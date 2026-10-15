"""
Smart Queue Management System  (multi-sector)
---------------------------------------------
A Flask + SQLite queue management system for hospitals, clinics, supermarkets,
banks and offices. One engine, configured by a "business type" preset.

Core features: priority scheduling, services with providers (doctors, counters,
officers) and their availability, per-provider and shared queues, round-robin
load balancing, concurrency-safe ticket assignment, retry/dead-letter handling
for no-shows, rate limiting, mobile-number token lookup, printable tokens,
time-based "your turn is near" notifications, and wait-time estimation
(baseline formula, with an optional Temporal Fusion Transformer upgrade).

Run locally:
    pip install -r requirements.txt
    python app.py
Then open http://127.0.0.1:5000  (customer view)
and   http://127.0.0.1:5000/admin (staff view)

See README.md for full setup, deployment, and testing instructions.

IMPORTANT ON TFT: torch and pytorch-forecasting are NEVER imported at
module load time. They are only imported lazily, inside
get_tft_wait_estimate(), at the moment a prediction is actually requested.
This guarantees the app starts and runs correctly even on a machine/host
(e.g. Render's free tier) where those heavy packages are not installed.
"""

import os
import re
import hmac
import time
import sqlite3
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from functools import wraps
from pathlib import Path
from zoneinfo import ZoneInfo

from flask import (
    Flask, render_template, request, redirect, url_for,
    jsonify, flash, g, abort, session
)

app = Flask(__name__)
# Set SECRET_KEY and ADMIN_PASSWORD as environment variables on Render.
# The fallback secret key is only for local development.
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "college-project-secret-key")
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

# Admin password. If it is not set, admin login is disabled (fails closed).
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")

DB_PATH = Path(__file__).parent / "queue.db"

MAX_RETRIES = 2            # how many times a no-show gets re-queued
NOTIFY_WINDOW = 2          # (console log) people ahead that counts as "almost your turn"
RATE_LIMIT_WINDOW = 60     # seconds
RATE_LIMIT_MAX = 5         # max joins per IP per window
RECENT_SERVICE_SAMPLE = 20 # how many past completed tickets to average
TFT_ENCODER_LENGTH = 96    # TFT needs this many past 15-min buckets of history

ORG_NAME = os.environ.get("ORG_NAME", "Smart Queue")       # default organisation name
TIMEZONE = os.environ.get("TIMEZONE", "Asia/Kolkata")      # provider timings + token times
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
KINDS = ("standard", "special")
TIME_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")

NOTIFY_MINUTE_CHOICES = [2, 5, 10, 15, 20]   # "notify me N minutes before my turn"
DEFAULT_NOTIFY_MINUTES = 5
DEFAULT_ORG_TYPE = "hospital"

MON_FRI = "Mon,Tue,Wed,Thu,Fri"
MON_SAT = "Mon,Tue,Wed,Thu,Fri,Sat"
EVERY_DAY = ",".join(DAY_NAMES)

# ---------------------------------------------------------------------------
# Business-type presets: wording + queue style + sample data.
#   mode "required": the customer must pick a provider (hospital, clinic)
#   mode "optional": the customer may pick a provider or "Any available"
#   mode "none":     customers only pick a service; counters serve the queue
# Sample providers: (name, kind, service, location, days, start, end)
# ---------------------------------------------------------------------------
PRESETS = {
    "hospital": {
        "title": "Hospital", "slip_title": "OPD TOKEN", "mode": "required",
        "provider": "Doctor", "providers": "Doctors",
        "service": "Department", "services": "Departments",
        "customer": "Patient", "kind_standard": "General", "kind_special": "Specialist",
        "priority": "Emergency / priority case",
        "sample_services": ["General Medicine", "Cardiology", "Dermatology", "Pediatrics"],
        "sample_providers": [
            ("Dr. Anand Kumar", "standard", "General Medicine", "Room 1", MON_SAT, "09:00", "17:00"),
            ("Dr. Priya Nair", "special", "Cardiology", "Room 2", "Mon,Wed,Fri", "10:00", "14:00"),
            ("Dr. Rahul Mehta", "special", "Dermatology", "Room 3", "Tue,Thu,Sat", "14:00", "18:00"),
            ("Dr. Sana Iqbal", "special", "Pediatrics", "Room 4", MON_FRI, "09:00", "13:00"),
        ],
        "sample_counters": ["Counter 1", "Counter 2"],
    },
    "clinic": {
        "title": "Clinic", "slip_title": "CLINIC TOKEN", "mode": "required",
        "provider": "Doctor", "providers": "Doctors",
        "service": "Department", "services": "Departments",
        "customer": "Patient", "kind_standard": "General", "kind_special": "Specialist",
        "priority": "Emergency / priority case",
        "sample_services": ["General Consultation", "Dental", "Eye Care"],
        "sample_providers": [
            ("Dr. Meera Das", "standard", "General Consultation", "Room A", MON_SAT, "09:00", "18:00"),
            ("Dr. Arun Pillai", "special", "Dental", "Room B", "Mon,Wed,Fri", "10:00", "16:00"),
            ("Dr. Lakshmi Rao", "special", "Eye Care", "Room C", "Tue,Thu,Sat", "10:00", "15:00"),
        ],
        "sample_counters": ["Counter 1", "Counter 2"],
    },
    "supermarket": {
        "title": "Supermarket", "slip_title": "QUEUE TOKEN", "mode": "optional",
        "provider": "Counter", "providers": "Counters",
        "service": "Service", "services": "Services",
        "customer": "Customer", "kind_standard": "Regular", "kind_special": "Express",
        "priority": "Senior citizen / priority",
        "sample_services": ["Billing", "Customer Service", "Returns & Exchange"],
        "sample_providers": [
            ("Counter 1", "standard", "Billing", "Billing Hall", EVERY_DAY, "08:00", "22:00"),
            ("Counter 2", "standard", "Billing", "Billing Hall", EVERY_DAY, "08:00", "22:00"),
            ("Express Counter (10 items or less)", "special", "Billing", "Billing Hall", EVERY_DAY, "08:00", "22:00"),
            ("Customer Care Desk", "standard", "Customer Service", "Front Desk", EVERY_DAY, "09:00", "21:00"),
            ("Returns Desk", "standard", "Returns & Exchange", "Front Desk", EVERY_DAY, "10:00", "20:00"),
        ],
        "sample_counters": ["Counter 1", "Counter 2"],
    },
    "bank": {
        "title": "Bank", "slip_title": "BANK TOKEN", "mode": "optional",
        "provider": "Counter", "providers": "Counters",
        "service": "Service", "services": "Services",
        "customer": "Customer", "kind_standard": "Regular", "kind_special": "Specialist",
        "priority": "Senior citizen / priority",
        "sample_services": ["Cash & Deposits", "Loans", "Account Services"],
        "sample_providers": [
            ("Counter 1", "standard", "Cash & Deposits", "Main Hall", MON_FRI, "10:00", "16:00"),
            ("Counter 2", "standard", "Cash & Deposits", "Main Hall", MON_FRI, "10:00", "16:00"),
            ("Loan Officer - Ms. Rao", "special", "Loans", "Cabin 1", MON_FRI, "10:00", "16:00"),
            ("Account Desk", "standard", "Account Services", "Desk 1", MON_FRI, "10:00", "16:00"),
        ],
        "sample_counters": ["Counter 1", "Counter 2"],
    },
    "office": {
        "title": "Office / Other", "slip_title": "TOKEN", "mode": "none",
        "provider": "Staff", "providers": "Staff",
        "service": "Service", "services": "Services",
        "customer": "Customer", "kind_standard": "Regular", "kind_special": "Special",
        "priority": "Priority / VIP request",
        "sample_services": ["General Service", "Customer Support", "Billing", "Other"],
        "sample_providers": [],
        "sample_counters": ["Counter 1", "Counter 2"],
    },
}

# ---------------------------------------------------------------------------
# Database connection handling
# ---------------------------------------------------------------------------

def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH, timeout=10)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


@app.teardown_appcontext
def close_db(exception=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


# ---------------------------------------------------------------------------
# Database setup, settings and sample data
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS counter (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS service (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS provider (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'standard',
    service_id INTEGER NOT NULL,
    location TEXT NOT NULL DEFAULT '',
    days TEXT NOT NULL DEFAULT 'Mon,Tue,Wed,Thu,Fri,Sat,Sun',
    start_time TEXT NOT NULL DEFAULT '09:00',
    end_time TEXT NOT NULL DEFAULT '17:00',
    active INTEGER NOT NULL DEFAULT 1,
    FOREIGN KEY (service_id) REFERENCES service(id)
);

CREATE TABLE IF NOT EXISTS ticket (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    token_no INTEGER NOT NULL,
    name TEXT NOT NULL,
    contact TEXT,
    service_type TEXT NOT NULL DEFAULT 'General Service',
    service_id INTEGER,
    provider_id INTEGER,
    served_by INTEGER,
    priority INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'waiting',
    counter_id INTEGER,
    retries INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    called_at TEXT,
    completed_at TEXT,
    notified INTEGER NOT NULL DEFAULT 0,
    notify_minutes INTEGER NOT NULL DEFAULT 5,
    FOREIGN KEY (counter_id) REFERENCES counter(id),
    FOREIGN KEY (service_id) REFERENCES service(id),
    FOREIGN KEY (provider_id) REFERENCES provider(id),
    FOREIGN KEY (served_by) REFERENCES provider(id)
);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE INDEX IF NOT EXISTS idx_ticket_status ON ticket(status);
CREATE INDEX IF NOT EXISTS idx_ticket_priority_created ON ticket(priority, created_at);
CREATE INDEX IF NOT EXISTS idx_ticket_provider ON ticket(provider_id, status);
CREATE INDEX IF NOT EXISTS idx_ticket_service ON ticket(service_id, status);
CREATE INDEX IF NOT EXISTS idx_ticket_served_by ON ticket(served_by, status);
CREATE INDEX IF NOT EXISTS idx_ticket_contact ON ticket(contact);
"""


def _is_old_layout(path):
    """True if the file holds a database from before services/providers existed."""
    if not Path(path).exists():
        return False
    con = sqlite3.connect(path)
    try:
        columns = [r[1] for r in con.execute("PRAGMA table_info(ticket)")]
    finally:
        con.close()
    return bool(columns) and "service_id" not in columns


def init_db():
    """Create tables on first startup and add the starter business setup.

    A database created by an older version (different table layout) is not
    modified: it is renamed to queue.db.old-<time> and a fresh one is created.
    """
    if _is_old_layout(DB_PATH):
        backup = DB_PATH.with_name(DB_PATH.name + ".old-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
        DB_PATH.replace(backup)
        print(f"[DB] Older database layout found. Saved as {backup.name}; starting a new database.")
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA)
    if get_meta(db, "org_type") is None:        # first start: set up the default business
        set_meta(db, "org_type", DEFAULT_ORG_TYPE)
        set_meta(db, "org_name", ORG_NAME)
        set_meta(db, "require_phone", "1")
        load_sample_data(db, DEFAULT_ORG_TYPE)
    db.commit()
    db.close()


def now_iso():
    return datetime.utcnow().isoformat(timespec="seconds")


def parse_dt(s):
    return datetime.fromisoformat(s) if s else None


# ---------------------------------------------------------------------------
# Settings and sample data
# ---------------------------------------------------------------------------

def get_settings(db):
    org_type = get_meta(db, "org_type", DEFAULT_ORG_TYPE)
    if org_type not in PRESETS:
        org_type = DEFAULT_ORG_TYPE
    return {
        "org_name": get_meta(db, "org_name", ORG_NAME),
        "org_type": org_type,
        "require_phone": get_meta(db, "require_phone", "1") == "1",
    }


def load_sample_data(db, org_type):
    """Replace ALL tickets, providers, services and counters with the sample
    setup of the chosen business type (used on first start and from Settings)."""
    preset = PRESETS[org_type]
    db.execute("DELETE FROM ticket")
    db.execute("DELETE FROM provider")
    db.execute("DELETE FROM service")
    db.execute("DELETE FROM counter")
    db.execute("DELETE FROM meta WHERE key='last_counter_id'")
    service_ids = {}
    for name in preset["sample_services"]:
        service_ids[name] = db.execute(
            "INSERT INTO service (name, active) VALUES (?, 1)", (name,)
        ).lastrowid
    for name, kind, service, location, days, start, end in preset["sample_providers"]:
        db.execute(
            "INSERT INTO provider (name, kind, service_id, location, days, start_time, end_time, active) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 1)",
            (name, kind, service_ids[service], location, days, start, end),
        )
    for name in preset["sample_counters"]:
        db.execute("INSERT INTO counter (name, active) VALUES (?, 1)", (name,))


# ---------------------------------------------------------------------------
# Time, phone-number, provider and service helpers
# ---------------------------------------------------------------------------

def _tz():
    try:
        return ZoneInfo(TIMEZONE)
    except Exception:
        return timezone(timedelta(hours=5, minutes=30))   # India, if tz data is missing


def local_now():
    return datetime.now(_tz())


def to_local(iso_str):
    """Times are stored in UTC; convert to local time for display."""
    if not iso_str:
        return None
    return datetime.fromisoformat(iso_str).replace(tzinfo=timezone.utc).astimezone(_tz())


@app.template_filter("localtime")
def localtime_filter(iso_str, fmt="%d %b %Y, %I:%M %p"):
    dt = to_local(iso_str)
    return dt.strftime(fmt).lstrip("0") if dt else "-"


def fmt_time(hhmm):
    """'14:30' -> '2:30 PM'"""
    h, m = (int(x) for x in hhmm.split(":"))
    return f"{(h % 12) or 12}:{m:02d} {'AM' if h < 12 else 'PM'}"


def days_text(days_csv):
    """'Mon,Tue,Wed,Fri' -> 'Mon-Wed, Fri'"""
    idx = sorted({DAY_NAMES.index(d) for d in days_csv.split(",") if d in DAY_NAMES})
    if not idx:
        return "No days set"
    if len(idx) == 7:
        return "Every day"
    runs, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            runs.append((start, prev))
            start = prev = i
    runs.append((start, prev))
    parts = []
    for a, b in runs:
        if a == b:
            parts.append(DAY_NAMES[a])
        elif b == a + 1:
            parts.append(f"{DAY_NAMES[a]}, {DAY_NAMES[b]}")
        else:
            parts.append(f"{DAY_NAMES[a]}\u2013{DAY_NAMES[b]}")
    return ", ".join(parts)


def provider_status(p, now=None):
    """Where is this provider (doctor, counter, ...) right now?"""
    now = now or local_now()
    if not p["active"]:
        return {"state": "leave", "label": "Not available"}
    if DAY_NAMES[now.weekday()] not in p["days"].split(","):
        return {"state": "off", "label": "Off today"}
    current = now.strftime("%H:%M")
    if current < p["start_time"]:
        return {"state": "later", "label": f"Arrives at {fmt_time(p['start_time'])}"}
    if current >= p["end_time"]:
        return {"state": "ended", "label": "Done for today"}
    return {"state": "available", "label": "Available now"}


def normalize_phone(raw):
    """Return a 10-digit Indian mobile number (digits only) or None if invalid.
    Accepts forms like 98765 43210, +91 98765 43210 and 098765 43210."""
    digits = re.sub(r"\D", "", raw or "")
    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    elif len(digits) == 11 and digits.startswith("0"):
        digits = digits[1:]
    return digits if re.fullmatch(r"[6-9]\d{9}", digits) else None


def mask_phone(contact):
    """9876543210 -> ******3210 (full numbers are never shown on public pages)."""
    if contact and re.fullmatch(r"\d{10}", contact):
        return "******" + contact[-4:]
    return None


def current_preset(db):
    return PRESETS[get_settings(db)["org_type"]]


@app.context_processor
def inject_business():
    """Makes the business wording (L), settings (org) and queue style (qmode)
    available in every template."""
    try:
        settings = get_settings(get_db())
    except Exception:
        settings = {"org_name": ORG_NAME, "org_type": DEFAULT_ORG_TYPE, "require_phone": True}
    preset = PRESETS[settings["org_type"]]
    return {"L": preset, "org": settings, "qmode": preset["mode"]}


def provider_dict(db, row, preset, now=None):
    """A provider row plus display text, today's status and queue numbers."""
    d = dict(row)
    d["kind_label"] = preset["kind_special"] if d["kind"] == "special" else preset["kind_standard"]
    d["days_text"] = days_text(d["days"])
    d["time_text"] = f"{fmt_time(d['start_time'])} \u2013 {fmt_time(d['end_time'])}"
    d["status"] = provider_status(d, now)
    own = db.execute(
        "SELECT COUNT(*) AS c FROM ticket WHERE provider_id=? AND status='waiting'", (d["id"],)
    ).fetchone()["c"]
    shared = db.execute(
        "SELECT COUNT(*) AS c FROM ticket "
        "WHERE provider_id IS NULL AND service_id=? AND status='waiting'", (d["service_id"],)
    ).fetchone()["c"]
    d["waiting_own"], d["waiting_shared"], d["waiting"] = own, shared, own + shared
    serving = db.execute(
        "SELECT id, token_no, name FROM ticket WHERE served_by=? AND status='serving' LIMIT 1",
        (d["id"],),
    ).fetchone()
    d["serving"] = dict(serving) if serving else None
    return d


def provider_dicts(db):
    preset, now = current_preset(db), local_now()
    rows = db.execute(
        """SELECT p.*, s.name AS service_name FROM provider p
           JOIN service s ON s.id = p.service_id
           ORDER BY s.name, CASE p.kind WHEN 'standard' THEN 0 ELSE 1 END, p.name"""
    ).fetchall()
    return [provider_dict(db, r, preset, now) for r in rows]


def service_dict(db, row):
    d = dict(row)
    d["providers"] = db.execute(
        "SELECT COUNT(*) AS c FROM provider WHERE service_id=? AND active=1", (d["id"],)
    ).fetchone()["c"]
    d["waiting"] = db.execute(
        "SELECT COUNT(*) AS c FROM ticket WHERE service_id=? AND status='waiting'", (d["id"],)
    ).fetchone()["c"]
    d["serving"] = db.execute(
        "SELECT COUNT(*) AS c FROM ticket WHERE service_id=? AND status='serving'", (d["id"],)
    ).fetchone()["c"]
    d["wait_minutes"] = service_wait_minutes(db, d["id"])
    return d


def service_dicts(db, only_active=False):
    sql = "SELECT * FROM service" + (" WHERE active=1" if only_active else "") + " ORDER BY name"
    return [service_dict(db, r) for r in db.execute(sql).fetchall()]


# ---------------------------------------------------------------------------
# Admin authentication (single shared password, session-based)
# ---------------------------------------------------------------------------
# The password comes from the ADMIN_PASSWORD environment variable. All admin
# routes are wrapped with admin_required; anyone not logged in is sent to
# /admin/login.

def admin_required(view_func):
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if not session.get("is_admin"):
            return redirect(url_for("admin_login", next=request.path))
        return view_func(*args, **kwargs)
    return wrapped


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

_rate_log = defaultdict(list)


def is_rate_limited(ip):
    now = time.time()
    _rate_log[ip] = [t for t in _rate_log[ip] if now - t < RATE_LIMIT_WINDOW]
    if len(_rate_log[ip]) >= RATE_LIMIT_MAX:
        return True
    _rate_log[ip].append(now)
    return False


# ---------------------------------------------------------------------------
# Queue logic helpers
# ---------------------------------------------------------------------------

def _average_duration(rows):
    durations = [
        (parse_dt(r["completed_at"]) - parse_dt(r["called_at"])).total_seconds()
        for r in rows
    ]
    return sum(durations) / len(durations) if durations else 180


def _recent_done(db, where="", params=()):
    return db.execute(
        f"""SELECT called_at, completed_at FROM ticket
            WHERE status='done' AND called_at IS NOT NULL AND completed_at IS NOT NULL {where}
            ORDER BY completed_at DESC LIMIT ?""",
        (*params, RECENT_SERVICE_SAMPLE),
    ).fetchall()


def avg_service_seconds(db, provider_id=None, service_id=None):
    """Average duration of the last RECENT_SERVICE_SAMPLE completed tickets.
    Uses that provider's own recent tickets if given and available, else the
    service's, else everyone's. Defaults to 3 minutes until data exists."""
    if provider_id is not None:
        rows = _recent_done(db, "AND served_by=?", (provider_id,))
        if rows:
            return _average_duration(rows)
    if service_id is not None:
        rows = _recent_done(db, "AND service_id=?", (service_id,))
        if rows:
            return _average_duration(rows)
    rows = _recent_done(db)
    return _average_duration(rows) if rows else 180


def active_counter_count(db):
    row = db.execute("SELECT COUNT(*) AS c FROM counter WHERE active=1").fetchone()
    return max(row["c"], 1)


def active_provider_count(db, service_id):
    return db.execute(
        "SELECT COUNT(*) AS c FROM provider WHERE service_id=? AND active=1", (service_id,)
    ).fetchone()["c"]


def count_ahead(db, priority, created_at, provider_id=None, ticket_id=None, service_id=None):
    """People waiting ahead of this ticket.
      - provider_id given: that provider's own queue
      - else service_id given: the shared queue of that service
        (tickets that did not ask for a particular provider)
      - neither: the whole queue
    With a ticket_id, tickets created in the same second are ordered by ticket
    number, the same way "Call Next" orders them, so nobody shares a position."""
    if ticket_id is not None:
        sql = """SELECT COUNT(*) AS c FROM ticket
                 WHERE status='waiting' AND (
                     priority > ? OR (priority = ? AND (
                         created_at < ? OR (created_at = ? AND id < ?)
                     ))
                 )"""
        params = [priority, priority, created_at, created_at, ticket_id]
    else:
        sql = """SELECT COUNT(*) AS c FROM ticket
                 WHERE status='waiting' AND (
                     priority > ? OR (priority = ? AND created_at < ?)
                 )"""
        params = [priority, priority, created_at]
    if provider_id is not None:
        sql += " AND provider_id = ?"
        params.append(provider_id)
    elif service_id is not None:
        sql += " AND provider_id IS NULL AND service_id = ?"
        params.append(service_id)
    return db.execute(sql, params).fetchone()["c"]


def estimate_wait_seconds(db, priority, created_at, provider_id=None, ticket_id=None, service_id=None):
    """Baseline estimator:
        wait = (people ahead * average service time) / servers
    servers = 1 for one provider's own queue; the number of active providers
    for a service's shared queue (or active counters if the service has no
    providers); active counters for the whole queue."""
    ahead = count_ahead(db, priority, created_at, provider_id, ticket_id, service_id)
    if provider_id is not None:
        servers = 1
    elif service_id is not None:
        servers = active_provider_count(db, service_id) or active_counter_count(db)
    else:
        servers = active_counter_count(db)
    avg_seconds = avg_service_seconds(db, provider_id, service_id)
    return (ahead * avg_seconds) / servers


def estimate_wait_minutes(db, priority, created_at, provider_id=None, ticket_id=None, service_id=None):
    return int(estimate_wait_seconds(db, priority, created_at, provider_id, ticket_id, service_id) // 60)


def service_wait_minutes(db, service_id):
    """Rough wait for someone joining this service's shared queue right now."""
    ahead = db.execute(
        "SELECT COUNT(*) AS c FROM ticket "
        "WHERE status='waiting' AND provider_id IS NULL AND service_id=?", (service_id,)
    ).fetchone()["c"]
    servers = active_provider_count(db, service_id) or active_counter_count(db)
    return int((ahead * avg_service_seconds(db, None, service_id) / servers) // 60)


def next_token(db):
    row = db.execute("SELECT MAX(token_no) AS m FROM ticket").fetchone()
    return (row["m"] or 0) + 1


def get_meta(db, key, default=None):
    row = db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def set_meta(db, key, value):
    db.execute(
        "INSERT INTO meta (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, str(value)),
    )


def pick_counter(db):
    """
    Choose the counter to assign the next customer to:
      1. Among active counters, find the minimum number currently 'serving'.
      2. If multiple counters share that minimum, break the tie using
         round-robin order (the counter after the last-assigned one, wrapping
         around), so load is distributed fairly over time rather than always
         picking the same (e.g. lowest-id) counter.
    """
    counters = db.execute(
        "SELECT * FROM counter WHERE active=1 ORDER BY id ASC"
    ).fetchall()
    if not counters:
        return None

    loads = {}
    for c in counters:
        loads[c["id"]] = db.execute(
            "SELECT COUNT(*) AS c FROM ticket WHERE counter_id=? AND status='serving'",
            (c["id"],),
        ).fetchone()["c"]

    min_load = min(loads.values())
    tied = [c for c in counters if loads[c["id"]] == min_load]

    if len(tied) == 1:
        chosen = tied[0]
    else:
        last_id = get_meta(db, "last_counter_id")
        last_id = int(last_id) if last_id is not None else None
        tied_ids = [c["id"] for c in tied]
        if last_id in tied_ids:
            idx = (tied_ids.index(last_id) + 1) % len(tied_ids)
        else:
            idx = 0
        chosen = next(c for c in tied if c["id"] == tied_ids[idx])

    set_meta(db, "last_counter_id", chosen["id"])
    return chosen


def waiting_list(db):
    return db.execute(
        """SELECT t.*, s.name AS service_name, p.name AS provider_name
           FROM ticket t
           LEFT JOIN service s ON s.id = t.service_id
           LEFT JOIN provider p ON p.id = t.provider_id
           WHERE t.status='waiting'
           ORDER BY t.priority DESC, t.created_at ASC, t.id ASC"""
    ).fetchall()


# ---------------------------------------------------------------------------
# Optional TFT forecasting (never imported at startup)
# ---------------------------------------------------------------------------

def build_recent_history_df(db, service_type):
    """
    Builds a DataFrame of the last TFT_ENCODER_LENGTH 15-minute time buckets
    from real ticket data, in the same column format the TFT model was
    trained on (see ml/generate_synthetic_data.py). Returns None if there
    isn't enough history yet to make a meaningful prediction.
    """
    import pandas as pd  # pandas is a normal (lightweight) dependency, safe to import eagerly elsewhere, but kept local here to keep this function self-contained

    rows = db.execute("SELECT * FROM ticket").fetchall()
    if not rows:
        return None

    df = pd.DataFrame([dict(r) for r in rows])
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["bucket"] = df["created_at"].dt.floor("15min")

    grouped = df.groupby("bucket").agg(
        queue_length=("id", "count"),
        priority_ratio=("priority", "mean"),
    ).reset_index()

    if len(grouped) < TFT_ENCODER_LENGTH:
        return None  # not enough real history yet - caller should fall back

    grouped = grouped.sort_values("bucket").tail(TFT_ENCODER_LENGTH).reset_index(drop=True)
    grouped["time_idx"] = range(len(grouped))
    grouped["hour"] = grouped["bucket"].dt.hour + grouped["bucket"].dt.minute / 60
    grouped["day_of_week"] = grouped["bucket"].dt.dayofweek
    grouped["is_weekend"] = (grouped["day_of_week"] >= 5).astype(int)
    grouped["counter_id"] = "1"
    grouped["service_type"] = service_type
    grouped["active_counters"] = active_counter_count(db)
    grouped["avg_service_seconds"] = avg_service_seconds(db)

    return grouped


def get_tft_wait_estimate(db, priority, created_at, service_type):
    """
    Attempts to use the trained TFT model for a quantile-based wait estimate
    (P10/P50/P90), accepting the ticket's service_type, recent queue
    history, the current active counter count, and current time features.

    Returns a dict {"p10": int, "p50": int, "p90": int} on success, or None
    if the TFT model/dependencies/history are unavailable or prediction
    fails for any reason - in which case the caller MUST fall back to the
    baseline estimator. This function never raises; it only returns None
    and logs why.
    """
    try:
        # Lazy import: torch / pytorch-forecasting are only touched here,
        # at prediction time, never at module import time.
        from ml.predict import predict_wait_tft, minutes_from_queue_length
    except Exception as e:
        print(f"[TFT] Not available ({e.__class__.__name__}: {e}). "
              f"Using baseline moving-average estimator instead.")
        return None

    try:
        history_df = build_recent_history_df(db, service_type)
        if history_df is None:
            print("[TFT] Not enough real ticket history yet for a TFT forecast. "
                  "Using baseline moving-average estimator instead.")
            return None

        quantiles = predict_wait_tft(history_df)
        avg_seconds = avg_service_seconds(db)
        counters = active_counter_count(db)

        p10 = minutes_from_queue_length(quantiles["p10"][0], avg_seconds, counters)
        p50 = minutes_from_queue_length(quantiles["p50"][0], avg_seconds, counters)
        p90 = minutes_from_queue_length(quantiles["p90"][0], avg_seconds, counters)
        return {"p10": p10, "p50": p50, "p90": p90}
    except Exception as e:
        print(f"[TFT] Prediction failed ({e.__class__.__name__}: {e}). "
              f"Using baseline moving-average estimator instead.")
        return None


def get_wait_estimate(db, priority, created_at, service_type, provider_id=None,
                      ticket_id=None, service_id=None):
    """
    Single entry point the routes call for a wait-time estimate. Tries TFT
    first; falls back to the baseline formula automatically and silently
    (from the user's point of view - a log line is printed either way).
    Returns a dict shaped for the template:
        {"mode": "tft", "p10": int, "p50": int, "p90": int}
        {"mode": "baseline", "minutes": int}
    """
    tft_result = get_tft_wait_estimate(db, priority, created_at, service_type)
    if tft_result is not None:
        print(f"[WAIT-ESTIMATE] Using TFT prediction: {tft_result}")
        return {"mode": "tft", **tft_result}

    minutes = estimate_wait_minutes(db, priority, created_at, provider_id, ticket_id, service_id)
    print(f"[WAIT-ESTIMATE] Using baseline moving-average estimate: {minutes} min")
    return {"mode": "baseline", "minutes": minutes}


# ---------------------------------------------------------------------------
# Routes: customer-facing
# ---------------------------------------------------------------------------

def _load_provider(db, provider_id, preset, now=None):
    if provider_id is None:
        return None
    row = db.execute(
        "SELECT p.*, s.name AS service_name FROM provider p "
        "JOIN service s ON s.id = p.service_id WHERE p.id=?", (provider_id,)
    ).fetchone()
    return provider_dict(db, row, preset, now) if row else None


def build_ticket_context(db, ticket):
    """Everything a page needs to show one token."""
    preset, now = current_preset(db), local_now()
    service = None
    if ticket["service_id"] is not None:
        row = db.execute("SELECT * FROM service WHERE id=?", (ticket["service_id"],)).fetchone()
        service = dict(row) if row else None
    requested = _load_provider(db, ticket["provider_id"], preset, now)
    served_by = _load_provider(db, ticket["served_by"], preset, now)

    position = wait_estimate = eta_seconds = counter_name = None
    if ticket["status"] == "waiting":
        scope = (ticket["provider_id"], ticket["id"], ticket["service_id"])
        position = count_ahead(db, ticket["priority"], ticket["created_at"], *scope) + 1
        wait_estimate = get_wait_estimate(
            db, ticket["priority"], ticket["created_at"], ticket["service_type"], *scope
        )
        eta_seconds = int(estimate_wait_seconds(db, ticket["priority"], ticket["created_at"], *scope))
    if ticket["status"] == "serving" and ticket["counter_id"] is not None:
        c = db.execute("SELECT name FROM counter WHERE id=?", (ticket["counter_id"],)).fetchone()
        counter_name = c["name"] if c else None
    return {
        "ticket": ticket,
        "service": service,
        "provider": requested,            # the provider the customer asked for (None = any)
        "served_by": served_by,           # the provider who called the ticket
        "shown_provider": served_by or requested,
        "position": position,
        "wait_estimate": wait_estimate,
        "eta_seconds": eta_seconds,
        "counter_name": counter_name,
        "phone_masked": mask_phone(ticket["contact"]),
        "notify_choices": NOTIFY_MINUTE_CHOICES,
    }


@app.route("/")
def index():
    db = get_db()
    preset = current_preset(db)
    services = service_dicts(db)
    providers = provider_dicts(db)
    providers_json = [
        {
            "id": p["id"], "name": p["name"], "kind": p["kind"], "kind_label": p["kind_label"],
            "service_id": p["service_id"], "location": p["location"],
            "days_text": p["days_text"], "time_text": p["time_text"],
            "status_label": p["status"]["label"], "waiting": p["waiting"],
            "active": bool(p["active"]),
        }
        for p in providers
    ]
    services_json = [{"id": s["id"], "name": s["name"]} for s in services if s["active"]]
    return render_template(
        "index.html",
        waiting=waiting_list(db),
        services=services,
        providers=providers,
        services_json=services_json,
        providers_json=providers_json,
        notify_choices=NOTIFY_MINUTE_CHOICES,
        default_notify=DEFAULT_NOTIFY_MINUTES,
    )


@app.route("/join", methods=["POST"])
def join():
    ip = request.remote_addr or "unknown"
    if is_rate_limited(ip):
        flash("Too many requests from your device. Please wait a minute and try again.", "error")
        return redirect(url_for("index"))

    name = request.form.get("name", "").strip()
    contact = request.form.get("contact", "").strip()
    priority = 1 if request.form.get("priority") == "on" else 0
    service_id = request.form.get("service_id", type=int)
    provider_id = request.form.get("provider_id", type=int)
    notify_minutes = request.form.get("notify_minutes", type=int)
    if notify_minutes not in NOTIFY_MINUTE_CHOICES:
        notify_minutes = DEFAULT_NOTIFY_MINUTES

    db = get_db()
    settings = get_settings(db)
    preset = PRESETS[settings["org_type"]]
    mode = preset["mode"]
    service_word, provider_word = preset["service"].lower(), preset["provider"].lower()

    def fail(message):
        flash(message, "error")
        return redirect(url_for("index"))

    # --- input validation ---
    if not name:
        return fail("Name is required.")
    if len(name) > 100:
        return fail("Name is too long (max 100 characters).")

    service = None
    if service_id is not None:
        service = db.execute("SELECT * FROM service WHERE id=? AND active=1", (service_id,)).fetchone()
    if service is None:
        return fail(f"Please select a {service_word}.")

    provider = None
    if mode != "none":
        providers = db.execute(
            "SELECT * FROM provider WHERE service_id=? AND active=1", (service["id"],)
        ).fetchall()
        if not providers:
            return fail(f"No {provider_word} is available for this {service_word} right now.")
        if provider_id is not None:
            provider = next((p for p in providers if p["id"] == provider_id), None)
            if provider is None:
                return fail(f"Please select an available {provider_word}.")
        elif mode == "required":
            return fail(f"Please select a {provider_word}.")

    phone = None
    if contact or settings["require_phone"]:
        phone = normalize_phone(contact)
        if phone is None:
            return fail("Please enter a valid 10-digit mobile number.")
        existing = db.execute(
            "SELECT id, token_no FROM ticket "
            "WHERE contact=? AND service_id=? AND status IN ('waiting','serving') LIMIT 1",
            (phone, service["id"]),
        ).fetchone()
        if existing:
            flash(f"You already have an active token (#{existing['token_no']}) for {service['name']}.", "info")
            return redirect(url_for("ticket_status", ticket_id=existing["id"]))

    try:
        token_no = next_token(db)
        cur = db.execute(
            """INSERT INTO ticket
               (token_no, name, contact, service_type, service_id, provider_id,
                priority, status, created_at, notify_minutes)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'waiting', ?, ?)""",
            (token_no, name, phone, service["name"], service["id"],
             provider["id"] if provider else None, priority, now_iso(), notify_minutes),
        )
        db.commit()
    except sqlite3.Error as e:
        flash(f"Could not create ticket due to a database error: {e}", "error")
        return redirect(url_for("index"))

    return redirect(url_for("ticket_status", ticket_id=cur.lastrowid))


@app.route("/ticket/<int:ticket_id>")
def ticket_status(ticket_id):
    db = get_db()
    ticket = db.execute("SELECT * FROM ticket WHERE id=?", (ticket_id,)).fetchone()
    if ticket is None:
        abort(404, description="Ticket not found.")

    ctx = build_ticket_context(db, ticket)
    if ticket["status"] == "waiting" and not ticket["notified"]:
        near = ctx["position"] <= NOTIFY_WINDOW or ctx["eta_seconds"] <= ticket["notify_minutes"] * 60
        if near:
            db.execute("UPDATE ticket SET notified=1 WHERE id=?", (ticket_id,))
            db.commit()
            print(f"[NOTIFY] {ticket['name']} ({ticket['contact']}): "
                  f"You're #{ctx['position']} in line, about {ctx['eta_seconds'] // 60} min to go.")

    return render_template("ticket.html", **ctx)


@app.route("/ticket/<int:ticket_id>/notify", methods=["POST"])
def set_notify_minutes(ticket_id):
    """Let the customer choose how many minutes before their turn to be alerted."""
    db = get_db()
    ticket = db.execute("SELECT id FROM ticket WHERE id=?", (ticket_id,)).fetchone()
    if ticket is None:
        abort(404, description="Ticket not found.")
    minutes = request.form.get("minutes", type=int)
    if minutes not in NOTIFY_MINUTE_CHOICES:
        flash("Please choose one of the listed times.", "error")
    else:
        db.execute(
            "UPDATE ticket SET notify_minutes=?, notified=0 WHERE id=? AND status='waiting'",
            (minutes, ticket_id),
        )
        db.commit()
        flash(f"We'll alert you about {minutes} minutes before your turn.", "info")
    return redirect(url_for("ticket_status", ticket_id=ticket_id))


@app.route("/ticket/<int:ticket_id>/print")
def print_token(ticket_id):
    """A clean, printable token slip."""
    db = get_db()
    ticket = db.execute("SELECT * FROM ticket WHERE id=?", (ticket_id,)).fetchone()
    if ticket is None:
        abort(404, description="Ticket not found.")
    ctx = build_ticket_context(db, ticket)
    return render_template(
        "print.html", printed_at=local_now().strftime("%d %b %Y, %I:%M %p").lstrip("0"), **ctx
    )


@app.route("/track", methods=["GET", "POST"])
def track():
    """Enter a mobile number to see all tokens booked with it."""
    cards, phone_masked = None, None
    if request.method == "POST":
        ip = request.remote_addr or "unknown"
        if is_rate_limited("lookup:" + ip):
            flash("Too many lookups. Please wait a minute and try again.", "error")
            return render_template("track.html", cards=None, phone_masked=None), 429
        phone = normalize_phone(request.form.get("phone", ""))
        if phone is None:
            flash("Please enter a valid 10-digit mobile number.", "error")
        else:
            db = get_db()
            rows = db.execute(
                "SELECT * FROM ticket WHERE contact=? ORDER BY id DESC LIMIT 10", (phone,)
            ).fetchall()
            cards = [build_ticket_context(db, t) for t in rows]
            phone_masked = mask_phone(phone)
    return render_template("track.html", cards=cards, phone_masked=phone_masked)


@app.route("/ticket/<int:ticket_id>/cancel", methods=["POST"])
def cancel_ticket(ticket_id):
    db = get_db()
    ticket = db.execute("SELECT * FROM ticket WHERE id=?", (ticket_id,)).fetchone()
    if ticket is None:
        abort(404, description="Ticket not found.")
    db.execute(
        "UPDATE ticket SET status='cancelled' WHERE id=? AND status='waiting'",
        (ticket_id,),
    )
    db.commit()
    return redirect(url_for("ticket_status", ticket_id=ticket_id))


@app.route("/ticket/<int:ticket_id>/status.json")
def ticket_status_json(ticket_id):
    db = get_db()
    ticket = db.execute("SELECT * FROM ticket WHERE id=?", (ticket_id,)).fetchone()
    if ticket is None:
        return jsonify(error="Ticket not found"), 404
    position = eta_seconds = notify_minutes = None
    if ticket["status"] == "waiting":
        scope = (ticket["provider_id"], ticket["id"], ticket["service_id"])
        position = count_ahead(db, ticket["priority"], ticket["created_at"], *scope) + 1
        eta_seconds = int(estimate_wait_seconds(db, ticket["priority"], ticket["created_at"], *scope))
        notify_minutes = ticket["notify_minutes"]
    return jsonify(status=ticket["status"], position=position,
                   eta_seconds=eta_seconds, notify_minutes=notify_minutes)


# ---------------------------------------------------------------------------
# Routes: staff / admin
# ---------------------------------------------------------------------------

def _safe_next(target):
    """Only allow redirects to paths inside this site."""
    if target and target.startswith("/") and not target.startswith("//"):
        return target
    return url_for("admin")


@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    next_url = request.values.get("next", "")
    if request.method == "POST":
        ip = request.remote_addr or "unknown"
        if is_rate_limited("login:" + ip):
            flash("Too many login attempts. Please wait a minute and try again.", "error")
            return render_template("admin_login.html", next_url=next_url), 429
        if not ADMIN_PASSWORD:
            flash("Admin login is disabled: the ADMIN_PASSWORD environment variable is not set.", "error")
            return render_template("admin_login.html", next_url=next_url)
        supplied = request.form.get("password", "")
        if hmac.compare_digest(supplied.encode(), ADMIN_PASSWORD.encode()):
            session.clear()
            session["is_admin"] = True
            return redirect(_safe_next(next_url))
        flash("Incorrect password.", "error")
    return render_template("admin_login.html", next_url=next_url)


@app.route("/admin/logout", methods=["POST"])
def admin_logout():
    session.clear()
    return redirect(url_for("index"))


@app.route("/admin")
def admin():
    # Read-only for everyone; action buttons are shown only to logged-in admins,
    # and every action route below still requires admin_required.
    db = get_db()
    waiting = waiting_list(db)
    serving = db.execute(
        """SELECT t.*, s.name AS service_name, p.name AS provider_name,
                  p.location AS provider_location
           FROM ticket t
           LEFT JOIN service s ON s.id = t.service_id
           LEFT JOIN provider p ON p.id = t.served_by
           WHERE t.status='serving'"""
    ).fetchall()
    counters = db.execute("SELECT * FROM counter ORDER BY id ASC").fetchall()

    since = (datetime.utcnow() - timedelta(hours=24)).isoformat(timespec="seconds")
    completed_today = db.execute(
        "SELECT COUNT(*) AS c FROM ticket WHERE status='done' AND completed_at >= ?",
        (since,),
    ).fetchone()["c"]
    dead_letter = db.execute(
        "SELECT COUNT(*) AS c FROM ticket WHERE status='dead_letter'"
    ).fetchone()["c"]
    avg_wait = round(avg_service_seconds(db) / 60, 1)

    return render_template(
        "admin.html",
        waiting=waiting,
        serving=serving,
        counters=counters,
        services=service_dicts(db),
        providers=provider_dicts(db),
        day_names=DAY_NAMES,
        completed_today=completed_today,
        dead_letter=dead_letter,
        avg_wait=avg_wait,
        queue_length=len(waiting),
    )


@app.route("/admin/call_next", methods=["POST"])
@admin_required
def call_next():
    """Call the next waiting ticket.
      - with provider_id: for that provider (their own queue and their
        service's shared queue; one customer at a time)
      - with service_id:  from that service's shared queue, onto a counter
      - with neither:     from the shared queue of any service, onto a counter
    """
    db = get_db()
    provider_id = request.form.get("provider_id", type=int)
    service_id = request.form.get("service_id", type=int)

    if provider_id is not None:
        provider = db.execute("SELECT * FROM provider WHERE id=?", (provider_id,)).fetchone()
        if provider is None:
            abort(404, description="Not found.")
        busy = db.execute(
            "SELECT token_no FROM ticket WHERE served_by=? AND status='serving' LIMIT 1",
            (provider_id,),
        ).fetchone()
        if busy:
            flash(f"{provider['name']} is already serving token #{busy['token_no']}. "
                  f"Complete that token first.", "error")
            return redirect(url_for("admin"))
        next_ticket = db.execute(
            """SELECT * FROM ticket WHERE status='waiting'
                 AND (provider_id = ? OR (provider_id IS NULL AND service_id = ?))
               ORDER BY priority DESC, created_at ASC, id ASC LIMIT 1""",
            (provider_id, provider["service_id"]),
        ).fetchone()
        if not next_ticket:
            flash(f"No one is waiting for {provider['name']}.", "info")
            return redirect(url_for("admin"))
        # Atomic: only succeeds if the ticket is still waiting AND this provider
        # has nobody serving, so simultaneous clicks cannot double-assign.
        try:
            cur = db.execute(
                """UPDATE ticket SET status='serving', served_by=?, counter_id=NULL, called_at=?
                   WHERE id=? AND status='waiting'
                     AND NOT EXISTS (SELECT 1 FROM ticket t2
                                     WHERE t2.served_by=? AND t2.status='serving')""",
                (provider_id, now_iso(), next_ticket["id"], provider_id),
            )
            db.commit()
        except sqlite3.Error as e:
            flash(f"Database error while calling next ticket: {e}", "error")
            return redirect(url_for("admin"))
        if cur.rowcount == 0:
            flash("That ticket was already taken.", "error")
        return redirect(url_for("admin"))

    # --- shared queue, served at a counter ---
    if service_id is not None and db.execute(
        "SELECT id FROM service WHERE id=?", (service_id,)
    ).fetchone() is None:
        abort(404, description="Not found.")

    counter = pick_counter(db)
    if not counter:
        flash("No active counters available.", "error")
        return redirect(url_for("admin"))

    sql = "SELECT * FROM ticket WHERE status='waiting' AND provider_id IS NULL"
    params = []
    if service_id is not None:
        sql += " AND service_id=?"
        params.append(service_id)
    sql += " ORDER BY priority DESC, created_at ASC, id ASC LIMIT 1"
    next_ticket = db.execute(sql, params).fetchone()
    if not next_ticket:
        flash("Queue is empty.", "info")
        return redirect(url_for("admin"))

    # Atomic conditional update: only succeeds if still 'waiting', so two
    # simultaneous "Call Next" clicks can never assign the same ticket twice.
    try:
        cur = db.execute(
            "UPDATE ticket SET status='serving', counter_id=?, called_at=? "
            "WHERE id=? AND status='waiting'",
            (counter["id"], now_iso(), next_ticket["id"]),
        )
        db.commit()
    except sqlite3.Error as e:
        flash(f"Database error while calling next ticket: {e}", "error")
        return redirect(url_for("admin"))

    if cur.rowcount == 0:
        flash("That ticket was already taken by another counter.", "error")
    return redirect(url_for("admin"))


@app.route("/admin/complete/<int:ticket_id>", methods=["POST"])
@admin_required
def complete_ticket(ticket_id):
    db = get_db()
    ticket = db.execute("SELECT * FROM ticket WHERE id=?", (ticket_id,)).fetchone()
    if ticket is None:
        abort(404, description="Ticket not found.")
    db.execute(
        "UPDATE ticket SET status='done', completed_at=? WHERE id=? AND status='serving'",
        (now_iso(), ticket_id),
    )
    db.commit()
    return redirect(url_for("admin"))


@app.route("/admin/no_show/<int:ticket_id>", methods=["POST"])
@admin_required
def no_show(ticket_id):
    """Retry logic + dead-letter queue for repeat no-shows."""
    db = get_db()
    ticket = db.execute("SELECT * FROM ticket WHERE id=?", (ticket_id,)).fetchone()
    if ticket is None:
        abort(404, description="Ticket not found.")

    retries = ticket["retries"] + 1
    if retries > MAX_RETRIES:
        db.execute(
            "UPDATE ticket SET status='dead_letter', retries=? WHERE id=? AND status='serving'",
            (retries, ticket_id),
        )
    else:
        db.execute(
            "UPDATE ticket SET status='waiting', counter_id=NULL, served_by=NULL, called_at=NULL, "
            "notified=0, retries=? WHERE id=? AND status='serving'",
            (retries, ticket_id),
        )
    db.commit()
    return redirect(url_for("admin"))


@app.route("/admin/counters/toggle/<int:counter_id>", methods=["POST"])
@admin_required
def toggle_counter(counter_id):
    db = get_db()
    counter = db.execute("SELECT * FROM counter WHERE id=?", (counter_id,)).fetchone()
    if counter is None:
        abort(404, description="Counter not found.")
    db.execute("UPDATE counter SET active = 1 - active WHERE id=?", (counter_id,))
    db.commit()
    return redirect(url_for("admin"))


@app.route("/admin/counters/add", methods=["POST"])
@admin_required
def add_counter():
    db = get_db()
    count = db.execute("SELECT COUNT(*) AS c FROM counter").fetchone()["c"]
    name = request.form.get("counter_name", "").strip() or f"Counter {count + 1}"
    if len(name) > 50:
        flash("Counter name is too long (max 50 characters).", "error")
        return redirect(url_for("admin"))
    db.execute("INSERT INTO counter (name, active) VALUES (?, 1)", (name,))
    db.commit()
    return redirect(url_for("admin"))


# ---------------------------------------------------------------------------
# Routes: services, providers and settings (admin only)
# ---------------------------------------------------------------------------

def _read_service_name(form, db, exclude_id=None):
    name = form.get("name", "").strip()
    if not name or len(name) > 60:
        return None, "Name is required (max 60 characters)."
    rows = db.execute("SELECT id FROM service WHERE LOWER(name)=LOWER(?)", (name,)).fetchall()
    if any(r["id"] != exclude_id for r in rows):
        return None, "A service with that name already exists."
    return name, None


@app.route("/admin/services/add", methods=["POST"])
@admin_required
def add_service():
    db = get_db()
    name, error = _read_service_name(request.form, db)
    if error:
        flash(error, "error")
        return redirect(url_for("admin"))
    db.execute("INSERT INTO service (name, active) VALUES (?, 1)", (name,))
    db.commit()
    flash(f"{name} added.", "info")
    return redirect(url_for("admin"))


@app.route("/admin/services/<int:service_id>/toggle", methods=["POST"])
@admin_required
def toggle_service(service_id):
    db = get_db()
    if db.execute("SELECT id FROM service WHERE id=?", (service_id,)).fetchone() is None:
        abort(404, description="Not found.")
    db.execute("UPDATE service SET active = 1 - active WHERE id=?", (service_id,))
    db.commit()
    return redirect(url_for("admin"))


@app.route("/admin/services/<int:service_id>/edit", methods=["GET", "POST"])
@admin_required
def edit_service(service_id):
    db = get_db()
    row = db.execute("SELECT * FROM service WHERE id=?", (service_id,)).fetchone()
    if row is None:
        abort(404, description="Not found.")
    if request.method == "POST":
        name, error = _read_service_name(request.form, db, exclude_id=service_id)
        if error:
            flash(error, "error")
            return redirect(url_for("edit_service", service_id=service_id))
        db.execute("UPDATE service SET name=? WHERE id=?", (name, service_id))
        db.execute("UPDATE ticket SET service_type=? WHERE service_id=?", (name, service_id))
        db.commit()
        flash("Service renamed.", "info")
        return redirect(url_for("admin"))
    return render_template("admin_service_edit.html", svc=dict(row))


def _read_provider_form(form, db):
    """Validate the add/edit provider form. Returns (values, error_message)."""
    name = form.get("name", "").strip()
    kind = form.get("kind", "standard")
    service_id = form.get("service_id", type=int)
    location = form.get("location", "").strip()
    days = [d for d in DAY_NAMES if d in form.getlist("days")]
    start = form.get("start_time", "").strip()
    end = form.get("end_time", "").strip()

    if not name or len(name) > 100:
        return None, "Name is required (max 100 characters)."
    if kind not in KINDS:
        return None, "Choose a valid type."
    if service_id is None or db.execute("SELECT id FROM service WHERE id=?", (service_id,)).fetchone() is None:
        return None, "Choose a service."
    if len(location) > 60:
        return None, "Location is too long (max 60 characters)."
    if not days:
        return None, "Select at least one working day."
    if not (TIME_RE.match(start) and TIME_RE.match(end)):
        return None, "Enter valid start and end times."
    if start >= end:
        return None, "End time must be after start time."
    return {
        "name": name, "kind": kind, "service_id": service_id, "location": location,
        "days": ",".join(days), "start_time": start, "end_time": end,
    }, None


@app.route("/admin/providers/add", methods=["POST"])
@admin_required
def add_provider():
    db = get_db()
    values, error = _read_provider_form(request.form, db)
    if error:
        flash(error, "error")
        return redirect(url_for("admin"))
    db.execute(
        "INSERT INTO provider (name, kind, service_id, location, days, start_time, end_time, active) "
        "VALUES (:name, :kind, :service_id, :location, :days, :start_time, :end_time, 1)",
        values,
    )
    db.commit()
    flash(f"{values['name']} added.", "info")
    return redirect(url_for("admin"))


@app.route("/admin/providers/<int:provider_id>/toggle", methods=["POST"])
@admin_required
def toggle_provider(provider_id):
    db = get_db()
    if db.execute("SELECT id FROM provider WHERE id=?", (provider_id,)).fetchone() is None:
        abort(404, description="Not found.")
    db.execute("UPDATE provider SET active = 1 - active WHERE id=?", (provider_id,))
    db.commit()
    return redirect(url_for("admin"))


@app.route("/admin/providers/<int:provider_id>/edit", methods=["GET", "POST"])
@admin_required
def edit_provider(provider_id):
    db = get_db()
    row = db.execute("SELECT * FROM provider WHERE id=?", (provider_id,)).fetchone()
    if row is None:
        abort(404, description="Not found.")
    if request.method == "POST":
        values, error = _read_provider_form(request.form, db)
        if error:
            flash(error, "error")
            return redirect(url_for("edit_provider", provider_id=provider_id))
        values["id"] = provider_id
        db.execute(
            "UPDATE provider SET name=:name, kind=:kind, service_id=:service_id, location=:location, "
            "days=:days, start_time=:start_time, end_time=:end_time WHERE id=:id",
            values,
        )
        db.commit()
        flash("Updated.", "info")
        return redirect(url_for("admin"))
    return render_template(
        "admin_provider_edit.html", prov=dict(row), day_names=DAY_NAMES,
        service_options=db.execute("SELECT * FROM service ORDER BY name").fetchall(),
    )


@app.route("/admin/settings", methods=["GET", "POST"])
@admin_required
def settings_page():
    db = get_db()
    if request.method == "POST":
        org_name = request.form.get("org_name", "").strip()
        org_type = request.form.get("org_type", "")
        if not org_name or len(org_name) > 80:
            flash("Organisation name is required (max 80 characters).", "error")
            return redirect(url_for("settings_page"))
        if org_type not in PRESETS:
            flash("Choose a business type.", "error")
            return redirect(url_for("settings_page"))
        set_meta(db, "org_name", org_name)
        set_meta(db, "org_type", org_type)
        set_meta(db, "require_phone", "1" if request.form.get("require_phone") == "on" else "0")
        if request.form.get("load_samples") == "on":
            load_sample_data(db, org_type)
            flash(f"Settings saved and the sample {PRESETS[org_type]['title']} setup was loaded.", "info")
        else:
            flash("Settings saved.", "info")
        db.commit()
        return redirect(url_for("admin"))
    return render_template("settings.html", settings=get_settings(db), presets=PRESETS)


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", message=str(e.description or "Page not found.")), 404


@app.errorhandler(500)
def server_error(e):
    return render_template("error.html", message="Something went wrong on our end."), 500


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

# Initialize DB immediately on import so it also works under gunicorn (Render).
init_db()

if __name__ == "__main__":
    app.run(debug=True)
