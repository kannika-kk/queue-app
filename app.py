"""
Smart Queue Management System
------------------------------
A Flask + SQLite queue management system with priority scheduling,
round-robin load balancing, concurrency-safe ticket assignment, retry/
dead-letter handling for no-shows, rate limiting, and wait-time estimation
(baseline moving-average, with an optional Temporal Fusion Transformer
upgrade for quantile-based forecasts).

Run locally:
    pip install -r requirements.txt
    python app.py
Then open http://127.0.0.1:5000  (customer view)
and   http://127.0.0.1:5000/admin (admin/counter view)

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
NOTIFY_WINDOW = 2          # notify when this many people (or fewer) are ahead
RATE_LIMIT_WINDOW = 60     # seconds
RATE_LIMIT_MAX = 5         # max joins per IP per window
RECENT_SERVICE_SAMPLE = 20 # how many past completed tickets to average

SERVICE_TYPES = ["General Service", "Customer Support", "Billing", "Other"]
TFT_ENCODER_LENGTH = 96    # TFT needs this many past 15-min buckets of history

ORG_NAME = os.environ.get("ORG_NAME", "Smart Queue OPD")   # printed on token slips
TIMEZONE = os.environ.get("TIMEZONE", "Asia/Kolkata")      # doctor timings + token times
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DOCTOR_TYPES = ["General", "Specialist"]
TIME_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")

# Sample doctors, added only when the doctor table is empty (admins can edit them).
# (name, type, specialty, working days, start, end)
SAMPLE_DOCTORS = [
    ("Dr. Anand Kumar", "General", "General Medicine", "Mon,Tue,Wed,Thu,Fri,Sat", "09:00", "17:00"),
    ("Dr. Priya Nair", "Specialist", "Cardiology", "Mon,Wed,Fri", "10:00", "14:00"),
    ("Dr. Rahul Mehta", "Specialist", "Dermatology", "Tue,Thu,Sat", "14:00", "18:00"),
    ("Dr. Sana Iqbal", "Specialist", "Pediatrics", "Mon,Tue,Wed,Thu,Fri", "09:00", "13:00"),
]


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


def init_db():
    """Create tables on first startup, upgrade older databases, and seed
    sample counters and doctors if none exist."""
    db = sqlite3.connect(DB_PATH)
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS counter (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS doctor (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            doctor_type TEXT NOT NULL DEFAULT 'General',
            specialty TEXT NOT NULL DEFAULT 'General Medicine',
            days TEXT NOT NULL DEFAULT 'Mon,Tue,Wed,Thu,Fri,Sat',
            start_time TEXT NOT NULL DEFAULT '09:00',
            end_time TEXT NOT NULL DEFAULT '13:00',
            active INTEGER NOT NULL DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS ticket (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            token_no INTEGER NOT NULL,
            name TEXT NOT NULL,
            contact TEXT,
            service_type TEXT NOT NULL DEFAULT 'General Service',
            priority INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'waiting',
            counter_id INTEGER,
            retries INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            called_at TEXT,
            completed_at TEXT,
            notified INTEGER NOT NULL DEFAULT 0,
            doctor_id INTEGER,
            FOREIGN KEY (counter_id) REFERENCES counter(id),
            FOREIGN KEY (doctor_id) REFERENCES doctor(id)
        );

        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_ticket_status ON ticket(status);
        CREATE INDEX IF NOT EXISTS idx_ticket_priority_created
            ON ticket(priority, created_at);
        """
    )
    # Databases created before doctors existed: add the doctor link column.
    columns = [r[1] for r in db.execute("PRAGMA table_info(ticket)")]
    if "doctor_id" not in columns:
        db.execute("ALTER TABLE ticket ADD COLUMN doctor_id INTEGER REFERENCES doctor(id)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_ticket_doctor ON ticket(doctor_id, status)")

    row = db.execute("SELECT COUNT(*) AS c FROM counter").fetchone()
    if row[0] == 0:
        db.execute("INSERT INTO counter (name, active) VALUES ('Counter 1', 1)")
        db.execute("INSERT INTO counter (name, active) VALUES ('Counter 2', 1)")
    if db.execute("SELECT COUNT(*) FROM doctor").fetchone()[0] == 0:
        db.executemany(
            "INSERT INTO doctor (name, doctor_type, specialty, days, start_time, end_time, active) "
            "VALUES (?, ?, ?, ?, ?, ?, 1)",
            SAMPLE_DOCTORS,
        )
    db.commit()
    db.close()


def now_iso():
    return datetime.utcnow().isoformat(timespec="seconds")


def parse_dt(s):
    return datetime.fromisoformat(s) if s else None


# ---------------------------------------------------------------------------
# Time, phone-number and doctor helpers
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


def doctor_status(doc, now=None):
    """Where is this doctor right now? Returns {'state': ..., 'label': ...}."""
    now = now or local_now()
    if not doc["active"]:
        return {"state": "leave", "label": "On leave"}
    if DAY_NAMES[now.weekday()] not in doc["days"].split(","):
        return {"state": "off", "label": "Off today"}
    current = now.strftime("%H:%M")
    if current < doc["start_time"]:
        return {"state": "later", "label": f"Arrives at {fmt_time(doc['start_time'])}"}
    if current >= doc["end_time"]:
        return {"state": "ended", "label": "Done for today"}
    return {"state": "available", "label": "Available now"}


def doctor_dict(db, row, now=None):
    """A doctor row plus display text, today's status and queue numbers."""
    d = dict(row)
    d["days_text"] = days_text(d["days"])
    d["time_text"] = f"{fmt_time(d['start_time'])} \u2013 {fmt_time(d['end_time'])}"
    d["status"] = doctor_status(d, now)
    d["waiting"] = db.execute(
        "SELECT COUNT(*) AS c FROM ticket WHERE doctor_id=? AND status='waiting'", (d["id"],)
    ).fetchone()["c"]
    serving = db.execute(
        "SELECT id, token_no, name FROM ticket WHERE doctor_id=? AND status='serving' LIMIT 1",
        (d["id"],),
    ).fetchone()
    d["serving"] = dict(serving) if serving else None
    return d


def doctor_dicts(db):
    now = local_now()
    rows = db.execute(
        "SELECT * FROM doctor ORDER BY CASE doctor_type WHEN 'General' THEN 0 ELSE 1 END, name"
    ).fetchall()
    return [doctor_dict(db, r, now) for r in rows]


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


def avg_service_seconds(db, doctor_id=None):
    """Average duration of the last RECENT_SERVICE_SAMPLE completed tickets.
    With a doctor_id, that doctor's own recent consultations are used when
    they have any; otherwise the overall average is used."""
    if doctor_id is not None:
        rows = db.execute(
            """SELECT called_at, completed_at FROM ticket
               WHERE status='done' AND doctor_id=?
                 AND called_at IS NOT NULL AND completed_at IS NOT NULL
               ORDER BY completed_at DESC LIMIT ?""",
            (doctor_id, RECENT_SERVICE_SAMPLE),
        ).fetchall()
        if rows:
            return _average_duration(rows)
    rows = db.execute(
        """SELECT called_at, completed_at FROM ticket
           WHERE status='done' AND called_at IS NOT NULL AND completed_at IS NOT NULL
           ORDER BY completed_at DESC LIMIT ?""",
        (RECENT_SERVICE_SAMPLE,),
    ).fetchall()
    if not rows:
        return 180  # default guess (3 min) until real data accumulates
    return _average_duration(rows)


def active_counter_count(db):
    row = db.execute("SELECT COUNT(*) AS c FROM counter WHERE active=1").fetchone()
    return max(row["c"], 1)


def count_ahead(db, priority, created_at, doctor_id=None, ticket_id=None):
    """People waiting ahead of this ticket (in the same doctor's queue if a
    doctor_id is given, otherwise in the whole queue). With a ticket_id,
    tickets created in the same second are ordered by ticket number, the same
    way "Call Next" orders them, so two people never share a position."""
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
    if doctor_id is not None:
        sql += " AND doctor_id = ?"
        params.append(doctor_id)
    return db.execute(sql, params).fetchone()["c"]


def estimate_wait_minutes(db, priority, created_at, doctor_id=None, ticket_id=None):
    """Baseline estimator:
    estimated wait = (people ahead * avg service duration) / servers
    where servers = active counters for the shared queue, or 1 for a single
    doctor's queue (a doctor sees one patient at a time).
    """
    ahead = count_ahead(db, priority, created_at, doctor_id, ticket_id)
    servers = 1 if doctor_id is not None else active_counter_count(db)
    avg_seconds = avg_service_seconds(db, doctor_id)
    seconds = (ahead * avg_seconds) / servers
    return int(seconds // 60)


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
        """SELECT t.*, d.name AS doctor_name FROM ticket t
           LEFT JOIN doctor d ON d.id = t.doctor_id
           WHERE t.status='waiting'
           ORDER BY t.priority DESC, t.created_at ASC, t.id ASC"""
    ).fetchall()


# ---------------------------------------------------------------------------
# TFT wait-time estimation (optional, lazy-loaded)
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


def get_wait_estimate(db, priority, created_at, service_type, doctor_id=None, ticket_id=None):
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

    minutes = estimate_wait_minutes(db, priority, created_at, doctor_id, ticket_id)
    print(f"[WAIT-ESTIMATE] Using baseline moving-average estimate: {minutes} min")
    return {"mode": "baseline", "minutes": minutes}


# ---------------------------------------------------------------------------
# Routes: customer-facing
# ---------------------------------------------------------------------------

def build_ticket_context(db, ticket):
    """Everything a page needs to show one token."""
    doctor = None
    if ticket["doctor_id"] is not None:
        row = db.execute("SELECT * FROM doctor WHERE id=?", (ticket["doctor_id"],)).fetchone()
        if row is not None:
            doctor = doctor_dict(db, row)
    position = wait_estimate = counter_name = None
    if ticket["status"] == "waiting":
        position = count_ahead(
            db, ticket["priority"], ticket["created_at"], ticket["doctor_id"], ticket["id"]
        ) + 1
        wait_estimate = get_wait_estimate(
            db, ticket["priority"], ticket["created_at"], ticket["service_type"],
            ticket["doctor_id"], ticket["id"],
        )
    if ticket["status"] == "serving" and ticket["counter_id"] is not None:
        c = db.execute("SELECT name FROM counter WHERE id=?", (ticket["counter_id"],)).fetchone()
        counter_name = c["name"] if c else None
    return {
        "ticket": ticket,
        "doctor": doctor,
        "position": position,
        "wait_estimate": wait_estimate,
        "counter_name": counter_name,
        "phone_masked": mask_phone(ticket["contact"]),
    }


@app.route("/")
def index():
    db = get_db()
    doctors = doctor_dicts(db)
    doctors_json = [
        {
            "id": d["id"], "name": d["name"], "type": d["doctor_type"],
            "specialty": d["specialty"], "days_text": d["days_text"],
            "time_text": d["time_text"], "status_label": d["status"]["label"],
            "waiting": d["waiting"], "active": bool(d["active"]),
        }
        for d in doctors
    ]
    return render_template(
        "index.html",
        waiting=waiting_list(db),
        service_types=SERVICE_TYPES,
        doctors=doctors,
        doctors_json=doctors_json,
        doctors_enabled=any(d["active"] for d in doctors),
    )


@app.route("/join", methods=["POST"])
def join():
    ip = request.remote_addr or "unknown"
    if is_rate_limited(ip):
        flash("Too many requests from your device. Please wait a minute and try again.", "error")
        return redirect(url_for("index"))

    name = request.form.get("name", "").strip()
    contact = request.form.get("contact", "").strip()
    service_type = request.form.get("service_type", "").strip()
    priority = 1 if request.form.get("priority") == "on" else 0
    doctor_id = request.form.get("doctor_id", type=int)

    # --- input validation ---
    if not name:
        flash("Name is required.", "error")
        return redirect(url_for("index"))
    if len(name) > 100:
        flash("Name is too long (max 100 characters).", "error")
        return redirect(url_for("index"))

    db = get_db()
    active_doctors = db.execute("SELECT * FROM doctor WHERE active=1").fetchall()
    if active_doctors:
        # Doctor mode: the patient picks a doctor and gives a mobile number.
        doctor = next((d for d in active_doctors if d["id"] == doctor_id), None)
        if doctor is None:
            flash("Please select an available doctor.", "error")
            return redirect(url_for("index"))
        phone = normalize_phone(contact)
        if phone is None:
            flash("Please enter a valid 10-digit mobile number.", "error")
            return redirect(url_for("index"))
        existing = db.execute(
            "SELECT id, token_no FROM ticket "
            "WHERE contact=? AND doctor_id=? AND status IN ('waiting','serving') LIMIT 1",
            (phone, doctor["id"]),
        ).fetchone()
        if existing:
            flash(f"You already have an active token (#{existing['token_no']}) with {doctor['name']}.", "info")
            return redirect(url_for("ticket_status", ticket_id=existing["id"]))
        contact = phone
        service_type = doctor["specialty"]
        doctor_pk = doctor["id"]
    else:
        # No doctors set up: behave like the plain queue system.
        if contact and len(contact) > 100:
            flash("Contact is too long (max 100 characters).", "error")
            return redirect(url_for("index"))
        if service_type not in SERVICE_TYPES:
            flash("Please select a valid service type.", "error")
            return redirect(url_for("index"))
        doctor_pk = None

    try:
        token_no = next_token(db)
        cur = db.execute(
            """INSERT INTO ticket
               (token_no, name, contact, service_type, priority, status, created_at, doctor_id)
               VALUES (?, ?, ?, ?, ?, 'waiting', ?, ?)""",
            (token_no, name, contact, service_type, priority, now_iso(), doctor_pk),
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
    position = ctx["position"]
    if ticket["status"] == "waiting":
        # simulated notification when close to being served
        if position <= NOTIFY_WINDOW and not ticket["notified"]:
            db.execute("UPDATE ticket SET notified=1 WHERE id=?", (ticket_id,))
            db.commit()
            print(f"[NOTIFY] {ticket['name']} ({ticket['contact']}): "
                  f"You're #{position} in line, almost your turn!")

    return render_template("ticket.html", **ctx)


@app.route("/ticket/<int:ticket_id>/print")
def print_token(ticket_id):
    """A clean, printable token slip."""
    db = get_db()
    ticket = db.execute("SELECT * FROM ticket WHERE id=?", (ticket_id,)).fetchone()
    if ticket is None:
        abort(404, description="Ticket not found.")
    ctx = build_ticket_context(db, ticket)
    return render_template(
        "print.html", org_name=ORG_NAME,
        printed_at=local_now().strftime("%d %b %Y, %I:%M %p").lstrip("0"), **ctx
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
    position = None
    if ticket["status"] == "waiting":
        position = count_ahead(
            db, ticket["priority"], ticket["created_at"], ticket["doctor_id"], ticket["id"]
        ) + 1
    return jsonify(status=ticket["status"], position=position)


# ---------------------------------------------------------------------------
# Routes: admin / counter operations
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
        """SELECT t.*, d.name AS doctor_name FROM ticket t
           LEFT JOIN doctor d ON d.id = t.doctor_id
           WHERE t.status='serving'"""
    ).fetchall()
    counters = db.execute("SELECT * FROM counter ORDER BY id ASC").fetchall()
    doctors = doctor_dicts(db)

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
        doctors=doctors,
        day_names=DAY_NAMES,
        completed_today=completed_today,
        dead_letter=dead_letter,
        avg_wait=avg_wait,
        queue_length=len(waiting),
    )


@app.route("/admin/call_next", methods=["POST"])
@admin_required
def call_next():
    """Call the next waiting ticket - from one doctor's queue if a doctor_id
    is submitted, otherwise from the whole queue."""
    db = get_db()
    doctor_id = request.form.get("doctor_id", type=int)
    doctor = None
    if doctor_id is not None:
        doctor = db.execute("SELECT * FROM doctor WHERE id=?", (doctor_id,)).fetchone()
        if doctor is None:
            abort(404, description="Doctor not found.")
        busy = db.execute(
            "SELECT token_no FROM ticket WHERE doctor_id=? AND status='serving' LIMIT 1",
            (doctor_id,),
        ).fetchone()
        if busy:
            flash(f"{doctor['name']} is already seeing token #{busy['token_no']}. "
                  f"Complete that token first.", "error")
            return redirect(url_for("admin"))

    counter = pick_counter(db)
    if not counter:
        flash("No active counters available.", "error")
        return redirect(url_for("admin"))

    sql = "SELECT * FROM ticket WHERE status='waiting'"
    params = []
    if doctor_id is not None:
        sql += " AND doctor_id=?"
        params.append(doctor_id)
    sql += " ORDER BY priority DESC, created_at ASC, id ASC LIMIT 1"
    next_ticket = db.execute(sql, params).fetchone()
    if not next_ticket:
        flash("Queue is empty." if doctor is None else f"No one is waiting for {doctor['name']}.", "info")
        return redirect(url_for("admin"))

    # Atomic conditional update: only succeeds if still 'waiting', so two
    # simultaneous "Call Next" clicks can never assign the same ticket twice.
    # For a doctor, it also only succeeds while that doctor has nobody serving.
    update_sql = ("UPDATE ticket SET status='serving', counter_id=?, called_at=? "
                  "WHERE id=? AND status='waiting'")
    update_params = [counter["id"], now_iso(), next_ticket["id"]]
    if doctor_id is not None:
        update_sql += (" AND NOT EXISTS (SELECT 1 FROM ticket t2 "
                       "WHERE t2.doctor_id=? AND t2.status='serving')")
        update_params.append(doctor_id)
    try:
        cur = db.execute(update_sql, update_params)
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
        "UPDATE ticket SET status='done', completed_at=? WHERE id=?",
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
            "UPDATE ticket SET status='dead_letter', retries=? WHERE id=?",
            (retries, ticket_id),
        )
    else:
        db.execute(
            "UPDATE ticket SET status='waiting', counter_id=NULL, called_at=NULL, "
            "retries=? WHERE id=?",
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
# Routes: doctor management (admin only)
# ---------------------------------------------------------------------------

def _read_doctor_form(form):
    """Validate the add/edit doctor form. Returns (values, error_message)."""
    name = form.get("name", "").strip()
    doctor_type = form.get("doctor_type", "General")
    specialty = form.get("specialty", "").strip()
    days = [d for d in DAY_NAMES if d in form.getlist("days")]
    start = form.get("start_time", "").strip()
    end = form.get("end_time", "").strip()

    if not name or len(name) > 100:
        return None, "Doctor name is required (max 100 characters)."
    if doctor_type not in DOCTOR_TYPES:
        return None, "Choose General or Specialist."
    if not specialty and doctor_type == "General":
        specialty = "General Medicine"
    if not specialty:
        return None, "Enter the specialty (for example Cardiology)."
    if len(specialty) > 60:
        return None, "Specialty is too long (max 60 characters)."
    if not days:
        return None, "Select at least one working day."
    if not (TIME_RE.match(start) and TIME_RE.match(end)):
        return None, "Enter valid start and end times."
    if start >= end:
        return None, "End time must be after start time."
    return {
        "name": name, "doctor_type": doctor_type, "specialty": specialty,
        "days": ",".join(days), "start_time": start, "end_time": end,
    }, None


@app.route("/admin/doctors/add", methods=["POST"])
@admin_required
def add_doctor():
    values, error = _read_doctor_form(request.form)
    if error:
        flash(error, "error")
        return redirect(url_for("admin"))
    db = get_db()
    db.execute(
        "INSERT INTO doctor (name, doctor_type, specialty, days, start_time, end_time, active) "
        "VALUES (:name, :doctor_type, :specialty, :days, :start_time, :end_time, 1)",
        values,
    )
    db.commit()
    flash(f"{values['name']} added.", "info")
    return redirect(url_for("admin"))


@app.route("/admin/doctors/<int:doctor_id>/toggle", methods=["POST"])
@admin_required
def toggle_doctor(doctor_id):
    db = get_db()
    if db.execute("SELECT id FROM doctor WHERE id=?", (doctor_id,)).fetchone() is None:
        abort(404, description="Doctor not found.")
    db.execute("UPDATE doctor SET active = 1 - active WHERE id=?", (doctor_id,))
    db.commit()
    return redirect(url_for("admin"))


@app.route("/admin/doctors/<int:doctor_id>/edit", methods=["GET", "POST"])
@admin_required
def edit_doctor(doctor_id):
    db = get_db()
    row = db.execute("SELECT * FROM doctor WHERE id=?", (doctor_id,)).fetchone()
    if row is None:
        abort(404, description="Doctor not found.")
    if request.method == "POST":
        values, error = _read_doctor_form(request.form)
        if error:
            flash(error, "error")
            return redirect(url_for("edit_doctor", doctor_id=doctor_id))
        values["id"] = doctor_id
        db.execute(
            "UPDATE doctor SET name=:name, doctor_type=:doctor_type, specialty=:specialty, "
            "days=:days, start_time=:start_time, end_time=:end_time WHERE id=:id",
            values,
        )
        db.commit()
        flash("Doctor updated.", "info")
        return redirect(url_for("admin"))
    return render_template("admin_doctor_edit.html", doc=dict(row), day_names=DAY_NAMES)


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
