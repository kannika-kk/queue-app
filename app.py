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
import hmac
import time
import sqlite3
from datetime import datetime, timedelta
from collections import defaultdict
from functools import wraps
from pathlib import Path

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
    """Create tables on first startup and seed 2 sample counters if none exist."""
    db = sqlite3.connect(DB_PATH)
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS counter (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
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
            FOREIGN KEY (counter_id) REFERENCES counter(id)
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
    row = db.execute("SELECT COUNT(*) AS c FROM counter").fetchone()
    if row[0] == 0:
        db.execute("INSERT INTO counter (name, active) VALUES ('Counter 1', 1)")
        db.execute("INSERT INTO counter (name, active) VALUES ('Counter 2', 1)")
    db.commit()
    db.close()


def now_iso():
    return datetime.utcnow().isoformat(timespec="seconds")


def parse_dt(s):
    return datetime.fromisoformat(s) if s else None


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

def avg_service_seconds(db):
    """Average duration of the last RECENT_SERVICE_SAMPLE completed tickets."""
    rows = db.execute(
        """SELECT called_at, completed_at FROM ticket
           WHERE status='done' AND called_at IS NOT NULL AND completed_at IS NOT NULL
           ORDER BY completed_at DESC LIMIT ?""",
        (RECENT_SERVICE_SAMPLE,),
    ).fetchall()
    if not rows:
        return 180  # default guess (3 min) until real data accumulates
    durations = [
        (parse_dt(r["completed_at"]) - parse_dt(r["called_at"])).total_seconds()
        for r in rows
    ]
    return sum(durations) / len(durations) if durations else 180


def active_counter_count(db):
    row = db.execute("SELECT COUNT(*) AS c FROM counter WHERE active=1").fetchone()
    return max(row["c"], 1)


def count_ahead(db, priority, created_at):
    row = db.execute(
        """SELECT COUNT(*) AS c FROM ticket
           WHERE status='waiting' AND (
               priority > ? OR (priority = ? AND created_at < ?)
           )""",
        (priority, priority, created_at),
    ).fetchone()
    return row["c"]


def estimate_wait_minutes(db, priority, created_at):
    """Baseline estimator:
    estimated wait = (people ahead * avg service duration) / active counters
    """
    ahead = count_ahead(db, priority, created_at)
    counters = active_counter_count(db)
    avg_seconds = avg_service_seconds(db)
    seconds = (ahead * avg_seconds) / counters
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
        "SELECT * FROM ticket WHERE status='waiting' ORDER BY priority DESC, created_at ASC"
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


def get_wait_estimate(db, priority, created_at, service_type):
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

    minutes = estimate_wait_minutes(db, priority, created_at)
    print(f"[WAIT-ESTIMATE] Using baseline moving-average estimate: {minutes} min")
    return {"mode": "baseline", "minutes": minutes}


# ---------------------------------------------------------------------------
# Routes: customer-facing
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    db = get_db()
    return render_template(
        "index.html", waiting=waiting_list(db), service_types=SERVICE_TYPES
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

    # --- input validation ---
    if not name:
        flash("Name is required.", "error")
        return redirect(url_for("index"))
    if len(name) > 100:
        flash("Name is too long (max 100 characters).", "error")
        return redirect(url_for("index"))
    if contact and len(contact) > 100:
        flash("Contact is too long (max 100 characters).", "error")
        return redirect(url_for("index"))
    if service_type not in SERVICE_TYPES:
        flash("Please select a valid service type.", "error")
        return redirect(url_for("index"))

    try:
        db = get_db()
        token_no = next_token(db)
        cur = db.execute(
            """INSERT INTO ticket
               (token_no, name, contact, service_type, priority, status, created_at)
               VALUES (?, ?, ?, ?, ?, 'waiting', ?)""",
            (token_no, name, contact, service_type, priority, now_iso()),
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

    position = None
    wait_estimate = None
    if ticket["status"] == "waiting":
        position = count_ahead(db, ticket["priority"], ticket["created_at"]) + 1
        wait_estimate = get_wait_estimate(
            db, ticket["priority"], ticket["created_at"], ticket["service_type"]
        )

        # simulated notification when close to being served
        if position <= NOTIFY_WINDOW and not ticket["notified"]:
            db.execute("UPDATE ticket SET notified=1 WHERE id=?", (ticket_id,))
            db.commit()
            print(f"[NOTIFY] {ticket['name']} ({ticket['contact']}): "
                  f"You're #{position} in line, almost your turn!")

    return render_template(
        "ticket.html", ticket=ticket, position=position, wait_estimate=wait_estimate
    )


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
        position = count_ahead(db, ticket["priority"], ticket["created_at"]) + 1
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
    serving = db.execute("SELECT * FROM ticket WHERE status='serving'").fetchall()
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
        completed_today=completed_today,
        dead_letter=dead_letter,
        avg_wait=avg_wait,
        queue_length=len(waiting),
    )


@app.route("/admin/call_next", methods=["POST"])
@admin_required
def call_next():
    db = get_db()
    counter = pick_counter(db)
    if not counter:
        flash("No active counters available.", "error")
        return redirect(url_for("admin"))

    next_ticket = db.execute(
        "SELECT * FROM ticket WHERE status='waiting' ORDER BY priority DESC, created_at ASC LIMIT 1"
    ).fetchone()
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
