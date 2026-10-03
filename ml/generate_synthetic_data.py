"""
Generate synthetic historical queue data for training the TFT model.

Why synthetic data: a brand-new queue system has no history to learn from.
This script simulates several weeks of realistic queue activity (with
morning/evening peaks, weekday vs weekend patterns, and some random noise)
so the TFT model has enough data to learn meaningful patterns from before
your real app has accumulated its own history.

Each (counter_id, service_type) combination is generated as its own time
series, since service_type is a static covariate the live app also sends
at prediction time (see predict.py's interface contract).

Once your deployed app has collected a few weeks of REAL ticket data, you
should export that instead (see export_real_data() below) and retrain on
it for a model that reflects your actual queue, not a simulation.

Run:
    python generate_synthetic_data.py
Produces:
    queue_history.csv
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG = np.random.default_rng(42)

INTERVAL_MINUTES = 15
DAYS_OF_HISTORY = 42          # 6 weeks
COUNTERS = [1, 2, 3]
SERVICE_TYPES = ["General Service", "Customer Support", "Billing", "Other"]
START_DATE = datetime(2026, 1, 1, 0, 0)

# Different service types have different typical demand levels/shapes.
SERVICE_TYPE_WEIGHT = {
    "General Service": 1.0,
    "Customer Support": 0.7,
    "Billing": 0.5,
    "Other": 0.3,
}


def hourly_base_load(hour: float) -> float:
    """Two peaks: late morning and early evening, quiet overnight."""
    morning_peak = 10 * np.exp(-((hour - 11) ** 2) / (2 * 2.0 ** 2))
    evening_peak = 14 * np.exp(-((hour - 17) ** 2) / (2 * 2.5 ** 2))
    overnight_floor = 0.5
    return overnight_floor + morning_peak + evening_peak


def generate_series(counter_id: int, service_type: str) -> pd.DataFrame:
    n_steps = int(DAYS_OF_HISTORY * 24 * 60 / INTERVAL_MINUTES)
    timestamps = [START_DATE + timedelta(minutes=INTERVAL_MINUTES * i) for i in range(n_steps)]
    weight = SERVICE_TYPE_WEIGHT[service_type]

    rows = []
    for i, ts in enumerate(timestamps):
        hour = ts.hour + ts.minute / 60
        day_of_week = ts.weekday()  # 0 = Monday
        is_weekend = 1 if day_of_week >= 5 else 0

        base = hourly_base_load(hour) * weight
        weekend_factor = 0.6 if is_weekend else 1.0
        noise = RNG.normal(0, 1.0 * weight + 0.2)
        queue_length = max(0, base * weekend_factor + noise)

        priority_ratio = np.clip(RNG.normal(0.2, 0.08), 0, 1)
        active_counters = 3 if 8 <= hour <= 20 else 1
        avg_service_seconds = max(60, RNG.normal(180, 30))

        rows.append({
            "time_idx": i,
            "timestamp": ts,
            "counter_id": str(counter_id),       # static covariate (categorical)
            "service_type": service_type,         # static covariate (categorical)
            "hour": hour,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "active_counters": active_counters,
            "priority_ratio": round(priority_ratio, 3),
            "avg_service_seconds": round(avg_service_seconds, 1),
            "queue_length": round(queue_length, 2),  # <-- forecasting target
        })

    return pd.DataFrame(rows)


def export_real_data(db_path="../queue.db", output_csv="queue_history_real.csv", interval_minutes=15):
    """
    OPTIONAL: once your deployed app has real historical tickets, use this
    instead of synthetic data. It aggregates actual ticket rows from
    queue.db into the same time-bucketed format the TFT model expects,
    split by service_type so each service type keeps its own series.

    Run manually once you have a few weeks of real usage:
        python -c "from generate_synthetic_data import export_real_data; export_real_data()"
    """
    import sqlite3

    conn = sqlite3.connect(db_path)
    tickets = pd.read_sql_query("SELECT * FROM ticket", conn)
    conn.close()

    if tickets.empty:
        print("No real ticket data found yet - keep using synthetic data for now.")
        return

    tickets["created_at"] = pd.to_datetime(tickets["created_at"])
    tickets["bucket"] = tickets["created_at"].dt.floor(f"{interval_minutes}min")

    grouped = tickets.groupby(["bucket", "service_type"]).agg(
        queue_length=("id", "count"),
        priority_ratio=("priority", "mean"),
    ).reset_index()

    grouped["hour"] = grouped["bucket"].dt.hour + grouped["bucket"].dt.minute / 60
    grouped["day_of_week"] = grouped["bucket"].dt.dayofweek
    grouped["is_weekend"] = (grouped["day_of_week"] >= 5).astype(int)
    grouped["counter_id"] = "1"          # adapt if you track per-counter history
    grouped["active_counters"] = 2        # adapt to your real counter count over time
    grouped["avg_service_seconds"] = 180  # adapt: compute real rolling average if available

    # time_idx must be sequential per series (counter_id + service_type group)
    grouped = grouped.sort_values("bucket")
    grouped["time_idx"] = grouped.groupby(["counter_id", "service_type"]).cumcount()

    grouped.to_csv(output_csv, index=False)
    print(f"Exported {len(grouped)} real time-buckets to {output_csv}")


if __name__ == "__main__":
    all_series = pd.concat(
        [generate_series(cid, stype) for cid in COUNTERS for stype in SERVICE_TYPES],
        ignore_index=True,
    )
    all_series.to_csv("queue_history.csv", index=False)
    print(
        f"Generated {len(all_series)} rows across {len(COUNTERS)} counters x "
        f"{len(SERVICE_TYPES)} service types -> queue_history.csv"
    )
    print(all_series.head(10))
