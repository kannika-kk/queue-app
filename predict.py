"""
Loads the trained TFT model and exposes predict_wait_tft(), used by the
Flask app's /ticket/<id> route to show a quantile-based wait estimate
instead of a plain average.

INTERFACE CONTRACT
-------------------
predict_wait_tft() accepts a pandas DataFrame covering the most recent
TFT_ENCODER_LENGTH (96) time buckets, with exactly these columns - the
same ones produced by generate_synthetic_data.py and by app.py's
build_recent_history_df():

    time_idx              int    sequential index, 0..N-1
    counter_id            str    static covariate (e.g. "1")
    service_type          str    static covariate - MUST be one of the
                                  same values used by the live app:
                                  "General Service", "Customer Support",
                                  "Billing", "Other"
    hour                  float  known future input (0-23.75)
    day_of_week           int    known future input (0=Mon..6=Sun)
    is_weekend            int    known future input (0/1)
    active_counters       int    known future input - current active
                                  counter count from the live app
    priority_ratio        float  observed past input
    avg_service_seconds   float  observed past input
    queue_length          float  observed past input / forecasting target

This keeps the model's inputs consistent with exactly what the live
Flask app can supply in real time: recent queue history, the current
active counter count, the ticket's service_type, and current time
features (hour/day/weekend derived from "now").

FAIL-SAFE DESIGN
-----------------
This module is written to fail safe: if torch/pytorch-forecasting aren't
installed (e.g. on a lightweight Render deployment) or the trained model
file doesn't exist yet, importing this module raises ImportError /
FileNotFoundError, which app.py catches and falls back to the simple
moving-average estimate automatically. The app never crashes either way.
"""

from pathlib import Path

MODEL_PATH = Path(__file__).parent / "saved_model" / "tft_model.ckpt"
VALID_SERVICE_TYPES = ["General Service", "Customer Support", "Billing", "Other"]

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"No trained TFT model found at {MODEL_PATH}. "
        "Run generate_synthetic_data.py then train_tft.py on a machine "
        "with torch + pytorch-forecasting installed (see requirements-ml.txt)."
    )

# These imports are intentionally placed after the "file exists" check so
# that environments without torch installed fail with a clear, early error
# (handled by app.py's try/except) rather than a confusing import crash.
import torch                                                # noqa: E402
import pandas as pd                                          # noqa: E402
from pytorch_forecasting import TemporalFusionTransformer    # noqa: E402

_model = TemporalFusionTransformer.load_from_checkpoint(str(MODEL_PATH))
_model.eval()


def predict_wait_tft(recent_history_df: "pd.DataFrame") -> dict:
    """
    Given a DataFrame of the most recent time buckets (see interface
    contract above), returns a quantile forecast of queue length for the
    next prediction steps.

    Returns:
        {
          "p10": [...],  # low (optimistic) estimate per future interval
          "p50": [...],  # median (most likely) estimate
          "p90": [...],  # high (conservative) estimate
        }
    """
    if "service_type" not in recent_history_df.columns:
        raise ValueError("recent_history_df must include a 'service_type' column.")

    bad_types = set(recent_history_df["service_type"].unique()) - set(VALID_SERVICE_TYPES)
    if bad_types:
        raise ValueError(
            f"Unknown service_type value(s) {bad_types}; must be one of {VALID_SERVICE_TYPES}."
        )

    with torch.no_grad():
        raw_predictions = _model.predict(recent_history_df, mode="quantiles")

    # raw_predictions shape: [n_series, prediction_length, n_quantiles]
    # Default TFT quantiles are [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98] unless
    # customized in train_tft.py's QuantileLoss() - adjust indices below if changed.
    quantiles = raw_predictions[0]  # first (only) series in this batch
    return {
        "p10": quantiles[:, 1].tolist(),
        "p50": quantiles[:, 3].tolist(),
        "p90": quantiles[:, 5].tolist(),
    }


def minutes_from_queue_length(predicted_queue_length: float, avg_service_seconds: float, active_counters: int) -> int:
    """
    Converts a predicted queue length into an estimated wait in minutes.

    Conversion formula (documented here and in README.md):
        estimated_wait_seconds = (predicted_queue_length * avg_service_seconds)
                                  / active_counters
        estimated_wait_minutes = estimated_wait_seconds // 60

    This mirrors the baseline formula in app.py's estimate_wait_minutes(),
    so TFT and the baseline estimator stay directly comparable.
    """
    seconds = (predicted_queue_length * avg_service_seconds) / max(active_counters, 1)
    return int(seconds // 60)
