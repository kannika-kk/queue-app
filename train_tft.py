"""
Train a Temporal Fusion Transformer (TFT) to forecast queue length,
using service_type and counter_id as static covariates so the model can
differentiate demand patterns per service type (e.g. "Billing" tends to
be quieter than "General Service").

IMPORTANT: This requires torch + pytorch-forecasting + pytorch-lightning,
which are NOT installed in a lightweight deployment environment (and are
too heavy for Render's free tier). Run this on your own laptop:

    pip install -r requirements-ml.txt
    python generate_synthetic_data.py      # creates queue_history.csv
    python train_tft.py                    # trains and saves the model

Output:
    saved_model/tft_model.ckpt    <- load this in predict.py
    training_forecast_sample.png  <- a plot of real vs predicted (P10/P50/P90)
"""

import pandas as pd
import torch
import matplotlib.pyplot as plt

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping


MAX_ENCODER_LENGTH = 96    # look back 24 hours (96 x 15-min steps)
MAX_PREDICTION_LENGTH = 8  # forecast ahead 2 hours (8 x 15-min steps)


def build_dataset(csv_path="queue_history.csv"):
    df = pd.read_csv(csv_path)
    df["counter_id"] = df["counter_id"].astype(str)
    df["service_type"] = df["service_type"].astype(str)

    # time_idx must be sequential WITHIN each (counter_id, service_type) series
    df = df.sort_values(["counter_id", "service_type", "time_idx"])

    training_cutoff = df["time_idx"].max() - MAX_PREDICTION_LENGTH

    training = TimeSeriesDataSet(
        df[df.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="queue_length",
        group_ids=["counter_id", "service_type"],
        max_encoder_length=MAX_ENCODER_LENGTH,
        max_prediction_length=MAX_PREDICTION_LENGTH,

        # Static covariates: fixed per series, don't change over time
        static_categoricals=["counter_id", "service_type"],

        # Known future inputs: we know these ahead of time for any future timestep
        time_varying_known_reals=["hour", "day_of_week", "is_weekend", "active_counters"],

        # Observed past inputs: only known after the fact
        time_varying_unknown_reals=["queue_length", "priority_ratio", "avg_service_seconds"],

        target_normalizer=GroupNormalizer(groups=["counter_id", "service_type"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training, df, predict=True, stop_randomization=True
    )

    return training, validation, df


def train():
    training, validation, df = build_dataset()

    train_dataloader = training.to_dataloader(train=True, batch_size=64, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=64, num_workers=0)

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=16,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=8,
        loss=QuantileLoss(),           # <-- gives us P10/P50/P90 style outputs
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    print(f"Model has {sum(p.numel() for p in tft.parameters())} parameters")

    early_stop = EarlyStopping(monitor="val_loss", patience=5, mode="min")
    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="auto",
        gradient_clip_val=0.1,
        callbacks=[early_stop],
        enable_progress_bar=True,
    )

    trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    trainer.save_checkpoint("saved_model/tft_model.ckpt")
    print("Saved trained model to saved_model/tft_model.ckpt")

    # Quick sanity-check plot: actual vs predicted quantiles on validation data
    raw_predictions = tft.predict(val_dataloader, mode="raw", return_x=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    tft.plot_prediction(raw_predictions.x, raw_predictions.output, idx=0, ax=ax)
    plt.title("Sample TFT Forecast: Actual vs Predicted (with P10-P90 band)")
    plt.tight_layout()
    plt.savefig("training_forecast_sample.png", dpi=150)
    print("Saved training_forecast_sample.png")


if __name__ == "__main__":
    train()
