"""Baseline: historical mean demand per (zone, hour_of_day, day_of_week)."""
from __future__ import annotations

import logging

import numpy as np
import polars as pl

from src.models.base import BaseForecaster

logger = logging.getLogger(__name__)


class SeasonalMeanBaseline(BaseForecaster):
    """Predicts historical mean demand for a (zone, hour_of_day, day_of_week) key.

    Falls back to (zone, hour_of_day) mean then global mean if key unseen.
    """

    name = "seasonal_mean"

    def __init__(self) -> None:
        self.key_mean: pl.DataFrame | None = None
        self.zone_hour_mean: pl.DataFrame | None = None
        self.global_mean: float = 0.0

    def fit(
        self, train: pl.DataFrame, feature_cols: list[str], target: str = "demand"
    ) -> "SeasonalMeanBaseline":
        df = train.select(
            ["zone_id", "hour_of_day", "day_of_week", target]
        )
        self.key_mean = (
            df.group_by(["zone_id", "hour_of_day", "day_of_week"])
            .agg(pl.col(target).mean().alias("_pred"))
        )
        self.zone_hour_mean = (
            df.group_by(["zone_id", "hour_of_day"])
            .agg(pl.col(target).mean().alias("_fallback1"))
        )
        self.global_mean = float(df[target].mean() or 0.0)
        logger.info(
            "SeasonalMeanBaseline fitted: keys=%s, fallbacks=%s, global=%.2f",
            self.key_mean.height,
            self.zone_hour_mean.height,
            self.global_mean,
        )
        return self

    def predict(self, df: pl.DataFrame, feature_cols: list[str]) -> np.ndarray:
        if self.key_mean is None or self.zone_hour_mean is None:
            raise RuntimeError("Model not fitted")
        base = df.select(["zone_id", "hour_of_day", "day_of_week"])
        merged = (
            base.join(self.key_mean, on=["zone_id", "hour_of_day", "day_of_week"], how="left")
            .join(self.zone_hour_mean, on=["zone_id", "hour_of_day"], how="left")
            .with_columns(
                pl.coalesce(
                    [pl.col("_pred"), pl.col("_fallback1"), pl.lit(self.global_mean)]
                ).alias("_final")
            )
        )
        return np.clip(merged["_final"].to_numpy().astype(np.float64), 0.0, None)
