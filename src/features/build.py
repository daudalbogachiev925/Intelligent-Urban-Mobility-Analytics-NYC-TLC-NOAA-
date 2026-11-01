"""Build time-series features for hourly zonal demand forecasting.

Features per (zone_id, hour):
- calendar: hour_of_day, day_of_week, is_weekend, month
- lags: demand at t-1, t-2, ..., t-168
- rolling: mean/std/max over 3/6/12/24/168 hours (past only)
- Fourier: daily and weekly seasonality
- weather: temp_max_c, temp_min_c, precip_mm, snow_mm, wind_ms
- zone: zone_id as categorical
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import polars as pl

from src.config import AppConfig, get_app_config

logger = logging.getLogger(__name__)


@dataclass
class FeatureBundle:
    frame: pl.DataFrame
    feature_cols: list[str]
    target_col: str = "demand"


def _fourier(t: np.ndarray, period: int, orders: list[int], prefix: str) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for k in orders:
        out[f"{prefix}_sin_{k}"] = np.sin(2 * np.pi * k * t / period)
        out[f"{prefix}_cos_{k}"] = np.cos(2 * np.pi * k * t / period)
    return out


def build_features(df: pl.DataFrame, cfg: AppConfig | None = None) -> FeatureBundle:
    cfg = cfg or get_app_config()
    fc = cfg.features
    df = df.sort(["zone_id", "hour"])

    # calendar
    df = df.with_columns(
        pl.col("hour").dt.hour().alias("hour_of_day"),
        pl.col("hour").dt.weekday().alias("day_of_week"),
        (pl.col("hour").dt.weekday() >= 6).cast(pl.Int8).alias("is_weekend"),
        pl.col("hour").dt.month().alias("month"),
        (pl.col("hour").dt.hour() // 86_400_000).alias("_dummy"),
    ).drop("_dummy")

    # Fourier: use hour-of-day (period 24) and day-of-week (period 7)
    t_hour = df["hour_of_day"].to_numpy().astype(np.float64)
    t_dow = df["day_of_week"].to_numpy().astype(np.float64)
    daily = _fourier(t_hour, 24, fc.fourier_orders_daily, "daily")
    weekly = _fourier(t_dow, 7, fc.fourier_orders_weekly, "weekly")
    df = pl.concat([df, pl.DataFrame({**daily, **weekly})], how="horizontal")

    # lags
    lag_exprs = [pl.col("demand").shift(L).over("zone_id").alias(f"lag_{L}") for L in fc.lags_hours]
    df = df.with_columns(lag_exprs)

    # rolling (past only: shift(1) before rolling)
    roll_exprs = []
    for W in fc.rolling_hours:
        roll_exprs.append(
            pl.col("demand").shift(1).rolling_mean(W).over("zone_id").alias(f"roll_mean_{W}")
        )
        roll_exprs.append(
            pl.col("demand").shift(1).rolling_std(W).over("zone_id").alias(f"roll_std_{W}")
        )
        roll_exprs.append(
            pl.col("demand").shift(1).rolling_max(W).over("zone_id").alias(f"roll_max_{W}")
        )
    df = df.with_columns(roll_exprs)

    # zone as categorical code
    df = df.with_columns(
        pl.col("zone_id").cast(pl.Categorical).to_physical().cast(pl.Int32).alias("zone_code")
    )

    base_cols = ["zone_code", "hour_of_day", "day_of_week", "is_weekend", "month"]
    weather_cols = [
        c for c in ["temp_max_c", "temp_min_c", "precip_mm", "snow_mm", "wind_ms"]
        if c in df.columns
    ]
    lag_cols = [c for c in df.columns if c.startswith("lag_")]
    roll_cols = [c for c in df.columns if c.startswith("roll_")]
    fourier_cols = [c for c in df.columns if c.startswith(("daily_", "weekly_"))]

    feature_cols = base_cols + weather_cols + lag_cols + roll_cols + fourier_cols

    # drop warmup rows (max lag/rolling window)
    max_lag = max(max(fc.lags_hours), max(fc.rolling_hours))
    df = df.with_columns(pl.int_range(pl.len()).over("zone_id").alias("_idx"))
    df = df.filter(pl.col("_idx") >= max_lag).drop("_idx")

    logger.info("Feature frame: %s rows, %s features", df.height, len(feature_cols))
    return FeatureBundle(frame=df, feature_cols=feature_cols)
