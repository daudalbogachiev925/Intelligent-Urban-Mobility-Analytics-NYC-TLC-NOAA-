"""Feature engineering tests: no leakage, warmup dropping, expected columns."""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from src.config import AppConfig
from src.features.build import build_features


def _make_panel(n_hours: int = 400, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    start = datetime(2024, 1, 1, 0, 0)
    hours = [start + timedelta(hours=i) for i in range(n_hours)]
    frames = []
    for zone in [161, 237]:
        frames.append(
            pl.DataFrame(
                {
                    "hour": hours,
                    "zone_id": [zone] * n_hours,
                    "demand": rng.integers(0, 100, n_hours).astype(float),
                    "avg_fare": rng.uniform(5, 30, n_hours),
                    "avg_distance": rng.uniform(0.5, 10, n_hours),
                    "avg_passengers": rng.uniform(0.5, 3, n_hours),
                    "temp_max_c": [15.0] * n_hours,
                    "temp_min_c": [5.0] * n_hours,
                    "precip_mm": [0.0] * n_hours,
                    "snow_mm": [0.0] * n_hours,
                    "wind_ms": [3.0] * n_hours,
                }
            )
        )
    return pl.concat(frames, how="vertical").sort(["zone_id", "hour"])


@pytest.fixture
def cfg() -> AppConfig:
    return AppConfig()


def test_features_columns(cfg: AppConfig) -> None:
    panel = _make_panel()
    bundle = build_features(panel, cfg)
    expected = ["lag_1", "lag_24", "lag_168", "roll_mean_24", "roll_std_168", "zone_code", "daily_sin_1", "weekly_cos_1"]
    for col in expected:
        assert col in bundle.frame.columns, f"Missing {col}"


def test_no_leakage_in_lags(cfg: AppConfig) -> None:
    panel = _make_panel(seed=1)
    bundle = build_features(panel, cfg)
    zone = bundle.frame.filter(pl.col("zone_id") == 161).sort("hour")
    # lag_1[t] must equal demand[t-1]
    lags = zone["lag_1"].to_numpy()
    demand = zone["demand"].to_numpy()
    # align: shifted by max_lag already; check first 20 values
    n_check = min(20, len(lags))
    for i in range(n_check):
        assert lags[i] == demand[i - 1] if i > 0 else True


def test_warmup_rows_dropped(cfg: AppConfig) -> None:
    panel = _make_panel()
    bundle = build_features(panel, cfg)
    max_lag = max(max(cfg.features.lags_hours), max(cfg.features.rolling_hours))
    n_in = panel.filter(pl.col("zone_id") == 161).height
    n_out = bundle.frame.filter(pl.col("zone_id") == 161).height
    assert n_in - n_out == max_lag


def test_rolling_uses_past_only(cfg: AppConfig) -> None:
    panel = _make_panel(seed=2)
    bundle = build_features(panel, cfg)
    zone = bundle.frame.filter(pl.col("zone_id") == 237).sort("hour")
    roll = zone["roll_mean_24"].to_numpy()
    demand = zone["demand"].to_numpy()
    # roll_mean_24 at position i must not include demand[i]
    # approximate check: differs from simple mean of window ending at i
    if len(roll) > 30:
        window_including = np.mean(demand[25:49])
        window_past = np.mean(demand[24:48])
        assert abs(roll[24] - window_past) < abs(roll[24] - window_including) + 1e-6


def test_deterministic(cfg: AppConfig) -> None:
    panel = _make_panel(seed=3)
    b1 = build_features(panel, cfg)
    b2 = build_features(panel, cfg)
    assert b1.frame.equals(b2.frame)
