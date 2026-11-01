"""Model and metrics tests."""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from src.evaluation.metrics import all_metrics, bias, rmse, smape, wape
from src.models.baseline import SeasonalMeanBaseline
from src.models.lightgbm_demand import LightGBMDemandForecaster
from src.config import LightGBMConfig


def _toy_frame(n: int = 500) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    start = datetime(2024, 1, 1)
    hours = [start + timedelta(hours=i) for i in range(n)]
    return pl.DataFrame(
        {
            "hour": hours,
            "zone_id": [161] * n,
            "zone_code": [0] * n,
            "hour_of_day": [h.hour for h in hours],
            "day_of_week": [h.weekday() for h in hours],
            "is_weekend": [int(h.weekday() >= 5) for h in hours],
            "month": [h.month for h in hours],
            "demand": rng.integers(0, 50, n).astype(float),
            "f1": rng.normal(0, 1, n),
            "f2": rng.normal(0, 1, n),
        }
    )


def test_wape_perfect() -> None:
    y = np.array([1.0, 2.0, 3.0])
    assert wape(y, y) == 0.0


def test_wape_known() -> None:
    y = np.array([10.0, 10.0])
    p = np.array([15.0, 10.0])
    assert wape(y, p) == pytest.approx(0.25)


def test_smape_zero() -> None:
    y = np.array([5.0, 6.0])
    assert smape(y, y) == 0.0


def test_rmse() -> None:
    assert rmse(np.array([1.0, 2.0]), np.array([1.0, 4.0])) == pytest.approx(np.sqrt(2.0))


def test_bias_positive() -> None:
    assert bias(np.array([1.0, 1.0]), np.array([2.0, 2.0])) > 0


def test_all_metrics_keys() -> None:
    y = np.array([1.0, 2.0, 3.0])
    m = all_metrics(y, y)
    assert set(m.keys()) == {"wape", "smape", "rmse", "bias"}


def test_baseline_fit_predict() -> None:
    df = _toy_frame(300)
    train = df.head(250)
    test = df.tail(50)
    model = SeasonalMeanBaseline()
    model.fit(train, feature_cols=[])
    preds = model.predict(test, feature_cols=[])
    assert preds.shape == (50,)
    assert (preds >= 0).all()


def test_lightgbm_fit_predict() -> None:
    df = _toy_frame(500)
    train = df.head(400)
    test = df.tail(100)
    feature_cols = ["zone_code", "hour_of_day", "day_of_week", "is_weekend", "month", "f1", "f2"]
    model = LightGBMDemandForecaster(LightGBMConfig(n_estimators=30, learning_rate=0.1))
    model.fit(train, feature_cols, categorical_cols=["zone_code"])
    preds = model.predict(test, feature_cols)
    assert preds.shape == (100,)
    assert (preds >= 0).all()
    fi = model.feature_importance(feature_cols)
    assert fi.height == len(feature_cols)
