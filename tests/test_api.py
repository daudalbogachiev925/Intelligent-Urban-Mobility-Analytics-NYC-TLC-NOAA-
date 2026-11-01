"""FastAPI tests with mocked artifacts."""
from __future__ import annotations

import pickle
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient


class _DummyModel:
    def predict(self, df: pl.DataFrame, cols: list[str]) -> np.ndarray:
        return np.full(df.height, 3.0)


def _make_panel(n_hours: int = 200) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    start = datetime(2024, 1, 1)
    hours = [start + timedelta(hours=i) for i in range(n_hours)]
    return pl.DataFrame(
        {
            "hour": hours,
            "zone_id": [161] * n_hours,
            "demand": rng.integers(0, 30, n_hours).astype(float),
            "avg_fare": rng.uniform(5, 20, n_hours),
            "avg_distance": rng.uniform(1, 5, n_hours),
            "avg_passengers": rng.uniform(1, 2, n_hours),
            "temp_max_c": [15.0] * n_hours,
            "temp_min_c": [5.0] * n_hours,
            "precip_mm": [0.0] * n_hours,
            "snow_mm": [0.0] * n_hours,
            "wind_ms": [3.0] * n_hours,
        }
    )


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    panel = _make_panel()
    panel_path = tmp_path / "panel.parquet"
    panel.write_parquet(panel_path)

    zones = pl.DataFrame({"zone_id": [161], "zone_name": ["Midtown Center"], "borough": ["Manhattan"]})
    zones.write_parquet(tmp_path / "zones.parquet")

    feature_cols = [
        "zone_code", "hour_of_day", "day_of_week", "is_weekend", "month",
        "temp_max_c", "temp_min_c", "precip_mm", "snow_mm", "wind_ms",
        "lag_1", "lag_2", "lag_3", "lag_6", "lag_12", "lag_24", "lag_48", "lag_168",
        "roll_mean_3", "roll_std_3", "roll_max_3",
        "roll_mean_6", "roll_std_6", "roll_max_6",
        "roll_mean_12", "roll_std_12", "roll_max_12",
        "roll_mean_24", "roll_std_24", "roll_max_24",
        "roll_mean_168", "roll_std_168", "roll_max_168",
        "daily_sin_1", "daily_cos_1", "daily_sin_2", "daily_cos_2",
        "daily_sin_3", "daily_cos_3",
        "weekly_sin_1", "weekly_cos_1", "weekly_sin_2", "weekly_cos_2",
    ]

    with (tmp_path / "model.pkl").open("wb") as f:
        pickle.dump({"model": _DummyModel(), "feature_cols": feature_cols}, f)
    with (tmp_path / "metadata.pkl").open("wb") as f:
        pickle.dump({"last_hour": str(panel["hour"].max()), "cv_metrics": {}}, f)

    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    from src.config import get_settings, get_app_config
    from src.api import main as api_main

    get_settings.cache_clear()
    get_app_config.cache_clear()
    api_main.get_artifacts.cache_clear()

    with TestClient(api_main.app) as c:
        yield c


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_model_info(client: TestClient) -> None:
    r = client.get("/model-info")
    assert r.status_code == 200
    data = r.json()
    assert data["model_type"] == "_DummyModel"
    assert data["n_features"] > 0


def test_demand_ok(client: TestClient) -> None:
    r = client.get("/demand", params={"zone": 161, "horizon": 6})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["zone_id"] == 161
    assert data["horizon"] == 6
    assert len(data["points"]) == 6
    assert data["total"] == pytest.approx(18.0)


def test_demand_unknown_zone(client: TestClient) -> None:
    r = client.get("/demand", params={"zone": 9999, "horizon": 6})
    assert r.status_code == 404


def test_demand_invalid_horizon(client: TestClient) -> None:
    r = client.get("/demand", params={"zone": 161, "horizon": 0})
    assert r.status_code == 422


def test_zones_top(client: TestClient) -> None:
    r = client.get("/zones/top", params={"n": 5})
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    assert data[0]["zone_id"] == 161


def test_weather_sensitivity(client: TestClient) -> None:
    r = client.get("/weather-sensitivity")
    # either 200 with correlations or 400 if insufficient data
    assert r.status_code in (200, 400)
    if r.status_code == 200:
        data = r.json()
        assert len(data) == 3
        for item in data:
            assert -1.0 <= item["correlation"] <= 1.0
