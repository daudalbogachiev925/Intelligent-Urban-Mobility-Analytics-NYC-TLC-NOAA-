"""FastAPI app: /health, /model-info, /demand, /zones/top, /weather-sensitivity."""
from __future__ import annotations

import logging
import pickle
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import AsyncIterator

import numpy as np
import polars as pl
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    DemandResponse,
    ForecastPoint,
    HealthResponse,
    ModelInfoResponse,
    WeatherSensitivityPoint,
    ZoneKPI,
)
from src.config import get_app_config, get_settings
from src.features.build import build_features
from src.viz.choropleth import plot_zone_demand_choropleth

logger = logging.getLogger(__name__)
logging.basicConfig(level=get_settings().log_level)


class Artifacts:
    def __init__(self, path: Path):
        self.path = path
        self.model = None
        self.feature_cols: list[str] = []
        self.metadata: dict = {}
        self.panel: pl.DataFrame | None = None
        self.zones: pl.DataFrame | None = None

    def load(self) -> "Artifacts":
        model_path = self.path / "model.pkl"
        meta_path = self.path / "metadata.pkl"
        panel_path = self.path / "panel.parquet"
        zones_path = self.path / "zones.parquet"

        if model_path.exists():
            with model_path.open("rb") as f:
                payload = pickle.load(f)
            self.model = payload["model"]
            self.feature_cols = payload["feature_cols"]
            logger.info("Loaded model: %s", type(self.model).__name__)
        else:
            logger.warning("Model missing at %s", model_path)

        if meta_path.exists():
            with meta_path.open("rb") as f:
                self.metadata = pickle.load(f)

        if panel_path.exists():
            self.panel = pl.read_parquet(panel_path)
            logger.info("Loaded panel: %s rows", self.panel.height)

        if zones_path.exists():
            self.zones = pl.read_parquet(zones_path)

        return self

    @property
    def ready(self) -> bool:
        return self.model is not None and self.panel is not None


@lru_cache(maxsize=1)
def get_artifacts() -> Artifacts:
    settings = get_settings()
    return Artifacts(Path(settings.artifacts_dir)).load()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    get_artifacts()
    yield


app = FastAPI(
    title="Urban Mobility Analytics API",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=app.version)


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info(artifacts: Artifacts = Depends(get_artifacts)) -> ModelInfoResponse:
    if not artifacts.ready:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return ModelInfoResponse(
        model_type=type(artifacts.model).__name__,
        n_features=len(artifacts.feature_cols),
        feature_columns=artifacts.feature_cols,
        last_training_hour=artifacts.metadata.get("last_hour"),
    )


def _future_frame(
    artifacts: Artifacts, zone_id: int, horizon: int, start: datetime | None
) -> pl.DataFrame:
    """Construct feature frame for horizon hours ahead of training end."""
    assert artifacts.panel is not None
    panel = artifacts.panel
    zone_hist = panel.filter(pl.col("zone_id") == zone_id).sort("hour")
    if zone_hist.height == 0:
        raise HTTPException(status_code=404, detail=f"Unknown zone: {zone_id}")

    last_hour = zone_hist["hour"].max()
    start_hour = start or (last_hour + timedelta(hours=1))

    # weather: use last 30 days averages as naive projection
    tail = zone_hist.tail(24 * 30)
    weather_means = {}
    for c in ["temp_max_c", "temp_min_c", "precip_mm", "snow_mm", "wind_ms"]:
        if c in tail.columns:
            val = tail[c].mean()
            weather_means[c] = float(val) if val is not None else 0.0

    future_hours = [start_hour + timedelta(hours=i) for i in range(horizon)]
    future = pl.DataFrame(
        {
            "hour": future_hours,
            "zone_id": [zone_id] * horizon,
            "demand": [0.0] * horizon,
            "avg_fare": [float(zone_hist["avg_fare"].mean() or 0.0)] * horizon,
            "avg_distance": [float(zone_hist["avg_distance"].mean() or 0.0)] * horizon,
            "avg_passengers": [float(zone_hist["avg_passengers"].mean() or 0.0)] * horizon,
            **{k: [v] * horizon for k, v in weather_means.items()},
        }
    )
    combined = pl.concat([zone_hist.select(future.columns), future], how="vertical").sort("hour")
    bundle = build_features(combined)
    feature_frame = bundle.frame.filter(pl.col("hour") >= pl.lit(start_hour))
    if feature_frame.height == 0:
        raise HTTPException(status_code=500, detail="Could not build features")
    return feature_frame


@app.get("/demand", response_model=DemandResponse)
def demand(
    zone: int = Query(..., description="NYC taxi zone ID"),
    horizon: int = Query(24, ge=1, le=168),
    start: datetime | None = Query(None),
    artifacts: Artifacts = Depends(get_artifacts),
) -> DemandResponse:
    if not artifacts.ready:
        raise HTTPException(status_code=503, detail="Model not loaded")
    frame = _future_frame(artifacts, zone, horizon, start)
    try:
        preds = artifacts.model.predict(frame, artifacts.feature_cols)
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    preds = np.clip(np.asarray(preds, dtype=float), 0.0, None)
    hours = frame["hour"].to_list()
    points = [ForecastPoint(hour=h, forecast=float(v)) for h, v in zip(hours, preds)]
    return DemandResponse(
        zone_id=zone,
        horizon=horizon,
        model=type(artifacts.model).__name__,
        points=points,
        total=float(preds.sum()),
    )


@app.get("/zones/top", response_model=list[ZoneKPI])
def zones_top(
    n: int = Query(20, ge=1, le=263),
    artifacts: Artifacts = Depends(get_artifacts),
) -> list[ZoneKPI]:
    if artifacts.panel is None:
        raise HTTPException(status_code=503, detail="Panel not loaded")
    agg = (
        artifacts.panel.group_by("zone_id")
        .agg(
            pl.col("demand").sum().alias("total_demand"),
            pl.col("demand").mean().alias("mean_hourly_demand"),
            pl.col("demand").max().alias("peak_hourly_demand"),
        )
        .sort("total_demand", descending=True)
        .head(n)
    )
    if artifacts.zones is not None and "zone_name" in artifacts.zones.columns:
        agg = agg.join(artifacts.zones.select(["zone_id", "zone_name"]), on="zone_id", how="left")
    else:
        agg = agg.with_columns(pl.lit(None).alias("zone_name"))
    return [
        ZoneKPI(
            zone_id=int(r["zone_id"]),
            zone_name=r.get("zone_name"),
            total_demand=float(r["total_demand"]),
            mean_hourly_demand=float(r["mean_hourly_demand"]),
            peak_hourly_demand=float(r["peak_hourly_demand"]),
        )
        for r in agg.iter_rows(named=True)
    ]


@app.get("/weather-sensitivity", response_model=list[WeatherSensitivityPoint])
def weather_sensitivity(
    artifacts: Artifacts = Depends(get_artifacts),
) -> list[WeatherSensitivityPoint]:
    if artifacts.panel is None:
        raise HTTPException(status_code=503, detail="Panel not loaded")
    if "precip_mm" not in artifacts.panel.columns or "temp_max_c" not in artifacts.panel.columns:
        raise HTTPException(status_code=404, detail="Weather columns not present")

    df = artifacts.panel.select(["demand", "temp_max_c", "precip_mm", "wind_ms"]).drop_nulls()
    if df.height < 30:
        raise HTTPException(status_code=400, detail="Insufficient data")
    corr_temp = float(np.corrcoef(df["demand"], df["temp_max_c"])[0, 1])
    corr_precip = float(np.corrcoef(df["demand"], df["precip_mm"])[0, 1])
    corr_wind = float(np.corrcoef(df["demand"], df["wind_ms"])[0, 1])
    return [
        WeatherSensitivityPoint(bucket="temp_max_c", correlation=corr_temp),
        WeatherSensitivityPoint(bucket="precip_mm", correlation=corr_precip),
        WeatherSensitivityPoint(bucket="wind_ms", correlation=corr_wind),
    ]
