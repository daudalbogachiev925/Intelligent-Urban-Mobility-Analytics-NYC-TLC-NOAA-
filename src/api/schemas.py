"""Pydantic v2 schemas for API."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    version: str


class ForecastPoint(BaseModel):
    hour: datetime
    forecast: float = Field(..., ge=0.0)


class DemandResponse(BaseModel):
    zone_id: int
    horizon: int
    model: str
    points: list[ForecastPoint]
    total: float


class ZoneKPI(BaseModel):
    zone_id: int
    zone_name: str | None = None
    total_demand: float
    mean_hourly_demand: float
    peak_hourly_demand: float


class WeatherSensitivityPoint(BaseModel):
    bucket: str
    correlation: float


class ModelInfoResponse(BaseModel):
    model_type: str
    n_features: int
    feature_columns: list[str]
    last_training_hour: str | None = None
