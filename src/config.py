"""Configuration loader: pydantic-settings + YAML."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataConfig(BaseModel):
    tlc_year: int = 2024
    tlc_months: list[int] = Field(default_factory=lambda: [1, 2, 3])
    tlc_base_url: str = "https://d37ci6vzurychx.cloudfront.net/trip-data"
    zones_url: str = "https://data.cityofnewyork.us/api/geospatial/8meu-9t5y?method=export&format=GeoJSON"
    raw_dir: str = "./data/raw"
    processed_dir: str = "./data/processed"


class NOAAConfig(BaseModel):
    station_id: str = "GHCND:USW00094728"
    dataset: str = "GHCND"
    start_date: str = "2024-01-01"
    end_date: str = "2024-03-31"
    datatypes: list[str] = Field(default_factory=lambda: ["TMAX", "TMIN", "PRCP", "SNOW", "AWND"])


class FeaturesConfig(BaseModel):
    lags_hours: list[int] = Field(default_factory=lambda: [1, 2, 3, 6, 12, 24, 48, 168])
    rolling_hours: list[int] = Field(default_factory=lambda: [3, 6, 12, 24, 168])
    fourier_orders_daily: list[int] = Field(default_factory=lambda: [1, 2, 3])
    fourier_orders_weekly: list[int] = Field(default_factory=lambda: [1, 2])


class CVConfig(BaseModel):
    n_folds: int = 7
    horizon_hours: int = 24


class LightGBMConfig(BaseModel):
    n_estimators: int = 600
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_child_samples: int = 30
    subsample: float = 0.9
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1


class ModelConfig(BaseModel):
    lightgbm: LightGBMConfig = Field(default_factory=LightGBMConfig)


class APIConfig(BaseModel):
    default_horizon: int = 24


class AppConfig(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    noaa: NOAAConfig = Field(default_factory=NOAAConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    cv: CVConfig = Field(default_factory=CVConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    api: APIConfig = Field(default_factory=APIConfig)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    noaa_token: str = ""
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"
    data_dir: str = "./data"
    artifacts_dir: str = "./artifacts"
    config_path: str = "./configs/config.yaml"

    def load_app_config(self) -> AppConfig:
        path = Path(self.config_path)
        if not path.exists():
            return AppConfig()
        with path.open("r", encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
        return AppConfig(**raw)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@lru_cache(maxsize=1)
def get_app_config() -> AppConfig:
    return get_settings().load_app_config()


def ensure_dirs() -> None:
    settings = get_settings()
    for p in (settings.data_dir, settings.artifacts_dir):
        Path(p).mkdir(parents=True, exist_ok=True)
    cfg = get_app_config()
    Path(cfg.data.raw_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.data.processed_dir).mkdir(parents=True, exist_ok=True)
