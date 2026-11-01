"""NYC Taxi Zones: load GeoJSON, provide lookup and geometry access."""
from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import polars as pl

from src.config import AppConfig

logger = logging.getLogger(__name__)


def load_zones_geodataframe(cfg: AppConfig) -> gpd.GeoDataFrame:
    path = Path(cfg.data.raw_dir) / "zones" / "taxi_zones.geojson"
    if not path.exists():
        raise FileNotFoundError(f"Zones not found at {path}. Run download first.")
    gdf = gpd.read_file(path)
    gdf = gdf.rename(columns={"LocationID": "zone_id", "zone": "zone_name", "borough": "borough"})
    gdf["zone_id"] = gdf["zone_id"].astype(int)
    logger.info("Loaded zones: %s features", len(gdf))
    return gdf


def zones_to_polars(gdf: gpd.GeoDataFrame) -> pl.DataFrame:
    return pl.from_pandas(gdf.drop(columns=["geometry"]).rename(columns=str))


def zone_lookup(cfg: AppConfig) -> dict[int, str]:
    gdf = load_zones_geodataframe(cfg)
    return dict(zip(gdf["zone_id"].astype(int), gdf["zone_name"]))
