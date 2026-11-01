"""Lazy ETL: scan parquet, clean, enrich with weather, aggregate hourly demand per zone.

Uses Polars lazy scan with projection and predicate pushdown:
- only reads required columns (tpep_pickup_datetime, PULocationID, fare_amount, trip_distance)
- skips row groups outside the date window [citation:18]
"""
from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from src.config import AppConfig

logger = logging.getLogger(__name__)

TLC_COLUMNS = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "PULocationID",
    "DOLocationID",
    "passenger_count",
    "trip_distance",
    "fare_amount",
    "total_amount",
]


def scan_tlc(cfg: AppConfig) -> pl.LazyFrame:
    """Scan all TLC parquet files lazily with projection pushdown."""
    pattern = str(Path(cfg.data.raw_dir) / "tlc" / "yellow_tripdata_*.parquet")
    lf = pl.scan_parquet(pattern)
    schema = lf.collect_schema()
    keep = [c for c in TLC_COLUMNS if c in schema.names()]
    lf = lf.select(keep)
    logger.info("Scanned TLC with %s columns", len(keep))
    return lf


def clean_tlc(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Remove invalid rows: null pickup, non-positive fare, out-of-zone IDs."""
    return lf.filter(
        pl.col("tpep_pickup_datetime").is_not_null()
        & (pl.col("fare_amount") > 0)
        & (pl.col("trip_distance") >= 0)
        & pl.col("PULocationID").is_between(1, 263)
    )


def aggregate_hourly_zone(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Aggregate trips to hourly demand per zone + derived metrics."""
    return (
        lf.with_columns(
            pl.col("tpep_pickup_datetime")
            .dt.truncate("1h")
            .alias("hour"),
        )
        .group_by(["hour", "PULocationID"])
        .agg(
            pl.len().alias("demand"),
            pl.col("fare_amount").mean().alias("avg_fare"),
            pl.col("trip_distance").mean().alias("avg_distance"),
            pl.col("passenger_count").mean().alias("avg_passengers"),
        )
        .rename({"PULocationID": "zone_id"})
        .sort(["zone_id", "hour"])
    )


def enrich_weather(lf: pl.LazyFrame, cfg: AppConfig) -> pl.LazyFrame:
    """Join daily weather on date. Weather is daily, so we broadcast to hourly."""
    weather_path = Path(cfg.data.raw_dir) / "noaa" / "ghcnd_daily.parquet"
    if not weather_path.exists():
        logger.warning("Weather file not found; skipping enrichment")
        return lf
    weather = pl.read_parquet(weather_path).with_columns(
        pl.col("date").cast(pl.Date)
    )
    lf = lf.with_columns(pl.col("hour").dt.date().alias("date"))
    lf = lf.join(weather.lazy(), on="date", how="left").drop("date")
    return lf


def build_hourly_panel(cfg: AppConfig) -> pl.DataFrame:
    """Run lazy pipeline and collect to DataFrame. Writes result to parquet."""
    lf = scan_tlc(cfg)
    lf = clean_tlc(lf)
    lf = aggregate_hourly_zone(lf)
    lf = enrich_weather(lf, cfg)
    logger.info("Collecting hourly panel...")
    df = lf.collect(engine="streaming")
    out_path = Path(cfg.data.processed_dir) / "hourly_zone_demand.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    logger.info("Saved hourly panel: %s rows x %s cols -> %s", df.height, df.width, out_path)
    return df
