"""Higher-level aggregates: daily, weekly, zone-level KPIs."""
from __future__ import annotations

import polars as pl


def daily_zone(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_columns(pl.col("hour").dt.date().alias("date"))
        .group_by(["date", "zone_id"])
        .agg(
            pl.col("demand").sum().alias("daily_demand"),
            pl.col("avg_fare").mean().alias("avg_fare"),
            pl.col("avg_distance").mean().alias("avg_distance"),
        )
        .sort(["zone_id", "date"])
    )


def zone_kpis(df: pl.DataFrame) -> pl.DataFrame:
    """Per-zone summary: total demand, peak hour, mean demand."""
    return (
        df.group_by("zone_id")
        .agg(
            pl.col("demand").sum().alias("total_demand"),
            pl.col("demand").mean().alias("mean_hourly_demand"),
            pl.col("demand").max().alias("peak_hourly_demand"),
            pl.col("hour").sort_by("demand").last().alias("peak_hour"),
        )
        .sort("total_demand", descending=True)
    )
