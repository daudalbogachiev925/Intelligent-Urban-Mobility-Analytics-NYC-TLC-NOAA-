"""ETL tests: lazy cleaning, aggregation, weather enrichment."""
from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest

from src.etl.lazy_pipeline import aggregate_hourly_zone, clean_tlc


def _raw_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "tpep_pickup_datetime": [
                datetime(2024, 1, 1, 8, 0),
                datetime(2024, 1, 1, 8, 15),
                datetime(2024, 1, 1, 8, 45),
                datetime(2024, 1, 1, 9, 0),
                datetime(2024, 1, 1, 9, 30),
                datetime(2024, 1, 1, 9, 45),
            ],
            "PULocationID": [161, 161, 161, 161, 237, 237],
            "fare_amount": [10.0, 12.0, 15.0, 0.0, 20.0, 22.0],
            "trip_distance": [1.0, 2.0, 1.5, 1.0, 3.0, 4.0],
            "passenger_count": [1, 2, 1, 0, 1, 2],
        }
    )


def test_clean_removes_zero_fare() -> None:
    lf = _raw_frame().lazy()
    out = clean_tlc(lf).collect()
    assert (out["fare_amount"] > 0).all()
    assert out.height == 5


def test_clean_removes_out_of_range_zone() -> None:
    df = _raw_frame().with_columns(
        pl.when(pl.arange(0, 6) == 0).then(pl.lit(999)).otherwise(pl.col("PULocationID")).alias("PULocationID")
    )
    out = clean_tlc(df.lazy()).collect()
    assert (out["PULocationID"] <= 263).all()


def test_aggregate_by_hour_zone() -> None:
    lf = _raw_frame().lazy()
    out = aggregate_hourly_zone(clean_tlc(lf)).collect().sort(["zone_id", "hour"])
    # zone 161: 3 trips at 08:00
    z161 = out.filter(pl.col("zone_id") == 161)
    assert z161.height == 1
    row = z161.to_dicts()[0]
    assert row["demand"] == 3
    assert row["avg_fare"] == pytest.approx((10.0 + 12.0 + 15.0) / 3)

    # zone 237: 2 trips at 09:00
    z237 = out.filter(pl.col("zone_id") == 237)
    assert z237.height == 1
    assert z237.to_dicts()[0]["demand"] == 2


def test_aggregate_preserves_avg_distance() -> None:
    lf = _raw_frame().lazy()
    out = aggregate_hourly_zone(clean_tlc(lf)).collect()
    z161 = out.filter(pl.col("zone_id") == 161).to_dicts()[0]
    assert z161["avg_distance"] == pytest.approx((1.0 + 2.0 + 1.5) / 3)
