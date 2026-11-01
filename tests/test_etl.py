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
            "fare_amount": [10.0, 12.0, 15.0, 0.0, 20.0, 22.0],  # zero-fare should be removed
            "trip_distance": [1.0, 2.0, 1.5, 1.0, 3.0, 4.0],
            "passenger_count": [1, 2, 1, 0, 1, 2],
        }
    )


def test_clean_removes_zero_fare() -> None:
    lf = _raw_frame().lazy()
    out = clean_tlc(lf).collect()
    assert (out["fare_amount"] > 0).all()


def test_aggregate_by_hour_zone() -> None:
    lf = _raw_frame().lazy()
    out = aggregate_hourly_zone(clean_tlc(lf)).collect()
    # 161 zone: 3 trips in hour 08:00; zone 237: 2 trips in hour
