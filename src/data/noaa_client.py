"""NOAA Climate Data Online (GHCND) client with pagination."""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import httpx
import polars as pl

from src.config import AppConfig, get_settings

logger = logging.getLogger(__name__)

NOAA_BASE = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"


def fetch_ghcnd_daily(
    station_id: str,
    start_date: str,
    end_date: str,
    datatypes: list[str],
    token: str,
    client: httpx.Client | None = None,
) -> pl.DataFrame:
    """Fetch daily summaries from NOAA CDO. Handles 1-year limit and pagination."""
    if not token:
        raise RuntimeError("NOAA_TOKEN is not set. Get one at https://www.ncdc.noaa.gov/cdo-web/token")

    close_client = client is None
    client = client or httpx.Client(timeout=60.0)
    headers = {"token": token}

    all_records: list[dict] = []
    limit = 1000
    offset = 1

    try:
        while True:
            params = {
                "datasetid": "GHCND",
                "stationid": station_id,
                "startdate": start_date,
                "enddate": end_date,
                "datatypeid": ",".join(datatypes),
                "limit": limit,
                "offset": offset,
                "units": "metric",
            }
            resp = client.get(NOAA_BASE, params=params, headers=headers)
            if resp.status_code == 429:
                logger.warning("NOAA rate limit hit; sleeping 5s")
                import time
                time.sleep(5)
                continue
            resp.raise_for_status()
            payload = resp.json()
            results = payload.get("results", [])
            if not results:
                break
            all_records.extend(results)
            logger.info("NOAA fetched %s records (offset=%s)", len(results), offset)
            if len(results) < limit:
                break
            offset += limit
    finally:
        if close_client:
            client.close()

    if not all_records:
        return pl.DataFrame()

    df = pl.DataFrame(all_records)
    # pivot long (datatype/value) to wide per date
    wide = (
        df.select(["date", "datatype", "value"])
        .with_columns(pl.col("date").str.slice(0, 10).str.strptime(pl.Date, "%Y-%m-%d"))
        .pivot(index="date", on="datatype", values="value", aggregate_function="first")
        .sort("date")
    )
    # rename columns for clarity
    rename_map = {
        "TMAX": "temp_max_c",
        "TMIN": "temp_min_c",
        "PRCP": "precip_mm",
        "SNOW": "snow_mm",
        "AWND": "wind_ms",
    }
    wide = wide.rename({k: v for k, v in rename_map.items() if k in wide.columns})
    logger.info("NOAA wide frame: %s rows x %s cols", wide.height, wide.width)
    return wide


def download_noaa(cfg: AppConfig) -> Path:
    settings = get_settings()
    out_dir = Path(cfg.data.raw_dir) / "noaa"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ghcnd_daily.parquet"
    if out_path.exists():
        logger.info("Skip existing NOAA: %s", out_path)
        return out_path

    df = fetch_ghcnd_daily(
        station_id=cfg.noaa.station_id,
        start_date=cfg.noaa.start_date,
        end_date=cfg.noaa.end_date,
        datatypes=cfg.noaa.datatypes,
        token=settings.noaa_token,
    )
    df.write_parquet(out_path)
    logger.info("Saved NOAA: %s (%s rows)", out_path, df.height)
    return out_path
