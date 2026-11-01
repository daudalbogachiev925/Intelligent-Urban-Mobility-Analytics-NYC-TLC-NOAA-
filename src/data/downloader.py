"""Download NYC TLC Yellow Taxi parquet files and taxi zone GeoJSON."""
from __future__ import annotations

import logging
from pathlib import Path

import httpx
import polars as pl

from src.config import AppConfig

logger = logging.getLogger(__name__)


def tlc_parquet_url(base_url: str, year: int, month: int) -> str:
    return f"{base_url}/yellow_tripdata_{year}-{month:02d}.parquet"


def download_tlc_month(
    year: int, month: int, cfg: AppConfig, client: httpx.Client | None = None
) -> Path:
    url = tlc_parquet_url(cfg.data.tlc_base_url, year, month)
    out_dir = Path(cfg.data.raw_dir) / "tlc"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"yellow_tripdata_{year}-{month:02d}.parquet"

    if out_path.exists():
        logger.info("Skip existing: %s", out_path)
        return out_path

    logger.info("Downloading %s -> %s", url, out_path)
    close_client = client is None
    client = client or httpx.Client(timeout=120.0, follow_redirects=True)
    try:
        with client.stream("GET", url) as resp:
            resp.raise_for_status()
            with out_path.open("wb") as f:
                for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                    f.write(chunk)
    finally:
        if close_client:
            client.close()
    logger.info("Saved %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)
    return out_path


def download_tlc_range(cfg: AppConfig) -> list[Path]:
    paths: list[Path] = []
    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        for month in cfg.data.tlc_months:
            paths.append(download_tlc_month(cfg.data.tlc_year, month, cfg, client))
    return paths


def download_zones(cfg: AppConfig) -> Path:
    out_dir = Path(cfg.data.raw_dir) / "zones"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "taxi_zones.geojson"
    if out_path.exists():
        logger.info("Skip existing zones: %s", out_path)
        return out_path

    logger.info("Downloading taxi zones from %s", cfg.data.zones_url)
    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        resp = client.get(cfg.data.zones_url)
        resp.raise_for_status()
        out_path.write_bytes(resp.content)
    logger.info("Saved zones: %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)
    return out_path


def inspect_tlc_schema(path: Path) -> dict:
    lf = pl.scan_parquet(path)
    schema = lf.collect_schema()
    return {name: str(dtype) for name, dtype in schema.items()}
