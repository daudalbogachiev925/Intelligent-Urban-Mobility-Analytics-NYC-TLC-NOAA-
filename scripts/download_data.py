"""Download TLC parquet, taxi zones, and NOAA GHCND."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ensure_dirs, get_app_config, get_settings
from src.data.downloader import download_tlc_range, download_zones
from src.data.noaa_client import download_noaa

logging.basicConfig(
    level=get_settings().log_level,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("urban.download")


def main() -> int:
    ensure_dirs()
    cfg = get_app_config()
    logger.info("Downloading TLC for %s months=%s", cfg.data.tlc_year, cfg.data.tlc_months)
    tlc_paths = download_tlc_range(cfg)
    logger.info("Downloaded %s TLC files", len(tlc_paths))
    download_zones(cfg)
    try:
        download_noaa(cfg)
    except Exception as exc:
        logger.warning("NOAA download failed: %s (continuing without weather)", exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
