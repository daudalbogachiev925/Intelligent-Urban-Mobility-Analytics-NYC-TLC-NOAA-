"""Build the hourly zone panel from raw TLC + NOAA."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ensure_dirs, get_app_config, get_settings
from src.data.zones import load_zones_geodataframe, zones_to_polars
from src.etl.lazy_pipeline import build_hourly_panel

logging.basicConfig(
    level=get_settings().log_level,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("urban.features")


def main() -> int:
    ensure_dirs()
    cfg = get_app_config()
    panel = build_hourly_panel(cfg)
    logger.info("Panel built: %s rows x %s cols", panel.height, panel.width)

    try:
        gdf = load_zones_geodataframe(cfg)
        zones = zones_to_polars(gdf)
        out = Path(get_settings().artifacts_dir) / "zones.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        zones.write_parquet(out)
        logger.info("Zones saved: %s", out)
    except FileNotFoundError as exc:
        logger.warning("Zones not loaded: %s", exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
