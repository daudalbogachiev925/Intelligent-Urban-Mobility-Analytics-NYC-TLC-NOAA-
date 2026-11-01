"""Choropleth maps of demand per NYC taxi zone."""
from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import polars as pl

from src.config import AppConfig

logger = logging.getLogger(__name__)


def plot_zone_demand_choropleth(
    zone_demand: pl.DataFrame,
    cfg: AppConfig,
    value_col: str = "total_demand",
    title: str = "NYC Taxi Demand by Zone",
    cmap: str = "YlOrRd",
    out_path: str | Path | None = None,
) -> plt.Figure:
    """Render choropleth of zone-level demand on NYC taxi zone geometry.

    zone_demand: DataFrame with columns [zone_id, value_col]
    """
    zones_path = Path(cfg.data.raw_dir) / "zones" / "taxi_zones.geojson"
    if not zones_path.exists():
        raise FileNotFoundError(f"Zones not found at {zones_path}")
    gdf = gpd.read_file(zones_path).rename(
        columns={"LocationID": "zone_id", "zone": "zone_name", "borough": "borough"}
    )
    gdf["zone_id"] = gdf["zone_id"].astype(int)

    joined = gdf.merge(
        zone_demand.to_pandas(), on="zone_id", how="left"
    )

    fig, ax = plt.subplots(figsize=(11, 11))
    joined.plot(
        column=value_col,
        cmap=cmap,
        legend=True,
        legend_kwds={"label": value_col, "shrink": 0.7},
        ax=ax,
        edgecolor="white",
        linewidth=0.3,
        missing_kwds={"color": "lightgray", "label": "No data"},
    )
    ax.set_title(title, fontsize=14)
    ax.set_axis_off()
    plt.tight_layout()

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        logger.info("Saved choropleth to %s", out_path)
    return fig


def plot_weather_elasticity(
    elasticity_df: pl.DataFrame,
    out_path: str | Path | None = None,
) -> plt.Figure:
    """Bar chart of weather sensitivity per zone (or per borough)."""
    pdf = elasticity_df.to_pandas()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(pdf.iloc[:, 0].astype(str), pdf["correlation"], color="steelblue")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel(elasticity_df.columns[0])
    ax.set_ylabel("Correlation with demand")
    ax.set_title("Weather sensitivity of hourly demand")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        logger.info("Saved weather elasticity chart to %s", out_path)
    return fig
