"""Temporal plots: hourly pattern, weekly heatmap, anomaly overlay."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


def plot_hourly_profile(
    df: pl.DataFrame, zone_id: int, out_path: str | Path | None = None
) -> plt.Figure:
    sub = df.filter(pl.col("zone_id") == zone_id)
    if sub.height == 0:
        raise ValueError(f"No data for zone {zone_id}")
    profile = (
        sub.group_by("hour_of_day")
        .agg(pl.col("demand").mean().alias("mean_demand"))
        .sort("hour_of_day")
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(profile["hour_of_day"].to_list(), profile["mean_demand"].to_list(), marker="o")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean demand")
    ax.set_title(f"Hourly demand profile, zone {zone_id}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        logger.info("Saved hourly profile to %s", out_path)
    return fig


def plot_weekly_heatmap(
    df: pl.DataFrame, out_path: str | Path | None = None
) -> plt.Figure:
    """Heatmap of mean demand by (hour_of_day, day_of_week) across all zones."""
    agg = (
        df.group_by(["day_of_week", "hour_of_day"])
        .agg(pl.col("demand").mean().alias("mean_demand"))
        .sort(["day_of_week", "hour_of_day"])
    )
    pdf = agg.to_pandas().pivot(index="day_of_week", columns="hour_of_day", values="mean_demand")
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(pdf.to_numpy(), aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(pdf.shape[1]))
    ax.set_xticklabels([str(c) for c in pdf.columns], rotation=90, fontsize=8)
    ax.set_yticks(range(pdf.shape[0]))
    ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][: pdf.shape[0]])
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Day of week")
    ax.set_title("Mean hourly demand heatmap (all zones)")
    fig.colorbar(im, ax=ax, label="Mean demand")
    plt.tight_layout()
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        logger.info("Saved weekly heatmap to %s", out_path)
    return fig
