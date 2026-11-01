"""Train LightGBM demand model with rolling-origin CV and save artifacts."""
from __future__ import annotations

import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl

from src.config import ensure_dirs, get_app_config, get_settings
from src.evaluation.metrics import all_metrics
from src.features.build import build_features
from src.models.baseline import SeasonalMeanBaseline
from src.models.lightgbm_demand import LightGBMDemandForecaster

logging.basicConfig(
    level=get_settings().log_level,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("urban.train")


def rolling_origin_splits(
    df: pl.DataFrame, n_folds: int, horizon_hours: int
) -> list[tuple[pl.DataFrame, pl.DataFrame]]:
    hours = df.select("hour").unique().sort("hour")["hour"].to_list()
    n = len(hours)
    if n < n_folds * horizon_hours + 24:
        raise ValueError("Not enough hours for requested CV")
    folds: list[tuple[pl.DataFrame, pl.DataFrame]] = []
    end = n
    for k in range(n_folds):
        test_end = end - k * horizon_hours
        test_start = test_end - horizon_hours
        train_end = test_start
        train_hours = set(hours[:train_end])
        test_hours = set(hours[test_start:test_end])
        train = df.filter(pl.col("hour").is_in(list(train_hours)))
        test = df.filter(pl.col("hour").is_in(list(test_hours)))
        folds.append((train, test))
    return list(reversed(folds))


def evaluate_cv(
    bundle: pl.DataFrame, feature_cols: list[str], cfg, categorical_cols: list[str]
) -> dict[str, dict[str, float]]:
    results: dict[str, list[dict[str, float]]] = {"lightgbm": [], "baseline": []}
    folds = rolling_origin_splits(bundle, cfg.cv.n_folds, cfg.cv.horizon_hours)
    for k, (train, test) in enumerate(folds):
        if train.height < 24 or test.height == 0:
            continue
        y_true = test["demand"].to_numpy().astype(float)

        lgbm = LightGBMDemandForecaster(cfg.model.lightgbm)
        lgbm.fit(train, feature_cols, categorical_cols=categorical_cols)
        y_lgbm = lgbm.predict(test, feature_cols)
        results["lightgbm"].append(all_metrics(y_true, y_lgbm))

        base = SeasonalMeanBaseline()
        base.fit(train, feature_cols)
        y_base = base.predict(test, feature_cols)
        results["baseline"].append(all_metrics(y_true, y_base))
        logger.info(
            "Fold %s: LGBM WAPE=%.3f | Baseline WAPE=%.3f",
            k,
            results["lightgbm"][-1]["wape"],
            results["baseline"][-1]["wape"],
        )

    aggregated: dict[str, dict[str, float]] = {}
    for name, folds_metrics in results.items():
        if not folds_metrics:
            aggregated[name] = {}
            continue
        keys = folds_metrics[0].keys()
        aggregated[name] = {
            k: float(np.nanmean([m[k] for m in folds_metrics])) for k in keys
        }
    return aggregated


def main() -> int:
    ensure_dirs()
    cfg = get_app_config()

    panel_path = Path(cfg.data.processed_dir) / "hourly_zone_demand.parquet"
    if not panel_path.exists():
        raise FileNotFoundError(f"Panel not found at {panel_path}. Run build_features.py first.")
    panel = pl.read_parquet(panel_path)
    logger.info("Panel loaded: %s rows x %s cols", panel.height, panel.width)

    bundle = build_features(panel, cfg)
    categorical_cols = ["zone_code"]

    logger.info("Running CV...")
    cv_metrics = evaluate_cv(bundle.frame, bundle.feature_cols, cfg, categorical_cols)
    logger.info("CV metrics: %s", cv_metrics)

    logger.info("Training final model on all data...")
    final = LightGBMDemandForecaster(cfg.model.lightgbm)
    final.fit(bundle.frame, bundle.feature_cols, categorical_cols=categorical_cols)

    fi = final.feature_importance(bundle.feature_cols)
    logger.info("Top-10 features:\n%s", fi.head(10))

    out_dir = Path(get_settings().artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "model.pkl").open("wb") as f:
        pickle.dump({"model": final, "feature_cols": bundle.feature_cols}, f)
    with (out_dir / "metadata.pkl").open("wb") as f:
        pickle.dump(
            {
                "last_hour": str(bundle.frame["hour"].max()),
                "cv_metrics": cv_metrics,
                "trained_at": datetime.utcnow().isoformat(),
            },
            f,
        )
    bundle.frame.write_parquet(out_dir / "panel.parquet")
    logger.info("Artifacts saved to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
