"""Abstract forecaster interface."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import polars as pl


class BaseForecaster(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, train: pl.DataFrame, feature_cols: list[str], target: str = "demand") -> "BaseForecaster":
        ...

    @abstractmethod
    def predict(self, df: pl.DataFrame, feature_cols: list[str]) -> np.ndarray:
        ...
