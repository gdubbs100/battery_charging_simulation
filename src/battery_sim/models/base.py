from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque

import numpy as np
import pandas as pd
from scipy.stats import rv_continuous


class PriceModel(ABC):
    """Base class for all price models."""

    def __init__(self, buffer_size: int = 1000):
        self._buffer: deque[float] = deque(maxlen=buffer_size)

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform raw data into model-ready features. Override as needed."""
        return data

    def fit(self, data: pd.DataFrame) -> None:
        """Train/calibrate the model. Default no-op for pre-trained models."""
        pass

    def update(self, new_observation: float) -> None:
        """Feed a single new price observation."""
        self._buffer.append(new_observation)

    def reset(self) -> None:
        self._buffer.clear()


class DeterministicModel(PriceModel):
    """Models that produce point estimates."""

    @abstractmethod
    def predict(self, horizon: int) -> np.ndarray:
        """Return point forecast. Shape: (horizon,)."""
        ...

    def forecast(self, data: pd.DataFrame, horizon: int) -> np.ndarray:
        """Preprocess data, update buffer, then predict."""
        processed = self.preprocess(data)
        for price in processed.iloc[:, -1].values:
            self.update(float(price))
        return self.predict(horizon)


class StochasticModel(PriceModel):
    """Models that sample from a distribution."""

    @abstractmethod
    def get_distribution(self, horizon: int) -> list[rv_continuous]:
        """Return a scipy distribution object per forecast step."""
        ...

    def sample(self, horizon: int, n: int = 1) -> np.ndarray:
        """Sample n trajectories. Shape: (n, horizon)."""
        dists = self.get_distribution(horizon)
        return np.column_stack([d.rvs(size=n) for d in dists])

    def predict(self, horizon: int) -> np.ndarray:
        """Return mean of distribution as point estimate. Shape: (horizon,)."""
        dists = self.get_distribution(horizon)
        return np.array([d.mean() for d in dists])

    def forecast(self, data: pd.DataFrame, horizon: int) -> np.ndarray:
        processed = self.preprocess(data)
        for price in processed.iloc[:, -1].values:
            self.update(float(price))
        return self.predict(horizon)
