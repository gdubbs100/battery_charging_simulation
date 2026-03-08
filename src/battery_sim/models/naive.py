from __future__ import annotations

import numpy as np

from battery_sim.models.base import DeterministicModel


class PersistenceModel(DeterministicModel):
    """Predicts the last observed price for all future steps."""

    def predict(self, horizon: int) -> np.ndarray:
        if not self._buffer:
            raise ValueError("No observations yet. Call update() first.")
        return np.full(horizon, self._buffer[-1])


class MovingAverageModel(DeterministicModel):
    """Predicts the rolling mean over a window."""

    def __init__(self, window: int = 12, buffer_size: int = 1000):
        super().__init__(buffer_size=buffer_size)
        self.window = window

    def predict(self, horizon: int) -> np.ndarray:
        if not self._buffer:
            raise ValueError("No observations yet. Call update() first.")
        recent = list(self._buffer)[-self.window :]
        mean = np.mean(recent)
        return np.full(horizon, mean)
