from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import rv_continuous


@dataclass
class Observation:
    timestamp: datetime
    price: float  # $/MWh
    soc: float  # 0.0–1.0
    energy_mwh: float
    forecast: np.ndarray | None = None


@dataclass
class BatteryState:
    soc: float
    energy_mwh: float
    capacity_mwh: float


@dataclass
class StepRecord:
    timestamp: datetime
    price: float
    action: float  # MW
    reward: float  # $
    soc: float
    energy_mwh: float


@dataclass
class SimResult:
    trajectory: list[StepRecord] = field(default_factory=list)
    cumulative_revenue: float = 0.0
    total_steps: int = 0


@dataclass
class PredictionResult:
    distributions: list[rv_continuous]
    prices: np.ndarray
    medians: np.ndarray

    def quantiles(self, q: list[float] | None = None) -> np.ndarray:
        """Shape: (n_steps, len(q))."""
        if q is None:
            q = [0.025, 0.25, 0.5, 0.75, 0.975]
        return np.array([d.ppf(q) for d in self.distributions])

    def sample(self, n: int) -> np.ndarray:
        """IID samples from each marginal. Shape: (n_steps, n)."""
        return np.array([d.rvs(size=n) for d in self.distributions])

    def to_dataframe(self, q: list[float] | None = None) -> pd.DataFrame:
        if q is None:
            q = [0.025, 0.25, 0.5, 0.75, 0.975]
        quantiles = self.quantiles(q)
        df = pd.DataFrame({"price": self.prices, "median": self.medians})
        for i, qi in enumerate(q):
            df[f"q{qi:.3f}"] = quantiles[:, i]
        return df
