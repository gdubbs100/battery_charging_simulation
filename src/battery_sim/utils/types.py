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

    def to_dataframe(self) -> pd.DataFrame:
        rows = [
            {
                "timestamp": r.timestamp,
                "price": r.price,
                "action": r.action,
                "reward": r.reward,
                "soc": r.soc,
                "energy_mwh": r.energy_mwh,
            }
            for r in self.trajectory
        ]
        df = pd.DataFrame(rows)
        df["cumulative_revenue"] = df["reward"].cumsum()
        return df


@dataclass
class EvalResult:
    """Rich evaluation output for a single policy run."""

    name: str
    sim_result: SimResult
    df: pd.DataFrame  # time series (from SimResult.to_dataframe)
    cumulative_revenue: float
    daily_revenue: pd.Series  # revenue per day
    mean_daily_revenue: float
    std_daily_revenue: float

    @staticmethod
    def from_sim(name: str, sim_result: SimResult) -> "EvalResult":
        df = sim_result.to_dataframe()
        daily = df.set_index("timestamp")["reward"].resample("D").sum()
        return EvalResult(
            name=name,
            sim_result=sim_result,
            df=df,
            cumulative_revenue=sim_result.cumulative_revenue,
            daily_revenue=daily,
            mean_daily_revenue=float(daily.mean()),
            std_daily_revenue=float(daily.std()),
        )

    def summary(self) -> str:
        return (
            f"{self.name}:\n"
            f"  Cumulative revenue: ${self.cumulative_revenue:,.2f}\n"
            f"  Mean daily revenue: ${self.mean_daily_revenue:,.2f} "
            f"(+/- ${self.std_daily_revenue:,.2f})\n"
            f"  Total steps: {len(self.df)}"
        )


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
