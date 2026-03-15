from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import cauchy, iqr, rv_continuous

from battery_sim.models.base import StochasticModel
from battery_sim.utils.types import PredictionResult


class MedianReversionModel(StochasticModel):
    """Stochastic price model with reversion to a trailing rolling median.

    Model:
        Y_t = (p_t - p_{t-1}) / (mu_{t-1} - p_{t-1})
        kappa = 1 - median(Y)
        errors = Y / median(Y) ~ Cauchy(loc, scale)

    Public API:
        backtest(data)         — in-sample distributions at each row
        forecast(horizon)      — h-step ahead from current state (read-only)
        simulate(horizon, n)   — recursive Monte Carlo trajectories (read-only)
        step(price, horizon)   — online update + forecast
    """

    def __init__(self, window_days: int = 30, interval_minutes: int = 60):
        window_size = window_days * 24 * 60 // interval_minutes
        super().__init__(buffer_size=window_size)
        self.window_size = window_size
        self.window_days = window_days
        self.interval_minutes = interval_minutes
        self.kappa: float | None = None
        self.loc: float | None = None
        self.scale: float | None = None

    def _check_fitted(self):
        if self.kappa is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

    def fit(self, data: pd.DataFrame) -> None:
        """Estimate kappa, loc, scale from training data and seed the buffer."""
        prices = self.preprocess(data).iloc[:, -1]

        rolling_med = prices.rolling(window=self.window_size).median().shift(1)
        prev_prices = prices.shift(1)
        denominator = rolling_med - prev_prices
        numerator = prices - prev_prices
        Y = (numerator / (denominator + 1.0e-6)).dropna()

        median_y = np.median(Y)
        errors = Y / (median_y + 1.0e-6)
        self.kappa = 1 - median_y
        self.loc = 1.0
        self.scale = iqr(errors) / 2

        for p in prices.values[-self.window_size:]:
            self._buffer.append(float(p))

    # ── Properties ──────────────────────────────────────────────────

    @property
    def rolling_median(self) -> float:
        if not self._buffer:
            raise ValueError("No observations. Call fit() or update() first.")
        return float(np.median(self._buffer))

    @property
    def current_price(self) -> float:
        if not self._buffer:
            raise ValueError("No observations. Call fit() or update() first.")
        return self._buffer[-1]

    # ── Private primitives ──────────────────────────────────────────

    def _predict_loc(self, price: float, median: float, horizon: int) -> float:
        reversion = 1 - self.kappa ** horizon
        return price + reversion * (median - price) * self.loc

    def _predict_scale(self, price: float, median: float, horizon: int) -> float:
        reversion = 1 - self.kappa ** horizon
        return max(abs(reversion * (median - price)) * self.scale, 1e-8)

    def _distribution_at(self, price: float, median: float, horizon: int) -> list[rv_continuous]:
        """Cauchy distributions for h=1..horizon given price and median."""
        return [
            cauchy(
                loc=self._predict_loc(price, median, h),
                scale=self._predict_scale(price, median, h),
            )
            for h in range(1, horizon + 1)
        ]

    # ── Public API ──────────────────────────────────────────────────

    def backtest(self, data: pd.DataFrame) -> PredictionResult:
        """In-sample: distributions at each row using per-row rolling median."""
        self._check_fitted()
        prices = self.preprocess(data).iloc[:, -1]
        rolling_med = prices.rolling(window=self.window_size).median()
        mask = rolling_med.notna()
        p_arr = prices[mask].values
        m_arr = rolling_med[mask].values
        dists = [self._distribution_at(float(p), float(m), 1)[0] for p, m in zip(p_arr, m_arr)]
        return PredictionResult(
            distributions=dists,
            prices=p_arr.astype(float),
            medians=m_arr.astype(float),
        )

    def forecast(self, horizon: int) -> PredictionResult:
        """Forecast h steps ahead from current buffer state. Read-only."""
        self._check_fitted()
        price = self.current_price
        median = self.rolling_median
        dists = self._distribution_at(price, median, horizon)
        prices = np.array([d.median() for d in dists])
        medians = np.full(horizon, median)
        return PredictionResult(distributions=dists, prices=prices, medians=medians)

    def simulate(self, horizon: int, n: int = 1) -> np.ndarray:
        """Recursive Monte Carlo trajectories. Shape: (n, horizon). Read-only."""
        self._check_fitted()
        trajectories = np.zeros((n, horizon))
        p = np.full(n, self.current_price)
        mu = self.rolling_median
        reversion = 1 - self.kappa

        for t in range(horizon):
            loc = p + reversion * (mu - p) * self.loc
            scale = np.abs(reversion * (mu - p)) * self.scale
            trajectories[:, t] = cauchy.rvs(loc=loc, scale=scale)
            p = trajectories[:, t]

        return trajectories

    def step(self, price: float, horizon: int = 1) -> PredictionResult:
        """Update buffer with new price, then forecast."""
        self._check_fitted()
        self.update(price)
        return self.forecast(horizon)

    # ── Base class compatibility ────────────────────────────────────

    def get_distribution(self, horizon: int) -> list[rv_continuous]:
        return self.forecast(horizon).distributions

    def sample(self, horizon: int, n: int = 1) -> np.ndarray:
        return self.simulate(horizon, n)
