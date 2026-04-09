from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from battery_sim.agents.policies.base import Policy
from battery_sim.models.median_reversion import MedianReversionModel
from battery_sim.optimization.policy_optimizer import evaluate_policy
from battery_sim.utils.types import Observation

_DEFAULT_GRID: dict = {
    "spread": [10, 25, 50, 75, 100, 150],
}

_DEFAULT_PROB_MED_GRID: dict = {
    "spread": [10, 25, 50, 75, 100, 150],
    "horizon": [3, 6, 12],
}


class MedianThresholdPolicy(Policy):
    """Charge/discharge based on deviation from 30-day rolling median.

    action = clip((median - price) / spread, -1, 1)

    When price < median: positive action (charge).
    When price > median: negative action (discharge).
    Spread controls sensitivity — smaller spread means more aggressive trading.
    """

    def __init__(
        self,
        model: MedianReversionModel,
        spread: float = 50.0,
    ):
        self.model = model
        self.spread = spread
        self._prev_price: float | None = None

    def select_action(self, observation: Observation) -> float:
        # Use p_{t-1} and median_{t-1} to choose the action
        median = self.model.rolling_median
        prev = self._prev_price if self._prev_price is not None else observation.price
        self._prev_price = observation.price
        self.model.step(observation.price)
        action = (median - prev) / self.spread
        return float(np.clip(action, -1.0, 1.0))

    def learn(
        self,
        train_data: pd.DataFrame,
        battery,
        num_iters: int = 10,
        window_len: int = 24,
        param_grid: dict | None = None,
        **_,
    ) -> None:
        if param_grid is None:
            param_grid = _DEFAULT_GRID

        self.model.fit(train_data)

        best_score, best_spread = float("-inf"), self.spread
        for (spread,) in product(param_grid["spread"]):
            self.spread = spread
            score = evaluate_policy(
                self, train_data, battery,
                n_windows=num_iters, window_len=window_len,
                interval_minutes=self.model.interval_minutes,
            )
            self.model.fit(train_data)
            if score > best_score:
                best_score, best_spread = score, spread

        self.spread = best_spread
        self.model.fit(train_data)

    def reset(self) -> None:
        self._prev_price = None


class ProbabilisticMedianThresholdPolicy(Policy):
    """Probabilistic upgrade of MedianThresholdPolicy.

    Deterministic analog: action = clip((median - price) / spread, -1, 1)
    This policy:         action = clip(median(median - simulated_prices) / spread, -1, 1)

    Uses multi-step Monte Carlo simulation to estimate the robust central
    tendency of future deviations from the rolling median. The median
    estimator handles the Cauchy model's fat tails correctly (Cauchy has
    no finite mean, but the median is well-defined).

    Over multiple simulated steps, compounding reversion dynamics create
    path-dependent outcomes that a single-step forecast cannot capture.
    """

    def __init__(
        self,
        model: MedianReversionModel,
        horizon: int = 6,
        n_trajectories: int = 50,
        spread: float = 50.0,
    ):
        self.model = model
        self.horizon = horizon
        self.n_trajectories = n_trajectories
        self.spread = spread

    def select_action(self, observation: Observation) -> float:
        # Get median_{t-1} before updating buffer
        median = self.model.rolling_median
        # model.current_price is p_{t-1} (buffer not yet updated with p_t)
        trajs = self.model.simulate(self.horizon, self.n_trajectories)  # (N, H)
        # Robust estimate of future deviation from median
        signal = np.median(median - trajs)
        # Now update buffer with current price for next call
        self.model.step(observation.price)
        return float(np.clip(signal / self.spread, -1, 1))

    def learn(
        self,
        train_data: pd.DataFrame,
        battery,
        num_iters: int = 10,
        window_len: int = 24,
        param_grid: dict | None = None,
        **_,
    ) -> None:
        if param_grid is None:
            param_grid = _DEFAULT_PROB_MED_GRID

        self.model.fit(train_data)

        best_score, best_params = float("-inf"), (self.spread, self.horizon)
        for spread, horizon in product(param_grid["spread"], param_grid["horizon"]):
            self.spread = spread
            self.horizon = horizon
            score = evaluate_policy(
                self, train_data, battery,
                n_windows=num_iters, window_len=window_len,
                interval_minutes=self.model.interval_minutes,
            )
            self.model.fit(train_data)
            if score > best_score:
                best_score, best_params = score, (spread, horizon)

        self.spread, self.horizon = best_params
        self.model.fit(train_data)

    def reset(self) -> None:
        pass
