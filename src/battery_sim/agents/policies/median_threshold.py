from __future__ import annotations

from itertools import product

import numpy as np  # used by ProbabilisticMedianThresholdPolicy
import pandas as pd

from battery_sim.agents.policies.base import Policy
from battery_sim.models.median_reversion import MedianReversionModel
from battery_sim.optimization.policy_optimizer import evaluate_policy
from battery_sim.utils.types import Observation

_DEFAULT_GRID: dict = {
    "buy_dev": [10, 25, 50, 75, 100],
    "sell_dev": [10, 25, 50, 75, 100],
}

_DEFAULT_PROB_MED_GRID: dict = {
    "buy_dev": [10, 25, 50, 75],
    "sell_dev": [10, 25, 50, 75],
}


class MedianThresholdPolicy(Policy):
    """Charge/discharge based on deviation from 30-day rolling median.

    deviation = median_{t-1} - p_{t-1}

    action =  1.0  if deviation >  buy_dev   (price cheap enough → charge)
    action = -1.0  if deviation < -sell_dev  (price expensive enough → discharge)
    action =  0.0  otherwise                 (hold)

    Deterministic analog of ProbabilisticMedianThresholdPolicy — same
    deviation thresholds applied to the current price rather than to
    probabilities over simulated trajectories.
    """

    def __init__(
        self,
        model: MedianReversionModel,
        buy_dev: float = 25.0,
        sell_dev: float = 25.0,
    ):
        self.model = model
        self.buy_dev = buy_dev
        self.sell_dev = sell_dev
        self._prev_price: float | None = None

    def select_action(self, observation: Observation) -> float:
        median = self.model.rolling_median
        prev = self._prev_price if self._prev_price is not None else observation.price
        self._prev_price = observation.price
        self.model.step(observation.price)
        deviation = median - prev
        if deviation > self.buy_dev:
            return 1.0
        elif deviation < -self.sell_dev:
            return -1.0
        return 0.0

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

        best_score = float("-inf")
        best_params = (self.buy_dev, self.sell_dev)
        for buy_dev, sell_dev in product(param_grid["buy_dev"], param_grid["sell_dev"]):
            self.buy_dev = buy_dev
            self.sell_dev = sell_dev
            score = evaluate_policy(
                self, train_data, battery,
                n_windows=num_iters, window_len=window_len,
                interval_minutes=self.model.interval_minutes,
            )
            self.model.fit(train_data)
            if score > best_score:
                best_score, best_params = score, (buy_dev, sell_dev)

        self.buy_dev, self.sell_dev = best_params
        self.model.fit(train_data)

    def reset(self) -> None:
        self._prev_price = None
        self.model.reset()


class ProbabilisticMedianThresholdPolicy(Policy):
    """Uses analytical CDF of the forecast distribution to compute action.

    For each step in the forecast horizon, evaluates:
        p_buy  = P(p_future < median - buy_dev)   averaged over horizon
        p_sell = P(p_future > median + sell_dev)  averaged over horizon
        action = p_buy - p_sell
    """

    def __init__(
        self,
        model: MedianReversionModel,
        buy_dev: float = 25.0,
        sell_dev: float = 25.0,
        horizon: int = 6,
    ):
        self.model = model
        self.buy_dev = buy_dev
        self.sell_dev = sell_dev
        self.horizon = horizon

    def select_action(self, observation: Observation) -> float:
        median = self.model.rolling_median
        result = self.model.forecast(self.horizon)
        p_buy = float(np.mean([d.cdf(median - self.buy_dev) for d in result.distributions]))
        p_sell = float(np.mean([1 - d.cdf(median + self.sell_dev) for d in result.distributions]))
        self.model.step(observation.price)
        return p_buy - p_sell

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

        best_score = float("-inf")
        best_params = (self.buy_dev, self.sell_dev)
        for buy_dev, sell_dev in product(param_grid["buy_dev"], param_grid["sell_dev"]):
            self.buy_dev = buy_dev
            self.sell_dev = sell_dev
            score = evaluate_policy(
                self, train_data, battery,
                n_windows=num_iters, window_len=window_len,
                interval_minutes=self.model.interval_minutes,
            )
            self.model.fit(train_data)
            if score > best_score:
                best_score, best_params = score, (buy_dev, sell_dev)

        self.buy_dev, self.sell_dev = best_params
        self.model.fit(train_data)

    def reset(self) -> None:
        self.model.reset()
