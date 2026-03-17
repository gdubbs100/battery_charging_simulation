from __future__ import annotations

from itertools import product

import pandas as pd

from battery_sim.agents.policies.base import Policy
from battery_sim.models.median_reversion import MedianReversionModel
from battery_sim.optimization.policy_optimizer import evaluate_policy
from battery_sim.utils.types import Observation

_DEFAULT_CONT_GRID: dict = {
    "buy_threshold": [30, 50, 75, 100],
    "sell_threshold": [100, 150, 200, 300],
}


class ContinuousProbabilisticPolicy(Policy):
    """Continuous-action variant of ProbabilisticThresholdPolicy.

    action = signal * rate, where
    signal = P(p_{t+1} <= buy_threshold) - P(p_{t+1} >= sell_threshold)
    """

    def __init__(
        self,
        model: MedianReversionModel,
        buy_threshold: float = 50.0,
        sell_threshold: float = 200.0,
        charge_rate: float = 1.0,
        discharge_rate: float = 1.0,
        forecast_horizon: int = 1,
    ):
        self.model = model
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.charge_rate = charge_rate
        self.discharge_rate = discharge_rate
        self.forecast_horizon = forecast_horizon

    def select_action(self, observation: Observation) -> float:
        result = self.model.step(observation.price, self.forecast_horizon)
        dist = result.distributions[0]

        p_buy = dist.cdf(self.buy_threshold)
        p_sell = 1 - dist.cdf(self.sell_threshold)
        signal = p_buy - p_sell

        if signal >= 0:
            return signal * self.charge_rate
        return signal * self.discharge_rate

    def learn(
        self,
        train_data: pd.DataFrame,
        battery,
        num_iters: int = 10,
        window_len: int = 24,
        param_grid: dict | None = None,
    ) -> None:
        if param_grid is None:
            param_grid = _DEFAULT_CONT_GRID

        self.model.fit(train_data)

        best_score, best_params = float("-inf"), None
        for buy_threshold, sell_threshold in product(
            param_grid["buy_threshold"], param_grid["sell_threshold"]
        ):
            if buy_threshold >= sell_threshold:
                continue
            self.buy_threshold = buy_threshold
            self.sell_threshold = sell_threshold
            score = evaluate_policy(
                self, train_data, battery,
                n_windows=num_iters, window_len=window_len,
                interval_minutes=self.model.interval_minutes,
            )
            self.model.fit(train_data)
            if score > best_score:
                best_score, best_params = score, (buy_threshold, sell_threshold)

        self.buy_threshold, self.sell_threshold = best_params
        self.model.fit(train_data)

    def reset(self) -> None:
        pass
