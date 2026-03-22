from __future__ import annotations

import pandas as pd
from itertools import product

from battery_sim.optimization.policy_optimizer import evaluate_policy
from battery_sim.agents.policies.base import Policy
from battery_sim.utils.types import Observation


_DEFAULT_THRESHOLD_GRID: dict = {
    "buy_threshold":  [30, 50, 75, 100, 150],
    "sell_threshold": [100, 150, 200, 300, 500],
}


class ThresholdPolicy(Policy):
    def __init__(
        self,
        buy_threshold: float,
        sell_threshold: float,
    ):
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def select_action(self, observation: Observation) -> float:
        """Return normalized action in [-1, 1].

        -1 = max discharge, 0 = hold, 1 = max charge
        """
        if observation.price < self.buy_threshold:
            return 1.0
        elif observation.price > self.sell_threshold:
            return -1.0
        return 0.0

    def learn(
        self,
        train_data: pd.DataFrame,
        battery,
        num_iters: int = 10,
        window_len: int = 24,
        param_grid: dict | None = None,
    ) -> None:
        if param_grid is None:
            param_grid = _DEFAULT_THRESHOLD_GRID

        best_score, best_params = float("-inf"), None
        for buy_threshold, sell_threshold in product(
            param_grid["buy_threshold"], param_grid["sell_threshold"]
        ):
            if buy_threshold >= sell_threshold:
                continue
            self.buy_threshold = buy_threshold
            self.sell_threshold = sell_threshold
            score = evaluate_policy(self, train_data, battery, n_windows=num_iters, window_len=window_len)
            if score > best_score:
                best_score, best_params = score, (buy_threshold, sell_threshold)

        self.buy_threshold, self.sell_threshold = best_params


class TimeOfUsePolicy(Policy):
    def __init__(
        self,
        charge_hours: set[int],
        discharge_hours: set[int],
    ):
        self.charge_hours = charge_hours
        self.discharge_hours = discharge_hours

    def select_action(self, observation: Observation) -> float:
        """Return normalized action in [-1, 1].

        -1 = max discharge, 0 = hold, 1 = max charge
        """
        hour = observation.timestamp.hour
        if hour in self.charge_hours:
            return 1.0
        elif hour in self.discharge_hours:
            return -1.0
        return 0.0
