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

    def select_action(self, observation: Observation) -> float:
        self.model.step(observation.price)
        median = self.model.rolling_median
        action = (median - observation.price) / self.spread
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
        pass
