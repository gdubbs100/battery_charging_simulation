from __future__ import annotations

from itertools import product

import pandas as pd

from battery_sim.agents.policies.base import Policy
from battery_sim.models.median_reversion import MedianReversionModel
from battery_sim.optimization.policy_optimizer import evaluate_policy
from battery_sim.utils.types import Observation


_DEFAULT_PROB_GRID: dict = {
    "buy_threshold": [30, 50, 75, 100],
    "sell_threshold": [100, 150, 200, 300],
}


class ProbabilisticThresholdPolicy(Policy):
    """Charges when P(price < buy_threshold) is highest,
    discharges when P(price > sell_threshold) is highest.

    Uses a MedianReversionModel to produce price distributions,
    then acts on the CDF probabilities.
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

        p_below_buy = dist.cdf(self.buy_threshold)
        p_above_sell = 1 - dist.cdf(self.sell_threshold)

        if p_below_buy > p_above_sell:
            return self.charge_rate
        elif p_above_sell > p_below_buy:
            return -self.discharge_rate
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
            param_grid = _DEFAULT_PROB_GRID

        # 1. Fit the price model on training data
        self.model.fit(train_data)

        # 2. Grid search over buy/sell thresholds
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
            # Re-fit buffer after each evaluation (evaluate mutates via step())
            self.model.fit(train_data)
            if score > best_score:
                best_score, best_params = score, (buy_threshold, sell_threshold)

        self.buy_threshold, self.sell_threshold = best_params
        # Final fit so buffer is clean for downstream use
        self.model.fit(train_data)

    def reset(self) -> None:
        pass
