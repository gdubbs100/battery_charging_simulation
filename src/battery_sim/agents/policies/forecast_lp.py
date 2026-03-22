from __future__ import annotations

import numpy as np
import pandas as pd

from battery_sim.agents.policies.base import Policy
from battery_sim.agents.policies._utils import solve_lp
from battery_sim.battery.battery import Battery
from battery_sim.models.median_reversion import MedianReversionModel
from battery_sim.utils.types import Observation


class ForecastLPPolicy(Policy):
    """MPC-style policy using median-reversion forecasts + LP optimization.

    At each step, forecasts the next `horizon` hours using median estimates
    from the price model, then solves an LP to get optimal charge/discharge.
    Takes only the first action (receding horizon).
    """

    def __init__(
        self,
        model: MedianReversionModel,
        battery: Battery,
        horizon: int = 12,
        interval_minutes: int = 60,
    ):
        self.model = model
        self.battery = battery
        self.horizon = horizon
        self.interval_minutes = interval_minutes

    def select_action(self, observation: Observation) -> float:
        """Return normalized action in [-1, 1].

        -1 = max discharge, 0 = hold, 1 = max charge
        """
        # Update model buffer with current price and forecast future prices
        self.model.step(observation.price)
        pred = self.model.forecast(self.horizon)

        # Build price array: [current] + horizon forecasts
        prices = np.concatenate([[observation.price], pred.prices])

        # Solve LP starting from current battery state
        actions = solve_lp(
            prices,
            self.battery,
            self.interval_minutes,
            e_0=observation.energy_mwh,
        )

        # Normalize first action to [-1, 1]
        action_mw = np.clip(
            actions[0],
            -self.battery.max_discharge_rate_mw,
            self.battery.max_charge_rate_mw,
        )
        action_normalized = action_mw / self.battery.max_charge_rate_mw
        return float(action_normalized)

    def learn(self, train_data: pd.DataFrame, battery: Battery, **_) -> None:
        self.battery = battery
        self.model.fit(train_data)

    def reset(self) -> None:
        pass
