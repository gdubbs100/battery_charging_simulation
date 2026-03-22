from __future__ import annotations

import numpy as np
import pandas as pd

from battery_sim.agents.policies.base import Policy
from battery_sim.agents.policies._utils import solve_lp
from battery_sim.battery.battery import Battery
from battery_sim.utils.types import Observation


class OraclePolicy(Policy):
    """Perfect-foresight baseline that solves an LP over known prices.

    Maximises revenue = sum_t (d[t] - c[t]*eff) * price[t+1] * dt
    subject to battery rate and SoC constraints.
    """

    def __init__(self, price_data: pd.DataFrame, battery: Battery,
                 interval_minutes: int = 60):
        self.battery = battery
        prices = price_data["price"].values
        self._actions = solve_lp(prices, battery, interval_minutes)
        self._step = 0

    def select_action(self, observation: Observation) -> float:
        """Return normalized action in [-1, 1].

        -1 = max discharge, 0 = hold, 1 = max charge
        """
        idx = min(self._step, len(self._actions) - 1)
        self._step += 1
        action_mw = self._actions[idx]
        # Normalize to [-1, 1]
        action_normalized = action_mw / self.battery.max_charge_rate_mw
        return float(action_normalized)

    def reset(self) -> None:
        self._step = 0
