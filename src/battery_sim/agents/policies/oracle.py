from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from battery_sim.agents.policies.base import Policy
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
        self._actions = self._solve(prices, battery, interval_minutes)
        self._step = 0

    def _solve(self, prices: np.ndarray, b: Battery,
               interval_minutes: int) -> np.ndarray:
        dt = interval_minutes / 60
        T = len(prices) - 1  # N prices → N-1 tradeable steps
        exec_prices = prices[1:]

        # Variables: x = [c_0..c_{T-1}, d_0..d_{T-1}]
        # Minimise: sum_t c[t]*eff*price[t+1]*dt - d[t]*price[t+1]*dt
        c_obj = np.concatenate([
            b.efficiency * exec_prices * dt,   # charge cost
            -exec_prices * dt,                  # discharge revenue (negative = good)
        ])

        # Bounds: 0 <= c <= max_charge, 0 <= d <= max_discharge
        bounds = ([(0, b.max_charge_rate_mw)] * T
                  + [(0, b.max_discharge_rate_mw)] * T)

        # SoC constraints: min_e <= e[0] + cumsum(c*eff*dt - d*dt) <= max_e
        min_e = b.min_soc * b.capacity_mwh
        max_e = b.max_soc * b.capacity_mwh
        e_0 = b.initial_soc * b.capacity_mwh

        # 2T inequality rows: lower + upper bound per timestep
        A_ub = np.zeros((2 * T, 2 * T))
        b_ub = np.zeros(2 * T)

        for t in range(1, T + 1):
            # Lower: -sum c*eff*dt + sum d*dt <= e_0 - min_e
            A_ub[2 * (t - 1), :t] = -b.efficiency * dt
            A_ub[2 * (t - 1), T:T + t] = dt
            b_ub[2 * (t - 1)] = e_0 - min_e

            # Upper: sum c*eff*dt - sum d*dt <= max_e - e_0
            A_ub[2 * (t - 1) + 1, :t] = b.efficiency * dt
            A_ub[2 * (t - 1) + 1, T:T + t] = -dt
            b_ub[2 * (t - 1) + 1] = max_e - e_0

        result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                         method="highs")
        if not result.success:
            raise RuntimeError(f"LP failed: {result.message}")

        return result.x[:T] - result.x[T:]  # net power MW

    def select_action(self, observation: Observation) -> float:
        idx = min(self._step, len(self._actions) - 1)
        self._step += 1
        return float(self._actions[idx])

    def reset(self) -> None:
        self._step = 0
