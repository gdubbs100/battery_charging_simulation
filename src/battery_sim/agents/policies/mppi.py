from __future__ import annotations

import numpy as np
import pandas as pd
from itertools import product
from scipy.special import softmax

from battery_sim.agents.policies.base import Policy
from battery_sim.battery.battery import Battery
from battery_sim.models.median_reversion import MedianReversionModel
from battery_sim.optimization.policy_optimizer import evaluate_policy
from battery_sim.utils.types import Observation

_DEFAULT_MPPI_GRID: dict = {
    "noise_sigma": [0.2, 0.5, 1.0],
    "temperature": [0.5, 1.0, 5.0],
}


class MPPIPolicy(Policy):
    """MPPI (Model Predictive Path Integral) controller for battery dispatch.

    Uses median-then-softmax aggregation: for each candidate action sequence,
    the median profit across Monte Carlo price trajectories is used as its score,
    then action sequences are blended via softmax weighting.
    """

    def __init__(
        self,
        model: MedianReversionModel,
        battery: Battery,
        horizon: int = 12,
        n_samples: int = 100,
        n_trajectories: int = 50,
        temperature: float = 1.0,
        noise_sigma: float = 0.5,
    ):
        self.model = model
        self.battery = battery
        self.horizon = horizon
        self.n_samples = n_samples
        self.n_trajectories = n_trajectories
        self.temperature = temperature
        self.noise_sigma = noise_sigma
        self._warm_start = np.zeros(horizon)

    def _rollout(self, obs: Observation, actions: np.ndarray, prices: np.ndarray) -> float:
        """Total profit for one action sequence against one price trajectory."""
        b = self.battery
        energy = obs.energy_mwh
        total = 0.0
        for t in range(len(actions)):
            a = actions[t]
            delta = a * b.efficiency if a >= 0 else a
            delta = np.clip(delta,
                            b.min_soc * b.capacity_mwh - energy,
                            b.max_soc * b.capacity_mwh - energy)
            total += prices[t] * (-delta)
            energy += delta
        return total

    def select_action(self, observation: Observation) -> float:
        b = self.battery
        self.model.step(observation.price)
        price_trajs = self.model.simulate(self.horizon, self.n_trajectories)

        noise = np.random.randn(self.n_samples, self.horizon) * self.noise_sigma
        action_seqs = self._warm_start + noise
        action_seqs = np.clip(action_seqs, -b.max_discharge_rate_mw, b.max_charge_rate_mw)

        scores = np.zeros(self.n_samples)
        for k in range(self.n_samples):
            profits = np.array([
                self._rollout(observation, action_seqs[k], price_trajs[j])
                for j in range(self.n_trajectories)
            ])
            scores[k] = np.median(profits)

        weights = softmax(scores / self.temperature)
        optimal = weights @ action_seqs

        self._warm_start[:-1] = optimal[1:]
        self._warm_start[-1] = 0.0

        return float(np.clip(optimal[0], -b.max_discharge_rate_mw, b.max_charge_rate_mw))

    def learn(self, train_data: pd.DataFrame, battery, num_iters=10,
              window_len=24, param_grid: dict | None = None, **_):
        """Fit model, update battery ref, tune noise_sigma and temperature."""
        self.battery = battery
        self.model.fit(train_data)

        if param_grid is None:
            param_grid = _DEFAULT_MPPI_GRID

        best_score, best_params = float("-inf"), None
        for sigma, temp in product(param_grid["noise_sigma"], param_grid["temperature"]):
            self.noise_sigma = sigma
            self.temperature = temp
            score = evaluate_policy(
                self, train_data, battery,
                n_windows=num_iters, window_len=window_len,
                interval_minutes=self.model.interval_minutes,
            )
            self.model.fit(train_data)
            if score > best_score:
                best_score, best_params = score, (sigma, temp)

        self.noise_sigma, self.temperature = best_params
        self.model.fit(train_data)

    def reset(self) -> None:
        self._warm_start = np.zeros(self.horizon)
