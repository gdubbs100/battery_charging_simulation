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
    "noise_sigma": [0.05, 0.1, 0.2],  # Relative to max_charge_rate_mw
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
        noise_sigma: float = 0.1,
    ):
        self.model = model
        self.battery = battery
        self.horizon = horizon
        self.n_samples = n_samples
        self.n_trajectories = n_trajectories
        self.temperature = temperature
        self.noise_sigma = noise_sigma
        self._warm_start = np.zeros(horizon)

    def _rollout_batch(self, obs: Observation, action_seqs: np.ndarray,
                       price_trajs: np.ndarray) -> np.ndarray:
        """Vectorized rollout over all action seqs x price trajectories.

        Args:
            action_seqs: (K, H) action sequences
            price_trajs: (N, H) price trajectories
        Returns:
            scores: (K,) median profit per action sequence
        """
        b = self.battery
        K, H = action_seqs.shape
        N = price_trajs.shape[0]
        min_e = b.min_soc * b.capacity_mwh
        max_e = b.max_soc * b.capacity_mwh

        energy = np.full((K, N), obs.energy_mwh)
        total = np.zeros((K, N))

        for t in range(H):
            a = action_seqs[:, t][:, np.newaxis]                # (K, 1)
            # Energy delta: charge stores a*efficiency, discharge draws a/efficiency
            delta = np.where(a >= 0, a * b.efficiency, a / b.efficiency)  # (K, 1)
            delta = np.clip(delta, min_e - energy, max_e - energy)  # (K, N)
            total += price_trajs[np.newaxis, :, t] * (-delta)   # (K, N)
            energy += delta

        return np.median(total, axis=1)  # (K,)

    def select_action(self, observation: Observation) -> float:
        """Return normalized action in [-1, 1].

        -1 = max discharge, 0 = hold, 1 = max charge
        """
        b = self.battery
        self.model.step(observation.price)
        price_trajs = self.model.simulate(self.horizon, self.n_trajectories)

        # Scale noise relative to battery capacity
        noise = np.random.randn(self.n_samples, self.horizon) * self.noise_sigma * b.max_charge_rate_mw
        action_seqs = self._warm_start + noise
        action_seqs = np.clip(action_seqs, -b.max_discharge_rate_mw, b.max_charge_rate_mw)

        scores = self._rollout_batch(observation, action_seqs, price_trajs)

        weights = softmax(scores / self.temperature)
        optimal = weights @ action_seqs

        self._warm_start[:-1] = optimal[1:]
        self._warm_start[-1] = 0.0

        # Normalize to [-1, 1]
        action_mw = np.clip(optimal[0], -b.max_discharge_rate_mw, b.max_charge_rate_mw)
        action_normalized = action_mw / b.max_charge_rate_mw
        return float(action_normalized)

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
