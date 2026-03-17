from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from datetime import timedelta

import torch
import torch.nn as nn

from battery_sim.agents.policies.base import Policy
from battery_sim.utils.types import Observation
from battery_sim.optimization.policy_optimizer import sample_windows, run_episode
from battery_sim.simulation.env import SimulationEnv


def default_features(obs: Observation) -> torch.Tensor:
    """Default feature extractor: [price/100, soc]."""
    return torch.tensor([obs.price / 100.0, obs.soc], dtype=torch.float32)


class ADPPolicy(Policy):
    def __init__(
        self,
        net: nn.Module,
        feature_fn: Callable[[Observation], torch.Tensor] = default_features,
        charge_rate: float = 1.0,
        discharge_rate: float = 1.0,
        n_actions: int = 11,
        efficiency: float = 0.9,
        min_soc: float = 0.1,
        max_soc: float = 0.9,
        capacity_mwh: float = 1.0,
        lr: float = 1e-3,
        gamma: float = 0.99,
        max_grad_norm: float = 10.0,
    ):
        self.net = net
        self.feature_fn = feature_fn
        self.actions = np.linspace(-discharge_rate, charge_rate, n_actions)
        self.efficiency = efficiency
        self.min_soc = min_soc
        self.max_soc = max_soc
        self.capacity_mwh = capacity_mwh
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.optim = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.loss_history: list[float] = []

    def _predict(self, obs: Observation) -> float:
        with torch.no_grad():
            return self.net(self.feature_fn(obs)).squeeze(-1).item()

    def _approx_next(self, obs: Observation, action: float) -> tuple[float, Observation]:
        delta = action * self.efficiency if action >= 0 else action
        delta = np.clip(
            delta,
            self.min_soc * self.capacity_mwh - obs.energy_mwh,
            self.max_soc * self.capacity_mwh - obs.energy_mwh,
        )
        next_obs = Observation(
            timestamp=obs.timestamp + timedelta(hours=1),
            price=obs.price,
            soc=(obs.energy_mwh + delta) / self.capacity_mwh,
            energy_mwh=obs.energy_mwh + delta,
        )
        output = -delta
        reward = next_obs.price * output
        return reward, next_obs

    def select_action(self, observation: Observation) -> float:
        best, best_val = 0.0, -np.inf
        for x in self.actions:
            r, next_obs = self._approx_next(observation, x)
            v = r + self.gamma * self._predict(next_obs)
            if v > best_val:
                best_val, best = v, x
        return float(best)

    def learn(self, train_data: pd.DataFrame, battery, num_iters=10,
              window_len=24, n_windows=20, **_):
        self.efficiency = battery.efficiency
        self.min_soc = battery.min_soc
        self.max_soc = battery.max_soc
        self.capacity_mwh = battery.capacity_mwh

        loss_fn = nn.MSELoss()
        self.loss_history = []

        for i in range(num_iters):
            windows = sample_windows(train_data, window_len, n_windows)
            feats, targets = [], []

            for window in windows:
                env = SimulationEnv(battery=battery, price_data=window)
                result = run_episode(env, self)
                traj = result.trajectory

                for j in range(len(traj) - 1):
                    curr, nxt = traj[j], traj[j + 1]
                    next_obs = Observation(
                        timestamp=nxt.timestamp, price=nxt.price,
                        soc=nxt.soc, energy_mwh=nxt.energy_mwh,
                    )
                    target = curr.reward + self.gamma * self._predict(next_obs)
                    curr_obs = Observation(
                        timestamp=curr.timestamp, price=curr.price,
                        soc=curr.soc, energy_mwh=curr.energy_mwh,
                    )
                    feats.append(self.feature_fn(curr_obs))
                    targets.append(target)

            X = torch.stack(feats)
            y = torch.tensor(targets, dtype=torch.float32)

            self.optim.zero_grad()
            loss = loss_fn(self.net(X).squeeze(-1), y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
            self.optim.step()

            loss_val = loss.item()
            self.loss_history.append(loss_val)
            print(f"  iter {i+1}/{num_iters}  loss={loss_val:.4f}")

    def reset(self) -> None:
        pass
