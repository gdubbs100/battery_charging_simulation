from __future__ import annotations
from typing import Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

from battery_sim.agents.policies.base import Policy
from battery_sim.utils.types import Observation
from battery_sim.optimization.policy_optimizer import sample_windows
from battery_sim.simulation.env import SimulationEnv
from battery_sim.agents.policies._utils import default_features, action_norm_to_mw, make_mlp


class REINFORCEPolicy(Policy):
    def __init__(
        self,
        net: nn.Module,
        feature_fn: Callable[[Observation], torch.Tensor] = default_features,
        lr: float = 1e-3,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
    ):
        self.net = net
        self.feature_fn = feature_fn
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.optim = torch.optim.Adam(self.net.parameters(), lr=lr)

        self.training = False
        self._log_probs = []
        self._entropies = []
        self._rewards = []

        # Batch trajectory data
        self._batch_log_probs = []
        self._batch_entropies = []
        self._batch_returns = []

        self.loss_history = []
        self.return_history = []

    def _get_alpha_beta(self, obs: Observation) -> tuple[torch.Tensor, torch.Tensor]:
        """Get Beta distribution parameters from network."""
        with torch.no_grad() if not self.training else torch.enable_grad():
            raw = self.net(self.feature_fn(obs))
        alpha = F.softplus(raw[0]) + 1e-4
        beta = F.softplus(raw[1]) + 1e-4
        return alpha, beta

    def select_action(self, observation: Observation) -> float:
        """Return normalized action in [-1, 1].

        -1 = max discharge, 0 = hold, 1 = max charge
        """
        feat = self.feature_fn(observation)

        if not self.training:
            with torch.no_grad():
                raw = self.net(feat)
            alpha = F.softplus(raw[0]) + 1e-4
            beta = F.softplus(raw[1]) + 1e-4
            # Use mode if interior, mean otherwise
            if alpha > 1 and beta > 1:
                x = (alpha - 1) / (alpha + beta - 2)
            else:
                x = alpha / (alpha + beta)
            x = float(x.clamp(0, 1).detach())
        else:
            raw = self.net(feat)
            alpha = F.softplus(raw[0]) + 1e-4
            beta = F.softplus(raw[1]) + 1e-4
            dist = Beta(alpha, beta)
            x = dist.rsample()
            self._log_probs.append(dist.log_prob(x))
            self._entropies.append(dist.entropy())
            x = float(x.clamp(0, 1).detach())

        # Map [0,1] to [-1,1]
        action = 2 * x - 1
        return float(action)

    def _accumulate_episode(self) -> None:
        """Accumulate episode data into batch buffers."""
        if not self._log_probs:
            return

        # Compute discounted returns for this episode
        returns = []
        running = 0.0
        for r in reversed(self._rewards):
            running = r + self.gamma * running
            returns.insert(0, running)

        # Store episode data for batch processing
        self._batch_log_probs.extend(self._log_probs)
        self._batch_entropies.extend(self._entropies)
        self._batch_returns.extend(returns)
        self.return_history.append(sum(self._rewards))

    def _update_policy(self) -> None:
        """REINFORCE gradient step: normalize batch returns and update policy."""
        if not self._batch_log_probs:
            return

        # Normalize returns across entire batch for stable gradient estimates
        returns_t = torch.tensor(self._batch_returns, dtype=torch.float32)
        returns_norm = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        log_probs_t = torch.stack(self._batch_log_probs)
        entropies_t = torch.stack(self._batch_entropies)

        policy_loss = -(log_probs_t * returns_norm).sum()
        entropy_bonus = self.entropy_coef * entropies_t.sum()
        loss = policy_loss - entropy_bonus

        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
        self.optim.step()

        self.loss_history.append(loss.item())

        # Clear batch buffers
        self._batch_log_probs = []
        self._batch_entropies = []
        self._batch_returns = []

    def learn(
        self,
        train_data,
        battery,
        num_iters: int = 20,
        window_len: int = 24,
        n_windows: int = 20,
        batch_size: int = 5,
        **_
    ) -> None:
        self.max_charge_rate_mw = battery.max_charge_rate_mw
        self.max_discharge_rate_mw = battery.max_discharge_rate_mw

        self.loss_history = []
        self.return_history = []
        self.training = True

        for iter_idx in range(num_iters):
            windows = sample_windows(train_data, window_len, n_windows)
            for batch_idx, window in enumerate(windows):
                env = SimulationEnv(battery=battery, price_data=window)
                obs = env.reset()
                self.reset()
                done = False

                while not done:
                    action = self.select_action(obs)
                    next_obs, reward, done = env.step(action)
                    self._rewards.append(reward)
                    obs = next_obs

                # Accumulate episode into batch
                self._accumulate_episode()

                # Update after batch_size episodes
                if (batch_idx + 1) % batch_size == 0:
                    self._update_policy()

            # Update any remaining episodes at end of iteration
            if len(self._batch_log_probs) > 0:
                self._update_policy()

            if (iter_idx + 1) % 5 == 0:
                mean_return = np.mean(self.return_history[-n_windows:])
                print(f"  iter {iter_idx+1}/{num_iters}  mean_return={mean_return:.2f}")

        self.training = False

    def reset(self) -> None:
        """Reset per-episode buffers (not batch buffers)."""
        self._log_probs = []
        self._entropies = []
        self._rewards = []
