from __future__ import annotations
import copy
import random
from collections import deque
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from battery_sim.agents.policies.base import Policy
from battery_sim.agents.policies._utils import default_features, log_transform_reward, make_mlp
from battery_sim.optimization.policy_optimizer import sample_windows
from battery_sim.simulation.env import SimulationEnv
from battery_sim.utils.types import Observation


class DQNPolicy(Policy):
    def __init__(
        self,
        net: nn.Module,
        feature_fn: Callable[[Observation], torch.Tensor] = default_features,
        n_actions: int = 11,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 2000,
        batch_size: int = 32,
        buffer_size: int = 2000,
        target_update_freq: int = 100,
    ):
        self.net = net
        self.feature_fn = feature_fn
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Target network for stable Q-value targets (deep copy of network)
        self.target_net = copy.deepcopy(net)
        self.target_net.eval()  # Target net doesn't need gradients

        self.optim = torch.optim.Adam(self.net.parameters(), lr=lr)

        # Precompute action grid (normalized [-1, 1])
        self._action_grid = np.linspace(-1.0, 1.0, n_actions)

        # Replay buffer
        self._buffer = deque(maxlen=buffer_size)

        # Training state
        self._step_count = 0
        self._epsilon = epsilon_start
        self._last_action_idx = None
        self._last_feat = None

        self.loss_history = []
        self.epsilon_history = []

    def _update_target_network(self) -> None:
        """Update target network weights from main network."""
        self.target_net.load_state_dict(self.net.state_dict())

    def select_action(self, observation: Observation) -> float:
        """Return normalized action in [-1, 1].

        -1 = max discharge, 0 = hold, 1 = max charge
        """
        feat = self.feature_fn(observation)
        self._last_feat = feat

        if np.random.random() < self._epsilon:
            idx = np.random.randint(self.n_actions)
        else:
            with torch.no_grad():
                q_vals = self.net(feat)
                idx = q_vals.argmax().item()

        self._last_action_idx = idx
        return float(self._action_grid[idx])

    def update(
        self,
        obs: Observation,
        action: float,
        reward: float,
        next_obs: Observation,
        done: bool,
    ) -> None:
        feat_next = self.feature_fn(next_obs)
        self._buffer.append((self._last_feat, self._last_action_idx, reward, feat_next, done))

        self._step_count += 1
        self._epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(
            -self._step_count / self.epsilon_decay
        )

        if len(self._buffer) >= self.batch_size:
            self._gradient_step()

    def _gradient_step(self) -> None:
        batch = random.sample(self._buffer, self.batch_size)
        feats, idxs, rewards, next_feats, dones = zip(*batch)

        feats_t = torch.stack(feats)
        next_feats_t = torch.stack(next_feats)
        dones_t = torch.tensor(dones, dtype=torch.float32)
        idxs_t = torch.tensor(idxs, dtype=torch.long)

        # Log-transform rewards for numerical stability
        # (battery rewards range [-5000, +1200], causing huge Q-values and gradients)
        # Transform: sign(r) * log(1 + |r|) compresses range to ~[-8.5, +7]
        rewards_transformed = torch.tensor(
            [log_transform_reward(r) for r in rewards],
            dtype=torch.float32
        )  # Scale to reasonable magnitude

        q_vals = self.net(feats_t)
        q_taken = q_vals.gather(1, idxs_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Use target network for stable Q-value targets (key DQN improvement)
            q_next = self.target_net(next_feats_t).max(1).values
            targets = rewards_transformed + self.gamma * q_next * (1 - dones_t)

        loss = F.mse_loss(q_taken, targets)
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
        self.optim.step()

        self.loss_history.append(loss.item())

        # Update target network periodically
        if self._step_count % self.target_update_freq == 0:
            self._update_target_network()

    def learn(
        self,
        train_data,
        battery,
        num_iters: int = 10,
        window_len: int = 24,
        n_windows: int = 20,
        **_
    ) -> None:
        self._epsilon = self.epsilon_start
        self._step_count = 0
        self._buffer.clear()
        self._update_target_network()
        self.loss_history = []
        self.epsilon_history = []

        for iter_idx in range(num_iters):
            windows = sample_windows(train_data, window_len, n_windows)
            for window in windows:
                env = SimulationEnv(battery=battery, price_data=window)
                obs = env.reset()
                self.reset()
                done = False

                while not done:
                    action = self.select_action(obs)
                    next_obs, reward, done = env.step(action)
                    self.update(obs, action, reward, next_obs, done)
                    obs = next_obs

                self.epsilon_history.append(self._epsilon)

            if (iter_idx + 1) % 5 == 0:
                mean_loss = np.mean(self.loss_history[-100:]) if self.loss_history else 0
                print(f"  iter {iter_idx+1}/{num_iters}  loss={mean_loss:.4f}  epsilon={self._epsilon:.4f}")

        self._epsilon = 0.0

    def reset(self) -> None:
        self._last_action_idx = None
        self._last_feat = None
