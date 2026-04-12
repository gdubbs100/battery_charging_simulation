from __future__ import annotations

import numpy as np
import pandas as pd

from battery_sim.agents.policies.base import Policy
from battery_sim.battery.battery import Battery
from battery_sim.simulation.env import SimulationEnv
from battery_sim.utils.types import EvalResult, SimResult


def run_episode(env: SimulationEnv, policy: Policy) -> SimResult:
    """Run one full episode. Policy-agnostic."""
    obs = env.reset()
    policy.reset()
    done = False
    while not done:
        action = policy.select_action(obs)
        obs, reward, done = env.step(action)
    return env.get_result()


def sample_windows(data: pd.DataFrame, window_len: int, n: int) -> list[pd.DataFrame]:
    """Sample n non-overlapping random windows of length window_len rows from data."""
    num_slots = len(data) // window_len
    if num_slots == 0:
        raise ValueError(f"window_len={window_len} exceeds data length {len(data)}")
    slots = np.random.choice(num_slots, size=min(n, num_slots), replace=False)
    return [data.iloc[s * window_len : (s + 1) * window_len].reset_index(drop=True) for s in slots]


def evaluate_policy(
    policy: Policy,
    data: pd.DataFrame,
    battery: Battery,
    n_windows: int = 10,
    window_len: int = 24,
    interval_minutes: int = 60,
) -> float:
    """Mean cumulative revenue over n random windows. Policy-agnostic."""
    windows = sample_windows(data, window_len, n_windows)
    revenues = []
    for window in windows:
        env = SimulationEnv(battery=battery, price_data=window, interval_minutes=interval_minutes)
        result = run_episode(env, policy)
        revenues.append(result.cumulative_revenue)
    return float(np.mean(revenues))


def evaluate_full(
    name: str,
    policy: Policy,
    data: pd.DataFrame,
    battery: Battery,
    interval_minutes: int = 60,
) -> EvalResult:
    """Run a full evaluation on data and return rich EvalResult."""
    env = SimulationEnv(battery=battery, price_data=data, interval_minutes=interval_minutes)
    sim_result = run_episode(env, policy)
    return EvalResult.from_sim(name, sim_result)
