import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linprog
from battery_sim.utils.types import Observation


def default_features(obs: Observation) -> torch.Tensor:
    """Extract [price/100, soc] features from observation."""
    return torch.tensor([obs.price / 100.0, obs.soc], dtype=torch.float32)


def action_norm_to_mw(x: float, max_charge: float, max_discharge: float) -> float:
    """Map normalized action x in [0,1] to MW.

    x=0 -> -max_discharge (full discharge)
    x=0.5 -> (max_charge - max_discharge) / 2
    x=1 -> max_charge (full charge)
    """
    return x * (max_charge + max_discharge) - max_discharge


def make_mlp(in_features: int, out_features: int, hidden: int = 16) -> nn.Sequential:
    """Small MLP: in -> hidden(Tanh) -> out."""
    return nn.Sequential(
        nn.Linear(in_features, hidden),
        nn.Tanh(),
        nn.Linear(hidden, out_features)
    )


def solve_lp(
    prices: np.ndarray,
    battery,
    interval_minutes: int = 60,
    e_0: float | None = None,
) -> np.ndarray:
    """Solve LP for optimal battery charge/discharge given price series.

    Args:
        prices: array shape (T+1,) where prices[0] is current price,
                prices[1:T+1] are execution prices
        battery: Battery object with capacity, rate limits, efficiency, SoC bounds
        interval_minutes: time delta per step (default 60 for hourly)
        e_0: starting energy in MWh (default: battery.initial_soc * capacity_mwh)

    Returns:
        net power array shape (T,) where positive=charge, negative=discharge
    """
    b = battery
    dt = interval_minutes / 60.0
    T = len(prices) - 1
    exec_prices = prices[1:]

    if e_0 is None:
        e_0 = b.initial_soc * b.capacity_mwh

    # Decision variables: [c_0, ..., c_{T-1}, d_0, ..., d_{T-1}]
    # Charge cost: c[t] * efficiency * price[t+1] * dt (energy stored after losses)
    c_obj = b.efficiency * exec_prices * dt
    # Discharge revenue: d[t] * price[t+1] * dt, but cost to battery is d[t] / efficiency * dt
    # So effective cost: -d[t] / efficiency * price[t+1] * dt (negative = revenue)
    d_obj = -exec_prices / b.efficiency * dt
    c_obj_full = np.concatenate([c_obj, d_obj])

    # Bounds
    bounds = [(0, b.max_charge_rate_mw)] * T + [(0, b.max_discharge_rate_mw)] * T

    # SoC constraints
    min_e = b.min_soc * b.capacity_mwh
    max_e = b.max_soc * b.capacity_mwh
    A_ub = np.zeros((2 * T, 2 * T))
    b_ub = np.zeros(2 * T)

    for t in range(1, T + 1):
        # Energy balance: e[t] = e_0 + sum_{s<=t} (c[s]*eff - d[s]/eff) * dt
        # Lower bound: e[t] >= min_e => -sum c*eff*dt + sum d/eff*dt <= e_0 - min_e
        A_ub[2 * (t - 1), :t] = -b.efficiency * dt
        A_ub[2 * (t - 1), T:T + t] = dt / b.efficiency
        b_ub[2 * (t - 1)] = e_0 - min_e

        # Upper bound: e[t] <= max_e => sum c*eff*dt - sum d/eff*dt <= max_e - e_0
        A_ub[2 * (t - 1) + 1, :t] = b.efficiency * dt
        A_ub[2 * (t - 1) + 1, T:T + t] = -dt / b.efficiency
        b_ub[2 * (t - 1) + 1] = max_e - e_0

    result = linprog(c_obj_full, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if not result.success:
        raise RuntimeError(f"LP failed: {result.message}")

    return result.x[:T] - result.x[T:]
