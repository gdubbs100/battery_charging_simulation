"""End-to-end example: historical replay with a threshold policy."""

import pandas as pd

from battery_sim.agents.agent import BatteryAgent
from battery_sim.agents.policies.rule_based import ThresholdPolicy
from battery_sim.battery.battery import Battery
from battery_sim.models.naive import PersistenceModel
from battery_sim.simulation.env import SimulationEnv


def run_with_synthetic_data():
    """Run a simulation with synthetic price data (no API key needed)."""
    # Generate synthetic price data: low overnight, high during peaks
    timestamps = pd.date_range("2024-01-01", periods=48, freq="30min")
    prices = [50 + 30 * (1 if 7 <= t.hour <= 20 else -1) for t in timestamps]
    price_data = pd.DataFrame({"timestamp": timestamps, "price": prices})

    battery = Battery(
        capacity_mwh=2.0,
        max_charge_rate_mw=1.0,
        max_discharge_rate_mw=1.0,
        efficiency=0.9,
    )

    policy = ThresholdPolicy(
        buy_threshold=30.0,
        sell_threshold=70.0,
        charge_rate=1.0,
        discharge_rate=1.0,
    )

    model = PersistenceModel()
    agent = BatteryAgent(policy=policy, price_model=model, forecast_horizon=6)

    env = SimulationEnv(battery=battery, price_data=price_data, interval_minutes=30)
    obs = env.reset()
    done = False

    while not done:
        action = agent.act(obs)
        next_obs, reward, done = env.step(action)
        agent.update(obs, action, reward, next_obs, done)
        obs = next_obs

    result = env.get_result()
    print(f"Total steps: {result.total_steps}")
    print(f"Cumulative revenue: ${result.cumulative_revenue:.2f}")
    print(f"Final SoC: {result.trajectory[-1].soc:.2%}")

    print("\nFirst 10 steps:")
    for rec in result.trajectory[:10]:
        print(
            f"  {rec.timestamp} | price=${rec.price:.0f} | "
            f"action={rec.action:+.1f}MW | reward=${rec.reward:.2f} | "
            f"soc={rec.soc:.2%}"
        )


def run_with_api_data():
    """Run a simulation with real NEM data (requires API key in .env)."""
    from battery_sim.data.client import PriceDataClient

    client = PriceDataClient()
    price_data = client.fetch_prices(
        network="NEM",
        region="NSW1",
        start="2024-01-01",
        end="2024-01-07",
        interval="5m",
    )

    battery = Battery(
        capacity_mwh=2.0,
        max_charge_rate_mw=1.0,
        max_discharge_rate_mw=1.0,
        efficiency=0.9,
    )

    policy = ThresholdPolicy(
        buy_threshold=50.0,
        sell_threshold=150.0,
        charge_rate=1.0,
        discharge_rate=1.0,
    )

    agent = BatteryAgent(policy=policy)
    env = SimulationEnv(battery=battery, price_data=price_data, interval_minutes=30)
    obs = env.reset()
    done = False

    while not done:
        action = agent.act(obs)
        next_obs, reward, done = env.step(action)
        agent.update(obs, action, reward, next_obs, done)
        obs = next_obs

    result = env.get_result()
    print(f"Total steps: {result.total_steps}")
    print(f"Cumulative revenue: ${result.cumulative_revenue:.2f}")


if __name__ == "__main__":
    print("=== Synthetic Data Simulation ===")
    run_with_synthetic_data()
    # Uncomment to run with real API data:
    print("\n=== Real NEM Data Simulation ===")
    run_with_api_data()
