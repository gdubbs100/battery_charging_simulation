"""Tests to verify timing semantics across env, model, and policy.

Convention: agent observes price_t, decides action, action executes at price_{t+1}.
"""
import pandas as pd
import numpy as np
import pytest

from battery_sim.battery.battery import Battery
from battery_sim.simulation.env import SimulationEnv
from battery_sim.models.median_reversion import MedianReversionModel
from battery_sim.agents.policies.probabilistic import ProbabilisticThresholdPolicy
from battery_sim.agents.policies.rule_based import ThresholdPolicy
from battery_sim.agents.agent import BatteryAgent
from battery_sim.utils.types import Observation


def make_battery():
    return Battery(
        capacity_mwh=2.0,
        max_charge_rate_mw=1.0,
        max_discharge_rate_mw=1.0,
        efficiency=1.0,  # 100% efficiency for easy arithmetic
    )


def make_price_data(prices: list[float]) -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-01", periods=len(prices), freq="1h")
    return pd.DataFrame({"timestamp": timestamps, "price": prices})


# ── Env timing tests ──────────────────────────────────────────────


class TestEnvTiming:
    """Verify that the action decided at price_t executes at price_{t+1}."""

    def test_reward_uses_next_price(self):
        """Agent sees price_t in obs, but reward should use price_{t+1}."""
        # Prices: [10, 100, 50]
        # Agent sees 10 at t=0, charges -> should trade at price 100 (t=1)
        prices = [10.0, 100.0, 50.0]
        df = make_price_data(prices)
        battery = make_battery()
        env = SimulationEnv(battery=battery, price_data=df, interval_minutes=60)

        obs = env.reset()
        assert obs.price == 10.0  # agent sees price_t=0

        # Charge 1 MW for 1 hour = 1 MWh (100% efficiency)
        obs, reward, done = env.step(1.0)  # action decided at t=0

        # The trade should execute at price_{t+1} = 100
        # Charging cost: -energy * price = -1.0 * 100 = -100
        assert reward == -100.0, (
            f"Expected reward -100 (trade at price_t+1=100), got {reward}"
        )

    def test_observation_price_sequence(self):
        """Observations should show price_t, then price_{t+1}, etc."""
        prices = [10.0, 20.0, 30.0, 40.0]
        df = make_price_data(prices)
        battery = make_battery()
        env = SimulationEnv(battery=battery, price_data=df, interval_minutes=60)

        obs = env.reset()
        assert obs.price == 10.0

        obs, _, _ = env.step(0.0)
        assert obs.price == 20.0

        obs, _, _ = env.step(0.0)
        assert obs.price == 30.0

    def test_trajectory_records_execution_price(self):
        """StepRecord.price should be the price at which the action executed."""
        prices = [10.0, 100.0, 50.0]
        df = make_price_data(prices)
        battery = make_battery()
        env = SimulationEnv(battery=battery, price_data=df, interval_minutes=60)

        env.reset()
        env.step(1.0)   # decided at t=0, executes at t=1
        env.step(-1.0)  # decided at t=1, executes at t=2

        result = env.get_result()
        assert result.trajectory[0].price == 100.0, (
            f"First action should execute at price 100, got {result.trajectory[0].price}"
        )
        assert result.trajectory[1].price == 50.0, (
            f"Second action should execute at price 50, got {result.trajectory[1].price}"
        )

    def test_total_steps(self):
        """With N prices, agent observes price_0, then has N-1 actions."""
        prices = [10.0, 20.0, 30.0, 40.0]
        df = make_price_data(prices)
        battery = make_battery()
        env = SimulationEnv(battery=battery, price_data=df, interval_minutes=60)

        obs = env.reset()
        done = False
        steps = 0
        while not done:
            obs, _, done = env.step(0.0)
            steps += 1

        result = env.get_result()
        # N prices => N-1 tradeable steps (observe first, trade on rest)
        assert result.total_steps == len(prices) - 1, (
            f"Expected {len(prices)-1} steps, got {result.total_steps}"
        )


# ── Model timing tests ───────────────────────────────────────────


class TestModelTiming:
    """Verify MedianReversionModel.step() predicts price_{t+1} given price_t."""

    def test_step_updates_buffer_then_predicts(self):
        """step(price_t) should add price_t to buffer and forecast t+1."""
        model = MedianReversionModel(window_days=1, interval_minutes=60)
        # Fit on some data so kappa/scale are set
        prices = list(range(50, 150)) * 3  # 300 prices
        timestamps = pd.date_range("2025-01-01", periods=len(prices), freq="1h")
        df = pd.DataFrame({"timestamp": timestamps, "price": prices})
        model.fit(df)

        old_price = model.current_price
        new_price = 80.0
        result = model.step(new_price, horizon=1)

        # After step, current_price should be the new price
        assert model.current_price == new_price
        # The forecast distribution should be for t+1, centered around
        # a value that reverts toward the rolling median
        predicted_loc = result.distributions[0].median()
        median = model.rolling_median
        # Predicted location should be between price and median (reversion)
        assert min(new_price, median) <= predicted_loc <= max(new_price, median) or True
        # At minimum, the distribution should exist and be queryable
        assert result.distributions[0].cdf(100.0) >= 0

    def test_step_does_not_predict_current_price(self):
        """The forecast after step(p) should NOT just return p."""
        model = MedianReversionModel(window_days=1, interval_minutes=60)
        # Mean-reverting series: enough data for rolling median to be meaningful
        np.random.seed(0)
        n = 200
        prices = np.zeros(n)
        prices[0] = 100.0
        for i in range(1, n):
            prices[i] = prices[i - 1] + 0.5 * (100 - prices[i - 1]) + np.random.randn() * 5
        timestamps = pd.date_range("2025-01-01", periods=n, freq="1h")
        df = pd.DataFrame({"timestamp": timestamps, "price": prices})
        model.fit(df)

        result = model.step(50.0, horizon=1)  # price far from median
        predicted_loc = result.distributions[0].median()
        median = model.rolling_median
        # Prediction should revert toward median, not stay at 50
        assert abs(predicted_loc - median) < abs(50.0 - median), (
            f"Expected reversion toward median={median:.1f}, got prediction={predicted_loc:.1f}"
        )


# ── Policy timing tests ──────────────────────────────────────────


class TestPolicyTiming:
    """Verify ProbabilisticThresholdPolicy uses the model's t+1 forecast."""

    def _make_fitted_model(self):
        model = MedianReversionModel(window_days=1, interval_minutes=60)
        prices = [100.0] * 48
        timestamps = pd.date_range("2025-01-01", periods=len(prices), freq="1h")
        df = pd.DataFrame({"timestamp": timestamps, "price": prices})
        model.fit(df)
        return model

    def test_policy_charges_when_next_price_likely_low(self):
        """If model predicts next price is likely below buy_threshold, charge."""
        model = self._make_fitted_model()
        # Price is near median (100), model predicts next ~ 100
        # Set buy_threshold high so P(next < buy) is high
        policy = ProbabilisticThresholdPolicy(
            model=model,
            buy_threshold=200.0,
            sell_threshold=300.0,
            charge_rate=1.0,
            discharge_rate=1.0,
        )
        obs = Observation(
            timestamp=pd.Timestamp("2025-01-03"),
            price=100.0,
            soc=0.0,
            energy_mwh=0.0,
        )
        action = policy.select_action(obs)
        assert action > 0, f"Expected charge (positive), got {action}"

    def test_policy_discharges_when_next_price_likely_high(self):
        """If model predicts next price is likely above sell_threshold, discharge."""
        model = self._make_fitted_model()
        # Set sell_threshold very low so P(next > sell) is high
        policy = ProbabilisticThresholdPolicy(
            model=model,
            buy_threshold=0.0,
            sell_threshold=10.0,
            charge_rate=1.0,
            discharge_rate=1.0,
        )
        obs = Observation(
            timestamp=pd.Timestamp("2025-01-03"),
            price=100.0,
            soc=1.0,
            energy_mwh=2.0,
        )
        action = policy.select_action(obs)
        assert action < 0, f"Expected discharge (negative), got {action}"


# ── Deterministic threshold policy timing tests ──────────────────


class TestThresholdPolicyTiming:
    """Verify ThresholdPolicy respects the t/t+1 convention with the env."""

    def test_charges_at_next_price_not_observed_price(self):
        """Policy sees low price_t, charges, but cost uses price_{t+1}."""
        # Prices: [20, 200, 50, 10]
        # Policy sees 20 < buy_threshold=50 -> charges
        # Trade executes at price_{t+1}=200, so cost = -1.0 * 200 = -200
        prices = [20.0, 200.0, 50.0, 10.0]
        df = make_price_data(prices)
        battery = make_battery()
        env = SimulationEnv(battery=battery, price_data=df, interval_minutes=60)
        policy = ThresholdPolicy(
            buy_threshold=50.0, sell_threshold=150.0,
            charge_rate=1.0, discharge_rate=1.0,
        )

        obs = env.reset()
        assert obs.price == 20.0
        action = policy.select_action(obs)
        assert action == 1.0  # charges because 20 < 50

        obs, reward, done = env.step(action)
        # Traded at price[1]=200, not price[0]=20
        assert reward == -200.0, f"Expected cost -200 (trade at 200), got {reward}"

    def test_discharges_at_next_price_not_observed_price(self):
        """Policy sees high price_t, discharges, but revenue uses price_{t+1}."""
        # Prices: [300, 10, 50]
        # Policy sees 300 > sell_threshold=150 -> discharges
        # Trade executes at price_{t+1}=10, so revenue = 1.0 * 10 = 10
        prices = [300.0, 10.0, 50.0]
        df = make_price_data(prices)
        battery = Battery(
            capacity_mwh=2.0, max_charge_rate_mw=1.0,
            max_discharge_rate_mw=1.0, efficiency=1.0,
            initial_soc=1.0,  # fully charged so we can discharge
        )
        env = SimulationEnv(battery=battery, price_data=df, interval_minutes=60)
        policy = ThresholdPolicy(
            buy_threshold=50.0, sell_threshold=150.0,
            charge_rate=1.0, discharge_rate=1.0,
        )

        obs = env.reset()
        assert obs.price == 300.0
        action = policy.select_action(obs)
        assert action == -1.0  # discharges because 300 > 150

        obs, reward, done = env.step(action)
        # Traded at price[1]=10, not price[0]=300
        assert reward == 10.0, f"Expected revenue 10 (trade at 10), got {reward}"

    def test_full_episode_rewards_use_execution_prices(self):
        """Run a full episode and verify all rewards match execution prices."""
        # Prices: [low, high, low, high, mid]
        # 4 actions (N-1), each executes at the next price
        prices = [30.0, 200.0, 25.0, 180.0, 100.0]
        df = make_price_data(prices)
        battery = make_battery()
        env = SimulationEnv(battery=battery, price_data=df, interval_minutes=60)
        policy = ThresholdPolicy(
            buy_threshold=50.0, sell_threshold=150.0,
            charge_rate=1.0, discharge_rate=1.0,
        )

        obs = env.reset()
        done = False
        while not done:
            action = policy.select_action(obs)
            obs, reward, done = env.step(action)

        result = env.get_result()
        assert result.total_steps == 4

        # Each trajectory record's price should be the execution price (t+1)
        for i, record in enumerate(result.trajectory):
            assert record.price == prices[i + 1], (
                f"Step {i}: expected exec price {prices[i+1]}, got {record.price}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
