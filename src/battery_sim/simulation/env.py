from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from battery_sim.battery.battery import Battery
from battery_sim.models.base import StochasticModel
from battery_sim.utils.types import Observation, SimResult, StepRecord


class SimulationEnv:
    def __init__(
        self,
        battery: Battery,
        interval_minutes: int = 5,
        # Mode 1: historical replay
        price_data: pd.DataFrame | None = None,
        # Mode 2: stochastic dynamics
        price_model: StochasticModel | None = None,
        n_steps: int | None = None,
        initial_price: float | None = None,
    ):
        if price_data is None and price_model is None:
            raise ValueError("Provide either price_data or price_model")
        if price_data is not None and price_model is not None:
            raise ValueError("Provide only one of price_data or price_model")
        if price_model is not None and n_steps is None:
            raise ValueError("n_steps required for stochastic mode")

        self.battery = battery
        self.interval_minutes = interval_minutes
        self.duration_hours = interval_minutes / 60.0

        # Mode 1
        self._price_data = price_data
        self._prices: list[float] = []
        self._timestamps: list[datetime] = []

        # Mode 2
        self._price_model = price_model
        self._n_steps = n_steps
        self._initial_price = initial_price

        self._step_idx = 0
        self._max_steps = 0
        self._trajectory: list[StepRecord] = []
        self._cumulative_revenue = 0.0

    def reset(self) -> Observation:
        self.battery.reset()
        self._step_idx = 0
        self._trajectory = []
        self._cumulative_revenue = 0.0

        if self._price_data is not None:
            df = self._price_data.copy()
            self._timestamps = pd.to_datetime(df.iloc[:, 0]).tolist()
            self._prices = df.iloc[:, 1].astype(float).tolist()
            self._max_steps = len(self._prices)
        else:
            # Stochastic mode: seed the model and generate first price
            self._max_steps = self._n_steps  # type: ignore
            if self._initial_price is not None:
                self._price_model.update(self._initial_price)  # type: ignore
            self._prices = []
            self._timestamps = []
            now = datetime.now()
            # Generate first price by sampling 1 step ahead
            first_price = self._price_model.sample(1, 1)[0, 0]  # type: ignore
            self._prices.append(float(first_price))
            self._timestamps.append(now)

        state = self.battery.get_state()
        return Observation(
            timestamp=self._timestamps[0],
            price=self._prices[0],
            soc=state.soc,
            energy_mwh=state.energy_mwh,
        )

    def step(self, action: float) -> tuple[Observation, float, bool]:
        if self._step_idx >= self._max_steps:
            raise RuntimeError("Episode is done. Call reset().")

        price = self._prices[self._step_idx]
        energy = self.battery.apply_action(action, self.duration_hours)

        # Reward: revenue from discharge (negative energy = discharged)
        # energy > 0 means charged (cost), energy < 0 means discharged (revenue)
        reward = -energy * price  # discharge: -(-E)*P = E*P > 0; charge: -(+E)*P < 0

        state = self.battery.get_state()
        self._trajectory.append(StepRecord(
            timestamp=self._timestamps[self._step_idx],
            price=price,
            action=action,
            reward=reward,
            soc=state.soc,
            energy_mwh=state.energy_mwh,
        ))
        self._cumulative_revenue += reward
        self._step_idx += 1

        done = self._step_idx >= self._max_steps

        if not done:
            # Generate next price for stochastic mode
            if self._price_model is not None:
                self._price_model.update(price)  # type: ignore
                next_price = self._price_model.sample(1, 1)[0, 0]  # type: ignore
                self._prices.append(float(next_price))
                self._timestamps.append(
                    self._timestamps[-1] + timedelta(minutes=self.interval_minutes)
                )

            obs = Observation(
                timestamp=self._timestamps[self._step_idx],
                price=self._prices[self._step_idx],
                soc=state.soc,
                energy_mwh=state.energy_mwh,
            )
        else:
            obs = Observation(
                timestamp=self._timestamps[-1],
                price=price,
                soc=state.soc,
                energy_mwh=state.energy_mwh,
            )

        return obs, reward, done

    def get_result(self) -> SimResult:
        return SimResult(
            trajectory=list(self._trajectory),
            cumulative_revenue=self._cumulative_revenue,
            total_steps=len(self._trajectory),
        )
