from __future__ import annotations

from battery_sim.agents.policies.base import Policy
from battery_sim.models.base import DeterministicModel, PriceModel, StochasticModel
from battery_sim.utils.types import Observation


class BatteryAgent:
    def __init__(
        self,
        policy: Policy,
        price_model: PriceModel | None = None,
        forecast_horizon: int = 12,
        n_samples: int = 100,
    ):
        self.policy = policy
        self.price_model = price_model
        self.forecast_horizon = forecast_horizon
        self.n_samples = n_samples

    def act(self, observation: Observation) -> float:
        if self.price_model is not None:
            self.price_model.update(observation.price)
            if isinstance(self.price_model, StochasticModel):
                forecast = self.price_model.sample(self.forecast_horizon, self.n_samples)
            elif isinstance(self.price_model, DeterministicModel):
                forecast = self.price_model.predict(self.forecast_horizon)
            else:
                forecast = None
            observation = Observation(
                timestamp=observation.timestamp,
                price=observation.price,
                soc=observation.soc,
                energy_mwh=observation.energy_mwh,
                forecast=forecast,
            )
        return self.policy.select_action(observation)

    def update(
        self,
        obs: Observation,
        action: float,
        reward: float,
        next_obs: Observation,
        done: bool,
    ) -> None:
        self.policy.update(obs, action, reward, next_obs, done)

    def reset(self) -> None:
        self.policy.reset()
        if self.price_model is not None:
            self.price_model.reset()
