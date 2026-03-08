from __future__ import annotations

from abc import ABC, abstractmethod

from battery_sim.utils.types import Observation


class Policy(ABC):
    @abstractmethod
    def select_action(self, observation: Observation) -> float:
        """Return power in MW. Positive=charge, negative=discharge."""
        ...

    def reset(self) -> None:
        pass

    def update(
        self,
        obs: Observation,
        action: float,
        reward: float,
        next_obs: Observation,
        done: bool,
    ) -> None:
        """Gym-style transition update. No-op for rule-based policies."""
        pass
