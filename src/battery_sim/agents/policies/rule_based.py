from __future__ import annotations

from battery_sim.agents.policies.base import Policy
from battery_sim.utils.types import Observation


class ThresholdPolicy(Policy):
    def __init__(
        self,
        charge_below: float,
        discharge_above: float,
        charge_rate: float,
        discharge_rate: float,
    ):
        self.charge_below = charge_below
        self.discharge_above = discharge_above
        self.charge_rate = charge_rate
        self.discharge_rate = discharge_rate

    def select_action(self, observation: Observation) -> float:
        if observation.price < self.charge_below:
            return self.charge_rate
        elif observation.price > self.discharge_above:
            return -self.discharge_rate
        return 0.0


class TimeOfUsePolicy(Policy):
    def __init__(
        self,
        charge_hours: set[int],
        discharge_hours: set[int],
        charge_rate: float,
        discharge_rate: float,
    ):
        self.charge_hours = charge_hours
        self.discharge_hours = discharge_hours
        self.charge_rate = charge_rate
        self.discharge_rate = discharge_rate

    def select_action(self, observation: Observation) -> float:
        hour = observation.timestamp.hour
        if hour in self.charge_hours:
            return self.charge_rate
        elif hour in self.discharge_hours:
            return -self.discharge_rate
        return 0.0
