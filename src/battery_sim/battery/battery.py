from __future__ import annotations

from battery_sim.utils.types import BatteryState


class Battery:
    def __init__(
        self,
        capacity_mwh: float,
        max_charge_rate_mw: float,
        max_discharge_rate_mw: float,
        efficiency: float = 0.9,
        initial_soc: float = 0.0,
        min_soc: float = 0.0,
        max_soc: float = 1.0,
    ):
        self.capacity_mwh = capacity_mwh
        self.max_charge_rate_mw = max_charge_rate_mw
        self.max_discharge_rate_mw = max_discharge_rate_mw
        self.efficiency = efficiency
        self.initial_soc = initial_soc
        self.min_soc = min_soc
        self.max_soc = max_soc
        self.energy_mwh = initial_soc * capacity_mwh

    @property
    def soc(self) -> float:
        return self.energy_mwh / self.capacity_mwh

    def apply_action(self, power_mw: float, duration_hours: float) -> float:
        """Apply charge (positive) or discharge (negative) action.

        Returns grid-side energy in MWh (positive = drawn from grid for charging,
        negative = delivered to grid from discharging).
        """
        if power_mw >= 0:
            # Charge: clip to rate limit
            power_mw = min(power_mw, self.max_charge_rate_mw)
            raw_energy = power_mw * duration_hours
            usable_energy = raw_energy * self.efficiency
            # Clip to available capacity
            headroom = (self.max_soc * self.capacity_mwh) - self.energy_mwh
            actual_stored = min(usable_energy, max(headroom, 0.0))
            self.energy_mwh += actual_stored
            # Grid-side: we drew actual_stored / efficiency from the grid
            grid_energy = actual_stored / self.efficiency if actual_stored > 0 else 0.0
            return grid_energy
        else:
            # Discharge: clip to rate limit
            power_mw = max(power_mw, -self.max_discharge_rate_mw)
            requested_output = abs(power_mw) * duration_hours
            # Due to efficiency, battery must discharge more than the requested output
            required_discharge = requested_output / self.efficiency
            # Clip to available energy
            available = self.energy_mwh - (self.min_soc * self.capacity_mwh)
            actual_discharge_from_battery = min(required_discharge, max(available, 0.0))
            # Grid-side: we deliver efficiency * actual_discharge to the grid
            actual_output = actual_discharge_from_battery * self.efficiency
            self.energy_mwh -= actual_discharge_from_battery
            return -actual_output

    def get_state(self) -> BatteryState:
        return BatteryState(
            soc=self.soc,
            energy_mwh=self.energy_mwh,
            capacity_mwh=self.capacity_mwh,
        )

    def reset(self, initial_soc: float | None = None) -> None:
        soc = initial_soc if initial_soc is not None else self.initial_soc
        self.energy_mwh = soc * self.capacity_mwh
