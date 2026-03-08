from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np


@dataclass
class Observation:
    timestamp: datetime
    price: float  # $/MWh
    soc: float  # 0.0–1.0
    energy_mwh: float
    forecast: np.ndarray | None = None


@dataclass
class BatteryState:
    soc: float
    energy_mwh: float
    capacity_mwh: float


@dataclass
class StepRecord:
    timestamp: datetime
    price: float
    action: float  # MW
    reward: float  # $
    soc: float
    energy_mwh: float


@dataclass
class SimResult:
    trajectory: list[StepRecord] = field(default_factory=list)
    cumulative_revenue: float = 0.0
    total_steps: int = 0
