# Battery Charging Simulation - Implementation Plan

## Context

Build a modular battery charging simulation using real Australian electricity price data from the openElectricity API. Simulate battery charge/discharge decisions against historical price data or stochastic price dynamics. Evaluate different policies (rule-based, MPC, RL). Designed for the NEM (National Electricity Market).

---

## Project Structure

```
battery_charging_simulation/
├── pyproject.toml
├── .env                        # OPENELECTRICITY_API_KEY=...
├── .env.example                # Template
├── CLAUDE.md
├── plan.md                     # Copy of this plan
├── src/
│   └── battery_sim/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   └── client.py       # OpenElectricity API wrapper
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py         # PriceModel, DeterministicModel, StochasticModel
│       │   └── naive.py        # Persistence, MovingAverage baselines
│       ├── simulation/
│       │   ├── __init__.py
│       │   └── env.py          # Gym-like SimulationEnv
│       ├── battery/
│       │   ├── __init__.py
│       │   └── battery.py      # Battery physics model
│       ├── agents/
│       │   ├── __init__.py
│       │   ├── agent.py        # BatteryAgent wrapper
│       │   └── policies/
│       │       ├── __init__.py
│       │       ├── base.py     # Abstract Policy (RL-ready)
│       │       └── rule_based.py
│       └── utils/
│           ├── __init__.py
│           └── types.py        # Shared dataclasses & enums
├── data/
│   └── cache/                  # CSV cache for fetched price data
└── examples/
    └── basic_sim.py
```

---

## Module Details

### 1. Data Collection — `src/battery_sim/data/client.py`

Uses the official `openelectricity` Python package.

**Class: `PriceDataClient`**
- Wraps `OEClient` with convenience methods for price data
- `fetch_prices(network, region, start, end, interval) -> pd.DataFrame`
- `fetch_demand(network, region, start, end, interval) -> pd.DataFrame`
- API key loaded from `.env` file via `python-dotenv`
- **Caching**: CSV files in `data/cache/`, keyed by `{network}_{region}_{start}_{end}_{interval}.csv`. Checks cache before API calls.
- Networks: `NEM`, `WEM` | Regions: `NSW1`, `QLD1`, `VIC1`, `SA1`, `TAS1`
- Intervals: `5m`, `30m`, `1h`, `1d`

### 2. Price Forecast Models — `src/battery_sim/models/`

Three-level hierarchy:

```python
class PriceModel(ABC):
    """Base class — shared interface."""
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame: ...  # public
    def fit(self, data: pd.DataFrame) -> None: ...                 # optional, default no-op
    def update(self, new_observation: float) -> None: ...          # online/rolling update
    def reset(self) -> None: ...

class DeterministicModel(PriceModel):
    """Models that produce point estimates."""
    def predict(self, horizon: int) -> np.ndarray: ...             # shape (horizon,)
    def forecast(self, data: pd.DataFrame, horizon: int) -> np.ndarray: ...

class StochasticModel(PriceModel):
    """Models that sample from a distribution."""
    def get_distribution(self, horizon: int) -> list[rv_continuous]: ...  # scipy dist per step
    def sample(self, horizon: int, n: int = 1) -> np.ndarray: ...        # shape (n, horizon)
    def predict(self, horizon: int) -> np.ndarray: ...                   # returns mean of distribution
    def forecast(self, data: pd.DataFrame, horizon: int) -> np.ndarray: ...
```

- `preprocess`: public — transform raw data into model-ready features
- `fit`: train/calibrate (default no-op for pre-trained models)
- `update`: feed one new price for rolling/online state
- Stateful: maintains internal buffer of recent observations
- **Fully separate** from the simulation env — agent owns the model

**Pre-trained model integration**: subclass `DeterministicModel` or `StochasticModel`, pass trained model to `__init__`, implement `preprocess` + `predict`/`sample`.

**Baselines** (`naive.py`):
- `PersistenceModel(DeterministicModel)` — predicts last observed price
- `MovingAverageModel(DeterministicModel)` — rolling mean over a window

### 3. Simulation Environment — `src/battery_sim/simulation/env.py`

Passive Gym-like environment. Agent drives the loop **externally**. Two operating modes:

**Mode 1 — Historical replay**: step through a real price DataFrame
**Mode 2 — Stochastic dynamics**: sample next price from a `StochasticModel` each step

```python
class SimulationEnv:
    def __init__(
        self,
        battery: Battery,
        interval_minutes: int = 5,
        # Mode 1: historical
        price_data: pd.DataFrame | None = None,
        # Mode 2: stochastic
        price_model: StochasticModel | None = None,
        n_steps: int | None = None,           # required for mode 2
        initial_price: float | None = None,    # optional seed price for mode 2
    ): ...
    def reset(self) -> Observation: ...
    def step(self, action: float) -> tuple[Observation, float, bool]: ...
    def get_result(self) -> SimResult: ...
```

- Must provide either `price_data` (mode 1) or `price_model` + `n_steps` (mode 2)
- **Action space**: continuous float (MW). Positive = charge, negative = discharge. Battery clips to feasible range.
- `reset()` → initial `Observation` (resets battery, rewinds to start or generates fresh price sequence)
- `step(action)` → applies action to battery, advances time, returns `(next_obs, reward, done)`
- **Reward** = `price * energy_discharged - price * energy_charged` (simple revenue per step)
- Configurable interval: resamples price_data if needed (5m, 30m, 1h)
- Records full trajectory internally for `get_result()`
- **No forecast model** inside env — agent handles forecasting separately

**Usage — historical replay:**
```python
env = SimulationEnv(battery=battery, price_data=df, interval_minutes=30)
obs = env.reset()
done = False
while not done:
    action = agent.act(obs)
    obs, reward, done = env.step(action)
result = env.get_result()
```

**Usage — stochastic dynamics:**
```python
env = SimulationEnv(battery=battery, price_model=stoch_model, n_steps=288, interval_minutes=5)
obs = env.reset()
# same loop as above — env samples price from model each step
```

### 4a. Battery Model — `src/battery_sim/battery/battery.py`

```python
class Battery:
    def __init__(self, capacity_mwh, max_charge_rate_mw, max_discharge_rate_mw,
                 efficiency=0.9, initial_soc=0.0, min_soc=0.0, max_soc=1.0): ...
    def apply_action(self, power_mw: float, duration_hours: float) -> float: ...
    def get_state(self) -> BatteryState: ...
    def reset(self, initial_soc=None) -> None: ...
```

- Single `apply_action`: positive = charge, negative = discharge
- Returns actual energy transacted (after clipping + efficiency)
- Enforces capacity limits (min/max SoC) and rate limits
- Efficiency loss applied on charge (round-trip)
- Clips to feasible range silently

### 4b. Agent & Policies — `src/battery_sim/agents/`

**Policy interface** (`policies/base.py`) — RL-ready, Gym-style:
```python
class Policy(ABC):
    def select_action(self, observation: Observation) -> float: ...
    def reset(self) -> None: ...
    def update(self, obs: Observation, action: float, reward: float,
               next_obs: Observation, done: bool) -> None: ...  # no-op for rule-based
```

- Action is a continuous float (MW power)
- `update` uses Gym-style transition tuple for RL compatibility

**Rule-based policies** (`policies/rule_based.py`):
- `ThresholdPolicy(charge_below, discharge_above, charge_rate, discharge_rate)`
- `TimeOfUsePolicy(charge_hours, discharge_hours, charge_rate, discharge_rate)`

**Agent** (`agents/agent.py`):
```python
class BatteryAgent:
    def __init__(self, policy: Policy, price_model: PriceModel | None = None,
                 forecast_horizon: int = 12): ...
    def act(self, observation: Observation) -> float: ...
    def update(self, obs, action, reward, next_obs, done) -> None: ...
    def reset(self) -> None: ...
```
- Owns an optional `PriceModel`. Each step:
  1. `price_model.update(obs.price)`
  2. `predict(horizon)` (deterministic) or `sample(horizon, n)` (stochastic)
  3. Attaches forecast to observation, passes to policy
- Delegates action selection to policy. Does not own battery.

### 5. Shared Types — `src/battery_sim/utils/types.py`

```python
@dataclass
class Observation:
    timestamp: datetime
    price: float              # $/MWh
    soc: float                # 0.0–1.0
    energy_mwh: float         # current energy in battery
    forecast: np.ndarray | None = None  # populated by agent

@dataclass
class BatteryState:
    soc: float
    energy_mwh: float
    capacity_mwh: float

@dataclass
class StepRecord:
    timestamp: datetime
    price: float
    action: float         # MW
    reward: float         # $ this step
    soc: float
    energy_mwh: float

@dataclass
class SimResult:
    trajectory: list[StepRecord]
    cumulative_revenue: float
    total_steps: int
```

---

## Dependencies

```toml
[project]
name = "battery-sim"
requires-python = ">=3.10"
dependencies = [
    "openelectricity[analysis]",
    "pandas",
    "numpy",
    "scipy",
    "python-dotenv",
]
```

---

## Implementation Order

1. `pyproject.toml` + package structure + `__init__.py` files + `.env.example`
2. `utils/types.py` — shared dataclasses
3. `battery/battery.py` — battery physics
4. `agents/policies/base.py` — abstract policy
5. `agents/policies/rule_based.py` — threshold + ToU policies
6. `agents/agent.py` — agent wrapper with optional forecast model
7. `data/client.py` — API client with CSV caching
8. `models/base.py` — PriceModel, DeterministicModel, StochasticModel
9. `models/naive.py` — baseline models
10. `simulation/env.py` — env with historical + stochastic modes
11. `examples/basic_sim.py` — end-to-end demo

---

## Verification

1. **Unit tests**: Battery clips correctly, policy selects expected actions, env steps with synthetic data
2. **Integration**: `examples/basic_sim.py` fetches real NEM data, runs a full historical simulation
3. **Sanity**: Threshold policy ("buy low sell high") outperforms an always-idle baseline
4. **Stochastic mode**: Run env with a simple stochastic model, verify prices are sampled and sim completes
