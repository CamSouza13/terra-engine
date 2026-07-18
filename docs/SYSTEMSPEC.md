# Authoring a domain (SystemSpec)

A domain is not a fork of the engine — it is a `SystemSpec`. To add one, supply
a state vector, a process model, the sensor channels, and the safety limits
worth forecasting. The same `TerraEngine` runs it.

## Minimal checklist

1. **States** — name every state, including the one hidden health/efficiency
   variable you never measure but that fails first.
2. **Process model** — a continuous `deriv(x, u, params)` and a vectorised
   `deriv_batch(X, u, params)` (used by the Monte-Carlo forecast). Return dx/dt.
   The hidden state's derivative is `0` (it drifts via process noise).
3. **Channels** — a `Channel(obs, noise, state)` per sensor. `obs(x)` maps state
   to the measured quantity (identity for a direct probe, a formula for a proxy).
4. **Safety targets** — `SafetyTarget(name, value, limit, direction, units)` for
   each quantity to forecast a breach probability against.
5. **`params`** — a dataclass of kinetic/physical parameters.

```python
from dataclasses import dataclass
import numpy as np
from terra.core import SystemSpec, Channel, SafetyTarget

STATES = ["A", "B", "health"]

@dataclass
class MyParams:
    k: float = 1.0

def deriv(x, u, p):
    A, B, h = x
    r = h * p.k * A
    return np.array([-r, r, 0.0])

def deriv_batch(X, u, p):
    A, B, h = X[:, 0], X[:, 1], X[:, 2]
    r = h * p.k * A
    d = np.empty_like(X)
    d[:, 0] = -r; d[:, 1] = r; d[:, 2] = 0.0
    return d

def build_spec():
    return SystemSpec(
        name="mydomain", state_names=STATES,
        x0=np.array([1.0, 0.0, 1.0]),
        P0=np.diag([0.1, 0.1, 0.05]),
        process_std=np.array([0.02, 0.02, 0.02]),
        deriv=deriv, deriv_batch=deriv_batch,
        channels={"A": Channel(lambda x: x[0], 0.05, state=0)},
        params=MyParams(),
        hidden="health", hidden_baseline=1.0, hidden_alert_frac=0.85,
        safety=[SafetyTarget("B level", lambda X, p, e: X[1],
                             limit=0.9, direction=">", units="")],
        nonneg=np.array([True, True, False]),
    )
```

Register it in `terra/domains/__init__.py` and it works with the engine, the
CLI, and the validation harness immediately.

## To make it calibratable

Add a `CalModel` in `terra/calibrate/models.py`: a JAX mirror of `deriv`, the
`fit_params` with `prior`s, and an `obs_map` of `channel -> (state_index, scale,
offset)`. See [`CALIBRATION.md`](CALIBRATION.md). The inference machinery is
shared; only the differentiable model differs.

## Conventions that matter

- **`nonneg`** — a bool mask of states floored at zero each step (concentrations,
  populations). Leave energy-like states (DO, O2, ORP) unfloored.
- **Proxy channels** — fine to use (`EC`, `ORP`, `CO2_flux`); the estimator
  fuses them, but the raw-gauge baseline ignores non-identity channels.
- **Forecast inputs** — pass `u_forecast` for scheduled cyclic drivers (feeding)
  so a momentary peak does not trip a false forecast.
