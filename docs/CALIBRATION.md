# Offline Bayesian calibration

The edge engine (`TerraEngine` + UKF) is a real-time **state** estimator: it
assumes the process parameters are known and tracks the hidden state on a
Raspberry-Pi-class node with numpy only. This module solves the complementary
**parameter** problem offline.

Given a logged run from a site, `terra.calibrate` infers the process
parameters — nitrification rates, oxygen transfer, respiration — with full
posterior uncertainty using Hamiltonian Monte Carlo (NUTS). You calibrate a
site once against real data, then hand the fitted parameters to the edge engine
so it tracks state with site-specific kinetics instead of illustrative
defaults. This is the README's stated next step made concrete.

## Why it is a separate layer

- **The edge core stays numpy-only.** Nothing in `terra/core.py` or
  `terra/ukf.py` imports JAX, and `import terra` never pulls in JAX. The
  calibration stack is an optional extra that runs on a workstation, not the
  edge node.
- **State vs parameters.** The UKF estimates the fast-moving state (and the
  hidden health variable) assuming fixed kinetics. NUTS estimates the slow,
  fixed kinetics from a whole run. Different problems, different tools.
- **Differentiable model.** NUTS needs gradients, so the process model is
  mirrored in JAX (`terra/calibrate/jax_models.py`) alongside the numpy model
  the edge engine uses. The two are kept structurally identical.

## Install

```bash
pip install terra-engine[calibrate]     # adds jax + numpyro
```

## Use

```python
from terra.calibrate import fit_nuts
from terra.domains import aquaculture

spec, sim = aquaculture.simulate(hours=24, fault=False)   # or your own log
res = fit_nuts(sim["t"], sim["u"], sim["meas"], spec,
               num_warmup=300, num_samples=300)

res.print_report(truth={"k1": 1.6, "k2": 1.8, "kLa": 2.2, "resp": 0.9})
res.medians()                    # {"k1": ..., "k2": ..., "kLa": ..., "resp": ...}

calibrated = res.apply_to(spec.params)   # a RASParams with fitted kinetics
```

Feed a real log the same way `terra.ingest` does: `times` is a sequence of
hours, `u_series` is the input (excretion, optionally an exchange multiplier)
at each timestamp, and `meas` is a list of `{channel: value}` dicts where
missing channels are simply omitted. The likelihood is masked per channel, so
partial and irregular logs calibrate as-is.

## What it fits

For the aquaculture RAS domain the fit parameters and their priors are declared
in `terra/calibrate/jax_models.py`:

| Parameter | Meaning | Prior (LogNormal median) |
|---|---|---|
| `k1` | max TAN → NO₂ nitrification rate | 1.6 /h |
| `k2` | max NO₂ → NO₃ nitrification rate | 1.8 /h |
| `kLa` | oxygen transfer coefficient | 2.2 /h |
| `resp` | biological O₂ respiration | 0.9 mg/L/h |

Structural constants (tank volume, half-saturation, stoichiometry) are held
fixed. Efficiency is held constant over the calibration window so it does not
trade off against `k1`/`k2` — calibrate on fault-free stretches, then let the
edge engine track efficiency departures at run time.

## Try it

```bash
python scripts/run_calibrate.py --hours 24 --samples 400
python tests/test_calibrate.py          # recovery test (skips without the extra)
```

## All four domains

Calibration is domain-agnostic. Each domain has a `CalModel` in
`terra/calibrate/models.py` — a differentiable `deriv`, the parameters to fit
with priors, and a channel-to-state map — and `fit_nuts` resolves it from
`spec.name`. The same call calibrates any domain:

| Domain | Fit parameters |
|---|---|
| `aquaculture` | `k1`, `k2`, `kLa`, `resp` |
| `soil` | `k`, `uptake`, `base_resp` |
| `bioremediation` | `k`, `y` |
| `blss` | `Pmax`, `crew_co2`, `crew_o2` |

```python
from terra.calibrate import fit_nuts
from terra.domains import soil
spec, sim = soil.simulate(fault=False)
res = fit_nuts(sim["t"], sim["u"], sim["meas"], spec)   # model picked by spec.name
```

To add a new domain, add a `CalModel` to `models.py` (a JAX `deriv`, a
`fit_params`/`prior` block, and an `obs_map`) and register it. The rollout and
inference are unchanged.

## Sensor drift / biofouling

Real logs are corrupted by slow sensor drift (biofouling, calibration wander).
Pass `fit_drift` to jointly infer a per-channel linear drift rate
(`obs = signal + rate * t`) alongside the process parameters, so the fit
separates a real process change from a lying sensor instead of baking the drift
into the kinetics.

```python
res = fit_nuts(sim["t"], sim["u"], meas, spec, fit_drift=["TAN"])   # or fit_drift=True
res.drift()          # {"TAN": rate_per_hour}
res.medians()["k1"]  # kinetics, corrected for the drift
```

This is the differentiating capability: Bayesian fusion that stays honest under
drifting, fouling sensors rather than threshold-alerting on raw gauges.
