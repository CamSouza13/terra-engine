# Terra Engine

[![CI](https://github.com/CamSouza13/terra-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/CamSouza13/terra-engine/actions/workflows/ci.yml)

**One Bayesian inference engine for every closed biogeochemical loop.**

Terra measures, models, and stabilizes the mass and energy budget of a living
system in real time — from a recirculating fish farm to a field of contaminated
soil to a life-support core off-world. This repository is the engine: a
domain-agnostic state estimator that does three things a wall of raw gauges
cannot.

1. **State estimation.** It holds a probability distribution over the *true*
   state of a partly observed loop — including a hidden "health" variable it can
   never measure directly (a biofilter's efficiency, a soil's microbial
   activity, a remediation's active biomass, a crop's photosynthetic capacity).
   That hidden variable is almost always the thing that fails first.
2. **Budget closure.** It tracks the mass balance to zero and flags when the
   books stop closing (a mass residual plus a model/measurement consistency
   check). That divergence is the earliest sign of failure.
3. **Calibrated forecast.** It projects the posterior forward and reports the
   probability of breaching a safety limit within a horizon, with an expected
   time-to-breach — hours before a raw-threshold alarm would trip.

It runs on the edge (NumPy only, no cloud), and it rides on whatever sensors you
have: **channels can be enabled or disabled per timestep** based on data
availability.

4. **Control (optional).** A separate `Controller` turns a forecast into a
   corrective action — it searches the available actuators (water exchange,
   fertigation, donor dosing, life-support backup) for the least-aggressive move
   that brings the breach probability back under a safe threshold. It
   **recommends by default and enacts only when an operator has authorized
   autonomy**. Perception and action are deliberately separate components.

## One engine, four loops

A *domain* is not a fork of the engine — it is a `SystemSpec`: a set of state
variables, a process model, a set of sensor channels, and the safety limits
worth forecasting. The same `TerraEngine` runs all of them.

| Domain | Hidden state | Sensors | Forecast target | Demo fault |
|---|---|---|---|---|
| `aquaculture` | biofilter efficiency | TAN, NO₂, NO₃, DO | un-ionized NH₃, NO₂ | biofilter crash |
| `soil` | microbial activity | NH₄, NO₃, CO₂ flux, EC | available nitrate | activity stall |
| `bioremediation` | degrader activity | contaminant, ORP | electron-donor depletion | dosing-pump failure |
| `blss` | crop photosynthetic capacity | CO₂, O₂ | cabin O₂ / CO₂ | crop lighting failure |

Each demo runs a synthetic loop with an injected fault and shows the engine
inferring the hidden failure and forecasting the breach before it happens.

## Quick start

```bash
pip install numpy matplotlib          # matplotlib only needed for plots
python scripts/run_demo.py            # run all four domains + a sensor-dropout demo
python scripts/run_demo.py --domain blss --no-plot
python tests/test_engine.py           # 8 tests, no framework required

pip install -e .                      # then use the CLI:
terra domains                         # list domains
terra demo --domain aquaculture       # run a domain, print the event log
terra validate --domain soil          # score engine vs ground truth + baseline
```

### Validation

`terra validate` (or `terra.validate.run_validation`) scores the engine against
ground truth: hidden-state RMSE, 95% credible-band coverage, and how many hours
earlier it warns of a breach than a raw-gauge baseline. See
[`docs/VALIDATION.md`](docs/VALIDATION.md). The full build backlog lives in
[`ROADMAP.md`](ROADMAP.md).

### Minimal usage

```python
from terra import TerraEngine, EngineConfig
from terra.domains import aquaculture

spec, sim = aquaculture.simulate()                 # or bring your own SystemSpec
engine = TerraEngine(spec, EngineConfig(forecast_horizon_h=12))

t, dt, uf = sim["t"], sim["t"][1] - sim["t"][0], sim.get("u_forecast")
for i in range(len(t)):
    est = engine.step(t[i], dt, sim["meas"][i], sim["u"][i], u_forecast=uf)
    # est.hidden, est.hidden_std  -> the health variable you can't measure
    # est.risks["un-ionized NH3-N"]["p"], ["t_cross"]  -> calibrated forecast
    # est.budget_residual, est.nis  -> budget closure / consistency
for t_ev, level, msg in engine.events:
    print(f"{t_ev:5.1f}h  {level}  {msg}")
```

### Running on real data

Two ways in.

**A logged CSV.** Point `terra.ingest.run_csv` at a file whose columns are a
timestamp plus one column per sensor channel (names matching the domain's
channels). Blank cells or missing columns are treated as that sensor being
offline, so partial and irregular logs work as-is.

```bash
python scripts/make_sample_csv.py data/aquaculture_sample.csv   # example log
python scripts/run_csv.py --domain aquaculture \
    --csv data/aquaculture_sample.csv --time-col timestamp \
    --u-col excretion_kg_h --u-forecast 0.225
```

```python
from terra.ingest import run_csv
from terra.domains import aquaculture

res = run_csv(aquaculture.build_spec(), "my_tank_log.csv",
              time_col="timestamp", u_col="feed_kg_h", u_forecast=0.225)
for t, level, msg in res.events:
    print(f"{t:.1f}h  {level}  {msg}")
```

**A live loop.** Call `engine.step(t, dt, {channel: value}, u)` each cycle with
only the channels that reported. To adapt to your rig, edit the domain's
`build_spec()` (channels, noise, safety limits) or write a new `SystemSpec`.

For a plain-language, code-included tour of the engine and all four domains, see
[`docs/WALKTHROUGH.md`](docs/WALKTHROUGH.md).

## How it works

- **Estimator:** a compact Unscented Kalman Filter (`terra/ukf.py`), NumPy only,
  no Jacobians. `update(z, hx, R)` accepts a variable subset of channels, which
  is what makes sensor enable/disable per timestep clean.
- **Models:** each domain supplies a continuous `deriv(x, u, params)` and a
  vectorized `deriv_batch` (used by the Monte-Carlo forecast). The core
  integrates with RK4 and floors non-negative states.
- **Forecast:** samples the posterior, propagates it under an expected input,
  and estimates the probability and timing of each safety breach. Scheduled,
  cyclic drivers (feeding) are forecast under their expected average via
  `u_forecast`; step changes (a pump failure) are forecast under the current
  value — so a momentary feed peak never causes a false alarm.

## Project layout

```
terra/
  core.py                 # SystemSpec, TerraEngine, shared simulator, RK4
  ukf.py                  # Unscented Kalman Filter (variable channels)
  control.py              # controller: forecast -> recommended/enacted action
  ingest.py               # replay logged sensor CSVs through a domain
  validate.py             # score engine vs ground truth: RMSE, coverage, lead time
  baselines.py            # raw-gauge threshold baseline for benchmarking
  diagnostics.py          # pure-numpy split-R-hat / ESS
  cli.py                  # the `terra` command
  node/                   # edge-node runtime (driver, service loop, self-test)
  domains/
    aquaculture.py        # RAS nitrogen loop
    soil.py               # root-zone nitrogen / CEA
    bioremediation.py     # in-situ contaminant drawdown
    blss.py               # closed-habitat air loop
  calibrate/              # OPTIONAL offline parameter fit (jax/numpyro, off-edge)
    spec.py               # CalModel: the calibratable-model description
    models.py             # differentiable models for all four domains + registry
    nuts.py               # RK4 rollout + NUTS posterior, drift, close-the-loop
scripts/
  run_demo.py             # multi-domain demo + plots + sensor dropout
  run_csv.py              # replay a CSV log through a domain
  run_calibrate.py        # fit kinetics from a run, apply to the edge engine
  run_close_loop.py       # log -> calibrated spec -> tuned edge estimator
  make_sample_csv.py      # emit an example log with a known fault
tests/
  test_engine.py          # generic + per-domain tests
  test_ingest.py          # CSV ingestion tests
.github/workflows/ci.yml  # runs the suite on 3.10-3.12 every push/PR
```

## Offline calibration (optional)

The edge engine tracks *state* assuming the kinetics are known. The
`terra.calibrate` layer solves the complementary offline problem: given a
logged run, it infers the process *parameters* (nitrification rates, oxygen
transfer, respiration) with full posterior uncertainty using HMC/NUTS, so a
site can be calibrated against real data before the edge engine is trusted.

All four domains are calibratable (each has a differentiable `CalModel` in
`terra/calibrate/models.py`), and the fit can jointly infer per-channel sensor
drift / biofouling (`fit_drift=...`) so a lying sensor is separated from a real
process change instead of biasing the kinetics. It is a deliberately separate,
optional layer — the edge core stays numpy-only and never imports JAX.

```bash
pip install terra-engine[calibrate]     # adds jax + numpyro
python scripts/run_calibrate.py         # recover RAS kinetics from a synthetic run
```

```python
from terra.calibrate import fit_nuts
from terra.domains import aquaculture

spec, sim = aquaculture.simulate(hours=24, fault=False)
res = fit_nuts(sim["t"], sim["u"], sim["meas"], spec)
calibrated = res.apply_to(spec.params)   # RASParams with fitted kinetics
```

See [`docs/CALIBRATION.md`](docs/CALIBRATION.md) for the full walkthrough.

## Status and honesty

This is a **validated-in-simulation prototype**, not a field-proven product. The
dynamics are physically reasonable and the numbers are plausible, but every
result here is against *synthetic* ground truth. The parameters (kinetic rates,
sensor noise, safety limits) are illustrative defaults and must be calibrated
per site before any operational use. Next step: replay real logged sensor data
through a domain spec and compare the inferred hidden state against ground-truth
lab samples.

## License

MIT — see `LICENSE`.
