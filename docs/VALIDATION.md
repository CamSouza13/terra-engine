# Validating the engine

The engine's claim is that modelling the coupled loop recovers a hidden failure
and warns earlier than watching raw gauges. `terra.validate` scores exactly
that against ground truth.

```python
from terra.validate import run_validation
from terra.domains import aquaculture

spec, sim = aquaculture.simulate(fault=True)
run_validation(spec, sim).print_report()
```

```
Validation — aquaculture  (193 steps)
  hidden-state RMSE   : 0.278
  hidden-state bias   : +0.05
  95% band coverage   : 86%
  mean NIS/dof        : 1.10
  'un-ionized NH3-N' lead time  : engine 9.3 h  |  raw gauge 0.5 h
```

Or from the CLI:

```bash
terra validate --domain soil
```

## What it measures

- **Hidden-state RMSE / bias** — how well the engine recovers the health
  variable it never measures (biofilter efficiency, microbial activity, degrader
  activity, crop capacity), against ground truth.
- **95% band coverage** — the fraction of truth that falls inside the engine's
  reported credible band. Near 95% means the uncertainty is honest; much lower
  means it is overconfident.
- **mean NIS/dof** — normalised innovation squared per channel; ~1 means the
  model explains the measurements, higher means it does not.
- **Lead time** — how many hours before the true breach the engine raises a
  forecast alert, next to a raw-gauge threshold baseline (`terra.baselines`).

## From synthetic to real

The harness takes any run object with `t`, `meas`, `u`, and a `truth` array.
Today `truth` comes from the simulator. For a real deployment, assemble the same
structure from a logged run plus interpolated lab samples as ground truth, and
the identical metrics apply. That swap is the "validated on real data, not
synthetic" milestone in [`ROADMAP.md`](../ROADMAP.md).

## Convergence diagnostics

For the Bayesian calibration layer, `terra.diagnostics` provides pure-numpy
`split_rhat` and `ess` over grouped MCMC samples:

```python
from terra.diagnostics import report
report(mcmc.get_samples(group_by_chain=True))   # {param: {"rhat": ..., "ess": ...}}
```

R-hat near 1.0 and a healthy ESS mean the fit converged; R-hat above ~1.1 means
it did not and the posterior should not be trusted.
