# The data flywheel

Every deployment makes the next one better. This is the end-to-end loop that
turns logged site data into a sharper engine, and the commands that run it.

## The loop

1. **Per-site calibration.** Each site logs its sensor stream. `calibrate_and_build`
   fits that site's kinetics with NUTS, starting from the generic prior, and
   reports convergence (split-R-hat / ESS). Output: a per-site posterior
   (mean + std per parameter).

2. **Fleet pooling.** `terra.fleet.pool_posteriors` pools the per-site posteriors
   into a shared prior using a random-effects model. The pooled population mean
   tightens as sites join (~tau/sqrt(N)); the prior a *new* site starts from
   stays as wide as the fleet genuinely is, because between-site spread (tau) is
   estimated, not assumed.

3. **Backtest.** `terra.ingest.backtest_csv` replays a logged CSV through the
   engine and, if lab ground truth is present, scores hidden-state RMSE,
   coverage, and warning lead — generic default vs site-calibrated.

## Commands

```bash
# whole loop on stand-in field logs (heterogeneous simulated sites)
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python scripts/run_flywheel.py --sites 6

# pool per-site posteriors ({param: [mean, std]} JSONs) into a shared prior
terra fleet-pool --sites ./site_posteriors --out shared_prior.json

# score the engine on a real logged CSV (with a hidden-state truth column)
terra backtest farm_log.csv --domain aquaculture --truth eff_lab --u-col feed
```

## What the demo shows

Running `run_flywheel.py` on six heterogeneous stand-in sites: each site's
kinetics are recovered within roughly 3–10%, every fit converges, the pooled
population mean tightens with N, and on held-out fault logs the site-calibrated
model improves uncertainty calibration (coverage ~85% -> ~91%) at comparable
RMSE. On real pilot logs the same three commands run unchanged; only the numbers
firm up.
