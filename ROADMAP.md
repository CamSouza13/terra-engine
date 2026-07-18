# Terra Engine roadmap

Where the engine is and what is left to build to get from a
validated-in-simulation prototype to an autonomous, edge-deployed, real-data
system. Checked items are in the repo today.

## Done

- [x] Domain-agnostic UKF state estimator, numpy-only, edge-ready (`terra/core.py`, `terra/ukf.py`)
- [x] Four domains: aquaculture, soil, bioremediation, BLSS
- [x] Hidden-state inference, budget closure, calibrated Monte-Carlo forecast
- [x] Control layer: recommend / enact behind an autonomy gate (`terra/control.py`)
- [x] CSV ingestion for logged sensor data (`terra/ingest.py`)
- [x] Offline Bayesian calibration, all four domains, HMC/NUTS (`terra/calibrate/`)
- [x] Close-the-loop: fit a run → calibrated `SystemSpec` → tuned edge engine
- [x] Sensor drift / biofouling inference in the fit (`fit_drift=`)
- [x] Validation harness: RMSE, interval coverage, lead-time vs a baseline (`terra/validate.py`)
- [x] Raw-gauge baseline for benchmarking (`terra/baselines.py`)
- [x] Convergence diagnostics: split-R-hat, ESS (`terra/diagnostics.py`)
- [x] Unified `terra` CLI (`terra/cli.py`)
- [x] CI across Python 3.10–3.12 plus an optional calibration job

## Near-term — the "validated on real data" milestone (critical path)

- [ ] Ingest a real logged sensor dataset (bench RAS Pi logs) end to end
- [ ] Compare inferred hidden state against lab ground truth on real data
- [ ] Validation report artifact (plots + metrics) from a real run
- [ ] Baseline comparison written up with numbers (engine vs threshold vs plain EKF)

## Inference robustness & trust

- [x] Surface R-hat / ESS from every NUTS fit (`CalibrationResult.diagnostics()` / `converged()`, printed by the CLI)
- [ ] Posterior predictive checks and a per-site identifiability report
- [x] Propagate parameter posterior into the forecast (`EngineConfig.param_draws`; `mc_forecast` marginalises over draws)
- [ ] Variational inference path (fast, edge-friendly) — claimed on the site, not built
- [ ] Simulation-based inference (SBI) for novel sensor regimes — claimed, not built
- [ ] Online joint state + parameter estimation (adaptive kinetics on the edge)

## Sensor & process-model realism

- [~] Richer drift/fouling models: exponential fouling, recalibration steps, stuck sensors (spike/outlier rejection done via `EngineConfig.outlier_sigma`)
- [ ] Irregular / multi-rate timestamps and sensor-lag alignment (fit currently assumes ~uniform dt)
- [ ] Nonlinear / known sensor response curves beyond linear scale+offset
- [ ] Temperature and pH dependence of kinetics; mechanistic hidden-failure models
- [ ] Coupled multi-loop BLSS (air + water + waste + crop together)

## Edge deployment — make "runs on a Pi" literal

- [ ] Hardware driver layer (I2C / serial) feeding `engine.step` (abstraction + `SimulatedDriver` done; real driver TODO)
- [x] Daemon / service runtime: systemd unit, watchdog (Restart), atomic state persistence, self-test (`terra/node/`, `deploy/`)
- [x] Actuator interface behind the autonomy gate with an audit log (`terra/node/actuator.py`, wired into the node loop)
- [ ] On-device performance + memory benchmarking on Pi-class hardware
- [x] Offline resilience: store-and-forward outbox that buffers offline and flushes on reconnect
- [x] Node hardware spec + BOM + build-vs-buy + Terra-branded board plan (`docs/HARDWARE.md`)
- [x] Branded Raspberry Pi node image assets + provisioning recipe (`deploy/image/`)

## Control-layer maturity

- [ ] Multi-actuator, multi-objective optimization with action costs and constraints
- [ ] Robustness testing of the controller under model uncertainty
- [ ] Safety interlocks, rate limits, human-in-the-loop approval workflow

## Product surfaces & ops

- [x] Unified CLI
- [x] PyPI packaging (dynamic version, metadata) + tagged-release workflow (`.github/workflows/release.yml`); changelog still TODO
- [ ] Live dashboard from a running node (current `simulation.html` is static)
- [x] Lint (ruff), type-check (mypy), and coverage all run in the CI quality job
- [x] SystemSpec authoring guide (`docs/SYSTEMSPEC.md`)

## Notes

Items marked "claimed on the site, not built" (VI, SBI) are advertised on the
marketing site and should either be built or softened in the copy. The
real-data milestone is the single highest-leverage next step for grants and the
raise; everything else can follow it.
