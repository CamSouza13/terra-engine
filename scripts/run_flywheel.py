"""End-to-end data flywheel on a fleet of (stand-in) field logs.

Until a pilot provides real sensor logs, each 'site' here is a held-out
simulation with its own perturbed kinetics -- a heterogeneous farm the generic
default priors do NOT match. The pipeline is exactly what runs on real data:

  1. per-site calibration: fit each site's kinetics from its log (NUTS),
     starting from the generic prior, and record convergence;
  2. fleet pooling: pool the per-site posteriors into a shared prior that a new
     site starts from, and watch it tighten as sites join;
  3. backtest uplift: on a held-out log from one site, score the engine with the
     generic default vs the site-calibrated model (RMSE, coverage, warning lead).

Run:  JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python scripts/run_flywheel.py
"""
from __future__ import annotations

import dataclasses
import json
import os
import tempfile

import numpy as np

from terra.calibrate import calibrate_and_build
from terra.core import EngineConfig
from terra.domains import aquaculture
from terra.fleet import pool_posteriors, prior_tightening
from terra.validate import run_validation

FIT = ("k1", "k2", "kLa", "resp")
rng = np.random.default_rng(11)


def site_params(base, factors):
    return dataclasses.replace(base, **{k: getattr(base, k) * f
                                        for k, f in zip(FIT, factors)})


def main(n_sites=6, warmup=150, samples=150):
    base = aquaculture.build_spec().params
    outdir = tempfile.mkdtemp(prefix="flywheel_")
    print(f"writing per-site posteriors to {outdir}\n")

    site_truth, site_post, site_cspec = [], [], []
    print("1) PER-SITE CALIBRATION (fit each site's kinetics from its log)")
    for s in range(n_sites):
        factors = rng.uniform(0.75, 1.25, len(FIT))     # farm heterogeneity
        truth = site_params(base, factors)
        spec, sim = aquaculture.simulate(hours=18, fault=False, seed=s, params=truth)
        cspec, res = calibrate_and_build(spec, sim["t"], sim["u"], sim["meas"],
                                         num_warmup=warmup, num_samples=samples)
        summ = res.summary()
        post = {k: (summ[k]["mean"], summ[k]["std"]) for k in FIT}
        site_truth.append(truth); site_post.append(post); site_cspec.append(cspec)
        with open(os.path.join(outdir, f"site{s}.json"), "w") as f:
            json.dump({k: [v[0], v[1]] for k, v in post.items()}, f)
        conv = "converged" if res.converged() else "NOT converged"
        recov = np.mean([abs(summ[k]["mean"] - getattr(truth, k)) / getattr(truth, k)
                         for k in FIT])
        print(f"  site {s}: {conv}; mean kinetic recovery error {recov*100:.1f}%")

    print("\n2) FLEET POOLING (population mean tightens; new-site prior stays as")
    print("   wide as the fleet genuinely is)")
    pooled = pool_posteriors(site_post)
    for k in FIT:
        curve = prior_tightening(site_post, k)   # index 1 = 2 sites (tau estimable)
        print(f"  {k:5s} pooled-mean std {curve[1]['std']:.3g} (2 sites) -> "
              f"{curve[-1]['std']:.3g} ({len(curve)} sites); "
              f"between-site tau {pooled[k]['tau']:.3g}")
    with open(os.path.join(outdir, "shared_prior.json"), "w") as f:
        json.dump({k: [d["mean"], d["prior_std"]] for k, d in pooled.items()}, f, indent=2)

    print("\n3) BACKTEST UPLIFT (generic vs site-calibrated, mean of 3 held-out")
    print("   fault logs from site 0)")
    dr = dc = cr = cc = 0.0
    for seed in (99, 100, 101):
        _, held = aquaculture.simulate(hours=48, fault=True, seed=seed, params=site_truth[0])
        d = run_validation(aquaculture.build_spec(), held, EngineConfig())
        c = run_validation(site_cspec[0], held, EngineConfig())
        dr += d.hidden_rmse; cr += c.hidden_rmse
        dc += d.hidden_coverage; cc += c.hidden_coverage
    print(f"  default   : RMSE {dr/3:.3f}  coverage {dc/3:.0%}")
    print(f"  calibrated: RMSE {cr/3:.3f}  coverage {cc/3:.0%}")
    print(f"\nartifacts in {outdir} (site*.json + shared_prior.json)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="end-to-end data flywheel")
    ap.add_argument("--sites", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=150)
    ap.add_argument("--samples", type=int, default=150)
    a = ap.parse_args()
    main(n_sites=max(a.sites, 2), warmup=a.warmup, samples=a.samples)
