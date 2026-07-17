"""Offline calibration demo: recover RAS kinetics from a synthetic run.

Generates a fault-free aquaculture log (constant efficiency, so the kinetic
parameters are identifiable), fits them with NUTS, prints recovered-vs-truth,
and shows the fitted parameters applied back onto the edge engine's RASParams.

    python scripts/run_calibrate.py                # default: 18 h, 200 samples
    python scripts/run_calibrate.py --hours 24 --samples 400

Requires the optional extra: pip install terra-engine[calibrate]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hours", type=float, default=18.0)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--n-sub", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from terra.calibrate import HAS_JAX
    if not HAS_JAX:
        print("jax/numpyro not installed. Install with: "
              "pip install terra-engine[calibrate]")
        return 1

    from terra.calibrate import fit_nuts
    from terra.calibrate import jax_models as M
    from terra.domains import aquaculture

    print(f"Simulating {args.hours:.0f} h fault-free RAS run (constant efficiency)...")
    spec, sim = aquaculture.simulate(hours=args.hours, fault=False, seed=11)
    n_obs = sum(len(m) for m in sim["meas"])
    print(f"  {len(sim['t'])} timesteps, {n_obs} sensor readings across "
          f"{', '.join(M.OBS_CHANNELS)}\n")

    print(f"Running NUTS ({args.warmup} warmup + {args.samples} samples)...")
    res = fit_nuts(sim["t"], sim["u"], sim["meas"], spec,
                   num_warmup=args.warmup, num_samples=args.samples,
                   n_sub=args.n_sub, seed=args.seed)

    truth = {"k1": 1.6, "k2": 1.8, "kLa": 2.2, "resp": 0.9}
    print("\nPosterior vs ground truth:")
    res.print_report(truth=truth)

    fitted = res.apply_to(spec.params)
    print("\nApplied to edge engine RASParams:")
    print(f"  k1 {spec.params.k1:.2f} -> {fitted.k1:.2f}   "
          f"k2 {spec.params.k2:.2f} -> {fitted.k2:.2f}   "
          f"kLa {spec.params.kLa:.2f} -> {fitted.kLa:.2f}   "
          f"resp {spec.params.resp:.2f} -> {fitted.resp:.2f}")
    print("\nHand `fitted` to TerraEngine via a calibrated SystemSpec to run the "
          "edge estimator with site-specific kinetics.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
