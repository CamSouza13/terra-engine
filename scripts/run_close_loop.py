"""Close the loop: logged run -> calibrated kinetics -> tuned edge estimator.

Simulates a *site* whose biofilter is weaker than the library defaults, fits
its kinetics from a fault-free training window, builds a calibrated SystemSpec,
then replays a held-out faulted run through two engines — one on library
defaults, one calibrated — and compares how well each model explains the data
(mean NIS/dof; lower is a better-specified model).

    python scripts/run_close_loop.py
    python scripts/run_close_loop.py --warmup 300 --samples 400

Requires the optional extra: pip install terra-engine[calibrate]
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _mean_nis(spec, sim):
    from terra import TerraEngine, EngineConfig
    import numpy as np
    eng = TerraEngine(spec, EngineConfig(forecast_horizon_h=12))
    dt = sim["t"][1] - sim["t"][0]
    vals = []
    for i, t in enumerate(sim["t"]):
        est = eng.step(t, dt, sim["meas"][i], sim["u"][i],
                       u_forecast=sim.get("u_forecast"))
        if est.used_channels:
            vals.append(est.nis / max(len(est.used_channels), 1))
    return eng, float(np.mean(vals))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--warmup", type=int, default=250)
    ap.add_argument("--samples", type=int, default=350)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from terra.calibrate import HAS_JAX
    if not HAS_JAX:
        print("jax/numpyro not installed. Install with: "
              "pip install terra-engine[calibrate]")
        return 1

    from terra.calibrate import calibrate_and_build
    from terra.domains import aquaculture
    from terra.domains.aquaculture import RASParams

    # a real site: weaker/colder biofilter than the library defaults
    true = replace(RASParams(), k1=1.05, k2=1.25, kLa=1.70, resp=1.05)
    truth = {"k1": 1.05, "k2": 1.25, "kLa": 1.70, "resp": 1.05}

    print("Site truth differs from library defaults "
          "(k1 1.60->1.05, k2 1.80->1.25, kLa 2.20->1.70, resp 0.90->1.05).")
    print("Fitting kinetics from a 24 h fault-free training window...\n")
    _, sim_train = aquaculture.simulate(hours=24, fault=False, params=true, seed=1)
    cal_spec, res = calibrate_and_build(
        aquaculture.build_spec(), sim_train["t"], sim_train["u"],
        sim_train["meas"], num_warmup=args.warmup, num_samples=args.samples,
        seed=args.seed)
    res.print_report(truth=truth)

    # held-out run from the same site, now with a biofilter fault
    _, sim_test = aquaculture.simulate(hours=48, fault=True, params=true, seed=2)

    eng_def, nis_def = _mean_nis(aquaculture.build_spec(), sim_test)   # defaults
    eng_cal, nis_cal = _mean_nis(cal_spec, sim_test)                   # calibrated

    print("\nHeld-out faulted run — mean NIS/dof (lower = model explains data better):")
    print(f"  library defaults : {nis_def:.2f}")
    print(f"  calibrated       : {nis_cal:.2f}   "
          f"({100 * (1 - nis_cal / nis_def):+.0f}% vs default)")

    def first_alert(engine):
        for t, level, msg in engine.events:
            if level in ("WARN", "ALERT"):
                return t, msg
        return None, None

    td, md = first_alert(eng_def)
    tc, mc = first_alert(eng_cal)
    print("\nFirst hidden/forecast alert on the faulted run:")
    print(f"  library defaults : {('%.1f h' % td) if td is not None else 'none'}")
    print(f"  calibrated       : {('%.1f h' % tc) if tc is not None else 'none'}")
    print("\nSame engine, same code — only the fitted kinetics differ. That is the "
          "one-call path from a logged run to a site-tuned edge estimator.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
