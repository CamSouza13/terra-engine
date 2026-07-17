"""Offline calibration recovery test.

Skips cleanly (exit 0) when the optional `calibrate` extra is not installed, so
the numpy-only CI stays green. When jax/numpyro are present, it fits a
fault-free synthetic RAS run and checks the posterior recovers the true
kinetics within a tolerance.

    python tests/test_calibrate.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from terra.calibrate import HAS_JAX
except Exception:
    HAS_JAX = False

if not HAS_JAX:
    print("SKIP  test_calibrate (optional extra not installed: "
          "pip install terra-engine[calibrate])")
    sys.exit(0)

import numpy as np
from terra.calibrate import fit_nuts
from terra.domains import aquaculture


def test_recovers_kinetics():
    # fault-free run -> constant efficiency -> k1/k2 are identifiable
    spec, sim = aquaculture.simulate(hours=18.0, fault=False, seed=11)
    res = fit_nuts(sim["t"], sim["u"], sim["meas"], spec,
                   num_warmup=150, num_samples=150, n_sub=4, seed=0)
    med = res.medians()

    for k in ("k1", "k2", "kLa", "resp"):
        assert np.isfinite(med[k]), (k, med)

    # truth: k1=1.6, k2=1.8 ; loose bands for a short chain
    assert 1.0 < med["k1"] < 2.4, med
    assert 1.1 < med["k2"] < 2.6, med

    print(f"  recovered  k1={med['k1']:.2f} (truth 1.60)  "
          f"k2={med['k2']:.2f} (truth 1.80)  "
          f"kLa={med['kLa']:.2f} (2.20)  resp={med['resp']:.2f} (0.90)")


def test_apply_to_updates_params():
    spec, sim = aquaculture.simulate(hours=12.0, fault=False, seed=5)
    res = fit_nuts(sim["t"], sim["u"], sim["meas"], spec,
                   num_warmup=80, num_samples=80, n_sub=4, seed=1)
    updated = res.apply_to(spec.params)
    assert abs(updated.k1 - res.medians()["k1"]) < 1e-9
    assert updated.V == spec.params.V           # untouched structural constants
    print(f"  apply_to set k1={updated.k1:.2f}, left V={updated.V:.0f} fixed")


if __name__ == "__main__":
    n = 0
    for fn in (test_recovers_kinetics, test_apply_to_updates_params):
        fn()
        print(f"PASS  {fn.__name__}")
        n += 1
    print(f"\n{n}/{n} passed")
