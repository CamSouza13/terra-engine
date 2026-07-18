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


def test_close_loop_beats_default():
    # a site whose kinetics differ from the library defaults
    from dataclasses import replace
    from terra.calibrate import calibrate_and_build
    from terra.domains import aquaculture
    from terra.domains.aquaculture import RASParams
    from terra import TerraEngine, EngineConfig

    true = replace(RASParams(), k1=1.05, k2=1.25, kLa=1.70, resp=1.05)
    _, tr = aquaculture.simulate(hours=18.0, fault=False, params=true, seed=1)
    cal_spec, _ = calibrate_and_build(
        aquaculture.build_spec(), tr["t"], tr["u"], tr["meas"],
        num_warmup=120, num_samples=120, seed=0)

    _, te = aquaculture.simulate(hours=36.0, fault=True, params=true, seed=2)

    def mean_nis(spec):
        eng = TerraEngine(spec, EngineConfig(forecast_horizon_h=12))
        dt = te["t"][1] - te["t"][0]
        vals = []
        for i, t in enumerate(te["t"]):
            e = eng.step(t, dt, te["meas"][i], te["u"][i],
                         u_forecast=te.get("u_forecast"))
            if e.used_channels:
                vals.append(e.nis / max(len(e.used_channels), 1))
        return float(np.mean(vals))

    nis_def = mean_nis(aquaculture.build_spec())   # library defaults (mis-specified)
    nis_cal = mean_nis(cal_spec)                    # tuned to the site
    assert nis_cal < nis_def, (nis_cal, nis_def)
    print(f"  held-out NIS/dof  default={nis_def:.2f}  calibrated={nis_cal:.2f}"
          f"  ({100 * (1 - nis_cal / nis_def):+.0f}%)")


def test_all_domains_calibrate():
    from terra.domains import soil, bioremediation, blss
    cases = [
        (soil, dict(hours=48.0), "k", (0.8, 2.4), 1.5),
        (bioremediation, dict(hours=18.0), "k", (0.25, 0.9), 0.5),
        (blss, dict(hours=18.0), "Pmax", (340.0, 620.0), 466.0),
    ]
    for mod, kw, key, (lo, hi), truth in cases:
        spec, sim = mod.simulate(fault=False, seed=4, **kw)
        med = fit_nuts(sim["t"], sim["u"], sim["meas"], spec,
                       num_warmup=100, num_samples=100, seed=0).medians()
        assert all(np.isfinite(v) for v in med.values()), (spec.name, med)
        assert lo < med[key] < hi, (spec.name, key, med[key])
        print(f"  {spec.name:14} {key}={med[key]:.3f} (truth {truth})")


def test_drift_inference_separates_sensor_from_process():
    # inject a slow biofouling drift on the TAN probe
    spec, sim = aquaculture.simulate(hours=20.0, fault=False, seed=7)
    dr = 0.03  # mg/L per hour
    meas = [dict(m) for m in sim["meas"]]
    for i, t in enumerate(sim["t"]):
        if "TAN" in meas[i]:
            meas[i]["TAN"] += dr * t

    naive = fit_nuts(sim["t"], sim["u"], meas, spec,
                     num_warmup=120, num_samples=120, seed=0).medians()
    aware = fit_nuts(sim["t"], sim["u"], meas, spec, fit_drift=["TAN"],
                     num_warmup=150, num_samples=150, seed=0).medians()

    assert abs(aware["drift_TAN"] - dr) < 0.02, aware          # recovered drift
    # separating drift pulls k1 back toward the truth vs ignoring it
    assert abs(aware["k1"] - 1.6) < abs(naive["k1"] - 1.6), (naive["k1"], aware["k1"])
    print(f"  drift_TAN={aware['drift_TAN']:.3f} (truth {dr})  "
          f"k1 naive={naive['k1']:.2f} -> aware={aware['k1']:.2f} (truth 1.60)")


if __name__ == "__main__":
    n = 0
    for fn in (test_recovers_kinetics, test_apply_to_updates_params,
               test_close_loop_beats_default, test_all_domains_calibrate,
               test_drift_inference_separates_sensor_from_process):
        fn()
        print(f"PASS  {fn.__name__}")
        n += 1
    print(f"\n{n}/{n} passed")
