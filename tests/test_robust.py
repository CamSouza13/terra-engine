"""Robustness features: outlier rejection and parameter-marginalized forecast."""
import os
import sys
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from terra import TerraEngine, EngineConfig
from terra.domains import aquaculture
from terra.domains.aquaculture import RASParams


def _hidden_rmse(cfg, sim, spec):
    eng = TerraEngine(spec, cfg)
    hi = spec.idx(spec.hidden)
    truth = sim["truth"]
    t = sim["t"]
    dt = float(t[1] - t[0])
    err = []
    for i, tt in enumerate(t):
        est = eng.step(tt, dt, sim["meas"][i], sim["u"][i],
                       u_forecast=sim.get("u_forecast"))
        err.append((est.hidden - truth[i][hi]) ** 2)
    return float(np.sqrt(np.mean(err)))


def test_outlier_rejection_survives_spikes():
    spec, sim = aquaculture.simulate(hours=48.0, fault=True, seed=7)
    meas = [dict(m) for m in sim["meas"]]
    for i in range(10, len(meas), 25):          # periodic +5 mg/L TAN spikes
        if "TAN" in meas[i]:
            meas[i]["TAN"] += 5.0
    sim2 = dict(sim); sim2["meas"] = meas

    base = EngineConfig(forecast_horizon_h=12, forecast_samples=120)
    gated = EngineConfig(forecast_horizon_h=12, forecast_samples=120, outlier_sigma=5.0)
    e_none = _hidden_rmse(base, sim2, spec)
    e_gate = _hidden_rmse(gated, sim2, spec)
    assert e_gate <= e_none, (e_gate, e_none)
    print(f"  hidden RMSE  no-gate={e_none:.3f}  gate={e_gate:.3f}")


def test_forecast_marginalises_over_params():
    spec, sim = aquaculture.simulate(hours=36.0, fault=True, seed=7)
    draws = [replace(RASParams(), k1=k) for k in (1.2, 1.6, 2.0)]
    cfg = EngineConfig(forecast_horizon_h=12, forecast_samples=150, param_draws=draws)
    eng = TerraEngine(spec, cfg)
    t = sim["t"]; dt = float(t[1] - t[0]); last = None
    for i, tt in enumerate(t):
        last = eng.step(tt, dt, sim["meas"][i], sim["u"][i],
                        u_forecast=sim.get("u_forecast"))
    r = last.risks.get("un-ionized NH3-N")
    assert r is not None
    assert 0.0 <= r["p"] <= 1.0, r
    print(f"  parameter-marginalised NH3 breach p={r['p']:.2f}")


if __name__ == "__main__":
    n = 0
    for fn in (test_outlier_rejection_survives_spikes,
               test_forecast_marginalises_over_params):
        fn()
        print(f"PASS  {fn.__name__}")
        n += 1
    print(f"\n{n}/{n} passed")
