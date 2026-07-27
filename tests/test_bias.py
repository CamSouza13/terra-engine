"""Online sensor-bias / drift compensation.

Inject a growing additive drift onto one channel and check that (a) the tracker
recovers the injected bias, and (b) it lowers hidden-state RMSE versus running
with the drift uncorrected.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from terra.core import EngineConfig, TerraEngine
from terra.domains import aquaculture


def _run(sim, spec, cfg, drift_ch, rate):
    eng = TerraEngine(spec, cfg)
    hi = spec.idx(spec.hidden)
    truth = np.asarray(sim["truth"], float)
    times = np.asarray(sim["t"], float)
    dt = float(times[1] - times[0])
    err2, last = [], 0.0
    for i, t in enumerate(times):
        meas = dict(sim["meas"][i])
        if drift_ch in meas and meas[drift_ch] is not None:
            meas[drift_ch] = meas[drift_ch] + rate * t          # inject drift
        est = eng.step(t, dt, meas, sim["u"][i], u_forecast=sim.get("u_forecast"))
        err2.append((est.hidden - truth[i][hi]) ** 2)
        last = est.sensor_bias.get(drift_ch, 0.0)
    return float(np.sqrt(np.mean(err2))), last, times[-1]


def test_bias_recovery_and_rmse():
    spec, sim = aquaculture.simulate()
    ch = "TAN"
    rate = 0.01                      # mg/L per hour drift on the TAN sensor

    off = _run(sim, spec, EngineConfig(track_bias=False), ch, rate)
    spec2, sim2 = aquaculture.simulate()
    on = _run(sim2, spec2, EngineConfig(track_bias=True, bias_tau_h=12.0), ch, rate)

    rmse_off, _, _ = off
    rmse_on, bias_on, tend = on
    injected = rate * tend

    # the tracker should recover most of the injected drift
    assert bias_on > 0.4 * injected, (bias_on, injected)
    # and compensation should not be worse than ignoring the drift
    assert rmse_on <= rmse_off + 1e-9, (rmse_on, rmse_off)
    print(f"  bias tracked {bias_on:.3f} of injected {injected:.3f} "
          f"mg/L; hidden RMSE {rmse_off:.3f} -> {rmse_on:.3f}")


if __name__ == "__main__":
    test_bias_recovery_and_rmse()
    print("all bias tests passed")
