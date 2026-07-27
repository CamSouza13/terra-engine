"""CUSUM changepoint detection + uncertainty-gated actuation."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from terra import Controller, EngineConfig, TerraEngine
from terra.control import policy_for
from terra.domains import aquaculture


def _replay(cfg, break_at=None, break_ch="TAN", break_mag=3.0):
    spec, sim = aquaculture.simulate(fault=False)
    eng = TerraEngine(spec, cfg)
    t = sim["t"]; dt = t[1] - t[0]
    for i in range(len(t)):
        meas = dict(sim["meas"][i])
        if break_at is not None and t[i] >= break_at and meas.get(break_ch) is not None:
            meas[break_ch] = meas[break_ch] + break_mag      # abrupt regime shift
        eng.step(t[i], dt, meas, sim["u"][i], u_forecast=sim.get("u_forecast"))
    return eng


def test_cusum_fires_on_abrupt_break():
    eng = _replay(EngineConfig(cusum_threshold=5.0), break_at=20.0)
    hits = [(tt, m) for tt, lv, m in eng.events if "regime shift" in m]
    assert hits, "CUSUM should flag the abrupt break"
    assert hits[0][0] >= 20.0
    # a healthy run (no break) does not fire
    eng0 = _replay(EngineConfig(cusum_threshold=5.0))
    assert not [m for _, _, m in eng0.events if "regime shift" in m]
    # off by default even with the break present
    engoff = _replay(EngineConfig(), break_at=20.0)
    assert not [m for _, _, m in engoff.events if "regime shift" in m]
    print(f"  CUSUM fired at t={hits[0][0]:.0f}h: {hits[0][1]}")


def test_uncertainty_gate_holds_action():
    spec, sim = aquaculture.simulate(fault=True)
    cfg = EngineConfig(forecast_horizon_h=12, forecast_dt=0.25, forecast_samples=200)
    eng = TerraEngine(spec, cfg)
    pol = policy_for("aquaculture")
    pol.defer_above_std = 0.0            # never confident -> always hold
    ctl = Controller(spec, cfg, pol, authorized=True)
    t = sim["t"]; dt = t[1] - t[0]
    held = None
    for i in range(len(t)):
        est = eng.step(t[i], dt, sim["meas"][i], sim["u"][i], u_forecast=sim.get("u_forecast"))
        rec = ctl.recommend(est, sim["u"][i])
        if rec is not None:
            held = rec
            break
    assert held is not None
    assert held.enacted is False and held.confident is False, held
    assert "holding for operator" in held.message()
    print(f"  gated: {held.message()}")


if __name__ == "__main__":
    test_cusum_fires_on_abrupt_break()
    test_uncertainty_gate_holds_action()
    print("all control-safety tests passed")
