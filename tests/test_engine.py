"""Tests for the domain-agnostic Terra Engine and all four domains."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from terra import TerraEngine, EngineConfig
from terra.domains import DOMAINS, aquaculture

CFG = {
    "aquaculture":    EngineConfig(forecast_horizon_h=12, forecast_dt=0.25, forecast_samples=200),
    "soil":           EngineConfig(forecast_horizon_h=48, forecast_dt=1.0,  forecast_samples=150),
    "bioremediation": EngineConfig(forecast_horizon_h=12, forecast_dt=0.25, forecast_samples=200),
    "blss":           EngineConfig(forecast_horizon_h=18, forecast_dt=0.25, forecast_samples=200),
}

_CACHE = {}


def run(name):
    if name not in _CACHE:
        mod = DOMAINS[name]
        spec, sim = mod.simulate()
        eng = TerraEngine(spec, CFG[name])
        t = sim["t"]; dt = t[1] - t[0]
        uf = sim.get("u_forecast")
        for i in range(len(t)):
            eng.step(t[i], dt, sim["meas"][i], sim["u"][i], u_forecast=uf)
        _CACHE[name] = (spec, sim, eng)
    return _CACHE[name]


# ---- generic behaviour across every domain -----------------------------------

def test_all_domains_infer_hidden_state():
    for name in DOMAINS:
        spec, sim, eng = run(name)
        hi = spec.idx(spec.hidden)
        est = np.mean([e.hidden for e in eng.history[-6:]])
        true = sim["truth"][-1, hi]
        assert abs(est - true) < 0.15, (name, est, true)


def test_all_domains_raise_a_forecast_alert():
    for name in DOMAINS:
        spec, sim, eng = run(name)
        levels = {lv for _, lv, _ in eng.events}
        assert "ALERT" in levels, (name, eng.events)


def test_forecast_probabilities_bounded():
    for name in DOMAINS:
        spec, sim, eng = run(name)
        for e in eng.history:
            for r in e.risks.values():
                assert 0.0 <= r["p"] <= 1.0


def test_no_false_alarm_before_fault():
    # aquaculture fault starts at h24; no WARN/ALERT should precede h20
    spec, sim, eng = run("aquaculture")
    early = [ev for ev in eng.events if ev[0] < 20.0 and ev[1] in ("WARN", "ALERT")]
    assert not early, early


def test_hidden_failure_domains_flag_efficiency():
    # domains whose fault IS a hidden-state collapse should emit a WARN
    for name in ("aquaculture", "soil", "blss"):
        spec, sim, eng = run(name)
        assert any(lv == "WARN" for _, lv, _ in eng.events), (name, eng.events)


# ---- estimation quality ------------------------------------------------------

def test_estimate_beats_raw_sensor_aquaculture():
    spec, sim, eng = run("aquaculture")
    truth = sim["truth"]
    est = np.array([e.x for e in eng.history])
    total_raw = total_est = 0.0
    for cname, ch in spec.channels.items():
        si = ch.state
        raw = np.array([m.get(cname, np.nan) for m in sim["meas"]])
        # channel obs is identity for these; compare to state truth
        total_raw += np.nanmean((raw - truth[:, si]) ** 2)
        total_est += np.mean((est[:, si] - truth[:, si]) ** 2)
    assert total_est < total_raw, (total_est, total_raw)


# ---- data availability: sensors can drop out ---------------------------------

def test_channel_dropout_still_estimates():
    def avail(t, ch):
        return not (ch == "TAN" and t >= 20.0)
    spec, sim = aquaculture.simulate(available=avail)
    eng = TerraEngine(spec, CFG["aquaculture"])
    t = sim["t"]; dt = t[1] - t[0]
    uf = sim.get("u_forecast")
    used_after = []
    for i in range(len(t)):
        e = eng.step(t[i], dt, sim["meas"][i], sim["u"][i], u_forecast=uf)
        if t[i] >= 20.0:
            used_after.append(len(e.used_channels))
    # ammonia channel absent after h20 -> fewer channels, but still runs
    assert max(used_after) <= 3
    truth = sim["truth"][:, 0]
    est = np.array([e.x[0] for e in eng.history])
    after = t >= 20.0
    rmse = np.sqrt(np.mean((est[after] - truth[after]) ** 2))
    assert rmse < 1.0, rmse  # still tracks ammonia with no ammonia probe


def test_empty_measurements_is_prediction_only():
    spec, sim = aquaculture.simulate()
    eng = TerraEngine(spec, CFG["aquaculture"])
    t = sim["t"]; dt = t[1] - t[0]
    e = eng.step(t[0], dt, {}, sim["u"][0])   # no sensors at all
    assert len(e.used_channels) == 0
    assert np.all(np.isfinite(e.x))


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS  {fn.__name__}"); passed += 1
        except Exception:
            print(f"FAIL  {fn.__name__}"); traceback.print_exc()
    print(f"\n{passed}/{len(fns)} passed")
    raise SystemExit(0 if passed == len(fns) else 1)
