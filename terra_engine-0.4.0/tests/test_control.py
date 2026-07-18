"""Tests for the controller: recommend, reduce forecast risk, honor autonomy."""
import os
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from terra import TerraEngine, EngineConfig, Controller, policy_for
from terra.domains import DOMAINS

CFG = {
    "aquaculture":    EngineConfig(forecast_horizon_h=12, forecast_dt=0.25, forecast_samples=200),
    "soil":           EngineConfig(forecast_horizon_h=48, forecast_dt=1.0,  forecast_samples=150),
    "bioremediation": EngineConfig(forecast_horizon_h=12, forecast_dt=0.25, forecast_samples=200),
    "blss":           EngineConfig(forecast_horizon_h=18, forecast_dt=0.25, forecast_samples=200),
}


def _first_recommendation(name, authorized=False):
    mod = DOMAINS[name]
    spec, sim = mod.simulate(fault=True)
    eng = TerraEngine(spec, CFG[name])
    ctl = Controller(spec, CFG[name], policy_for(name), authorized=authorized)
    t = sim["t"]; dt = t[1] - t[0]; uf = sim.get("u_forecast")
    for i in range(len(t)):
        est = eng.step(t[i], dt, sim["meas"][i], sim["u"][i], u_forecast=uf)
        rec = ctl.recommend(est, sim["u"][i])
        if rec is not None:
            return spec, sim, eng, rec, t[i]
    return spec, sim, eng, None, None


def test_controller_recommends_when_at_risk():
    for name in DOMAINS:
        spec, sim, eng, rec, t_act = _first_recommendation(name)
        assert rec is not None, name
        assert rec.p_after < rec.p_before, (name, rec)


def test_recommendation_reduces_forecast_risk_below_trigger():
    for name in DOMAINS:
        _, _, _, rec, _ = _first_recommendation(name)
        # the chosen action should get risk under the safe threshold (or be the
        # best available effort that still cuts risk substantially)
        assert rec.p_after <= 0.30, (name, rec.p_after)


def test_autonomy_flag_gates_enactment():
    _, _, _, rec_advisory, _ = _first_recommendation("bioremediation", authorized=False)
    _, _, _, rec_auto, _ = _first_recommendation("bioremediation", authorized=True)
    assert rec_advisory.enacted is False
    assert rec_auto.enacted is True
    assert "Recommended" in rec_advisory.message()
    assert "Enacted" in rec_auto.message()


def test_enacting_adequate_action_recovers_the_loop():
    # the control lever, driven at adequate strength, brings the loop back to
    # safety even as the fault fully develops (a deepening fault needs the
    # stronger end of the range — that is why the controller re-evaluates).
    for name in ("aquaculture", "soil", "bioremediation", "blss"):
        spec, sim, eng, rec, t_act = _first_recommendation(name, authorized=True)
        assert rec is not None
        act = policy_for(name).actuators[0]
        strong_u = act.make_u(sim["u"][0], act.levels[-1])   # full strength
        mod = DOMAINS[name]
        spec2, sim2 = mod.simulate(fault=True, intervene_t=t_act, intervene_u=strong_u)
        eng2 = TerraEngine(spec2, CFG[name])
        t = sim2["t"]; dt = t[1] - t[0]; base_uf = sim2.get("u_forecast")
        end_p = None
        for i in range(len(t)):
            ufi = strong_u if (name == "aquaculture" and t[i] >= t_act) else base_uf
            est = eng2.step(t[i], dt, sim2["meas"][i], sim2["u"][i], u_forecast=ufi)
            end_p = est.risks[policy_for(name).risk]["p"]
        assert end_p < 0.5, (name, end_p)


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
