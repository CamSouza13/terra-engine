"""Independent verification of the Terra Engine.

These checks are deliberately *not* the unit tests. They probe statistical
soundness (is the filter's own uncertainty honest?), physical soundness (is
mass conserved?), and reproducibility — the things that decide whether the
numbers can be trusted, not just whether the code runs.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from terra import TerraEngine, EngineConfig, rk4
from terra.domains import DOMAINS, aquaculture

CFG = {
    "aquaculture":    EngineConfig(forecast_horizon_h=12, forecast_dt=0.25, forecast_samples=300),
    "soil":           EngineConfig(forecast_horizon_h=48, forecast_dt=1.0,  forecast_samples=250),
    "bioremediation": EngineConfig(forecast_horizon_h=12, forecast_dt=0.25, forecast_samples=300),
    "blss":           EngineConfig(forecast_horizon_h=18, forecast_dt=0.25, forecast_samples=300),
}


def run(name, seed=None):
    mod = DOMAINS[name]
    spec, sim = (mod.simulate(seed=seed) if seed is not None else mod.simulate())
    eng = TerraEngine(spec, CFG[name])
    t = sim["t"]; dt = t[1] - t[0]; uf = sim.get("u_forecast")
    hist = []
    for i in range(len(t)):
        hist.append(eng.step(t[i], dt, sim["meas"][i], sim["u"][i], u_forecast=uf))
    return spec, sim, eng, hist


def check_calibration():
    """NEES: for a consistent filter, mean (x-truth)' P^-1 (x-truth) ~ dim(state).
    Also NIS during the healthy window should be ~ #channels."""
    print("[1] Filter calibration (NEES ~ n_state, NIS ~ n_channels)")
    for name in DOMAINS:
        spec, sim, eng, hist = run(name)
        truth = sim["truth"]; t = sim["t"]
        n = len(spec.state_names)
        nees = []
        for i, e in enumerate(hist):
            err = e.x - truth[i]
            try:
                nees.append(err @ np.linalg.solve(e.P, err))
            except np.linalg.LinAlgError:
                pass
        # healthy window = before any injected fault (first 25% of run)
        healthy = t < 0.2 * t[-1]
        nis_healthy = np.mean([hist[i].nis for i in range(len(t))
                               if healthy[i] and hist[i].nis > 0])
        print(f"    {name:14s} NEES mean {np.mean(nees):6.2f} (n={n})   "
              f"NIS healthy {nis_healthy:5.2f} (ch={len(spec.channels)})")


def check_beats_raw():
    print("[2] Fused estimate vs raw sensor (RMSE, lower is better)")
    for name in DOMAINS:
        spec, sim, eng, hist = run(name)
        truth = sim["truth"]; est = np.array([e.x for e in hist])
        improved = 0; total = 0
        for cname, ch in spec.channels.items():
            si = ch.state
            if si is None:
                continue
            raw = np.array([m.get(cname, np.nan) for m in sim["meas"]])
            # only compare identity channels (obs == state) for a fair RMSE
            if abs(ch.obs(np.eye(len(spec.state_names))[si]) - 1.0) > 1e-9:
                continue
            rr = np.sqrt(np.nanmean((raw - truth[:, si]) ** 2))
            re = np.sqrt(np.mean((est[:, si] - truth[:, si]) ** 2))
            total += 1; improved += int(re < rr)
        print(f"    {name:14s} {improved}/{total} channels improved by fusion")


def check_hidden_recovery():
    print("[3] Hidden-state recovery (|estimate - truth| at end)")
    for name in DOMAINS:
        spec, sim, eng, hist = run(name)
        hi = spec.idx(spec.hidden)
        err = abs(np.mean([e.hidden for e in hist[-6:]]) - sim["truth"][-1, hi])
        print(f"    {name:14s} hidden [{spec.hidden}] err {err:.3f}")


def check_mass_conservation():
    print("[4] Mass conservation (RAS closed: no feed, no flow, no reaction loss)")
    from terra.domains.aquaculture import RASParams, deriv
    p = RASParams(Q=0.0)
    x = np.array([2.0, 1.0, 5.0, 8.0, 1.0])
    tn0 = x[:3].sum()
    for _ in range(6000):
        x = rk4(x, 1 / 60, deriv, 0.0, p, np.array([1, 1, 1, 0, 0], bool))
    drift = abs(x[:3].sum() - tn0)
    print(f"    total N drift over 100 h: {drift:.2e} mg-N/L  "
          f"({'PASS' if drift < 1e-6 else 'FAIL'})")


def check_forecast_soundness():
    print("[5] Forecast rises with the fault (P should be ~0 before, high after)")
    for name in DOMAINS:
        spec, sim, eng, hist = run(name)
        t = sim["t"]
        r0 = list(hist[0].risks.keys())[0]
        p = np.array([e.risks[r0]["p"] for e in hist])
        before = p[t < 0.3 * t[-1]].max()
        after = p[t > 0.7 * t[-1]].max()
        print(f"    {name:14s} max P before {before:.2f}  ->  after {after:.2f}  "
              f"({'ok' if after > before else 'CHECK'})")


def check_determinism():
    print("[6] Determinism (same seed -> identical result)")
    for name in DOMAINS:
        _, _, _, h1 = run(name, seed=7)
        _, _, _, h2 = run(name, seed=7)
        d = max(abs(a.hidden - b.hidden) for a, b in zip(h1, h2))
        print(f"    {name:14s} max hidden diff across identical runs: {d:.2e}")


def check_dropout_robustness():
    print("[7] Sensor dropout: estimate ammonia with NO ammonia probe from h20")
    def avail(tt, ch):
        return not (ch == "TAN" and tt >= 20.0)
    spec, sim = aquaculture.simulate(available=avail)
    eng = TerraEngine(spec, CFG["aquaculture"]); t = sim["t"]; dt = t[1] - t[0]
    est = []
    for i in range(len(t)):
        est.append(eng.step(t[i], dt, sim["meas"][i], sim["u"][i], u_forecast=0.225).x[0])
    est = np.array(est); truth = sim["truth"][:, 0]; after = t >= 20.0
    rmse = np.sqrt(np.mean((est[after] - truth[after]) ** 2))
    print(f"    TAN RMSE with no ammonia probe (h20-48): {rmse:.2f} mg/L "
          f"(true rises to {truth[-1]:.1f})")


if __name__ == "__main__":
    print("=" * 68)
    print("TERRA ENGINE — INDEPENDENT VERIFICATION")
    print("=" * 68)
    check_calibration()
    check_beats_raw()
    check_hidden_recovery()
    check_mass_conservation()
    check_forecast_soundness()
    check_determinism()
    check_dropout_robustness()
    print("=" * 68)
