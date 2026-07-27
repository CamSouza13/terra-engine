"""Quantitative analyses over the engine: robustness, sensor detectability,
edge cost, and economic value. Pure functions returning plain data, so they can
back a CLI, a report, or a test without pulling in a plotting stack.
"""
from __future__ import annotations

import time
import tracemalloc
from typing import Any

import numpy as np

from .core import EngineConfig, TerraEngine
from .validate import run_validation


# ---- 1. graceful degradation: how metrics hold as sensors drop ----------------

def degradation_sweep(spec, sim, drop_order: list[str] | None = None,
                      config: EngineConfig | None = None) -> list[dict]:
    """Re-score the engine as channels are removed one at a time, in ``drop_order``
    (default: the spec's channel order). Shows how gracefully the estimate
    degrades toward a single surviving sensor."""
    order = drop_order or list(spec.channels)
    rows = []
    dropped: list[str] = []
    for keep_n in range(len(order), 0, -1):
        active = [c for c in spec.channels if c not in dropped]
        sub = dict(sim)
        sub["meas"] = [{c: m.get(c) for c in active} for m in sim["meas"]]
        rep = run_validation(spec, sub, config or EngineConfig())
        rows.append({
            "n_channels": len(active),
            "active": list(active),
            "rmse": rep.hidden_rmse,
            "coverage": rep.hidden_coverage,
            "lead": {k: v.get("engine_lead") for k, v in rep.lead.items()},
        })
        if keep_n > 1:
            dropped.append(order[len(order) - keep_n])
    return rows


# ---- 2. drift detectability: min recoverable sensor-drift rate -----------------

def drift_detectability(spec, channel: str, windows_h=(12, 24, 48, 96, 168),
                        dt: float = 0.25, phi: float = 0.85, trials: int = 400,
                        power: float = 0.8, seed: int = 7) -> dict:
    """Minimum sustained drift rate on ``channel`` (units/hour) separable from the
    channel's own noise at ``power``, per observation window. Exact-GLS detector
    (the engine knows its calibrated noise), false alarms held at ~5%."""
    rng = np.random.default_rng(seed)
    ch = spec.channels[channel]
    sigma = float(ch.noise)
    st = int(ch.state) if ch.state is not None else spec.idx(spec.hidden)
    proc_step = float(np.asarray(spec.process_std, float)[st])
    proc_var = proc_step ** 2 / (1 - phi ** 2)

    def ar1(n):
        e = rng.normal(0, proc_step, n)
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = phi * x[i - 1] + e[i]
        return x

    out: dict[str, Any] = {"channel": channel, "sigma": sigma,
                           "process_wander": float(np.sqrt(proc_var)), "windows": {}}
    for T in windows_h:
        n = max(int(T / dt), 3)
        idx = np.arange(n)
        C = sigma ** 2 * np.eye(n) + proc_var * phi ** np.abs(idx[:, None] - idx[None, :])
        Cinv = np.linalg.inv(C)
        t = idx * dt
        X = np.column_stack([np.ones(n), t])
        XtCiX_inv = np.linalg.inv(X.T @ Cinv @ X)
        proj = XtCiX_inv @ X.T @ Cinv
        se = float(np.sqrt(XtCiX_inv[1, 1]))
        rate = 0.0
        step = max(sigma / (T * 4), 1e-5)
        for _ in range(400):                     # climb until we clear `power`
            hits = 0
            for _ in range(trials):
                y = rate * t + ar1(n) + rng.normal(0, sigma, n)
                if abs((proj @ y)[1]) > 1.96 * se:
                    hits += 1
            if hits / trials >= power:
                break
            rate += step
        out["windows"][int(T)] = {"min_rate_per_h": rate, "min_rate_per_day": rate * 24}
    return out


# ---- 3. edge compute budget ---------------------------------------------------

def edge_benchmark(spec, sim, steps: int = 300,
                   config: EngineConfig | None = None) -> dict:
    """Wall-clock and memory cost of one estimator cycle, to size the edge node.
    Replays real measurements/inputs from ``sim`` so the dynamics are exercised."""
    eng = TerraEngine(spec, config or EngineConfig())
    times = np.asarray(sim["t"], float)
    dt_step = float(times[1] - times[0])
    m = len(sim["meas"])
    tracemalloc.start()
    t0 = time.perf_counter()
    for k in range(steps):
        j = k % m
        eng.step(k * dt_step, dt_step, sim["meas"][j], sim["u"][j])
    dt = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {"steps": steps, "ms_per_step": 1e3 * dt / steps,
            "steps_per_s": steps / dt, "peak_mem_mb": peak / 1e6,
            "state_dim": len(spec.state_names), "channels": len(spec.channels)}


# ---- 4. alert quality: engine vs raw-gauge baseline ---------------------------

def alert_quality(domain_module, seeds: int = 8,
                  config: EngineConfig | None = None) -> dict:
    """Detection rate, false-alarm rate, and mean warning lead for the engine
    versus a raw-threshold gauge, over independent fault and no-fault runs."""
    cfg = config or EngineConfig()
    eng_det = eng_lead = base_det = targets = 0
    for s in range(seeds):                      # fault runs: measure detection
        spec, sim = domain_module.simulate(seed=s, fault=True)
        for d in run_validation(spec, sim, cfg).lead.values():
            targets += 1
            el, bl = d.get("engine_lead"), d.get("baseline_lead")
            if el is not None and el > 0:
                eng_det += 1; eng_lead += el
            if bl is not None and bl > 0:
                base_det += 1
    eng_fa = base_fa = 0
    for s in range(seeds):                      # healthy runs: should never alert
        spec, sim = domain_module.simulate(seed=100 + s, fault=False)
        for d in run_validation(spec, sim, cfg).lead.values():
            eng_fa += d.get("engine_alert") is not None
            base_fa += d.get("baseline_alarm") is not None
    denom = max(targets, 1)
    return {
        "seeds": seeds,
        "engine_detect_rate": eng_det / denom,
        "baseline_detect_rate": base_det / denom,
        "engine_mean_lead_h": (eng_lead / eng_det) if eng_det else None,
        "engine_false_alarms": int(eng_fa),
        "baseline_false_alarms": int(base_fa),
    }


# ---- 5. economic value distribution -------------------------------------------

def roi_distribution(n: int = 20000, seed: int = 0, *,
                     p_event: float = 0.35, mortality_value: float = 300_000.0,
                     feed_gain: float = 50_000.0, labor: float = 50_000.0,
                     catch_rate: float = 0.7) -> dict:
    """Monte-Carlo annual value delivered at a mid-size site: expected prevented
    mortality (event probability x value x fraction the engine catches in time)
    plus feed-conversion and labor savings, each with spread. Returns percentiles
    so the headline number is a distribution, not a point."""
    rng = np.random.default_rng(seed)
    events = rng.random(n) < p_event
    caught = rng.random(n) < catch_rate
    mort = events * caught * mortality_value * rng.uniform(0.6, 1.4, n)
    feed = np.clip(rng.normal(feed_gain, feed_gain * 0.4, n), 0, None)
    lab = np.clip(rng.normal(labor, labor * 0.3, n), 0, None)
    total = mort + feed + lab
    pct = {p: float(np.percentile(total, p)) for p in (10, 25, 50, 75, 90)}
    return {"mean": float(total.mean()), "percentiles": pct,
            "p_any_mortality_prevented": float((mort > 0).mean())}
