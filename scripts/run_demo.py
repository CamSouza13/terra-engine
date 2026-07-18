"""One engine, every closed loop.

Runs the same Terra Engine over four domains — aquaculture, soil/farming,
bioremediation, and closed-habitat life support — each just a SystemSpec. For
each: infers the hidden state it can't measure, closes the budget, and forecasts
the safety breach before a raw-threshold alarm would.

    python scripts/run_demo.py             # all domains, text + plots
    python scripts/run_demo.py --domain blss
    python scripts/run_demo.py --no-plot
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from terra import TerraEngine, EngineConfig            # noqa: E402
from terra.domains import DOMAINS                       # noqa: E402

# per-domain forecast settings (timescales differ across loops)
CFG = {
    "aquaculture":    EngineConfig(forecast_horizon_h=12, forecast_dt=0.25, forecast_samples=250),
    "soil":           EngineConfig(forecast_horizon_h=48, forecast_dt=1.0,  forecast_samples=200),
    "bioremediation": EngineConfig(forecast_horizon_h=12, forecast_dt=0.25, forecast_samples=250),
    "blss":           EngineConfig(forecast_horizon_h=18, forecast_dt=0.25, forecast_samples=250),
}


def run_domain(name):
    mod = DOMAINS[name]
    spec, sim = mod.simulate()
    eng = TerraEngine(spec, CFG[name])
    t = sim["t"]; dt = t[1] - t[0]
    uf = sim.get("u_forecast")
    est = []
    for i in range(len(t)):
        est.append(eng.step(t[i], dt, sim["meas"][i], sim["u"][i], u_forecast=uf))
    return spec, sim, eng, est


def report(name, spec, sim, eng, est):
    hi = spec.idx(spec.hidden)
    truth = sim["truth"]
    print("=" * 72)
    print(f"DOMAIN: {name}    ({sim['fault']})")
    print("=" * 72)
    print(f"  hidden [{spec.hidden}]: est {est[-1].hidden:.2f}  "
          f"(true {truth[-1, hi]:.2f}, |err| {abs(est[-1].hidden - truth[-1, hi]):.2f})")
    print("  EVENT LOG")
    if not eng.events:
        print("    (no events)")
    for et, lv, msg in eng.events:
        print(f"    {et:6.1f}h  {lv:5s}  {msg}")
    # early-warning delta for the first ALERT
    alert = next((e for e in eng.events if e[1] == "ALERT"), None)
    if alert:
        print(f"  first ALERT at {alert[0]:.1f}h.")
    print()


def sensor_dropout_demo():
    """Same aquaculture loop, but the ammonia probe goes offline at h20.
    The engine keeps estimating ammonia from the model + the remaining
    (NO2/NO3/DO) channels — data availability, handled."""
    from terra.domains import aquaculture

    def avail(t, ch):
        return not (ch == "TAN" and t >= 20.0)   # ammonia probe drops at h20

    spec_full, sim_full = aquaculture.simulate()
    spec, sim = aquaculture.simulate(available=avail)
    eng = TerraEngine(spec, CFG["aquaculture"])
    t = sim["t"]; dt = t[1] - t[0]
    uf = sim.get("u_forecast")
    tan_est, used_counts = [], []
    for i in range(len(t)):
        e = eng.step(t[i], dt, sim["meas"][i], sim["u"][i], u_forecast=uf)
        tan_est.append(e.x[0]); used_counts.append(len(e.used_channels))
    tan_est = np.array(tan_est)
    truth = sim["truth"][:, 0]
    after = t >= 20.0
    rmse_after = np.sqrt(np.mean((tan_est[after] - truth[after]) ** 2))
    print("=" * 72)
    print("DATA-AVAILABILITY DEMO: ammonia probe offline from h20")
    print("=" * 72)
    print(f"  channels used before h20: {max(used_counts)}   after h20: {min(used_counts)}")
    print(f"  ammonia (TAN) still estimated after probe loss, RMSE {rmse_after:.2f} mg/L")
    print(f"  final TAN est {tan_est[-1]:.2f} vs true {truth[-1]:.2f} "
          f"(no ammonia sensor for the last {t[-1]-20:.0f} h)")
    print()


def plots(results, outdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as ex:  # pragma: no cover
        print(f"[plots skipped: {ex}]")
        return
    os.makedirs(outdir, exist_ok=True)
    for name, (spec, sim, eng, est) in results.items():
        hi = spec.idx(spec.hidden)
        t = sim["t"]; truth = sim["truth"]
        est_x = np.array([e.x for e in est])
        risks = list(est[0].risks.keys())
        r0 = risks[0]
        pl = np.array([e.risks[r0]["p"] for e in est])
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        # hidden state
        ax[0].plot(t, truth[:, hi], "k-", lw=1.5, label=f"true {spec.hidden}")
        ax[0].plot(t, est_x[:, hi], "b-", lw=1.6, label="estimate")
        ax[0].axhline(spec.hidden_alert_frac * spec.hidden_baseline, ls="--",
                      color="#e67e22")
        ax[0].set_title(f"{name}: hidden {spec.hidden} (never measured)")
        ax[0].legend(); ax[0].set_xlabel("hours")
        # a key observed state
        key = 0
        cname = spec.state_names[key]
        raw = np.array([m.get(list(spec.channels)[0], np.nan) for m in sim["meas"]])
        ax[1].plot(t, raw, ".", ms=3, color="#bbb", label="raw sensor")
        ax[1].plot(t, truth[:, key], "k-", lw=1.4, label=f"true {cname}")
        ax[1].plot(t, est_x[:, key], "r-", lw=1.6, label="estimate")
        ax[1].set_title(f"{name}: {cname}"); ax[1].legend(); ax[1].set_xlabel("hours")
        # forecast risk
        ax[2].plot(t, pl, "r-", lw=1.8)
        ax[2].set_ylim(-0.02, 1.02)
        ax[2].set_title(f"{name}: forecast P({r0} breach)")
        ax[2].set_xlabel("hours")
        fig.tight_layout()
        p = os.path.join(outdir, f"terra_{name}.png")
        fig.savefig(p, dpi=120); plt.close(fig)
        print(f"[plot saved: {p}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=list(DOMAINS), default=None)
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--outdir", default=os.path.join(os.getcwd(), "outputs"))
    args = ap.parse_args()

    names = [args.domain] if args.domain else list(DOMAINS)
    results = {}
    for name in names:
        spec, sim, eng, est = run_domain(name)
        report(name, spec, sim, eng, est)
        results[name] = (spec, sim, eng, est)

    if not args.domain:
        sensor_dropout_demo()
    if not args.no_plot:
        plots(results, args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
