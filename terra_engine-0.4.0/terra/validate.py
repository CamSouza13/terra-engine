"""Validation harness: score the engine against ground truth.

Given a run with known ground truth (the simulator provides it; a real
deployment provides lab samples), this replays the engine and reports how well
it did on the three things that matter: recovering the hidden state it can't
measure, the calibration of its uncertainty, and how many hours earlier it
warns of a breach than a raw-gauge baseline.

This is the scaffold for the "validated on real data, not synthetic" milestone:
swap the synthetic ``sim`` for a logged run plus lab ground truth and the same
metrics apply.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .core import TerraEngine, EngineConfig
from .baselines import threshold_alarm, true_breach_times


@dataclass
class ValidationReport:
    domain: str
    n_steps: int
    hidden_rmse: float
    hidden_bias: float
    hidden_coverage: float           # fraction of truth inside the 95% band
    mean_nis_dof: float
    lead: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "domain": self.domain,
            "n_steps": self.n_steps,
            "hidden_rmse": self.hidden_rmse,
            "hidden_bias": self.hidden_bias,
            "hidden_coverage": self.hidden_coverage,
            "mean_nis_dof": self.mean_nis_dof,
            "lead": self.lead,
        }

    def print_report(self) -> None:
        print(f"Validation — {self.domain}  ({self.n_steps} steps)")
        print(f"  hidden-state RMSE   : {self.hidden_rmse:.3f}")
        print(f"  hidden-state bias   : {self.hidden_bias:+.3f}")
        print(f"  95% band coverage   : {self.hidden_coverage:.0%}")
        print(f"  mean NIS/dof        : {self.mean_nis_dof:.2f}")
        for name, d in self.lead.items():
            el = d["engine_lead"]
            bl = d["baseline_lead"]
            el_s = f"{el:.1f} h" if el is not None else "no warning"
            bl_s = f"{bl:.1f} h" if bl is not None else "no warning"
            print(f"  '{name}' lead time  : engine {el_s}  |  raw gauge {bl_s}")


def run_validation(spec, sim, config: EngineConfig | None = None) -> ValidationReport:
    eng = TerraEngine(spec, config or EngineConfig())
    hi = spec.idx(spec.hidden)
    truth = np.asarray(sim["truth"], float)
    times = np.asarray(sim["t"], float)
    dt = float(times[1] - times[0])
    uf = sim.get("u_forecast")

    hid_est, hid_std, hid_true, nis = [], [], [], []
    for i, t in enumerate(times):
        est = eng.step(t, dt, sim["meas"][i], sim["u"][i], u_forecast=uf)
        hid_est.append(est.hidden)
        hid_std.append(est.hidden_std)
        hid_true.append(truth[i][hi])
        if est.used_channels:
            nis.append(est.nis / max(len(est.used_channels), 1))

    hid_est = np.array(hid_est)
    hid_std = np.array(hid_std)
    hid_true = np.array(hid_true)
    rmse = float(np.sqrt(np.mean((hid_est - hid_true) ** 2)))
    bias = float(np.mean(hid_est - hid_true))
    lo = hid_est - 1.96 * hid_std
    up = hid_est + 1.96 * hid_std
    coverage = float(np.mean((hid_true >= lo) & (hid_true <= up)))

    # engine alert time per safety target (first forecast ALERT naming it)
    engine_alert = {s.name: None for s in spec.safety}
    for t_ev, level, msg in eng.events:
        if level == "ALERT":
            for s in spec.safety:
                if engine_alert[s.name] is None and s.name in msg:
                    engine_alert[s.name] = float(t_ev)

    base = threshold_alarm(spec, sim)
    breach = true_breach_times(spec, sim)
    lead = {}
    for s in spec.safety:
        tb, ea, ba = breach[s.name], engine_alert[s.name], base[s.name]
        lead[s.name] = {
            "engine_alert": ea,
            "baseline_alarm": ba,
            "true_breach": tb,
            "engine_lead": (tb - ea) if (tb is not None and ea is not None) else None,
            "baseline_lead": (tb - ba) if (tb is not None and ba is not None) else None,
        }

    return ValidationReport(
        domain=spec.name, n_steps=len(times), hidden_rmse=rmse,
        hidden_bias=bias, hidden_coverage=coverage,
        mean_nis_dof=float(np.mean(nis)) if nis else float("nan"), lead=lead)
