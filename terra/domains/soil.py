"""Soil / controlled-environment farming domain — root-zone nitrogen budget.

Hidden state: microbial nitrification activity. A silent stall (cold, compaction,
waterlogging, toxicity) lets ammonium build while available nitrate falls, so the
crop starves of nitrate days before a quarterly lab test would reveal it. The CO2
respiration channel is a proxy that exposes the activity drop even without N probes.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..core import SystemSpec, Channel, SafetyTarget, simulate_truth

STATES = ["NH4", "NO3", "resp", "act"]


@dataclass
class SoilParams:
    k: float = 1.5          # max nitrification, mg-N/L/h
    K: float = 1.0
    uptake: float = 0.30    # crop nitrate uptake, mg-N/L/h
    base_resp: float = 2.0  # healthy microbial CO2 flux
    krelax: float = 0.4     # how fast respiration tracks activity


def _unpack(u):
    """u = (mineralization, drainage[, nitrate_dose]). Dose is the control lever."""
    min_in, drain = u[0], u[1]
    no3_dose = u[2] if len(u) > 2 else 0.0
    return min_in, drain, no3_dose


def deriv(x, u, p: SoilParams):
    NH4, NO3, resp, act = x
    NH4 = max(NH4, 0.0); NO3 = max(NO3, 0.0)
    act = max(act, 0.0)
    min_in, drain, no3_dose = _unpack(u)
    r = act * p.k * NH4 / (p.K + NH4)
    return np.array([
        min_in - r,
        r - drain * NO3 - p.uptake + no3_dose,   # fertigation adds nitrate
        p.krelax * (act * p.base_resp - resp),
        0.0,
    ])


def deriv_batch(X, u, p: SoilParams):
    NH4 = np.clip(X[:, 0], 0, None); NO3 = np.clip(X[:, 1], 0, None)
    resp = X[:, 2]; act = np.clip(X[:, 3], 0, None)
    min_in, drain, no3_dose = _unpack(u)
    r = act * p.k * NH4 / (p.K + NH4)
    d = np.empty_like(X)
    d[:, 0] = min_in - r
    d[:, 1] = r - drain * NO3 - p.uptake + no3_dose
    d[:, 2] = p.krelax * (act * p.base_resp - resp)
    d[:, 3] = 0.0
    return d


def build_spec() -> SystemSpec:
    p = SoilParams()

    def budget(x, u, p):
        NH4 = max(x[0], 0.0)
        healthy = 1.0 * p.k * NH4 / (p.K + NH4)
        actual = max(x[3], 0.0) * p.k * NH4 / (p.K + NH4)
        return float(healthy - actual)

    safety = [
        SafetyTarget("available nitrate", lambda X, p, e: X[1],
                     limit=4.0, direction="<", units="mg-N/L"),
    ]
    channels = {
        "NH4": Channel(lambda x: x[0], 1.0, state=0),
        "NO3": Channel(lambda x: x[1], 1.5, state=1),
        "CO2_flux": Channel(lambda x: x[2], 0.05, state=2),
        "EC": Channel(lambda x: 0.1 * x[1], 0.05, state=1),  # proxy for nitrate
    }
    return SystemSpec(
        name="soil", state_names=STATES,
        x0=np.array([0.5, 10.0, 2.0, 1.0]),
        P0=np.diag([0.5, 4.0, 0.2, 0.05]),
        process_std=np.array([0.05, 0.10, 0.03, 0.02]),
        deriv=deriv, deriv_batch=deriv_batch, channels=channels, params=p,
        hidden="act", hidden_baseline=1.0, hidden_alert_frac=0.85,
        env={}, safety=safety, budget=budget,
        nonneg=np.array([True, True, True, False]),
    )


def simulate(hours=120.0, seed=11, available=None, fault=True,
             intervene_t=None, intervene_u=None):
    spec = build_spec()
    dt = 1.0 / 60.0

    def u_of_t(t):
        if intervene_t is not None and t >= intervene_t:
            return intervene_u          # controller action: fertigate nitrate
        return (0.5, 0.02, 0.0)  # mineralization mg/L/h, drainage /h, no dose

    def hidden_of_t(t):
        if not fault:
            return 1.0
        a, b, final = 24.0, 40.0, 0.25
        if t <= a:
            return 1.0
        if t >= b:
            return final
        return 1.0 + (t - a) / (b - a) * (final - 1.0)

    sim = simulate_truth(spec, hours, dt, sensor_dt=1.0,
                         u_of_t=u_of_t, hidden_of_t=hidden_of_t, seed=seed,
                         available=available)
    sim["fault"] = "microbial activity 1.00 -> 0.25 over h24-40"
    return spec, sim
