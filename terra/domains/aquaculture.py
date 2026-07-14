"""Aquaculture (RAS) domain — the nitrogen loop of a recirculating fish farm.

Hidden state: biofilter nitrification efficiency. It is never measured; a silent
drop (clogging, cold shock, toxicity, overfeeding) is how a RAS loop kills stock
in hours. Warm-water default (tilapia-type) where ammonia toxicity is a genuine
hours-to-loss risk.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..core import SystemSpec, Channel, SafetyTarget, simulate_truth

STATES = ["TAN", "NO2", "NO3", "DO", "eff"]


@dataclass
class RASParams:
    V: float = 20000.0
    Q: float = 800.0
    k1: float = 1.6
    K1: float = 0.5
    k2: float = 1.8
    K2: float = 0.5
    DO_sat: float = 8.0
    kLa: float = 2.2
    resp: float = 0.9
    o2_per_tan: float = 3.43
    o2_per_no2: float = 1.14
    TAN_in: float = 0.0
    NO2_in: float = 0.0
    NO3_in: float = 4.0
    a_tan: float = 0.03            # kg TAN per kg feed


def tan_production(excretion_kg_h: float, p: RASParams) -> float:
    return excretion_kg_h * p.a_tan * 1.0e6 / p.V


def unionized_nh3_fraction(pH: float, temp_c: float) -> float:
    pKa = 0.09018 + 2729.92 / (273.15 + temp_c)
    return 1.0 / (1.0 + 10.0 ** (pKa - pH))


def deriv(x, u, p: RASParams):
    TAN, NO2, NO3, DO, eff = x
    TAN = max(TAN, 0.0); NO2 = max(NO2, 0.0); NO3 = max(NO3, 0.0)
    eff = min(max(eff, 0.0), 1.5)
    p_tan = tan_production(u, p)
    r1 = eff * p.k1 * TAN / (p.K1 + TAN)
    r2 = eff * p.k2 * NO2 / (p.K2 + NO2)
    dil = p.Q / p.V
    return np.array([
        p_tan - r1 - dil * (TAN - p.TAN_in),
        r1 - r2 - dil * (NO2 - p.NO2_in),
        r2 - dil * (NO3 - p.NO3_in),
        p.kLa * (p.DO_sat - DO) - p.resp - (p.o2_per_tan * r1 + p.o2_per_no2 * r2),
        0.0,
    ])


def deriv_batch(X, u, p: RASParams):
    TAN = np.clip(X[:, 0], 0, None); NO2 = np.clip(X[:, 1], 0, None)
    NO3 = np.clip(X[:, 2], 0, None); DO = X[:, 3]
    eff = np.clip(X[:, 4], 0, 1.5)
    p_tan = tan_production(u, p)
    r1 = eff * p.k1 * TAN / (p.K1 + TAN)
    r2 = eff * p.k2 * NO2 / (p.K2 + NO2)
    dil = p.Q / p.V
    d = np.empty_like(X)
    d[:, 0] = p_tan - r1 - dil * (TAN - p.TAN_in)
    d[:, 1] = r1 - r2 - dil * (NO2 - p.NO2_in)
    d[:, 2] = r2 - dil * (NO3 - p.NO3_in)
    d[:, 3] = p.kLa * (p.DO_sat - DO) - p.resp - (p.o2_per_tan * r1 + p.o2_per_no2 * r2)
    d[:, 4] = 0.0
    return d


def build_spec(pH: float = 7.5, temp_c: float = 27.0) -> SystemSpec:
    p = RASParams()
    env = {"pH": pH, "temp_c": temp_c}

    def budget(x, u, p):
        TAN = max(x[0], 0.0)
        healthy = 1.0 * p.k1 * TAN / (p.K1 + TAN)
        actual = min(max(x[4], 0), 1.5) * p.k1 * TAN / (p.K1 + TAN)
        return float(healthy - actual)

    frac = unionized_nh3_fraction(pH, temp_c)
    safety = [
        SafetyTarget("un-ionized NH3-N", lambda X, p, e: X[0] * frac,
                     limit=0.05, direction=">", units="mg/L"),
        SafetyTarget("nitrite NO2-N", lambda X, p, e: X[1],
                     limit=0.5, direction=">", units="mg/L"),
    ]
    channels = {
        "TAN": Channel(lambda x: x[0], 0.12, state=0),
        "NO2": Channel(lambda x: x[1], 0.06, state=1),
        "NO3": Channel(lambda x: x[2], 0.80, state=2),
        "DO":  Channel(lambda x: x[3], 0.15, state=3),
    }
    return SystemSpec(
        name="aquaculture", state_names=STATES,
        x0=np.array([0.35, 0.10, 6.0, 8.9, 1.0]),
        P0=np.diag([0.1, 0.05, 2.0, 0.5, 0.05]),
        process_std=np.array([0.012, 0.010, 0.20, 0.10, 0.030]),
        deriv=deriv, deriv_batch=deriv_batch, channels=channels, params=p,
        hidden="eff", hidden_baseline=1.0, hidden_alert_frac=0.85,
        env=env, safety=safety, budget=budget,
        nonneg=np.array([True, True, True, True, False]),
    )


def simulate(hours=48.0, seed=7, available=None):
    spec = build_spec()
    feed_hours = (7.0, 12.0, 17.0)
    feed_kg, window, tau = 1.8, 0.5, 3.0
    # excretion lag state carried via closure
    state = {"exc": 0.0, "t": -1.0}
    dt = 1.0 / 60.0

    def feed_rate(t):
        r = 0.0
        for day in range(int(hours // 24) + 1):
            for fh in feed_hours:
                s = day * 24 + fh
                if s <= t < s + window:
                    r += feed_kg / window
        return r

    def u_of_t(t):
        # advance a first-order excretion lag as time marches (dt-consistent)
        if t <= state["t"]:
            return state["exc"]
        state["exc"] += (dt / tau) * (feed_rate(t) - state["exc"])
        state["t"] = t
        return state["exc"]

    def hidden_of_t(t):
        a, b, final = 24.0, 28.0, 0.15
        if t <= a:
            return 1.0
        if t >= b:
            return final
        return 1.0 + (t - a) / (b - a) * (final - 1.0)

    sim = simulate_truth(spec, hours, dt, sensor_dt=0.25,
                         u_of_t=u_of_t, hidden_of_t=hidden_of_t, seed=seed,
                         available=available)
    sim["fault"] = "biofilter efficiency 1.00 -> 0.15 over h24-28"
    # feeding is scheduled/known: forecast under the expected daily-average
    # excretion so a post-feeding peak never trips a false forecast.
    sim["u_forecast"] = feed_kg * len(feed_hours) / 24.0
    return spec, sim
