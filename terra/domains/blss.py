"""Bioregenerative life support (BLSS) domain — a closed habitat air loop.

The hardest version of the problem, and the reason the engine exists. Crew and
plants regenerate each other's air with no resupply. Hidden state: crop
photosynthetic capacity. A silent drop (lighting fault, disease) lets CO2 climb
and O2 fall. The engine forecasts the crew-safety breach hours ahead.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..core import SystemSpec, Channel, SafetyTarget, simulate_truth

STATES = ["CO2", "O2", "act"]      # CO2 ppm, O2 %, crop photosynthetic capacity


@dataclass
class BLSSParams:
    crew_co2: float = 300.0    # ppm/h produced by crew
    crew_o2: float = 0.30      # %/h consumed by crew
    Pmax: float = 466.0        # max photosynthetic CO2 uptake, ppm/h
    Kco2: float = 500.0
    k_o2: float = 0.001        # %O2 produced per ppm CO2 fixed
    leak: float = 0.02         # cabin exchange /h
    CO2_amb: float = 400.0
    O2_amb: float = 20.9


def deriv(x, u, p: BLSSParams):
    CO2, O2, act = x
    CO2 = max(CO2, 0.0); act = max(act, 0.0)
    light = u
    photo = act * p.Pmax * light * CO2 / (p.Kco2 + CO2)
    return np.array([
        p.crew_co2 - photo + p.leak * (p.CO2_amb - CO2),
        -p.crew_o2 + p.k_o2 * photo + p.leak * (p.O2_amb - O2),
        0.0,
    ])


def deriv_batch(X, u, p: BLSSParams):
    CO2 = np.clip(X[:, 0], 0, None); O2 = X[:, 1]; act = np.clip(X[:, 2], 0, None)
    light = u
    photo = act * p.Pmax * light * CO2 / (p.Kco2 + CO2)
    d = np.empty_like(X)
    d[:, 0] = p.crew_co2 - photo + p.leak * (p.CO2_amb - CO2)
    d[:, 1] = -p.crew_o2 + p.k_o2 * photo + p.leak * (p.O2_amb - O2)
    d[:, 2] = 0.0
    return d


def build_spec() -> SystemSpec:
    p = BLSSParams()
    safety = [
        SafetyTarget("cabin O2", lambda X, p, e: X[1],
                     limit=19.5, direction="<", units="%"),
        SafetyTarget("cabin CO2", lambda X, p, e: X[0],
                     limit=5000.0, direction=">", units="ppm"),
    ]
    channels = {
        "CO2": Channel(lambda x: x[0], 40.0, state=0),
        "O2": Channel(lambda x: x[1], 0.10, state=1),
    }
    return SystemSpec(
        name="blss", state_names=STATES,
        x0=np.array([900.0, 20.9, 1.0]),
        P0=np.diag([2500.0, 0.25, 0.05]),
        process_std=np.array([20.0, 0.05, 0.02]),
        deriv=deriv, deriv_batch=deriv_batch, channels=channels, params=p,
        hidden="act", hidden_baseline=1.0, hidden_alert_frac=0.85,
        env={}, safety=safety, budget=None,
        nonneg=np.array([True, False, False]),
    )


def simulate(hours=48.0, seed=3, available=None):
    spec = build_spec()
    dt = 1.0 / 60.0

    def u_of_t(t):
        return 1.0                        # continuous LED light

    def hidden_of_t(t):
        a, b, final = 12.0, 16.0, 0.30
        if t <= a:
            return 1.0
        if t >= b:
            return final
        return 1.0 + (t - a) / (b - a) * (final - 1.0)

    sim = simulate_truth(spec, hours, dt, sensor_dt=0.25,
                         u_of_t=u_of_t, hidden_of_t=hidden_of_t, seed=seed,
                         available=available)
    sim["fault"] = "crop photosynthetic capacity 1.00 -> 0.30 over h12-16"
    return spec, sim
