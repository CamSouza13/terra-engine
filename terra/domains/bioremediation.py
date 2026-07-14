"""Bioremediation domain — a living cleanup drawing a contaminant down in situ.

The biology is rarely the hard part; proving, continuously, that the system is
actually closing is. Here the failure is an electron-donor supply loss (dosing
pump). The engine infers that the degrader activity is fine, watches the donor
fall, and forecasts the depletion that will stall the drawdown — before the
contaminant rebounds. ORP is a proxy channel for the donor/redox state.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..core import SystemSpec, Channel, SafetyTarget, simulate_truth

STATES = ["C", "D", "act"]      # contaminant, electron donor, degrader activity


@dataclass
class BioParams:
    k: float = 0.5
    Kc: float = 20.0
    Kd: float = 1.0
    y: float = 3.0              # donor consumed per unit contaminant degraded
    infl: float = 0.005        # slow back-diffusion from source zone
    C_src: float = 60.0


def _rate(C, D, act, p):
    return act * p.k * C / (p.Kc + C) * D / (p.Kd + D)


def deriv(x, u, p: BioParams):
    C, D, act = x
    C = max(C, 0.0); D = max(D, 0.0); act = max(act, 0.0)
    dose = u
    rate = _rate(C, D, act, p)
    return np.array([
        -rate + p.infl * (p.C_src - C),
        dose - p.y * rate,
        0.0,
    ])


def deriv_batch(X, u, p: BioParams):
    C = np.clip(X[:, 0], 0, None); D = np.clip(X[:, 1], 0, None)
    act = np.clip(X[:, 2], 0, None)
    dose = u
    rate = _rate(C, D, act, p)
    d = np.empty_like(X)
    d[:, 0] = -rate + p.infl * (p.C_src - C)
    d[:, 1] = dose - p.y * rate
    d[:, 2] = 0.0
    return d


def build_spec() -> SystemSpec:
    p = BioParams()

    def budget(x, u, p):
        # expected drawdown at full donor vs actual (donor-limited) drawdown
        healthy = _rate(max(x[0], 0), 5.0, max(x[2], 0), p)
        actual = _rate(max(x[0], 0), max(x[1], 0), max(x[2], 0), p)
        return float(healthy - actual)

    safety = [
        SafetyTarget("electron donor", lambda X, p, e: X[1],
                     limit=0.5, direction="<", units="mg/L"),
    ]
    channels = {
        "contaminant": Channel(lambda x: x[0], 2.0, state=0),
        "ORP": Channel(lambda x: -50.0 - 30.0 * x[1], 15.0, state=1),  # mV proxy
    }
    return SystemSpec(
        name="bioremediation", state_names=STATES,
        x0=np.array([50.0, 5.0, 1.0]),
        P0=np.diag([9.0, 1.0, 0.05]),
        process_std=np.array([0.15, 0.05, 0.02]),
        deriv=deriv, deriv_batch=deriv_batch, channels=channels, params=p,
        hidden="act", hidden_baseline=1.0, hidden_alert_frac=0.80,
        env={}, safety=safety, budget=budget,
        nonneg=np.array([True, True, False]),
    )


def simulate(hours=48.0, seed=5, available=None, fault=True):
    spec = build_spec()
    dt = 1.0 / 60.0

    def u_of_t(t):
        if not fault:
            return 1.0                    # dosing holds steady
        return 1.0 if t < 20.0 else 0.0   # dosing pump fails at h20

    def hidden_of_t(t):
        return 1.0                        # biology stays healthy; supply fails

    sim = simulate_truth(spec, hours, dt, sensor_dt=0.25,
                         u_of_t=u_of_t, hidden_of_t=hidden_of_t, seed=seed,
                         available=available)
    sim["fault"] = "electron-donor dosing fails at h20 (biology healthy)"
    return spec, sim
