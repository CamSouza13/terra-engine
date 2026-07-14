"""Controller — the layer that decides what to do about a forecast.

The engine answers "what is the state, and where is it heading?" The controller
answers "given that, what action pulls it back?" They are deliberately separate
components: perception/prediction vs. policy/actuation.

Policy here is a forecast-driven search: when a safety target's breach
probability crosses a trigger, try the available actuators at escalating
strength and return the *least aggressive* action whose forecast brings the risk
back under a safe threshold. Each actuator maps to a real control lever the
sensor kit can drive (water exchange, fertigation, donor dosing, life-support
backup).

Autonomy stance: recommend by default; enact only if explicitly authorized. A
`Controller` with `authorized=False` will produce recommendations but never
claim to have acted. Actuation on real hardware is always operator-gated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
import numpy as np

from .core import SystemSpec, EngineConfig, mc_forecast


@dataclass
class Actuator:
    name: str                                   # human label
    make_u: Callable[[Any, float], Any]         # (base_u, level) -> control input
    levels: list[float]                         # escalating candidate strengths
    fmt: Callable[[float], str]                 # level -> readable magnitude


@dataclass
class Recommendation:
    actuator: str
    level: float
    detail: str
    p_before: float
    p_after: float
    control_u: Any
    reaches_safe: bool
    enacted: bool = False

    def message(self) -> str:
        verb = "Enacted" if self.enacted else "Recommended"
        return (f"{verb}: {self.actuator} ({self.detail}) — forecast risk "
                f"{self.p_before:.2f} → {self.p_after:.2f}"
                + ("" if self.reaches_safe else " (best available)"))


@dataclass
class ControlPolicy:
    risk: str                                   # safety-target name to control
    actuators: list[Actuator]
    trigger_p: float = 0.30
    safe_p: float = 0.10


class Controller:
    """Recommends (and, if authorized, enacts) corrective actions."""

    def __init__(self, spec: SystemSpec, cfg: EngineConfig, policy: ControlPolicy,
                 authorized: bool = False, seed: int = 101):
        self.spec = spec
        self.cfg = cfg
        self.policy = policy
        self.authorized = authorized
        self._rng = np.random.default_rng(seed)   # own RNG: never perturbs the filter

    def recommend(self, estimate, base_u) -> Recommendation | None:
        """Return a corrective action if the controlled risk is over the trigger."""
        r = estimate.risks.get(self.policy.risk)
        if not r or r["p"] < self.policy.trigger_p:
            return None
        p0 = r["p"]
        best: Recommendation | None = None
        for act in self.policy.actuators:
            for lv in act.levels:
                cu = act.make_u(base_u, lv)
                pa = mc_forecast(self.spec, self.cfg, self._rng,
                                 estimate.x, estimate.P, cu).get(
                                     self.policy.risk, {}).get("p", 1.0)
                cand = Recommendation(act.name, lv, act.fmt(lv), p0, pa, cu,
                                      reaches_safe=pa <= self.policy.safe_p,
                                      enacted=False)
                if cand.reaches_safe:
                    cand.enacted = self.authorized
                    return cand
                if best is None or pa < best.p_after:
                    best = cand
        if best is not None:
            best.enacted = self.authorized
        return best


# --------------------------------------------------------------------------- #
#  Per-domain control policies — each actuator is a real sensor-kit lever      #
# --------------------------------------------------------------------------- #

def policy_for(name: str) -> ControlPolicy:
    if name == "aquaculture":
        return ControlPolicy(
            risk="un-ionized NH3-N",
            actuators=[Actuator(
                "emergency water exchange + feed hold",
                make_u=lambda base, lv: (0.0, lv),
                levels=[3.0, 6.0, 10.0],
                fmt=lambda lv: f"{lv:.0f}x exchange, feed held")])
    if name == "soil":
        return ControlPolicy(
            risk="available nitrate",
            actuators=[Actuator(
                "fertigation",
                make_u=lambda base, lv: (base[0], base[1], lv),
                levels=[0.15, 0.30, 0.50],
                fmt=lambda lv: f"+{lv:.2f} mg-N/L/h nitrate")])
    if name == "bioremediation":
        return ControlPolicy(
            risk="electron donor",
            actuators=[Actuator(
                "restore donor dosing (backup pump)",
                make_u=lambda base, lv: lv,
                levels=[1.0, 1.5, 2.0],
                fmt=lambda lv: f"dose {lv:.1f} mg/L/h")])
    if name == "blss":
        return ControlPolicy(
            risk="cabin O2",
            actuators=[Actuator(
                "engage physico-chemical life-support backup",
                make_u=lambda base, lv: (1.0, lv),
                levels=[0.4, 0.7, 1.0],
                fmt=lambda lv: f"backup {lv * 100:.0f}%")])
    raise ValueError(f"no control policy for domain {name!r}")
