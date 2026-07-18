"""Actuator drivers — the hardware-facing side of the controller.

The controller decides *what* to do (a `Recommendation`); an `ActuatorDriver`
carries it out on a real lever (a relay, a dosing pump, a valve). Perception and
actuation stay separate, and — like the controller — enactment is gated: the
node only calls `apply` when the recommendation is `enacted` (operator-authorized
autonomy).

`SimulatedActuatorDriver` records what it would have done, so the whole control
loop is testable without hardware. A real driver implements `apply` against I2C
relays / EZO-PMP pumps.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class ActuatorDriver(ABC):
    @abstractmethod
    def apply(self, recommendation) -> bool:
        """Carry out a recommendation. Return True on success."""
        raise NotImplementedError


class SimulatedActuatorDriver(ActuatorDriver):
    """No-op driver that logs enacted actions for inspection and audit."""

    def __init__(self):
        self.log: list = []

    def apply(self, recommendation) -> bool:
        self.log.append({
            "actuator": recommendation.actuator,
            "level": recommendation.level,
            "detail": recommendation.detail,
            "p_before": recommendation.p_before,
            "p_after": recommendation.p_after,
        })
        return True
