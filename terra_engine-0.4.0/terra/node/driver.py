"""Sensor drivers for the edge node.

The node runtime reads from a ``SensorDriver`` and does not care whether the
readings come from real probes or a replayed simulation. That abstraction is
what lets the whole node stack (loop, state persistence, event log, self-test)
be built and tested today with zero hardware, then run unchanged on a real board
by dropping in an I2C/analog driver later.

A driver yields ``(t, dt, measurements, u)`` per cycle:
  - ``t``  : timestamp in hours
  - ``dt`` : hours since the previous cycle
  - ``measurements`` : ``{channel: value}`` (omit or None for offline channels)
  - ``u``  : the known process input this cycle
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class SensorDriver(ABC):
    """Base class for anything that feeds the node cycles."""

    #: forecast input to assume when projecting forward (optional)
    u_forecast = None

    @abstractmethod
    def steps(self):
        """Yield ``(t, dt, measurements, u)`` tuples, one per cycle."""
        raise NotImplementedError


class SimulatedDriver(SensorDriver):
    """Replays a domain ``simulate()`` output as a live sensor stream.

    This is the hardware-free driver used for development, the self-test, and
    CI. It faithfully exercises the same code path a real driver will.
    """

    def __init__(self, spec, sim):
        self.spec = spec
        self._t = list(sim["t"])
        self._meas = sim["meas"]
        self._u = sim["u"]
        self.u_forecast = sim.get("u_forecast")

    def __len__(self):
        return len(self._t)

    def steps(self):
        t0 = self._t
        base_dt = (t0[1] - t0[0]) if len(t0) > 1 else 1.0
        for i in range(len(t0)):
            dt = base_dt if i == 0 else (t0[i] - t0[i - 1])
            yield t0[i], dt, self._meas[i], self._u[i]
