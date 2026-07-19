"""Real sensor drivers for the edge node.

Where ``SimulatedDriver`` replays a run, ``HardwareDriver`` reads live probes on a
cadence and yields the same ``(t, dt, meas, u)`` cycles the runtime expects — so
the estimator, state persistence, and reporting code run unchanged on real
hardware.

Probes implement a one-method interface (``read() -> float``):

* ``AtlasEZO``     — Atlas Scientific EZO smart sensors over I2C (pH, DO, EC, ORP,
                     RTD). Real protocol; needs ``smbus2`` and a real bus.
* ``LinearProbe``  — any analog probe read through your ADC: supply a callable that
                     returns the raw signal (volts/counts) and a linear calibration
                     ``value = scale * raw + offset``. Keeps us ADC-agnostic.
* ``ConstantProbe``— a fixed value, for bring-up and tests.

Nothing here imports hardware libraries at module load, so the module is safe to
import (and unit-test) on any machine; the I2C dependency is resolved lazily when
an ``AtlasEZO`` is actually read.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Callable

from .driver import SensorDriver


class Probe(ABC):
    @abstractmethod
    def read(self) -> float:
        """Return the current calibrated reading, or raise on failure."""
        raise NotImplementedError


class ConstantProbe(Probe):
    def __init__(self, value: float):
        self.value = float(value)

    def read(self) -> float:
        return self.value


class LinearProbe(Probe):
    """An analog probe read through any ADC.

    ``raw`` returns the ADC signal (volts or counts); the reading is
    ``scale * raw + offset``. Determine scale/offset from a two-point calibration.
    """

    def __init__(self, raw: Callable[[], float], scale: float = 1.0, offset: float = 0.0):
        self._raw = raw
        self.scale = float(scale)
        self.offset = float(offset)

    def read(self) -> float:
        return self.scale * float(self._raw()) + self.offset


class AtlasEZO(Probe):
    """Atlas Scientific EZO sensor over I2C.

    Protocol: write an ASCII command (``"R"`` to take a reading), wait for the
    processing delay, then read back the response bytes and parse the float.
    Addresses default per product (pH 0x63, EC 0x64, DO 0x61, ORP 0x62, RTD 0x66).
    """

    def __init__(self, address: int, bus: int = 1, read_delay_s: float = 0.9):
        self.address = address
        self.bus_num = bus
        self.read_delay = read_delay_s
        self._bus = None

    def _open(self):
        if self._bus is None:
            try:
                import smbus2  # lazy: only needed on real hardware
            except Exception as e:  # pragma: no cover - env without smbus2
                raise RuntimeError("AtlasEZO needs smbus2 on a real I2C bus: "
                                   "pip install smbus2") from e
            self._bus = smbus2.SMBus(self.bus_num)
        return self._bus

    def read(self) -> float:
        import smbus2  # pragma: no cover - hardware only
        bus = self._open()
        bus.i2c_wr = smbus2.i2c_msg.write(self.address, b"R")
        bus.i2c_rd = smbus2.i2c_msg.read(self.address, 32)
        bus.i2c_rdwr(bus.i2c_wr)
        time.sleep(self.read_delay)
        bus.i2c_rdwr(bus.i2c_rd)
        data = bytes(bus.i2c_rd)
        # EZO returns a status byte (1 = success) then ASCII digits, NUL-padded
        if data and data[0] == 1:
            text = data[1:].split(b"\x00", 1)[0].decode("ascii", "ignore").strip()
            return float(text)
        raise RuntimeError(f"EZO 0x{self.address:02x} read failed (status {data[:1]!r})")


class HardwareDriver(SensorDriver):
    """Reads a set of probes on a fixed cadence and yields node cycles.

    ``probes`` maps engine channel keys to ``Probe`` instances. ``u`` is the known
    process input — a fixed list, or a callable returning one each cycle. A probe
    that raises is skipped for that cycle (the estimator handles missing channels).
    """

    def __init__(self, spec, probes: dict, u, poll_interval_s: float = 60.0,
                 max_cycles: int | None = None, u_forecast=None, _clock=time.time,
                 _sleep=time.sleep):
        self.spec = spec
        self.probes = probes
        self._u = u
        self.poll = float(poll_interval_s)
        self.max_cycles = max_cycles
        self.u_forecast = u_forecast
        self._clock = _clock
        self._sleep = _sleep

    def _u_now(self):
        return list(self._u() if callable(self._u) else self._u)

    def steps(self):
        t0 = self._clock()
        last = t0
        n = 0
        while self.max_cycles is None or n < self.max_cycles:
            now = self._clock()
            t_h = (now - t0) / 3600.0
            dt_h = max((now - last) / 3600.0, 1e-6)
            last = now
            meas = {}
            for key, probe in self.probes.items():
                try:
                    meas[key] = float(probe.read())
                except Exception:
                    pass  # skip a flaky probe this cycle
            yield t_h, dt_h, meas, self._u_now()
            n += 1
            if self.max_cycles is None or n < self.max_cycles:
                self._sleep(self.poll)
