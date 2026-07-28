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

import struct
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


class ModbusProbe(Probe):
    """One reading from a Modbus device (TCP or RS-485/RTU).

    The instruments most aquaculture sites already run — In-Situ Aqua TROLL, YSI
    EXO, Hach SC controllers, OxyGuard, Pentair Point Four — publish their values
    as Modbus registers. Give the register, how many 16-bit words it spans, and the
    decode type; the reading is ``scale * decoded + offset``. Needs ``pymodbus`` on
    a real bus; a client with ``read_holding_registers``/``read_input_registers``
    can be injected for testing.
    """

    _FMT = {"float32": ">f", "float64": ">d", "int32": ">i", "uint32": ">I",
            "int16": ">h", "uint16": ">H"}

    def __init__(self, register, *, host=None, port=502, unit=1,
                 serial_port=None, baudrate=9600, count=2, data_type="float32",
                 word_order="big", fn="holding", scale=1.0, offset=0.0,
                 _client=None):
        if data_type not in self._FMT:
            raise ValueError(f"data_type must be one of {list(self._FMT)}")
        self.register = int(register)
        self.host = host
        self.port = int(port)
        self.unit = int(unit)
        self.serial_port = serial_port
        self.baudrate = int(baudrate)
        self.count = int(count)
        self.data_type = data_type
        self.word_order = word_order
        self.fn = fn
        self.scale = float(scale)
        self.offset = float(offset)
        self._client = _client

    def _connect(self):
        if self._client is not None:
            return self._client
        try:
            if self.serial_port:                # RS-485 / RTU
                from pymodbus.client import ModbusSerialClient
                self._client = ModbusSerialClient(port=self.serial_port,
                                                  baudrate=self.baudrate)
            else:                               # TCP
                from pymodbus.client import ModbusTcpClient
                self._client = ModbusTcpClient(self.host, port=self.port)
        except Exception as e:  # pragma: no cover - env without pymodbus
            raise RuntimeError("ModbusProbe needs pymodbus: pip install pymodbus") from e
        self._client.connect()
        return self._client

    def _decode(self, regs) -> float:
        words = [w & 0xFFFF for w in list(regs)[:self.count]]
        if self.word_order == "little":
            words = words[::-1]
        raw = b"".join(struct.pack(">H", w) for w in words)
        fmt = self._FMT[self.data_type]
        n = struct.calcsize(fmt)
        return float(struct.unpack(fmt, raw[:n])[0])

    def read(self) -> float:
        client = self._connect()
        reader = getattr(client, "read_input_registers" if self.fn == "input"
                         else "read_holding_registers")
        try:
            rr = reader(self.register, count=self.count)
        except TypeError:                       # older pymodbus positional API
            rr = reader(self.register, self.count)
        regs = getattr(rr, "registers", None)
        if not regs:
            raise RuntimeError(f"Modbus read failed at {self.register}: {rr!r}")
        return self.scale * self._decode(regs) + self.offset


class SDI12Probe(Probe):
    """One value from an SDI-12 sonde (In-Situ, YSI, and most environmental sondes).

    SDI-12 is the standard 3-wire digital sonde bus. Wired through a USB/serial
    SDI-12 adapter, this runs a measurement (``aM!``), waits, reads the data
    (``aD0!``) and returns the field at ``index``. A transport callable
    ``send(cmd) -> response`` can be injected for testing.
    """

    def __init__(self, address="0", index=0, *, serial_port=None, baudrate=1200,
                 measure_wait_s=1.0, scale=1.0, offset=0.0, _transport=None,
                 _sleep=time.sleep):
        self.address = str(address)
        self.index = int(index)
        self.serial_port = serial_port
        self.baudrate = int(baudrate)
        self.measure_wait_s = float(measure_wait_s)
        self.scale = float(scale)
        self.offset = float(offset)
        self._transport = _transport
        self._sleep = _sleep

    def _send(self, cmd: str) -> str:
        if self._transport is not None:
            return self._transport(cmd)
        raise RuntimeError("SDI12Probe needs a serial SDI-12 adapter transport")

    @staticmethod
    def _values(resp: str) -> list:
        import re
        return [float(m) for m in re.findall(r"[-+]?\d*\.?\d+", resp)]

    def read(self) -> float:
        self._send(f"{self.address}M!")
        self._sleep(self.measure_wait_s)
        resp = self._send(f"{self.address}D0!")
        vals = self._values(resp)
        # SDI-12 data responses lead with the address digit; drop it if present
        if resp.strip().startswith(self.address) and len(vals) > 1:
            vals = vals[1:]
        if self.index >= len(vals):
            raise RuntimeError(f"SDI-12 index {self.index} out of range: {vals}")
        return self.scale * vals[self.index] + self.offset


class HTTPPollProbe(Probe):
    """Pull one value from a JSON REST endpoint (vendor cloud or on-site gateway).

    Covers cloud-connected systems — In-Situ HydroVu, Eruvaka, Innovasea — and any
    local gateway that serves JSON. ``path`` indexes into the response, e.g.
    ``"parameters.DO.value"`` or ``"readings[0].mgL"``. Auth belongs in ``headers``,
    never the URL. ``_opener(url, headers) -> obj`` can be injected for testing.
    Stdlib only (urllib); no extra dependency.
    """

    def __init__(self, url, path, *, headers=None, timeout=10.0,
                 scale=1.0, offset=0.0, _opener=None):
        self.url = url
        self.path = path
        self.headers = dict(headers or {})
        self.timeout = float(timeout)
        self.scale = float(scale)
        self.offset = float(offset)
        self._opener = _opener

    def _fetch(self):
        if self._opener is not None:
            return self._opener(self.url, self.headers)
        import json as _json
        from urllib.request import Request, urlopen
        req = Request(self.url, headers=self.headers)
        with urlopen(req, timeout=self.timeout) as r:  # noqa: S310 - operator URL
            return _json.loads(r.read().decode("utf-8"))

    @staticmethod
    def _extract(obj, path):
        import re
        for token in re.findall(r"[^.\[\]]+", str(path)):
            if isinstance(obj, dict):
                obj = obj[token]
            elif isinstance(obj, (list, tuple)):
                obj = obj[int(token)]
            else:
                raise KeyError(path)
        return obj

    def read(self) -> float:
        val = self._extract(self._fetch(), self.path)
        return self.scale * float(val) + self.offset


class MQTTProbe(Probe):
    """Latest value from an MQTT topic (modern gateways and brokers).

    Subscribes in the background and returns the most recent message. ``path``
    extracts a field from a JSON payload; omit it if the payload is a bare number.
    Needs ``paho-mqtt`` on a real broker; a client can be injected for testing and
    ``_on_message`` can be fed directly.
    """

    def __init__(self, topic, path=None, *, host="localhost", port=1883,
                 username=None, password=None, scale=1.0, offset=0.0, _client=None):
        self.topic = topic
        self.path = path
        self.host = host
        self.port = int(port)
        self.username = username
        self.password = password
        self.scale = float(scale)
        self.offset = float(offset)
        self._client = _client
        self._latest = None

    def _on_message(self, _client=None, _userdata=None, msg=None):
        payload = getattr(msg, "payload", msg)
        if isinstance(payload, (bytes, bytearray)):
            payload = payload.decode("utf-8", "ignore")
        try:
            import json as _json
            data = _json.loads(payload)
            val = HTTPPollProbe._extract(data, self.path) if self.path else data
        except (ValueError, TypeError, KeyError, IndexError):
            val = payload
        self._latest = float(val)

    def start(self):
        if self._client is None:
            try:
                import paho.mqtt.client as mqtt
            except Exception as e:  # pragma: no cover - env without paho
                raise RuntimeError("MQTTProbe needs paho-mqtt: pip install paho-mqtt") from e
            self._client = mqtt.Client()
            if self.username:
                self._client.username_pw_set(self.username, self.password)
            self._client.connect(self.host, self.port)
        self._client.on_message = self._on_message
        self._client.subscribe(self.topic)
        self._client.loop_start()
        return self

    def read(self) -> float:
        if self._latest is None:
            raise RuntimeError(f"no MQTT message yet on {self.topic}")
        return self.scale * self._latest + self.offset


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
