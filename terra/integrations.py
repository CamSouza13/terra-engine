"""Connect Terra to the sensor hardware a site already runs.

The engine rides on whatever channels report each cycle (see ``terra/ingest.py``
and ``terra/node/hardware.py``). This module maps the field instruments common in
aquaculture — In-Situ Aqua TROLL, YSI EXO, Hach SC, Atlas Scientific EZO,
OxyGuard, Pentair Point Four, plus cloud systems like In-Situ HydroVu and
Eruvaka — onto that channel schema, so an operator points Terra at existing
probes instead of installing new ones.

A bridge config is a plain dict (JSON/YAML-loadable)::

    system: insitu_aquatroll
    transport: modbus
    connection: {host: 10.0.0.20, port: 502, unit: 1}
    channels:
      DO:  {register: 5450, data_type: float32}
      TAN: {register: 5454, data_type: float32}
    u: [1.8, 1.0]
    poll_interval_s: 60

``build_probes(cfg)`` returns ``{channel: Probe}`` ready for ``HardwareDriver``;
``build_bridge(spec, cfg)`` returns the driver itself. Stdlib + the node's probe
classes only; transport libraries (pymodbus, paho-mqtt, pyserial) are imported
lazily and only when a real bus is used.
"""
from __future__ import annotations

from typing import Any

# ---- canonical channels + vendor parameter aliases --------------------------
# The engine estimates on domain channels (aquaculture: TAN, NO2, NO3, DO).
# pH / temp / EC / ORP are commonly reported too and carried as environment/
# context; normalize_param maps any vendor label onto a canonical key.
CANONICAL = ["TAN", "NO2", "NO3", "DO", "pH", "temp", "EC", "ORP",
             "salinity", "turbidity"]

_ALIASES: dict[str, list[str]] = {
    "TAN": ["tan", "ammonia", "nh3", "nh4", "nh3-n", "nh4-n", "total ammonia",
            "ammonium", "total ammonia nitrogen"],
    "NO2": ["no2", "nitrite", "no2-n"],
    "NO3": ["no3", "nitrate", "no3-n"],
    "DO":  ["do", "odo", "rdo", "ldo", "dissolved oxygen", "do mgl", "domgl",
            "o2", "oxygen"],
    "pH":  ["ph"],
    "temp": ["temp", "temperature", "temp c", "tempc", "wtemp", "water temp",
             "water temperature"],
    "EC":  ["ec", "conductivity", "spcond", "sp cond", "cond",
            "specific conductance"],
    "ORP": ["orp", "redox", "redox potential"],
    "salinity": ["sal", "salinity", "psu"],
    "turbidity": ["turb", "turbidity", "ntu", "fnu"],
}
_LOOKUP = {a: c for c, al in _ALIASES.items() for a in al}


def normalize_param(name: Any) -> str | None:
    """Map a vendor parameter label onto a canonical Terra channel, or None."""
    if name is None:
        return None
    key = str(name).strip().lower().replace("_", " ")
    if key in _LOOKUP:
        return _LOOKUP[key]
    for c in CANONICAL:
        if key == c.lower():
            return c
    return None


# ---- catalog of common aquaculture instruments ------------------------------
# ``transport`` is the default bus; ``params`` are the channels the system can
# report. Register maps are user-configurable per install (vendors let you remap
# Modbus registers), so the bridge config supplies the exact registers/paths.
SYSTEMS: dict[str, dict] = {
    "insitu_aquatroll": {
        "vendor": "In-Situ", "model": "Aqua TROLL 500/600",
        "transport": "modbus", "buses": ["Modbus RTU (RS-485)", "SDI-12"],
        "params": ["DO", "pH", "temp", "EC", "ORP", "salinity", "turbidity"],
        "notes": "Multiparameter sonde. Registers are user-mapped in VuSitu; "
                 "values are float32. Also speaks SDI-12 (transport: sdi12).",
    },
    "ysi_exo": {
        "vendor": "YSI (Xylem)", "model": "EXO2 / EXO3",
        "transport": "modbus", "buses": ["Modbus RTU/TCP", "SDI-12"],
        "params": ["DO", "pH", "temp", "EC", "ORP", "salinity", "turbidity"],
        "notes": "Via an EXO signal-output adapter (DCP/Modbus). float32 registers.",
    },
    "hach_sc": {
        "vendor": "Hach", "model": "SC200 / SC4500 controller",
        "transport": "modbus", "buses": ["Modbus RTU/TCP", "Profibus", "4-20 mA"],
        "params": ["DO", "pH", "temp", "ORP"],
        "notes": "Controller with LDO dissolved-oxygen and pH sensors. Modbus "
                 "register map per the SC manual.",
    },
    "atlas_ezo": {
        "vendor": "Atlas Scientific", "model": "EZO (pH/DO/EC/ORP/RTD)",
        "transport": "i2c", "buses": ["I2C", "UART"],
        "params": ["DO", "pH", "EC", "ORP", "temp"],
        "notes": "Common in DIY/OEM RAS builds. I2C addresses: pH 0x63, EC 0x64, "
                 "DO 0x61, ORP 0x62, RTD 0x66.",
    },
    "oxyguard": {
        "vendor": "OxyGuard", "model": "Commander / Pacific",
        "transport": "modbus", "buses": ["Modbus RS-485", "4-20 mA"],
        "params": ["DO", "temp", "pH"],
        "notes": "Dissolved-oxygen probes and controllers widely used in RAS.",
    },
    "pointfour": {
        "vendor": "Pentair", "model": "Point Four RIU",
        "transport": "modbus", "buses": ["Modbus RS-485"],
        "params": ["DO", "temp", "pH"],
        "notes": "Regulating & Instrumentation Unit; monitors DO and controls "
                 "degassing.",
    },
    "insitu_hydrovu": {
        "vendor": "In-Situ", "model": "HydroVu cloud",
        "transport": "http", "buses": ["REST (HTTPS)"],
        "params": ["DO", "pH", "temp", "EC", "ORP", "turbidity"],
        "notes": "Cloud API for Aqua TROLL deployments. Token in headers; JSON "
                 "path to each parameter value.",
    },
    "eruvaka": {
        "vendor": "Eruvaka (Nutreco)", "model": "PondGuard / PondMother",
        "transport": "http", "buses": ["REST (HTTPS)", "GSM"],
        "params": ["DO", "temp", "pH"],
        "notes": "Pond aerator monitors/controllers. Pull readings from the "
                 "account API; auth via headers.",
    },
    "generic_modbus": {
        "vendor": "Generic", "model": "Any Modbus device",
        "transport": "modbus", "buses": ["Modbus TCP", "Modbus RTU"],
        "params": CANONICAL,
        "notes": "Supply register + data_type per channel.",
    },
    "generic_sdi12": {
        "vendor": "Generic", "model": "Any SDI-12 sonde",
        "transport": "sdi12", "buses": ["SDI-12"],
        "params": CANONICAL,
        "notes": "Supply address + index per channel.",
    },
    "generic_mqtt": {
        "vendor": "Generic", "model": "Any MQTT gateway",
        "transport": "mqtt", "buses": ["MQTT"],
        "params": CANONICAL,
        "notes": "Supply topic (+ optional JSON path) per channel.",
    },
    "generic_http": {
        "vendor": "Generic", "model": "Any JSON REST endpoint",
        "transport": "http", "buses": ["REST"],
        "params": CANONICAL,
        "notes": "Supply url + JSON path per channel; auth in headers.",
    },
}


def describe_systems() -> list[dict]:
    """A flat list of supported systems for the CLI / console."""
    return [{"id": k, **v} for k, v in SYSTEMS.items()]


def _resolve_transport(cfg: dict) -> str:
    t = cfg.get("transport")
    if not t and cfg.get("system") in SYSTEMS:
        t = SYSTEMS[cfg["system"]]["transport"]
    if not t:
        raise ValueError("config needs a 'transport' (or a known 'system')")
    return t


def build_probes(cfg: dict, *, _clients: dict | None = None) -> dict:
    """Build ``{channel: Probe}`` from a bridge config.

    ``_clients`` optionally injects shared transport clients for testing, keyed by
    transport name: ``{"modbus": client, "sdi12": send, "http": opener,
    "mqtt": client}``.
    """
    from .node.hardware import (ModbusProbe, SDI12Probe, HTTPPollProbe,
                                MQTTProbe, AtlasEZO)
    clients = _clients or {}
    transport = _resolve_transport(cfg)
    conn = dict(cfg.get("connection") or {})
    channels = cfg.get("channels") or {}
    if not channels:
        raise ValueError("config needs a non-empty 'channels' map")

    probes: dict = {}
    for ch, raw in channels.items():
        opt = dict(raw or {})
        if transport == "modbus":
            if "register" not in opt:
                raise ValueError(f"channel {ch!r} needs a 'register'")
            probes[ch] = ModbusProbe(
                opt.pop("register"), _client=clients.get("modbus"),
                host=conn.get("host"), port=conn.get("port", 502),
                unit=conn.get("unit", 1), serial_port=conn.get("serial_port"),
                baudrate=conn.get("baudrate", 9600), **opt)
        elif transport == "sdi12":
            probes[ch] = SDI12Probe(
                opt.pop("address", "0"), opt.pop("index", 0),
                serial_port=conn.get("serial_port"),
                baudrate=conn.get("baudrate", 1200),
                _transport=clients.get("sdi12"), **opt)
        elif transport in ("http", "cloud"):
            probes[ch] = HTTPPollProbe(
                opt.pop("url", conn.get("url")), opt.pop("path"),
                headers=conn.get("headers"), _opener=clients.get("http"), **opt)
        elif transport == "mqtt":
            probes[ch] = MQTTProbe(
                opt.pop("topic"), opt.pop("path", None),
                host=conn.get("host", "localhost"), port=conn.get("port", 1883),
                username=conn.get("username"), password=conn.get("password"),
                _client=clients.get("mqtt"), **opt)
        elif transport == "i2c":
            probes[ch] = AtlasEZO(int(opt.pop("address")),
                                  bus=conn.get("bus", 1), **opt)
        else:
            raise ValueError(f"unknown transport {transport!r}")
    return probes


def sample_path() -> str:
    """Path to the bundled sample aquaculture log."""
    import os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(here, "data", "aquaculture_sample.csv")


def sample_bridge_driver(spec, path: str | None = None, *, max_cycles=None):
    """A driver that streams the bundled sample log as if it were live hardware.

    Yields the same ``(t, dt, meas, u)`` cycles a real ``HardwareDriver`` produces,
    so ``terra bridge --simulate`` exercises the live streaming path — engine,
    state persistence, events — with no sensors attached.
    """
    from .node.driver import SensorDriver
    from .ingest import load_csv, _parse_time, _to_hours

    path = path or sample_path()
    with open(path) as f:
        header = f.readline().strip().split(",")
    tcol = "t" if "t" in header else ("timestamp" if "timestamp" in header
                                      else header[0])
    _, rows = load_csv(path, tcol)
    times = _to_hours([_parse_time(r[tcol]) for r in rows])
    chans = [c for c in spec.channels if c in header]
    ucol = "excretion_kg_h" if "excretion_kg_h" in header else None
    n = len(rows) if max_cycles is None else min(int(max_cycles), len(rows))

    class _SampleDriver(SensorDriver):
        def steps(self):
            for i in range(n):
                meas = {}
                for c in chans:
                    raw = (rows[i].get(c) or "").strip()
                    if raw:
                        try:
                            meas[c] = float(raw)
                        except ValueError:
                            pass
                u = 0.0
                if ucol:
                    uraw = (rows[i].get(ucol) or "").strip()
                    if uraw:
                        try:
                            u = (float(uraw), 1.0)
                        except ValueError:
                            pass
                dt = (times[i] - times[i - 1] if i > 0
                      else (times[1] - times[0] if len(times) > 1 else 1.0))
                yield times[i], dt, meas, u

    return _SampleDriver()


def build_bridge(spec, cfg: dict, *, u=None, poll_interval_s=None,
                 max_cycles=None, _clients: dict | None = None):
    """Return a ``HardwareDriver`` that streams the configured hardware into the
    engine. Any probe with a ``start()`` (e.g. MQTT) is started here."""
    from .node.hardware import HardwareDriver
    probes = build_probes(cfg, _clients=_clients)
    for p in probes.values():
        if hasattr(p, "start"):
            p.start()
    u_val = u if u is not None else cfg.get("u", 0.0)
    poll = (poll_interval_s if poll_interval_s is not None
            else float(cfg.get("poll_interval_s", 60.0)))
    return HardwareDriver(spec, probes, u_val, poll_interval_s=poll,
                          max_cycles=max_cycles)
