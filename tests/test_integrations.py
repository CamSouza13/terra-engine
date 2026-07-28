"""Hardware-integration tests: protocol adapters + the bridge (no real hardware).

Every transport client is injected, so these exercise decoding, JSON-path
extraction, SDI-12 parsing, MQTT payload handling, the system catalog, and the
full probe -> HardwareDriver -> engine path deterministically.
"""
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from terra import integrations as I
from terra.node.hardware import (ModbusProbe, SDI12Probe, HTTPPollProbe,
                                 MQTTProbe, HardwareDriver)
from terra.domains import aquaculture


class FakeModbus:
    """Minimal Modbus client: maps register -> list of 16-bit words."""

    def __init__(self, table):
        self.table = table

    def _read(self, address, count=2):
        words = self.table[address]

        class R:
            registers = list(words)[:count]
        return R()

    read_holding_registers = _read
    read_input_registers = _read


def _f32_words(value):
    raw = struct.pack(">f", value)
    hi, lo = struct.unpack(">HH", raw)
    return [hi, lo]


def test_modbus_decodes_float32():
    table = {5450: _f32_words(7.85), 5454: _f32_words(0.42)}
    client = FakeModbus(table)
    do = ModbusProbe(5450, _client=client)
    tan = ModbusProbe(5454, _client=client)
    assert abs(do.read() - 7.85) < 1e-4
    assert abs(tan.read() - 0.42) < 1e-4


def test_modbus_scale_offset_and_int():
    # a 4-20 mA style scaled int16 register
    client = FakeModbus({10: [1234]})
    p = ModbusProbe(10, count=1, data_type="uint16", scale=0.01, offset=1.0,
                    _client=client)
    assert abs(p.read() - (1234 * 0.01 + 1.0)) < 1e-9


def test_modbus_word_order_swap():
    words = _f32_words(3.14159)
    client = FakeModbus({0: words[::-1]})
    p = ModbusProbe(0, word_order="little", _client=client)
    assert abs(p.read() - 3.14159) < 1e-4


def test_http_poll_extracts_nested_path():
    payload = {"parameters": {"DO": {"value": 6.7, "unit": "mg/L"}}}
    p = HTTPPollProbe("https://api.example.com/x", "parameters.DO.value",
                      _opener=lambda url, headers: payload)
    assert p.read() == 6.7


def test_http_poll_extracts_list_index():
    payload = {"readings": [{"mgL": 5.1}, {"mgL": 9.9}]}
    p = HTTPPollProbe("https://api.example.com/x", "readings[1].mgL",
                      _opener=lambda url, headers: payload)
    assert p.read() == 9.9


def test_http_poll_auth_in_headers_not_url():
    seen = {}

    def opener(url, headers):
        seen["url"] = url
        seen["headers"] = headers
        return {"v": 1.0}

    p = HTTPPollProbe("https://api.example.com/x", "v",
                      headers={"Authorization": "Bearer secret"}, _opener=opener)
    p.read()
    assert "secret" not in seen["url"]
    assert seen["headers"]["Authorization"] == "Bearer secret"


def test_sdi12_parses_field_by_index():
    # aM! then aD0! -> "0+8.23+7.10+25.4" (address echo, DO, pH, temp)
    def transport(cmd):
        return "0+8.23+7.10+25.4" if cmd.endswith("D0!") else "00013"

    do = SDI12Probe("0", index=0, _transport=transport, _sleep=lambda s: None)
    temp = SDI12Probe("0", index=2, _transport=transport, _sleep=lambda s: None)
    assert abs(do.read() - 8.23) < 1e-9
    assert abs(temp.read() - 25.4) < 1e-9


def test_mqtt_latest_message_and_json_path():
    p = MQTTProbe("farm/tank1/do", path="value", _client=object())

    class Msg:
        payload = b'{"value": 6.4}'

    p._on_message(msg=Msg())
    assert p.read() == 6.4


def test_mqtt_bare_number_payload():
    p = MQTTProbe("farm/tank1/do", _client=object())

    class Msg:
        payload = b"7.2"

    p._on_message(msg=Msg())
    assert p.read() == 7.2


def test_normalize_param_aliases():
    assert I.normalize_param("ODO") == "DO"
    assert I.normalize_param("dissolved oxygen") == "DO"
    assert I.normalize_param("NH3-N") == "TAN"
    assert I.normalize_param("Nitrite") == "NO2"
    assert I.normalize_param("Temperature") == "temp"
    assert I.normalize_param("spCond") == "EC"
    assert I.normalize_param("mystery") is None


def test_systems_catalog_wellformed():
    for sid, s in I.SYSTEMS.items():
        assert s["transport"] in ("modbus", "sdi12", "http", "mqtt", "i2c")
        assert s["params"], f"{sid} has no params"
        assert s["vendor"] and s["model"]


def test_build_probes_from_config():
    cfg = {
        "system": "insitu_aquatroll",
        "transport": "modbus",
        "connection": {"host": "10.0.0.20", "port": 502, "unit": 1},
        "channels": {
            "DO":  {"register": 5450, "data_type": "float32"},
            "TAN": {"register": 5454, "data_type": "float32"},
        },
    }
    client = FakeModbus({5450: _f32_words(7.9), 5454: _f32_words(0.35)})
    probes = I.build_probes(cfg, _clients={"modbus": client})
    assert set(probes) == {"DO", "TAN"}
    assert abs(probes["DO"].read() - 7.9) < 1e-4


def test_bridge_feeds_engine_end_to_end():
    """Full path: configured Modbus probes -> HardwareDriver -> engine cycles."""
    spec, _ = aquaculture.simulate()
    cfg = {
        "transport": "modbus",
        "connection": {"host": "x", "port": 502},
        "channels": {
            "TAN": {"register": 0}, "NO2": {"register": 2},
            "NO3": {"register": 4}, "DO": {"register": 6},
        },
        "u": [1.8, 1.0],
        "poll_interval_s": 0,
    }
    client = FakeModbus({0: _f32_words(0.35), 2: _f32_words(0.10),
                         4: _f32_words(6.0), 6: _f32_words(8.5)})
    driver = I.build_bridge(spec, cfg, max_cycles=5,
                            _clients={"modbus": client})
    assert isinstance(driver, HardwareDriver)

    from terra.core import TerraEngine, EngineConfig
    engine = TerraEngine(spec, EngineConfig())
    n, last = 0, None
    for t, dt, meas, u in driver.steps():
        assert set(meas) == {"TAN", "NO2", "NO3", "DO"}
        last = engine.step(t, dt, meas, u)
        n += 1
    assert n == 5
    assert last is not None and last.hidden is not None


def test_sample_assets_bundled():
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assert os.path.isfile(os.path.join(root, "data", "aquaculture_sample.csv"))
    # served statically by the console for the one-click demo
    assert os.path.isfile(os.path.join(root, "terra", "web", "assets",
                                       "aquaculture_sample.csv"))
    assert os.path.isfile(I.sample_path())


def test_sample_bridge_driver_catches_fault():
    """The --simulate path: stream the sample as if live, engine infers the
    biofilter collapse and raises events — no hardware."""
    from terra.core import TerraEngine, EngineConfig
    spec, _ = aquaculture.simulate()
    driver = I.sample_bridge_driver(spec)
    engine = TerraEngine(spec, EngineConfig(outlier_sigma=5.0, track_bias=True,
                                            cusum_threshold=8.0))
    last, n = None, 0
    for t, dt, meas, u in driver.steps():
        last = engine.step(t, dt, meas, u)
        n += 1
    assert n > 150                       # full 48h sample replayed
    assert last.hidden < 0.5             # inferred efficiency collapsed
    assert len(engine.events) >= 1       # at least one warning/alert raised


def test_sample_bridge_respects_max_cycles():
    spec, _ = aquaculture.simulate()
    driver = I.sample_bridge_driver(spec, max_cycles=10)
    assert sum(1 for _ in driver.steps()) == 10


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} passed")
