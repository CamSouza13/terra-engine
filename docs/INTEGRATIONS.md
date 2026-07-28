# Connecting Terra to existing sensor hardware

Terra does not require its own probes. The engine runs on whatever channels
report each cycle, so it can ride on the instruments a site already operates.
The `terra bridge` command reads from common aquaculture hardware, normalizes
each value onto Terra's channel schema, and feeds the same estimator that the
node and the CSV replay path use.

You do not need a live node to start. A logged CSV export from any of these
systems replays through the engine today (`terra backtest`, or **Calibrate →
Data** in the console). The bridge is for streaming live.

## Supported systems

| System | Vendor | Bus | Reports |
| --- | --- | --- | --- |
| Aqua TROLL 500/600 | In-Situ | Modbus RTU (RS-485), SDI-12 | DO, pH, temp, EC, ORP, salinity, turbidity |
| EXO2 / EXO3 | YSI (Xylem) | Modbus RTU/TCP, SDI-12 | DO, pH, temp, EC, ORP, turbidity |
| SC200 / SC4500 | Hach | Modbus, Profibus, 4–20 mA | DO (LDO), pH, temp, ORP |
| EZO (pH/DO/EC/ORP/RTD) | Atlas Scientific | I2C, UART | DO, pH, EC, ORP, temp |
| Commander / Pacific | OxyGuard | Modbus RS-485, 4–20 mA | DO, temp, pH |
| Point Four RIU | Pentair | Modbus RS-485 | DO, temp, pH |
| HydroVu cloud | In-Situ | REST (HTTPS) | DO, pH, temp, EC, ORP, turbidity |
| PondGuard / PondMother | Eruvaka (Nutreco) | REST (HTTPS), GSM | DO, temp, pH |

Anything speaking Modbus, SDI-12, MQTT, or a JSON REST endpoint is covered by
the `generic_*` entries. List them at any time:

```
terra bridge --list
```

## How it maps onto the engine

The aquaculture domain estimates on `TAN`, `NO2`, `NO3`, `DO`. Sensors report
under many labels (`ODO`, `RDO`, `dissolved oxygen`, `NH3-N`, `spCond`, …);
`normalize_param` maps those onto canonical channels. Parameters the estimator
does not use directly (pH, temperature, EC) are still read and carried as
context. Partial and irregular reporting is fine — a channel that is silent on a
cycle is simply skipped, and the estimator rides on whatever is present.

## Bridge config

A config is plain JSON or YAML.

Modbus (In-Situ Aqua TROLL over RS-485/TCP):

```yaml
system: insitu_aquatroll
transport: modbus
connection: { host: 10.0.0.20, port: 502, unit: 1 }
channels:
  DO:  { register: 5450, data_type: float32 }
  TAN: { register: 5454, data_type: float32 }
u: [1.8, 1.0]          # known process input (feed, exchange multiplier)
poll_interval_s: 60
```

Register numbers and word order come from the instrument's Modbus manual;
`data_type` is one of `float32`, `float64`, `int16/32`, `uint16/32`. Use
`word_order: little` if a 32-bit value reads swapped.

Cloud REST (In-Situ HydroVu / Eruvaka). Put credentials in `headers`, never in
the URL:

```yaml
system: insitu_hydrovu
transport: http
connection:
  headers: { Authorization: "Bearer ${HYDROVU_TOKEN}" }
channels:
  DO:  { url: "https://www.hydrovu.com/public-api/v1/locations/123/data", path: "parameters.DO.value" }
```

MQTT gateway:

```yaml
transport: mqtt
connection: { host: broker.local, port: 1883 }
channels:
  DO: { topic: "farm/tank1/do", path: "value" }
```

SDI-12 sonde through a USB adapter:

```yaml
transport: sdi12
connection: { serial_port: /dev/ttyUSB0, baudrate: 1200 }
channels:
  DO:   { address: "0", index: 0 }
  temp: { address: "0", index: 2 }
```

## Verify wiring, then run

Read each configured probe once and print the value before committing to a run:

```
terra bridge --config farm.yaml --check
```

```
  [ok ] DO       = 7.85   -> channel DO
  [ok ] TAN      = 0.42   -> channel TAN
bridge check: PASS
```

Then stream live into the engine (and, with node credentials, report to the
console):

```
terra bridge --config farm.yaml
```

## Optional dependencies

Core Terra is numpy-only. Transport libraries are imported lazily and only when
a real bus is used:

- Modbus: `pip install pymodbus`
- MQTT: `pip install paho-mqtt`
- SDI-12 / serial: `pip install pyserial`
- HTTP/REST: none (standard library)

Every adapter accepts an injected client, so the whole path is unit-tested
without hardware (see `tests/test_integrations.py`).
