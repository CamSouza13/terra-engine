# Deploying a Terra edge node

The node runtime is numpy-only and runs on a Raspberry Pi Zero 2 W (512 MB) or
larger. Calibration (jax/numpyro) runs off-node on a workstation, not here.

## Quick provision (Raspberry Pi OS Lite)

```bash
sudo bash firstboot.sh aquaculture      # or soil / bioremediation / blss
```

This brands the node (`terra-node` hostname, `terra` user), installs the engine,
enables I2C for smart sensors, and starts the `terra-node` systemd service with
a bring-up self-test. Then:

```bash
journalctl -u terra-node -f             # live event log
```

## What runs

- `terra-node.service` runs `terra node --domain <d>`, which loops the estimator
  over the driver, surfaces events, and persists state atomically to
  `/var/lib/terra/state.json` so a reboot resumes mid-run.
- `ExecStartPre` runs `terra node --selftest` and refuses to start on failure.
- `Restart=always` brings it back after crashes or power loss.

## Going from simulated to real sensors

Today the service uses `SimulatedDriver` (replays a domain sim) so the whole
stack runs with no hardware. To go live, implement a `SensorDriver` that reads
your probes and yields `(t, dt, measurements, u)`:

- **Atlas EZO / I2C smart sensors:** read each EZO over I2C directly on the Pi —
  no ADC, no microcontroller.
- **Raw analog probes:** sample via an ADS1115/ADS1220 ADC (optionally through an
  RP2040 front-end for clean, isolated timing), convert to engineering units,
  and yield the same tuple.

Drop the new driver into `NodeRunner(spec, MyDriver(...))`; nothing else changes.
See `docs/HARDWARE.md` for the board spec and BOM.
