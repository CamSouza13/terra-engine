# Node hardware: spec, board choice, and BOM

The engine is the moat; the node is a commodity. This documents what the
physical TLK node needs to be, why it is a Raspberry Pi (not an Arduino), and
what it actually costs — because the sensor bill of materials, not the compute,
is what sets the price.

## The one architectural fact

Reliable online **DO, pH, ORP, and EC** probes are digital "smart" sensors
(e.g. Atlas Scientific EZO, I2C/UART). A lab-grade **DO kit alone is ~$355**.
Reliable online **TAN / NO2 / NO3** ion-selective probes are either very
expensive or drift so badly they are useless unattended — which is precisely
the failure Terra exists to solve.

So the node does **not** try to directly sense the nitrogen species. It measures
the cheap, stable channels (DO, pH, ORP, temp, EC, flow) and the **engine infers
TAN / NO2 / NO3** from the coupled model. That is the product: fewer, cheaper,
more reliable sensors, and inference fills the gap. It also means the BOM is
dominated by DO and pH, and the price tiers below are really "how good a DO
probe."

## Arduino vs Raspberry Pi

Not really a versus — they do different jobs, and the engine forces the choice.

| | Raspberry Pi (Zero 2 W / 4 / 5) | Arduino / RP2040 (microcontroller) |
|---|---|---|
| Runs the engine? | **Yes** — Linux, Python, numpy UKF + Monte-Carlo forecast | No — no OS, KB of RAM, weak float |
| Analog inputs | None built in (needs I2C ADC, or use I2C smart sensors) | Built-in 10-bit ADC, deterministic timing |
| Real-time sampling | OS can jitter timing | Microsecond, guaranteed |
| Power | Pi 4 ~15 W; Zero 2 W ~1–2 W | ~0.25 W, sleeps to µA |
| Role at Terra | **The node brain** — inference, forecast, control, comms | Optional front-end: clean analog sampling, isolation, watchdog, low power |

Conclusion:

- **The engine must run on a Pi (or Pi-class SBC).** An Arduino cannot run the
  Bayesian estimator. This is non-negotiable and matches the site's
  "Raspberry-Pi-class deployment."
- **If you use I2C smart sensors (Atlas EZO), you do not need an Arduino or an
  ADC at all** — the Pi reads them directly over I2C. This is the fastest path
  and what the bench should use.
- **Add a microcontroller front-end (RP2040/Pico or Arduino) only** when you use
  raw analog probes (needs a clean ADC + isolation), or when you want a hardware
  watchdog, deterministic sampling, and low-power sleep for remote field nodes.
- **Calibration (jax/numpyro) never runs on the node.** It is heavy and runs
  offline on a laptop/server; the node only runs the numpy edge engine, which
  fits comfortably on a Pi Zero 2 W (512 MB).

Recommended split:

- **Bench / pilot:** Raspberry Pi 4 (4 GB) + Atlas EZO I2C sensors. No Arduino.
- **Field / cost-down:** Pi Zero 2 W + RP2040 front-end + analog probes + ADS1115 ADC + isolation.

## Technical spec (TLK-AQUA reference node)

- **Compute:** Raspberry Pi 4 (4 GB) bench / Pi Zero 2 W field. Quad-core ARM,
  Linux, runs the numpy engine, forecast, controller, and comms.
- **Directly sensed:** dissolved oxygen, pH, ORP, temperature, EC, flow.
- **Inferred by the engine:** TAN, NO2-N, NO3-N (nitrogen loop), plus the hidden
  biofilter-efficiency state.
- **Sampling:** 0.25–1 min cadence; channels enable/disable per timestep (the
  engine already handles dropout).
- **Isolation:** electrically isolated carrier per probe (mandatory in water —
  probes cross-talk otherwise).
- **Anti-fouling:** DO anti-fouling screen; periodic recalibration workflow. The
  engine's drift/biofouling inference (`fit_drift`) is the software half of this.
- **Actuation (optional):** relay or peristaltic dosing pump (e.g. EZO-PMP) for
  water exchange / dosing, behind the controller's authorize-autonomy gate.
- **Power:** 12 V + buck, or PoE HAT; brown-out-safe with state persistence.
- **Comms:** Wi-Fi on-site; cellular or LoRa for remote; store-and-forward when
  offline (the engine runs fully local regardless).
- **Enclosure:** IP65, DIN-rail or wall mount.

## Bill of materials and pricing

Compute and DO are confirmed current prices; other sensor lines are estimates
(labeled) and should be confirmed against the vendor before quoting.

### Tier A — Research / lab-grade (bench, pilots, the STTR narrative)

| Item | ~Unit |
|---|---|
| Raspberry Pi 4 (4 GB) | $55–60 |
| Atlas EZO Dissolved Oxygen kit (probe + circuit + isolated carrier + cal) | **$355** (confirmed) |
| Atlas EZO pH kit (est.) | ~$150 |
| Atlas EZO ORP kit (est.) | ~$120 |
| Atlas EZO conductivity kit (est.) | ~$165 |
| PT-1000 inline temperature kit | ~$85 (confirmed line) |
| Enclosure, power, wiring, flow (est.) | ~$120 |
| **Node total** | **~$1,050** |

Volume pricing on the DO kit alone drops to ~$319 at 25+ units, so a batch build
trims this. This is the honest cost of a lab-grade multi-probe node.

### Tier B — Field / mid-grade (the ~$390 price point)

| Item | ~Unit (est.) |
|---|---|
| Raspberry Pi Zero 2 W | $15 (confirmed) |
| RP2040 front-end + ADS1115 ADC | ~$25 |
| Galvanic/optical DO probe (mid-grade) | ~$90–190 |
| pH + ORP analog probes | ~$70 |
| EC probe | ~$40 |
| DS18B20 temperature | ~$6 |
| Isolation, enclosure, power, wiring | ~$80 |
| **Node total** | **~$330–430** |

This is where the site's "from $390 / node" lives: mid-grade sensors, inference
covering the nitrogen loop. Accuracy is lower per-probe, which is exactly what
the Bayesian fusion is designed to tolerate.

### Tier C — Bring-your-own-sensors

| Item | ~Unit |
|---|---|
| Pi Zero 2 W + carrier + isolation | ~$60 |
| Integration to existing farm probes | labor |
| **Node total** | **~$100–150** |

"Your system, your data, our inference" — the node is just the brain riding on
sensors the site already has.

## Building it Terra-branded, from scratch

Yes, this is buildable as a fully Terra-branded product. The key is knowing what
to build and what to buy, because "from scratch" does not mean fabricating
electrochemistry.

**Do not build:** the raw sensing elements (the pH glass bulb, the DO membrane
and electrode, the ORP electrode). These are decades of chemistry and are sold
as OEM components. Every instrument company buys the electrode and builds the
instrument around it. Buy the bare electrodes/probes as parts.

**Do build, and brand as Terra:**

1. **The TLK board** — your own PCB in Raspberry Pi HAT form factor with the
   Terra silkscreen. It carries the analog front-ends the OEM probes need:
   a potentiostat for DO/ORP, a high-impedance buffer for pH, an AC-excitation
   driver for EC, a precision 16–24-bit ADC (ADS1220-class), per-channel
   galvanic isolation, an RTC, one-wire/RS-485 for temperature, screw terminals,
   and a HAT-ID EEPROM. This is a real but very achievable EE project and it is
   what lets you drop the Atlas carriers and own the margin.
2. **On-board co-processor** — an RP2040 on the TLK board does the deterministic,
   isolated sampling and exposes a clean register map to the Pi over I2C. Terra
   firmware, OTA-updatable. (This is the "Arduino role" done properly, on your
   own board, not a separate Uno.)
3. **The node image** — a Terra-branded Raspberry Pi OS image: boot splash,
   `terra` hostname, the engine running as a systemd service, first-boot
   provisioning and self-test. Fully brandable today with the software already
   in this repo.
4. **Enclosure + industrial design** — IP65 enclosure with Terra label/silkscreen,
   DIN-rail/wall mount. Start with an off-the-shelf enclosure, custom faceplate;
   move to a molded enclosure at volume.

**Manufacturing path:** prototype PCBA at JLCPCB / PCBWay (cheap, fast); move
low-volume production to a US contract manufacturer (MacroFab, Screaming
Circuits) — which also strengthens the SBIR/STTR domestic-build story.

**Regulatory:** FCC/CE (using a pre-certified Pi radio module de-risks this),
RoHS, and calibration traceability for any accuracy claims.

**Margin:** at Tier B (~$330–430 cost) the Atlas carrier boards are a big chunk;
a Terra TLK board replacing them both cuts cost and creates the margin that a
"sell at cost" line does not have. Owning the board is what turns hardware from
a cost center into a branded product.

**Sequence:**

- **v0** — a socketed dev board that hosts commodity front-end modules (fast,
  de-risks the analog design, still Terra-branded firmware + image).
- **v1** — an integrated 4-channel TLK-AQUA board (DO, pH, ORP, temp + EC/flow).
- **v2** — domain variants: TLK-SOIL, TLK-FLOW, TLK-CORE on the same platform.

## v0 dev board — detailed BOM

The goal of v0 is to de-risk the analog design and run the real Terra node
firmware/software, while socketing commodity modules so nothing is committed to
a custom PCB yet. Everything below is off-the-shelf; prices are approximate and
should be confirmed before ordering.

| # | Part | Purpose | ~Unit |
|---|---|---|---|
| 1 | Raspberry Pi Zero 2 W | node brain, runs the engine | $15 |
| 2 | microSD 32 GB (A1) | OS + Terra node image | $8 |
| 3 | Raspberry Pi Pico (RP2040) | co-processor: deterministic, isolated sampling | $4 |
| 4 | ADS1220 24-bit ADC breakout | precision analog reads | $12 |
| 5 | ADS1115 16-bit ADC breakout | spare/aux channels | $6 |
| 6 | Galvanic isolators (I2C/ADC, per channel) | probe cross-talk protection | $6 × 4 = $24 |
| 7 | DO probe (galvanic, mid-grade) | dissolved oxygen | $90 |
| 8 | pH probe + BNC + buffer board | pH | $35 |
| 9 | ORP probe + board | redox | $30 |
| 10 | EC probe + driver board | conductivity | $40 |
| 11 | DS18B20 (1-wire) | temperature | $6 |
| 12 | DC-DC buck + 12 V supply | power | $18 |
| 13 | Perfboard / socketed carrier + connectors | integration | $20 |
| 14 | IP65 enclosure + glands | housing | $25 |
| | **v0 node total** | | **~$330** |

Notes:

- Swap items 7–10 for Atlas EZO I2C kits to get lab-grade accuracy (and skip the
  ADC entirely) at roughly +$500–700 — that is the bench/pilot build, not v0.
- Items 3–6 are the whole reason the v1 TLK board exists: v1 folds the RP2040,
  ADC, isolation, and front-ends onto one Terra-branded HAT, cutting cost, size,
  and assembly labor.
- Firmware (RP2040) and the node image are already spec'd in this repo
  (`terra/node/`, `deploy/`), so v0 is mostly wiring plus writing the sensor
  driver that replaces `SimulatedDriver`.

## What still needs to be built (hardware track)

- Sensor driver layer: read EZO I2C (bench) and ADS1115 analog (field) into `engine.step`
- Node service: boot-to-run, watchdog, state persistence, store-and-forward buffer
- Actuator driver + safety interlocks behind the autonomy gate (dosing, exchange)
- Calibration + anti-fouling field workflow (pairs with `fit_drift`)
- Enclosure + power + comms hardening; a bring-up/self-test routine
- A reference BOM per domain (soil, remediation, BLSS) mirroring TLK-AQUA

See [`ROADMAP.md`](../ROADMAP.md) for how this sequences with the software work.
