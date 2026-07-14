# How the Terra Engine works — a plain-language walkthrough

One engine runs every closed loop on the Terra map: a recirculating fish farm, a
crop root-zone, a contaminated site, a spacecraft cabin. This document explains
the shared machinery once, then shows how each domain fills it in — with the
real code.

## The one big idea

Picture a kid guessing how much candy is in a jar. The dumb way: believe
whatever the last person shouted. The smart way: keep **a best guess AND how
sure you are**, and every time a new clue arrives, blend the clue with what you
already believed — moving a lot when you're unsure, barely moving when you're
confident and the clue is noisy.

That is the whole engine. The jar is the living system. The guess is its true
state. The clues are noisy sensors. Terra keeps a guess plus a confidence and
blends them every cycle. And the most important number in every loop is one it
can **never measure directly** — a hidden "health" variable that fails first.

## The shared machinery (same for every domain)

**1. A rulebook — how the system changes over time.** Each domain supplies a
`deriv(x, u, p)` giving the rate of change of every tracked quantity. The engine
rolls it forward with Runge–Kutta (`rk4`).

**2. Predict, then correct.** Every cycle:

```python
# terra/core.py — TerraEngine.step
self.ukf.predict(dt)                 # roll the rulebook forward; grow less sure
if used:                             # whichever sensors reported this cycle
    z = np.array([measurements[c] for c in used])
    self.ukf.update(z, hx, R)        # blend readings into the guess
```

The blend (`terra/ukf.py`) hinges on the Kalman gain `K` — the "how much do I
trust this clue" knob:

```python
K = Pxz @ Sinv                       # my uncertainty ÷ total uncertainty
self.x = self.x + K @ innovation     # nudge guess toward the surprise
self.P = self.P - K @ S @ K.T        # ...and become more sure
```

`innovation` is *(sensor reading) − (what the rulebook expected)*. Here is the
trick that recovers the hidden variable: when a reading disagrees with the
rulebook, the only knob that can explain the disagreement is the hidden health
state, so the filter quietly adjusts it. **It infers what it cannot measure from
the mismatch in what it can.**

**3. Sigma points (the "unscented" part).** The rules are curvy (multiply,
divide), so the engine represents its uncertainty with a handful of sample
points, pushes each through the real rules, and re-measures the cloud after.

**4. Budget closure.** A running "how surprised have I been lately" score
(`nis`) flags when the model stops explaining the data — the earliest tripwire.

**5. Forecast.** A Monte-Carlo daydream: draw a few hundred plausible "true
systems" from the current guess-cloud, fast-forward each, and count how many hit
the danger line.

```python
# terra/core.py — _forecast()
X = self._rng.multivariate_normal(x, P, size=n)   # plausible true states
for k in range(steps):
    X = rk4_batch(X, dt, ...)                      # fast-forward all of them
    crossed |= (value(X) crosses the limit)
p = crossed.mean()                                 # probability of breach
```

**6. Sensors can vanish.** Correction blends only the channels that reported, so
a dead probe just drops out and the engine leans on the rulebook and the rest.

Everything below only changes the rulebook, the tracked states, the sensors, and
the danger line. The engine never changes.

---

## Domain 1 — Aquaculture (RAS) · the Aqua kit

**The loop.** A recirculating fish farm. Feed fish → they excrete ammonia →
biofilter bacteria turn ammonia → nitrite → nitrate.

**Tracked:** `TAN` (ammonia), `NO2`, `NO3`, `DO` (oxygen), and the hidden
**biofilter efficiency `eff`**.

**Hidden failure it catches:** the biofilter silently loses efficiency (clog,
cold shock, toxicity). Ammonia stops being eaten and fish die within hours.

```python
# terra/domains/aquaculture.py
r1 = eff * p.k1 * TAN / (p.K1 + TAN)   # ammonia -> nitrite
r2 = eff * p.k2 * NO2 / (p.K2 + NO2)   # nitrite -> nitrate
# ammonia reads high -> the only explanation is a low eff -> filter lowers it
safety = [SafetyTarget("un-ionized NH3-N", lambda X,p,e: X[0]*frac, 0.05, ">")]
```

**Warns:** *un-ionized NH₃ forecast to breach 0.05 mg/L, ~4 h out.*

## Domain 2 — Soil & controlled-environment agriculture · the Soil kit

**The loop.** A crop root-zone. Organic matter mineralizes to ammonium →
microbes nitrify it to nitrate → the crop takes up nitrate.

**Tracked:** `NH4`, `NO3`, `resp` (CO₂ respiration), and the hidden **microbial
activity `act`**.

**Hidden failure it catches:** microbial activity stalls (cold, compaction,
waterlogging). Ammonium builds, nitrate falls, and the crop starves of nitrate
**days** before a quarterly lab test would reveal it. Note the CO₂ channel is a
*proxy* — it exposes the activity drop even with no nitrogen probes.

```python
# terra/domains/soil.py
r = act * p.k * NH4 / (p.K + NH4)            # nitrification (needs microbes)
d[0] = min_in - r                            # ammonium: added, minus nitrified
d[1] = r - drain*NO3 - p.uptake              # nitrate: made, drained, taken up
d[2] = p.krelax * (act*p.base_resp - resp)   # CO2 respiration tracks activity
channels = {"NH4":..., "NO3":..., "CO2_flux":..., "EC":...}
safety = [SafetyTarget("available nitrate", lambda X,p,e: X[1], 4.0, "<")]
```

**Warns:** *available nitrate forecast to fall below 4 mg-N/L* (crop N deficiency).

This same spec covers both "Controlled-Env Agriculture" and "Soil & Land" on the
site — indoor beds and open farmland are the same root-zone budget.

## Domain 3 — Bioremediation & water/wastewater · the Flow kit

**The loop.** A living cleanup (or a treatment bioreactor). Microbes degrade a
contaminant, fed by an electron donor you dose in.

**Tracked:** `C` (contaminant), `D` (electron donor), and the hidden **degrader
activity `act`**.

**Failure it catches:** here the biology is healthy — the **dosing pump fails**.
The donor falls, the drawdown will stall, and the contaminant rebounds. The
engine confirms the microbes are fine (`act` stays ~1) and pins the problem on
supply. ORP is a proxy channel for the donor/redox state.

```python
# terra/domains/bioremediation.py
rate = act * p.k * C/(p.Kc+C) * D/(p.Kd+D)  # degradation needs BOTH microbes and donor
d[0] = -rate + p.infl*(p.C_src - C)          # contaminant drawn down (+ slow influx)
d[1] = dose - p.y*rate                        # donor: dosed in, consumed
channels = {"contaminant":..., "ORP": lambda x: -50 - 30*x[1]}  # ORP proxies donor
safety = [SafetyTarget("electron donor", lambda X,p,e: X[1], 0.5, "<")]
```

**Warns:** *electron donor forecast to deplete below 0.5 mg/L, ~9 h out* — the
event that stalls the cleanup. A wastewater bioreactor is the same math with a
permit limit as the danger line.

## Domain 4 — Closed habitats & life support (BLSS) · the Core kit

**The loop.** A sealed cabin where crew and plants regenerate each other's air —
no resupply. The hardest version of the problem, and the reason the engine
exists.

**Tracked:** `CO2` (ppm), `O2` (%), and the hidden **crop photosynthetic
capacity `act`**.

**Hidden failure it catches:** crop capacity silently drops (lighting fault,
disease). Photosynthesis can't keep up, CO₂ climbs and O₂ falls toward crew-unsafe
levels.

```python
# terra/domains/blss.py
photo = act * p.Pmax * light * CO2/(p.Kco2+CO2)   # plants fix CO2, make O2
d[0] = p.crew_co2 - photo + p.leak*(p.CO2_amb-CO2) # CO2: crew adds, plants remove
d[1] = -p.crew_o2 + p.k_o2*photo + ...            # O2: crew uses, plants make
safety = [SafetyTarget("cabin O2",  lambda X,p,e: X[1], 19.5, "<"),
          SafetyTarget("cabin CO2", lambda X,p,e: X[0], 5000, ">")]
```

**Warns:** *cabin O₂ forecast to fall below 19.5%, ~13 h out* — hours before the
crew would feel it.

---

## The same story, four times

| Domain | Hidden variable (never measured) | Sensors | Forecast danger line | Fault caught |
|---|---|---|---|---|
| Aquaculture | biofilter efficiency | TAN, NO₂, NO₃, DO | un-ionized ammonia | biofilter crash |
| Soil / CEA | microbial activity | NH₄, NO₃, CO₂ flux, EC | available nitrate | activity stall |
| Bioremediation / water | degrader activity | contaminant, ORP | donor depletion | dosing-pump failure |
| Closed habitat (BLSS) | crop photosynthetic capacity | CO₂, O₂ | cabin O₂ / CO₂ | crop lighting failure |

Every row is the same engine: keep a guess and a confidence, predict with the
rulebook, correct with whatever sensors reported, infer the hidden health you
can't measure, and daydream forward to warn early. From the fish farm to the
Moon, it's one inference core — only the form gets filled in differently.

## See it yourself

```bash
python scripts/run_demo.py                 # all four domains + a sensor-dropout demo
python scripts/run_demo.py --domain blss   # one at a time
python scripts/verify.py                   # calibration, conservation, determinism
```
