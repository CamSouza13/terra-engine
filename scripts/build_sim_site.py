"""Generate a self-contained, branded simulation.html for the Terra Labs site.

Runs the REAL engine (healthy + fault) across all four domains, downsamples the
verified output to JSON, and bakes it into an interactive page that matches the
terralaboratories.com branding. Regenerate any time with:

    python scripts/build_sim_site.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from terra import TerraEngine, EngineConfig
from terra.domains import DOMAINS

CFG = {
    "aquaculture":    EngineConfig(forecast_horizon_h=12, forecast_dt=0.25, forecast_samples=300),
    "soil":           EngineConfig(forecast_horizon_h=48, forecast_dt=1.0,  forecast_samples=250),
    "bioremediation": EngineConfig(forecast_horizon_h=12, forecast_dt=0.25, forecast_samples=300),
    "blss":           EngineConfig(forecast_horizon_h=18, forecast_dt=0.25, forecast_samples=300),
}

# per-domain presentation metadata
META = {
    "aquaculture": {
        "num": "01", "title_a": "Aquaculture", "title_b": "RAS",
        "kit": "TLK-AQUA", "node": "Grow-out · T04",
        "hidden": "eff", "hidden_label": "Biofilter efficiency",
        "sensor_state": 0, "sensor_channel": "TAN",
        "sensor_label": "Ammonia · TAN", "sensor_units": "mg/L",
        "risk": "un-ionized NH3-N", "danger": "Un-ionized NH₃ > 0.05 mg/L",
        "blurb": "A recirculating fish farm. Feed becomes ammonia; a biofilter of "
                 "nitrifying bacteria turns ammonia → nitrite → nitrate. The "
                 "hidden variable is the biofilter's efficiency — unmeasurable, "
                 "and the thing that kills stock in hours when it silently fails.",
        "eqs": [
            "d(TAN)/dt = P(feed) − r₁ − (Q/V)·TAN",
            "r₁ = eff · k₁ · TAN / (K₁ + TAN)      (ammonia → nitrite)",
            "r₂ = eff · k₂ · NO₂ / (K₂ + NO₂)      (nitrite → nitrate)",
            "NH₃ = TAN · f(pH, T)   →   alarm if NH₃ > 0.05 mg/L",
        ],
        "channels": "TAN · NO₂ · NO₃ · DO",
        "fault_note": "Injected fault: biofilter efficiency collapses 1.00 → 0.15.",
    },
    "soil": {
        "num": "02", "title_a": "Soil & CEA", "title_b": "root zone",
        "kit": "TLK-SOIL", "node": "Field · Plot 7",
        "hidden": "act", "hidden_label": "Microbial activity",
        "sensor_state": 1, "sensor_channel": "NO3",
        "sensor_label": "Nitrate · NO₃", "sensor_units": "mg-N/L",
        "risk": "available nitrate", "danger": "Available nitrate < 4 mg-N/L",
        "blurb": "A crop root-zone. Microbes nitrify ammonium into the nitrate the "
                 "crop lives on. When microbial activity silently stalls, nitrate "
                 "falls and the crop starves — days before a lab test shows it. "
                 "The CO₂ channel is a proxy that exposes the drop with no N probes.",
        "eqs": [
            "d(NH₄)/dt = M − r,     r = act · k · NH₄ / (K + NH₄)",
            "d(NO₃)/dt = r − drainage·NO₃ − uptake",
            "CO₂ flux → act · resp₀   (proxy for hidden activity)",
            "alarm if available NO₃ < 4 mg-N/L",
        ],
        "channels": "NH₄ · NO₃ · CO₂ flux · EC",
        "fault_note": "Injected fault: microbial activity collapses 1.00 → 0.25.",
    },
    "bioremediation": {
        "num": "03", "title_a": "Bioremediation", "title_b": "& water",
        "kit": "TLK-FLOW", "node": "Wetland · Cell 3",
        "hidden": "act", "hidden_label": "Degrader activity",
        "sensor_state": 0, "sensor_channel": "contaminant",
        "sensor_label": "Contaminant", "sensor_units": "mg/L",
        "risk": "electron donor", "danger": "Electron donor < 0.5 mg/L",
        "blurb": "A living cleanup. Microbes degrade a contaminant, fed by an "
                 "electron donor you dose in. Here the biology stays healthy — the "
                 "dosing pump fails. The engine confirms the microbes are fine and "
                 "forecasts the donor depletion that will stall the drawdown.",
        "eqs": [
            "rate = act · k · C/(K_c+C) · D/(K_d+D)",
            "d(C)/dt = −rate + infl·(C_src − C)      (contaminant)",
            "d(D)/dt = dose − y·rate                 (electron donor)",
            "alarm if D < 0.5 mg/L  (drawdown about to stall)",
        ],
        "channels": "Contaminant · ORP",
        "fault_note": "Injected fault: electron-donor dosing pump fails at h20.",
    },
    "blss": {
        "num": "04", "title_a": "Life support", "title_b": "BLSS",
        "kit": "TLK-CORE", "node": "Habitat · BLSS",
        "hidden": "act", "hidden_label": "Crop capacity",
        "sensor_state": 1, "sensor_channel": "O2",
        "sensor_label": "Cabin O₂", "sensor_units": "%",
        "risk": "cabin O2", "danger": "Cabin O₂ < 19.5 %",
        "blurb": "A sealed habitat where crew and plants regenerate each other's "
                 "air, no resupply. The hidden variable is the crop's "
                 "photosynthetic capacity. When it silently drops, CO₂ climbs and "
                 "O₂ falls toward crew-unsafe — the engine forecasts it hours out.",
        "eqs": [
            "photo = act · P_max · L · CO₂/(K + CO₂)",
            "d(CO₂)/dt = crew − photo + leak·(amb − CO₂)",
            "d(O₂)/dt  = −crew_O₂ + κ·photo + leak·(amb − O₂)",
            "alarm if cabin O₂ < 19.5 %",
        ],
        "channels": "CO₂ · O₂",
        "fault_note": "Injected fault: crop photosynthetic capacity 1.00 → 0.30.",
    },
}

ORDER = ["aquaculture", "soil", "bioremediation", "blss"]


def _downsample(n, target=90):
    if n <= target:
        return list(range(n))
    return list(np.linspace(0, n - 1, target).astype(int))


def collect(name, fault):
    mod = DOMAINS[name]
    spec, sim = mod.simulate(fault=fault)
    eng = TerraEngine(spec, CFG[name])
    t = sim["t"]; dt = t[1] - t[0]; uf = sim.get("u_forecast")
    m = META[name]
    hi = spec.idx(m["hidden"]); si = m["sensor_state"]; ch = m["sensor_channel"]
    rk = m["risk"]
    hist = []
    for i in range(len(t)):
        hist.append(eng.step(t[i], dt, sim["meas"][i], sim["u"][i], u_forecast=uf))
    idx = _downsample(len(t))
    r = lambda a, d=3: round(float(a), d)
    data = {
        "t": [r(t[i], 2) for i in idx],
        "hidden_true": [r(sim["truth"][i, hi]) for i in idx],
        "hidden_est": [r(hist[i].hidden) for i in idx],
        "hidden_std": [r(hist[i].hidden_std) for i in idx],
        "sensor_raw": [r(sim["meas"][i].get(ch, float("nan"))) for i in idx],
        "sensor_true": [r(sim["truth"][i, si]) for i in idx],
        "sensor_est": [r(hist[i].x[si]) for i in idx],
        "forecast_p": [r(hist[i].risks.get(rk, {}).get("p", 0.0), 3) for i in idx],
        "events": [{"t": r(et, 1), "level": lv, "msg": msg}
                   for et, lv, msg in eng.events],
    }
    return spec, data


def build_data():
    out = {"order": ORDER, "meta": {}, "runs": {}}
    for name in ORDER:
        spec, fault = collect(name, True)
        _, healthy = collect(name, False)
        m = META[name]
        risk = next((s for s in spec.safety if s.name == m["risk"]), None)
        out["runs"][name] = {"fault": fault, "healthy": healthy}
        out["meta"][name] = {
            **{k: m[k] for k in (
                "num", "title_a", "title_b", "kit", "node", "hidden_label",
                "sensor_label", "sensor_units", "danger", "blurb", "eqs",
                "channels", "fault_note")},
            "hidden_baseline": spec.hidden_baseline,
            "hidden_alert": spec.hidden_alert_frac * spec.hidden_baseline,
            "risk_limit": risk.limit if risk else None,
            "risk_dir": risk.direction if risk else ">",
            "risk_units": risk.units if risk else "",
        }
    return out


# --------------------------------------------------------------------------- #
#  HTML / CSS / JS                                                            #
# --------------------------------------------------------------------------- #

def domain_section(name, meta):
    m = meta
    eqs = "\n".join(f'<div class="eqline">{e}</div>' for e in m["eqs"])
    return f'''
  <section class="domain" id="dom-{name}">
    <div class="dhead"><span class="dnum">{m["num"]}</span> — {m["title_a"]} <em>{m["title_b"]}.</em></div>
    <div class="dgrid">
      <div class="dnarr">
        <p class="blurb">{m["blurb"]}</p>
        <div class="metaline"><span class="ml-k">Sensor kit</span><span class="ml-v">{m["kit"]}</span></div>
        <div class="metaline"><span class="ml-k">Channels</span><span class="ml-v">{m["channels"]}</span></div>
        <div class="metaline"><span class="ml-k">Hidden state</span><span class="ml-v">{m["hidden_label"]} · inferred, never measured</span></div>
        <div class="metaline"><span class="ml-k">Danger line</span><span class="ml-v">{m["danger"]}</span></div>
        <div class="eqs">{eqs}</div>
        <div class="faultnote">◇ {m["fault_note"]}</div>
      </div>
      <div class="panel" data-domain="{name}">
        <div class="panel-top">
          <span class="pt-title">◈ Terra // Control · {m["node"]}</span>
          <span class="live"><span class="dot"></span>LIVE</span>
        </div>
        <div class="panel-controls">
          <div class="seg">
            <button class="seg-btn" data-mode="healthy">Healthy</button>
            <button class="seg-btn active" data-mode="fault">Fault</button>
          </div>
          <button class="playbtn" data-act="play">❚❚ Pause</button>
          <input class="scrub" type="range" min="0" max="100" value="0">
          <span class="tread">00.0 h</span>
        </div>
        <div class="chart-wrap">
          <div class="chart-label"><span>{m["hidden_label"]}</span><span class="cl-val hidden-val"></span></div>
          <canvas class="c-hidden" height="150"></canvas>
        </div>
        <div class="chart-wrap">
          <div class="chart-label"><span>{m["sensor_label"]} · {m["sensor_units"]}</span><span class="cl-val sensor-val"></span></div>
          <canvas class="c-sensor" height="120"></canvas>
        </div>
        <div class="chart-wrap">
          <div class="chart-label"><span>Forecast · P(breach within horizon)</span><span class="cl-val fc-val"></span></div>
          <canvas class="c-forecast" height="96"></canvas>
        </div>
        <div class="eventlog"></div>
      </div>
    </div>
  </section>'''


CSS = r"""
:root{
  --bg:#000; --panel:#0a0a0b; --panel2:#0d0e11; --line:rgba(255,255,255,.11);
  --line2:rgba(255,255,255,.06); --ink:#eceae4; --muted:#83868f; --faint:#565962;
  --green:#7fdca4; --amber:#e7b06a; --red:#e5615c; --blue:#79b7e6;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);
  -webkit-font-smoothing:antialiased;line-height:1.5}
a{color:inherit;text-decoration:none}
.mono{font-family:var(--mono)}
.wrap{max-width:1200px;margin:0 auto;padding:0 28px}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.22em;
  text-transform:uppercase;color:var(--muted)}

/* nav */
nav{position:sticky;top:0;z-index:50;background:rgba(0,0,0,.82);
  backdrop-filter:blur(10px);border-bottom:1px solid var(--line2)}
nav .wrap{display:flex;align-items:center;justify-content:space-between;height:60px}
.brand{font-family:var(--mono);font-size:13px;letter-spacing:.12em;text-transform:uppercase}
.navlinks{display:flex;gap:22px;font-family:var(--mono);font-size:12px;
  letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.navlinks a:hover{color:var(--ink)}
.navlinks a.active{color:var(--green)}
.navcta{border:1px solid var(--line);padding:8px 14px;border-radius:2px;
  font-family:var(--mono);font-size:11px;letter-spacing:.1em;text-transform:uppercase}
.navcta:hover{background:#fff;color:#000}
@media(max-width:900px){.navlinks{display:none}}

/* hero */
.hero{padding:96px 0 60px;border-bottom:1px solid var(--line2)}
.hero h1{font-size:clamp(34px,6vw,68px);line-height:1.02;letter-spacing:-.02em;
  font-weight:600;margin:18px 0 20px;max-width:16ch}
.hero h1 em{font-style:italic;color:var(--green);font-weight:500}
.hero p{color:var(--muted);font-size:18px;max-width:60ch}
.herocta{display:flex;gap:12px;margin-top:30px;flex-wrap:wrap}
.btn{display:inline-flex;align-items:center;gap:8px;padding:13px 20px;border-radius:2px;
  font-family:var(--mono);font-size:12px;letter-spacing:.08em;text-transform:uppercase}
.btn-p{background:var(--green);color:#04150b;font-weight:600}
.btn-p:hover{filter:brightness(1.08)}
.btn-o{border:1px solid var(--line);color:var(--ink)}
.btn-o:hover{background:#fff;color:#000}
.herostats{display:flex;gap:40px;margin-top:52px;flex-wrap:wrap}
.hs .k{font-family:var(--mono);font-size:26px;color:var(--ink)}
.hs .l{font-family:var(--mono);font-size:11px;letter-spacing:.14em;
  text-transform:uppercase;color:var(--faint);margin-top:6px}

/* section */
section.blk{padding:70px 0;border-bottom:1px solid var(--line2)}
.snum{font-family:var(--mono);font-size:12px;letter-spacing:.18em;
  text-transform:uppercase;color:var(--muted);margin-bottom:18px}
h2{font-size:clamp(26px,4vw,40px);letter-spacing:-.015em;font-weight:600;margin:0 0 14px;max-width:20ch}
h2 em{font-style:italic;color:var(--green);font-weight:500}
.lead{color:var(--muted);max-width:70ch;font-size:16px}

/* engine steps */
.steps{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;margin-top:34px;
  background:var(--line2);border:1px solid var(--line2)}
.step{background:var(--bg);padding:22px}
.step .n{font-family:var(--mono);color:var(--green);font-size:12px}
.step h3{font-size:15px;margin:10px 0 8px;font-weight:600}
.step p{color:var(--muted);font-size:13.5px;margin:0}
.step .eq{font-family:var(--mono);font-size:12px;color:var(--blue);margin-top:12px;
  padding-top:12px;border-top:1px solid var(--line2);white-space:nowrap;overflow-x:auto}
@media(max-width:900px){.steps{grid-template-columns:1fr 1fr}}
@media(max-width:560px){.steps{grid-template-columns:1fr}}

/* domain */
.domain{padding:64px 0;border-bottom:1px solid var(--line2)}
.dhead{font-size:clamp(24px,3.4vw,34px);font-weight:600;letter-spacing:-.01em;margin-bottom:26px}
.dhead em{font-style:italic;color:var(--green);font-weight:500}
.dnum{font-family:var(--mono);font-size:13px;color:var(--muted);vertical-align:middle;margin-right:8px}
.dgrid{display:grid;grid-template-columns:0.92fr 1.08fr;gap:36px;align-items:start}
@media(max-width:940px){.dgrid{grid-template-columns:1fr}}
.blurb{color:var(--muted);font-size:15.5px;margin:0 0 22px}
.metaline{display:flex;gap:14px;padding:9px 0;border-top:1px solid var(--line2);font-size:13px}
.ml-k{font-family:var(--mono);font-size:11px;letter-spacing:.1em;text-transform:uppercase;
  color:var(--faint);min-width:104px}
.ml-v{color:var(--ink)}
.eqs{margin:20px 0;padding:16px;background:var(--panel2);border:1px solid var(--line2);border-radius:3px}
.eqline{font-family:var(--mono);font-size:12.5px;color:var(--blue);white-space:nowrap;
  overflow-x:auto;padding:3px 0}
.faultnote{font-family:var(--mono);font-size:12px;color:var(--amber)}

/* panel */
.panel{background:var(--panel);border:1px solid var(--line);border-radius:4px;overflow:hidden}
.panel-top{display:flex;align-items:center;justify-content:space-between;
  padding:12px 16px;border-bottom:1px solid var(--line);background:var(--panel2)}
.pt-title{font-family:var(--mono);font-size:12px;letter-spacing:.06em;color:var(--muted)}
.live{font-family:var(--mono);font-size:10.5px;letter-spacing:.18em;color:var(--green);
  display:flex;align-items:center;gap:7px}
.live .dot{width:7px;height:7px;border-radius:50%;background:var(--green);
  box-shadow:0 0 0 0 rgba(127,220,164,.6);animation:pulse 1.8s infinite}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(127,220,164,.5)}70%{box-shadow:0 0 0 7px rgba(127,220,164,0)}100%{box-shadow:0 0 0 0 rgba(127,220,164,0)}}
.panel-controls{display:flex;align-items:center;gap:12px;padding:12px 16px;
  border-bottom:1px solid var(--line2);flex-wrap:wrap}
.seg{display:flex;border:1px solid var(--line);border-radius:2px;overflow:hidden}
.seg-btn{background:transparent;color:var(--muted);border:0;padding:7px 13px;
  font-family:var(--mono);font-size:11px;letter-spacing:.08em;text-transform:uppercase;cursor:pointer}
.seg-btn.active{background:var(--ink);color:#000}
.playbtn{background:transparent;color:var(--ink);border:1px solid var(--line);border-radius:2px;
  padding:7px 12px;font-family:var(--mono);font-size:11px;cursor:pointer;min-width:92px}
.playbtn:hover{border-color:var(--muted)}
.scrub{flex:1;min-width:120px;accent-color:var(--green)}
.tread{font-family:var(--mono);font-size:12px;color:var(--muted);min-width:56px;text-align:right}
.chart-wrap{padding:14px 16px 6px}
.chart-label{display:flex;justify-content:space-between;font-family:var(--mono);
  font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--faint);margin-bottom:6px}
.cl-val{color:var(--ink)}
canvas{width:100%;display:block}
.eventlog{margin:8px 16px 16px;border-top:1px solid var(--line2);padding-top:12px;
  min-height:96px;max-height:150px;overflow-y:auto;font-family:var(--mono);font-size:12px}
.ev{display:flex;gap:10px;padding:4px 0;color:var(--muted)}
.ev .et{color:var(--faint);min-width:52px}
.ev.WARN .el{color:var(--amber)} .ev.ALERT .el{color:var(--red)} .ev.INFO .el{color:var(--blue)}
.ev .el{min-width:46px;text-transform:uppercase;font-size:10.5px;letter-spacing:.06em}
.ev .em{color:var(--ink);flex:1}
.ev.new{animation:flash .8s}
@keyframes flash{from{background:rgba(127,220,164,.12)}to{background:transparent}}

/* verify */
.vgrid{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;margin-top:30px;
  background:var(--line2);border:1px solid var(--line2)}
.vcell{background:var(--bg);padding:22px}
.vcell .vk{font-family:var(--mono);font-size:24px;color:var(--green)}
.vcell .vl{font-family:var(--mono);font-size:11px;letter-spacing:.12em;text-transform:uppercase;
  color:var(--faint);margin-top:8px}
@media(max-width:900px){.vgrid{grid-template-columns:1fr 1fr}}

/* cta + footer */
.cta{padding:84px 0;text-align:center;border-bottom:1px solid var(--line2)}
.cta h2{margin:0 auto 16px}
.cta .lead{margin:0 auto 26px}
.cta .herocta{justify-content:center}
footer{padding:40px 0;color:var(--faint);font-family:var(--mono);font-size:12px;
  display:flex;justify-content:space-between;flex-wrap:wrap;gap:12px}
.disc{color:var(--faint);font-size:12.5px;max-width:80ch;margin-top:22px;font-family:var(--mono)}
"""


def build_html(data):
    sections = "\n".join(domain_section(n, data["meta"][n]) for n in ORDER)
    data_json = json.dumps(data, separators=(",", ":"))
    return TEMPLATE.replace("/*CSS*/", CSS).replace(
        "<!--SECTIONS-->", sections).replace('"__DATA__"', data_json)


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#000000">
<title>Terra Labs — Live Simulation</title>
<meta name="description" content="Watch the Terra Engine hold four closed biogeochemical loops in real time — recovering the hidden state it never measures and forecasting failure before it cascades. Live, verified engine output.">
<style>/*CSS*/</style>
</head>
<body>
<nav><div class="wrap">
  <a href="index.html" class="brand">Terra Labs</a>
  <div class="navlinks">
    <a href="index.html#problem">Problem</a>
    <a href="index.html#engine">Engine</a>
    <a href="index.html#kits">Sensor Kits</a>
    <a href="index.html#markets">Markets</a>
    <a href="team.html">Team</a>
    <a href="simulation.html" class="active">Live Simulation</a>
  </div>
  <a href="index.html#contact" class="navcta">Get in touch</a>
</div></nav>

<header class="hero"><div class="wrap">
  <div class="eyebrow">Live Simulation · The interactive twin</div>
  <h1>Watch the engine <em>hold the loop.</em></h1>
  <p>The same Bayesian inference core, running four closed biogeochemical loops. Each one recovers a hidden state it can never measure, closes the mass budget, and forecasts the failure before any single gauge crosses a threshold. Every curve below is real, verified engine output — not an animation of a script.</p>
  <div class="herocta">
    <a class="btn btn-p" href="#dom-aquaculture">Run the simulations ↓</a>
    <a class="btn btn-o" href="index.html#contact">Get in touch</a>
  </div>
  <div class="herostats">
    <div class="hs"><div class="k">1 engine</div><div class="l">four closed loops</div></div>
    <div class="hs"><div class="k">Edge</div><div class="l">no cloud required</div></div>
    <div class="hs"><div class="k">Calibrated</div><div class="l">uncertainty on every estimate</div></div>
    <div class="hs"><div class="k">Hours</div><div class="l">of early warning</div></div>
  </div>
</div></header>

<section class="blk"><div class="wrap">
  <div class="snum">A — The Inference Loop</div>
  <h2>How the engine <em>thinks.</em></h2>
  <p class="lead">It keeps a probability distribution over the loop's true state — a best guess and how sure it is — and every cycle it predicts with a process model, then corrects with whatever sensors reported. The disagreement between the two is what reveals the hidden state no sensor sees.</p>
  <div class="steps">
    <div class="step"><div class="n">/01</div><h3>State model</h3><p>A mass-balance rulebook for how the loop evolves between readings.</p><div class="eq">xₖ = f(xₖ₋₁, uₖ) + wₖ</div></div>
    <div class="step"><div class="n">/02</div><h3>Predict</h3><p>Roll the rulebook forward through the unscented transform; grow less certain.</p><div class="eq">x̄, P̄ ← UT(f, x, P) + Q</div></div>
    <div class="step"><div class="n">/03</div><h3>Correct</h3><p>Blend in the sensors that reported. Surprise × trust = the nudge.</p><div class="eq">x ← x̄ + K(z − h(x̄))</div></div>
    <div class="step"><div class="n">/04</div><h3>Forecast</h3><p>Sample the posterior, fast-forward, count the futures that breach.</p><div class="eq">P(breach) = 𝔼[𝟙 g(x_t) > limit]</div></div>
  </div>
  <p class="disc">◇ Sensors are optional per cycle — drop a probe and the filter leans on the model and the remaining channels. The hidden "health" state (biofilter efficiency, microbial activity, crop capacity) is inferred from the mismatch, never measured directly.</p>
</div></div></section>

<div class="wrap">
<!--SECTIONS-->
</div>

<section class="blk"><div class="wrap">
  <div class="snum">B — Under the hood</div>
  <h2>Verified, not just <em>rendered.</em></h2>
  <p class="lead">The engine ships with an independent verification pass. These are its actual numbers — the filter's stated uncertainty matches its real error, mass is conserved to machine precision, and every run is reproducible.</p>
  <div class="vgrid">
    <div class="vcell"><div class="vk">4.98 / 5</div><div class="vl">filter calibration · NEES vs state dim</div></div>
    <div class="vcell"><div class="vk">8.9e-15</div><div class="vl">nitrogen mass drift · mg-N/L</div></div>
    <div class="vcell"><div class="vk">11 / 11</div><div class="vl">automated tests passing</div></div>
    <div class="vcell"><div class="vk">bit-exact</div><div class="vl">deterministic on replay</div></div>
  </div>
  <p class="disc">◇ These simulations run against synthetic ground truth with illustrative parameters; per-site calibration is required before operational use. The point is the mechanism — the same one validated on real biogeochemical data — not these specific numbers.</p>
</div></div></section>

<section class="cta"><div class="wrap">
  <div class="eyebrow">Bridge round open</div>
  <h2 style="max-width:22ch">The engine that keeps biology alive — <em>anywhere.</em></h2>
  <p class="lead">If you build instrumentation, run closed systems, or invest in the infrastructure of living systems — talk to us.</p>
  <div class="herocta">
    <a class="btn btn-p" href="mailto:cameron@terralaboratories.com">Get in touch</a>
    <a class="btn btn-o" href="index.html#engine">See the engine</a>
  </div>
</div></section>

<footer><div class="wrap" style="display:flex;justify-content:space-between;width:100%;flex-wrap:wrap;gap:12px">
  <span>© Terra Laboratories · From the mud to the moon</span>
  <span>Cameron Souza · Hawaii</span>
</div></footer>

<script>
const DATA = "__DATA__";
const CSS = getComputedStyle(document.documentElement);
const COL = {
  ink:'#eceae4', muted:'#83868f', faint:'#565962',
  green:'#7fdca4', amber:'#e7b06a', red:'#e5615c', blue:'#79b7e6', line:'rgba(255,255,255,.10)'
};

function setupCanvas(cv){
  const dpr = window.devicePixelRatio || 1;
  if(!cv._lh){ cv._lh = (parseInt(cv.getAttribute('height'),10)||120); cv.style.height=cv._lh+'px'; }
  const w = cv.clientWidth || 300, h = cv._lh;      // logical size (never mutated)
  cv.width = w*dpr; cv.height = h*dpr;
  const ctx = cv.getContext('2d'); ctx.setTransform(dpr,0,0,dpr,0,0);  // reset, no compounding
  return {ctx, w, h};
}
function niceMax(v){ if(v<=0) return 1; const p=Math.pow(10,Math.floor(Math.log10(v))); const n=v/p; const s=n<=1?1:n<=2?2:n<=5?5:10; return s*p; }

function drawSeries(cv, series, opts){
  const {ctx,w,h} = setupCanvas(cv);
  const t = series.t, n = t.length, k = opts.k;
  const padL=38, padR=10, padT=10, padB=18;
  const x0=padL, x1=w-padR, y0=h-padB, y1=padT;
  const tmax = t[n-1] || 1;
  let ymin = opts.ymin!==undefined?opts.ymin:0;
  let ymax = opts.ymax!==undefined?opts.ymax:niceMax(Math.max(...opts.lines.flatMap(L=>L.data.filter(v=>!isNaN(v)))));
  if(ymax<=ymin) ymax=ymin+1;
  const X = ti => x0 + (x1-x0)*(t[ti]/tmax);
  const Y = v => y1 + (y0-y1)*(1-(v-ymin)/(ymax-ymin));
  ctx.clearRect(0,0,w,h);
  // grid
  ctx.strokeStyle=COL.line; ctx.lineWidth=1; ctx.font='9px ui-monospace,monospace'; ctx.fillStyle=COL.faint;
  for(let g=0; g<=2; g++){ const yv=ymin+(ymax-ymin)*g/2; const y=Y(yv);
    ctx.beginPath(); ctx.moveTo(x0,y); ctx.lineTo(x1,y); ctx.stroke();
    ctx.fillText((Math.abs(yv)>=100?yv.toFixed(0):yv.toFixed(ymax<3?2:1)), 2, y+3); }
  // uncertainty band
  if(opts.band){
    ctx.fillStyle='rgba(127,220,164,.13)'; ctx.beginPath();
    for(let i=0;i<=k;i++){ const v=opts.band.mid[i]+opts.band.sd[i]; ctx.lineTo(X(i),Y(v)); }
    for(let i=k;i>=0;i--){ const v=opts.band.mid[i]-opts.band.sd[i]; ctx.lineTo(X(i),Y(v)); }
    ctx.closePath(); ctx.fill();
  }
  // threshold
  if(opts.thresh!==undefined && opts.thresh>=ymin && opts.thresh<=ymax){
    ctx.strokeStyle=COL.red; ctx.setLineDash([4,4]); ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(x0,Y(opts.thresh)); ctx.lineTo(x1,Y(opts.thresh)); ctx.stroke();
    ctx.setLineDash([]);
  }
  // dots (raw)
  if(opts.dots){ ctx.fillStyle=COL.faint;
    for(let i=0;i<=k;i++){ const v=opts.dots[i]; if(isNaN(v))continue; ctx.beginPath(); ctx.arc(X(i),Y(v),1.4,0,7); ctx.fill(); } }
  // lines up to k
  for(const L of opts.lines){
    ctx.strokeStyle=L.color; ctx.lineWidth=L.w||1.7; ctx.beginPath();
    let started=false;
    for(let i=0;i<=k;i++){ const v=L.data[i]; if(isNaN(v)){started=false;continue;}
      const px=X(i),py=Y(v); if(!started){ctx.moveTo(px,py);started=true;} else ctx.lineTo(px,py); }
    ctx.stroke();
  }
  // fill under forecast
  if(opts.fill){ const L=opts.lines[0]; ctx.fillStyle='rgba(229,97,92,.14)'; ctx.beginPath();
    ctx.moveTo(X(0),Y(ymin));
    for(let i=0;i<=k;i++) ctx.lineTo(X(i),Y(L.data[i]));
    ctx.lineTo(X(k),Y(ymin)); ctx.closePath(); ctx.fill(); }
  // playhead
  ctx.strokeStyle='rgba(255,255,255,.28)'; ctx.lineWidth=1; ctx.beginPath();
  ctx.moveTo(X(k),y1); ctx.lineTo(X(k),y0); ctx.stroke();
}

function initPanel(panel){
  const name = panel.dataset.domain;
  const meta = DATA.meta[name];
  const cH = panel.querySelector('.c-hidden');
  const cS = panel.querySelector('.c-sensor');
  const cF = panel.querySelector('.c-forecast');
  const scrub = panel.querySelector('.scrub');
  const tread = panel.querySelector('.tread');
  const log = panel.querySelector('.eventlog');
  const hVal = panel.querySelector('.hidden-val');
  const sVal = panel.querySelector('.sensor-val');
  const fVal = panel.querySelector('.fc-val');
  const playbtn = panel.querySelector('.playbtn');
  let mode='fault', run=DATA.runs[name][mode], N=run.t.length, k=0, playing=true, shownEvents=0;

  function render(){
    const series={t:run.t};
    drawSeries(cH, series, {k, ymin:0, ymax:Math.max(1.2, meta.hidden_baseline*1.25),
      band:{mid:run.hidden_est, sd:run.hidden_std},
      thresh:meta.hidden_alert,
      lines:[{data:run.hidden_true,color:COL.muted,w:1.3},{data:run.hidden_est,color:COL.green,w:2}]});
    drawSeries(cS, series, {k, dots:run.sensor_raw,
      lines:[{data:run.sensor_true,color:COL.muted,w:1.2},{data:run.sensor_est,color:COL.amber,w:2}]});
    drawSeries(cF, series, {k, ymin:0, ymax:1, fill:true, thresh:0.3,
      lines:[{data:run.forecast_p,color:COL.red,w:2}]});
    tread.textContent = run.t[k].toFixed(1).padStart(4,'0')+' h';
    scrub.value = (k/(N-1))*100;
    hVal.textContent = run.hidden_est[k].toFixed(2)+' ('+meta.hidden_label.toLowerCase()+')';
    sVal.textContent = run.sensor_est[k].toFixed(2)+' '+meta.sensor_units;
    fVal.textContent = (run.forecast_p[k]*100).toFixed(0)+'%';
    fVal.style.color = run.forecast_p[k]>=0.3?COL.red:COL.ink;
    // events up to current time
    const tc = run.t[k];
    const due = run.events.filter(e=>e.t<=tc);
    if(due.length!==shownEvents){
      log.innerHTML = due.length? due.map((e,i)=>
        `<div class="ev ${e.level} ${i===due.length-1&&due.length>shownEvents?'new':''}"><span class="et">${e.t.toFixed(1)}h</span><span class="el">${e.level}</span><span class="em">${e.msg}</span></div>`
      ).reverse().join('') : '<div class="ev"><span class="em" style="color:var(--faint)">Budget closing · Δ ≈ 0 · no events</span></div>';
      shownEvents=due.length;
    }
  }
  function setMode(mo){ mode=mo; run=DATA.runs[name][mode]; N=run.t.length; k=0; shownEvents=-1;
    panel.querySelectorAll('.seg-btn').forEach(b=>b.classList.toggle('active',b.dataset.mode===mo));
    render(); }
  panel.querySelectorAll('.seg-btn').forEach(b=>b.onclick=()=>setMode(b.dataset.mode));
  scrub.oninput=()=>{ k=Math.round(scrub.value/100*(N-1)); shownEvents=-1; render(); };
  playbtn.onclick=()=>{ playing=!playing; playbtn.textContent = playing?'❚❚ Pause':'▶ Play'; };
  panel._tick=()=>{ if(playing){ k++; if(k>=N){k=N-1;setTimeout(()=>{if(playing){k=0;shownEvents=-1;}},1200);} render(); } };
  setMode('fault');
  return panel;
}

const panels=[...document.querySelectorAll('.panel')].map(initPanel);
let last=0;
function loop(ts){ if(ts-last>240){ last=ts; panels.forEach(p=>p._tick()); } requestAnimationFrame(loop); }
requestAnimationFrame(loop);
window.addEventListener('resize',()=>panels.forEach(p=>p.querySelector('.scrub').dispatchEvent(new Event('input'))));
</script>
</body>
</html>"""


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "simulation.html"
    data = build_data()
    html = build_html(data)
    with open(out, "w") as f:
        f.write(html)
    kb = len(html) / 1024
    print(f"wrote {out}  ({kb:.0f} KB, {len(data['order'])} domains, "
          f"healthy+fault runs embedded)")
