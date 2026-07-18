"""The Terra platform server: real engine, real storage, real config.

`terra serve` starts a stdlib HTTP server that runs the actual engine, persists
configuration and the estimate history to disk, ingests real logged sensor data,
runs real calibration, and serves the web console from the same origin. Nothing
is simulated in the browser and no numbers are fabricated — every value the
console shows is computed by the Python engine here and stored under
``$TERRA_HOME`` (default ``~/.terra``).

Data source, in order of preference:
  1. a hardware ``SensorDriver`` (when wired — see terra/node/driver.py),
  2. a real logged CSV you ingest (POST /api/ingest or `terra serve --log`),
  3. the repo's sample log, so the engine has real rows to run on out of the box.

Endpoints:
  GET  /                -> web console
  GET  /api/status      -> domain, running, cycles, source, autonomy, offline
  GET  /api/config      -> persisted config (domain, channels, hardware, params)
  POST /api/config      -> update + persist + rebuild engine
  GET  /api/state       -> latest engine estimate (channels, hidden, risks, events)
  GET  /api/history?n=  -> last n persisted snapshots (for the live chart)
  GET  /api/export      -> the stored history as CSV (download)
  POST /api/ingest      -> upload a CSV log; engine replays it for real
  POST /api/calibrate   -> run the real NUTS fit on the active log; persist params
  POST /api/control     -> {"autonomy": bool}
  POST /api/offline     -> {"offline": bool}   (edge-autonomous mode)
"""
from __future__ import annotations

import csv
import io
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .core import TerraEngine, EngineConfig
from .domains import DOMAINS
from .control import Controller, policy_for

HOME = os.environ.get("TERRA_HOME", os.path.expanduser("~/.terra"))
DATA_DIR = os.path.join(HOME, "data")
CONFIG_PATH = os.path.join(HOME, "config.json")
WEB_DIR = os.path.join(os.path.dirname(__file__), "web")

# nominal process input per domain when a log doesn't carry one
NOMINAL_U = {"aquaculture": [0.22, 1.0], "soil": [0.5, 0.02, 0.0],
             "bioremediation": [1.0], "blss": [1.0, 0.0]}

DEFAULT_CONFIG = {
    "domain": "aquaculture",
    "channels": None,                       # None = all; else list of enabled keys
    "hardware": {"board": "Raspberry Pi 4 (4 GB)", "tier": "Field / mid-grade"},
    "params": {},                           # calibrated parameter overrides
    "autonomy": False,
    "speed": 0.4,
}


def _ensure_home():
    os.makedirs(DATA_DIR, exist_ok=True)


def load_config() -> dict:
    _ensure_home()
    cfg = dict(DEFAULT_CONFIG)
    if os.path.exists(CONFIG_PATH):
        try:
            cfg.update(json.load(open(CONFIG_PATH)))
        except Exception:
            pass
    return cfg


def save_config(cfg: dict) -> None:
    _ensure_home()
    tmp = CONFIG_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=2)
    os.replace(tmp, CONFIG_PATH)


def parse_log(text: str, spec) -> list:
    """Parse a CSV log into engine rows: [{t, meas:{ch:val}, u:[...]}].

    Time column is the first column whose name contains 'time'/'t'; channel
    columns are matched to the spec's channel names; an optional input column
    (e.g. excretion_kg_h) feeds u, else the domain nominal input is used.
    """
    reader = csv.DictReader(io.StringIO(text))
    cols = reader.fieldnames or []
    tcol = next((c for c in cols if c.lower() in ("timestamp", "time", "t", "hours")), cols[0])
    ucol = next((c for c in cols if any(k in c.lower()
                 for k in ("excretion", "feed", "input", "dose", "u_"))), None)
    chan_keys = [k for k in spec.channels if k in cols]
    nominal = list(NOMINAL_U.get(spec.name, [0.0]))
    rows = []
    t0 = None
    for i, r in enumerate(reader):
        try:
            tv = r.get(tcol, "")
            t = float(tv) if _isnum(tv) else float(i) * 0.25
        except Exception:
            t = float(i) * 0.25
        if t0 is None:
            t0 = t
        meas = {}
        for k in chan_keys:
            v = r.get(k, "")
            if v not in ("", None) and _isnum(v):
                meas[k] = float(v)
        u = list(nominal)
        if ucol and _isnum(r.get(ucol, "")):
            u[0] = float(r[ucol])
        rows.append({"t": t - t0, "meas": meas, "u": u})
    return rows


def _isnum(v) -> bool:
    try:
        float(v)
        return True
    except Exception:
        return False


class Platform:
    def __init__(self, log_path: str | None = None):
        _ensure_home()
        self.cfg = load_config()
        self.autonomy = bool(self.cfg.get("autonomy", False))
        self.offline = False
        self.speed = float(self.cfg.get("speed", 0.4))
        self.source_name = "none"
        self.rows: list = []
        self.i = 0
        self.cycles = 0
        self.latest = None
        self.rec = None
        self.hist_path = os.path.join(DATA_DIR, "history.jsonl")
        self.active_log = log_path or os.path.join(DATA_DIR, "active.csv")
        self._build_engine()
        self._load_initial_source()
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    # ---- engine / config ----
    def _build_engine(self):
        self.spec = DOMAINS[self.cfg["domain"]].build_spec()
        self._apply_params()
        self.engine = TerraEngine(self.spec, EngineConfig(
            forecast_horizon_h=12, forecast_samples=120,
            outlier_sigma=5.0))
        try:
            self.controller = Controller(self.spec, self.engine.cfg,
                                          policy_for(self.cfg["domain"]),
                                          authorized=self.autonomy)
        except Exception:
            self.controller = None

    def _apply_params(self):
        import dataclasses
        p = self.cfg.get("params") or {}
        good = {k: v for k, v in p.items() if hasattr(self.spec.params, k)}
        if good:
            self.spec = dataclasses.replace(
                self.spec, params=dataclasses.replace(self.spec.params, **good))

    def _enabled_channel(self, key: str) -> bool:
        ch = self.cfg.get("channels")
        return True if ch is None else (key in ch)

    def set_config(self, patch: dict):
        with self.lock():
            for k in ("domain", "channels", "hardware", "params", "autonomy", "speed"):
                if k in patch:
                    self.cfg[k] = patch[k]
            self.autonomy = bool(self.cfg.get("autonomy", self.autonomy))
            save_config(self.cfg)
            self._build_engine()
            self.i = 0
            self.cycles = 0
            self.latest = None

    # ---- data source ----
    def _load_initial_source(self):
        if os.path.exists(self.active_log):
            try:
                self._load_text(open(self.active_log).read(), os.path.basename(self.active_log))
                return
            except Exception:
                pass
        # fall back to the repo sample so the engine runs on real rows
        sample = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              "data", "aquaculture_sample.csv")
        if os.path.exists(sample) and self.cfg["domain"] == "aquaculture":
            try:
                self._load_text(open(sample).read(), "aquaculture_sample.csv")
            except Exception:
                pass

    def _load_text(self, text: str, name: str):
        rows = parse_log(text, self.spec)
        if not rows:
            raise ValueError("no usable rows")
        self.rows = rows
        self.i = 0
        self.source_name = f"replay:{name}"

    def ingest(self, text: str, name: str = "uploaded.csv"):
        _ensure_home()
        with open(self.active_log, "w") as f:
            f.write(text)
        with self.lock():
            self._load_text(text, name)
            self.cycles = 0

    def lock(self):
        if not hasattr(self, "_lock"):
            self._lock = threading.Lock()
        return self._lock

    # ---- run loop ----
    def _loop(self):
        while self.running:
            with self.lock():
                self._step()
            time.sleep(self.speed)

    def _step(self):
        if not self.rows:
            return
        if self.i >= len(self.rows):
            self.i = 0                       # loop the log
        row = self.rows[self.i]
        dt = 0.25
        if self.i + 1 < len(self.rows):
            dt = max(self.rows[self.i + 1]["t"] - row["t"], 1e-3)
        meas = {k: v for k, v in row["meas"].items() if self._enabled_channel(k)}
        # an enacted controller action overrides the input on the next step
        u = self._override_u if getattr(self, "_override_u", None) is not None else row["u"]
        self._override_u = None
        est = self.engine.step(row["t"], dt, meas, u,
                               u_forecast=NOMINAL_U.get(self.cfg["domain"]))
        if self.controller is not None:
            self.controller.authorized = self.autonomy
            self.rec = self.controller.recommend(est, row["u"])
            if self.rec and self.rec.enacted:
                self._override_u = self.rec.control_u
        self.latest = est
        self.i += 1
        self.cycles += 1
        self._persist(est)

    def _persist(self, est):
        rec = {"t": est.t, "cycles": self.cycles,
               "channels": {k: (float(est.x[ch.state]) if ch.state is not None
                                else float(ch.obs(est.x)))
                            for k, ch in self.spec.channels.items()},
               "hidden": est.hidden, "hidden_std": est.hidden_std, "nis": est.nis}
        try:
            with open(self.hist_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:
            pass

    # ---- snapshots ----
    def status(self):
        return {"domain": self.cfg["domain"], "running": bool(self.rows),
                "cycles": self.cycles, "source": self.source_name,
                "autonomy": self.autonomy, "offline": self.offline,
                "hardware": self.cfg.get("hardware", {})}

    def config(self):
        d = self.spec
        return {"domain": self.cfg["domain"],
                "channels_enabled": self.cfg.get("channels"),
                "hidden": d.hidden,
                "channels": [{"key": k, "state": ch.state, "noise": ch.noise}
                             for k, ch in d.channels.items()],
                "safety": [{"name": s.name, "limit": s.limit,
                            "direction": s.direction, "units": s.units} for s in d.safety],
                "hardware": self.cfg.get("hardware", {}),
                "params": self.cfg.get("params", {})}

    def state(self):
        with self.lock():
            est = self.latest
            if est is None:
                return {"ready": False, "source": self.source_name}
            d = self.spec
            channels = {k: (float(est.x[ch.state]) if ch.state is not None
                            else float(ch.obs(est.x))) for k, ch in d.channels.items()}
            risks = {n: {"p": r["p"], "t_cross": r["t_cross"], "limit": r["limit"],
                         "direction": r["direction"], "units": r["units"]}
                     for n, r in est.risks.items()}
            events = [[float(t), lv, m] for (t, lv, m) in self.engine.events[-24:]]
            return {"ready": True, "t": est.t, "cycles": self.cycles,
                    "source": self.source_name, "channels": channels,
                    "used": est.used_channels, "hidden": est.hidden,
                    "hidden_std": est.hidden_std, "nis": est.nis,
                    "budget": est.budget_residual, "risks": risks, "events": events,
                    "recommendation": (self.rec.message() if self.rec else None),
                    "autonomy": self.autonomy, "offline": self.offline}

    def history(self, n=200):
        if not os.path.exists(self.hist_path):
            return []
        try:
            lines = open(self.hist_path).read().splitlines()[-int(n):]
            return [json.loads(x) for x in lines if x.strip()]
        except Exception:
            return []

    def export_csv(self):
        recs = self.history(100000)
        if not recs:
            return "t,hidden,nis\n"
        chans = list(recs[-1].get("channels", {}).keys())
        out = io.StringIO()
        w = csv.writer(out)
        w.writerow(["t"] + chans + ["hidden", "nis"])
        for r in recs:
            w.writerow([r.get("t")] + [r.get("channels", {}).get(c) for c in chans]
                       + [r.get("hidden"), r.get("nis")])
        return out.getvalue()

    def calibrate(self):
        try:
            from .calibrate import HAS_JAX
        except Exception:
            HAS_JAX = False
        if not HAS_JAX:
            return {"error": "calibration needs the optional extra: "
                             "pip install terra-engine[calibrate]"}
        if not self.rows:
            return {"error": "no active log to calibrate on — ingest a CSV first"}
        from .calibrate import fit_nuts
        times = [r["t"] for r in self.rows]
        u_series = [r["u"] for r in self.rows]
        meas = [r["meas"] for r in self.rows]
        res = fit_nuts(times, u_series, meas, self.spec,
                       num_warmup=300, num_samples=300)
        med = res.medians()
        diag = res.diagnostics()
        drift = res.drift()
        with self.lock():
            self.cfg["params"] = {k: v for k, v in med.items()
                                  if hasattr(self.spec.params, k)}
            save_config(self.cfg)
            self._build_engine()
        return {"medians": med, "diagnostics": diag, "drift": drift,
                "converged": res.converged()}


def make_handler(pf: Platform):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            if isinstance(body, (dict, list)):
                body = json.dumps(body).encode()
            elif isinstance(body, str):
                body = body.encode()
            if body:
                self.wfile.write(body)

        def do_OPTIONS(self):
            self._send(204, b"")

        def do_GET(self):
            from urllib.parse import urlparse, parse_qs
            u = urlparse(self.path)
            p, q = u.path, parse_qs(u.query)
            if p == "/api/status":
                return self._send(200, pf.status())
            if p == "/api/config":
                return self._send(200, pf.config())
            if p == "/api/state":
                return self._send(200, pf.state())
            if p == "/api/history":
                return self._send(200, pf.history(int((q.get("n", ["200"])[0]))))
            if p == "/api/export":
                self.send_response(200)
                self.send_header("Content-Type", "text/csv")
                self.send_header("Content-Disposition", "attachment; filename=terra-history.csv")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                return self.wfile.write(pf.export_csv().encode())
            if p in ("/", "/index.html"):
                f = os.path.join(WEB_DIR, "index.html")
                if os.path.exists(f):
                    return self._send(200, open(f, "rb").read(), "text/html; charset=utf-8")
                return self._send(404, {"error": "console not bundled"})
            return self._send(404, {"error": "not found"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(n) if n else b""
            p = self.path.split("?")[0]
            if p == "/api/ingest":
                pf.ingest(raw.decode("utf-8", "replace"), "uploaded.csv")
                return self._send(200, pf.status())
            try:
                data = json.loads(raw or b"{}")
            except Exception:
                data = {}
            if p == "/api/config":
                pf.set_config(data)
                return self._send(200, pf.config())
            if p == "/api/control":
                pf.autonomy = bool(data.get("autonomy"))
                pf.cfg["autonomy"] = pf.autonomy
                save_config(pf.cfg)
                return self._send(200, pf.status())
            if p == "/api/offline":
                pf.offline = bool(data.get("offline"))
                return self._send(200, pf.status())
            if p == "/api/calibrate":
                return self._send(200, pf.calibrate())
            return self._send(404, {"error": "not found"})
    return H


def serve(domain: str = None, port: int = 8700, autonomy: bool = False,
          host: str = "0.0.0.0", log: str = None) -> int:
    if domain or autonomy:
        cfg = load_config()
        if domain:
            cfg["domain"] = domain
        if autonomy:
            cfg["autonomy"] = True
        save_config(cfg)
    pf = Platform(log_path=log)
    httpd = ThreadingHTTPServer((host, port), make_handler(pf))
    print(f"Terra platform on http://localhost:{port}  (console + /api)")
    print(f"  home={HOME}  domain={pf.cfg['domain']}  source={pf.source_name}")
    print("  real engine, real storage — ingest your logs at /api/ingest.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pf.running = False
        httpd.shutdown()
    return 0
