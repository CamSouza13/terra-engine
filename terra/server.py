"""Platform HTTP server.

``terra serve`` starts a standard-library HTTP server that runs the engine,
persists configuration and estimate history to disk, ingests logged sensor data,
runs calibration, and serves the web console from the same origin. Every value the
console shows is computed here and stored under ``$TERRA_HOME`` (default
``~/.terra``); nothing is computed in the browser.

Data source, in order of preference:
  1. a hardware ``SensorDriver`` (see terra/node/driver.py),
  2. a logged CSV ingested via the API or ``terra serve --log``,
  3. the bundled sample log, only when ``TERRA_DEMO`` is set.

Endpoints:
  GET  /                -> web console
  GET  /api/status      -> domain, running, cycles, source, autonomy, offline
  GET  /api/config      -> persisted config (domain, channels, hardware, params)
  POST /api/config      -> update + persist + rebuild engine
  GET  /api/state       -> latest engine estimate (channels, hidden, risks, events)
  GET  /api/history?n=  -> last n persisted snapshots (for the live chart)
  GET  /api/export      -> the stored history as CSV (download)
  POST /api/ingest      -> upload a CSV log for the engine to replay (auth required)
  POST /api/calibrate   -> run the NUTS fit on the active log; persist params
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
from . import accounts as acc
from . import alerts as alertmod
from . import registry as reg
from . import billing
from . import support
from . import audit
from . import reports

HOME = os.environ.get("TERRA_HOME", os.path.expanduser("~/.terra"))
DATA_DIR = os.path.join(HOME, "data")
CONFIG_PATH = os.path.join(HOME, "config.json")
WEB_DIR = os.path.join(os.path.dirname(__file__), "web")

# auth is off for local single-user use; set TERRA_AUTH=1 for the hosted platform
AUTH_ENFORCED = bool(os.environ.get("TERRA_AUTH"))
LOCAL_USER = {"user_id": 0, "email": "local", "role": "owner", "workspace_id": 0,
              "workspace": "local", "plan": "pro", "raw_plan": "pro", "trial_ends": None}
# the workspace/node this server instance reports under (a deployed single node
# sets these; locally they default to the built-in workspace 0 / node "local")
LOCAL_WS = int(os.environ.get("TERRA_WORKSPACE", "0"))
LOCAL_NODE = os.environ.get("TERRA_NODE", "local")

# request hardening config
MAX_BODY = int(os.environ.get("TERRA_MAX_BODY", str(2 * 1024 * 1024)))  # 2 MB default
# CORS: comma-separated allowlist of origins; empty means same-origin only (no ACAO)
CORS_ORIGINS = {o.strip() for o in os.environ.get("TERRA_CORS_ORIGINS", "").split(",") if o.strip()}
# billing stub (free plan set) only allowed when explicitly opted in
ALLOW_STUB_BILLING = bool(os.environ.get("TERRA_ALLOW_STUB_BILLING"))

# generic in-memory rate limiter: buckets of recent event timestamps per key
_RL: dict = {}
LOGIN_MAX_FAILS = 6
LOGIN_WINDOW = 300.0


def _rl_blocked(key: str, limit: int, window: float) -> bool:
    now = time.time()
    hits = [t for t in _RL.get(key, []) if now - t < window]
    _RL[key] = hits
    return len(hits) >= limit


def _rl_hit(key: str):
    _RL.setdefault(key, []).append(time.time())


def _login_blocked(key: str) -> bool:
    return _rl_blocked("login:" + key, LOGIN_MAX_FAILS, LOGIN_WINDOW)


def _login_fail(key: str):
    _rl_hit("login:" + key)

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
        self.workspace_id = LOCAL_WS
        self.node_id = LOCAL_NODE
        self._last_alert_eval = 0.0
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
        # No simulation or demo data: the platform starts empty and only runs on
        # real ingested logs or live node reports. Set TERRA_DEMO=1 to opt into the
        # bundled sample for a local walkthrough.
        if not os.environ.get("TERRA_DEMO"):
            self.source_name = "none"
            return
        sample = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              "data", "aquaculture_sample.csv")
        if os.path.exists(sample) and self.cfg["domain"] == "aquaculture":
            try:
                self._load_text(open(sample).read(), "sample (TERRA_DEMO)")
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
        self._watch(est)

    def _metrics(self, est) -> dict:
        channels = {k: (float(est.x[ch.state]) if ch.state is not None
                        else float(ch.obs(est.x))) for k, ch in self.spec.channels.items()}
        risks = {n: r["p"] for n, r in est.risks.items()}
        hidden = est.hidden
        h0 = next(iter(hidden.values())) if isinstance(hidden, dict) and hidden else None
        return {"channels": channels, "risks": risks, "nis": est.nis, "hidden": h0}

    def _watch(self, est):
        """Evaluate alert rules and refresh this node's registry heartbeat."""
        now = time.time()
        if now - self._last_alert_eval < 2.0:   # don't hammer the DB every step
            return
        self._last_alert_eval = now
        try:
            m = self._metrics(est)
            reg.touch_node(self.node_id, self.workspace_id, domain=self.cfg["domain"],
                           status={"cycles": self.cycles, "nis": est.nis,
                                   "autonomy": self.autonomy, "offline": self.offline})
            alertmod.evaluate(self.workspace_id, self.node_id, m)
            reg.append_history(self.node_id, m)
        except Exception:
            pass

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

    def config_bundle(self):
        """The provisioning bundle a node applies to run this configuration —
        domain, enabled channels, calibrated params, safety limits, and speed."""
        d = self.spec
        return {"domain": self.cfg["domain"],
                "channels": self.cfg.get("channels"),
                "params": self.cfg.get("params") or {},
                "speed": float(self.cfg.get("speed", 0.4)),
                "hidden": d.hidden,
                "safety": [{"name": s.name, "limit": s.limit,
                            "direction": s.direction, "units": s.units} for s in d.safety],
                "generated": time.time()}

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

        def _cors(self):
            # same-origin console needs no CORS; only echo an explicitly allowlisted origin
            origin = self.headers.get("Origin")
            if origin and origin in CORS_ORIGINS:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers",
                                 "Content-Type, Authorization, X-Node-Key, X-API-Key")

        def _send(self, code, body, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self._cors()
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            if isinstance(body, (dict, list)):
                body = json.dumps(body).encode()
            elif isinstance(body, str):
                body = body.encode()
            if body:
                self.wfile.write(body)

        def do_OPTIONS(self):
            self._send(204, b"")

        def _token(self):
            h = self.headers.get("Authorization", "")
            return h[7:] if h.startswith("Bearer ") else None

        def _user(self):
            u = acc.session_user(self._token())
            if u:
                return u
            return None if AUTH_ENFORCED else dict(LOCAL_USER)

        def _api_key(self):
            k = self.headers.get("X-API-Key")
            if k:
                return k
            h = self.headers.get("Authorization", "")
            if h.startswith("Bearer ") and h[7:].startswith("sk_"):
                return h[7:]
            return None

        def _api_user(self):
            """Resolve a workspace from an API key, for the public /api/v1 surface."""
            k = self._api_key()
            if k:
                ws = reg.verify_api_key(k)
                if ws is not None:
                    return acc.workspace_user(ws)
            return None if AUTH_ENFORCED else dict(LOCAL_USER)

        def _ws(self):
            """Workspace id for the signed-in user, or the local workspace."""
            u = self._user()
            return u.get("workspace_id") if u else LOCAL_WS

        def _origin(self):
            proto = self.headers.get("X-Forwarded-Proto", "http")
            host = self.headers.get("Host", "localhost")
            return f"{proto}://{host}"

        def _gate(self, feature, min_role=None):
            u = self._user()
            if u is None:
                self._send(401, {"error": "sign in to continue"})
                return None
            if not acc.has_feature(u, feature):
                self._send(402, {"error": "your plan does not include this",
                                 "feature": feature, "plan": u["plan"], "upgrade": True})
                return None
            if min_role and not acc.has_role(u, min_role):
                self._send(403, {"error": "your role can't do this",
                                 "need": min_role, "role": u.get("role")})
                return None
            return u

        def _role(self, min_role):
            u = self._user()
            if u is None:
                self._send(401, {"error": "sign in to continue"})
                return None
            if not acc.has_role(u, min_role):
                self._send(403, {"error": "your role can't do this",
                                 "need": min_role, "role": u.get("role")})
                return None
            return u

        def do_GET(self):
            from urllib.parse import urlparse, parse_qs
            u = urlparse(self.path)
            p, q = u.path, parse_qs(u.query)
            if p == "/api/auth/me":
                u = self._user()
                return self._send(200, {"auth_enforced": AUTH_ENFORCED,
                    "authenticated": bool(u and u.get("email") != "local"),
                    "user": u, "trial_days_left": (acc.trial_days_left(u) if u else 0),
                    "trial_hours_left": (acc.trial_hours_left(u) if u else 0)})
            if p in ("/api/status", "/api/config", "/api/state"):
                if self._user() is None:
                    return self._send(401, {"error": "sign in to continue"})
                if p == "/api/status":
                    return self._send(200, pf.status())
                if p == "/api/config":
                    return self._send(200, pf.config())
                return self._send(200, pf.state())
            if p == "/api/history":
                u = self._user()
                n = int(q.get("n", ["200"])[0])
                if not (u and acc.has_feature(u, "history")):
                    n = min(n, acc.FREE_HISTORY_ROWS)
                return self._send(200, pf.history(n))
            if p == "/api/fleet":
                ws = self._ws()
                nodes = reg.list_nodes(ws)
                for nd in nodes:
                    nd["alerts_1h"] = alertmod.active_count(ws, nd["node_id"])
                return self._send(200, {"nodes": nodes})
            if p == "/api/nodes/history":
                if self._user() is None:
                    return self._send(401, {"error": "sign in"})
                node = q.get("node", [""])[0]
                n = int(q.get("n", ["300"])[0])
                return self._send(200, {"node": node,
                                        "history": reg.node_history(node, self._ws(), n)})
            if p == "/api/overview":
                usr = self._user()
                ws = usr.get("workspace_id") if usr else LOCAL_WS
                nodes = reg.list_nodes(ws)
                live = sum(1 for n in nodes if not n["stale"])
                events = alertmod.list_events(ws, 5)
                params = pf.cfg.get("params") or {}
                st = pf.status()
                return self._send(200, {
                    "plan": (usr or {}).get("plan"),
                    "trial_days_left": acc.trial_days_left(usr) if usr else 0,
                    "trial_hours_left": acc.trial_hours_left(usr) if usr else 0,
                    "workspace": (usr or {}).get("workspace"),
                    "domain": st["domain"], "source": st["source"],
                    "running": st["running"], "cycles": st["cycles"],
                    "nodes_total": len(nodes), "nodes_live": live,
                    "rules": len(alertmod.list_rules(ws)),
                    "recent_alerts": events,
                    "calibrated": bool(params), "calibrated_params": list(params.keys()),
                    "features": sorted(acc.PLAN_FEATURES.get((usr or {}).get("plan", "free"), [])),
                })
            if p == "/api/audit":
                if self._user() is None:
                    return self._send(401, {"error": "sign in"})
                n = int(q.get("n", ["100"])[0])
                return self._send(200, {"events": audit.list_events(self._ws(), n)})
            if p == "/api/team":
                u = self._user()
                if u is None:
                    return self._send(401, {"error": "sign in"})
                ws = u.get("workspace_id")
                return self._send(200, {"me": {"user_id": u.get("user_id"), "role": u.get("role")},
                                        "members": acc.list_members(ws),
                                        "invites": acc.list_invites(ws),
                                        "can_manage": acc.has_role(u, "admin")})
            if p == "/api/report":
                if self._gate("history") is None:
                    return
                if not reports.HAS_REPORTLAB:
                    return self._send(200, {"error": "PDF reports need the optional extra: "
                                                     "pip install terra-engine[reports]"})
                ws = self._ws()
                ctx = {
                    "workspace": (self._user() or {}).get("workspace"),
                    "domain": pf.cfg["domain"], "source": pf.source_name,
                    "cycles": pf.cycles, "calibrated": pf.cfg.get("params") or {},
                    "nodes": reg.list_nodes(ws), "alerts": alertmod.list_events(ws, 20),
                    "history": pf.history(2000),
                    "channels": [k for k in pf.spec.channels],
                }
                pdf = reports.build_pdf(ctx)
                audit.log(ws, (self._user() or {}).get("email", "local"), "report.download",
                          f"{len(ctx['history'])} points")
                self.send_response(200)
                self.send_header("Content-Type", "application/pdf")
                self.send_header("Content-Disposition", "attachment; filename=terra-report.pdf")
                self._cors()
                self.end_headers()
                return self.wfile.write(pdf)
            if p == "/api/orders":
                if self._user() is None:
                    return self._send(401, {"error": "sign in"})
                return self._send(200, {"orders": support.list_orders(self._ws())})
            if p == "/api/tickets":
                if self._user() is None:
                    return self._send(401, {"error": "sign in"})
                return self._send(200, {"tickets": support.list_tickets(self._ws())})
            if p == "/api/alerts/rules":
                if self._gate("alerts") is None:
                    return
                return self._send(200, {"rules": alertmod.list_rules(self._ws())})
            if p == "/api/alerts/events":
                if self._gate("alerts") is None:
                    return
                n = int(q.get("n", ["50"])[0])
                return self._send(200, {"events": alertmod.list_events(self._ws(), n)})
            if p == "/api/keys":
                if self._gate("api") is None:
                    return
                return self._send(200, {"keys": reg.list_api_keys(self._ws())})
            if p == "/api/v1/status":
                u = self._api_user()
                if u is None:
                    return self._send(401, {"error": "provide an API key"})
                if not acc.has_feature(u, "api"):
                    return self._send(402, {"error": "API access needs a paid plan"})
                return self._send(200, pf.status())
            if p == "/api/v1/history":
                u = self._api_user()
                if u is None:
                    return self._send(401, {"error": "provide an API key"})
                if not acc.has_feature(u, "api"):
                    return self._send(402, {"error": "API access needs a paid plan"})
                return self._send(200, pf.history(int(q.get("n", ["200"])[0])))
            if p == "/api/v1/config":
                # the provisioning bundle a node fetches so it runs the workspace's
                # configured domain, channels, and calibrated parameters offline.
                ws = reg.verify_node_key(self.headers.get("X-Node-Id"),
                                         self.headers.get("X-Node-Key", ""))
                if ws is None:
                    au = self._api_user()
                    ws = au.get("workspace_id") if au else None
                if ws is None:
                    return self._send(401, {"error": "provide a node key or API key"})
                bundle = pf.config_bundle()
                bundle["alerts"] = alertmod.list_rules(ws)
                return self._send(200, bundle)
            if p == "/api/export":
                if self._user() is None:
                    return self._send(401, {"error": "sign in"})
                self.send_response(200)
                self.send_header("Content-Type", "text/csv")
                self.send_header("Content-Disposition", "attachment; filename=terra-history.csv")
                self._cors()
                self.end_headers()
                return self.wfile.write(pf.export_csv().encode())
            if p.startswith("/assets/"):
                name = os.path.basename(p)
                f = os.path.join(WEB_DIR, "assets", name)
                if os.path.isfile(f):
                    ext = name.rsplit(".", 1)[-1].lower()
                    ct = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                          "webp": "image/webp", "mp4": "video/mp4", "webm": "video/webm",
                          "svg": "image/svg+xml", "ico": "image/x-icon"}.get(ext, "application/octet-stream")
                    return self._send(200, open(f, "rb").read(), ct)
                return self._send(404, {"error": "asset not found"})
            if p == "/favicon.ico":
                f = os.path.join(WEB_DIR, "assets", "favicon.ico")
                if os.path.isfile(f):
                    return self._send(200, open(f, "rb").read(), "image/x-icon")
                return self._send(404, {"error": "no favicon"})
            for _page in ("pricing", "order", "support", "docs", "terms", "privacy"):
                if p in ("/" + _page, "/" + _page + ".html"):
                    f = os.path.join(WEB_DIR, _page + ".html")
                    if os.path.exists(f):
                        return self._send(200, open(f, "rb").read(), "text/html; charset=utf-8")
                    return self._send(404, {"error": _page + " not bundled"})
            if p in ("/app", "/console", "/index.html"):
                f = os.path.join(WEB_DIR, "index.html")
                if os.path.exists(f):
                    return self._send(200, open(f, "rb").read(), "text/html; charset=utf-8")
                return self._send(404, {"error": "console not bundled"})
            if p in ("/", "/home", "/landing"):
                f = os.path.join(WEB_DIR, "landing.html")
                if not os.path.exists(f):   # fall back to the console if no landing bundled
                    f = os.path.join(WEB_DIR, "index.html")
                if os.path.exists(f):
                    return self._send(200, open(f, "rb").read(), "text/html; charset=utf-8")
                return self._send(404, {"error": "not bundled"})
            return self._send(404, {"error": "not found"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0) or 0)
            if n > MAX_BODY:
                self.close_connection = True   # don't read an oversized body into memory
                return self._send(413, {"error": "request body too large"})
            raw = self.rfile.read(n) if n else b""
            p = self.path.split("?")[0]
            if p == "/api/ingest":
                if self._gate("control", "member") is None:
                    return
                pf.ingest(raw.decode("utf-8", "replace"), "uploaded.csv")
                return self._send(200, pf.status())
            try:
                data = json.loads(raw or b"{}")
            except Exception:
                data = {}
            if p == "/api/auth/signup":
                ip = self.client_address[0]
                if _rl_blocked("signup:" + ip, 10, 3600):
                    return self._send(429, {"error": "too many signups — try later"})
                _rl_hit("signup:" + ip)
                try:
                    r = acc.create_account(data.get("email"), data.get("password"),
                                           data.get("workspace"),
                                           invite_token=data.get("invite_token"))
                except ValueError as e:
                    return self._send(400, {"error": str(e)})
                joined = bool(data.get("invite_token"))
                audit.log(r["workspace_id"], data.get("email"), "account.create",
                          f"joined as {r.get('role')}" if joined else "trial started")
                if not joined:
                    try:
                        from . import mailer
                        mailer.send_welcome(data.get("email"),
                                            data.get("workspace") or "your workspace",
                                            acc.TRIAL_HOURS, origin=self._origin())
                        vtok = acc.create_verify(r["user_id"], data.get("email"))
                        mailer.send_verify(data.get("email"),
                                           f"{self._origin()}/app?verify={vtok}")
                    except Exception:
                        pass
                return self._send(200, {"token": r["token"],
                                        "user": acc.session_user(r["token"])})
            if p == "/api/auth/login":
                ident = (data.get("email") or "").strip().lower() or self.client_address[0]
                if _login_blocked(ident):
                    return self._send(429, {"error": "too many attempts — wait a few minutes"})
                tok = acc.login(data.get("email"), data.get("password"))
                if not tok:
                    _login_fail(ident)
                    return self._send(401, {"error": "invalid email or password"})
                _RL.pop("login:" + ident, None)
                u2 = acc.session_user(tok)
                audit.log(u2.get("workspace_id") if u2 else None, data.get("email"), "auth.login", "")
                return self._send(200, {"token": tok, "user": u2})
            if p == "/api/auth/logout":
                t = self._token()
                if t:
                    acc.logout(t)
                return self._send(200, {"ok": True})
            if p == "/api/auth/reset-request":
                ip = self.client_address[0]
                if _rl_blocked("reset:" + ip, 5, 3600):
                    return self._send(429, {"error": "too many reset requests — try later"})
                _rl_hit("reset:" + ip)
                tok = acc.create_reset(data.get("email"))
                if tok:
                    try:
                        from . import mailer
                        mailer.send_reset(data.get("email"),
                                          f"{self._origin()}/app?reset={tok}")
                    except Exception:
                        pass
                # always 200 so we don't reveal whether an email is registered
                return self._send(200, {"ok": True})
            if p == "/api/auth/reset":
                if acc.reset_password(data.get("token"), data.get("password")):
                    return self._send(200, {"ok": True})
                return self._send(400, {"error": "invalid or expired link, or weak password"})
            if p == "/api/auth/verify":
                if acc.verify_email(data.get("token")):
                    return self._send(200, {"ok": True})
                return self._send(400, {"error": "invalid or expired link"})
            if p == "/api/auth/verify/resend":
                u = self._user()
                if not u:
                    return self._send(401, {"error": "sign in"})
                try:
                    from . import mailer
                    vtok = acc.create_verify(u["user_id"], u["email"])
                    mailer.send_verify(u["email"], f"{self._origin()}/app?verify={vtok}")
                except Exception:
                    pass
                return self._send(200, {"ok": True})
            if p == "/api/account/delete":
                u = self._role("owner")
                if u is None:
                    return
                audit.log(u["workspace_id"], u.get("email"), "workspace.delete", "")
                acc.delete_workspace(u["workspace_id"])
                return self._send(200, {"ok": True})
            if p == "/api/billing/upgrade":
                u = self._role("admin")
                if u is None:
                    return
                if not u.get("workspace_id"):
                    return self._send(401, {"error": "sign in first"})
                plan = data.get("plan", "pro")
                if billing.enabled():
                    origin = self._origin()
                    qty = max(1, len(reg.list_nodes(u["workspace_id"])))  # bill per node
                    try:
                        sess = billing.create_checkout_session(
                            u["workspace_id"], plan,
                            success_url=origin + "/app?upgraded=1",
                            cancel_url=origin + "/pricing", quantity=qty)
                    except Exception as e:
                        return self._send(400, {"error": f"billing error: {e}"})
                    return self._send(200, {"url": sess.get("url"), "plan": plan})
                # Stripe not configured. The stub sets the plan directly for local dev
                # only; on a real deploy this would be a free self-upgrade, so it is
                # disabled unless explicitly opted in.
                if not ALLOW_STUB_BILLING:
                    return self._send(503, {"error": "billing is not configured"})
                acc.set_plan(u["workspace_id"], plan)
                audit.log(u["workspace_id"], u.get("email"), "billing.plan", plan + " (stub)")
                return self._send(200, {"ok": True, "plan": plan, "stub": True})
            if p == "/api/billing/webhook":
                sig = self.headers.get("Stripe-Signature", "")
                if not billing.verify_signature(raw, sig):
                    return self._send(400, {"error": "bad signature"})
                try:
                    ev = json.loads(raw or b"{}")
                    plan = billing.handle_event(ev)
                    if plan:
                        obj = (ev.get("data") or {}).get("object") or {}
                        ws = (obj.get("metadata") or {}).get("workspace_id") or obj.get("client_reference_id")
                        if ws:
                            audit.log(int(ws), "stripe", "billing.plan", plan)
                except Exception:
                    pass
                return self._send(200, {"received": True})
            if p == "/api/orders":
                u = self._user()
                ws = u.get("workspace_id") if u else None
                try:
                    oid = support.create_order(
                        data.get("email") or (u or {}).get("email"),
                        data.get("board", ""), data.get("tier", ""),
                        int(data.get("qty", 1)), data.get("notes", ""), workspace_id=ws)
                except (ValueError, TypeError) as e:
                    return self._send(400, {"error": str(e)})
                audit.log(ws, (u or {}).get("email") or data.get("email"), "order.create",
                          f"{data.get('qty', 1)}x {data.get('board', '')}")
                return self._send(200, {"ok": True, "order_id": oid})
            if p == "/api/support":
                u = self._user()
                ws = u.get("workspace_id") if u else None
                try:
                    tid = support.create_ticket(
                        data.get("email") or (u or {}).get("email"),
                        data.get("subject", "Support request"), data.get("body", ""),
                        workspace_id=ws)
                except (ValueError, TypeError) as e:
                    return self._send(400, {"error": str(e)})
                audit.log(ws, (u or {}).get("email") or data.get("email"), "support.ticket",
                          data.get("subject", ""))
                return self._send(200, {"ok": True, "ticket_id": tid})
            if p == "/api/config":
                if self._role("member") is None:
                    return
                pf.set_config(data)
                return self._send(200, pf.config())
            if p == "/api/control":
                u = self._gate("control", "member")
                if u is None:
                    return
                pf.autonomy = bool(data.get("autonomy"))
                pf.cfg["autonomy"] = pf.autonomy
                save_config(pf.cfg)
                audit.log(u.get("workspace_id"), u.get("email"), "control.autonomy",
                          "on" if pf.autonomy else "off")
                return self._send(200, pf.status())
            if p == "/api/offline":
                if self._gate("control", "member") is None:
                    return
                pf.offline = bool(data.get("offline"))
                return self._send(200, pf.status())
            if p == "/api/calibrate":
                u = self._gate("calibrate", "member")
                if u is None:
                    return
                audit.log(u.get("workspace_id"), u.get("email"), "engine.calibrate", pf.cfg["domain"])
                return self._send(200, pf.calibrate())
            if p == "/api/team/invite":
                u = self._role("admin")
                if u is None:
                    return
                email = (data.get("email") or "").strip().lower()
                role = data.get("role", "member")
                if "@" not in email:
                    return self._send(400, {"error": "a valid email is required"})
                tok = acc.create_invite(u["workspace_id"], email, role)
                link = f"{self._origin()}/app?invite={tok}"
                try:
                    from . import mailer
                    mailer.send_invite(email, u.get("workspace") or "the workspace",
                                       role, link, inviter=u.get("email"))
                except Exception:
                    pass
                audit.log(u["workspace_id"], u.get("email"), "team.invite", f"{email} ({role})")
                return self._send(200, {"ok": True, "invite_link": link,
                                        "invites": acc.list_invites(u["workspace_id"])})
            if p == "/api/team/role":
                u = self._role("admin")
                if u is None:
                    return
                ok = acc.set_role(u["workspace_id"], int(data.get("user_id", 0)),
                                  data.get("role", "member"))
                if ok:
                    audit.log(u["workspace_id"], u.get("email"), "team.role",
                              f"user {data.get('user_id')} -> {data.get('role')}")
                return self._send(200 if ok else 400,
                                  {"ok": ok, "members": acc.list_members(u["workspace_id"])})
            if p == "/api/team/remove":
                u = self._role("admin")
                if u is None:
                    return
                ok = acc.remove_member(u["workspace_id"], int(data.get("user_id", 0)))
                if ok:
                    audit.log(u["workspace_id"], u.get("email"), "team.remove",
                              f"user {data.get('user_id')}")
                return self._send(200 if ok else 400,
                                  {"ok": ok, "members": acc.list_members(u["workspace_id"])})
            if p == "/api/alerts/rules":
                if self._gate("alerts") is None:
                    return
                try:
                    rid = alertmod.create_rule(
                        self._ws(), data.get("name") or "Alert", data.get("metric"),
                        data.get("op", ">"), float(data.get("threshold", 0)),
                        node_id=data.get("node_id", "*"),
                        delivery=data.get("delivery", "none"), dest=data.get("dest", ""),
                        cooldown=int(data.get("cooldown", alertmod.DEFAULT_COOLDOWN)))
                except (ValueError, TypeError) as e:
                    return self._send(400, {"error": str(e)})
                audit.log(self._ws(), (self._user() or {}).get("email"), "alert.rule.create",
                          data.get("name", ""))
                return self._send(200, {"id": rid, "rules": alertmod.list_rules(self._ws())})
            if p == "/api/alerts/rules/delete":
                if self._gate("alerts") is None:
                    return
                alertmod.delete_rule(self._ws(), int(data.get("id", 0)))
                return self._send(200, {"rules": alertmod.list_rules(self._ws())})
            if p == "/api/alerts/rules/toggle":
                if self._gate("alerts") is None:
                    return
                alertmod.set_enabled(self._ws(), int(data.get("id", 0)),
                                     bool(data.get("enabled")))
                return self._send(200, {"rules": alertmod.list_rules(self._ws())})
            if p == "/api/keys/create":
                if self._gate("api", "admin") is None:
                    return
                k = reg.create_api_key(self._ws(), data.get("name", "default"))
                audit.log(self._ws(), (self._user() or {}).get("email"), "apikey.create",
                          k.get("prefix", ""))
                return self._send(200, k)   # full key returned once
            if p == "/api/keys/revoke":
                if self._gate("api", "admin") is None:
                    return
                reg.revoke_api_key(self._ws(), int(data.get("id", 0)))
                return self._send(200, {"keys": reg.list_api_keys(self._ws())})
            if p == "/api/nodes/enroll-token":
                u = self._role("admin")
                if u is None:
                    return
                # free plan is capped at one node; multi_node lifts the cap
                if not acc.has_feature(u, "multi_node") and \
                        len(reg.list_nodes(u["workspace_id"])) >= acc.FREE_NODE_LIMIT:
                    return self._send(402, {"error": "your plan is limited to "
                                            f"{acc.FREE_NODE_LIMIT} node(s)",
                                            "feature": "multi_node", "upgrade": True})
                tok = reg.create_enroll_token(u["workspace_id"])
                audit.log(u["workspace_id"], u.get("email"), "node.enroll_token", "")
                return self._send(200, {"enroll_token": tok, "expires_in": reg.ENROLL_TOKEN_TTL})
            if p == "/api/v1/enroll":
                r = reg.enroll_node(data.get("enroll_token"), data.get("node_id"),
                                    data.get("name"), data.get("domain"))
                if not r:
                    return self._send(401, {"error": "invalid or expired enrollment token"})
                audit.log(r["workspace_id"], "node:" + r["node_id"], "node.enroll",
                          data.get("name", ""))
                return self._send(200, r)   # node_key returned once
            if p == "/api/v1/ingest":
                nid = data.get("node_id") or self.headers.get("X-Node-Id")
                ws = reg.verify_node_key(nid, self.headers.get("X-Node-Key", ""))
                if ws is None:
                    ws = self._api_user() and self._api_user().get("workspace_id")
                if ws is None:
                    return self._send(401, {"error": "provide a valid node key or API key"})
                reg.touch_node(nid or "node", ws, name=data.get("name"),
                               domain=data.get("domain"), status=data.get("status"))
                if data.get("metrics"):
                    alertmod.evaluate(ws, nid or "node", data["metrics"])
                    reg.append_history(nid or "node", data["metrics"])
                if data.get("csv"):
                    pf.ingest(data["csv"], f"{nid or 'node'}.csv")
                return self._send(200, {"ok": True, "node_id": nid, "workspace_id": ws})
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
