"""HTTP API + console host for a running Terra node.

`terra serve` starts a stdlib HTTP server that (1) runs the engine live in a
background thread and (2) exposes it as JSON, plus serves the web console from
the same origin so the browser connects with no CORS/mixed-content friction:

    GET  /                -> the Terra Console (web/index.html)
    GET  /api/status      -> {domain, running, cycles, autonomy, offline}
    GET  /api/state       -> latest estimate: channels, hidden, risks, events
    GET  /api/config      -> domain channels + safety limits
    POST /api/control     -> {"autonomy": true|false}
    POST /api/offline     -> {"offline": true|false}   (edge-autonomous mode)
    POST /api/fault       -> inject a demo fault

Numpy-only; no third-party web dependencies. On a real node the live loop would
be fed by a hardware `SensorDriver`; here it is driven by the domain simulator
so the whole stack is exercisable end to end.
"""
from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .core import TerraEngine, EngineConfig
from .domains import DOMAINS
from .control import Controller, policy_for

WEB_DIR = os.path.join(os.path.dirname(__file__), "web")


class NodeService:
    """Runs one domain's engine continuously and exposes a JSON snapshot."""

    def __init__(self, domain: str, autonomy: bool = False, speed: float = 0.4):
        if domain not in DOMAINS:
            raise ValueError(f"unknown domain {domain!r}; choices: {sorted(DOMAINS)}")
        self.domain = domain
        self.autonomy = autonomy
        self.offline = False
        self.speed = speed
        self.fault = False
        self.lock = threading.Lock()
        self._build_stream()
        self.cycles = 0
        self.latest = None
        self.rec = None
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _build_stream(self):
        mod = DOMAINS[self.domain]
        self.spec, self.sim = mod.simulate(fault=self.fault)
        self.engine = getattr(self, "engine", None) or TerraEngine(
            self.spec, EngineConfig(forecast_horizon_h=12, forecast_samples=120))
        try:
            self.controller = Controller(self.spec, self.engine.cfg,
                                          policy_for(self.domain), authorized=self.autonomy)
        except Exception:
            self.controller = None
        self.i = 0

    def _loop(self):
        while self.running:
            with self.lock:
                self._step()
            time.sleep(self.speed)

    def _step(self):
        t = self.sim["t"]
        dt = float(t[1] - t[0])
        if self.i >= len(t):
            self._build_stream()                       # loop the scenario
        i = self.i
        est = self.engine.step(t[i], dt, self.sim["meas"][i], self.sim["u"][i],
                               u_forecast=self.sim.get("u_forecast"))
        if self.controller is not None:
            self.controller.authorized = self.autonomy
            self.rec = self.controller.recommend(est, self.sim["u"][i])
        self.latest = est
        self.i += 1
        self.cycles += 1

    # ---- control ----
    def set_autonomy(self, on: bool):
        with self.lock:
            self.autonomy = bool(on)

    def set_offline(self, on: bool):
        with self.lock:
            self.offline = bool(on)

    def inject_fault(self):
        with self.lock:
            self.fault = True
            self.engine = None                         # rebuild engine on next stream
            self._build_stream()

    # ---- snapshots ----
    def status(self):
        return {"domain": self.domain, "running": self.running, "cycles": self.cycles,
                "autonomy": self.autonomy, "offline": self.offline}

    def config(self):
        d = self.spec
        return {
            "domain": self.domain,
            "hidden": d.hidden,
            "channels": [{"key": k, "state": ch.state} for k, ch in d.channels.items()],
            "safety": [{"name": s.name, "limit": s.limit, "direction": s.direction,
                        "units": s.units} for s in d.safety],
        }

    def state(self):
        with self.lock:
            est = self.latest
            if est is None:
                return {"ready": False}
            d = self.spec
            channels = {k: float(est.x[ch.state]) if ch.state is not None
                        else float(ch.obs(est.x)) for k, ch in d.channels.items()}
            risks = {name: {"p": r["p"], "t_cross": r["t_cross"], "limit": r["limit"],
                            "direction": r["direction"], "units": r["units"]}
                     for name, r in est.risks.items()}
            events = [[float(t), lv, m] for (t, lv, m) in self.engine.events[-24:]]
            return {
                "ready": True, "t": est.t, "cycles": self.cycles,
                "channels": channels, "hidden": est.hidden, "hidden_std": est.hidden_std,
                "nis": est.nis, "budget": est.budget_residual, "risks": risks,
                "events": events,
                "recommendation": (self.rec.message() if self.rec else None),
                "autonomy": self.autonomy, "offline": self.offline,
            }


def make_handler(service: NodeService):
    class Handler(BaseHTTPRequestHandler):
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
            p = self.path.split("?")[0]
            if p == "/api/status":
                return self._send(200, service.status())
            if p == "/api/state":
                return self._send(200, service.state())
            if p == "/api/config":
                return self._send(200, service.config())
            if p in ("/", "/index.html"):
                f = os.path.join(WEB_DIR, "index.html")
                if os.path.exists(f):
                    with open(f, "rb") as fh:
                        return self._send(200, fh.read(), "text/html; charset=utf-8")
                return self._send(404, {"error": "console not bundled"})
            return self._send(404, {"error": "not found"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0) or 0)
            try:
                data = json.loads(self.rfile.read(n) or b"{}")
            except Exception:
                data = {}
            p = self.path.split("?")[0]
            if p == "/api/control":
                service.set_autonomy(bool(data.get("autonomy")))
                return self._send(200, service.status())
            if p == "/api/offline":
                service.set_offline(bool(data.get("offline")))
                return self._send(200, service.status())
            if p == "/api/fault":
                service.inject_fault()
                return self._send(200, {"ok": True})
            return self._send(404, {"error": "not found"})
    return Handler


def serve(domain: str = "aquaculture", port: int = 8700, autonomy: bool = False,
          host: str = "0.0.0.0") -> int:
    service = NodeService(domain, autonomy=autonomy)
    httpd = ThreadingHTTPServer((host, port), make_handler(service))
    print(f"Terra node serving {domain} on http://localhost:{port}  (console + /api)")
    print("edge-autonomous: the engine runs locally; the console is optional.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        service.running = False
        httpd.shutdown()
    return 0
