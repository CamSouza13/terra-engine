"""Node-to-platform reporting: enrollment and authenticated heartbeats.

Turns a running edge node into a member of a hosted workspace. ``enroll`` redeems
a one-time enrollment token minted in the console and stores the node's permanent
key locally; ``ServerReporter`` then POSTs a heartbeat (channel values, hidden
state, NIS, and any risks) to ``/api/v1/ingest`` on a cadence, authenticating with
the node key. Reports are best-effort — if the server is unreachable the node
keeps running and simply resumes reporting when it returns (the local run loop is
the source of truth; the platform is a mirror).

All stdlib: no requests, no SDK.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request

HOME = os.environ.get("TERRA_HOME", os.path.expanduser("~/.terra"))
CREDS_PATH = os.path.join(HOME, "node_creds.json")
CONFIG_PATH = os.path.join(HOME, "node_config.json")


def _post(url: str, payload: dict, headers: dict = None, timeout: float = 10.0) -> dict:
    data = json.dumps(payload).encode()
    h = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(url, data=data, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read() or b"{}")


def _get(url: str, headers: dict = None, timeout: float = 10.0) -> dict:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read() or b"{}")


def _save_json(path: str, obj: dict):
    os.makedirs(HOME, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def enroll(server: str, enroll_token: str, node_id: str = None,
           name: str = None, domain: str = None) -> dict:
    """Redeem an enrollment token; persist the returned node key locally."""
    server = server.rstrip("/")
    r = _post(server + "/api/v1/enroll",
              {"enroll_token": enroll_token, "node_id": node_id,
               "name": name, "domain": domain})
    if not r.get("node_key"):
        raise SystemExit("enrollment failed: invalid or expired token")
    creds = {"server": server, "node_id": r["node_id"], "node_key": r["node_key"],
             "name": name, "domain": domain}
    _save_json(CREDS_PATH, creds)
    os.chmod(CREDS_PATH, 0o600)
    fetch_config(creds)   # provision the node with the workspace's configuration
    return creds


def load_creds(server: str = None) -> dict | None:
    if not os.path.exists(CREDS_PATH):
        return None
    try:
        with open(CREDS_PATH) as f:
            c = json.load(f)
    except Exception:
        return None
    if server:
        c["server"] = server.rstrip("/")
    return c


def fetch_config(creds: dict) -> dict | None:
    """Fetch the workspace's config bundle from the platform and cache it locally."""
    try:
        bundle = _get(creds["server"].rstrip("/") + "/api/v1/config",
                      {"X-Node-Key": creds["node_key"], "X-Node-Id": creds["node_id"]})
    except Exception:
        return None
    if bundle.get("domain"):
        _save_json(CONFIG_PATH, bundle)
        return bundle
    return None


def load_config() -> dict | None:
    if not os.path.exists(CONFIG_PATH):
        return None
    try:
        with open(CONFIG_PATH) as f:
            return json.load(f)
    except Exception:
        return None


def build_spec_from_bundle(bundle: dict):
    """Reconstruct a domain spec with the bundle's calibrated parameters applied."""
    import dataclasses

    from ..domains import DOMAINS
    spec = DOMAINS[bundle["domain"]].build_spec()
    params = {k: v for k, v in (bundle.get("params") or {}).items()
              if hasattr(spec.params, k)}
    if params:
        spec = dataclasses.replace(spec, params=dataclasses.replace(spec.params, **params))
    return spec


def build_metrics(spec, est) -> dict:
    channels = {k: (float(est.x[ch.state]) if ch.state is not None
                    else float(ch.obs(est.x))) for k, ch in spec.channels.items()}
    risks = {n: r["p"] for n, r in est.risks.items()}
    hidden = est.hidden
    h0 = next(iter(hidden.values())) if isinstance(hidden, dict) and hidden else None
    return {"channels": channels, "risks": risks, "nis": est.nis, "hidden": h0}


class ServerReporter:
    """Sends authenticated heartbeats to the platform on a fixed cadence."""

    def __init__(self, spec, creds: dict, interval_s: float = 10.0):
        self.spec = spec
        self.server = creds["server"].rstrip("/")
        self.node_id = creds["node_id"]
        self.name = creds.get("name")
        self.domain = creds.get("domain") or spec.name
        self.headers = {"X-Node-Key": creds["node_key"], "X-Node-Id": self.node_id}
        self.interval = interval_s
        self._last = 0.0
        self.sent = 0
        self.failures = 0

    def on_cycle(self, est, t):
        now = time.time()
        if now - self._last < self.interval:
            return
        self._last = now
        try:
            _post(self.server + "/api/v1/ingest", {
                "node_id": self.node_id, "name": self.name, "domain": self.domain,
                "status": {"cycles": self.sent + 1, "nis": est.nis, "t": est.t},
                "metrics": build_metrics(self.spec, est),
            }, self.headers, timeout=8.0)
            self.sent += 1
        except Exception:
            self.failures += 1   # best-effort; node keeps running and retries next tick
