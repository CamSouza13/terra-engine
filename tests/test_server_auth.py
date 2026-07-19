"""End-to-end HTTP tests for auth, plan gating, enrollment, and the public API.

Spins up the real ThreadingHTTPServer on an ephemeral port with auth enforced.
"""
import importlib
import json
import os
import sys
import tempfile
import threading
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _boot(tmp):
    os.environ["TERRA_HOME"] = tmp
    os.environ["TERRA_AUTH"] = "1"
    from terra import accounts, alerts, registry, billing, server
    for m in (accounts, alerts, registry, billing, server):
        importlib.reload(m)
    return server


def _call(base, path, method="GET", body=None, headers=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method=method,
                                 headers={"Content-Type": "application/json",
                                          **(headers or {})})
    try:
        r = urllib.request.urlopen(req, timeout=10)
        return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def test_auth_gating_enrollment_and_api():
    from http.server import ThreadingHTTPServer
    with tempfile.TemporaryDirectory() as tmp:
        srv = _boot(tmp)
        pf = srv.Platform()
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.make_handler(pf))
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            assert _call(base, "/api/auth/me")[1]["authenticated"] is False
            _, su = _call(base, "/api/auth/signup", "POST",
                          {"email": "cam@terra.io", "password": "password1"})
            H = {"Authorization": "Bearer " + su["token"]}
            assert su["user"]["plan"] == "trial"

            # gating: anonymous calibrate is 401
            assert _call(base, "/api/calibrate", "POST", {})[0] == 401
            # a free workspace cannot calibrate (402)
            srv.acc.set_plan(su["user"]["workspace_id"], "free")
            assert _call(base, "/api/calibrate", "POST", {}, H)[0] == 402
            srv.acc.set_plan(su["user"]["workspace_id"], "pro")

            # enroll a node, then ingest with its key; a bad key is rejected
            et = _call(base, "/api/nodes/enroll-token", "POST", {}, H)[1]["enroll_token"]
            enr = _call(base, "/api/v1/enroll", "POST",
                        {"enroll_token": et, "name": "Pond A", "domain": "aquaculture"})[1]
            NK = {"X-Node-Key": enr["node_key"], "X-Node-Id": enr["node_id"]}
            assert _call(base, "/api/v1/ingest", "POST",
                         {"node_id": enr["node_id"], "metrics": {"nis": 1}}, NK)[0] == 200
            assert _call(base, "/api/v1/ingest", "POST", {"node_id": enr["node_id"]},
                         {"X-Node-Key": "nk_wrong"})[0] == 401

            # fleet shows the node; API key unlocks the public status endpoint
            assert any(n["name"] == "Pond A" for n in _call(base, "/api/fleet", "GET", None, H)[1]["nodes"])
            key = _call(base, "/api/keys/create", "POST", {"name": "prod"}, H)[1]["key"]
            assert _call(base, "/api/v1/status")[0] == 401
            assert _call(base, "/api/v1/status", "GET", None, {"X-API-Key": key})[0] == 200

            # login rate limiting kicks in after repeated failures
            codes = [_call(base, "/api/auth/login", "POST",
                           {"email": "cam@terra.io", "password": "wrong"})[0] for _ in range(7)]
            assert codes[-1] == 429
            print("  server: gating, enrollment, node/api keys, rate limit OK")
        finally:
            pf.running = False
            httpd.shutdown()


if __name__ == "__main__":
    test_auth_gating_enrollment_and_api()
    print("server auth tests passed")
