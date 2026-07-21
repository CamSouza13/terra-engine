"""Plan gating: 72-hour trial and the free-plan single-node limit."""
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
    from terra import accounts, registry, alerts, server
    for m in (accounts, registry, alerts, server):
        importlib.reload(m)
    return server


def _call(base, path, method="GET", body=None, headers=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method=method,
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        r = urllib.request.urlopen(req, timeout=10)
        return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def test_trial_and_node_limit():
    from http.server import ThreadingHTTPServer
    with tempfile.TemporaryDirectory() as tmp:
        server = _boot(tmp)
        pf = server.Platform()
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(pf))
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            su = _call(base, "/api/auth/signup", "POST",
                       {"email": "cam@t.io", "password": "password1", "workspace": "Terra"})[1]
            H = {"Authorization": "Bearer " + su["token"]}
            ws = su["user"]["workspace_id"]

            # trial is measured in hours, close to 72
            me = _call(base, "/api/auth/me", "GET", None, H)[1]
            assert 70 <= me["trial_hours_left"] <= 72

            # drop to free: one node allowed, a second is blocked (402)
            server.acc.set_plan(ws, "free")
            et = _call(base, "/api/nodes/enroll-token", "POST", {}, H)
            assert et[0] == 200
            enr = _call(base, "/api/v1/enroll", "POST",
                        {"enroll_token": et[1]["enroll_token"], "name": "n1",
                         "domain": "aquaculture"})[1]
            assert enr.get("node_key")
            blocked = _call(base, "/api/nodes/enroll-token", "POST", {}, H)
            assert blocked[0] == 402 and blocked[1].get("feature") == "multi_node"

            # pro lifts the cap
            server.acc.set_plan(ws, "pro")
            assert _call(base, "/api/nodes/enroll-token", "POST", {}, H)[0] == 200
            print("  gating: 72h trial + free single-node limit + pro lifts it OK")
        finally:
            pf.running = False
            httpd.shutdown()


if __name__ == "__main__":
    test_trial_and_node_limit()
    print("all gating tests passed")
