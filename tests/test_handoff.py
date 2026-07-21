"""Online-to-offline config handoff: a node fetches the workspace's configured
domain and calibrated parameters and rebuilds its engine spec from them."""
import dataclasses
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
    from terra.node import report
    for m in (accounts, registry, alerts, server, report):
        importlib.reload(m)
    return server, report


def _call(base, path, method="GET", body=None, headers=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method=method,
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        r = urllib.request.urlopen(req, timeout=10)
        return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def test_config_handoff():
    from http.server import ThreadingHTTPServer
    from terra.domains import aquaculture
    pname = dataclasses.fields(aquaculture.build_spec().params)[0].name
    newval = getattr(aquaculture.build_spec().params, pname) * 1.1 + 0.001

    with tempfile.TemporaryDirectory() as tmp:
        server, report = _boot(tmp)
        pf = server.Platform()
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(pf))
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            owner = _call(base, "/api/auth/signup", "POST",
                          {"email": "cam@t.io", "password": "password1", "workspace": "Terra"})[1]
            H = {"Authorization": "Bearer " + owner["token"]}
            # configure a calibrated parameter through the console API (online setup)
            assert _call(base, "/api/config", "POST", {"params": {pname: newval}}, H)[0] == 200
            # enroll a node and fetch its config bundle with the node key
            et = _call(base, "/api/nodes/enroll-token", "POST", {}, H)[1]["enroll_token"]
            enr = _call(base, "/api/v1/enroll", "POST",
                        {"enroll_token": et, "name": "pond-a", "domain": "aquaculture"})[1]
            NK = {"X-Node-Key": enr["node_key"], "X-Node-Id": enr["node_id"]}
            st, bundle = _call(base, "/api/v1/config", "GET", None, NK)
            assert st == 200 and bundle["domain"] == "aquaculture"
            assert abs(bundle["params"][pname] - newval) < 1e-9
            assert "safety" in bundle and "alerts" in bundle
            # anonymous access is rejected
            assert _call(base, "/api/v1/config")[0] == 401
            # the node reconstructs its spec with the calibrated parameter applied
            spec = report.build_spec_from_bundle(bundle)
            assert abs(getattr(spec.params, pname) - newval) < 1e-9
            # and the node-side fetch persists the bundle locally for offline use
            creds = {"server": base, "node_id": enr["node_id"], "node_key": enr["node_key"]}
            saved = report.fetch_config(creds)
            assert saved and saved["domain"] == "aquaculture"
            assert report.load_config()["params"][pname]
            print("  handoff: bundle fetch by node key, param round-trip, spec rebuild OK")
        finally:
            pf.running = False
            httpd.shutdown()


if __name__ == "__main__":
    test_config_handoff()
    print("all handoff tests passed")
