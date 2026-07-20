"""Security-hardening regression tests: engine-endpoint auth, body cap, rate
limits, disabled stub billing, and fail-closed webhook verification."""
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
    os.environ["TERRA_MAX_BODY"] = "1000"
    for k in ("TERRA_ALLOW_STUB_BILLING", "STRIPE_SECRET_KEY", "STRIPE_WEBHOOK_SECRET"):
        os.environ.pop(k, None)
    from terra import accounts, billing, registry, alerts, support, audit, reports, server
    for m in (accounts, billing, registry, alerts, support, audit, reports, server):
        importlib.reload(m)
    return server, billing


def _call(base, path, method="GET", body=None, headers=None, raw=None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(base + path, data=data, method=method,
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        r = urllib.request.urlopen(req, timeout=10)
        return r.status
    except urllib.error.HTTPError as e:
        return e.code


def test_hardening():
    from http.server import ThreadingHTTPServer
    with tempfile.TemporaryDirectory() as tmp:
        server, billing = _boot(tmp)
        pf = server.Platform()
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(pf))
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            # H2: anonymous reads of the engine are blocked
            assert _call(base, "/api/status") == 401
            assert _call(base, "/api/state") == 401
            assert _call(base, "/api/config") == 401
            assert _call(base, "/api/export") == 401
            # H1: anonymous ingest is blocked
            assert _call(base, "/api/ingest", "POST", raw=b"t,do\n0,5\n") == 401
            # H3: oversized body -> 413 (cap set to 1000 bytes above)
            assert _call(base, "/api/ingest", "POST", raw=b"x" * 2000) == 413
            # signed-in user CAN read the engine
            import urllib.request as u
            r = u.urlopen(u.Request(base + "/api/auth/signup", method="POST",
                          data=json.dumps({"email": "a@b.io", "password": "password1"}).encode(),
                          headers={"Content-Type": "application/json"}))
            tok = json.loads(r.read())["token"]
            H = {"Authorization": "Bearer " + tok}
            assert _call(base, "/api/status", "GET", None, H) == 200
            # M5: stub billing is disabled without the opt-in flag
            assert _call(base, "/api/billing/upgrade", "POST", {"plan": "pro"}, H) == 503
            # M2: signup is rate limited per IP
            codes = [_call(base, "/api/auth/signup", "POST",
                           {"email": f"x{i}@b.io", "password": "password1"}) for i in range(12)]
            assert 429 in codes
            print("  security: engine auth, 413 body cap, stub-off, signup rate-limit OK")
        finally:
            pf.running = False
            httpd.shutdown()

    # M1: webhook verification fails closed when billing is live but no secret is set
    os.environ["STRIPE_SECRET_KEY"] = "sk_test_dummy"
    os.environ.pop("STRIPE_WEBHOOK_SECRET", None)
    importlib.reload(billing)
    assert billing.enabled() is True
    assert billing.verify_signature(b"{}", "") is False
    os.environ.pop("STRIPE_SECRET_KEY", None)
    print("  security: webhook fails closed without signing secret OK")


if __name__ == "__main__":
    test_hardening()
    print("all security tests passed")
