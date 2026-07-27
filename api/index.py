"""Vercel serverless entrypoint.

Runs the entire Terra console (terra/server.py) as one Python function. Vercel
invocations are short-lived and stateless, so this adapter drives the existing
http.server request handler headlessly per request instead of running a
long-lived socket server.

Persistence note: TERRA_HOME points at /tmp, which is writable but ephemeral on
serverless (it resets on a cold start). For durable accounts/history, set the
TURSO_* / external-DB env vars (step 2) — until then this is a working but
non-persistent deployment, fine for a live demo.
"""
import io
import json
import os

os.environ.setdefault("TERRA_HOME", "/tmp/terra-data")
os.environ.setdefault("TERRA_AUTH", "1")
os.environ.setdefault("TERRA_SERVERLESS", "1")
os.makedirs(os.environ["TERRA_HOME"], exist_ok=True)

from terra import server as S  # noqa: E402

_pf = S.Platform()
_Handler = S.make_handler(_pf)


class _Driver(_Handler):
    """Drive the BaseHTTPRequestHandler routing without a socket."""

    def __init__(self, environ):
        self.command = environ["REQUEST_METHOD"]
        qs = environ.get("QUERY_STRING", "")
        self.path = environ.get("PATH_INFO", "/") + (("?" + qs) if qs else "")
        length = int(environ.get("CONTENT_LENGTH") or 0)
        self.rfile = io.BytesIO(environ["wsgi.input"].read(length) if length else b"")
        self.wfile = io.BytesIO()
        self.request_version = "HTTP/1.1"
        self.client_address = (environ.get("REMOTE_ADDR", "127.0.0.1"), 0)
        import email.message
        h = email.message.Message()
        for k, v in environ.items():
            if k.startswith("HTTP_"):
                h[k[5:].replace("_", "-").title()] = v
        if environ.get("CONTENT_TYPE"):
            h["Content-Type"] = environ["CONTENT_TYPE"]
        if environ.get("CONTENT_LENGTH"):
            h["Content-Length"] = environ["CONTENT_LENGTH"]
        self.headers = h
        self._status = 200
        self._headers = []

    # capture response instead of writing HTTP wire format
    def log_message(self, *a, **k):
        pass

    def send_response(self, code, message=None):
        self._status = code

    def send_response_only(self, code, message=None):
        self._status = code

    def send_header(self, key, value):
        self._headers.append((key, str(value)))

    def end_headers(self):
        pass

    def date_time_string(self, timestamp=None):
        return ""

    def version_string(self):
        return "terra"


def app(environ, start_response):
    d = _Driver(environ)
    try:
        method = d.command
        if method == "GET":
            d.do_GET()
        elif method == "POST":
            d.do_POST()
        elif method == "OPTIONS":
            d.do_OPTIONS()
        else:
            d._status = 405
            d._headers = [("Content-Type", "application/json")]
            d.wfile = io.BytesIO(b'{"error":"method not allowed"}')
    except Exception as e:  # never 500 blank — return a JSON error
        d._status = 500
        d._headers = [("Content-Type", "application/json")]
        d.wfile = io.BytesIO(json.dumps({"error": str(e)}).encode())

    body = d.wfile.getvalue()
    headers = d._headers or [("Content-Type", "application/json")]
    if not any(k.lower() == "content-length" for k, _ in headers):
        headers.append(("Content-Length", str(len(body))))
    start_response(f"{d._status} OK", headers)
    return [body]
