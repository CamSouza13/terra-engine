# Terra Platform — Security Review

Manual review of the hosted platform (`terra/server.py` and the auth/registry/billing
layer), July 2026. Findings are evidence-based with file:line references. This is a
first-pass application review, not a substitute for a professional pentest before
onboarding third-party customer data.

Severity: **High** = fix before any external user with real data · **Medium** = fix
before general availability · **Low** = hardening / defense-in-depth.

---

## High

### H1 · Unauthenticated write to the shared engine — `/api/ingest`
`server.py:690` — `POST /api/ingest` calls `pf.ingest(raw)` with **no auth check**.
Any anonymous caller can overwrite the running engine's active data log and reset its
state. Combined with H4 (single shared engine), one anonymous request corrupts what
every signed-in user sees.
**Fix:** gate behind `_gate("control","member")` (or an API/node key); or remove the
legacy unauthenticated `/api/ingest` entirely and keep only `/api/v1/ingest` (which is
key-authenticated).

### H2 · Unauthenticated read of engine status/config/state
`server.py:526–531` — `GET /api/status`, `/api/config`, `/api/state` return the live
engine's configuration and full estimate with no auth. Information disclosure; in a
multi-tenant deployment it exposes the shared engine to anonymous users.
**Fix:** require `self._user()` (and, once per-tenant engines exist, scope to the
caller's workspace).

### H3 · No request body size limit — memory-exhaustion DoS
`server.py:687–688` — `n = int(Content-Length); raw = self.rfile.read(n)`. A client can
declare a huge Content-Length and stream a large body; the server reads it entirely
into memory. Trivial single-request DoS on a 512 MB machine.
**Fix:** cap `n` (e.g. reject > 2 MB with 413), and read in bounded chunks.

### H4 · Single shared engine across all workspaces (tenant isolation)
The platform runs one global `Platform` instance (`pf`). `/api/status|state|config|
ingest|control|offline|calibrate` all act on that one engine regardless of the caller's
workspace. Accounts, registry, alerts, keys, and audit **are** workspace-scoped, but the
live engine/dashboard is **not**. Two customers would share and could overwrite one
engine view.
**Fix:** instantiate/select a `Platform` per workspace (keyed by `workspace_id`), or make
the platform authoritative only over data the node pushes via `/api/v1/ingest` and drop
the shared local engine from the multi-tenant path.

---

## Medium

### M1 · Stripe webhook forgeable when secret is unset
`billing.py verify_signature` returns `True` when `STRIPE_WEBHOOK_SECRET` is absent. If
`STRIPE_SECRET_KEY` is set in prod but the webhook secret isn't, `POST
/api/billing/webhook` accepts unsigned events — an attacker can forge
`checkout.session.completed` and upgrade their own workspace to any plan for free.
**Fix:** when billing is enabled, require the webhook secret; fail closed (reject) if it
is missing rather than accepting.

### M2 · Rate limiting only covers login
`server.py` — only `/api/auth/login` is throttled. `/api/auth/signup`,
`/api/auth/reset-request` (email bombing / user-enumeration by side effects),
`/api/v1/ingest`, `/api/v1/enroll`, and API-key auth are unthrottled.
**Fix:** add per-IP and per-account throttles to signup, reset-request, and the `/api/v1`
surface.

### M3 · Permissive CORS (`Access-Control-Allow-Origin: *`)
`server.py:430` — every response allows any origin. Because auth is a bearer header (not
cookies), this doesn't auto-leak credentials, but it lets any website script the API.
**Fix:** allowlist your own origin(s) for the console API; keep `*` only on genuinely
public, unauthenticated endpoints if needed.

### M4 · Not a hardened production server
Runs on stdlib `ThreadingHTTPServer` — unbounded thread/connection creation, no request
timeouts, no slowloris protection, single process. Fine for a demo, not for exposed prod.
**Fix:** front it with a real reverse proxy / WAF, set connection and body timeouts, and
consider a production ASGI/WSGI server for the API.

### M5 · Billing "stub" performs free plan upgrades
`server.py:774–793` — when Stripe is not configured, `POST /api/billing/upgrade` (admin
role) sets the plan directly. Convenient for local dev, but on a live deploy without
Stripe it's a self-serve free upgrade.
**Fix:** disable the stub unless an explicit `TERRA_ALLOW_STUB_BILLING` flag is set.

---

## Low / hardening

- **L1 · Non-constant-time secret comparison.** `accounts.login` and
  `registry.verify_node_key` compare hashes with `!=`. Use `hmac.compare_digest`.
- **L2 · Email verification is soft** — not enforced for any gated action; enables
  throwaway accounts. Enforce for sensitive actions if abuse appears.
- **L3 · Cookie auth path exists but unused** (`_token` reads a `terra_session` cookie the
  server never sets). Remove it, or if you add cookie sessions, set `HttpOnly`,
  `Secure`, `SameSite=Lax` and add CSRF protection — and note `*` CORS is incompatible
  with cookie credentials.
- **L4 · No dependency / vulnerability scanning.** Enable Dependabot / `pip-audit` in CI.
- **L5 · No verified backups / DR** for the SQLite volume. Snapshot the Fly volume on a
  schedule and test restore.
- **L6 · No 2FA.** Consider TOTP for owner/admin accounts before enterprise customers.

---

## What is already sound

Parameterized SQL throughout (SQL-injection resistant) · PBKDF2-SHA256 password hashing
(200k iterations) with per-user salts · opaque expiring session tokens · hashed API keys,
node keys, and enrollment/verify/reset tokens · role- and plan-based access control
enforced server-side · append-only audit log · security headers (`X-Frame-Options: DENY`,
`nosniff`, `no-referrer`) · TLS in transit · Stripe webhook signature *mechanism* present
(see M1) · secrets kept in env/Fly secrets, not in code.

## Not applicable at this stage

Zero-knowledge proofs, Merkle trees, and HSMs have no use case in this product. Honeypots,
tripwires, and red/blue-team exercises are premature for a pre-GA build — close the High
and Medium findings and get a professional review first. (A hash-chained audit log for
tamper-evidence is a reasonable optional add later.)

## Suggested order of work

1. H1, H2, H3 (auth on engine endpoints + body-size cap) — small, high-impact code changes.
2. H4 (tenant isolation) — the real architectural item; decide per-workspace engine vs. node-push-only.
3. M1, M2, M5 (fail-closed webhook, rate limits, disable stub billing).
4. M3, M4 (CORS allowlist, reverse proxy / WAF).
5. Low items as hardening.
