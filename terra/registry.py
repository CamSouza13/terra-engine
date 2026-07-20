"""Node registry, secure enrollment, and workspace API keys.

Three pieces of real infrastructure the platform needs once it leaves a single
laptop:

* **Node registry** — every node that reports in is recorded with its last-seen
  time and last status, so the console's fleet view can show which loops are
  live and which have gone stale.
* **Enrollment** — an owner mints a short-lived enrollment token in the console;
  a physical node redeems it once to receive its own long-lived node key. The
  node then authenticates every ingest with that key instead of trusting the
  network. Tokens and keys are stored hashed.
* **API keys** — per-workspace keys (``sk_live_…``) for the public API. The full
  key is shown once at creation; only a salted hash and a display prefix are
  stored.

All of it is stdlib SQLite in the same ``$TERRA_HOME/terra.db`` as accounts.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time

from .accounts import _conn, init_db as _acc_init

ENROLL_TOKEN_TTL = 3600          # enrollment tokens last one hour
STALE_AFTER = 120.0              # a node unheard-from this long is "stale" (s)
NODE_HIST_CAP = 3000             # per-node heartbeat rows retained on disk


def _node_dir() -> str:
    d = os.path.join(os.environ.get("TERRA_HOME", os.path.expanduser("~/.terra")), "nodes")
    os.makedirs(d, exist_ok=True)
    return d


def _hist_path(node_id: str) -> str:
    safe = "".join(ch for ch in (node_id or "node") if ch.isalnum() or ch in "-_")
    return os.path.join(_node_dir(), safe + ".jsonl")


def append_history(node_id: str, metrics: dict):
    """Append one heartbeat's metrics to the node's on-disk history (capped)."""
    if not node_id:
        return
    rec = {"ts": time.time(), "nis": metrics.get("nis"), "hidden": metrics.get("hidden"),
           "channels": metrics.get("channels") or {}}
    p = _hist_path(node_id)
    try:
        with open(p, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if os.path.getsize(p) > NODE_HIST_CAP * 400:   # occasional trim to the cap
            lines = open(p).read().splitlines()[-NODE_HIST_CAP:]
            with open(p, "w") as f:
                f.write("\n".join(lines) + "\n")
    except Exception:
        pass


def node_history(node_id: str, workspace_id: int, n: int = 300) -> list:
    """Return the last n heartbeats for a node the workspace owns."""
    c = _conn()
    row = c.execute("SELECT workspace_id FROM nodes WHERE node_id=?", (node_id,)).fetchone()
    c.close()
    if not row or row["workspace_id"] != workspace_id:
        return []
    p = _hist_path(node_id)
    if not os.path.exists(p):
        return []
    try:
        lines = open(p).read().splitlines()[-int(n):]
        return [json.loads(x) for x in lines if x.strip()]
    except Exception:
        return []


def init_db():
    _acc_init()
    c = _conn()
    c.executescript("""
    CREATE TABLE IF NOT EXISTS nodes(
      node_id TEXT PRIMARY KEY, workspace_id INTEGER, name TEXT, domain TEXT,
      created REAL, last_seen REAL, last_status TEXT);
    CREATE TABLE IF NOT EXISTS enroll_tokens(
      token_hash TEXT PRIMARY KEY, workspace_id INTEGER, created REAL,
      expires REAL, used_by TEXT);
    CREATE TABLE IF NOT EXISTS node_creds(
      node_id TEXT PRIMARY KEY, workspace_id INTEGER, key_hash TEXT, created REAL);
    CREATE TABLE IF NOT EXISTS api_keys(
      id INTEGER PRIMARY KEY, workspace_id INTEGER, name TEXT, prefix TEXT,
      key_hash TEXT, created REAL, last_used REAL);
    """)
    c.commit()
    c.close()


def _h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


# ---- nodes -------------------------------------------------------------------

def touch_node(node_id: str, workspace_id: int, name: str = None,
               domain: str = None, status: dict = None):
    """Upsert a node's last-seen time and status. Called on every report."""
    init_db()
    import json
    now = time.time()
    c = _conn()
    row = c.execute("SELECT node_id FROM nodes WHERE node_id=?", (node_id,)).fetchone()
    st = json.dumps(status) if status is not None else None
    if row:
        c.execute("UPDATE nodes SET last_seen=?, last_status=COALESCE(?,last_status), "
                  "name=COALESCE(?,name), domain=COALESCE(?,domain) WHERE node_id=?",
                  (now, st, name, domain, node_id))
    else:
        c.execute("INSERT INTO nodes(node_id,workspace_id,name,domain,created,"
                  "last_seen,last_status) VALUES(?,?,?,?,?,?,?)",
                  (node_id, workspace_id, name or node_id, domain, now, now, st))
    c.commit()
    c.close()


def list_nodes(workspace_id: int) -> list:
    init_db()
    import json
    c = _conn()
    rows = c.execute("SELECT * FROM nodes WHERE workspace_id=? ORDER BY name",
                     (workspace_id,)).fetchall()
    c.close()
    now = time.time()
    out = []
    for r in rows:
        age = now - (r["last_seen"] or 0)
        try:
            status = json.loads(r["last_status"]) if r["last_status"] else None
        except Exception:
            status = None
        out.append({"node_id": r["node_id"], "name": r["name"], "domain": r["domain"],
                    "last_seen": r["last_seen"], "age_s": age,
                    "stale": age > STALE_AFTER, "status": status})
    return out


# ---- enrollment --------------------------------------------------------------

def create_enroll_token(workspace_id: int) -> str:
    init_db()
    tok = "et_" + secrets.token_urlsafe(18)
    now = time.time()
    c = _conn()
    c.execute("INSERT INTO enroll_tokens(token_hash,workspace_id,created,expires,used_by) "
              "VALUES(?,?,?,?,NULL)", (_h(tok), workspace_id, now, now + ENROLL_TOKEN_TTL))
    c.commit()
    c.close()
    return tok


def enroll_node(enroll_token: str, node_id: str = None, name: str = None,
                domain: str = None) -> dict | None:
    """Redeem an enrollment token once; return the node's permanent key."""
    init_db()
    c = _conn()
    row = c.execute("SELECT * FROM enroll_tokens WHERE token_hash=?",
                    (_h(enroll_token or ""),)).fetchone()
    if not row or row["used_by"] or (row["expires"] or 0) < time.time():
        c.close()
        return None
    ws = row["workspace_id"]
    node_id = node_id or ("node_" + secrets.token_hex(4))
    node_key = "nk_" + secrets.token_urlsafe(24)
    now = time.time()
    c.execute("UPDATE enroll_tokens SET used_by=? WHERE token_hash=?", (node_id, _h(enroll_token)))
    c.execute("INSERT OR REPLACE INTO node_creds(node_id,workspace_id,key_hash,created) "
              "VALUES(?,?,?,?)", (node_id, ws, _h(node_key), now))
    c.execute("INSERT OR IGNORE INTO nodes(node_id,workspace_id,name,domain,created,last_seen) "
              "VALUES(?,?,?,?,?,?)", (node_id, ws, name or node_id, domain, now, now))
    c.commit()
    c.close()
    return {"node_id": node_id, "node_key": node_key, "workspace_id": ws}


def verify_node_key(node_id: str, key: str) -> int | None:
    if not node_id or not key:
        return None
    c = _conn()
    row = c.execute("SELECT workspace_id,key_hash FROM node_creds WHERE node_id=?",
                    (node_id,)).fetchone()
    c.close()
    if not row or not hmac.compare_digest(row["key_hash"], _h(key)):
        return None
    return row["workspace_id"]


# ---- API keys ----------------------------------------------------------------

def create_api_key(workspace_id: int, name: str = "default") -> dict:
    init_db()
    secret = secrets.token_urlsafe(24)
    key = "sk_live_" + secret
    prefix = "sk_live_" + secret[:6]
    now = time.time()
    c = _conn()
    kid = c.execute("INSERT INTO api_keys(workspace_id,name,prefix,key_hash,created,last_used) "
                    "VALUES(?,?,?,?,?,NULL)",
                    (workspace_id, name, prefix, _h(key), now)).lastrowid
    c.commit()
    c.close()
    return {"id": kid, "name": name, "prefix": prefix, "key": key}


def list_api_keys(workspace_id: int) -> list:
    init_db()
    c = _conn()
    rows = c.execute("SELECT id,name,prefix,created,last_used FROM api_keys "
                     "WHERE workspace_id=? ORDER BY id DESC", (workspace_id,)).fetchall()
    c.close()
    return [dict(r) for r in rows]


def revoke_api_key(workspace_id: int, key_id: int):
    c = _conn()
    c.execute("DELETE FROM api_keys WHERE id=? AND workspace_id=?", (key_id, workspace_id))
    c.commit()
    c.close()


def verify_api_key(key: str) -> int | None:
    if not key or not key.startswith("sk_"):
        return None
    c = _conn()
    row = c.execute("SELECT id,workspace_id FROM api_keys WHERE key_hash=?",
                    (_h(key),)).fetchone()
    if not row:
        c.close()
        return None
    c.execute("UPDATE api_keys SET last_used=? WHERE id=?", (time.time(), row["id"]))
    c.commit()
    c.close()
    return row["workspace_id"]
