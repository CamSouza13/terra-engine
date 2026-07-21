"""Accounts, workspaces, sessions, and plan/feature gating.

SQLite-backed (under ``$TERRA_HOME``, default ``~/.terra``) with no third-party
dependencies. The plan/feature model here is the source of truth the server
enforces; ``set_plan`` is the entry point a Stripe webhook calls. Passwords use
salted PBKDF2; sessions are opaque tokens.

Plans:
  trial  all features for 14 days, then downgrades to free
  free   dashboard, email alerts, short history
  pro    calibration, control, full history, alerts, API
  fleet  pro plus multi-node fleet, SSO, roles
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sqlite3
import time

HOME = os.environ.get("TERRA_HOME", os.path.expanduser("~/.terra"))
DB_PATH = os.path.join(HOME, "terra.db")
TRIAL_HOURS = 72          # free trial length: full access for 72 hours
SESSION_DAYS = 30

PLAN_FEATURES = {
    "trial": {"calibrate", "control", "alerts", "api", "history", "multi_node"},
    "free":  {"alerts"},
    "pro":   {"calibrate", "control", "alerts", "api", "history", "multi_node"},
    "fleet": {"calibrate", "control", "alerts", "api", "history", "multi_node", "sso", "roles"},
}
FREE_HISTORY_ROWS = 500
FREE_NODE_LIMIT = 1       # free plan: one node; multi_node feature lifts this

# role hierarchy: higher rank can do everything a lower rank can
ROLES = {"owner": 3, "admin": 2, "member": 1, "viewer": 0}
INVITE_DAYS = 7


def role_rank(role: str | None) -> int:
    return ROLES.get(role or "viewer", 0)


def has_role(user: dict, min_role: str) -> bool:
    return role_rank((user or {}).get("role")) >= role_rank(min_role)


def _conn():
    os.makedirs(HOME, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    c = _conn()
    c.executescript("""
    CREATE TABLE IF NOT EXISTS workspaces(
      id INTEGER PRIMARY KEY, name TEXT, plan TEXT DEFAULT 'trial',
      trial_ends REAL, created REAL);
    CREATE TABLE IF NOT EXISTS users(
      id INTEGER PRIMARY KEY, email TEXT UNIQUE, salt TEXT, pw TEXT,
      workspace_id INTEGER, role TEXT DEFAULT 'owner', created REAL);
    CREATE TABLE IF NOT EXISTS sessions(
      token TEXT PRIMARY KEY, user_id INTEGER, created REAL, expires REAL);
    CREATE TABLE IF NOT EXISTS invites(
      token_hash TEXT PRIMARY KEY, workspace_id INTEGER, email TEXT, role TEXT,
      created REAL, expires REAL, used INTEGER DEFAULT 0);
    CREATE TABLE IF NOT EXISTS tokens(
      token_hash TEXT PRIMARY KEY, kind TEXT, user_id INTEGER, email TEXT,
      created REAL, expires REAL, used INTEGER DEFAULT 0);
    """)
    # add the email-verified flag to existing installs (no-op if already present)
    try:
        c.execute("ALTER TABLE users ADD COLUMN verified INTEGER DEFAULT 0")
    except Exception:
        pass
    c.commit()
    c.close()


def _hash(pw: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), 200_000).hex()


def _h(s: str) -> str:
    return hashlib.sha256((s or "").encode()).hexdigest()


def has_users() -> bool:
    init_db()
    c = _conn()
    n = c.execute("SELECT COUNT(*) n FROM users").fetchone()["n"]
    c.close()
    return n > 0


def create_account(email: str, password: str, workspace: str = None,
                   invite_token: str = None) -> dict:
    init_db()
    email = (email or "").strip().lower()
    if "@" not in email or len(password or "") < 8:
        raise ValueError("valid email and an 8+ character password are required")
    c = _conn()
    if c.execute("SELECT 1 FROM users WHERE email=?", (email,)).fetchone():
        c.close()
        raise ValueError("an account with that email already exists")
    now = time.time()
    salt = secrets.token_hex(16)
    if invite_token:
        # joining an existing workspace with the invited role (no new trial)
        inv = c.execute("SELECT * FROM invites WHERE token_hash=?",
                        (_h(invite_token),)).fetchone()
        if not inv or inv["used"] or (inv["expires"] or 0) < now:
            c.close()
            raise ValueError("invalid or expired invite")
        ws = inv["workspace_id"]
        role = inv["role"] if inv["role"] in ROLES else "member"
        uid = c.execute(
            "INSERT INTO users(email, salt, pw, workspace_id, role, created) VALUES(?,?,?,?,?,?)",
            (email, salt, _hash(password, salt), ws, role, now)).lastrowid
        c.execute("UPDATE invites SET used=1 WHERE token_hash=?", (_h(invite_token),))
        c.commit()
        c.close()
        return {"user_id": uid, "workspace_id": ws, "role": role, "token": _new_session(uid)}
    ws = c.execute(
        "INSERT INTO workspaces(name, plan, trial_ends, created) VALUES(?,?,?,?)",
        (workspace or email.split("@")[0], "trial", now + TRIAL_HOURS * 3600, now)
    ).lastrowid
    uid = c.execute(
        "INSERT INTO users(email, salt, pw, workspace_id, role, created) VALUES(?,?,?,?,?,?)",
        (email, salt, _hash(password, salt), ws, "owner", now)
    ).lastrowid
    c.commit()
    c.close()
    return {"user_id": uid, "workspace_id": ws, "role": "owner", "token": _new_session(uid)}


def login(email: str, password: str) -> str | None:
    init_db()
    email = (email or "").strip().lower()
    c = _conn()
    u = c.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    c.close()
    if not u or not hmac.compare_digest(_hash(password, u["salt"]), u["pw"]):
        return None
    return _new_session(u["id"])


def _new_session(uid: int) -> str:
    tok = secrets.token_urlsafe(24)
    now = time.time()
    c = _conn()
    c.execute("INSERT INTO sessions(token, user_id, created, expires) VALUES(?,?,?,?)",
              (tok, uid, now, now + SESSION_DAYS * 86400))
    c.commit()
    c.close()
    return tok


def logout(token: str):
    c = _conn()
    c.execute("DELETE FROM sessions WHERE token=?", (token,))
    c.commit()
    c.close()


def session_user(token: str) -> dict | None:
    if not token:
        return None
    c = _conn()
    s = c.execute("SELECT * FROM sessions WHERE token=?", (token,)).fetchone()
    if not s or s["expires"] < time.time():
        c.close()
        return None
    u = c.execute("SELECT * FROM users WHERE id=?", (s["user_id"],)).fetchone()
    ws = c.execute("SELECT * FROM workspaces WHERE id=?", (u["workspace_id"],)).fetchone()
    c.close()
    verified = bool(u["verified"]) if "verified" in u.keys() else True
    return {"user_id": u["id"], "email": u["email"], "role": u["role"],
            "workspace_id": ws["id"], "workspace": ws["name"],
            "plan": effective_plan(ws), "raw_plan": ws["plan"],
            "trial_ends": ws["trial_ends"], "verified": verified}


def workspace_user(workspace_id: int) -> dict | None:
    """A user-shaped dict for a workspace, for API-key (non-human) access."""
    c = _conn()
    ws = c.execute("SELECT * FROM workspaces WHERE id=?", (workspace_id,)).fetchone()
    c.close()
    if not ws:
        return None
    return {"user_id": 0, "email": "apikey", "role": "service",
            "workspace_id": ws["id"], "workspace": ws["name"],
            "plan": effective_plan(ws), "raw_plan": ws["plan"],
            "trial_ends": ws["trial_ends"]}


def create_invite(workspace_id: int, email: str, role: str = "member") -> str:
    """Create a one-time invite to join a workspace; returns the raw token."""
    init_db()
    if role not in ROLES or role == "owner":
        role = "member"
    tok = "inv_" + secrets.token_urlsafe(18)
    now = time.time()
    c = _conn()
    c.execute("INSERT INTO invites(token_hash,workspace_id,email,role,created,expires,used) "
              "VALUES(?,?,?,?,?,?,0)",
              (_h(tok), workspace_id, (email or "").strip().lower(), role,
               now, now + INVITE_DAYS * 86400))
    c.commit()
    c.close()
    return tok


def list_invites(workspace_id: int) -> list:
    init_db()
    c = _conn()
    rows = c.execute("SELECT email,role,created,expires FROM invites "
                     "WHERE workspace_id=? AND used=0 AND expires>? ORDER BY created DESC",
                     (workspace_id, time.time())).fetchall()
    c.close()
    return [dict(r) for r in rows]


def list_members(workspace_id: int) -> list:
    init_db()
    c = _conn()
    rows = c.execute("SELECT id,email,role,created FROM users WHERE workspace_id=? "
                     "ORDER BY created", (workspace_id,)).fetchall()
    c.close()
    return [dict(r) for r in rows]


def _owner_count(c, workspace_id: int) -> int:
    return c.execute("SELECT COUNT(*) n FROM users WHERE workspace_id=? AND role='owner'",
                     (workspace_id,)).fetchone()["n"]


def set_role(workspace_id: int, user_id: int, role: str) -> bool:
    if role not in ROLES:
        return False
    c = _conn()
    u = c.execute("SELECT * FROM users WHERE id=? AND workspace_id=?",
                  (user_id, workspace_id)).fetchone()
    if not u:
        c.close()
        return False
    # never demote the last owner
    if u["role"] == "owner" and role != "owner" and _owner_count(c, workspace_id) <= 1:
        c.close()
        return False
    c.execute("UPDATE users SET role=? WHERE id=?", (role, user_id))
    c.commit()
    c.close()
    return True


def remove_member(workspace_id: int, user_id: int) -> bool:
    c = _conn()
    u = c.execute("SELECT * FROM users WHERE id=? AND workspace_id=?",
                  (user_id, workspace_id)).fetchone()
    if not u:
        c.close()
        return False
    if u["role"] == "owner" and _owner_count(c, workspace_id) <= 1:
        c.close()
        return False
    c.execute("DELETE FROM sessions WHERE user_id=?", (user_id,))
    c.execute("DELETE FROM users WHERE id=?", (user_id,))
    c.commit()
    c.close()
    return True


def _new_token(kind: str, user_id: int, email: str, ttl_days: int) -> str:
    tok = kind[:3] + "_" + secrets.token_urlsafe(20)
    now = time.time()
    c = _conn()
    c.execute("INSERT INTO tokens(token_hash,kind,user_id,email,created,expires,used) "
              "VALUES(?,?,?,?,?,?,0)",
              (_h(tok), kind, user_id, email, now, now + ttl_days * 86400))
    c.commit()
    c.close()
    return tok


def _consume_token(token: str, kind: str):
    c = _conn()
    row = c.execute("SELECT * FROM tokens WHERE token_hash=? AND kind=?",
                    (_h(token), kind)).fetchone()
    if not row or row["used"] or (row["expires"] or 0) < time.time():
        c.close()
        return None
    c.execute("UPDATE tokens SET used=1 WHERE token_hash=?", (_h(token),))
    c.commit()
    c.close()
    return row


def create_reset(email: str) -> str | None:
    """Password-reset token for an existing account, or None if no such email."""
    init_db()
    email = (email or "").strip().lower()
    c = _conn()
    u = c.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone()
    c.close()
    if not u:
        return None
    return _new_token("reset", u["id"], email, ttl_days=1)


def reset_password(token: str, new_password: str) -> bool:
    init_db()
    if len(new_password or "") < 8:
        return False
    row = _consume_token(token, "reset")
    if not row:
        return False
    salt = secrets.token_hex(16)
    c = _conn()
    c.execute("UPDATE users SET salt=?, pw=? WHERE id=?",
              (salt, _hash(new_password, salt), row["user_id"]))
    c.execute("DELETE FROM sessions WHERE user_id=?", (row["user_id"],))  # log out everywhere
    c.commit()
    c.close()
    return True


def create_verify(user_id: int, email: str) -> str:
    init_db()
    return _new_token("verify", user_id, email, ttl_days=7)


def verify_email(token: str) -> bool:
    init_db()
    row = _consume_token(token, "verify")
    if not row:
        return False
    c = _conn()
    c.execute("UPDATE users SET verified=1 WHERE id=?", (row["user_id"],))
    c.commit()
    c.close()
    return True


def delete_workspace(workspace_id: int) -> bool:
    """Permanently delete a workspace and all data scoped to it (GDPR/CCPA)."""
    init_db()
    c = _conn()
    uids = [r["id"] for r in c.execute("SELECT id FROM users WHERE workspace_id=?",
                                       (workspace_id,)).fetchall()]
    for uid in uids:
        c.execute("DELETE FROM sessions WHERE user_id=?", (uid,))
        c.execute("DELETE FROM tokens WHERE user_id=?", (uid,))
    c.execute("DELETE FROM users WHERE workspace_id=?", (workspace_id,))
    c.execute("DELETE FROM invites WHERE workspace_id=?", (workspace_id,))
    c.execute("DELETE FROM workspaces WHERE id=?", (workspace_id,))
    # best-effort cleanup of data owned by other modules (tables may not all exist)
    for tbl in ("nodes", "node_creds", "enroll_tokens", "api_keys", "alert_rules",
                "alert_events", "orders", "tickets", "audit"):
        try:
            c.execute(f"DELETE FROM {tbl} WHERE workspace_id=?", (workspace_id,))
        except Exception:
            pass
    c.commit()
    c.close()
    return True


def effective_plan(ws) -> str:
    plan = ws["plan"]
    if plan == "trial" and ws["trial_ends"] and ws["trial_ends"] < time.time():
        return "free"
    return plan


def has_feature(user: dict, feature: str) -> bool:
    return feature in PLAN_FEATURES.get(user.get("plan", "free"), set())


def set_plan(workspace_id: int, plan: str):
    """Set a workspace's plan. A Stripe webhook calls this on checkout/renewal."""
    c = _conn()
    c.execute("UPDATE workspaces SET plan=? WHERE id=?", (plan, workspace_id))
    c.commit()
    c.close()


def trial_days_left(user: dict) -> int:
    te = user.get("trial_ends")
    if user.get("raw_plan") != "trial" or not te:
        return 0
    return max(0, int((te - time.time()) / 86400 + 0.5))


def trial_hours_left(user: dict) -> int:
    te = user.get("trial_ends")
    if user.get("raw_plan") != "trial" or not te:
        return 0
    return max(0, int((te - time.time()) / 3600 + 0.5))
