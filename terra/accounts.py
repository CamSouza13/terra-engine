"""Accounts, workspaces, sessions, and plan/paywall logic.

A real, dependency-free auth + billing-state layer for the platform, backed by
SQLite under ``$TERRA_HOME`` (default ``~/.terra``). It is the seam a hosted
deployment swaps for Postgres + Stripe: the plan/feature model here is the
source of truth the server enforces, and ``set_plan`` is what a Stripe webhook
would call. Passwords are salted PBKDF2 (stdlib); sessions are opaque tokens.

Plans and what they unlock:
  trial  — everything, for 14 days, then auto-downgrades to free
  free   — dashboard + email alerts + short history; no calibration or control
  pro    — calibration, control/autonomy, full history, alerts, API
  fleet  — pro + multi-node fleet, SSO, roles
"""
from __future__ import annotations

import hashlib
import os
import secrets
import sqlite3
import time

HOME = os.environ.get("TERRA_HOME", os.path.expanduser("~/.terra"))
DB_PATH = os.path.join(HOME, "terra.db")
TRIAL_DAYS = 14
SESSION_DAYS = 30

PLAN_FEATURES = {
    "trial": {"calibrate", "control", "alerts", "api", "history", "multi_node"},
    "free":  {"alerts"},
    "pro":   {"calibrate", "control", "alerts", "api", "history"},
    "fleet": {"calibrate", "control", "alerts", "api", "history", "multi_node", "sso", "roles"},
}
FREE_HISTORY_ROWS = 500


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
    """)
    c.commit()
    c.close()


def _hash(pw: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), 200_000).hex()


def has_users() -> bool:
    init_db()
    c = _conn()
    n = c.execute("SELECT COUNT(*) n FROM users").fetchone()["n"]
    c.close()
    return n > 0


def create_account(email: str, password: str, workspace: str = None) -> dict:
    init_db()
    email = (email or "").strip().lower()
    if "@" not in email or len(password or "") < 8:
        raise ValueError("valid email and an 8+ character password are required")
    c = _conn()
    if c.execute("SELECT 1 FROM users WHERE email=?", (email,)).fetchone():
        c.close()
        raise ValueError("an account with that email already exists")
    now = time.time()
    ws = c.execute(
        "INSERT INTO workspaces(name, plan, trial_ends, created) VALUES(?,?,?,?)",
        (workspace or email.split("@")[0], "trial", now + TRIAL_DAYS * 86400, now)
    ).lastrowid
    salt = secrets.token_hex(16)
    uid = c.execute(
        "INSERT INTO users(email, salt, pw, workspace_id, role, created) VALUES(?,?,?,?,?,?)",
        (email, salt, _hash(password, salt), ws, "owner", now)
    ).lastrowid
    c.commit()
    c.close()
    return {"user_id": uid, "workspace_id": ws, "token": _new_session(uid)}


def login(email: str, password: str) -> str | None:
    init_db()
    email = (email or "").strip().lower()
    c = _conn()
    u = c.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    c.close()
    if not u or _hash(password, u["salt"]) != u["pw"]:
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
    return {"user_id": u["id"], "email": u["email"], "role": u["role"],
            "workspace_id": ws["id"], "workspace": ws["name"],
            "plan": effective_plan(ws), "raw_plan": ws["plan"],
            "trial_ends": ws["trial_ends"]}


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
