"""Append-only audit log.

A durable record of who did what in a workspace — sign-ins, plan changes,
calibration runs, control toggles, alert-rule edits, key and node lifecycle,
and orders. This is a Fleet-tier expectation (and a compliance one): the log is
append-only from the application's side (no update/delete API) and every entry is
timestamped with an actor. Stdlib SQLite in the shared database.
"""
from __future__ import annotations

import time

from .accounts import _conn, init_db as _acc_init


def init_db():
    _acc_init()
    c = _conn()
    c.executescript("""
    CREATE TABLE IF NOT EXISTS audit(
      id INTEGER PRIMARY KEY, workspace_id INTEGER, actor TEXT, action TEXT,
      detail TEXT, created REAL);
    """)
    c.commit()
    c.close()


def log(workspace_id, actor: str, action: str, detail: str = ""):
    """Record an event. Best-effort: never raises into the request path."""
    try:
        init_db()
        c = _conn()
        c.execute("INSERT INTO audit(workspace_id,actor,action,detail,created) "
                  "VALUES(?,?,?,?,?)",
                  (workspace_id if workspace_id is not None else -1,
                   actor or "anon", action, detail, time.time()))
        c.commit()
        c.close()
    except Exception:
        pass


def list_events(workspace_id: int, n: int = 100) -> list:
    init_db()
    c = _conn()
    rows = c.execute("SELECT actor,action,detail,created FROM audit WHERE workspace_id=? "
                     "ORDER BY id DESC LIMIT ?", (workspace_id, int(n))).fetchall()
    c.close()
    return [dict(r) for r in rows]
