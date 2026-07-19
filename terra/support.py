"""Parts orders and support tickets.

Real, durable storage for two customer actions: ordering node hardware and asking
for help. Both accept a workspace id when the requester is signed in and fall back
to just an email for public (pre-login) submissions. On write we best-effort email
a notification to ``TERRA_SUPPORT_EMAIL`` (via the alerts SMTP config) so requests
don't sit unseen; the row is stored regardless.
"""
from __future__ import annotations

import os
import time

from .accounts import _conn, init_db as _acc_init


def init_db():
    _acc_init()
    c = _conn()
    c.executescript("""
    CREATE TABLE IF NOT EXISTS orders(
      id INTEGER PRIMARY KEY, workspace_id INTEGER, email TEXT, board TEXT,
      tier TEXT, qty INTEGER, notes TEXT, status TEXT DEFAULT 'new', created REAL);
    CREATE TABLE IF NOT EXISTS tickets(
      id INTEGER PRIMARY KEY, workspace_id INTEGER, email TEXT, subject TEXT,
      body TEXT, status TEXT DEFAULT 'open', created REAL);
    """)
    c.commit()
    c.close()


def _notify(subject: str, body: str):
    """Internal heads-up to the team address (if TERRA_SUPPORT_EMAIL is set)."""
    to = os.environ.get("TERRA_SUPPORT_EMAIL")
    if not to:
        return
    try:
        from .mailer import send
        send(to, f"[Terra] {subject}", body)
    except Exception:
        pass


def create_order(email: str, board: str, tier: str, qty: int = 1,
                 notes: str = "", workspace_id: int = None) -> int:
    init_db()
    email = (email or "").strip().lower()
    if "@" not in email:
        raise ValueError("a valid email is required")
    c = _conn()
    oid = c.execute("INSERT INTO orders(workspace_id,email,board,tier,qty,notes,"
                    "status,created) VALUES(?,?,?,?,?,?, 'new', ?)",
                    (workspace_id, email, board, tier, int(qty or 1), notes,
                     time.time())).lastrowid
    c.commit()
    c.close()
    _notify(f"New parts order #{oid}",
            f"{email}\n{qty} x {board} / {tier}\n\n{notes}")
    try:
        from .mailer import send_order_confirmation
        send_order_confirmation(email, oid, board, tier, qty)   # confirm to the customer
    except Exception:
        pass
    return oid


def create_ticket(email: str, subject: str, body: str,
                  workspace_id: int = None) -> int:
    init_db()
    email = (email or "").strip().lower()
    if "@" not in email:
        raise ValueError("a valid email is required")
    c = _conn()
    tid = c.execute("INSERT INTO tickets(workspace_id,email,subject,body,status,"
                    "created) VALUES(?,?,?,?, 'open', ?)",
                    (workspace_id, email, subject, body, time.time())).lastrowid
    c.commit()
    c.close()
    _notify(f"New support ticket #{tid}: {subject}", f"{email}\n\n{body}")
    try:
        from .mailer import send_ticket_confirmation
        send_ticket_confirmation(email, tid, subject)   # confirm to the customer
    except Exception:
        pass
    return tid


def list_orders(workspace_id: int) -> list:
    init_db()
    c = _conn()
    rows = c.execute("SELECT * FROM orders WHERE workspace_id=? ORDER BY id DESC",
                     (workspace_id,)).fetchall()
    c.close()
    return [dict(r) for r in rows]


def list_tickets(workspace_id: int) -> list:
    init_db()
    c = _conn()
    rows = c.execute("SELECT * FROM tickets WHERE workspace_id=? ORDER BY id DESC",
                     (workspace_id,)).fetchall()
    c.close()
    return [dict(r) for r in rows]
