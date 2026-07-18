"""Alert rules, evaluation, and delivery.

Turns the engine's live estimate into notifications the way the pricing page
promises: a workspace defines rules ("dissolved_oxygen below 5", "nis above 12",
"crash risk over 0.3"), and on every engine step the platform evaluates them and,
when one fires (subject to a per-rule cooldown so it never spams), records an
event and delivers it — email via SMTP, Slack via incoming webhook, or a plain
webhook POST. Everything is stdlib; delivery config comes from the environment,
and if none is set the event is still recorded so the console shows it.

Rule metrics:
  <channel key>   a live channel value (e.g. "dissolved_oxygen")
  nis             the filter's normalized innovation squared (consistency)
  hidden          the primary hidden state estimate
  risk:<name>     the breach probability of a named safety target (0..1)
"""
from __future__ import annotations

import json
import os
import smtplib
import time
import urllib.request
from email.message import EmailMessage

from .accounts import _conn, init_db as _acc_init

DEFAULT_COOLDOWN = 300  # seconds between repeat firings of the same rule


def init_db():
    _acc_init()
    c = _conn()
    c.executescript("""
    CREATE TABLE IF NOT EXISTS alert_rules(
      id INTEGER PRIMARY KEY, workspace_id INTEGER, node_id TEXT,
      name TEXT, metric TEXT, op TEXT, threshold REAL,
      delivery TEXT DEFAULT 'none', dest TEXT, cooldown INTEGER DEFAULT 300,
      enabled INTEGER DEFAULT 1, created REAL, last_fired REAL);
    CREATE TABLE IF NOT EXISTS alert_events(
      id INTEGER PRIMARY KEY, workspace_id INTEGER, node_id TEXT, rule_id INTEGER,
      name TEXT, message TEXT, value REAL, created REAL, delivered INTEGER DEFAULT 0);
    """)
    c.commit()
    c.close()


def create_rule(workspace_id: int, name: str, metric: str, op: str, threshold: float,
                node_id: str = "*", delivery: str = "none", dest: str = "",
                cooldown: int = DEFAULT_COOLDOWN) -> int:
    init_db()
    if op not in (">", "<"):
        raise ValueError("op must be '>' or '<'")
    if delivery not in ("none", "email", "slack", "webhook"):
        raise ValueError("delivery must be none|email|slack|webhook")
    c = _conn()
    rid = c.execute(
        "INSERT INTO alert_rules(workspace_id,node_id,name,metric,op,threshold,"
        "delivery,dest,cooldown,enabled,created) VALUES(?,?,?,?,?,?,?,?,?,1,?)",
        (workspace_id, node_id or "*", name, metric, op, float(threshold),
         delivery, dest, int(cooldown), time.time())).lastrowid
    c.commit()
    c.close()
    return rid


def list_rules(workspace_id: int, node_id: str = None) -> list:
    init_db()
    c = _conn()
    if node_id:
        rows = c.execute(
            "SELECT * FROM alert_rules WHERE workspace_id=? AND node_id IN(?, '*') "
            "ORDER BY id DESC", (workspace_id, node_id)).fetchall()
    else:
        rows = c.execute("SELECT * FROM alert_rules WHERE workspace_id=? ORDER BY id DESC",
                         (workspace_id,)).fetchall()
    c.close()
    return [dict(r) for r in rows]


def delete_rule(workspace_id: int, rule_id: int):
    c = _conn()
    c.execute("DELETE FROM alert_rules WHERE id=? AND workspace_id=?", (rule_id, workspace_id))
    c.commit()
    c.close()


def set_enabled(workspace_id: int, rule_id: int, enabled: bool):
    c = _conn()
    c.execute("UPDATE alert_rules SET enabled=? WHERE id=? AND workspace_id=?",
              (1 if enabled else 0, rule_id, workspace_id))
    c.commit()
    c.close()


def list_events(workspace_id: int, n: int = 50, node_id: str = None) -> list:
    init_db()
    c = _conn()
    if node_id:
        rows = c.execute(
            "SELECT * FROM alert_events WHERE workspace_id=? AND node_id=? "
            "ORDER BY id DESC LIMIT ?", (workspace_id, node_id, int(n))).fetchall()
    else:
        rows = c.execute("SELECT * FROM alert_events WHERE workspace_id=? "
                         "ORDER BY id DESC LIMIT ?", (workspace_id, int(n))).fetchall()
    c.close()
    return [dict(r) for r in rows]


def active_count(workspace_id: int, node_id: str, window: float = 3600.0) -> int:
    """Events fired for a node in the last `window` seconds — the fleet badge."""
    c = _conn()
    n = c.execute("SELECT COUNT(*) n FROM alert_events WHERE workspace_id=? AND node_id=? "
                  "AND created>?", (workspace_id, node_id, time.time() - window)).fetchone()["n"]
    c.close()
    return int(n)


def _metric_value(metric: str, metrics: dict):
    if metric in metrics:
        return metrics[metric]
    if metric.startswith("risk:"):
        return (metrics.get("risks") or {}).get(metric[5:])
    return (metrics.get("channels") or {}).get(metric)


def evaluate(workspace_id: int, node_id: str, metrics: dict) -> list:
    """Check this workspace/node's rules against a metrics snapshot.

    `metrics` may carry channel values at the top level plus optional
    ``channels`` and ``risks`` sub-dicts. Returns the list of fired events.
    """
    rules = [r for r in list_rules(workspace_id, node_id) if r["enabled"]]
    if not rules:
        return []
    now = time.time()
    fired = []
    c = _conn()
    for r in rules:
        val = _metric_value(r["metric"], metrics)
        if val is None:
            continue
        try:
            val = float(val)
        except Exception:
            continue
        hit = (val > r["threshold"]) if r["op"] == ">" else (val < r["threshold"])
        if not hit:
            continue
        if r["last_fired"] and (now - r["last_fired"]) < (r["cooldown"] or DEFAULT_COOLDOWN):
            continue
        msg = f"{r['name']}: {r['metric']} = {val:.4g} {r['op']} {r['threshold']:g}"
        delivered = _deliver(r, msg)
        c.execute("INSERT INTO alert_events(workspace_id,node_id,rule_id,name,message,"
                  "value,created,delivered) VALUES(?,?,?,?,?,?,?,?)",
                  (workspace_id, node_id, r["id"], r["name"], msg, val, now,
                   1 if delivered else 0))
        c.execute("UPDATE alert_rules SET last_fired=? WHERE id=?", (now, r["id"]))
        fired.append({"rule_id": r["id"], "name": r["name"], "message": msg,
                      "value": val, "delivered": delivered})
    c.commit()
    c.close()
    return fired


# ---- delivery ----------------------------------------------------------------

def _deliver(rule: dict, message: str) -> bool:
    d = rule.get("delivery") or "none"
    try:
        if d == "email":
            return _send_email(rule.get("dest"), rule.get("name"), message)
        if d == "slack":
            return _post_json(rule.get("dest"), {"text": f":rotating_light: {message}"})
        if d == "webhook":
            return _post_json(rule.get("dest"),
                              {"alert": rule.get("name"), "message": message,
                               "value": None, "ts": time.time()})
    except Exception:
        return False
    return False


def _send_email(to: str | None, subject: str | None, body: str) -> bool:
    host = os.environ.get("TERRA_SMTP_HOST")
    if not host or not to:
        return False  # not configured; event is still recorded
    port = int(os.environ.get("TERRA_SMTP_PORT", "587"))
    user = os.environ.get("TERRA_SMTP_USER")
    pw = os.environ.get("TERRA_SMTP_PASS")
    sender = os.environ.get("TERRA_SMTP_FROM", user or "alerts@terra.local")
    m = EmailMessage()
    m["From"] = sender
    m["To"] = to
    m["Subject"] = f"[Terra] {subject}"
    m.set_content(body)
    with smtplib.SMTP(host, port, timeout=10) as s:
        s.starttls()
        if user and pw:
            s.login(user, pw)
        s.send_message(m)
    return True


def _post_json(url: str | None, payload: dict) -> bool:
    if not url:
        return False
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return 200 <= resp.status < 300
