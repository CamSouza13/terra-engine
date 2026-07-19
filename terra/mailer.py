"""Customer-facing email: one sender, branded templates.

Every message the platform sends to a *customer* — a welcome on signup, an order
confirmation, a support-ticket receipt, and the alert notifications — goes through
``send`` here. It's a no-op unless SMTP is configured (``TERRA_SMTP_*``), so local
and unconfigured deployments never fail; when configured it sends a multipart
text+HTML email from your ``TERRA_SMTP_FROM`` address.

Stdlib only. For good deliverability to real customers, point ``TERRA_SMTP_FROM``
at an address on a verified domain and use a transactional provider.
"""
from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage

BRAND = "Terra"
TAGLINE = "from the mud to the moon"


def configured() -> bool:
    return bool(os.environ.get("TERRA_SMTP_HOST"))


def _wrap_html(heading: str, body_html: str, cta=None) -> str:
    button = ""
    if cta:
        label, url = cta
        button = (f'<a href="{url}" style="display:inline-block;margin-top:18px;'
                  f'background:#12151d;color:#fff;text-decoration:none;font-size:14px;'
                  f'padding:11px 20px;border-radius:10px">{label}</a>')
    return f"""<div style="background:#0d1017;padding:28px 0">
  <div style="max-width:520px;margin:0 auto;background:#ffffff;border-radius:16px;overflow:hidden;font-family:-apple-system,Segoe UI,Roboto,sans-serif">
    <div style="background:#12151d;padding:20px 28px;color:#f4f6fb;font-size:16px;font-weight:600">{BRAND}</div>
    <div style="padding:26px 28px;color:#1c1c1e;line-height:1.6">
      <div style="font-size:18px;font-weight:600;margin-bottom:10px">{heading}</div>
      <div style="font-size:14px;color:#3a3a3e">{body_html}</div>
      {button}
    </div>
    <div style="padding:16px 28px;color:#a1a1a6;font-size:11px;border-top:1px solid #eee">{BRAND} · {TAGLINE}</div>
  </div>
</div>"""


def send(to: str, subject: str, text: str, html: str = None) -> bool:
    """Send one email. Returns True if actually sent, False if not configured."""
    host = os.environ.get("TERRA_SMTP_HOST")
    if not host or not to:
        return False
    port = int(os.environ.get("TERRA_SMTP_PORT", "587"))
    user = os.environ.get("TERRA_SMTP_USER")
    pw = os.environ.get("TERRA_SMTP_PASS")
    sender = os.environ.get("TERRA_SMTP_FROM", user or "hello@terra.local")
    m = EmailMessage()
    m["From"] = f"{BRAND} <{sender}>"
    m["To"] = to
    m["Subject"] = subject
    m.set_content(text)
    if html:
        m.add_alternative(html, subtype="html")
    try:
        with smtplib.SMTP(host, port, timeout=10) as s:
            s.starttls()
            if user and pw:
                s.login(user, pw)
            s.send_message(m)
        return True
    except Exception:
        return False


# ---- templates ---------------------------------------------------------------

def send_welcome(to: str, workspace: str, trial_days: int, origin: str = "") -> bool:
    text = (f"Welcome to {BRAND}.\n\nYour workspace '{workspace}' is ready and your "
            f"{trial_days}-day free trial has started — every feature is unlocked, no "
            f"card required.\n\nOpen the console: {origin or 'your Terra console'}\n\n"
            f"Reply to this email if you need anything.\n\n{BRAND} · {TAGLINE}")
    html = _wrap_html(
        f"Welcome to {BRAND}",
        f"Your workspace <b>{workspace}</b> is ready, and your <b>{trial_days}-day free "
        f"trial</b> has started — every feature unlocked, no card required. Connect a "
        f"node when you're ready and watch your first loop go live.",
        cta=("Open the console", origin or "#") if origin else None)
    return send(to, f"Welcome to {BRAND} — your trial has started", text, html)


def send_order_confirmation(to: str, order_id: int, board: str, tier: str,
                            qty: int, origin: str = "") -> bool:
    text = (f"Thanks for your order (#{order_id}).\n\n{qty} x {board} / {tier}\n\n"
            f"This is a build request, not a charge — we'll email you a quote and lead "
            f"time shortly.\n\n{BRAND} · {TAGLINE}")
    html = _wrap_html(
        f"Order received (#{order_id})",
        f"We've got your request for <b>{qty} × {board}</b> on the <b>{tier}</b> tier. "
        f"This is a build request, not a charge — we'll follow up by email with a quote "
        f"and lead time.")
    return send(to, f"{BRAND} order #{order_id} received", text, html)


def send_invite(to: str, workspace: str, role: str, link: str,
                inviter: str = "") -> bool:
    by = f" by {inviter}" if inviter else ""
    text = (f"You've been invited{by} to join the {workspace} workspace on {BRAND} "
            f"as {role}.\n\nAccept your invite: {link}\n\n{BRAND} · {TAGLINE}")
    html = _wrap_html(
        f"Join {workspace} on {BRAND}",
        f"You've been invited{by} to join the <b>{workspace}</b> workspace as "
        f"<b>{role}</b>. Set a password to accept and you're in.",
        cta=("Accept invite", link))
    return send(to, f"You're invited to {workspace} on {BRAND}", text, html)


def send_ticket_confirmation(to: str, ticket_id: int, subject: str,
                             origin: str = "") -> bool:
    text = (f"We received your support request (#{ticket_id}): {subject}\n\n"
            f"We'll reply to this email address as soon as we can.\n\n{BRAND} · {TAGLINE}")
    html = _wrap_html(
        f"We're on it (#{ticket_id})",
        f"Thanks for reaching out about <b>{subject}</b>. We received your request and "
        f"will reply to this email address as soon as we can.")
    return send(to, f"{BRAND} support #{ticket_id} received", text, html)
