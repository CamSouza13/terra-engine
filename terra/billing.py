"""Stripe billing: real Checkout sessions and webhook-driven plan changes.

This is the other half of the ``accounts.set_plan`` seam. When the Stripe
environment is configured the platform creates a real hosted Checkout session and
returns its URL; after the customer pays, Stripe calls the webhook and we flip the
workspace's plan. When Stripe is *not* configured everything still works — the
server falls back to a documented stub — so local and trial flows never depend on
live keys.

Configuration (set as Fly secrets, never committed):
  STRIPE_SECRET_KEY       sk_live_… / sk_test_…
  STRIPE_PRICE_PRO        price_…  (the Pro recurring price)
  STRIPE_PRICE_FLEET      price_…  (optional)
  STRIPE_WEBHOOK_SECRET   whsec_…  (verifies webhook authenticity)

Stdlib only: Stripe's REST API over urllib, HMAC signature verification by hand.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import urllib.parse
import urllib.request

from . import accounts as acc

API = "https://api.stripe.com/v1"


def enabled() -> bool:
    return bool(os.environ.get("STRIPE_SECRET_KEY"))


def _price_for(plan: str) -> str | None:
    return {"pro": os.environ.get("STRIPE_PRICE_PRO"),
            "fleet": os.environ.get("STRIPE_PRICE_FLEET")}.get(plan)


def create_checkout_session(workspace_id: int, plan: str,
                            success_url: str, cancel_url: str) -> dict:
    """Create a subscription Checkout session; returns Stripe's JSON (has 'url')."""
    key = os.environ["STRIPE_SECRET_KEY"]
    price = _price_for(plan)
    if not price:
        raise ValueError(f"no Stripe price configured for plan '{plan}'")
    params = {
        "mode": "subscription",
        "success_url": success_url,
        "cancel_url": cancel_url,
        "client_reference_id": str(workspace_id),
        "line_items[0][price]": price,
        "line_items[0][quantity]": "1",
        "metadata[workspace_id]": str(workspace_id),
        "metadata[plan]": plan,
        "subscription_data[metadata][workspace_id]": str(workspace_id),
        "subscription_data[metadata][plan]": plan,
    }
    body = urllib.parse.urlencode(params).encode()
    req = urllib.request.Request(API + "/checkout/sessions", data=body, headers={
        "Authorization": "Bearer " + key,
        "Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read() or b"{}")


def verify_signature(payload: bytes, sig_header: str) -> bool:
    """Verify a Stripe-Signature header. No secret set => accept (dev only)."""
    secret = os.environ.get("STRIPE_WEBHOOK_SECRET")
    if not secret:
        return True
    if not sig_header:
        return False
    parts = dict(p.split("=", 1) for p in sig_header.split(",") if "=" in p)
    t, v1 = parts.get("t"), parts.get("v1")
    if not t or not v1:
        return False
    signed = (t + "." + payload.decode()).encode()
    mac = hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()
    return hmac.compare_digest(mac, v1)


def handle_event(event: dict) -> str | None:
    """Apply a Stripe event to the workspace plan. Returns the new plan or None."""
    typ = event.get("type")
    obj = (event.get("data") or {}).get("object") or {}
    meta = obj.get("metadata") or {}
    ws = meta.get("workspace_id") or obj.get("client_reference_id")
    if not ws:
        return None
    ws = int(ws)
    if typ in ("checkout.session.completed", "customer.subscription.created",
               "customer.subscription.updated"):
        plan = meta.get("plan", "pro")
        acc.set_plan(ws, plan)
        return plan
    if typ == "customer.subscription.deleted":
        acc.set_plan(ws, "free")
        return "free"
    return None
