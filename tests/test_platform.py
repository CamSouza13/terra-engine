"""Accounts, alerts, registry, and billing — the platform/commercial layer.

All stdlib SQLite in a throwaway TERRA_HOME, so these run anywhere with no network.
"""
import importlib
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _fresh(tmp):
    """Point the account/registry/alerts modules at a clean temp DB."""
    from terra import accounts, alerts, registry, billing
    for m in (accounts, alerts, registry, billing):
        importlib.reload(m)
    accounts.HOME = tmp
    accounts.DB_PATH = os.path.join(tmp, "terra.db")
    accounts.init_db()
    return accounts, alerts, registry, billing


def test_accounts_trial_login_and_plan():
    with tempfile.TemporaryDirectory() as tmp:
        acc, _, _, _ = _fresh(tmp)
        r = acc.create_account("cam@terra.io", "password1", "Terra")
        assert r["workspace_id"] and r["token"]
        u = acc.session_user(r["token"])
        assert u["plan"] == "trial" and acc.has_feature(u, "calibrate")
        # duplicate + weak password rejected
        for bad in (("cam@terra.io", "password1"), ("x@y.io", "short")):
            try:
                acc.create_account(*bad)
                assert False, "should have raised"
            except ValueError:
                pass
        assert acc.login("cam@terra.io", "password1")
        assert acc.login("cam@terra.io", "nope") is None
        acc.set_plan(r["workspace_id"], "free")
        u2 = acc.session_user(r["token"])
        assert u2["plan"] == "free" and not acc.has_feature(u2, "calibrate")
        print("  accounts: trial->features->downgrade OK")


def test_alerts_fire_cooldown_and_events():
    with tempfile.TemporaryDirectory() as tmp:
        acc, alerts, _, _ = _fresh(tmp)
        ws = acc.create_account("a@b.io", "password1")["workspace_id"]
        alerts.create_rule(ws, "Low DO", "dissolved_oxygen", "<", 5.0)
        alerts.create_rule(ws, "Divergence", "nis", ">", 12.0, cooldown=300)
        f1 = alerts.evaluate(ws, "n1", {"channels": {"dissolved_oxygen": 4.0}, "nis": 3})
        assert len(f1) == 1
        f2 = alerts.evaluate(ws, "n1", {"channels": {"dissolved_oxygen": 4.0}, "nis": 3})
        assert f2 == []                                  # cooldown suppresses repeat
        f3 = alerts.evaluate(ws, "n1", {"channels": {"dissolved_oxygen": 6}, "nis": 20})
        assert len(f3) == 1 and "nis" in f3[0]["message"]
        assert alerts.active_count(ws, "n1") == 2
        assert len(alerts.list_events(ws)) == 2
        print("  alerts: fire, cooldown, nis breach, events OK")


def test_registry_enrollment_and_api_keys():
    with tempfile.TemporaryDirectory() as tmp:
        acc, _, reg, _ = _fresh(tmp)
        ws = acc.create_account("c@d.io", "password1")["workspace_id"]
        tok = reg.create_enroll_token(ws)
        node = reg.enroll_node(tok, name="Pond A", domain="aquaculture")
        assert node and node["node_key"].startswith("nk_")
        assert reg.verify_node_key(node["node_id"], node["node_key"]) == ws
        assert reg.verify_node_key(node["node_id"], "nk_wrong") is None
        assert reg.enroll_node(tok) is None              # one-time token
        reg.touch_node(node["node_id"], ws, domain="aquaculture", status={"cycles": 3})
        assert reg.list_nodes(ws)[0]["name"] == "Pond A"
        k = reg.create_api_key(ws, "prod")
        assert reg.verify_api_key(k["key"]) == ws
        assert reg.verify_api_key("sk_live_nope") is None
        reg.revoke_api_key(ws, k["id"])
        assert reg.verify_api_key(k["key"]) is None      # revoked
        print("  registry: enroll, node key, api key, revoke OK")


def test_billing_signature_and_events():
    with tempfile.TemporaryDirectory() as tmp:
        acc, _, _, billing = _fresh(tmp)
        ws = acc.create_account("e@f.io", "password1")["workspace_id"]
        assert billing.enabled() is False
        # with no webhook secret, signatures are accepted (dev)
        assert billing.verify_signature(b"{}", "") is True
        # with a secret, a hand-computed v1 verifies and a bad one fails
        import hashlib
        import hmac
        os.environ["STRIPE_WEBHOOK_SECRET"] = "whsec_test"
        payload = b'{"hello":"world"}'
        mac = hmac.new(b"whsec_test", (b"123." + payload), hashlib.sha256).hexdigest()
        assert billing.verify_signature(payload, f"t=123,v1={mac}") is True
        assert billing.verify_signature(payload, "t=123,v1=deadbeef") is False
        del os.environ["STRIPE_WEBHOOK_SECRET"]
        # a completed checkout upgrades the workspace plan
        billing.handle_event({"type": "checkout.session.completed",
                              "data": {"object": {"metadata": {"workspace_id": str(ws),
                                                               "plan": "pro"}}}})
        assert acc.session_user(acc.login("e@f.io", "password1"))["plan"] == "pro"
        print("  billing: signature verify + webhook plan change OK")


if __name__ == "__main__":
    test_accounts_trial_login_and_plan()
    test_alerts_fire_cooldown_and_events()
    test_registry_enrollment_and_api_keys()
    test_billing_signature_and_events()
    print("all platform tests passed")
