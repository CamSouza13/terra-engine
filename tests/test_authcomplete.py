"""Auth completeness (reset, verify, delete), per-node history, and hardware driver."""
import importlib
import itertools
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _fresh(tmp):
    os.environ["TERRA_HOME"] = tmp
    from terra import accounts, registry
    for m in (accounts, registry):
        importlib.reload(m)
    accounts.HOME = tmp
    accounts.DB_PATH = os.path.join(tmp, "terra.db")
    accounts.init_db()
    return accounts, registry


def test_password_reset_and_verify():
    with tempfile.TemporaryDirectory() as tmp:
        acc, _ = _fresh(tmp)
        r = acc.create_account("cam@t.io", "password1", "Terra")
        # reset for unknown email returns None; known email returns a token
        assert acc.create_reset("nobody@t.io") is None
        tok = acc.create_reset("cam@t.io")
        assert tok
        assert acc.reset_password(tok, "short") is False        # weak rejected
        assert acc.reset_password(tok, "newpassword1") is True
        assert acc.reset_password(tok, "again12345") is False    # one-time
        assert acc.login("cam@t.io", "password1") is None        # old password gone
        assert acc.login("cam@t.io", "newpassword1")             # new works

        # email verification
        assert acc.session_user(acc.login("cam@t.io", "newpassword1"))["verified"] is False
        vt = acc.create_verify(r["user_id"], "cam@t.io")
        assert acc.verify_email(vt) is True
        assert acc.verify_email(vt) is False                     # one-time
        assert acc.session_user(acc.login("cam@t.io", "newpassword1"))["verified"] is True
        print("  auth: reset (one-time, invalidates old) + verify OK")


def test_delete_workspace_cascade():
    with tempfile.TemporaryDirectory() as tmp:
        acc, reg = _fresh(tmp)
        r = acc.create_account("own@t.io", "password1", "Terra")
        ws = r["workspace_id"]
        reg.create_api_key(ws, "k")
        reg.touch_node("n1", ws, domain="aquaculture", status={"cycles": 1})
        assert len(reg.list_nodes(ws)) == 1 and len(reg.list_api_keys(ws)) == 1
        assert acc.delete_workspace(ws) is True
        assert acc.session_user(r["token"]) is None              # sessions gone
        assert reg.list_nodes(ws) == [] and reg.list_api_keys(ws) == []
        print("  auth: workspace deletion cascades across tables OK")


def test_node_history_scoped():
    with tempfile.TemporaryDirectory() as tmp:
        acc, reg = _fresh(tmp)
        a = acc.create_account("a@t.io", "password1")["workspace_id"]
        b = acc.create_account("b@t.io", "password1")["workspace_id"]
        reg.touch_node("n1", a, domain="soil")
        for do in (5.0, 6.0, 7.0):
            reg.append_history("n1", {"nis": 1.0, "channels": {"do": do}})
        assert len(reg.node_history("n1", a)) == 3
        assert reg.node_history("n1", b) == []                   # other workspace can't read
        print("  analytics: per-node history stored and workspace-scoped OK")


def test_hardware_driver_runs():
    from terra.node import HardwareDriver, NodeRunner, NodeConfig, ConstantProbe, LinearProbe
    from terra.domains import aquaculture
    spec, _ = aquaculture.simulate()
    keys = list(spec.channels)[:2]
    probes = {keys[0]: ConstantProbe(1.0), keys[1]: LinearProbe(lambda: 2.0, scale=1.5, offset=0.1)}
    clk = itertools.count(0, 60)
    drv = HardwareDriver(spec, probes, u=[0.22, 1.0], poll_interval_s=0, max_cycles=5,
                         _clock=lambda: next(clk), _sleep=lambda s: None)
    fd, tmp = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    os.remove(tmp)
    try:
        res = NodeRunner(spec, drv, NodeConfig(state_path=tmp, max_cycles=5)).run()
        assert res["run_cycles"] == 5
        assert abs(probes[keys[1]].read() - 3.1) < 1e-9          # 1.5*2 + 0.1
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    print("  hardware: LinearProbe calibration + HardwareDriver through the runner OK")


if __name__ == "__main__":
    test_password_reset_and_verify()
    test_delete_workspace_cascade()
    test_node_history_scoped()
    test_hardware_driver_runs()
    print("all auth/analytics/hardware tests passed")
