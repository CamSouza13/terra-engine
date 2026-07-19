"""Audit log and PDF report tests.

The report test adapts to whether the optional reportlab extra is installed:
with it, we assert real PDF bytes; without it, we assert the documented hint.
Audit and channel-stats run everywhere.
"""
import importlib
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _fresh(tmp):
    from terra import accounts, audit, reports
    for m in (accounts, audit, reports):
        importlib.reload(m)
    accounts.HOME = tmp
    accounts.DB_PATH = os.path.join(tmp, "terra.db")
    accounts.init_db()
    return accounts, audit, reports


def test_audit_append_and_list():
    with tempfile.TemporaryDirectory() as tmp:
        acc, audit, _ = _fresh(tmp)
        ws = acc.create_account("cam@terra.io", "password1")["workspace_id"]
        audit.log(ws, "cam@terra.io", "auth.login", "")
        audit.log(ws, "cam@terra.io", "engine.calibrate", "aquaculture")
        audit.log(ws + 99, "other", "noise", "")   # different workspace
        ev = audit.list_events(ws)
        assert len(ev) == 2
        assert ev[0]["action"] == "engine.calibrate"     # newest first
        assert ev[0]["detail"] == "aquaculture"
        print("  audit: append-only log + per-workspace listing OK")


def test_channel_stats():
    with tempfile.TemporaryDirectory() as tmp:
        _, _, reports = _fresh(tmp)
        hist = [{"channels": {"do": 5.0}}, {"channels": {"do": 7.0}},
                {"channels": {"do": 6.0}}, {"channels": {}}]
        rows = reports.channel_stats(hist, ["do", "ph"])
        do = rows[0]
        assert do[0] == "do" and do[1] == "5" and do[3] == "7" and do[5] == "3"
        assert rows[1] == ["ph", "—", "—", "—", "—", "0"]
        print("  reports: channel_stats min/mean/max/last/N OK")


def test_report_build_or_hint():
    with tempfile.TemporaryDirectory() as tmp:
        _, _, reports = _fresh(tmp)
        ctx = {"workspace": "Terra", "domain": "aquaculture", "source": "replay",
               "cycles": 120, "calibrated": {"k_nit": 0.51},
               "nodes": [{"name": "Pond A", "domain": "aquaculture",
                          "last_seen": 1.0, "stale": False}],
               "alerts": [{"created": 1.0, "message": "Low DO", "delivered": 0}],
               "history": [{"channels": {"dissolved_oxygen": 6.0}}],
               "channels": ["dissolved_oxygen"]}
        if reports.HAS_REPORTLAB:
            pdf = reports.build_pdf(ctx)
            assert pdf[:4] == b"%PDF" and len(pdf) > 1000
            print(f"  reports: built a {len(pdf)}-byte PDF")
        else:
            try:
                reports.build_pdf(ctx)
                assert False, "should have raised without reportlab"
            except RuntimeError as e:
                assert "extra" in str(e)
            print("  reports: correct hint when reportlab absent")


if __name__ == "__main__":
    test_audit_append_and_list()
    test_channel_stats()
    test_report_build_or_hint()
    print("all report/audit tests passed")
