"""Tests for the CSV ingestion path: replay real-shaped logs through a domain."""
import csv
import os
import sys
import tempfile
from datetime import datetime, timedelta

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from terra import EngineConfig
from terra.domains import aquaculture
from terra.ingest import run_csv, load_csv

CFG = EngineConfig(forecast_horizon_h=12, forecast_dt=0.25, forecast_samples=200)


def _write_csv(drop=(), numeric_time=False):
    """Export the aquaculture sim to a temp CSV; optionally drop channel columns."""
    spec, sim = aquaculture.simulate()
    t = sim["t"]
    start = datetime(2026, 7, 1)
    cols = ["timestamp", "TAN", "NO2", "NO3", "DO", "excretion_kg_h"]
    cols = [c for c in cols if c not in drop]
    fd, path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i in range(len(t)):
            m = sim["meas"][i]
            ts = float(t[i]) if numeric_time else (start + timedelta(hours=float(t[i]))).isoformat()
            row = {"timestamp": ts, "excretion_kg_h": f"{sim['u'][i]:.4f}"}
            for ch in ("TAN", "NO2", "NO3", "DO"):
                row[ch] = f"{m[ch]:.4f}" if ch in m else ""
            w.writerow([row[c] for c in cols])
    return path


def test_replay_recovers_hidden_and_alerts():
    path = _write_csv()
    try:
        res = run_csv(aquaculture.build_spec(), path, time_col="timestamp",
                      u_col="excretion_kg_h", u_forecast=0.225, config=CFG)
    finally:
        os.remove(path)
    assert res.estimates[-1].hidden < 0.6            # tracked the crash
    assert any(lv == "ALERT" for _, lv, _ in res.events), res.events


def test_iso_and_numeric_time_agree():
    p_iso = _write_csv(numeric_time=False)
    p_num = _write_csv(numeric_time=True)
    try:
        a = run_csv(aquaculture.build_spec(), p_iso, time_col="timestamp",
                    u_col="excretion_kg_h", u_forecast=0.225, config=CFG)
        b = run_csv(aquaculture.build_spec(), p_num, time_col="timestamp",
                    u_col="excretion_kg_h", u_forecast=0.225, config=CFG)
    finally:
        os.remove(p_iso); os.remove(p_num)
    # identical data, different time encodings -> same final estimate
    assert abs(a.estimates[-1].hidden - b.estimates[-1].hidden) < 1e-6


def test_missing_column_is_treated_as_offline_sensor():
    path = _write_csv(drop=("TAN",))              # no ammonia column at all
    try:
        _, rows = load_csv(path, time_col="timestamp")
        assert "TAN" not in rows[0]
        res = run_csv(aquaculture.build_spec(), path, time_col="timestamp",
                      u_col="excretion_kg_h", u_forecast=0.225, config=CFG)
    finally:
        os.remove(path)
    # engine still runs and never uses a TAN channel
    assert all("TAN" not in e.used_channels for e in res.estimates)
    assert max(len(e.used_channels) for e in res.estimates) == 3
    # ammonia state is still estimated from model + other channels
    assert np.isfinite(res.estimates[-1].x[0])


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS  {fn.__name__}"); passed += 1
        except Exception:
            print(f"FAIL  {fn.__name__}"); traceback.print_exc()
    print(f"\n{passed}/{len(fns)} passed")
    raise SystemExit(0 if passed == len(fns) else 1)
