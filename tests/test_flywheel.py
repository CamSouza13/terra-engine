"""Backtest harness (score a logged CSV) and fleet-pool CLI (the flywheel job)."""
import csv
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from terra.domains import bioremediation
from terra.ingest import backtest_csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _write_csv(path):
    spec, sim = bioremediation.simulate(fault=True)
    hi = spec.idx(spec.hidden)
    truth = np.asarray(sim["truth"], float)
    chans = list(spec.channels)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t"] + chans + ["u", "truth"])
        for i, t in enumerate(sim["t"]):
            row = [t] + [sim["meas"][i].get(c, "") for c in chans]
            u = sim["u"][i]
            row.append(float(u) if np.isscalar(u) else "")
            row.append(truth[i][hi])
            w.writerow(row)
    return spec


def test_backtest_scores_against_truth():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "log.csv")
        spec = _write_csv(path)
        res = backtest_csv(spec, path, truth_col="truth", u_col="u")
        assert res["n_steps"] > 10
        assert "hidden_rmse" in res and np.isfinite(res["hidden_rmse"])
        assert 0.0 <= res["hidden_coverage"] <= 1.0
        print(f"  backtest RMSE {res['hidden_rmse']:.3f} "
              f"coverage {res['hidden_coverage']:.0%}")


def test_fleet_pool_cli():
    with tempfile.TemporaryDirectory() as d:
        for i, mean in enumerate([0.9, 1.0, 1.1, 1.05, 0.95]):
            with open(os.path.join(d, f"site{i}.json"), "w") as f:
                json.dump({"k_deg": [mean, 0.1]}, f)
        out = os.path.join(d, "prior.json")
        r = subprocess.run([sys.executable, "-m", "terra.cli", "fleet-pool",
                            "--sites", d, "--out", out],
                           cwd=ROOT, capture_output=True, text=True,
                           env={**os.environ, "PYTHONPATH": ROOT})
        assert r.returncode == 0, r.stderr
        with open(out) as f:
            prior = json.load(f)
        assert "k_deg" in prior and abs(prior["k_deg"][0] - 1.0) < 0.1
        print(f"  fleet-pool -> shared prior k_deg {prior['k_deg']}")


if __name__ == "__main__":
    test_backtest_scores_against_truth()
    test_fleet_pool_cli()
    print("all flywheel tests passed")
