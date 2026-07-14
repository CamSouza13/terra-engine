"""Emit an example sensor log (CSV) from the aquaculture simulator, so the CSV
ingestion path has a realistic, runnable input with a known injected fault.

    python scripts/make_sample_csv.py data/aquaculture_sample.csv
"""
from __future__ import annotations

import csv
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from terra.domains import aquaculture   # noqa: E402


def main(out):
    spec, sim = aquaculture.simulate()
    t = sim["t"]
    start = datetime(2026, 7, 1, 0, 0, 0)
    cols = ["timestamp", "TAN", "NO2", "NO3", "DO", "excretion_kg_h"]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i in range(len(t)):
            m = sim["meas"][i]
            ts = (start + timedelta(hours=float(t[i]))).isoformat()
            w.writerow([
                ts,
                f"{m.get('TAN', ''):.4f}" if "TAN" in m else "",
                f"{m.get('NO2', ''):.4f}" if "NO2" in m else "",
                f"{m.get('NO3', ''):.4f}" if "NO3" in m else "",
                f"{m.get('DO', ''):.4f}" if "DO" in m else "",
                f"{(sim['u'][i][0] if hasattr(sim['u'][i], '__len__') else sim['u'][i]):.4f}",
            ])
    print(f"wrote {out}  ({len(t)} rows)")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "data/aquaculture_sample.csv"
    main(out)
