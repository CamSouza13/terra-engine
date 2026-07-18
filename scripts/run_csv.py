"""Replay a logged sensor CSV through a domain and print the event log.

    python scripts/run_csv.py --domain aquaculture --csv data/aquaculture_sample.csv \
        --time-col timestamp --u-col excretion_kg_h

Columns are matched to the domain's channels by name (TAN, NO2, NO3, DO, ...).
Blank cells or missing columns are treated as that sensor being offline.
"""
from __future__ import annotations

import argparse
import os
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from terra import EngineConfig                    # noqa: E402
from terra.domains import DOMAINS                  # noqa: E402
from terra.ingest import run_csv                   # noqa: E402

CFG = {
    "aquaculture":    EngineConfig(forecast_horizon_h=12, forecast_dt=0.25),
    "soil":           EngineConfig(forecast_horizon_h=48, forecast_dt=1.0),
    "bioremediation": EngineConfig(forecast_horizon_h=12, forecast_dt=0.25),
    "blss":           EngineConfig(forecast_horizon_h=18, forecast_dt=0.25),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=list(DOMAINS))
    ap.add_argument("--csv", required=True)
    ap.add_argument("--time-col", default="timestamp")
    ap.add_argument("--u-col", default=None, help="column with the input drive")
    ap.add_argument("--u-const", type=float, default=None)
    ap.add_argument("--u-forecast", type=float, default=None)
    args = ap.parse_args()

    spec = DOMAINS[args.domain].build_spec()
    res = run_csv(
        spec, args.csv, time_col=args.time_col,
        u_col=args.u_col, u_const=args.u_const, u_forecast=args.u_forecast,
        config=CFG[args.domain],
    )

    last = res.estimates[-1]
    print(f"replayed {len(res.estimates)} rows over {res.t[-1]:.1f} h")
    print(f"final hidden [{spec.hidden}] = {last.hidden:.2f} "
          f"(+/- {last.hidden_std:.2f})")
    print("EVENT LOG")
    if not res.events:
        print("  (no events)")
    for t, level, msg in res.events:
        print(f"  {t:6.1f}h  {level:5s}  {msg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
