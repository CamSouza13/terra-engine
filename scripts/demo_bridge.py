#!/usr/bin/env python3
"""End-to-end hardware-bridge demo — no sensors required.

Streams the bundled sample aquaculture log through the same bridge path a live
node uses (sample -> driver -> TerraEngine), and narrates what the estimator
infers: the hidden biofilter efficiency, the forecast breach probability, and
the moment it raises the alert. Run:

    python scripts/demo_bridge.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from terra import integrations as I
from terra.core import TerraEngine, EngineConfig
from terra.domains import aquaculture


def main() -> int:
    spec, _ = aquaculture.simulate()
    driver = I.sample_bridge_driver(spec)
    engine = TerraEngine(spec, EngineConfig(outlier_sigma=5.0, track_bias=True,
                                            cusum_threshold=8.0))

    print("Terra bridge demo — streaming the sample farm log as if from live "
          "hardware\n")
    print(f"  {'hour':>5}  {'TAN':>6} {'NO2':>6}  {'eff (hidden)':>16}  status")
    print("  " + "-" * 58)

    first_alert = None
    last = None
    seen = 0
    for i, (t, dt, meas, u) in enumerate(driver.steps()):
        last = engine.step(t, dt, meas, u)
        if len(engine.events) > seen:
            for ev in engine.events[seen:]:
                if first_alert is None and "ALERT" in str(ev[1]).upper():
                    first_alert = (ev[0], ev[2])
            seen = len(engine.events)
        if i % 12 == 0 or (first_alert and abs(t - first_alert[0]) < 1e-6):
            tan = meas.get("TAN", float("nan"))
            no2 = meas.get("NO2", float("nan"))
            band = f"{last.hidden:5.2f} +/- {last.hidden_std:4.2f}"
            status = "healthy" if last.hidden > 0.85 else (
                "watch" if last.hidden > 0.5 else "AT RISK")
            print(f"  {t:5.1f}  {tan:6.2f} {no2:6.2f}  {band:>16}  {status}")

    print("\nresult:")
    print(f"  final biofilter efficiency (inferred): {last.hidden:.2f} "
          f"+/- {last.hidden_std:.2f}  (healthy = 1.0)")
    if first_alert:
        print(f"  first alert at ~{first_alert[0]:.1f} h: {first_alert[1]}")
    else:
        print("  no alert raised")
    print(f"  events raised: {len(engine.events)}")
    print("\nThe engine never measured biofilter efficiency directly — it "
          "inferred the collapse from the nitrogen channels and flagged it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
