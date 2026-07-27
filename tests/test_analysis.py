"""Analysis toolkit: degradation, drift detectability, edge benchmark, ROI."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from terra import analysis
from terra.domains import aquaculture


def test_degradation_monotone_length():
    spec, sim = aquaculture.simulate()
    rows = analysis.degradation_sweep(spec, sim)
    assert [r["n_channels"] for r in rows] == list(range(len(spec.channels), 0, -1))
    for r in rows:
        assert 0.0 <= r["coverage"] <= 1.0
        assert r["rmse"] >= 0.0
    print(f"  degradation: {len(rows)} levels, "
          f"full RMSE {rows[0]['rmse']:.3f} -> 1-ch RMSE {rows[-1]['rmse']:.3f}")


def test_drift_detectability_monotone():
    spec, sim = aquaculture.simulate()
    d = analysis.drift_detectability(spec, "DO", windows_h=(24, 96, 168), trials=150)
    rates = [d["windows"][w]["min_rate_per_day"] for w in (24, 96, 168)]
    # a longer window resolves an equal-or-smaller drift
    assert rates[0] >= rates[1] >= rates[2] - 1e-9, rates
    print(f"  drift DO min/day: 24h {rates[0]:.3g}  96h {rates[1]:.3g}  168h {rates[2]:.3g}")


def test_alert_quality_beats_gauge():
    d = analysis.alert_quality(aquaculture, seeds=4)
    assert 0.0 <= d["engine_detect_rate"] <= 1.0
    assert d["engine_detect_rate"] >= d["baseline_detect_rate"]
    print(f"  alerts: engine {d['engine_detect_rate']:.0%} vs gauge "
          f"{d['baseline_detect_rate']:.0%}, lead {d['engine_mean_lead_h']}")


def test_bench_and_roi():
    spec, sim = aquaculture.simulate()
    b = analysis.edge_benchmark(spec, sim, steps=60)
    assert b["ms_per_step"] > 0 and b["peak_mem_mb"] > 0
    r = analysis.roi_distribution(n=5000)
    p = r["percentiles"]
    assert p[10] <= p[50] <= p[90]
    print(f"  bench {b['ms_per_step']:.3f} ms/step; ROI P50 ${p[50]:,.0f}")


if __name__ == "__main__":
    test_degradation_monotone_length()
    test_drift_detectability_monotone()
    test_alert_quality_beats_gauge()
    test_bench_and_roi()
    print("all analysis tests passed")
