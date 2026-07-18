"""Validation-harness tests (numpy only)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from terra.validate import run_validation
from terra.baselines import threshold_alarm, true_breach_times
from terra.domains import aquaculture, soil, bioremediation, blss


def test_recovers_hidden_and_warns_early():
    spec, sim = aquaculture.simulate(hours=48.0, fault=True, seed=7)
    r = run_validation(spec, sim)
    assert r.hidden_rmse < 0.5, r.hidden_rmse
    assert r.hidden_coverage >= 0.4, r.hidden_coverage

    nh3 = r.lead["un-ionized NH3-N"]
    assert nh3["true_breach"] is not None, nh3
    assert nh3["engine_lead"] is not None, nh3          # engine warned before breach
    if nh3["baseline_lead"] is not None:
        # engine warns no later than a raw gauge
        assert nh3["engine_lead"] >= nh3["baseline_lead"] - 1e-6, nh3
    print(f"  aquaculture: RMSE={r.hidden_rmse:.3f} cover={r.hidden_coverage:.0%} "
          f"engine_lead={nh3['engine_lead']:.1f}h "
          f"baseline_lead={nh3['baseline_lead']}")


def test_runs_all_domains():
    for mod in (aquaculture, soil, bioremediation, blss):
        spec, sim = mod.simulate(fault=True)
        r = run_validation(spec, sim)
        assert np.isfinite(r.hidden_rmse)
        assert 0.0 <= r.hidden_coverage <= 1.0
        # baseline + truth helpers return an entry per safety target
        base = threshold_alarm(spec, sim)
        breach = true_breach_times(spec, sim)
        for s in spec.safety:
            assert s.name in base and s.name in breach
        print(f"  {spec.name:14} RMSE={r.hidden_rmse:.3f} cover={r.hidden_coverage:.0%}")


if __name__ == "__main__":
    n = 0
    for fn in (test_recovers_hidden_and_warns_early, test_runs_all_domains):
        fn()
        print(f"PASS  {fn.__name__}")
        n += 1
    print(f"\n{n}/{n} passed")
