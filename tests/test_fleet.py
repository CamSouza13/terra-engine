"""Fleet prior pooling: the flywheel tightens the shared prior as sites join."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from terra.fleet import pool_posteriors, prior_tightening


def _make_sites(n, truth=1.0, between=0.05, within=0.1, seed=0):
    rng = np.random.default_rng(seed)
    sites = []
    for _ in range(n):
        site_mean = truth + rng.normal(0, between)          # site heterogeneity
        obs = site_mean + rng.normal(0, within)             # calibration noise
        sites.append({"k_nitrify": (float(obs), within)})
    return sites


def test_pool_between_site_means_and_tightens():
    sites = _make_sites(20)
    pooled = pool_posteriors(sites)["k_nitrify"]
    means = [m for m, _ in (s["k_nitrify"] for s in sites)]
    # pooled mean sits within the range of site means and near truth
    assert min(means) <= pooled["mean"] <= max(means)
    assert abs(pooled["mean"] - 1.0) < 0.05
    assert pooled["n_sites"] == 20
    # tau recovers real between-site spread in the right ballpark
    assert 0.0 < pooled["tau"] < 0.2

    curve = prior_tightening(sites, "k_nitrify")
    assert curve[-1]["std"] < curve[0]["std"]               # tightens with N
    # pooled-mean uncertainty shrinks roughly like 1/sqrt(N)
    assert curve[-1]["std"] < curve[0]["std"] / 2
    print(f"  pooled mean {pooled['mean']:.3f}, tau {pooled['tau']:.3f}; "
          f"prior std {curve[0]['std']:.3f} -> {curve[-1]['std']:.3f} over 20 sites")


def test_empty_and_single():
    assert pool_posteriors([]) == {}
    one = pool_posteriors([{"a": (2.0, 0.3)}])["a"]
    assert one["mean"] == 2.0 and one["tau"] == 0.0


if __name__ == "__main__":
    test_pool_between_site_means_and_tightens()
    test_empty_and_single()
    print("all fleet tests passed")
