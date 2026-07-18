"""Convergence-diagnostics tests (pure numpy, no jax)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from terra.diagnostics import split_rhat, ess, report


def test_rhat_near_one_when_mixed():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 600))
    r = split_rhat(x)
    assert 0.95 < r < 1.05, r


def test_rhat_flags_unmixed_chains():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(4, 600)) + np.arange(4)[:, None] * 3.0   # chains apart
    assert split_rhat(x) > 1.2, split_rhat(x)


def test_ess_large_for_independent():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(4, 500))
    assert ess(x) > 500, ess(x)


def test_report_shape():
    rng = np.random.default_rng(3)
    rep = report({"k1": rng.normal(size=(2, 300))})
    assert set(rep["k1"]) == {"rhat", "ess"}


if __name__ == "__main__":
    n = 0
    for fn in (test_rhat_near_one_when_mixed, test_rhat_flags_unmixed_chains,
               test_ess_large_for_independent, test_report_shape):
        fn()
        print(f"PASS  {fn.__name__}")
        n += 1
    print(f"\n{n}/{n} passed")
