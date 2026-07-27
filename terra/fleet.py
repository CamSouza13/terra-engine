"""Fleet-level prior pooling: the data flywheel, made concrete.

Each deployed site produces a calibrated posterior over the model parameters
(mean + std per parameter, e.g. from the offline NUTS fit). This module pools
those per-site posteriors into a shared prior that a *new* site starts from,
using a random-effects (empirical-Bayes) model so we borrow strength across
sites without pretending every farm is identical.

Two things fall out and both are real product behaviour:
  * the shared prior tightens as the fleet grows (more sites -> smaller
    uncertainty on the pooled mean), which is why each deployment makes the
    next one faster to calibrate;
  * the between-site spread (tau) is estimated, not assumed, so a genuinely
    heterogeneous fleet keeps a wide, honest prior instead of a false-confident
    one.
"""
from __future__ import annotations

import numpy as np

# a per-site posterior is {param_name: (mean, std)}
SitePosterior = dict


def _dersimonian_laird(means: np.ndarray, variances: np.ndarray) -> float:
    """Estimate between-site variance tau^2 (random-effects heterogeneity)."""
    k = len(means)
    if k < 2:
        return 0.0
    w = 1.0 / variances
    mu_fixed = float(np.sum(w * means) / np.sum(w))
    Q = float(np.sum(w * (means - mu_fixed) ** 2))
    c = float(np.sum(w) - np.sum(w ** 2) / np.sum(w))
    tau2 = (Q - (k - 1)) / c if c > 0 else 0.0
    return max(tau2, 0.0)


def pool_posteriors(sites: list[SitePosterior]) -> dict:
    """Pool per-site posteriors into a shared prior per parameter.

    Returns, for each parameter: the pooled mean, the std of that pooled mean,
    the estimated between-site std (tau), and the predictive std a new site
    should adopt as its prior (combines pooled-mean uncertainty with tau)."""
    if not sites:
        return {}
    params = set().union(*[set(s) for s in sites])
    out: dict[str, dict] = {}
    for p in sorted(params):
        rows = [s[p] for s in sites if p in s]
        means = np.array([m for m, _ in rows], float)
        variances = np.array([max(sd, 1e-12) ** 2 for _, sd in rows], float)
        tau2 = _dersimonian_laird(means, variances)
        w = 1.0 / (variances + tau2)                     # random-effects weights
        mu = float(np.sum(w * means) / np.sum(w))
        var_mu = float(1.0 / np.sum(w))                  # uncertainty of the mean
        out[p] = {
            "mean": mu,
            "std": float(np.sqrt(var_mu)),               # of the pooled mean
            "tau": float(np.sqrt(tau2)),                 # between-site spread
            "prior_std": float(np.sqrt(var_mu + tau2)),  # for a new site
            "n_sites": len(rows),
        }
    return out


def prior_tightening(site_stream: list[SitePosterior], param: str) -> list[dict]:
    """Replay the fleet growing one site at a time and record how the pooled
    prior for ``param`` tightens. This is the flywheel as a measurable curve."""
    curve = []
    for k in range(1, len(site_stream) + 1):
        pooled = pool_posteriors(site_stream[:k])
        if param in pooled:
            curve.append({"n_sites": k,
                          "mean": pooled[param]["mean"],
                          "std": pooled[param]["std"],
                          "prior_std": pooled[param]["prior_std"]})
    return curve
