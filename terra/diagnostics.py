"""Pure-numpy MCMC convergence diagnostics.

Import-safe without jax, so it can be tested and used off the edge as well as
inside the (optional) calibration layer. Feed a per-parameter array shaped
(num_chains, num_samples); the calibrate layer produces these via
``mcmc.get_samples(group_by_chain=True)``.

- ``split_rhat``: split-R-hat (Gelman-Rubin). Values near 1.0 indicate the
  chains have mixed; > ~1.01-1.1 is a warning sign.
- ``ess``: a Geyer initial-monotone-sequence effective sample size.
"""
from __future__ import annotations

import numpy as np


def split_rhat(x) -> float:
    """Split-R-hat for samples shaped (chains, draws) or (draws,)."""
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[None, :]
    _, n = x.shape
    if n < 4:
        return float("nan")
    half = n // 2
    s = np.concatenate([x[:, :half], x[:, half:2 * half]], axis=0)  # (2*chains, half)
    _, N = s.shape
    chain_means = s.mean(axis=1)
    chain_vars = s.var(axis=1, ddof=1)
    W = chain_vars.mean()
    B = N * chain_means.var(ddof=1)
    if W <= 0:
        return float("nan")
    var_hat = (N - 1) / N * W + B / N
    return float(np.sqrt(var_hat / W))


def ess(x) -> float:
    """Effective sample size for samples shaped (chains, draws) or (draws,)."""
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[None, :]
    m, n = x.shape
    if n < 4:
        return float("nan")
    xc = x - x.mean(axis=1, keepdims=True)
    var = float(np.mean(xc ** 2))
    if var <= 0:
        return float(m * n)
    # autocorrelation averaged across chains
    acf = np.empty(n)
    for lag in range(n):
        acf[lag] = float(np.mean(xc[:, : n - lag] * xc[:, lag:])) / var
    # Geyer initial monotone sequence: sum adjacent pairs until non-positive
    tau = 1.0
    for lag in range(1, n - 1, 2):
        pair = acf[lag] + acf[lag + 1]
        if pair <= 0:
            break
        tau += 2.0 * pair
    return float(m * n / tau)


def report(samples: dict) -> dict:
    """Diagnostics for a dict of param -> (chains, draws) arrays."""
    return {k: {"rhat": split_rhat(v), "ess": ess(v)} for k, v in samples.items()}
