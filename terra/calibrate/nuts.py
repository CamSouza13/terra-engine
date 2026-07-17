"""Offline Bayesian calibration of a domain's kinetic parameters via HMC/NUTS.

The edge engine (``TerraEngine`` + UKF) is a real-time *state* estimator: it
assumes the process parameters are known and tracks the state. This module
solves the complementary offline problem — given a logged run, infer the
*parameters* (nitrification rates, O2 transfer, respiration) with full
posterior uncertainty, so a site can be calibrated before the edge engine is
trusted operationally.

It integrates the JAX process model with RK4 (mirroring ``terra.core.rk4``),
places LogNormal priors on the fit parameters, and samples the posterior with
NumPyro's NUTS. The result carries posterior samples and an ``apply_to`` helper
that returns an updated ``RASParams`` for the edge engine to consume.

Requires the optional ``calibrate`` extra: ``pip install terra-engine[calibrate]``.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from . import jax_models as M


# ---- differentiable RK4 rollout ---------------------------------------------

def _rk4(x, dt, u, theta):
    k1 = M.deriv(x, u, theta)
    k2 = M.deriv(x + 0.5 * dt * k1, u, theta)
    k3 = M.deriv(x + 0.5 * dt * k2, u, theta)
    k4 = M.deriv(x + dt * k3, u, theta)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def rollout(theta, x0, u_arr, dt_meas, n_sub):
    """Integrate from ``x0`` under piecewise-constant inputs.

    ``u_arr`` has one row per interval (length N-1 for N measurement times);
    each interval is integrated with ``n_sub`` RK4 substeps. Returns the state
    at each of the N measurement times, shape (N, 5).
    """
    dt = dt_meas / n_sub

    def interval(x, u):
        def sub(xx, _):
            return _rk4(xx, dt, u, theta), None
        x_next, _ = jax.lax.scan(sub, x, None, length=n_sub)
        return x_next, x_next

    _, xs = jax.lax.scan(interval, x0, u_arr)          # (N-1, 5)
    return jnp.concatenate([x0[None, :], xs], axis=0)  # (N, 5)


# ---- probabilistic model -----------------------------------------------------

def _model(u_arr, x0, dt_meas, n_sub, obs, mask, sigma):
    theta = {}
    for name in M.FIT_PARAMS:
        med, s = M.PRIOR[name]
        theta[name] = numpyro.sample(name, dist.LogNormal(jnp.log(med), s))
    preds = rollout(theta, x0, u_arr, dt_meas, n_sub)        # (N, 5)
    pred_obs = preds[:, list(M.OBS_INDEX)]                    # (N, 4)
    # masked Gaussian likelihood — only channels that reported contribute
    numpyro.sample("obs", dist.Normal(pred_obs, sigma).mask(mask), obs=obs)


# ---- result container --------------------------------------------------------

@dataclass
class CalibrationResult:
    samples: dict
    channels: tuple

    def medians(self) -> dict:
        return {k: float(np.median(v)) for k, v in self.samples.items()}

    def summary(self) -> dict:
        out = {}
        for k, v in self.samples.items():
            out[k] = {
                "median": float(np.median(v)),
                "mean": float(np.mean(v)),
                "std": float(np.std(v)),
                "q5": float(np.percentile(v, 5)),
                "q95": float(np.percentile(v, 95)),
            }
        return out

    def apply_to(self, params: Any) -> Any:
        """Return a copy of a dataclass params object with fitted values set."""
        med = self.medians()
        updates = {k: v for k, v in med.items() if hasattr(params, k)}
        return dataclasses.replace(params, **updates)

    def print_report(self, truth: dict | None = None) -> None:
        s = self.summary()
        print(f"{'param':6}  {'median':>8}  {'90% CI':>18}" +
              ("  truth" if truth else ""))
        for k in self.samples:
            ci = f"[{s[k]['q5']:.3f}, {s[k]['q95']:.3f}]"
            line = f"{k:6}  {s[k]['median']:8.3f}  {ci:>18}"
            if truth and k in truth:
                line += f"  {truth[k]:.3f}"
            print(line)


# ---- input helpers -----------------------------------------------------------

def _u_pair(u) -> list:
    """Normalise an input to [excretion, exchange_multiplier]."""
    if hasattr(u, "__len__"):
        return [float(u[0]), float(u[1]) if len(u) > 1 else 1.0]
    return [float(u), 1.0]


def fit_nuts(times, u_series, meas, spec, *,
             channels=M.OBS_CHANNELS, num_warmup=300, num_samples=300,
             n_sub=4, seed=0, progress_bar=False) -> CalibrationResult:
    """Fit kinetic parameters from a logged run.

    Parameters
    ----------
    times : sequence of float
        Measurement timestamps (hours), assumed ~uniform.
    u_series : sequence
        Input at each timestamp; scalar excretion or (excretion, exchange).
    meas : list of dict
        Per-timestamp ``{channel: value}`` (missing channels allowed).
    spec : SystemSpec
        The domain spec (for x0 and per-channel sensor noise).
    """
    times = np.asarray(times, float)
    n = len(times)
    dt_meas = float(times[1] - times[0])

    u_arr = jnp.asarray(np.array([_u_pair(u_series[i]) for i in range(n - 1)]))
    x0 = jnp.asarray(np.asarray(spec.x0, float))

    obs = np.zeros((n, len(channels)), float)
    mask = np.zeros((n, len(channels)), bool)
    for i in range(n):
        row = meas[i] if i < len(meas) else {}
        for j, ch in enumerate(channels):
            v = row.get(ch)
            if v is not None and np.isfinite(v):
                obs[i, j] = float(v)
                mask[i, j] = True
    sigma = jnp.asarray([spec.channels[ch].noise for ch in channels])

    kernel = NUTS(_model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                progress_bar=progress_bar)
    mcmc.run(jax.random.PRNGKey(seed), u_arr, x0, dt_meas, n_sub,
             jnp.asarray(obs), jnp.asarray(mask), sigma)
    samples = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
    return CalibrationResult(samples, tuple(channels))
