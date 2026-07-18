"""Offline Bayesian calibration of a domain's parameters via HMC/NUTS.

The edge engine (``TerraEngine`` + UKF) is a real-time *state* estimator: it
assumes the process parameters are known and tracks the state. This module
solves the complementary offline problem — given a logged run, infer the
*parameters* (kinetics, transfer rates) with full posterior uncertainty, and
optionally the *sensor drift / biofouling* corrupting the log, so a site can be
calibrated before the edge engine is trusted operationally.

It is domain-agnostic: the differentiable model, the parameters to fit, and the
channel-to-state map come from a ``CalModel`` (see ``models.py``), resolved by
domain name. Requires the optional extra: ``pip install terra-engine[calibrate]``.
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

from .models import get_model
from .spec import CalModel


# ---- differentiable RK4 rollout ---------------------------------------------

def _rk4(deriv, x, dt, u, theta):
    k1 = deriv(x, u, theta)
    k2 = deriv(x + 0.5 * dt * k1, u, theta)
    k3 = deriv(x + 0.5 * dt * k2, u, theta)
    k4 = deriv(x + dt * k3, u, theta)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def rollout(deriv, theta, x0, u_arr, dt_meas, n_sub):
    """Integrate from ``x0`` under piecewise-constant inputs.

    ``u_arr`` has one row per interval (length N-1 for N measurement times);
    each interval takes ``n_sub`` RK4 substeps. Returns the state at each of the
    N measurement times, shape (N, S).
    """
    dt = dt_meas / n_sub

    def interval(x, u):
        def sub(xx, _):
            return _rk4(deriv, xx, dt, u, theta), None
        x_next, _ = jax.lax.scan(sub, x, None, length=n_sub)
        return x_next, x_next

    _, xs = jax.lax.scan(interval, x0, u_arr)          # (N-1, S)
    return jnp.concatenate([x0[None, :], xs], axis=0)  # (N, S)


# ---- probabilistic model (closure over a CalModel) ---------------------------

def _make_model(model: CalModel, channels, drift_cols):
    obs_map = model.obs_map

    def _model(u_arr, x0, dt_meas, n_sub, obs, mask, sigma, times, drift_sd):
        theta = {}
        for name in model.fit_params:
            med, s = model.prior[name]
            theta[name] = numpyro.sample(name, dist.LogNormal(jnp.log(med), s))

        drift = {}
        for j, c in enumerate(drift_cols):
            ch = channels[c]
            drift[c] = numpyro.sample("drift_" + ch,
                                      dist.Normal(0.0, drift_sd[j]))

        preds = rollout(model.deriv, theta, x0, u_arr, dt_meas, n_sub)  # (N, S)
        cols = []
        for c, ch in enumerate(channels):
            idx, scale, offset = obs_map[ch]
            col = scale * preds[:, idx] + offset
            if c in drift:
                col = col + drift[c] * times
            cols.append(col)
        pred_obs = jnp.stack(cols, axis=1)                              # (N, C)
        numpyro.sample("obs", dist.Normal(pred_obs, sigma).mask(mask), obs=obs)

    return _model


# ---- result container --------------------------------------------------------

@dataclass
class CalibrationResult:
    samples: dict
    channels: tuple
    fit_params: tuple

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

    def drift(self) -> dict:
        """Fitted sensor drift rates (channel-units per hour), if any."""
        return {k[len("drift_"):]: float(np.median(v))
                for k, v in self.samples.items() if k.startswith("drift_")}

    def diagnostics(self) -> dict:
        """Convergence diagnostics per parameter: split-R-hat and ESS.

        R-hat near 1.0 (and a healthy ESS) means the chain mixed; R-hat above
        ~1.1 means the fit did not converge and the posterior is untrustworthy.
        """
        from ..diagnostics import report
        return report(self.samples)

    def converged(self, rhat_max: float = 1.1) -> bool:
        return all(d["rhat"] < rhat_max for d in self.diagnostics().values())

    def apply_to(self, params: Any) -> Any:
        """Return a copy of a dataclass params object with fitted values set.

        Only process parameters that exist on ``params`` are applied; sensor
        drift terms are ignored (they describe the sensors, not the process).
        """
        med = self.medians()
        updates = {k: v for k, v in med.items() if hasattr(params, k)}
        return dataclasses.replace(params, **updates)

    def print_report(self, truth: dict | None = None) -> None:
        s = self.summary()
        header = f"{'param':10}  {'median':>9}  {'90% CI':>20}"
        if truth:
            header += "  truth"
        print(header)
        for k in self.samples:
            ci = f"[{s[k]['q5']:.3f}, {s[k]['q95']:.3f}]"
            line = f"{k:10}  {s[k]['median']:9.3f}  {ci:>20}"
            if truth and k in truth:
                line += f"  {truth[k]:.3f}"
            print(line)


# ---- input helpers -----------------------------------------------------------

def _u_row(u) -> list:
    """Normalise an input to a flat list of floats (any arity)."""
    if hasattr(u, "__len__"):
        return [float(v) for v in u]
    return [float(u)]


def fit_nuts(times, u_series, meas, spec, *, model=None, channels=None,
             fit_drift=None, num_warmup=300, num_samples=300, n_sub=4,
             seed=0, progress_bar=False) -> CalibrationResult:
    """Fit a domain's parameters (and optional sensor drift) from a logged run.

    Parameters
    ----------
    times, u_series, meas :
        Timestamps (hours), input at each timestamp, and per-timestamp
        ``{channel: value}`` dicts (missing channels allowed).
    spec : SystemSpec
        Domain spec (for x0, per-channel sensor noise, and the domain name used
        to resolve the calibration model).
    model : CalModel, optional
        Override the model; defaults to the one registered for ``spec.name``.
    fit_drift : sequence of str or True, optional
        Channels to give a linear drift/biofouling term (``obs = signal +
        rate * t``). ``True`` enables drift on every observed channel.
    """
    model = model or get_model(spec.name)
    channels = tuple(channels or model.channels)

    times = np.asarray(times, float)
    n = len(times)
    dt_meas = float(times[1] - times[0])
    t_rel = jnp.asarray(times - times[0])

    u_arr = jnp.asarray(np.array([_u_row(u_series[i]) for i in range(n - 1)]))
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
    sigma = np.array([spec.channels[ch].noise for ch in channels], float)

    if fit_drift is True:
        drift_names = list(channels)
    else:
        drift_names = list(fit_drift or [])
    drift_cols = [channels.index(ch) for ch in drift_names]
    drift_sd = jnp.asarray([sigma[c] for c in drift_cols]) if drift_cols \
        else jnp.asarray([], float)

    numpyro_model = _make_model(model, channels, drift_cols)
    mcmc = MCMC(NUTS(numpyro_model), num_warmup=num_warmup,
                num_samples=num_samples, progress_bar=progress_bar)
    mcmc.run(jax.random.PRNGKey(seed), u_arr, x0, dt_meas, n_sub,
             jnp.asarray(obs), jnp.asarray(mask), jnp.asarray(sigma),
             t_rel, drift_sd)
    samples = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
    return CalibrationResult(samples, channels, tuple(model.fit_params))


# ---- close the loop: fit -> calibrated spec -> edge engine -------------------

def calibrated_spec(spec, result: CalibrationResult):
    """Return a copy of ``spec`` whose process params carry the fitted medians.

    The returned SystemSpec drops straight into ``TerraEngine`` so the edge
    estimator runs with site-specific parameters instead of library defaults.
    """
    return dataclasses.replace(spec, params=result.apply_to(spec.params))


def calibrate_and_build(base_spec, times, u_series, meas, **fit_kwargs):
    """One call from a logged run to a calibrated spec.

    Fits the parameters with NUTS and returns
    ``(calibrated_spec, CalibrationResult)``. Pass the result's spec to
    ``TerraEngine`` to run the edge estimator tuned to this site.
    """
    result = fit_nuts(times, u_series, meas, base_spec, **fit_kwargs)
    return calibrated_spec(base_spec, result), result
