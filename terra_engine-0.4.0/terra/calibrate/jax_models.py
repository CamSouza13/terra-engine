"""JAX mirror of the aquaculture RAS process model, for offline calibration.

This is a differentiable copy of the numpy dynamics in
``terra/domains/aquaculture.py`` so HMC/NUTS can fit the kinetic parameters
against logged data. It is deliberately self-contained: nothing in the edge
core (``terra/core.py``, ``terra/ukf.py``) imports JAX, and nothing here is
imported at ``import terra`` time. The edge engine stays numpy-only.

Contract: ``deriv(x, u, theta)`` returns dx/dt for the RAS state
``[TAN, NO2, NO3, DO, eff]`` given input ``u = (excretion_kg_h, exchange_mult)``
and a dict ``theta`` of the parameters being fit. Structural constants (tank
volume, half-saturation, stoichiometry) are held fixed at their
``RASParams`` defaults during calibration.
"""
from __future__ import annotations

import jax.numpy as jnp

# ---- fixed structural constants (from RASParams defaults) --------------------
V = 20000.0          # L, system volume
Q = 800.0            # L/h, recirculation / exchange base flow
K1 = 0.5             # mg/L, TAN half-saturation
K2 = 0.5             # mg/L, NO2 half-saturation
DO_sat = 8.0         # mg/L
o2_per_tan = 3.43    # mg O2 per mg TAN oxidised
o2_per_no2 = 1.14    # mg O2 per mg NO2 oxidised
TAN_in = 0.0
NO2_in = 0.0
NO3_in = 4.0
a_tan = 0.03         # kg TAN per kg feed

# ---- parameters NUTS infers, with LogNormal (median, sigma_log) priors -------
FIT_PARAMS = ("k1", "k2", "kLa", "resp")
PRIOR = {
    "k1":   (1.6, 0.4),   # max TAN -> NO2 nitrification rate (1/h)
    "k2":   (1.8, 0.4),   # max NO2 -> NO3 nitrification rate (1/h)
    "kLa":  (2.2, 0.4),   # O2 transfer coefficient (1/h)
    "resp": (0.9, 0.4),   # biological O2 respiration (mg/L/h)
}

# channels observed and the state index each maps to
OBS_CHANNELS = ("TAN", "NO2", "NO3", "DO")
OBS_INDEX = (0, 1, 2, 3)


def deriv(x, u, theta):
    """dx/dt for [TAN, NO2, NO3, DO, eff]. `theta` is a dict of fit params.

    Efficiency ``eff`` is carried as a (constant) state so the state vector
    matches the numpy model; for calibration windows it is held at its initial
    value and ``k1``/``k2`` absorb the nitrification capacity.
    """
    TAN = jnp.clip(x[0], 0.0, None)
    NO2 = jnp.clip(x[1], 0.0, None)
    NO3 = jnp.clip(x[2], 0.0, None)
    DO = x[3]
    eff = jnp.clip(x[4], 0.0, 1.5)

    exc = u[0]
    qm = u[1]
    p_tan = exc * a_tan * 1.0e6 / V
    r1 = eff * theta["k1"] * TAN / (K1 + TAN)
    r2 = eff * theta["k2"] * NO2 / (K2 + NO2)
    dil = Q * qm / V

    return jnp.array([
        p_tan - r1 - dil * (TAN - TAN_in),
        r1 - r2 - dil * (NO2 - NO2_in),
        r2 - dil * (NO3 - NO3_in),
        theta["kLa"] * (DO_sat - DO) - theta["resp"]
        - (o2_per_tan * r1 + o2_per_no2 * r2),
        0.0,
    ])
