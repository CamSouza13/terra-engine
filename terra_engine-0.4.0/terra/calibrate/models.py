"""Calibratable JAX models for every domain, and the registry.

Each domain's differentiable ``deriv`` mirrors the numpy dynamics in
``terra/domains/<name>.py``. Structural constants (half-saturations, ambient
levels, stoichiometry) are held at their library defaults; only the parameters
listed in each ``CalModel.fit_params`` are inferred. The edge core imports none
of this.
"""
from __future__ import annotations

import jax.numpy as jnp

from .spec import CalModel
from . import jax_models as _aqua


# ---- soil (root-zone nitrogen) ----------------------------------------------
# state [NH4, NO3, resp, act];  u = (mineralization, drainage, nitrate_dose)
_SOIL_K = 1.0        # NH4 half-saturation (fixed)
_SOIL_KRELAX = 0.4   # respiration tracking speed (fixed)


def _soil_deriv(x, u, th):
    NH4 = jnp.clip(x[0], 0.0, None)
    NO3 = jnp.clip(x[1], 0.0, None)
    resp = x[2]
    act = jnp.clip(x[3], 0.0, None)
    min_in = u[0]
    drain = u[1]
    no3_dose = u[2]
    r = act * th["k"] * NH4 / (_SOIL_K + NH4)
    return jnp.array([
        min_in - r,
        r - drain * NO3 - th["uptake"] + no3_dose,
        _SOIL_KRELAX * (act * th["base_resp"] - resp),
        0.0,
    ])


SOIL = CalModel(
    name="soil", deriv=_soil_deriv,
    fit_params=("k", "uptake", "base_resp"),
    prior={"k": (1.5, 0.4), "uptake": (0.30, 0.4), "base_resp": (2.0, 0.4)},
    obs_map={"NH4": (0, 1.0, 0.0), "NO3": (1, 1.0, 0.0),
             "CO2_flux": (2, 1.0, 0.0), "EC": (1, 0.1, 0.0)},
)


# ---- bioremediation (contaminant drawdown) ----------------------------------
# state [C, D, act];  u = dose (scalar)
_BIO_KC = 20.0
_BIO_KD = 1.0
_BIO_INFL = 0.005
_BIO_CSRC = 60.0


def _bio_deriv(x, u, th):
    C = jnp.clip(x[0], 0.0, None)
    D = jnp.clip(x[1], 0.0, None)
    act = jnp.clip(x[2], 0.0, None)
    dose = u[0]
    rate = act * th["k"] * C / (_BIO_KC + C) * D / (_BIO_KD + D)
    return jnp.array([
        -rate + _BIO_INFL * (_BIO_CSRC - C),
        dose - th["y"] * rate,
        0.0,
    ])


BIOREMEDIATION = CalModel(
    name="bioremediation", deriv=_bio_deriv,
    fit_params=("k", "y"),
    prior={"k": (0.5, 0.4), "y": (3.0, 0.4)},
    obs_map={"contaminant": (0, 1.0, 0.0), "ORP": (1, -30.0, -50.0)},
)


# ---- blss (closed-habitat air loop) -----------------------------------------
# state [CO2, O2, act];  u = (light, backup)
_BLSS_KCO2 = 500.0
_BLSS_KO2 = 0.001
_BLSS_LEAK = 0.02
_BLSS_CO2AMB = 400.0
_BLSS_O2AMB = 20.9
_BLSS_SCRUB = 260.0
_BLSS_O2CAP = 0.28


def _blss_deriv(x, u, th):
    CO2 = jnp.clip(x[0], 0.0, None)
    O2 = x[1]
    act = jnp.clip(x[2], 0.0, None)
    light = u[0]
    backup = u[1]
    photo = act * th["Pmax"] * light * CO2 / (_BLSS_KCO2 + CO2)
    return jnp.array([
        th["crew_co2"] - photo + _BLSS_LEAK * (_BLSS_CO2AMB - CO2)
        - backup * _BLSS_SCRUB,
        -th["crew_o2"] + _BLSS_KO2 * photo + _BLSS_LEAK * (_BLSS_O2AMB - O2)
        + backup * _BLSS_O2CAP,
        0.0,
    ])


BLSS = CalModel(
    name="blss", deriv=_blss_deriv,
    fit_params=("Pmax", "crew_co2", "crew_o2"),
    prior={"Pmax": (466.0, 0.3), "crew_co2": (300.0, 0.3),
           "crew_o2": (0.30, 0.3)},
    obs_map={"CO2": (0, 1.0, 0.0), "O2": (1, 1.0, 0.0)},
)


# ---- aquaculture (from jax_models.py) ---------------------------------------
AQUACULTURE = CalModel(
    name="aquaculture", deriv=_aqua.deriv,
    fit_params=_aqua.FIT_PARAMS, prior=_aqua.PRIOR,
    obs_map={"TAN": (0, 1.0, 0.0), "NO2": (1, 1.0, 0.0),
             "NO3": (2, 1.0, 0.0), "DO": (3, 1.0, 0.0)},
)


MODELS = {
    "aquaculture": AQUACULTURE,
    "soil": SOIL,
    "bioremediation": BIOREMEDIATION,
    "blss": BLSS,
}


def get_model(name: str) -> CalModel:
    if name not in MODELS:
        raise KeyError(
            f"no calibration model for domain '{name}'; "
            f"available: {sorted(MODELS)}")
    return MODELS[name]
