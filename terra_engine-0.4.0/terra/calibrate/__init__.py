"""Offline Bayesian calibration layer (optional).

The edge engine is numpy-only and imports nothing here. This subpackage adds a
JAX/NumPyro HMC-NUTS fit for a domain's process parameters, used offline to
calibrate a site before the edge UKF is trusted operationally.

    from terra.calibrate import HAS_JAX, fit_nuts
    # requires: pip install terra-engine[calibrate]

``HAS_JAX`` is False when the optional dependencies are absent, so callers and
tests can degrade gracefully instead of failing at import.
"""
try:
    import jax  # noqa: F401
    import numpyro  # noqa: F401
    HAS_JAX = True
except Exception:  # pragma: no cover - exercised only without the extra
    HAS_JAX = False

if HAS_JAX:
    from .nuts import (
        fit_nuts, CalibrationResult, rollout,
        calibrated_spec, calibrate_and_build,
    )
    from .models import MODELS, get_model
    from .spec import CalModel
    from . import jax_models

__all__ = ["HAS_JAX"]
if HAS_JAX:
    __all__ += [
        "fit_nuts", "CalibrationResult", "rollout",
        "calibrated_spec", "calibrate_and_build",
        "MODELS", "get_model", "CalModel", "jax_models",
    ]
