"""Deprecated. Aquaculture math moved to terra.domains.aquaculture in v0.2.

Kept as a shim so old imports don't hard-fail.
"""
from .domains.aquaculture import (  # noqa: F401
    RASParams,
    tan_production,
    unionized_nh3_fraction,
    deriv,
    deriv_batch,
)
