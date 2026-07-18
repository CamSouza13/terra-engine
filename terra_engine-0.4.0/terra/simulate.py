"""Deprecated module path. Simulators now live per-domain in terra.domains
and share terra.core.simulate_truth (v0.2)."""
from .domains.aquaculture import simulate as simulate_ras  # noqa: F401
