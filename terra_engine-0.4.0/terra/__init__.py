"""Terra Engine — one Bayesian inference core for every closed biogeochemical loop.

    from terra import TerraEngine
    from terra.domains import aquaculture

    spec, sim = aquaculture.simulate()
    engine = TerraEngine(spec)
    for t, meas, u in zip(sim["t"], sim["meas"], sim["u"]):
        est = engine.step(t, sim["t"][1] - sim["t"][0], meas, u)

The same engine runs soil, bioremediation, and closed-habitat life support —
each is just a different SystemSpec. Sensors can be enabled or disabled per
timestep; the filter rides on whatever data is available.
"""
from .core import (
    TerraEngine,
    EngineConfig,
    SystemSpec,
    Channel,
    SafetyTarget,
    Estimate,
    simulate_truth,
    mc_forecast,
    rk4,
    rk4_batch,
)
from .ukf import UnscentedKalmanFilter
from .control import Controller, ControlPolicy, Actuator, Recommendation, policy_for
from . import domains

__all__ = [
    "TerraEngine", "EngineConfig", "SystemSpec", "Channel", "SafetyTarget",
    "Estimate", "simulate_truth", "mc_forecast", "rk4", "rk4_batch",
    "UnscentedKalmanFilter", "domains",
    "Controller", "ControlPolicy", "Actuator", "Recommendation", "policy_for",
]

__version__ = "0.4.0"
