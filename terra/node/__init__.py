"""Terra edge-node runtime.

    from terra.node import NodeRunner, SimulatedDriver
    from terra.domains import aquaculture

    spec, sim = aquaculture.simulate()
    NodeRunner(spec, SimulatedDriver(spec, sim)).run(on_event=print)

Numpy-only; runs on Pi-class hardware. Swap ``SimulatedDriver`` for an
I2C/analog driver to go live.
"""
from .driver import SensorDriver, SimulatedDriver
from .runner import NodeRunner, NodeConfig
from .selftest import self_test

__all__ = ["SensorDriver", "SimulatedDriver", "NodeRunner", "NodeConfig",
           "self_test"]
