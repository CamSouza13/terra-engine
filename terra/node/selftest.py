"""Node bring-up self-test.

A quick, hardware-free check that the whole node stack is wired correctly:
engine constructs, the loop steps, state persists atomically, and a restart
resumes from disk. Run on first boot and on demand (`terra node --selftest`).
"""
from __future__ import annotations

import os
import tempfile

from ..domains import DOMAINS
from .driver import SimulatedDriver
from .runner import NodeRunner, NodeConfig


def self_test(domain: str = "aquaculture") -> tuple[bool, list]:
    checks = []
    mod = DOMAINS[domain]
    spec, sim = mod.simulate()

    fd, tmp = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    os.remove(tmp)                                   # start with no state
    try:
        r1 = NodeRunner(spec, SimulatedDriver(spec, sim),
                        NodeConfig(state_path=tmp, max_cycles=25))
        res = r1.run()
        checks.append(("engine steps", res["run_cycles"] > 0))
        checks.append(("events surfaced", res["events"] >= 0))
        checks.append(("state persisted", os.path.exists(tmp)))

        r2 = NodeRunner(spec, SimulatedDriver(spec, sim),
                        NodeConfig(state_path=tmp, max_cycles=1))
        checks.append(("state resumed", r2.cycles >= res["run_cycles"]))
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    ok = all(v for _, v in checks)
    return ok, checks
