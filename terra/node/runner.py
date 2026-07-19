"""The Terra edge-node runtime.

Wraps a ``TerraEngine`` in a service loop: read a cycle from the driver, step
the estimator, surface new events, and persist state atomically so the node
survives restarts. Numpy-only, so it runs on a Raspberry Pi Zero 2 W (512 MB)
and is fully testable without hardware.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import numpy as np

from ..core import TerraEngine, EngineConfig

BANNER = r"""
  ______                   ____      _
 /_  __/__ ___________ _  / __/___ _(_)__  ___
  / / / -_) __/ __/ _ `/ / _// _ `/ / _ \/ -_)
 /_/  \__/_/ /_/  \_,_/ /___/\_, /_/_//_/\__/
   edge node  ·  from the mud/___/ to the moon
"""


@dataclass
class NodeConfig:
    state_path: str = "terra_node_state.json"
    buffer_path: str = "terra_node_buffer.json"   # store-and-forward outbox
    max_cycles: int | None = None
    sleep_s: float = 0.0          # wall-clock pacing; 0 for replay/tests
    persist_every: int = 1        # write state every N cycles


class NodeRunner:
    def __init__(self, spec, driver, config: NodeConfig | None = None,
                 engine_config: EngineConfig | None = None,
                 sink=None, controller=None, actuator=None):
        self.spec = spec
        self.driver = driver
        self.cfg = config or NodeConfig()
        self.engine = TerraEngine(spec, engine_config or EngineConfig())
        self.sink = sink              # callable(record) -> None; may raise if offline
        self.controller = controller  # optional terra.control.Controller
        self.actuator = actuator      # optional terra.node.ActuatorDriver
        self.recommendations: list = []
        self.buffer: list = []        # store-and-forward outbox
        self.cycles = 0               # lifetime cycles (persisted)
        self._load_state()
        self._load_buffer()

    # ---- state persistence (atomic) ----
    def _load_state(self) -> None:
        p = self.cfg.state_path
        if not os.path.exists(p):
            return
        try:
            with open(p) as f:
                d = json.load(f)
            self.engine.ukf.x = np.array(d["x"], float)  # type: ignore[assignment]
            self.engine.ukf.P = np.array(d["P"], float)  # type: ignore[assignment]
            self.cycles = int(d.get("cycles", 0))
        except Exception:
            pass                      # corrupt/partial state: start clean

    def _save_state(self) -> None:
        d = {
            "domain": self.spec.name,
            "x": self.engine.ukf.x.tolist(),
            "P": self.engine.ukf.P.tolist(),
            "cycles": self.cycles,
        }
        tmp = self.cfg.state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(d, f)
        os.replace(tmp, self.cfg.state_path)    # atomic swap

    # ---- store-and-forward outbox ----
    def _load_buffer(self) -> None:
        p = self.cfg.buffer_path
        if os.path.exists(p):
            try:
                with open(p) as f:
                    self.buffer = json.load(f)
            except Exception:
                self.buffer = []

    def _save_buffer(self) -> None:
        tmp = self.cfg.buffer_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.buffer, f)
        os.replace(tmp, self.cfg.buffer_path)

    def _emit(self, record: dict) -> None:
        """Queue a record for the sink; buffer and persist if the sink is offline."""
        if self.sink is None:
            return                              # nowhere to forward: don't buffer
        self.buffer.append(record)
        self._flush()

    def _flush(self) -> None:
        if self.sink is None:
            return
        while self.buffer:
            try:
                self.sink(self.buffer[0])
            except Exception:
                break                           # offline: keep the rest buffered
            self.buffer.pop(0)
        self._save_buffer()

    # ---- the service loop ----
    def run(self, on_event=None, banner: bool = False, on_cycle=None) -> dict:
        if banner:
            print(BANNER)
            print(f"  domain: {self.spec.name}  |  resuming at cycle {self.cycles}\n")
        uf = getattr(self.driver, "u_forecast", None)
        seen = len(self.engine.events)
        run_cycles = 0
        for t, dt, meas, u in self.driver.steps():
            est = self.engine.step(t, dt, meas, u, u_forecast=uf)
            self.cycles += 1
            run_cycles += 1
            if on_cycle is not None:
                try:
                    on_cycle(est, t)
                except Exception:
                    pass
            # controller: recommend, and enact if authorized + an actuator is wired
            if self.controller is not None:
                rec = self.controller.recommend(est, u)
                if rec is not None:
                    self.recommendations.append(rec)
                    if rec.enacted and self.actuator is not None:
                        self.actuator.apply(rec)
                    self.engine.events.append(
                        (t, "ACTION" if rec.enacted else "ADVICE", rec.message()))
            # surface + forward new events (store-and-forward if the sink is offline)
            for ev in self.engine.events[seen:]:
                if on_event:
                    on_event(ev)
                self._emit({"t": ev[0], "level": ev[1], "msg": ev[2]})
            seen = len(self.engine.events)
            if run_cycles % self.cfg.persist_every == 0:
                self._save_state()
            if self.cfg.sleep_s:
                time.sleep(self.cfg.sleep_s)
            if self.cfg.max_cycles and run_cycles >= self.cfg.max_cycles:
                break
        self._flush()
        self._save_state()
        return {"cycles": self.cycles, "run_cycles": run_cycles,
                "events": len(self.engine.events), "buffered": len(self.buffer)}
