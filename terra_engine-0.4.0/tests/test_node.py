"""Edge-node runtime tests (numpy only, no hardware)."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from terra.node import NodeRunner, NodeConfig, SimulatedDriver, self_test
from terra.domains import aquaculture


def test_node_runs_and_surfaces_events():
    spec, sim = aquaculture.simulate(fault=True)
    fd, tmp = tempfile.mkstemp(suffix=".json")
    os.close(fd); os.remove(tmp)
    events = []
    try:
        r = NodeRunner(spec, SimulatedDriver(spec, sim), NodeConfig(state_path=tmp))
        res = r.run(on_event=events.append)
        assert res["run_cycles"] == len(sim["t"])
        assert os.path.exists(tmp)                    # state persisted
        assert len(events) >= 1                       # a fault run raises events
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    print(f"  ran {res['run_cycles']} cycles, {len(events)} events")


def test_state_persists_and_resumes():
    spec, sim = aquaculture.simulate(fault=True)
    fd, tmp = tempfile.mkstemp(suffix=".json")
    os.close(fd); os.remove(tmp)
    try:
        NodeRunner(spec, SimulatedDriver(spec, sim),
                   NodeConfig(state_path=tmp, max_cycles=20)).run()
        resumed = NodeRunner(spec, SimulatedDriver(spec, sim),
                             NodeConfig(state_path=tmp, max_cycles=1))
        assert resumed.cycles >= 20                   # picked up prior lifetime cycles
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    print(f"  resumed at cycle {resumed.cycles}")


def test_selftest_passes():
    ok, checks = self_test("aquaculture")
    assert ok, checks
    print("  " + ", ".join(f"{n}={'ok' if v else 'FAIL'}" for n, v in checks))


def test_store_and_forward_buffers_then_flushes():
    spec, sim = aquaculture.simulate(fault=True)
    fd, tS = tempfile.mkstemp(suffix=".json"); os.close(fd); os.remove(tS)
    fd, tB = tempfile.mkstemp(suffix=".json"); os.close(fd); os.remove(tB)
    received = []
    offline = [True]

    def sink(rec):
        if offline[0]:
            raise RuntimeError("comms offline")
        received.append(rec)

    try:
        # offline: events accumulate in the buffer, nothing delivered
        r = NodeRunner(spec, SimulatedDriver(spec, sim),
                       NodeConfig(state_path=tS, buffer_path=tB), sink=sink)
        res = r.run()
        assert res["buffered"] >= 1, res
        assert received == []
        assert os.path.exists(tB)                       # persisted for later

        # reconnect: a fresh node loads the outbox and flushes it
        offline[0] = False
        r2 = NodeRunner(spec, SimulatedDriver(spec, sim),
                        NodeConfig(state_path=tS, buffer_path=tB), sink=sink)
        r2._flush()
        assert len(received) >= 1
        assert len(r2.buffer) == 0
    finally:
        for p in (tS, tB):
            if os.path.exists(p):
                os.remove(p)
    print(f"  buffered {res['buffered']} offline, flushed {len(received)} on reconnect")


def test_controller_enacts_through_actuator():
    from terra import EngineConfig
    from terra.control import Controller, policy_for
    from terra.node import SimulatedActuatorDriver

    spec, sim = aquaculture.simulate(fault=True)
    cfg = EngineConfig(forecast_horizon_h=12)
    ctrl = Controller(spec, cfg, policy_for("aquaculture"), authorized=True)
    act = SimulatedActuatorDriver()
    fd, tS = tempfile.mkstemp(suffix=".json"); os.close(fd); os.remove(tS)
    try:
        r = NodeRunner(spec, SimulatedDriver(spec, sim),
                       NodeConfig(state_path=tS, buffer_path=tS + ".b"),
                       controller=ctrl, actuator=act)
        r.run()
        assert len(act.log) >= 1, "expected at least one enacted action"
        assert len(r.recommendations) >= 1
    finally:
        for p in (tS, tS + ".b"):
            if os.path.exists(p):
                os.remove(p)
    print(f"  {len(r.recommendations)} recommendations, {len(act.log)} actions enacted")


if __name__ == "__main__":
    n = 0
    for fn in (test_node_runs_and_surfaces_events, test_state_persists_and_resumes,
               test_selftest_passes, test_store_and_forward_buffers_then_flushes,
               test_controller_enacts_through_actuator):
        fn()
        print(f"PASS  {fn.__name__}")
        n += 1
    print(f"\n{n}/{n} passed")
