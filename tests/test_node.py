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


if __name__ == "__main__":
    n = 0
    for fn in (test_node_runs_and_surfaces_events, test_state_persists_and_resumes,
               test_selftest_passes):
        fn()
        print(f"PASS  {fn.__name__}")
        n += 1
    print(f"\n{n}/{n} passed")
