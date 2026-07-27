"""Self-healing mesh: routing re-routes on failure, sensing imputes a dropped node."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from terra.mesh import build_grid, smooth_field


def test_routing_self_heals():
    net = build_grid(4, 4)                      # 16 nodes, gateway at (0,0)
    r0 = net.routes()
    assert net.connectivity() == 1.0            # everyone reaches the gateway
    far = (3, 3)
    hops0 = r0[far]["hops"]

    # sever the far node's cheapest hop; it must re-route, still connected
    nh = r0[far]["next_hop"]
    net.fail_link(far, nh)
    r1 = net.routes()
    assert r1[far] is not None                  # healed onto a surviving path
    assert r1[far]["hops"] >= hops0             # detour is at least as long
    assert net.connectivity() == 1.0

    # kill an interior node entirely; network stays fully connected via detours
    net.fail_node((1, 1))
    assert net.connectivity() == 1.0
    print(f"  routing: (3,3) {hops0}->{r1[far]['hops']} hops after link loss; "
          f"connectivity 100% after node loss")


def test_sensing_self_heals():
    net = build_grid(5, 5, seed=2)
    field = smooth_field(net.pos, seed=3)
    readings = dict(field)                       # every node reporting truth

    node = (2, 2)                                # drop this node's sensor
    truth = field[node]
    readings[node] = None
    est = net.impute(node, readings)
    assert est is not None
    err = abs(est - truth)
    # imputed value tracks the true local field within a tight bound
    assert err < 0.5, (est, truth, err)

    # average imputation error across all interior nodes stays small
    errs = []
    for n in net.pos:
        r = dict(field); r[n] = None
        e = net.impute(n, r)
        if e is not None:
            errs.append(abs(e - field[n]))
    mae = float(np.mean(errs))
    assert mae < 0.4, mae
    print(f"  sensing: dropped {node} imputed err {err:.3f}; fleet MAE {mae:.3f}")


if __name__ == "__main__":
    test_routing_self_heals()
    test_sensing_self_heals()
    print("all mesh tests passed")
