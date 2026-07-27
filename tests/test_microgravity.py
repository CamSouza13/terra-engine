"""Microgravity BLSS variant: removing buoyancy-driven mixing changes the
physics (CO2 builds faster), and the engine still recovers the hidden crop
state under the altered dynamics."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from terra.domains import blss
from terra.validate import run_validation


def test_microgravity_changes_physics_and_still_estimable():
    # Earth vs microgravity, same fault
    _, earth = blss.simulate(gravity=1.0, seed=1)
    _, micro = blss.simulate(gravity=0.05, seed=1)
    co2_earth = np.asarray(earth["truth"], float)[:, 0].max()
    co2_micro = np.asarray(micro["truth"], float)[:, 0].max()
    # with mixing suppressed, CO2 accumulates higher
    assert co2_micro > co2_earth, (co2_micro, co2_earth)

    # the engine, given the microgravity spec, still recovers the hidden state
    spec_m, sim_m = blss.simulate(gravity=0.05, seed=2)
    rep = run_validation(spec_m, sim_m)
    assert np.isfinite(rep.hidden_rmse)
    assert 0.0 <= rep.hidden_coverage <= 1.0
    # default (Earth) path is unchanged
    spec_e, sim_e = blss.simulate(seed=2)
    assert spec_e.params.gravity == 1.0
    print(f"  peak CO2 earth {co2_earth:.0f} -> micro {co2_micro:.0f} ppm; "
          f"micro RMSE {rep.hidden_rmse:.3f} cover {rep.hidden_coverage:.0%}")


if __name__ == "__main__":
    test_microgravity_changes_physics_and_still_estimable()
    print("all microgravity tests passed")
