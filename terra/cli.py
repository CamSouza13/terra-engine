"""Unified command-line interface for Terra Engine.

    terra domains                       # list available domains
    terra demo --domain aquaculture     # run a domain, print the event log
    terra validate --domain soil        # score the engine vs ground truth + baseline
    terra calibrate --domain blss       # offline Bayesian fit (needs the extra)

The core commands are numpy-only; ``calibrate`` lazily imports the optional
jax/numpyro extra and prints an install hint if it is missing.
"""
from __future__ import annotations

import argparse
import sys


def _get_domain(name: str):
    from terra.domains import DOMAINS
    if name not in DOMAINS:
        raise SystemExit(f"unknown domain '{name}'; choices: {sorted(DOMAINS)}")
    return DOMAINS[name]


def cmd_domains(_args) -> int:
    from terra.domains import DOMAINS
    for n in sorted(DOMAINS):
        print(n)
    return 0


def cmd_demo(args) -> int:
    from terra import TerraEngine, EngineConfig
    mod = _get_domain(args.domain)
    spec, sim = mod.simulate()
    eng = TerraEngine(spec, EngineConfig())
    times = sim["t"]
    dt = float(times[1] - times[0])
    uf = sim.get("u_forecast")
    for i, t in enumerate(times):
        eng.step(t, dt, sim["meas"][i], sim["u"][i], u_forecast=uf)
    print(f"{spec.name}: {sim.get('fault', 'run complete')}")
    if not eng.events:
        print("  (no events)")
    for t_ev, level, msg in eng.events:
        print(f"  {t_ev:6.1f} h  {level:5}  {msg}")
    return 0


def cmd_validate(args) -> int:
    from terra.validate import run_validation
    mod = _get_domain(args.domain)
    spec, sim = mod.simulate()
    run_validation(spec, sim).print_report()
    return 0


def cmd_calibrate(args) -> int:
    try:
        from terra.calibrate import HAS_JAX
    except Exception:
        HAS_JAX = False
    if not HAS_JAX:
        print("calibration needs the optional extra: "
              "pip install terra-engine[calibrate]")
        return 1
    from terra.calibrate import calibrate_and_build
    mod = _get_domain(args.domain)
    spec, sim = mod.simulate(fault=False)
    _, res = calibrate_and_build(
        spec, sim["t"], sim["u"], sim["meas"],
        num_warmup=args.warmup, num_samples=args.samples)
    res.print_report()
    return 0


def cmd_node(args) -> int:
    import os
    from terra.node import NodeRunner, NodeConfig, SimulatedDriver, self_test
    if args.selftest:
        ok, checks = self_test(args.domain)
        for name, passed in checks:
            print(f"  [{'ok ' if passed else 'FAIL'}] {name}")
        print("self-test:", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    mod = _get_domain(args.domain)
    spec, sim = mod.simulate()
    state = os.environ.get("TERRA_STATE", "terra_node_state.json")
    runner = NodeRunner(
        spec, SimulatedDriver(spec, sim),
        NodeConfig(state_path=state, max_cycles=args.cycles))
    runner.run(on_event=lambda ev: print(f"  {ev[0]:6.1f} h  {ev[1]:5}  {ev[2]}"),
               banner=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="terra", description="Terra Engine CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("domains", help="list domains").set_defaults(func=cmd_domains)

    d = sub.add_parser("demo", help="run a domain and print events")
    d.add_argument("--domain", default="aquaculture")
    d.set_defaults(func=cmd_demo)

    v = sub.add_parser("validate", help="score engine vs ground truth + baseline")
    v.add_argument("--domain", default="aquaculture")
    v.set_defaults(func=cmd_validate)

    c = sub.add_parser("calibrate", help="offline Bayesian fit (optional extra)")
    c.add_argument("--domain", default="aquaculture")
    c.add_argument("--warmup", type=int, default=200)
    c.add_argument("--samples", type=int, default=300)
    c.set_defaults(func=cmd_calibrate)

    n = sub.add_parser("node", help="run the edge-node service (simulated driver)")
    n.add_argument("--domain", default="aquaculture")
    n.add_argument("--cycles", type=int, default=None)
    n.add_argument("--selftest", action="store_true", help="run bring-up self-test")
    n.set_defaults(func=cmd_node)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
