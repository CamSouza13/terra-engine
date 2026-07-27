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


def cmd_analyze(args) -> int:
    from terra import analysis
    mod = _get_domain(args.domain)
    spec, sim = mod.simulate()
    if args.kind == "degradation":
        print(f"Graceful degradation — {spec.name}")
        for r in analysis.degradation_sweep(spec, sim):
            print(f"  {r['n_channels']} ch {str(r['active']):32s} "
                  f"RMSE {r['rmse']:.3f}  cover {r['coverage']:.0%}")
    elif args.kind == "drift":
        d = analysis.drift_detectability(spec, args.channel or list(spec.channels)[0])
        print(f"Drift detectability — {spec.name} / {d['channel']} "
              f"(noise {d['sigma']:.3g}, process wander {d['process_wander']:.3g})")
        for T, w in d["windows"].items():
            print(f"  {T:4d} h window: min recoverable drift "
                  f"{w['min_rate_per_day']:.3g} /day")
    elif args.kind == "bench":
        b = analysis.edge_benchmark(spec, sim)
        print(f"Edge benchmark — {spec.name}: {b['ms_per_step']:.3f} ms/step, "
              f"{b['steps_per_s']:.0f} steps/s, peak {b['peak_mem_mb']:.2f} MB, "
              f"state {b['state_dim']}, {b['channels']} channels")
    elif args.kind == "alerts":
        q = analysis.alert_quality(mod)
        lead = q["engine_mean_lead_h"]
        lead_s = f"{lead:.1f} h" if lead is not None else "n/a"
        print(f"Alert quality — {spec.name} ({q['seeds']} seeds): "
              f"engine detect {q['engine_detect_rate']:.0%} vs gauge "
              f"{q['baseline_detect_rate']:.0%}; mean lead {lead_s}; "
              f"false alarms engine {q['engine_false_alarms']} / "
              f"gauge {q['baseline_false_alarms']}")
    elif args.kind == "roi":
        r = analysis.roi_distribution()
        p = r["percentiles"]
        print(f"Annual value ($) — mean {r['mean']:,.0f}; "
              f"P10 {p[10]:,.0f}  P50 {p[50]:,.0f}  P90 {p[90]:,.0f}")
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
    print("\nConvergence (split-R-hat / ESS):")
    for k, d in res.diagnostics().items():
        print(f"  {k:6}  rhat={d['rhat']:.3f}  ess={d['ess']:.0f}")
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
    from terra.node import report as rep
    creds = None
    bundle = None
    if args.enroll or args.server or args.offline:
        if args.enroll:
            if not args.server:
                raise SystemExit("--enroll requires --server <platform url>")
            creds = rep.enroll(args.server, args.enroll, node_id=args.node_id,
                               name=args.name, domain=args.domain)
            print(f"enrolled as {creds['node_id']} -> {creds['server']}")
        else:
            creds = rep.load_creds(args.server)
            if creds is None and not args.offline:
                raise SystemExit("no saved node credentials; run with --enroll <token> first")
        if creds is not None and not args.offline:
            rep.fetch_config(creds)   # refresh the provisioning bundle from the platform
        bundle = rep.load_config()

    # build the engine spec from the provisioned bundle if present, else the CLI domain
    if bundle and bundle.get("domain"):
        spec = rep.build_spec_from_bundle(bundle)
        _, sim = _get_domain(bundle["domain"]).simulate()
        params = list((bundle.get("params") or {}).keys()) or "defaults"
        print(f"running provisioned config: domain={bundle['domain']} params={params}"
              f"{' (offline)' if args.offline else ''}")
    else:
        spec, sim = _get_domain(args.domain).simulate()

    reporter = None
    if creds is not None and not args.offline:
        reporter = rep.ServerReporter(spec, creds, interval_s=args.interval)
        print(f"reporting to {reporter.server} every {args.interval:g}s")

    state = os.environ.get("TERRA_STATE", "terra_node_state.json")
    runner = NodeRunner(
        spec, SimulatedDriver(spec, sim),
        NodeConfig(state_path=state, max_cycles=args.cycles))
    runner.run(on_event=lambda ev: print(f"  {ev[0]:6.1f} h  {ev[1]:5}  {ev[2]}"),
               banner=True, on_cycle=(reporter.on_cycle if reporter else None))
    if reporter:
        print(f"reported {reporter.sent} heartbeats ({reporter.failures} failed)")
    return 0


def cmd_serve(args) -> int:
    from terra.server import serve
    return serve(domain=args.domain, port=args.port, autonomy=args.autonomy)


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

    a = sub.add_parser("analyze", help="robustness / drift / edge-cost / value analyses")
    a.add_argument("kind", choices=["degradation", "drift", "bench", "alerts", "roi"])
    a.add_argument("--domain", default="aquaculture")
    a.add_argument("--channel", default=None, help="channel for drift analysis")
    a.set_defaults(func=cmd_analyze)

    c = sub.add_parser("calibrate", help="offline Bayesian fit (optional extra)")
    c.add_argument("--domain", default="aquaculture")
    c.add_argument("--warmup", type=int, default=200)
    c.add_argument("--samples", type=int, default=300)
    c.set_defaults(func=cmd_calibrate)

    n = sub.add_parser("node", help="run the edge-node service (simulated driver)")
    n.add_argument("--domain", default="aquaculture")
    n.add_argument("--cycles", type=int, default=None)
    n.add_argument("--selftest", action="store_true", help="run bring-up self-test")
    n.add_argument("--enroll", metavar="TOKEN", default=None,
                   help="redeem a one-time enrollment token from the platform")
    n.add_argument("--server", default=None,
                   help="platform URL to report heartbeats to")
    n.add_argument("--node-id", default=None, help="stable id for this node")
    n.add_argument("--name", default=None, help="display name for this node")
    n.add_argument("--interval", type=float, default=10.0,
                   help="seconds between heartbeats (default 10)")
    n.add_argument("--offline", action="store_true",
                   help="run from the saved config bundle with no network access")
    n.set_defaults(func=cmd_node)

    s = sub.add_parser("serve", help="serve the live engine API + web console")
    s.add_argument("--domain", default="aquaculture")
    s.add_argument("--port", type=int, default=8700)
    s.add_argument("--autonomy", action="store_true", help="enact actions autonomously")
    s.set_defaults(func=cmd_serve)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
