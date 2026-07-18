"""Naive baselines to benchmark the engine against.

The engine's whole claim is that modelling the coupled loop catches a
system-level failure earlier than watching individual gauges. These baselines
make that claim measurable. ``threshold_alarm`` is the incumbent approach: a
raw threshold on the directly-measured safety quantities. ``true_breach_times``
reads the real breach time from the simulator's ground truth so lead times can
be scored.
"""
from __future__ import annotations

import numpy as np


def _identity_state_channels(spec) -> dict:
    """State index -> channel name, for channels that measure a state directly.

    A channel is 'direct' if its observation function is the identity on the
    state it is tagged with (obs(e_i) == 1 and obs(0) == 0). Proxy channels
    (EC, ORP, CO2 flux) are intentionally excluded — a raw-gauge operator does
    not invert them, which is exactly the blind spot the engine covers.
    """
    out: dict = {}
    n = len(spec.state_names)
    zero = np.zeros(n)
    for name, ch in spec.channels.items():
        if ch.state is None:
            continue
        e = np.zeros(n)
        e[ch.state] = 1.0
        try:
            if abs(ch.obs(e) - 1.0) < 1e-9 and abs(ch.obs(zero)) < 1e-9:
                out.setdefault(ch.state, name)
        except Exception:
            continue
    return out


def _eval_safety(spec, x) -> dict:
    out = {}
    for s in spec.safety:
        lv = float(s.value(x[:, None], spec.params, spec.env)[0])
        out[s.name] = lv
    return out


def threshold_alarm(spec, sim) -> dict:
    """First raw-gauge crossing time per safety target (or None).

    Reconstructs a 'measured state' from the direct channels (held between
    reports, initialised at x0), evaluates each safety target on it, and returns
    the first time it crosses the limit.
    """
    idmap = _identity_state_channels(spec)
    x = np.array(spec.x0, float)
    times = sim["t"]
    meas = sim["meas"]
    first = {s.name: None for s in spec.safety}
    for i, t in enumerate(times):
        row = meas[i] if i < len(meas) else {}
        for si, ch in idmap.items():
            v = row.get(ch)
            if v is not None and np.isfinite(v):
                x[si] = float(v)
        levels = _eval_safety(spec, x)
        for s in spec.safety:
            if first[s.name] is None:
                lv = levels[s.name]
                hit = lv > s.limit if s.direction == ">" else lv < s.limit
                if hit:
                    first[s.name] = float(t)
    return first


def true_breach_times(spec, sim) -> dict:
    """First time each safety target truly breaches, from the simulator truth."""
    truth = sim["truth"]
    times = sim["t"]
    out = {s.name: None for s in spec.safety}
    for i, t in enumerate(times):
        levels = _eval_safety(spec, np.asarray(truth[i], float))
        for s in spec.safety:
            if out[s.name] is None:
                lv = levels[s.name]
                hit = lv > s.limit if s.direction == ">" else lv < s.limit
                if hit:
                    out[s.name] = float(t)
    return out
