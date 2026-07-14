"""Replay real logged sensor data through a domain spec.

The CSV is a time column plus one column per sensor channel (names matching the
domain's channels, or remapped via `channel_map`). A blank cell or a missing
column simply means that sensor did not report — the engine rides on whatever
channels are present, so partial and irregular logs work out of the box.

Stdlib + NumPy only; no pandas dependency.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable
import csv as _csv

import numpy as np

from .core import TerraEngine, EngineConfig, SystemSpec, Estimate


def _parse_time(raw: str) -> float | datetime:
    raw = raw.strip()
    try:
        return float(raw)                       # already numeric (hours)
    except ValueError:
        pass
    s = raw.replace("Z", "+00:00")
    return datetime.fromisoformat(s)            # ISO 8601 timestamp


def _to_hours(values: list) -> np.ndarray:
    if values and isinstance(values[0], datetime):
        t0 = values[0]
        return np.array([(v - t0).total_seconds() / 3600.0 for v in values])
    return np.array([float(v) for v in values])


def load_csv(path: str, time_col: str = "t") -> tuple[list[str], list[dict]]:
    """Return (header, rows) with each row a dict of column -> raw string."""
    with open(path, newline="") as f:
        reader = _csv.DictReader(f)
        rows = [dict(r) for r in reader]
        header = reader.fieldnames or []
    if time_col not in header:
        raise ValueError(f"time column {time_col!r} not found; have {header}")
    return header, rows


@dataclass
class ReplayResult:
    engine: TerraEngine
    t: np.ndarray
    estimates: list[Estimate]

    @property
    def events(self):
        return self.engine.events


def run_csv(
    spec: SystemSpec,
    path: str,
    *,
    time_col: str = "t",
    channel_map: dict[str, str] | None = None,
    u_col: str | None = None,
    u_const: Any | Callable[[float], Any] = None,
    u_forecast: Any = None,
    config: EngineConfig | None = None,
) -> ReplayResult:
    """Run `spec` over a CSV of logged readings.

    channel_map : {csv_column: channel_name}. Defaults to identity for any
                  column whose name matches a channel in the spec.
    u_col       : column holding the known input drive (e.g., feed/dosing).
    u_const     : scalar or callable(t) input, used when u_col is absent.
    u_forecast  : input to assume when forecasting (see TerraEngine.step).
    """
    header, rows = load_csv(path, time_col)
    if channel_map is None:
        channel_map = {c: c for c in spec.channels if c in header}
    # invert to channel -> csv column
    chan_to_col = {ch: col for col, ch in channel_map.items()}

    times = _to_hours([_parse_time(r[time_col]) for r in rows])
    engine = TerraEngine(spec, config or EngineConfig())
    estimates: list[Estimate] = []

    for i, r in enumerate(rows):
        meas = {}
        for ch in spec.channels:
            col = chan_to_col.get(ch)
            if col is None:
                continue
            raw = (r.get(col) or "").strip()
            if raw == "":
                continue                        # sensor silent this row
            try:
                meas[ch] = float(raw)
            except ValueError:
                continue

        if u_col is not None and (r.get(u_col) or "").strip() != "":
            u = float(r[u_col])
        elif callable(u_const):
            u = u_const(times[i])
        else:
            u = u_const

        dt = times[i] - times[i - 1] if i > 0 else (
            times[1] - times[0] if len(times) > 1 else 1.0)
        est = engine.step(times[i], dt, meas, u, u_forecast=u_forecast)
        estimates.append(est)

    return ReplayResult(engine=engine, t=times, estimates=estimates)
