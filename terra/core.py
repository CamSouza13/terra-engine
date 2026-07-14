"""Domain-agnostic biogeochemical inference core.

The same engine runs every closed loop. A *domain* — aquaculture, soil/farming,
bioremediation, closed-habitat life support — is nothing but a `SystemSpec`:
a set of state variables, a process model, a set of sensor channels, and the
safety limits worth forecasting against.

Two properties make it "one engine, every loop":

  * Channels are optional per timestep. The engine rides on whatever sensors
    report right now. Drop a probe and the filter leans on the model and the
    remaining channels instead of failing. This is how the same core adapts to
    wildly different instrumentation and data availability.

  * The hidden state (a biofilter's efficiency, a soil's microbial activity, a
    remediation's active biomass, a crop's photosynthetic capacity) is never
    measured. It is inferred from how the whole system explains the data — and
    it is almost always the thing that fails first.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
import numpy as np

from .ukf import UnscentedKalmanFilter


# ---- rk4 integrators over a domain-supplied derivative ------------------------
# (SystemSpec / Channel / SafetyTarget defined below; simulate_truth at bottom)

def rk4(x: np.ndarray, dt: float, deriv: Callable, u: Any, params: Any,
        nonneg: np.ndarray | None = None) -> np.ndarray:
    k1 = deriv(x, u, params)
    k2 = deriv(x + 0.5 * dt * k1, u, params)
    k3 = deriv(x + 0.5 * dt * k2, u, params)
    k4 = deriv(x + dt * k3, u, params)
    xn = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    if nonneg is not None:
        xn = np.where(nonneg, np.clip(xn, 0.0, None), xn)
    return xn


def rk4_batch(X: np.ndarray, dt: float, deriv_batch: Callable, u: Any, params: Any,
              nonneg: np.ndarray | None = None) -> np.ndarray:
    k1 = deriv_batch(X, u, params)
    k2 = deriv_batch(X + 0.5 * dt * k1, u, params)
    k3 = deriv_batch(X + 0.5 * dt * k2, u, params)
    k4 = deriv_batch(X + dt * k3, u, params)
    Xn = X + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    if nonneg is not None:
        Xn[:, nonneg] = np.clip(Xn[:, nonneg], 0.0, None)
    return Xn


# ---- domain description -------------------------------------------------------

@dataclass
class Channel:
    """A sensor. `obs(x)` maps state -> the measured quantity (may be a proxy)."""
    obs: Callable[[np.ndarray], float]
    noise: float                      # 1-sigma measurement noise
    state: int | None = None          # convenience: index it primarily reflects


@dataclass
class SafetyTarget:
    """A quantity to forecast a probability of breaching."""
    name: str
    value: Callable[[np.ndarray, Any, dict], float]  # (x, params, env) -> level
    limit: float
    direction: str = ">"              # ">" breach when above, "<" when below
    units: str = ""


@dataclass
class SystemSpec:
    name: str
    state_names: list[str]
    x0: np.ndarray
    P0: np.ndarray                    # diag covariance
    process_std: np.ndarray           # per-state process-noise std
    deriv: Callable[[np.ndarray, Any, Any], np.ndarray]
    deriv_batch: Callable[[np.ndarray, Any, Any], np.ndarray]
    channels: dict[str, Channel]
    params: Any
    hidden: str                       # name of the hidden health/efficiency state
    hidden_baseline: float = 1.0
    hidden_alert_frac: float = 0.85
    hidden_higher_is_healthy: bool = True
    env: dict = field(default_factory=dict)
    safety: list[SafetyTarget] = field(default_factory=list)
    budget: Callable[[np.ndarray, Any, Any], float] | None = None
    nonneg: np.ndarray | None = None  # bool mask of states floored at zero

    def idx(self, name: str) -> int:
        return self.state_names.index(name)


# ---- one estimator step of output --------------------------------------------

@dataclass
class Estimate:
    t: float
    x: np.ndarray
    P: np.ndarray
    used_channels: list[str]
    nis: float
    hidden: float
    hidden_std: float
    budget_residual: float | None
    risks: dict[str, dict]            # target name -> {p, t_cross, level}


# ---- the engine ---------------------------------------------------------------

@dataclass
class EngineConfig:
    forecast_horizon_h: float = 12.0
    forecast_samples: int = 300
    forecast_dt: float = 0.25
    nis_alert_per_channel: float = 4.0   # ~ chi-sq/dof threshold
    risk_alert: float = 0.30
    seed: int = 3


class TerraEngine:
    def __init__(self, spec: SystemSpec, config: EngineConfig | None = None):
        self.spec = spec
        self.cfg = config or EngineConfig()
        self._u: Any = None
        n = len(spec.state_names)
        self.ukf = UnscentedKalmanFilter(
            dim_x=n,
            fx=self._fx,
            Q=np.diag(np.asarray(spec.process_std, float) ** 2),
            alpha=1e-3, beta=2.0,
        )
        self.ukf.x = np.array(spec.x0, float)
        self.ukf.P = np.array(spec.P0, float)
        self._rng = np.random.default_rng(self.cfg.seed)
        self.history: list[Estimate] = []
        self.events: list[tuple[float, str, str]] = []
        self._announced: set[str] = set()
        self._unhealthy_run = 0

    # dynamics hook for the UKF (uses the input set at each step)
    def _fx(self, x: np.ndarray, dt: float) -> np.ndarray:
        return rk4(x, dt, self.spec.deriv, self._u, self.spec.params, self.spec.nonneg)

    def step(self, t: float, dt: float, measurements: dict[str, float],
             u: Any = None, u_forecast: Any = None) -> Estimate:
        """One measurement cycle.

        `u` is the current known input driving the model. `u_forecast` is the
        input to assume when projecting forward — for a cyclic, scheduled drive
        (feeding) pass its expected/average value so a momentary peak doesn't
        trigger a false forecast; for a step change (a pump failure) leave it as
        the current value so the forecast reacts immediately. Defaults to `u`.
        """
        self._u = u
        self.ukf.predict(dt)

        # assemble only the channels that actually reported this step
        used = [c for c in self.spec.channels
                if c in measurements and measurements[c] is not None
                and np.isfinite(measurements[c])]
        if used:
            z = np.array([measurements[c] for c in used], float)
            chans = [self.spec.channels[c] for c in used]
            hx = lambda x, _ch=chans: np.array([ch.obs(x) for ch in _ch])
            R = np.diag([ch.noise ** 2 for ch in chans])
            self.ukf.update(z, hx, R)
        else:
            # prediction-only step (no sensors reported): clear stale diagnostics
            self.ukf.innovation = np.zeros(0)
            self.ukf.nis = 0.0

        x = self.ukf.x.copy()
        P = self.ukf.P.copy()
        hi = self.spec.idx(self.spec.hidden)
        hidden = float(x[hi])
        hidden_std = float(np.sqrt(max(P[hi, hi], 0.0)))
        budget = (float(self.spec.budget(x, u, self.spec.params))
                  if self.spec.budget else None)
        risks = self._forecast(x, P, u if u_forecast is None else u_forecast)

        est = Estimate(t=t, x=x, P=P, used_channels=used, nis=self.ukf.nis,
                       hidden=hidden, hidden_std=hidden_std,
                       budget_residual=budget, risks=risks)
        self.history.append(est)
        self._log(est)
        return est

    def _forecast(self, x, P, u) -> dict[str, dict]:
        return mc_forecast(self.spec, self.cfg, self._rng, x, P, u)

    def _log(self, e: Estimate) -> None:
        s = self.spec
        base = s.hidden_baseline
        # hidden-state (efficiency/health) departure — require it to persist a
        # few steps so an initialization transient never trips a false alarm
        unhealthy = (e.hidden < s.hidden_alert_frac * base
                     if s.hidden_higher_is_healthy
                     else e.hidden > (2 - s.hidden_alert_frac) * base)
        confident = e.hidden_std < 0.3 * max(abs(base), 1e-6)
        self._unhealthy_run = self._unhealthy_run + 1 if (unhealthy and confident) else 0
        if self._unhealthy_run >= 3 and "hidden" not in self._announced:
            self._announced.add("hidden")
            pct = 100 * (e.hidden / base - 1) if base else 0.0
            self.events.append(
                (e.t, "WARN",
                 f"{s.hidden} estimated {e.hidden:.2f} "
                 f"({pct:+.0f}% vs baseline {base:.2f}); "
                 f"loop not closing as expected."))
        # forecast risks
        for name, r in e.risks.items():
            if r["p"] >= self.cfg.risk_alert and name not in self._announced:
                self._announced.add(name)
                when = f" in ~{r['t_cross']:.1f} h" if r["t_cross"] else ""
                op = ">" if r["direction"] == ">" else "<"
                self.events.append(
                    (e.t, "ALERT",
                     f"{name} forecast to breach ({op}{r['limit']:g} "
                     f"{r['units']}) with P={r['p']:.2f} within "
                     f"{self.cfg.forecast_horizon_h:.0f} h{when}."))
        # consistency breach (books stop balancing) — announce once
        dof = max(len(e.used_channels), 1)
        if e.nis / dof > self.cfg.nis_alert_per_channel and "nis" not in self._announced:
            self._announced.add("nis")
            self.events.append(
                (e.t, "INFO",
                 f"model/measurement consistency degraded "
                 f"(NIS/dof {e.nis/dof:.1f}); state posterior re-weighting."))


# ---- Monte-Carlo forecast (shared by the engine and the controller) ----------

def mc_forecast(spec: SystemSpec, cfg: "EngineConfig", rng, x, P, u) -> dict[str, dict]:
    """Propagate the posterior forward under input `u`; return per-target breach
    probability, expected crossing time, and current level. Pure in `rng`, so a
    controller can evaluate candidate actions with its own generator without
    disturbing the live filter."""
    if not spec.safety:
        return {}
    H, n, dt = cfg.forecast_horizon_h, cfg.forecast_samples, cfg.forecast_dt
    try:
        X = rng.multivariate_normal(x, P, size=n)
    except np.linalg.LinAlgError:
        X = x + rng.normal(size=(n, len(x))) * np.sqrt(np.diag(P))
    if spec.nonneg is not None:
        X[:, spec.nonneg] = np.clip(X[:, spec.nonneg], 0.0, None)

    steps = int(round(H / dt))
    crossed = {s.name: np.zeros(n, bool) for s in spec.safety}
    ctime = {s.name: np.full(n, np.nan) for s in spec.safety}
    for k in range(1, steps + 1):
        X = rk4_batch(X, dt, spec.deriv_batch, u, spec.params, spec.nonneg)
        for s in spec.safety:
            lv = s.value(X.T, spec.params, spec.env)
            hit = (lv > s.limit) if s.direction == ">" else (lv < s.limit)
            newly = (~crossed[s.name]) & hit
            ctime[s.name][newly] = k * dt
            crossed[s.name] |= hit
    out = {}
    for s in spec.safety:
        c = crossed[s.name]
        out[s.name] = {
            "p": float(c.mean()),
            "t_cross": float(np.nanmean(ctime[s.name])) if c.any() else None,
            "level": float(s.value(x[:, None], spec.params, spec.env)[0]),
            "limit": s.limit, "direction": s.direction, "units": s.units,
        }
    return out


# ---- shared simulator: ground truth + noisy, optionally-missing sensors -------

def simulate_truth(
    spec: SystemSpec,
    hours: float,
    dt: float,
    sensor_dt: float,
    u_of_t: Callable[[float], Any],
    hidden_of_t: Callable[[float], float],
    seed: int = 7,
    x0: np.ndarray | None = None,
    available: Callable[[float, str], bool] | None = None,
) -> dict:
    """Integrate the true system (with an imposed hidden trajectory) and sample
    its sensors with noise. `available(t, channel)` may return False to model a
    sensor being offline at time t (data-availability scenarios)."""
    rng = np.random.default_rng(seed)
    hi = spec.idx(spec.hidden)
    x = np.array(spec.x0 if x0 is None else x0, float)
    n_steps = int(round(hours / dt))
    ts, X, us = [], [], []
    t = 0.0
    for _ in range(n_steps + 1):
        x = x.copy()
        x[hi] = hidden_of_t(t)
        ts.append(t)
        X.append(x.copy())
        u = u_of_t(t)
        us.append(u)
        x = rk4(x, dt, spec.deriv, u, spec.params, spec.nonneg)
        t += dt
    ts = np.array(ts)
    X = np.array(X)

    step = max(int(round(sensor_dt / dt)), 1)
    idx = np.arange(0, len(ts), step)
    meas = []
    for i in idx:
        row = {}
        for name, ch in spec.channels.items():
            if available is not None and not available(ts[i], name):
                continue
            row[name] = float(ch.obs(X[i]) + rng.normal(0, ch.noise))
        meas.append(row)
    return {
        "t": ts[idx],
        "truth": X[idx],
        "t_fine": ts,
        "x_fine": X,
        "meas": meas,
        "u": [us[i] for i in idx],
        "spec": spec,
    }
