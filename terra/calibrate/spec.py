"""Description of a calibratable domain model.

A ``CalModel`` bundles everything the NUTS fit needs that is domain-specific:
the differentiable process model, which parameters to infer (with priors), and
how each observed channel maps to the state. The inference machinery in
``nuts.py`` is otherwise domain-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class CalModel:
    name: str
    deriv: Callable                 # (x_jnp, u_jnp, theta: dict) -> dx/dt (jnp)
    fit_params: tuple               # names of parameters NUTS infers
    prior: dict                     # name -> (median, sigma_log) for LogNormal
    # channel -> (state_index, scale, offset): obs = scale * state + offset
    obs_map: dict

    @property
    def channels(self) -> tuple:
        return tuple(self.obs_map.keys())
