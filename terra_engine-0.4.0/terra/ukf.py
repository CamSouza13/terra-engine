"""Compact Unscented Kalman Filter (Julier & Uhlmann / Wan & van der Merwe).

Self-contained, NumPy-only. Handles the nonlinear RAS dynamics without
requiring Jacobians. Exposes the innovation and normalized innovation squared
(NIS) so the engine can tell when the model has stopped explaining the data —
the "the books no longer balance" signal.
"""
from __future__ import annotations

from typing import Callable
import numpy as np


class UnscentedKalmanFilter:
    def __init__(
        self,
        dim_x: int,
        fx: Callable[[np.ndarray, float], np.ndarray],
        Q: np.ndarray,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float | None = None,
    ) -> None:
        self.dim_x = dim_x
        self.fx = fx
        self.Q = np.asarray(Q, float)

        self.x = np.zeros(dim_x)
        self.P = np.eye(dim_x)

        if kappa is None:
            kappa = 3.0 - dim_x
        self.lambda_ = alpha ** 2 * (dim_x + kappa) - dim_x
        self.alpha, self.beta, self.kappa = alpha, beta, kappa

        n = dim_x
        c = n + self.lambda_
        self.Wm = np.full(2 * n + 1, 1.0 / (2 * c))
        self.Wc = np.full(2 * n + 1, 1.0 / (2 * c))
        self.Wm[0] = self.lambda_ / c
        self.Wc[0] = self.lambda_ / c + (1 - alpha ** 2 + beta)
        self._c = c

        # diagnostics from the most recent update
        self.innovation = np.zeros(0)
        self.nis = 0.0

    def _sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        n = self.dim_x
        P = 0.5 * (P + P.T) + 1e-12 * np.eye(n)  # keep symmetric / PD
        try:
            # lower-triangular L with L @ L.T = c*P; the valid square-root
            # vectors are the COLUMNS of L (sum_i col_i col_i^T = L L^T = c*P).
            L = np.linalg.cholesky(self._c * P)
        except np.linalg.LinAlgError:
            # symmetric square root fallback if not PD (rows == columns here)
            w, V = np.linalg.eigh(self._c * P)
            w = np.clip(w, 1e-12, None)
            L = (V * np.sqrt(w)) @ V.T
        pts = np.zeros((2 * n + 1, n))
        pts[0] = x
        for i in range(n):
            col = L[:, i]
            pts[i + 1] = x + col
            pts[n + i + 1] = x - col
        return pts

    def predict(self, dt: float) -> None:
        pts = self._sigma_points(self.x, self.P)
        self.sigmas_f = np.array([self.fx(p, dt) for p in pts])
        self.x = self.Wm @ self.sigmas_f
        y = self.sigmas_f - self.x
        self.P = (self.Wc[:, None, None] * np.einsum("ki,kj->kij", y, y)).sum(0) + self.Q

    def update(
        self,
        z: np.ndarray,
        hx: Callable[[np.ndarray], np.ndarray],
        R: np.ndarray,
    ) -> None:
        """Measurement update against whatever channels are available this step.

        `hx` maps a state to the currently-available measurement vector and `R`
        is its noise covariance. Passing a subset lets the same engine ride on
        whatever sensors exist right now — drop a probe and the filter simply
        leans harder on the model and the remaining channels.
        """
        z = np.asarray(z, float)
        R = np.asarray(R, float)
        if z.size == 0:  # no data this step: prediction-only
            self.innovation = np.zeros(0)
            self.nis = 0.0
            return
        sigmas_h = np.array([hx(p) for p in self.sigmas_f])
        z_pred = self.Wm @ sigmas_h
        dz = sigmas_h - z_pred
        S = (self.Wc[:, None, None] * np.einsum("ki,kj->kij", dz, dz)).sum(0) + R
        dx = self.sigmas_f - self.x
        Pxz = (self.Wc[:, None, None] * np.einsum("ki,kj->kij", dx, dz)).sum(0)

        Sinv = np.linalg.inv(S)
        K = Pxz @ Sinv
        innovation = z - z_pred
        self.x = self.x + K @ innovation
        self.P = self.P - K @ S @ K.T

        self.innovation = innovation
        self.nis = float(innovation @ Sinv @ innovation)  # ~chi-square(len z) if consistent
