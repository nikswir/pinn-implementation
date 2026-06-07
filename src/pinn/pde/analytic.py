r"""Closed-form reference solution of the heat problem.

The PINN is validated against the analytic solution of

    u_t = u_xx + u_yy + x t^2 sin(y),   0<x<1, 0<y<pi/2, t>0
    u_x(0,y,t) = u_x(1,y,t) = 0           (Neumann in x)
    u(x,0,t)   = 0,  u_y(x,pi/2,t) = 0    (Dirichlet / Neumann in y)
    u(x,y,0)   = 0                        (initial condition)

Separating variables as ``u = sum_n c_n(t) cos(n*pi*x) sin(y)`` (the cosines
satisfy the x-Neumann conditions; ``sin y`` satisfies the y-conditions) and
expanding the source's ``x`` factor in the cosine basis gives, with
``lambda_n = (n*pi)^2 + 1``:

    n = 0 (odd contributions vanish for even n):
        c_0(t) = t^2/2 - t + 1 - e^{-t}
    odd n:
        c_n(t) = (-4 / ((n*pi)^2 * lambda_n^3))
                 * (lambda_n^2 t^2 - 2 lambda_n t + 2 - 2 e^{-lambda_n t})

This module evaluates that series.
"""

from __future__ import annotations

import math

import numpy as np


def _series_term(x: float, y: float, t: float, i: int) -> float:
    n = 2 * i + 1  # odd harmonics only
    lam = (math.pi * n) ** 2 + 1.0
    coeff = -4.0 / (((math.pi * n) ** 2) * (lam**3))
    time_part = (t * lam) ** 2 - 2.0 * t * lam + 2.0 * (1.0 - 1.0 / math.exp(lam * t))
    return coeff * time_part * math.sin(y) * math.cos(math.pi * n * x)


def u_exact_scalar(x: float, y: float, t: float, n_terms: int = 4) -> float:
    """Analytic solution at a single point, summing ``n_terms`` odd harmonics."""
    result = (-math.exp(-t) + (t**2) / 2.0 - t + 1.0) * math.sin(y)
    for i in range(n_terms):
        result += _series_term(x, y, t, i)
    return result


def u_exact(x, y, t, n_terms: int = 4) -> np.ndarray:
    """Vectorized analytic solution over arrays of points (broadcast together)."""
    x, y, t = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float),
                                  np.asarray(t, float))
    out = (-np.exp(-t) + t**2 / 2.0 - t + 1.0) * np.sin(y)
    for i in range(n_terms):
        n = 2 * i + 1
        lam = (math.pi * n) ** 2 + 1.0
        coeff = -4.0 / (((math.pi * n) ** 2) * (lam**3))
        time_part = (t * lam) ** 2 - 2.0 * t * lam + 2.0 * (1.0 - np.exp(-lam * t))
        out = out + coeff * time_part * np.sin(y) * np.cos(math.pi * n * x)
    return out
