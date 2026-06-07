"""The analytic solution must satisfy the PDE, boundary and initial conditions.

Derivatives are taken numerically (finite differences) on the closed-form
series, independently of the autograd engine, so this validates the reference
solution itself.
"""

from __future__ import annotations

import math

import numpy as np

from pinn.pde.analytic import u_exact

PI_2 = math.pi / 2.0
N_TERMS = 12


def _u(x, y, t):
    return u_exact(x, y, t, n_terms=N_TERMS)


def test_initial_condition_is_zero():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 20)
    y = rng.uniform(0, PI_2, 20)
    assert np.allclose(_u(x, y, 0.0), 0.0, atol=1e-6)


def test_dirichlet_y0_is_zero():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, 20)
    t = rng.uniform(0, 1, 20)
    assert np.allclose(_u(x, 0.0, t), 0.0, atol=1e-6)


def test_neumann_conditions():
    h = 1e-4
    rng = np.random.default_rng(2)
    y = rng.uniform(0.2, PI_2 - 0.2, 10)
    t = rng.uniform(0.2, 1.0, 10)
    # u_x = 0 at x=0 and x=1
    ux0 = (_u(0 + h, y, t) - _u(0 - h, y, t)) / (2 * h)
    ux1 = (_u(1 + h, y, t) - _u(1 - h, y, t)) / (2 * h)
    assert np.abs(ux0).max() < 1e-2
    assert np.abs(ux1).max() < 1e-2
    # u_y = 0 at y = pi/2
    x = rng.uniform(0.1, 0.9, 10)
    uy = (_u(x, PI_2 + h, t) - _u(x, PI_2 - h, t)) / (2 * h)
    assert np.abs(uy).max() < 1e-2


def test_pde_residual_is_small():
    """u_t - u_xx - u_yy - x t^2 sin(y) ~ 0 in the interior."""
    h = 1e-3
    rng = np.random.default_rng(3)
    x = rng.uniform(0.2, 0.8, 30)
    y = rng.uniform(0.3, PI_2 - 0.3, 30)
    t = rng.uniform(0.3, 0.9, 30)

    u_t = (_u(x, y, t + h) - _u(x, y, t - h)) / (2 * h)
    u_xx = (_u(x + h, y, t) - 2 * _u(x, y, t) + _u(x - h, y, t)) / h**2
    u_yy = (_u(x, y + h, t) - 2 * _u(x, y, t) + _u(x, y - h, t)) / h**2
    source = x * t**2 * np.sin(y)

    residual = u_t - u_xx - u_yy - source
    assert np.abs(residual).max() < 1e-2
