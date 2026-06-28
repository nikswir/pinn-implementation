"""Tests for the heat-equation PINN loss assembly.

The end-to-end tests only assert the loss shrinks, which leaves the residual
algebra unpinned: which derivative each term uses, the source term, the
residual signs, the per-boundary normalization, and which terms ``total_loss``
sums. These feed a manufactured analytic ``u`` whose every derivative — and so
every loss term — has a known closed form.
"""

from __future__ import annotations

import numpy as np

from pinn.pde.heat import PI_2, HeatPINN


class _Quadratic:
    """``u = a*x**2 + d*x + b*y**2 + c*t`` as differentiable Tensor ops.

    Quadratic in both x and y so no second derivative is constant-zero
    (``u_xx = 2a``, ``u_yy = 2b``), and the linear ``d*x`` term keeps the x=0
    Neumann derivative nonzero (``u_x = 2a x + d`` -> ``d`` at x=0) so a sign
    flip on that loss term is observable. ``u_t = c``.
    """

    def __init__(self, a: float, b: float, c: float, d: float = 0.0) -> None:
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def __call__(self, xyt):  # type: ignore[no-untyped-def]
        x = xyt[:, 0]
        y = xyt[:, 1]
        t = xyt[:, 2]
        return self.a * (x**2) + self.d * x + self.b * (y**2) + self.c * t


def test_pde_loss_matches_closed_form_residual():
    a, b, c = 0.5, 0.3, 1.5
    n = 5
    model = _Quadratic(a, b, c)
    pinn = HeatPINN(model, rng=np.random.default_rng(0), n_pde=n, n_bc=3)
    loss = pinn.pde_loss().item()

    # ── Replay the exact points pde_loss sampled (same seed, same order) ──
    r = np.random.default_rng(0)
    x = r.uniform(size=(n, 1))
    y = r.uniform(size=(n, 1)) * PI_2
    t = r.uniform(size=(n, 1))
    # residual = u_t - u_xx - u_yy - source = (c - 2a - 2b) - x t^2 sin y
    residual = (c - 2.0 * a - 2.0 * b) - x * (t**2) * np.sin(y)
    expected = float((residual**2).mean())
    assert np.isclose(loss, expected, rtol=1e-3, atol=1e-4)


def test_neumann_terms_have_closed_form_values():
    """Each Neumann term is the MSE of the normal derivative on its face; for
    this ``u`` those values are point-independent constants."""
    a, b, c, d = 0.5, 0.3, 1.5, 0.4
    model = _Quadratic(a, b, c, d)

    # u_x = 2a x + d -> d at x=0 (nonzero, so the term's sign is observable)
    p = HeatPINN(model, rng=np.random.default_rng(0), n_bc=4)
    assert np.isclose(p._neumann_loss(x=0.0).item(), d**2, rtol=1e-3, atol=1e-4)
    # u_x = 2a x + d -> 2a + d at x=1
    p = HeatPINN(model, rng=np.random.default_rng(0), n_bc=4)
    got = p._neumann_loss(x=1.0).item()
    assert np.isclose(got, (2.0 * a + d) ** 2, rtol=1e-3, atol=1e-4)
    # u_y = 2b y -> 2b*(pi/2) at y=pi/2
    p = HeatPINN(model, rng=np.random.default_rng(0), n_bc=4)
    got = p._neumann_loss(y=PI_2).item()
    assert np.isclose(got, (2.0 * b * PI_2) ** 2, rtol=1e-3, atol=1e-4)


def test_dirichlet_and_initial_terms_match_closed_form():
    a, b, c, d = 0.5, 0.3, 1.5, 0.4
    n = 4
    model = _Quadratic(a, b, c, d)
    p = HeatPINN(model, rng=np.random.default_rng(2), n_bc=n)
    dirichlet = p._dirichlet_loss(y=0.0).item()
    initial = p.initial_loss().item()

    # ── Replay both terms' samples in the order the methods draw them ──
    r = np.random.default_rng(2)
    t_d = r.uniform(size=(n, 1))  # _dirichlet_loss draws t, then xc
    xc_d = r.uniform(size=(n, 1))
    u_d = a * xc_d**2 + d * xc_d + c * t_d  # y=0 kills the b-term
    assert np.isclose(dirichlet, float((u_d**2).mean()), rtol=1e-3, atol=1e-4)

    x_i = r.uniform(size=(n, 1))  # initial_loss draws x, then y; t=0
    y_i = r.uniform(size=(n, 1)) * PI_2
    u_i = a * x_i**2 + d * x_i + b * y_i**2
    assert np.isclose(initial, float((u_i**2).mean()), rtol=1e-3, atol=1e-4)


def test_total_loss_is_the_sum_of_its_six_terms():
    model = _Quadratic(0.5, 0.3, 1.5, 0.4)
    total = (
        HeatPINN(
            model,
            rng=np.random.default_rng(1),
            n_pde=5,
            n_bc=3,
        )
        .total_loss()
        .item()
    )

    # ── Same seed + same call order -> identical sampled points ──
    p = HeatPINN(model, rng=np.random.default_rng(1), n_pde=5, n_bc=3)
    parts = (
        p.pde_loss()
        + p._neumann_loss(x=0.0)
        + p._neumann_loss(x=1.0)
        + p._neumann_loss(y=PI_2)
        + p._dirichlet_loss(y=0.0)
        + p.initial_loss()
    ).item()
    assert np.isclose(total, parts, rtol=1e-5)
