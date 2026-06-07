"""Gradient checks for the autograd engine against finite differences.

These tests are the core correctness evidence: every differentiation rule is
compared numerically, and second derivatives (needed by the PINN) are checked
too.
"""

from __future__ import annotations

import numpy as np
import pytest

from pinn.core.tensor import Tensor, grad


def fd_grad(f, x: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Central finite-difference gradient of scalar ``f`` at array ``x``."""
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        xp = x.copy()
        xp[idx] += eps
        xm = x.copy()
        xm[idx] -= eps
        g[idx] = (f(xp) - f(xm)) / (2 * eps)
    return g


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_add_mul_pow(rng):
    x = rng.standard_normal((3, 4)).astype(np.float32)

    def f(a):
        t = Tensor(a, requires_grad=True)
        return ((t * t + t * 2.0) ** 2).sum().item()

    t = Tensor(x, requires_grad=True)
    analytic = grad(((t * t + t * 2.0) ** 2).sum(), t).numpy()
    assert np.allclose(analytic, fd_grad(f, x), atol=1e-2)


def test_matmul(rng):
    a = rng.standard_normal((4, 3)).astype(np.float32)
    b = rng.standard_normal((3, 2)).astype(np.float32)
    bt = Tensor(b, requires_grad=True)

    def fa(av):
        return ((Tensor(av) @ Tensor(b)) ** 2).sum().item()

    at = Tensor(a, requires_grad=True)
    ga = grad(((at @ Tensor(b, requires_grad=True)) ** 2).sum(), at).numpy()
    assert np.allclose(ga, fd_grad(fa, a), atol=1e-2)

    def fb(bv):
        return ((Tensor(a) @ Tensor(bv)) ** 2).sum().item()

    gb = grad(((Tensor(a, requires_grad=True) @ bt) ** 2).sum(), bt).numpy()
    assert np.allclose(gb, fd_grad(fb, b), atol=1e-2)


def test_unary_functions(rng):
    x = rng.uniform(0.2, 1.5, size=(2, 5)).astype(np.float32)

    for fn_name in ("sin", "cos", "ln", "sigmoid"):

        def f(a, fn_name=fn_name):
            t = Tensor(a, requires_grad=True)
            return getattr(t, fn_name)().sum().item()

        t = Tensor(x, requires_grad=True)
        analytic = grad(getattr(t, fn_name)().sum(), t).numpy()
        assert np.allclose(analytic, fd_grad(f, x), atol=1e-2), fn_name


def test_broadcasting_bias(rng):
    """Gradient w.r.t. a (1, n) bias added to a (m, n) matrix sums over rows."""
    x = rng.standard_normal((5, 3)).astype(np.float32)
    b = rng.standard_normal((1, 3)).astype(np.float32)
    bt = Tensor(b, requires_grad=True)
    out = (Tensor(x) + bt).sum()
    gb = grad(out, bt).numpy()
    assert gb.shape == (1, 3)
    assert np.allclose(gb, np.full((1, 3), 5.0), atol=1e-4)


def test_column_indexing(rng):
    x = rng.standard_normal((4, 3)).astype(np.float32)
    t = Tensor(x, requires_grad=True)
    g = grad((t[:, 1] ** 2).sum(), t).numpy()
    expected = np.zeros_like(x)
    expected[:, 1] = 2 * x[:, 1]
    assert np.allclose(g, expected, atol=1e-3)


def test_second_order(rng):
    """d2/dx2 sum(sin(x)) = -sin(x)."""
    x = rng.uniform(0.1, 1.4, size=(1, 6)).astype(np.float32)
    t = Tensor(x, requires_grad=True)
    g1 = grad(t.sin().sum(), t, create_graph=True)
    g2 = grad(g1.sum(), t)
    assert np.allclose(g2.numpy(), -np.sin(x), atol=1e-3)


def test_second_order_sigmoid(rng):
    """Second derivative of the sigmoid: s(1-s)(1-2s)."""
    x = rng.uniform(-2, 2, size=(1, 6)).astype(np.float32)
    t = Tensor(x, requires_grad=True)
    g1 = grad(t.sigmoid().sum(), t, create_graph=True)
    g2 = grad(g1.sum(), t).numpy()
    s = 1.0 / (1.0 + np.exp(-x))
    assert np.allclose(g2, s * (1 - s) * (1 - 2 * s), atol=1e-3)
