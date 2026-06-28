"""Unit tests for the Adam optimizer.

The end-to-end PINN test only asserts the loss falls, which leaves Adam's
arithmetic (bias correction, the EMA coefficients, the update rule) unpinned.
These compare one and several steps against a hand-rolled canonical Adam.
"""

from __future__ import annotations

import pytest
import numpy as np

from pinn.nn import Adam
from pinn.core.tensor import Tensor


def _canonical_adam(
    p: np.ndarray,
    g: np.ndarray,
    *,
    lr: float,
    b1: float,
    b2: float,
    eps: float,
    steps: int,
) -> np.ndarray:
    """Reference Adam: the same constant grad applied for ``steps`` steps."""
    m = np.zeros_like(p)
    v = np.zeros_like(p)
    p = p.copy()
    for t in range(1, steps + 1):
        m = b1 * m + (1.0 - b1) * g
        v = b2 * v + (1.0 - b2) * (g * g)
        m_hat = m / (1.0 - b1**t)
        v_hat = v / (1.0 - b2**t)
        p = p - lr * m_hat / (np.sqrt(v_hat) + eps)
    return p


def test_single_step_matches_canonical_adam():
    p0 = np.array([[1.0, -2.0, 0.5]], dtype=np.float32)
    g = np.array([[0.1, 0.3, -0.2]], dtype=np.float32)
    p = Tensor(p0.copy(), requires_grad=True)
    opt = Adam([p], lr=1e-2)
    assert opt.m[0].shape == p0.shape  # moment buffers match the param
    assert opt.v[0].shape == p0.shape

    opt.step([Tensor(g)])
    assert opt.t == 1
    expected = _canonical_adam(
        p0,
        g,
        lr=1e-2,
        b1=0.9,
        b2=0.999,
        eps=1e-8,
        steps=1,
    )
    assert np.allclose(p.numpy(), expected, atol=1e-6)


def test_multi_step_bias_correction_matches_canonical():
    """Three steps with the same grad: pins the ``t``-dependent bias terms."""
    p0 = np.array([[0.5, -1.0]], dtype=np.float32)
    g = np.array([[0.2, -0.4]], dtype=np.float32)
    p = Tensor(p0.copy(), requires_grad=True)
    opt = Adam([p], lr=1e-2)

    for _ in range(3):
        opt.step([Tensor(g)])
    expected = _canonical_adam(
        p0,
        g,
        lr=1e-2,
        b1=0.9,
        b2=0.999,
        eps=1e-8,
        steps=3,
    )
    assert np.allclose(p.numpy(), expected, atol=1e-5)


def test_none_grad_leaves_its_param_untouched():
    """A ``None`` grad skips only that param; later params still update."""
    p0 = Tensor([[3.0]], requires_grad=True)
    p1 = Tensor([[4.0]], requires_grad=True)
    before0 = p0.numpy().copy()
    opt = Adam([p0, p1], lr=1e-2)

    opt.step([None, Tensor([[0.5]])])
    assert np.array_equal(p0.numpy(), before0)  # None grad: unchanged
    assert not np.array_equal(p1.numpy(), [[4.0]])  # real grad: stepped


def test_rejects_non_numpy_backed_param():
    """Adam is CPU/NumPy-only: a param whose ``.data`` is not an ndarray (as a
    CUDA ``DeviceArray`` would be) is rejected up front, not silently
    corrupted into a 0-d object array."""
    p = Tensor([[1.0]], requires_grad=True)
    p.data = object()  # a device descriptor stand-in (no numpy interface)
    with pytest.raises(TypeError, match="CPU/NumPy backend only"):
        Adam([p])
