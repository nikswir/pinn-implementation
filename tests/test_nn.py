"""Tests for the neural-network layers (Linear, MLP)."""

from __future__ import annotations

import numpy as np

from pinn.nn import MLP, Linear
from pinn.core.tensor import Tensor


def test_linear_applies_weight_and_bias():
    """``layer(x) == x @ W + b`` — with a nonzero bias, so its sign is pinned
    (the default zero bias hides a +/- swap)."""
    rng = np.random.default_rng(0)
    layer = Linear(3, 2, rng=rng)
    layer.bias = Tensor(np.array([[1.0, -2.0]]), requires_grad=True)
    x = rng.standard_normal((4, 3)).astype(np.float32)

    out = layer(Tensor(x)).numpy()
    expected = x @ layer.weight.numpy() + np.array([[1.0, -2.0]])
    assert np.allclose(out, expected, atol=1e-4)


def test_mlp_is_reproducible_from_seed():
    """The rng is threaded into every layer, so a fixed seed reproduces every
    parameter (a per-layer ``None`` rng would randomize them)."""
    m1 = MLP(3, 1, rng=np.random.default_rng(42))
    m2 = MLP(3, 1, rng=np.random.default_rng(42))

    p1 = m1.parameters()
    p2 = m2.parameters()
    assert len(p1) == len(p2)
    for a, b in zip(p1, p2, strict=True):
        assert np.array_equal(a.numpy(), b.numpy())


def test_mlp_maps_input_to_output_width():
    rng = np.random.default_rng(1)
    model = MLP(3, 1, hidden=8, rng=rng)
    x = rng.standard_normal((5, 3)).astype(np.float32)
    assert model(Tensor(x)).shape == (5, 1)
