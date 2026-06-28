"""Tests for the CPU (NumPy) reference backend.

The CUDA parity tests exercise most of these primitives, but only on a GPU.
This pins the reference backend itself on stage-1 CPU runs: coercion shapes,
the elementwise/reduction ops the parity suite reaches only via the GPU path,
and the float64 copy-out.
"""

from __future__ import annotations

import pytest
import numpy as np

from pinn.backend import cpu
from pinn.backend.memory import MemoryManager


def test_pool_is_created_with_backend_capacity():
    """The lazily-built CPU pool is a ``MemoryManager`` of ``POOL_CAPACITY``
    float32 slots — pins the constructor arguments of the live alloc path.

    The global is reset first so the lazy-creation body actually runs (it is
    cached after the first op anywhere), then restored so the GC-accounting
    test keeps a valid pool.
    """
    old = cpu._pool
    cpu._pool = None
    try:
        p = cpu.pool()
        assert isinstance(p, MemoryManager)
        assert p.capacity == cpu.POOL_CAPACITY
        assert p.buffer.dtype == np.float32
    finally:
        cpu._pool = old


def test_coerce_scalar_becomes_1x1():
    out = cpu.coerce(5.0)
    assert out.shape == (1, 1)
    assert out.dtype == np.float32
    assert out[0, 0] == 5.0


def test_coerce_zero_dim_array_becomes_1x1():
    assert cpu.coerce(np.float32(3.0)).shape == (1, 1)


def test_coerce_1d_becomes_row():
    assert cpu.coerce([1.0, 2.0, 3.0]).shape == (1, 3)


def test_coerce_2d_passes_through():
    assert cpu.coerce(np.ones((2, 4), dtype=np.float32)).shape == (2, 4)


def test_coerce_rejects_more_than_2d():
    with pytest.raises(ValueError, match="expected <=2 dims"):
        cpu.coerce(np.ones((2, 2, 2)))


def test_zeros_is_all_zero():
    out = cpu.zeros((2, 3))
    assert out.shape == (2, 3)
    assert np.array_equal(out, np.zeros((2, 3), dtype=np.float32))


def test_div_neg_exp_match_numpy():
    rng = np.random.default_rng(0)
    a = rng.uniform(0.5, 2.0, size=(3, 4)).astype(np.float32)
    b = rng.uniform(0.5, 2.0, size=(3, 4)).astype(np.float32)
    assert np.allclose(cpu.div(a, b), a / b, atol=1e-6)
    assert np.allclose(cpu.neg(a), -a, atol=1e-6)
    assert np.allclose(cpu.exp(a), np.exp(a), atol=1e-5)


def test_sum_axis_matches_numpy_on_both_axes():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((4, 5)).astype(np.float32)

    s0 = cpu.sum_axis(a, 0)
    assert s0.shape == (1, 5)
    assert np.allclose(s0, a.sum(axis=0, keepdims=True), atol=1e-4)

    s1 = cpu.sum_axis(a, 1)
    assert s1.shape == (4, 1)
    assert np.allclose(s1, a.sum(axis=1, keepdims=True), atol=1e-4)


def test_sum_axis_rejects_unsupported_axis():
    a = np.ones((2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="unsupported sum axis"):
        cpu.sum_axis(a, 2)


def test_to_numpy_copies_out_as_float64():
    out = cpu.to_numpy(cpu.asarray([[1.0, 2.0]]))
    assert out.dtype == np.float64
    assert np.allclose(out, [[1.0, 2.0]])
