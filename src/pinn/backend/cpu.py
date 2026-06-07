"""CPU compute backend (pure NumPy).

This is the reference backend: every primitive the autograd ``Tensor`` needs,
implemented with plain NumPy on 2-D ``float32`` arrays. It always runs (no GPU
required), so the test suite and CI exercise the exact same maths the CUDA
backend implements with hand-written kernels.

All arrays are kept 2-D and ``float32`` to mirror the on-device representation.
"""

from __future__ import annotations

import numpy as np

DTYPE = np.float32
name = "cpu"


# --- array creation ---------------------------------------------------------


def asarray(value) -> np.ndarray:
    """Coerce a Python scalar / 1-D / 2-D value into a 2-D float32 array."""
    if isinstance(value, (int | float)):
        return np.array([[value]], dtype=DTYPE)
    arr = np.asarray(value, dtype=DTYPE)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        return arr
    raise ValueError(f"expected <=2 dims, got shape {arr.shape}")


def full(shape, value) -> np.ndarray:
    return np.full(shape, value, dtype=DTYPE)


def zeros(shape) -> np.ndarray:
    return np.zeros(shape, dtype=DTYPE)


def to_numpy(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)


# --- elementwise (NumPy broadcasting) ---------------------------------------


def add(a, b):
    return (a + b).astype(DTYPE)


def mul(a, b):
    return (a * b).astype(DTYPE)


def div(a, b):
    return (a / b).astype(DTYPE)


def power(a, b):
    return np.power(a, b).astype(DTYPE)


def neg(a):
    return (-a).astype(DTYPE)


def sin(a):
    return np.sin(a).astype(DTYPE)


def cos(a):
    return np.cos(a).astype(DTYPE)


def log(a):
    return np.log(a).astype(DTYPE)


def exp(a):
    return np.exp(a).astype(DTYPE)


def sigmoid(a):
    return (1.0 / (1.0 + np.exp(-a))).astype(DTYPE)


# --- linear algebra / reductions --------------------------------------------


def matmul(a, b):
    return (a @ b).astype(DTYPE)


def transpose(a):
    return np.ascontiguousarray(a.T)


def sum_all(a):
    return np.array([[a.sum()]], dtype=DTYPE)


def sum_axis(a, axis):
    return a.sum(axis=axis, keepdims=True).astype(DTYPE)
