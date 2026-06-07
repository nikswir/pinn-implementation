"""CPU compute backend (NumPy) backed by the pooled allocator.

This is the reference backend: every primitive the autograd ``Tensor`` needs,
implemented with plain NumPy on 2-D ``float32`` arrays. It always runs (no GPU
required), so the test suite and CI exercise the exact same maths the CUDA
backend implements with hand-written kernels.

Every array an operation produces is a *slice of one big buffer* handed out by
:class:`~pinn.backend.memory.MemoryManager`, rather than a fresh NumPy
allocation. The slice is returned to the pool automatically when the array is
garbage-collected (via :func:`weakref.finalize`). This mirrors how real GPU
runtimes (e.g. PyTorch's caching allocator) avoid paying an allocation cost per
tensor; here it also means the allocator is the live allocation path, not a
showcase.

All arrays are kept 2-D and ``float32`` to mirror the on-device representation.
"""

from __future__ import annotations

import weakref

import numpy as np

from pinn.backend.memory import MemoryManager

DTYPE = np.float32
name = "cpu"

# Pool capacity in float32 elements (64 MB). One loss+grad step on the default
# problem peaks well under 3M elements, so this leaves generous headroom.
POOL_CAPACITY = 1 << 24

_pool: MemoryManager | None = None


def pool() -> MemoryManager:
    """The lazily-created allocation pool for this backend."""
    global _pool
    if _pool is None:
        _pool = MemoryManager(POOL_CAPACITY)
    return _pool


def empty(shape) -> np.ndarray:
    """Allocate an uninitialized 2-D array as a view into the pool buffer."""
    rows, cols = shape
    size = rows * cols
    p = pool()
    offset = p.allocate(size)
    view = p.buffer[offset : offset + size].reshape(rows, cols)
    weakref.finalize(view, p.free, offset, size)
    return view


# --- array creation ---------------------------------------------------------

def asarray(value) -> np.ndarray:
    """Coerce a scalar / 1-D / 2-D value into a pooled 2-D float32 array."""
    if isinstance(value, (int | float)):
        arr = np.array([[value]], dtype=DTYPE)
    else:
        arr = np.asarray(value, dtype=DTYPE)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim != 2:
            raise ValueError(f"expected <=2 dims, got shape {arr.shape}")
    out = empty(arr.shape)
    out[...] = arr
    return out


def full(shape, value) -> np.ndarray:
    out = empty(shape)
    out[...] = value
    return out


def zeros(shape) -> np.ndarray:
    return full(shape, 0.0)


def to_numpy(a: np.ndarray) -> np.ndarray:
    return a.astype(np.float64)  # copy out of the pool


# --- elementwise (NumPy broadcasting) ---------------------------------------

def add(a, b):
    out = empty(np.broadcast_shapes(a.shape, b.shape))
    np.add(a, b, out=out)
    return out


def mul(a, b):
    out = empty(np.broadcast_shapes(a.shape, b.shape))
    np.multiply(a, b, out=out)
    return out


def div(a, b):
    out = empty(np.broadcast_shapes(a.shape, b.shape))
    np.divide(a, b, out=out)
    return out


def power(a, b):
    out = empty(a.shape)
    np.power(a, DTYPE(b), out=out)
    return out


def neg(a):
    out = empty(a.shape)
    np.negative(a, out=out)
    return out


def sin(a):
    out = empty(a.shape)
    np.sin(a, out=out)
    return out


def cos(a):
    out = empty(a.shape)
    np.cos(a, out=out)
    return out


def log(a):
    out = empty(a.shape)
    np.log(a, out=out)
    return out


def exp(a):
    out = empty(a.shape)
    np.exp(a, out=out)
    return out


def sigmoid(a):
    out = empty(a.shape)
    np.negative(a, out=out)
    np.exp(out, out=out)
    np.add(out, DTYPE(1.0), out=out)
    np.divide(DTYPE(1.0), out, out=out)
    return out


# --- linear algebra / reductions --------------------------------------------

def matmul(a, b):
    out = empty((a.shape[0], b.shape[1]))
    np.matmul(a, b, out=out)
    return out


def transpose(a):
    out = empty((a.shape[1], a.shape[0]))
    out[...] = a.T
    return out


def sum_all(a):
    out = empty((1, 1))
    out[0, 0] = a.sum()
    return out


def sum_axis(a, axis):
    if axis == 0:
        out = empty((1, a.shape[1]))
    elif axis == 1:
        out = empty((a.shape[0], 1))
    else:
        raise ValueError(f"unsupported sum axis {axis!r}")
    np.sum(a, axis=axis, keepdims=True, out=out)
    return out
