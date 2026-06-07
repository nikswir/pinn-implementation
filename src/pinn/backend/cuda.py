"""CUDA compute backend (Numba kernels) backed by the pooled allocator.

Implements the same primitive surface as :mod:`pinn.backend.cpu`, but every
operation launches a hand-written kernel from :mod:`pinn.backend.kernels` and
returns a 2-D float32 *device* array. Like the CPU backend, each result is a
slice of one big device buffer handed out by
:class:`~pinn.backend.memory.MemoryManager` and returned to the pool when the
array is garbage-collected, so a per-tensor device allocation is never paid in
the training loop. The autograd ``Tensor`` is unchanged.

Imported lazily (only when ``backend.use("cuda")`` is called), so a GPU is never
required just to import the package.
"""

from __future__ import annotations

import weakref

import numpy as np
from numba import cuda, float32

from . import cpu, kernels
from .memory import MemoryManager

DTYPE = np.float32
TPB = (16, 16)
name = "cuda"

# Pool capacity in float32 elements (256 MB on the device).
POOL_CAPACITY = 1 << 26

_pool: MemoryManager | None = None


def pool() -> MemoryManager:
    """The lazily-created device allocation pool for this backend."""
    global _pool
    if _pool is None:
        buffer = cuda.device_array(POOL_CAPACITY, dtype=DTYPE)
        _pool = MemoryManager(POOL_CAPACITY, buffer=buffer)
    return _pool


# --- launch configuration ---------------------------------------------------

def _grid(rows: int, cols: int):
    bx = (rows + TPB[0] - 1) // TPB[0]
    by = (cols + TPB[1] - 1) // TPB[1]
    return (bx, by), TPB


def empty(shape):
    """Allocate an uninitialized 2-D device array as a view into the pool."""
    rows, cols = shape
    size = rows * cols
    p = pool()
    offset = p.allocate(size)
    view = p.buffer[offset : offset + size].reshape(rows, cols)
    weakref.finalize(view, p.free, offset, size)
    return view


# --- array creation ---------------------------------------------------------

def asarray(value):
    if cuda.is_cuda_array(value):
        return value
    host = cpu.coerce(value)
    out = empty(host.shape)
    out.copy_to_device(np.ascontiguousarray(host))
    return out


def full(shape, value):
    out = empty(shape)
    grid, block = _grid(shape[0], shape[1])
    kernels.fill_kernel[grid, block](out, float32(value))
    return out


def zeros(shape):
    return full(shape, 0.0)


def to_numpy(a) -> np.ndarray:
    host = a.copy_to_host() if cuda.is_cuda_array(a) else np.asarray(a)
    return host.astype(np.float64)


# --- helpers ----------------------------------------------------------------

def _broadcast_shape(a, b):
    return (max(a.shape[0], b.shape[0]), max(a.shape[1], b.shape[1]))


def _unary(kernel, a):
    out = empty(a.shape)
    grid, block = _grid(a.shape[0], a.shape[1])
    kernel[grid, block](out, a)
    return out


def _binary(kernel, a, b):
    shape = _broadcast_shape(a, b)
    out = empty(shape)
    grid, block = _grid(shape[0], shape[1])
    kernel[grid, block](out, a, b)
    return out


# --- elementwise ------------------------------------------------------------

def add(a, b):
    return _binary(kernels.add_kernel, a, b)


def mul(a, b):
    return _binary(kernels.mul_kernel, a, b)


def div(a, b):
    return _binary(kernels.div_kernel, a, b)


def power(a, b):
    out = empty(a.shape)
    grid, block = _grid(a.shape[0], a.shape[1])
    kernels.pow_scalar_kernel[grid, block](out, a, DTYPE(b))
    return out


def neg(a):
    return _unary(kernels.neg_kernel, a)


def sin(a):
    return _unary(kernels.sin_kernel, a)


def cos(a):
    return _unary(kernels.cos_kernel, a)


def log(a):
    return _unary(kernels.log_kernel, a)


def exp(a):
    return _unary(kernels.exp_kernel, a)


def sigmoid(a):
    return _unary(kernels.sigmoid_kernel, a)


# --- linear algebra / reductions --------------------------------------------

def matmul(a, b):
    m, n = a.shape[0], b.shape[1]
    out = empty((m, n))
    grid, block = _grid(m, n)
    kernels.matmul_kernel[grid, block](out, a, b)
    return out


def transpose(a):
    out = empty((a.shape[1], a.shape[0]))
    grid, block = _grid(a.shape[0], a.shape[1])
    kernels.transpose_kernel[grid, block](out, a)
    return out


def sum_all(a):
    out = zeros((1, 1))
    grid, block = _grid(a.shape[0], a.shape[1])
    kernels.sum_all_kernel[grid, block](out, a)
    return out


def sum_axis(a, axis):
    if axis == 0:
        out = zeros((1, a.shape[1]))
        kernel = kernels.sum_axis0_kernel
    elif axis == 1:
        out = zeros((a.shape[0], 1))
        kernel = kernels.sum_axis1_kernel
    else:
        raise ValueError(f"unsupported sum axis {axis!r}")
    grid, block = _grid(a.shape[0], a.shape[1])
    kernel[grid, block](out, a)
    return out
