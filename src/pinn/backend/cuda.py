"""CUDA compute backend (Numba kernels) over one flat device buffer.

A single device buffer (the pool) holds every tensor as a contiguous slice. An
array is just a lightweight descriptor :class:`DeviceArray` = ``(offset, rows,
cols)`` into that buffer; there is no per-tensor device allocation and no 2-D
device-array type. Each operation launches a hand-written kernel from
:mod:`pinn.backend.kernels` that addresses the buffer flatly with
``BUF[start + i * cols + j]`` — the same model as a CUDA C++ ``float*`` or a
PyTorch storage/offset/stride tensor.

The descriptor's slice is returned to the pool when the descriptor is
garbage-collected (``weakref.finalize``). The autograd ``Tensor`` is unchanged:
it only reads ``.shape`` and passes ``data`` back to these primitives.

Imported lazily (only when ``backend.use("cuda")`` is called), so a GPU is never
required just to import the package.
"""

from __future__ import annotations

import weakref

import numpy as np
from numba import cuda, float32

from pinn.backend import cpu, kernels
from pinn.backend.memory import MemoryManager

DTYPE = np.float32
TPB = (16, 16)
name = "cuda"

# Pool capacity in float32 elements (256 MB on the device).
POOL_CAPACITY = 1 << 26

_pool: MemoryManager | None = None


class DeviceArray:
    """A view into the pool buffer: a start offset plus its 2-D shape."""

    __slots__ = ("offset", "rows", "cols")

    def __init__(self, offset: int, rows: int, cols: int):
        self.offset = offset
        self.rows = rows
        self.cols = cols

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows, self.cols)

    @property
    def size(self) -> int:
        return self.rows * self.cols


def pool() -> MemoryManager:
    """The lazily-created device allocation pool for this backend."""
    global _pool
    if _pool is None:
        buffer = cuda.device_array(POOL_CAPACITY, dtype=DTYPE)
        _pool = MemoryManager(POOL_CAPACITY, buffer=buffer)
    return _pool


########################################
#         launch configuration         #
########################################


def _grid(rows: int, cols: int):
    bx = (rows + TPB[0] - 1) // TPB[0]
    by = (cols + TPB[1] - 1) // TPB[1]
    return (bx, by), TPB


def empty(shape) -> DeviceArray:
    """Reserve an uninitialized slice of the pool and describe it."""
    rows, cols = shape
    size = rows * cols
    p = pool()
    offset = p.allocate(size)
    arr = DeviceArray(offset, rows, cols)
    weakref.finalize(arr, p.free, offset, size)
    return arr


########################################
#            array creation            #
########################################


def asarray(value) -> DeviceArray:
    if isinstance(value, DeviceArray):
        return value
    host = cpu.coerce(value)
    out = empty(host.shape)
    flat = np.ascontiguousarray(host).reshape(-1)
    pool().buffer[out.offset : out.offset + out.size].copy_to_device(flat)
    return out


def full(shape, value) -> DeviceArray:
    out = empty(shape)
    grid, block = _grid(out.rows, out.cols)
    kernels.fill_kernel[grid, block](
        pool().buffer,
        out.offset,
        out.rows,
        out.cols,
        float32(value),
    )
    return out


def zeros(shape) -> DeviceArray:
    return full(shape, 0.0)


def to_numpy(a: DeviceArray) -> np.ndarray:
    flat = pool().buffer[a.offset : a.offset + a.size].copy_to_host()
    return flat.reshape(a.rows, a.cols).astype(np.float64)


########################################
#               helpers                #
########################################


def _unary(kernel, a: DeviceArray) -> DeviceArray:
    out = empty(a.shape)
    grid, block = _grid(a.rows, a.cols)
    kernel[grid, block](pool().buffer, a.offset, a.rows, a.cols, out.offset)
    return out


def _binary(kernel, a: DeviceArray, b: DeviceArray) -> DeviceArray:
    # Reject shapes NumPy would not broadcast, so the CUDA backend raises on
    # the same inputs the CPU backend does instead of silently reading garbage.
    np.broadcast_shapes(a.shape, b.shape)
    rows = max(a.rows, b.rows)
    cols = max(a.cols, b.cols)
    out = empty((rows, cols))
    grid, block = _grid(rows, cols)
    kernel[grid, block](
        pool().buffer,
        a.offset,
        a.rows,
        a.cols,
        b.offset,
        b.rows,
        b.cols,
        out.offset,
        rows,
        cols,
    )
    return out


########################################
#             elementwise              #
########################################


def add(a, b):
    return _binary(kernels.add_kernel, a, b)


def mul(a, b):
    return _binary(kernels.mul_kernel, a, b)


def div(a, b):
    return _binary(kernels.div_kernel, a, b)


def power(a, b):
    out = empty(a.shape)
    grid, block = _grid(a.rows, a.cols)
    kernels.pow_scalar_kernel[grid, block](
        pool().buffer,
        a.offset,
        a.rows,
        a.cols,
        out.offset,
        DTYPE(b),
    )
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


########################################
#     linear algebra / reductions      #
########################################


def matmul(a, b):
    if a.cols != b.rows:
        raise ValueError(
            f"matmul shape mismatch: {a.shape} @ {b.shape}",
        )
    out = empty((a.rows, b.cols))
    grid, block = _grid(a.rows, b.cols)
    kernels.matmul_kernel[grid, block](
        pool().buffer,
        a.offset,
        a.rows,
        a.cols,
        b.offset,
        b.rows,
        b.cols,
        out.offset,
    )
    return out


def transpose(a):
    out = empty((a.cols, a.rows))
    grid, block = _grid(a.rows, a.cols)
    kernels.transpose_kernel[grid, block](
        pool().buffer,
        a.offset,
        a.rows,
        a.cols,
        out.offset,
    )
    return out


def sum_all(a):
    out = empty((1, 1))
    kernels.sum_all_kernel[1, kernels.RED_TPB](
        pool().buffer,
        a.offset,
        a.size,
        out.offset,
    )
    return out


def sum_axis(a, axis):
    if axis == 0:
        out = empty((1, a.cols))
        kernel = kernels.sum_axis0_kernel
    elif axis == 1:
        out = empty((a.rows, 1))
        kernel = kernels.sum_axis1_kernel
    else:
        raise ValueError(f"unsupported sum axis {axis!r}")
    kernel[1, kernels.RED_TPB](
        pool().buffer,
        a.offset,
        a.rows,
        a.cols,
        out.offset,
    )
    return out
