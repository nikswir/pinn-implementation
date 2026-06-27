"""Hand-written CUDA kernels (Numba) over one flat float32 device buffer.

Every tensor is a contiguous slice of a single device buffer (the pool),
identified by a start offset plus its ``(rows, cols)``. Kernels address that
buffer directly with ``BUF[start + i * cols + j]`` — the same flat,
pointer-plus-offset model used in CUDA C++ (a raw ``float*``) and in PyTorch's
storage/offset/stride tensors. There is no 2-D array type; the 2-D shape is just
index arithmetic over linear memory.

This module is imported only when the CUDA backend is activated, so importing
``pinn`` never requires a GPU. The kernels mirror the NumPy CPU backend, which
is how the ``cuda``-marked tests validate them (output compared against the
reference backend).

Highlights:

- :func:`matmul_kernel` is a classic shared-memory **tiled** GEMM: each block
  cooperatively stages ``TPB x TPB`` tiles of A and B into shared memory and
  accumulates partial products, reusing each loaded value ``TPB`` times.
- Elementwise binary kernels support NumPy-style broadcasting by clamping the
  read index to 0 along any axis whose length is 1.
- Reductions use a shared-memory tree reduction in a single block.
"""

from __future__ import annotations

import math

from numba import cuda, float32

TPB = 16  # threads per block (one tile dimension)
RED_TPB = 32  # threads for a reduction block


########################################
#            matmul (tiled)            #
########################################


@cuda.jit
def matmul_kernel(
    BUF,
    a_start,
    a_rows,
    a_cols,
    b_start,
    b_rows,
    b_cols,
    c_start,
):
    # C = A @ B, with C of shape (a_rows, b_cols) and contraction dim a_cols.
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    row, col = cuda.grid(2)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    acc = float32(0.0)
    n_tiles = (a_cols + TPB - 1) // TPB
    for tile in range(n_tiles):
        a_col = tile * TPB + ty
        b_row = tile * TPB + tx
        if row < a_rows and a_col < a_cols:
            sA[tx, ty] = BUF[a_start + row * a_cols + a_col]
        else:
            sA[tx, ty] = float32(0.0)
        if b_row < b_rows and col < b_cols:
            sB[tx, ty] = BUF[b_start + b_row * b_cols + col]
        else:
            sB[tx, ty] = float32(0.0)
        cuda.syncthreads()
        for p in range(TPB):
            acc += sA[tx, p] * sB[p, ty]
        cuda.syncthreads()

    if row < a_rows and col < b_cols:
        BUF[c_start + row * b_cols + col] = acc


@cuda.jit
def transpose_kernel(BUF, a_start, a_rows, a_cols, c_start):
    i, j = cuda.grid(2)
    if i < a_rows and j < a_cols:
        BUF[c_start + j * a_rows + i] = BUF[a_start + i * a_cols + j]


@cuda.jit
def fill_kernel(BUF, c_start, c_rows, c_cols, value):
    i, j = cuda.grid(2)
    if i < c_rows and j < c_cols:
        BUF[c_start + i * c_cols + j] = value


########################################
#          elementwise binary          #
########################################


@cuda.jit
def add_kernel(
    BUF,
    a_start,
    a_rows,
    a_cols,
    b_start,
    b_rows,
    b_cols,
    c_start,
    c_rows,
    c_cols,
):
    i, j = cuda.grid(2)
    if i < c_rows and j < c_cols:
        ai = i if a_rows > 1 else 0
        aj = j if a_cols > 1 else 0
        bi = i if b_rows > 1 else 0
        bj = j if b_cols > 1 else 0
        av = BUF[a_start + ai * a_cols + aj]
        bv = BUF[b_start + bi * b_cols + bj]
        BUF[c_start + i * c_cols + j] = av + bv


@cuda.jit
def mul_kernel(
    BUF,
    a_start,
    a_rows,
    a_cols,
    b_start,
    b_rows,
    b_cols,
    c_start,
    c_rows,
    c_cols,
):
    i, j = cuda.grid(2)
    if i < c_rows and j < c_cols:
        ai = i if a_rows > 1 else 0
        aj = j if a_cols > 1 else 0
        bi = i if b_rows > 1 else 0
        bj = j if b_cols > 1 else 0
        av = BUF[a_start + ai * a_cols + aj]
        bv = BUF[b_start + bi * b_cols + bj]
        BUF[c_start + i * c_cols + j] = av * bv


@cuda.jit
def div_kernel(
    BUF,
    a_start,
    a_rows,
    a_cols,
    b_start,
    b_rows,
    b_cols,
    c_start,
    c_rows,
    c_cols,
):
    i, j = cuda.grid(2)
    if i < c_rows and j < c_cols:
        ai = i if a_rows > 1 else 0
        aj = j if a_cols > 1 else 0
        bi = i if b_rows > 1 else 0
        bj = j if b_cols > 1 else 0
        av = BUF[a_start + ai * a_cols + aj]
        bv = BUF[b_start + bi * b_cols + bj]
        BUF[c_start + i * c_cols + j] = av / bv


########################################
#          elementwise unary           #
########################################


@cuda.jit
def pow_scalar_kernel(BUF, a_start, a_rows, a_cols, c_start, p):
    i, j = cuda.grid(2)
    if i < a_rows and j < a_cols:
        BUF[c_start + i * a_cols + j] = BUF[a_start + i * a_cols + j] ** p


@cuda.jit
def neg_kernel(BUF, a_start, a_rows, a_cols, c_start):
    i, j = cuda.grid(2)
    if i < a_rows and j < a_cols:
        BUF[c_start + i * a_cols + j] = -BUF[a_start + i * a_cols + j]


@cuda.jit
def sin_kernel(BUF, a_start, a_rows, a_cols, c_start):
    i, j = cuda.grid(2)
    if i < a_rows and j < a_cols:
        BUF[c_start + i * a_cols + j] = math.sin(BUF[a_start + i * a_cols + j])


@cuda.jit
def cos_kernel(BUF, a_start, a_rows, a_cols, c_start):
    i, j = cuda.grid(2)
    if i < a_rows and j < a_cols:
        BUF[c_start + i * a_cols + j] = math.cos(BUF[a_start + i * a_cols + j])


@cuda.jit
def log_kernel(BUF, a_start, a_rows, a_cols, c_start):
    i, j = cuda.grid(2)
    if i < a_rows and j < a_cols:
        BUF[c_start + i * a_cols + j] = math.log(BUF[a_start + i * a_cols + j])


@cuda.jit
def exp_kernel(BUF, a_start, a_rows, a_cols, c_start):
    i, j = cuda.grid(2)
    if i < a_rows and j < a_cols:
        BUF[c_start + i * a_cols + j] = math.exp(BUF[a_start + i * a_cols + j])


@cuda.jit
def sigmoid_kernel(BUF, a_start, a_rows, a_cols, c_start):
    i, j = cuda.grid(2)
    if i < a_rows and j < a_cols:
        BUF[c_start + i * a_cols + j] = 1.0 / (
            1.0 + math.exp(-BUF[a_start + i * a_cols + j])
        )


########################################
#              reductions              #
########################################

# Shared-memory tree reduction in a single block of RED_TPB threads.


@cuda.jit
def sum_all_kernel(BUF, a_start, a_size, c_start):
    tx = cuda.threadIdx.x
    shared = cuda.shared.array(shape=RED_TPB, dtype=float32)

    acc = float32(0.0)
    idx = tx
    while idx < a_size:
        acc += BUF[a_start + idx]
        idx += RED_TPB
    shared[tx] = acc

    depth = RED_TPB // 2
    while depth > 0:
        cuda.syncthreads()
        if tx < depth:
            shared[tx] += shared[tx + depth]
        depth //= 2
    if tx == 0:
        BUF[c_start] = shared[0]


@cuda.jit
def sum_axis0_kernel(BUF, a_start, a_rows, a_cols, c_start):
    """Column sums -> (1, a_cols): reduce over rows for each column."""
    tx = cuda.threadIdx.x
    shared = cuda.shared.array(shape=RED_TPB, dtype=float32)

    for col in range(a_cols):
        acc = float32(0.0)
        i = tx
        while i < a_rows:
            acc += BUF[a_start + i * a_cols + col]
            i += RED_TPB
        shared[tx] = acc

        depth = RED_TPB // 2
        while depth > 0:
            cuda.syncthreads()
            if tx < depth:
                shared[tx] += shared[tx + depth]
            depth //= 2
        if tx == 0:
            BUF[c_start + col] = shared[0]
        cuda.syncthreads()


@cuda.jit
def sum_axis1_kernel(BUF, a_start, a_rows, a_cols, c_start):
    """Row sums -> (a_rows, 1): reduce over columns for each row."""
    tx = cuda.threadIdx.x
    shared = cuda.shared.array(shape=RED_TPB, dtype=float32)

    for row in range(a_rows):
        acc = float32(0.0)
        j = tx
        while j < a_cols:
            acc += BUF[a_start + row * a_cols + j]
            j += RED_TPB
        shared[tx] = acc

        depth = RED_TPB // 2
        while depth > 0:
            cuda.syncthreads()
            if tx < depth:
                shared[tx] += shared[tx + depth]
            depth //= 2
        if tx == 0:
            BUF[c_start + row] = shared[0]
        cuda.syncthreads()
