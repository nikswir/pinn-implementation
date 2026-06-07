"""Hand-written CUDA kernels (Numba) operating on 2-D float32 device arrays.

This module is imported only when the CUDA backend is activated, so importing
``pinn`` never requires a GPU. The kernels mirror the NumPy CPU backend exactly,
which is how the ``cuda``-marked tests validate them (output compared against the
reference backend).

Highlights:

- :func:`matmul_kernel` is a classic shared-memory **tiled** GEMM: each block
  cooperatively stages ``TPB x TPB`` tiles of A and B into shared memory and
  accumulates partial products, reusing each loaded value ``TPB`` times.
- Elementwise binary kernels support NumPy-style broadcasting between 2-D shapes
  by clamping the read index to 0 along any axis whose length is 1.
- Reductions use ``cuda.atomic.add`` into a zero-initialized output.
"""

from __future__ import annotations

import math

from numba import cuda, float32

TPB = 16  # threads per block (one tile dimension)


# --- matmul (tiled) ---------------------------------------------------------


@cuda.jit
def matmul_kernel(C, A, B):
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    row, col = cuda.grid(2)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    m, k = A.shape
    n = B.shape[1]

    acc = float32(0.0)
    n_tiles = (k + TPB - 1) // TPB
    for tile in range(n_tiles):
        a_col = tile * TPB + ty
        b_row = tile * TPB + tx
        sA[tx, ty] = A[row, a_col] if (row < m and a_col < k) else float32(0.0)
        sB[tx, ty] = B[b_row, col] if (b_row < k and col < n) else float32(0.0)
        cuda.syncthreads()
        for p in range(TPB):
            acc += sA[tx, p] * sB[p, ty]
        cuda.syncthreads()

    if row < m and col < n:
        C[row, col] = acc


@cuda.jit
def transpose_kernel(out, a):
    i, j = cuda.grid(2)
    if i < a.shape[0] and j < a.shape[1]:
        out[j, i] = a[i, j]


@cuda.jit
def fill_kernel(out, value):
    i, j = cuda.grid(2)
    if i < out.shape[0] and j < out.shape[1]:
        out[i, j] = value


# --- elementwise binary (with broadcasting) ---------------------------------


@cuda.jit
def add_kernel(out, a, b):
    i, j = cuda.grid(2)
    if i < out.shape[0] and j < out.shape[1]:
        ai = i if a.shape[0] > 1 else 0
        aj = j if a.shape[1] > 1 else 0
        bi = i if b.shape[0] > 1 else 0
        bj = j if b.shape[1] > 1 else 0
        out[i, j] = a[ai, aj] + b[bi, bj]


@cuda.jit
def mul_kernel(out, a, b):
    i, j = cuda.grid(2)
    if i < out.shape[0] and j < out.shape[1]:
        ai = i if a.shape[0] > 1 else 0
        aj = j if a.shape[1] > 1 else 0
        bi = i if b.shape[0] > 1 else 0
        bj = j if b.shape[1] > 1 else 0
        out[i, j] = a[ai, aj] * b[bi, bj]


@cuda.jit
def div_kernel(out, a, b):
    i, j = cuda.grid(2)
    if i < out.shape[0] and j < out.shape[1]:
        ai = i if a.shape[0] > 1 else 0
        aj = j if a.shape[1] > 1 else 0
        bi = i if b.shape[0] > 1 else 0
        bj = j if b.shape[1] > 1 else 0
        out[i, j] = a[ai, aj] / b[bi, bj]


# --- elementwise unary ------------------------------------------------------


@cuda.jit
def pow_scalar_kernel(out, a, p):
    i, j = cuda.grid(2)
    if i < a.shape[0] and j < a.shape[1]:
        out[i, j] = a[i, j] ** p


@cuda.jit
def neg_kernel(out, a):
    i, j = cuda.grid(2)
    if i < a.shape[0] and j < a.shape[1]:
        out[i, j] = -a[i, j]


@cuda.jit
def sin_kernel(out, a):
    i, j = cuda.grid(2)
    if i < a.shape[0] and j < a.shape[1]:
        out[i, j] = math.sin(a[i, j])


@cuda.jit
def cos_kernel(out, a):
    i, j = cuda.grid(2)
    if i < a.shape[0] and j < a.shape[1]:
        out[i, j] = math.cos(a[i, j])


@cuda.jit
def log_kernel(out, a):
    i, j = cuda.grid(2)
    if i < a.shape[0] and j < a.shape[1]:
        out[i, j] = math.log(a[i, j])


@cuda.jit
def exp_kernel(out, a):
    i, j = cuda.grid(2)
    if i < a.shape[0] and j < a.shape[1]:
        out[i, j] = math.exp(a[i, j])


@cuda.jit
def sigmoid_kernel(out, a):
    i, j = cuda.grid(2)
    if i < a.shape[0] and j < a.shape[1]:
        out[i, j] = 1.0 / (1.0 + math.exp(-a[i, j]))


# --- reductions -------------------------------------------------------------


@cuda.jit
def sum_all_kernel(out, a):
    i, j = cuda.grid(2)
    if i < a.shape[0] and j < a.shape[1]:
        cuda.atomic.add(out, (0, 0), a[i, j])


@cuda.jit
def sum_axis0_kernel(out, a):
    i, j = cuda.grid(2)
    if i < a.shape[0] and j < a.shape[1]:
        cuda.atomic.add(out, (0, j), a[i, j])


@cuda.jit
def sum_axis1_kernel(out, a):
    i, j = cuda.grid(2)
    if i < a.shape[0] and j < a.shape[1]:
        cuda.atomic.add(out, (i, 0), a[i, j])
