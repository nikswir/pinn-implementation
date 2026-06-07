"""A matrix multiply written in Triton (block/tile programming model).

Contrast with ``pinn/backend/kernels.py``: the Numba kernel is written from the
point of view of a single *thread* (explicit ``threadIdx``, shared-memory tiles,
``syncthreads``). Triton is written from the point of view of a single *block* —
you operate on whole tiles (``tl.load`` / ``tl.dot`` / ``tl.store``) and the
compiler generates the thread choreography, shared memory and synchronization.

Requires a GPU + Triton:  poetry install --extras gpu
"""

from __future__ import annotations

try:
    import torch
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - optional dependency
    HAVE_TRITON = False


if HAVE_TRITON:

    @triton.jit
    def _matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc, mask=mask)

    def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Triton-backed C = A @ B for 2-D float32 CUDA tensors."""
        m, k = a.shape
        _, n = b.shape
        c = torch.empty((m, n), device=a.device, dtype=torch.float32)
        block_m = block_n = 64
        block_k = 32
        grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
        _matmul_kernel[grid](
            a, b, c, m, n, k,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        )
        return c
