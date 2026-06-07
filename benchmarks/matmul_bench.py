"""Matmul throughput: hand-written kernels vs vendor libraries.

Times a square matmul across several backends and reports GFLOP/s, to make the
report's point concrete: the hand-written tiled kernel is a faithful *teaching*
implementation of what cuBLAS does, and is far from its tuned performance.

Backends are probed at runtime; whatever is unavailable is skipped:

- numpy           — CPU baseline (always available)
- pinn-numba      — the repo's hand-written CUDA tiled kernel
- cupy/cuBLAS     — NVIDIA's tuned GEMM
- triton          — the block-model kernel in triton_matmul.py
- torch/cuBLAS    — for reference

Run:  poetry run python benchmarks/matmul_bench.py
"""

from __future__ import annotations

import time

import numpy as np

SIZES = [128, 512, 1024, 2048]
REPEATS = 20


def _time(fn, sync=None) -> float:
    fn()  # warm up / compile
    if sync:
        sync()
    t0 = time.perf_counter()
    for _ in range(REPEATS):
        fn()
    if sync:
        sync()
    return (time.perf_counter() - t0) / REPEATS


def _gflops(n: int, seconds: float) -> float:
    return 2.0 * n**3 / seconds / 1e9


def bench_numpy(n):
    a = np.random.randn(n, n).astype(np.float32)
    b = np.random.randn(n, n).astype(np.float32)
    return _time(lambda: a @ b)


def bench_pinn_numba(n):
    from numba import cuda

    if not cuda.is_available():
        return None
    from pinn.backend import cuda as cu

    a = cu.asarray(np.random.randn(n, n).astype(np.float32))
    b = cu.asarray(np.random.randn(n, n).astype(np.float32))
    return _time(lambda: cu.matmul(a, b), sync=cuda.synchronize)


def bench_cupy(n):
    try:
        import cupy as cp
    except ImportError:
        return None
    a = cp.random.randn(n, n, dtype=cp.float32)
    b = cp.random.randn(n, n, dtype=cp.float32)
    return _time(lambda: a @ b, sync=cp.cuda.runtime.deviceSynchronize)


def bench_triton(n):
    import triton_matmul as tm

    if not tm.HAVE_TRITON:
        return None
    import torch

    if not torch.cuda.is_available():
        return None
    a = torch.randn(n, n, device="cuda", dtype=torch.float32)
    b = torch.randn(n, n, device="cuda", dtype=torch.float32)
    return _time(lambda: tm.matmul(a, b), sync=torch.cuda.synchronize)


def bench_torch(n):
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    a = torch.randn(n, n, device="cuda", dtype=torch.float32)
    b = torch.randn(n, n, device="cuda", dtype=torch.float32)
    return _time(lambda: a @ b, sync=torch.cuda.synchronize)


BACKENDS = {
    "numpy (CPU)": bench_numpy,
    "pinn-numba (CUDA)": bench_pinn_numba,
    "cupy/cuBLAS": bench_cupy,
    "triton": bench_triton,
    "torch/cuBLAS": bench_torch,
}


def main() -> None:
    header = f"{'backend':<22}" + "".join(f"{n:>12}" for n in SIZES)
    print(header)
    print("-" * len(header))
    for label, fn in BACKENDS.items():
        cells = []
        for n in SIZES:
            try:
                secs = fn(n)
            except Exception as exc:  # keep going if one size/backend fails
                secs = None
                print(f"  ! {label} n={n}: {exc}")
            cells.append("    skipped" if secs is None else f"{_gflops(n, secs):>9.1f}gf")
        print(f"{label:<22}" + "".join(cells))


if __name__ == "__main__":
    main()
