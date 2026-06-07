"""CUDA backend parity tests (skipped automatically when no GPU is present).

Each CUDA primitive is compared against the NumPy reference backend, and the
full autograd path (forward + gradients) is checked end-to-end on the GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from pinn import backend
from pinn.backend import cpu

pytestmark = pytest.mark.cuda


@pytest.fixture
def cuda_backend():
    backend.use("cuda")
    yield backend.active()
    backend.use("cpu")


def test_elementwise_and_matmul_match_cpu(cuda_backend):
    rng = np.random.default_rng(0)
    a = rng.standard_normal((20, 12)).astype(np.float32)
    b = rng.standard_normal((20, 12)).astype(np.float32)
    bias = rng.standard_normal((1, 12)).astype(np.float32)
    w = rng.standard_normal((12, 7)).astype(np.float32)

    da, db, dbias, dw = (cuda_backend.asarray(x) for x in (a, b, bias, w))

    cases = {
        "add": (cuda_backend.to_numpy(cuda_backend.add(da, db)), cpu.add(a, b)),
        "broadcast_add": (cuda_backend.to_numpy(cuda_backend.add(da, dbias)), cpu.add(a, bias)),
        "mul": (cuda_backend.to_numpy(cuda_backend.mul(da, db)), cpu.mul(a, b)),
        "sin": (cuda_backend.to_numpy(cuda_backend.sin(da)), cpu.sin(a)),
        "sigmoid": (cuda_backend.to_numpy(cuda_backend.sigmoid(da)), cpu.sigmoid(a)),
        "pow": (cuda_backend.to_numpy(cuda_backend.power(da, 3.0)), cpu.power(a, 3.0)),
        "matmul": (cuda_backend.to_numpy(cuda_backend.matmul(da, dw)), cpu.matmul(a, w)),
        "transpose": (cuda_backend.to_numpy(cuda_backend.transpose(da)), cpu.transpose(a)),
        "sum_all": (cuda_backend.to_numpy(cuda_backend.sum_all(da)), cpu.sum_all(a)),
        "sum0": (cuda_backend.to_numpy(cuda_backend.sum_axis(da, 0)), cpu.sum_axis(a, 0)),
        "sum1": (cuda_backend.to_numpy(cuda_backend.sum_axis(da, 1)), cpu.sum_axis(a, 1)),
    }
    for label, (got, want) in cases.items():
        assert np.allclose(got, want, atol=1e-4), label


def test_autograd_on_gpu(cuda_backend):
    from pinn.core.tensor import Tensor, grad

    rng = np.random.default_rng(1)
    x = rng.standard_normal((4, 3)).astype(np.float32)
    t = Tensor(x, requires_grad=True)
    g = grad((t.sin() * (t**2)).sum(), t).numpy()
    expected = np.cos(x) * x**2 + 2 * x * np.sin(x)
    assert np.allclose(g, expected, atol=1e-3)


def test_cuda_backend_allocates_from_pool_and_releases(cuda_backend):
    """The CUDA backend draws device memory from the pool and returns it on GC."""
    import gc

    from pinn.core.tensor import Tensor

    p = cuda_backend.pool()
    gc.collect()
    base = p.used

    a = Tensor(np.random.randn(64, 64))
    b = Tensor(np.random.randn(64, 64))
    c = a @ b
    assert p.used > base  # the result was allocated from the device pool

    del a, b, c
    gc.collect()
    assert p.used == base  # everything returned to the pool
