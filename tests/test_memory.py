"""Tests for the pooled memory allocator (runs on CPU, no GPU needed)."""

from __future__ import annotations

import numpy as np
import pytest

from pinn.backend.memory import MemoryManager


def test_basic_alloc_returns_distinct_nonoverlapping_slices():
    mm = MemoryManager(100)
    a = mm.allocate(10)
    b = mm.allocate(20)
    c = mm.allocate(5)
    spans = sorted([(a, 10), (b, 20), (c, 5)])
    for (s1, n1), (s2, _) in zip(spans, spans[1:], strict=False):
        assert s1 + n1 <= s2, "allocations overlap"
    assert mm.used == 35


def test_free_then_reuse():
    mm = MemoryManager(100)
    a = mm.allocate(40)
    mm.allocate(40)
    mm.free(a, 40)
    # the freed 40-slot block should be reusable
    d = mm.allocate(40)
    assert d == a
    assert mm.used == 80


def test_coalescing_merges_adjacent_gaps():
    mm = MemoryManager(100)
    a = mm.allocate(10)
    b = mm.allocate(10)
    c = mm.allocate(10)
    # free the two outer blocks, then the middle: all three should merge
    mm.free(a, 10)
    mm.free(c, 10)
    mm.free(b, 10)
    # the whole 100 capacity is free again -> a single 100-slot allocation fits
    assert mm.used == 0
    big = mm.allocate(100)
    assert big == 0


def test_out_of_capacity_raises():
    mm = MemoryManager(50)
    mm.allocate(50)
    with pytest.raises(MemoryError):
        mm.allocate(1)


def test_cpu_backend_allocates_from_pool_and_releases():
    """The CPU backend is the live allocation path: ops draw from the pool and
    return their memory once the tensors are garbage-collected."""
    import gc

    from pinn import backend
    from pinn.core.tensor import Tensor

    p = backend.cpu.pool()
    gc.collect()
    base = p.used

    a = Tensor(np.random.randn(64, 64))
    b = Tensor(np.random.randn(64, 64))
    c = a @ b
    assert p.used > base  # the result was allocated from the pool

    del a, b, c
    gc.collect()
    assert p.used == base  # everything returned to the pool


def test_randomized_no_overlap_invariant():
    rng = np.random.default_rng(0)
    mm = MemoryManager(1000)
    live: list[tuple[int, int]] = []
    for _ in range(500):
        if live and rng.random() < 0.5:
            i = rng.integers(len(live))
            start, size = live.pop(i)
            mm.free(start, size)
        else:
            size = int(rng.integers(1, 50))
            try:
                start = mm.allocate(size)
            except MemoryError:
                continue
            # check no overlap with currently live spans
            for s, n in live:
                assert start + size <= s or s + n <= start, "overlap detected"
            live.append((start, size))
