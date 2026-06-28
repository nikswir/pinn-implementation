"""Tests for the pooled memory allocator (runs on CPU, no GPU needed)."""

from __future__ import annotations

import pytest
import numpy as np

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


def test_remove_gap_scans_past_same_size_gaps():
    """Coalescing must delete the *right* gap when several share a size.

    ``_remove_gap`` binary-searches by size, then scans forward by start. With
    three 10-slot gaps at 0, 40 and 80, the one being coalesced (80) sits at
    index 2 of the size-ordered list — so a scan that stalls at a fixed index
    (or runs backward) would either hang or delete the wrong gap. Free the
    block at 90 to merge the 80-gap, then prove the structure stayed coherent.
    """
    mm = MemoryManager(100)
    starts = [mm.allocate(10) for _ in range(10)]  # blocks at 0,10,...,90
    mm.free(starts[0], 10)  # gap (10, 0)
    mm.free(starts[4], 10)  # gap (10, 40)
    mm.free(starts[8], 10)  # gap (10, 80) -> three same-size gaps
    assert mm.used == 70
    # freeing block@90 coalesces with the 80-gap (index 2 of three 10-gaps)
    mm.free(starts[9], 10)
    assert mm.used == 60
    # the merged 20-slot gap is the only one that fits a 20-slot request
    assert mm.allocate(20) == 80
    # the two untouched 10-gaps are still exactly where they were
    assert sorted([mm.allocate(10), mm.allocate(10)]) == [0, 40]
    assert mm.used == 100


def test_construction_buffer_capacity_and_fill_fraction():
    """A default manager owns a zeroed float32 buffer of its capacity, and
    ``fill_fraction`` tracks ``used / capacity``."""
    mm = MemoryManager(64)
    assert mm.capacity == 64
    assert mm.buffer.shape == (64,)
    assert mm.buffer.dtype == np.float32
    assert mm.fill_fraction == 0.0

    mm.allocate(16)
    assert mm.used == 16
    assert mm.fill_fraction == 16 / 64


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
