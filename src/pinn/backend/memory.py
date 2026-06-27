"""A pooled memory allocator over a single flat buffer.

Training repeatedly creates and destroys thousands of small tensors. Allocating
a fresh device buffer per tensor would dominate runtime, so instead one large
buffer is allocated up front and this manager hands out and reclaims slices of
it — a malloc/free for a contiguous array.

Free space is tracked as a list of gaps kept sorted by size, so a best-fit slot
is found in O(log n) via binary search. On free, a released slice is merged with
any adjacent free gaps to limit fragmentation. Three structures stay in sync:

- ``gaps_by_size``    : sorted list of ``(size, start)`` — best-fit search
- ``gap_at_start``    : ``start -> size`` — find a gap that begins right after a
                        freed slice (right neighbor)
- ``gap_at_end``      : ``end   -> size`` — find a gap that ends right before a
                        freed slice (left neighbor)
"""

from __future__ import annotations

import bisect

import numpy as np

from typing import Any

DTYPE = np.float32


class MemoryManager:
    def __init__(self, capacity: int, buffer: Any = None):
        # The allocator only tracks offsets; the backing storage is provided
        # externally (a device buffer for the CUDA backend) or defaults to a
        # NumPy buffer for the CPU backend.
        self.capacity = capacity
        self.buffer = (
            np.zeros(capacity, dtype=DTYPE) if buffer is None else buffer
        )
        self.gap_at_start: dict[int, int] = {0: capacity}
        self.gap_at_end: dict[int, int] = {capacity: capacity}
        self.gaps_by_size: list[tuple[int, int]] = [(capacity, 0)]
        self.used = 0

    ########################################
    #              allocation              #
    ########################################

    def allocate(self, size: int) -> int:
        """Reserve ``size`` contiguous slots, returning the start index."""
        i = bisect.bisect_left(self.gaps_by_size, size, key=lambda g: g[0])
        if i >= len(self.gaps_by_size):
            raise MemoryError("memory manager is out of capacity")

        gap_size, start = self.gaps_by_size[i]
        remainder = gap_size - size
        self.used += size

        del self.gaps_by_size[i]
        del self.gap_at_start[start]
        if remainder:
            self.gap_at_start[start + size] = remainder
            self.gap_at_end[start + gap_size] = remainder
            j = bisect.bisect_left(
                self.gaps_by_size,
                remainder,
                key=lambda g: g[0],
            )
            self.gaps_by_size.insert(j, (remainder, start + size))
        else:
            del self.gap_at_end[start + gap_size]
        return start

    ########################################
    #         freeing & coalescing         #
    ########################################

    def free(self, start: int, size: int) -> None:
        """Release a slice, merging with adjacent free gaps."""
        self.used -= size
        end = start + size
        left = start in self.gap_at_end
        right = end in self.gap_at_start

        if left:
            left_size = self.gap_at_end.pop(start)
            left_start = start - left_size
            self._remove_gap(left_size, left_start)
            start, size = left_start, size + left_size
        if right:
            right_size = self.gap_at_start.pop(end)
            self._remove_gap(right_size, end)
            size += right_size
            end = start + size

        self.gap_at_start[start] = size
        self.gap_at_end[start + size] = size
        j = bisect.bisect_left(self.gaps_by_size, size, key=lambda g: g[0])
        self.gaps_by_size.insert(j, (size, start))

    def _remove_gap(self, size: int, start: int) -> None:
        """Delete the gap ``(size, start)`` from the sorted list."""
        i = bisect.bisect_left(self.gaps_by_size, size, key=lambda g: g[0])
        while self.gaps_by_size[i][1] != start:
            i += 1
        del self.gaps_by_size[i]

    ########################################
    #             diagnostics              #
    ########################################

    @property
    def fill_fraction(self) -> float:
        return self.used / self.capacity
