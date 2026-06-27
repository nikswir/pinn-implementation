"""Shared pytest configuration: auto-skip CUDA tests when no GPU is present.

Also defines the stage-1 / stage-2 split. Stage 2 (heavy: long-running, GPU)
is skipped unless RUN_STAGE2=1, so the default `pytest` run stays fast and
deterministic. Mark a heavy test with `@stage2` (import it from this module).
"""

from __future__ import annotations

import os

import pytest

########################################
#          Stage-2 test gate           #
########################################

# Stage-2 tests are heavy (GPU, slow); skip them unless explicitly enabled
# (RUN_STAGE2=1) so the default suite stays offline and fast.
RUN_STAGE2 = os.environ.get("RUN_STAGE2") == "1"

stage2 = pytest.mark.skipif(
    not RUN_STAGE2,
    reason="set RUN_STAGE2=1 to run heavy stage-2 tests",
)


def _cuda_available() -> bool:
    try:
        from numba import cuda

        return cuda.is_available()
    except Exception:
        return False


CUDA_AVAILABLE = _cuda_available()


def pytest_collection_modifyitems(config, items):
    if CUDA_AVAILABLE:
        return
    skip = pytest.mark.skip(reason="no CUDA GPU available")
    for item in items:
        if "cuda" in item.keywords:
            item.add_marker(skip)
