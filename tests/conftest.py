"""Shared pytest configuration: auto-skip CUDA tests when no GPU is present."""

from __future__ import annotations

import pytest


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
