"""Compute-backend selection.

The autograd ``Tensor`` is backend-agnostic: it builds the computation graph and
delegates every array primitive to the *active* backend. Two backends exist:

- ``cpu``  — pure NumPy, always available (default, used by the test suite).
- ``cuda`` — hand-written Numba/CUDA kernels, used when a GPU is present.

Switch with :func:`use`; query with :func:`active`.
"""

from __future__ import annotations

from types import ModuleType

from pinn.backend import cpu

_active: ModuleType = cpu


def active() -> ModuleType:
    """Return the currently active backend module."""
    return _active


def use(name: str) -> ModuleType:
    """Activate a backend by name (``"cpu"`` or ``"cuda"``)."""
    global _active
    if name == "cpu":
        _active = cpu
    elif name == "cuda":
        from pinn.backend import cuda  # lazy: needs numba.cuda + a GPU

        _active = cuda
    else:
        raise ValueError(f"unknown backend {name!r} (expected 'cpu' or 'cuda')")
    return _active


def cuda_available() -> bool:
    """True if a CUDA GPU usable by Numba is present."""
    try:
        from numba import cuda as _c

        return bool(_c.is_available())
    except Exception:
        return False
