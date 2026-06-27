"""Public-API stability — pin the exported surface and its signatures.

If a test here breaks, the public contract changed: update it deliberately (and
bump the version), don't just edit the assertion away.
"""

from __future__ import annotations

import inspect
import importlib
import dataclasses

PKG = importlib.import_module("pinn")


def test_exports() -> None:
    """`__all__` is exactly the supported public surface."""
    assert set(PKG.__all__) == {
        "Tensor",
        "grad",
        "MLP",
        "Linear",
        "Sigmoid",
        "Adam",
        "active",
        "use",
        "cuda_available",
        "HeatPINN",
        "u_exact",
        "u_exact_scalar",
        "train",
        "evaluate",
        "TrainConfig",
    }


def test_train_signature() -> None:
    """`train` keeps its (cfg, verbose) contract."""
    sig = inspect.signature(PKG.train)
    assert list(sig.parameters) == ["cfg", "verbose"]


def test_evaluate_signature() -> None:
    """`evaluate` keeps its (model, n, t, n_terms) contract."""
    sig = inspect.signature(PKG.evaluate)
    assert list(sig.parameters) == ["model", "n", "t", "n_terms"]


def test_train_config_fields() -> None:
    """The TrainConfig dataclass keeps its public fields."""
    fields = [f.name for f in dataclasses.fields(PKG.TrainConfig)]
    assert fields == [
        "epochs",
        "lr",
        "hidden",
        "n_pde",
        "n_bc",
        "seed",
        "log_every",
    ]
