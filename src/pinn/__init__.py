"""pinn — a from-scratch autograd + CUDA runtime for the heat-equation PINN.

The public API: import from the package root, not from submodules.
"""

from __future__ import annotations

from pinn.core import grad, Tensor
from pinn.nn import MLP, Adam, Linear, Sigmoid
from pinn.train import train, evaluate, TrainConfig
from pinn.backend import use, active, cuda_available
from pinn.pde import u_exact, HeatPINN, u_exact_scalar

__version__ = "0.1.0"

__all__ = [
    "grad",
    "Tensor",
    "MLP",
    "Adam",
    "Linear",
    "Sigmoid",
    "train",
    "evaluate",
    "TrainConfig",
    "use",
    "active",
    "cuda_available",
    "u_exact",
    "HeatPINN",
    "u_exact_scalar",
]
