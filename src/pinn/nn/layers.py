"""Neural-network building blocks: a linear layer, sigmoid activation, and MLP.

Weights use Xavier-style initialization ``N(0, 1/in_features)``. Layers expose
:meth:`parameters` so an optimizer can collect every trainable tensor.
"""

from __future__ import annotations

import numpy as np

from pinn.core.tensor import Tensor


class Linear:
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rng: np.random.Generator | None = None,
    ):
        rng = rng or np.random.default_rng()
        scale = (1.0 / in_features) ** 0.5
        self.weight = Tensor(
            rng.standard_normal((in_features, out_features)) * scale,
            requires_grad=True,
        )
        self.bias = Tensor(np.zeros((1, out_features)), requires_grad=True)

    def __call__(self, x: Tensor) -> Tensor:
        return (x @ self.weight) + self.bias

    def parameters(self) -> list[Tensor]:
        return [self.weight, self.bias]


class Sigmoid:
    def __call__(self, x: Tensor) -> Tensor:
        return x.sigmoid()

    def parameters(self) -> list[Tensor]:
        return []


class MLP:
    """Multilayer perceptron: 4 linear layers with sigmoid activations.

    Architecture: ``in -> 32 -> 32 -> 32 -> out``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden: int = 32,
        rng: np.random.Generator | None = None,
    ):
        rng = rng or np.random.default_rng()
        self.layers: list[Linear | Sigmoid] = [
            Linear(in_features, hidden, rng),
            Sigmoid(),
            Linear(hidden, hidden, rng),
            Sigmoid(),
            Linear(hidden, hidden, rng),
            Sigmoid(),
            Linear(hidden, out_features, rng),
        ]

    def __call__(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> list[Tensor]:
        params: list[Tensor] = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
