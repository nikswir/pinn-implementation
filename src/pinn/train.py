"""Training loop for the heat-equation PINN.

Usage:
    uv run python -m pinn.train --epochs 2000
"""

from __future__ import annotations

import argparse
import numpy as np

from dataclasses import dataclass
from collections.abc import Callable

from pinn.nn import MLP, Adam
from pinn.pde.analytic import u_exact
from pinn.pde.heat import PI_2, HeatPINN
from pinn.core.tensor import grad, Tensor


@dataclass
class TrainConfig:
    epochs: int = 2000
    lr: float = 1e-2
    hidden: int = 32
    n_pde: int = 200
    n_bc: int = 50
    seed: int = 0
    log_every: int = 100


def train(
    cfg: TrainConfig | None = None,
    verbose: bool = True,
) -> tuple[MLP, list[float]]:
    """Train an MLP to solve the heat PDE; return (model, loss_history)."""
    cfg = cfg or TrainConfig()
    rng = np.random.default_rng(cfg.seed)
    model = MLP(3, 1, hidden=cfg.hidden, rng=rng)
    pinn = HeatPINN(model, rng=rng, n_pde=cfg.n_pde, n_bc=cfg.n_bc)
    params = model.parameters()
    opt = Adam(params, lr=cfg.lr)

    history: list[float] = []
    for epoch in range(cfg.epochs):
        loss = pinn.total_loss()
        grads = grad(loss, params)
        opt.step(grads)
        history.append(loss.item())
        if verbose and (epoch + 1) % cfg.log_every == 0:
            print(f"epoch {epoch + 1:5d}  loss {loss.item():.6e}")
    return model, history


def evaluate(
    model: Callable[[Tensor], Tensor],
    n: int = 50,
    t: float = 1.0,
    n_terms: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Return (X, Y, u_pred, u_exact, max_abs_error) on a grid at fixed t."""
    xs = np.linspace(0.0, 1.0, n)
    ys = np.linspace(0.0, PI_2, n)
    X, Y = np.meshgrid(xs, ys)
    pts = np.stack([X.ravel(), Y.ravel(), np.full(X.size, t)], axis=1)
    u_pred = model(Tensor(pts)).numpy().reshape(X.shape)
    u_ref = u_exact(X, Y, t, n_terms=n_terms)
    return X, Y, u_pred, u_ref, float(np.abs(u_pred - u_ref).max())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the heat-equation PINN.",
    )
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = TrainConfig(epochs=args.epochs, lr=args.lr, seed=args.seed)
    model, _ = train(cfg)
    *_, max_err = evaluate(model, t=1.0)
    print(f"max abs error vs analytic at t=1.0: {max_err:.4f}")


if __name__ == "__main__":
    main()
