"""Reference PINN for the same heat equation, written with PyTorch.

This is the "production-stack" counterpart to the from-scratch engine: the same
network, loss, and PDE, but using ``torch.autograd`` for differentiation. It
exists for two reasons — to cross-check that the hand-written solution converges
to the same place, and to make the comparison in the report concrete.

Run:
    poetry install --extras reference
    poetry run python examples/reference_pytorch.py
"""

from __future__ import annotations

import math

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "This example needs PyTorch. Install it with:\n"
        "    poetry install --extras reference"
    ) from exc

from pinn.pde.analytic import u_exact

PI_2 = math.pi / 2.0


class MLP(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.Sigmoid(),
            nn.Linear(hidden, hidden), nn.Sigmoid(),
            nn.Linear(hidden, hidden), nn.Sigmoid(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


def _grad(y, x):
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]


def pde_loss(model, n=200):
    x = torch.rand(n, 1)
    y = torch.rand(n, 1) * PI_2
    t = torch.rand(n, 1)
    xyt = torch.cat([x, y, t], dim=1).requires_grad_(True)
    u = model(xyt)
    g = _grad(u, xyt)
    u_x, u_y, u_t = g[:, 0:1], g[:, 1:2], g[:, 2:3]
    u_xx = _grad(u_x, xyt)[:, 0:1]
    u_yy = _grad(u_y, xyt)[:, 1:2]
    src = xyt[:, 0:1] * xyt[:, 2:3] ** 2 * torch.sin(xyt[:, 1:2])
    return ((u_t - u_xx - u_yy - src) ** 2).mean()


def neumann_loss(model, *, x=None, y=None, n=50):
    t = torch.rand(n, 1)
    if x is not None:
        xc = torch.full((n, 1), x)
        yc = torch.rand(n, 1) * PI_2
        axis = 0
    else:
        yc = torch.full((n, 1), y)
        xc = torch.rand(n, 1)
        axis = 1
    xyt = torch.cat([xc, yc, t], dim=1).requires_grad_(True)
    u = model(xyt)
    return (_grad(u, xyt)[:, axis : axis + 1] ** 2).mean()


def dirichlet_loss(model, *, y, n=50):
    t = torch.rand(n, 1)
    xyt = torch.cat([torch.rand(n, 1), torch.full((n, 1), y), t], dim=1)
    return (model(xyt) ** 2).mean()


def initial_loss(model, n=50):
    xyt = torch.cat([torch.rand(n, 1), torch.rand(n, 1) * PI_2, torch.zeros(n, 1)], dim=1)
    return (model(xyt) ** 2).mean()


def total_loss(model):
    return (
        pde_loss(model)
        + neumann_loss(model, x=0.0)
        + neumann_loss(model, x=1.0)
        + neumann_loss(model, y=PI_2)
        + dirichlet_loss(model, y=0.0)
        + initial_loss(model)
    )


def main(epochs: int = 2000, lr: float = 1e-2, seed: int = 0):
    torch.manual_seed(seed)
    model = MLP()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        opt.zero_grad()
        loss = total_loss(model)
        loss.backward()
        opt.step()
        if (epoch + 1) % 200 == 0:
            print(f"epoch {epoch + 1:5d}  loss {loss.item():.6e}")

    n = 50
    xs = np.linspace(0, 1, n)
    ys = np.linspace(0, PI_2, n)
    X, Y = np.meshgrid(xs, ys)
    pts = torch.tensor(
        np.stack([X.ravel(), Y.ravel(), np.ones(X.size)], axis=1), dtype=torch.float32
    )
    with torch.no_grad():
        u_pred = model(pts).numpy().reshape(X.shape)
    u_ref = u_exact(X, Y, 1.0)
    print(f"max abs error vs analytic at t=1.0: {np.abs(u_pred - u_ref).max():.4f}")


if __name__ == "__main__":
    main()
