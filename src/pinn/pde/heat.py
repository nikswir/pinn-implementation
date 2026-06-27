r"""Physics-Informed loss for the 2-D heat equation.

The network ``u(x, y, t)`` is trained to minimize the residual of the PDE
together with the boundary and initial conditions — no labelled data. Each loss
term is a mean-squared residual evaluated on points sampled from the relevant
region of the domain.

    PDE:        u_t - u_xx - u_yy - x t^2 sin(y) = 0   on the interior
    Neumann:    u_x = 0 at x=0 and x=1;  u_y = 0 at y=pi/2
    Dirichlet:  u   = 0 at y=0
    Initial:    u   = 0 at t=0

Second derivatives (``u_xx``, ``u_yy``) come from differentiating the first
derivatives again, which is why the autograd engine must keep gradients
differentiable (``create_graph=True``).
"""

from __future__ import annotations

import math

import numpy as np

from collections.abc import Callable

from pinn.core.tensor import grad, Tensor

PI_2 = math.pi / 2.0


class HeatPINN:
    """Loss assembler for the heat-equation PINN on (0,1) x (0,pi/2) x (0,1)."""

    def __init__(
        self,
        model: Callable[[Tensor], Tensor],
        rng: np.random.Generator | None = None,
        n_pde: int = 200,
        n_bc: int = 50,
    ) -> None:
        self.model = model
        self.rng = rng or np.random.default_rng()
        self.n_pde = n_pde
        self.n_bc = n_bc

    ########################################
    #            residual terms            #
    ########################################

    def pde_loss(self) -> Tensor:
        x = self.rng.uniform(size=(self.n_pde, 1))
        y = self.rng.uniform(size=(self.n_pde, 1)) * PI_2
        t = self.rng.uniform(size=(self.n_pde, 1))
        xyt = Tensor(np.concatenate([x, y, t], axis=1), requires_grad=True)
        u = self.model(xyt)

        first = grad(u.sum(), xyt, create_graph=True)
        u_x, u_y, u_t = first[:, 0], first[:, 1], first[:, 2]

        u_xx = grad(u_x.sum(), xyt, create_graph=True)[:, 0]
        u_yy = grad(u_y.sum(), xyt, create_graph=True)[:, 1]

        source = xyt[:, 0] * (xyt[:, 2] ** 2) * xyt[:, 1].sin()
        residual = u_t - u_xx - u_yy - source
        return (residual**2).sum() * (1.0 / self.n_pde)

    def _neumann_loss(
        self,
        *,
        x: float | None = None,
        y: float | None = None,
    ) -> Tensor:
        """MSE of the outward normal derivative on a face (x or y const)."""
        t = self.rng.uniform(size=(self.n_bc, 1))
        if x is not None:
            xc = np.full((self.n_bc, 1), x)
            yc = self.rng.uniform(size=(self.n_bc, 1)) * PI_2
            axis = 0
        else:
            yc = np.full((self.n_bc, 1), y)
            xc = self.rng.uniform(size=(self.n_bc, 1))
            axis = 1
        xyt = Tensor(np.concatenate([xc, yc, t], axis=1), requires_grad=True)
        u = self.model(xyt)
        deriv = grad(u.sum(), xyt, create_graph=True)[:, axis]
        return (deriv**2).sum() * (1.0 / self.n_bc)

    def _dirichlet_loss(self, *, y: float) -> Tensor:
        """MSE of u on a Dirichlet face (here u=0 at y=0)."""
        t = self.rng.uniform(size=(self.n_bc, 1))
        yc = np.full((self.n_bc, 1), y)
        xc = self.rng.uniform(size=(self.n_bc, 1))
        xyt = Tensor(np.concatenate([xc, yc, t], axis=1))
        u = self.model(xyt)
        return (u**2).sum() * (1.0 / self.n_bc)

    def initial_loss(self) -> Tensor:
        """MSE of u at t=0 (u=0)."""
        x = self.rng.uniform(size=(self.n_bc, 1))
        y = self.rng.uniform(size=(self.n_bc, 1)) * PI_2
        t = np.zeros((self.n_bc, 1))
        xyt = Tensor(np.concatenate([x, y, t], axis=1))
        u = self.model(xyt)
        return (u**2).sum() * (1.0 / self.n_bc)

    ########################################
    #                total                 #
    ########################################

    def total_loss(self) -> Tensor:
        return (
            self.pde_loss()
            + self._neumann_loss(x=0.0)
            + self._neumann_loss(x=1.0)
            + self._neumann_loss(y=PI_2)
            + self._dirichlet_loss(y=0.0)
            + self.initial_loss()
        )
