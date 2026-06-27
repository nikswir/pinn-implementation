"""Adam optimizer.

Operates directly on the raw backend arrays behind each parameter (no graph is
built during the update step). Implements canonical Adam: exponential moving
averages of the gradient and its square, bias-corrected (``m_hat``, ``v_hat``),
with ``eps`` added outside the square root (``sqrt(v_hat) + eps``).
"""

from __future__ import annotations

import numpy as np

from pinn.core.tensor import Tensor


class Adam:
    def __init__(
        self,
        params: list[Tensor],
        lr: float = 1e-2,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        self.params = list(params)
        self.lr = lr
        self.b1 = beta1
        self.b2 = beta2
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self, grads: list[Tensor | None]) -> None:
        self.t += 1
        bc1 = 1.0 - self.b1**self.t
        bc2 = 1.0 - self.b2**self.t
        for i, (p, g) in enumerate(zip(self.params, grads, strict=True)):
            if g is None:
                continue
            gd = g.data
            self.m[i] = self.b1 * self.m[i] + (1.0 - self.b1) * gd
            self.v[i] = self.b2 * self.v[i] + (1.0 - self.b2) * (gd * gd)
            m_hat = self.m[i] / bc1
            v_hat = self.v[i] / bc2
            update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            p.data = (p.data - update).astype(np.float32)
