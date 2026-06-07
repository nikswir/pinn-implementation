"""End-to-end PINN tests: gradients flow, loss decreases, solution converges."""

from __future__ import annotations

import numpy as np

from pinn.core.tensor import grad
from pinn.nn import MLP, Adam
from pinn.pde.heat import HeatPINN
from pinn.train import TrainConfig, evaluate, train


def test_gradients_reach_all_parameters():
    rng = np.random.default_rng(0)
    model = MLP(3, 1, rng=rng)
    pinn = HeatPINN(model, rng=rng, n_pde=16, n_bc=8)
    params = model.parameters()
    grads = grad(pinn.total_loss(), params)
    assert len(grads) == len(params)
    for g, p in zip(grads, params, strict=True):
        assert g is not None
        assert g.shape == p.shape
        assert np.isfinite(g.numpy()).all()


def test_training_reduces_loss():
    rng = np.random.default_rng(0)
    model = MLP(3, 1, rng=rng)
    pinn = HeatPINN(model, rng=rng, n_pde=64, n_bc=32)
    params = model.parameters()
    opt = Adam(params, lr=1e-2)
    first = pinn.total_loss().item()
    for _ in range(150):
        opt.step(grad(pinn.total_loss(), params))
    last = pinn.total_loss().item()
    assert last < 0.5 * first


def test_converges_to_analytic_slow():
    """Train long enough that the PINN approximates the analytic solution."""
    cfg = TrainConfig(epochs=800, lr=1e-2, seed=0)
    model, history = train(cfg, verbose=False)
    *_, max_err = evaluate(model, n=40, t=1.0)
    assert history[-1] < history[0]
    assert max_err < 0.2
