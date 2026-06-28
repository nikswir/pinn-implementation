"""Stage-1 smoke tests for the training driver: train, evaluate, run.

The fast suite otherwise re-implements the loop inline (test_pinn), leaving the
actual ``train`` / ``evaluate`` / ``run`` bodies exercised only by the @stage2
convergence test. These run a tiny 2-3 epoch config so a mutation in the driver
(a dropped ``history.append``, a ``.max()`` -> ``.min()`` in the error
reduction, a wrong epoch count) is caught on every commit, not just under
``RUN_STAGE2=1``.
"""

from __future__ import annotations

import numpy as np

from pathlib import Path
from omegaconf import OmegaConf

from pinn.run import run
from pinn.config_schema import Config
from pinn.train import train, evaluate, TrainConfig


def test_train_returns_history_of_epoch_length() -> None:
    cfg = TrainConfig(epochs=3, hidden=4, n_pde=8, n_bc=4, seed=0)
    model, history = train(cfg, verbose=False)
    assert len(history) == cfg.epochs
    assert np.isfinite(history).all()
    assert callable(model)


def test_evaluate_returns_grid_and_max_abs_error() -> None:
    cfg = TrainConfig(epochs=1, hidden=4, n_pde=8, n_bc=4, seed=0)
    model, _ = train(cfg, verbose=False)
    X, Y, u_pred, u_ref, max_err = evaluate(model, n=5, t=1.0)
    assert X.shape == (5, 5)
    assert u_pred.shape == (5, 5)
    assert u_ref.shape == (5, 5)
    # The reported error is the MAX absolute error over the grid (not min/mean).
    assert np.isclose(max_err, np.abs(u_pred - u_ref).max())
    assert max_err >= 0.0


def test_run_trains_and_writes_summary(tmp_path: Path) -> None:
    cfg = OmegaConf.structured(
        Config(train=TrainConfig(epochs=2, hidden=4, n_pde=8, n_bc=4)),
    )
    max_err = run(cfg, tmp_path)
    summary = (tmp_path / "summary.txt").read_text()
    assert "final_loss=" in summary
    assert f"max_abs_error_t1={max_err:.6f}" in summary
