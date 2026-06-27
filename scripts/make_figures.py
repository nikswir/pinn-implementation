"""Train the PINN and render the figures used in the README and the report.

Outputs (PNG, into docs/report/figures/):
    comparison_t1.png  - PINN prediction vs analytic solution at t=1
    error_field.png    - absolute error |u_pred - u_exact| at t=1
    loss_curve.png     - training loss vs epoch

Run:  uv sync --extra viz && uv run python scripts/make_figures.py
"""

from __future__ import annotations

import argparse
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from pathlib import Path  # noqa: E402

from pinn.train import train, evaluate, TrainConfig  # noqa: E402

FIG_DIR = Path(__file__).resolve().parent.parent / "docs" / "report" / "figures"


def _surface(ax, X, Y, Z, **kwargs):
    return ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True, **kwargs)


def make_comparison(X, Y, u_pred, u_ref):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    _surface(ax, X, Y, u_ref, color="tab:blue", alpha=0.6)
    _surface(ax, X, Y, u_pred, color="tab:red", alpha=0.6)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u")
    ax.set_title("PINN (red) vs analytic (blue) at t = 1")
    fig.tight_layout()
    return fig


def make_error(X, Y, u_pred, u_ref):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    err = abs(u_pred - u_ref)
    surf = _surface(ax, X, Y, err, cmap="viridis")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("|error|")
    ax.set_title(f"Absolute error at t = 1 (max {err.max():.3f})")
    fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.1)
    fig.tight_layout()
    return fig


def make_loss(history):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history, color="tab:purple")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("PINN loss")
    ax.set_title("Training loss")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"training {args.epochs} epochs ...")
    model, history = train(
        TrainConfig(epochs=args.epochs, seed=args.seed),
        verbose=False,
    )
    X, Y, u_pred, u_ref, max_err = evaluate(model, n=60, t=1.0)
    print(f"max abs error vs analytic at t=1: {max_err:.4f}")

    figures = {
        "comparison_t1.png": make_comparison(X, Y, u_pred, u_ref),
        "error_field.png": make_error(X, Y, u_pred, u_ref),
        "loss_curve.png": make_loss(history),
    }
    for filename, fig in figures.items():
        path = FIG_DIR / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"wrote {path.relative_to(FIG_DIR.parent.parent.parent)}")


if __name__ == "__main__":
    main()
