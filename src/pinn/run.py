"""Hydra entry point for pinn.

A run is composed from `configs/` (the `train` group) and written into Hydra's
per-run output directory, so repeated runs and `--multirun` sweeps never
collide. Things to try::

    python -m pinn.run --cfg job              # print the composed config
    python -m pinn.run train.epochs=500       # override a single field
    python -m pinn.run -m train.seed=0,1,2    # sweep over seeds
"""

from __future__ import annotations

import hydra

from typing import cast
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from hydra.core.hydra_config import HydraConfig

from pinn import config_schema
from pinn.train import train, evaluate, TrainConfig

# Register the structured-config schema so Hydra type-checks the composed YAML.
config_schema.register()

########################################
#               Core run               #
########################################


def run(cfg: DictConfig, out_dir: Path) -> float:
    """Train the PINN from the composed config; write a summary into out_dir."""
    # ── The `train` group is typed AS TrainConfig (config_schema), so rebuild
    #    the dataclass directly instead of copying field by field ──
    train_cfg = cast(TrainConfig, OmegaConf.to_object(cfg.train))
    model, history = train(train_cfg)
    *_, max_err = evaluate(model, t=1.0)

    summary = out_dir / "summary.txt"
    summary.write_text(
        f"final_loss={history[-1]:.6e}\nmax_abs_error_t1={max_err:.6f}\n",
    )
    return max_err


########################################
#             Entry point              #
########################################


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    # ── Hydra gives each run (and each --multirun job) its own output dir ──
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    max_err = run(cfg, out_dir)
    print(f"max abs error vs analytic at t=1.0: {max_err:.4f}")


if __name__ == "__main__":
    main()
