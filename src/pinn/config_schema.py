"""Structured-config schema — the run's typed contract.

Hydra validates `configs/` against these dataclasses: each field has a type and
either a literal default or `MISSING` (required — supplied by a config group).
Registering the root makes Hydra reject wrong types / unknown fields at startup,
instead of crashing deep inside the run. Add one dataclass per group.
"""

from __future__ import annotations

from dataclasses import field, dataclass
from hydra.core.config_store import ConfigStore

from pinn.train import TrainConfig

########################################
#            Group schemas             #
########################################

# The `train` group reuses the library's own `TrainConfig` (pinn.train) as the
# single source of truth for the training hyper-parameters and their defaults.

########################################
#           Root & registry            #
########################################


@dataclass
class Config:
    """The composed run config — one field per group."""

    train: TrainConfig = field(default_factory=TrainConfig)


def register() -> None:
    # ── Expose the schema as `config_schema` for config.yaml's defaults ──
    ConfigStore.instance().store(name="config_schema", node=Config)
