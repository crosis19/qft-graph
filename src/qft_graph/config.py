"""Hierarchical configuration system using dataclasses and OmegaConf."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


@dataclass
class LatticeConfig:
    """Spacetime lattice geometry configuration."""

    dimensions: tuple[int, ...] = (16, 16)
    spacing: float = 1.0
    boundary: str = "periodic"  # "periodic" | "open" | "antiperiodic"


@dataclass
class ScalarFieldConfig:
    """Scalar phi^4 field configuration."""

    mass_squared: float = -0.5  # m^2 (bare), negative for symmetry-broken phase
    coupling: float = 0.5  # lambda (quartic coupling)
    initialization: str = "hot"  # "hot" | "cold" | "gaussian"


@dataclass
class MCConfig:
    """Monte Carlo sampler configuration."""

    n_configs: int = 10000
    n_thermalization: int = 1000
    n_sweeps_between: int = 10
    step_size: float = 1.0  # Metropolis proposal width
    seed: int = 42


@dataclass
class ModelConfig:
    """GNN model architecture configuration."""

    hidden_dim: int = 64
    n_mp_blocks: int = 3  # number of 3-stage message passing blocks
    encoder_layers: int = 2
    dropout: float = 0.0
    activation: str = "gelu"  # "relu" | "gelu" | "silu"
    readout: str = "energy"  # "energy" | "correlator" | "both"


@dataclass
class TrainingConfig:
    """Training loop configuration."""

    epochs: int = 200
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-5
    scheduler: str = "cosine"  # "cosine" | "plateau"
    loss: str = "energy_matching"  # "energy_matching" | "kl_divergence"
    checkpoint_every: int = 20
    seed: int = 42


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration composing all sub-configs."""

    lattice: LatticeConfig = field(default_factory=LatticeConfig)
    field: ScalarFieldConfig = field(default_factory=ScalarFieldConfig)
    mc: MCConfig = field(default_factory=MCConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    device: str = "cuda"
    experiment_name: str = "default"


def load_config(
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> ExperimentConfig:
    """Load experiment config from YAML with optional CLI overrides.

    Args:
        config_path: Path to YAML config file. If None, uses defaults.
        overrides: Dict of dotted-path overrides, e.g. {"lattice.dimensions": [32, 32]}.

    Returns:
        Fully resolved ExperimentConfig.
    """
    schema = OmegaConf.structured(ExperimentConfig)

    if config_path is not None:
        file_conf = OmegaConf.load(config_path)
        schema = OmegaConf.merge(schema, file_conf)

    if overrides:
        override_conf = OmegaConf.create(overrides)
        schema = OmegaConf.merge(schema, override_conf)

    return OmegaConf.to_object(schema)
