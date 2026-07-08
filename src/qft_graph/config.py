"""Hierarchical configuration system using dataclasses and OmegaConf."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
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

    n_configs: int = 5000
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
    a_in_edges: bool = True  # put lattice spacing in edge displacements instead of nodes
    volume_scaling: bool = True  # multiply per-site energy by a^d
    # "coords" | "constant": constant replaces coordinates with a single 1.0
    # feature (geometry via displacement edges only) — translation-invariant,
    # transfers across lattice sizes (task P1-6)
    spacetime_features: str = "coords"
    # Feed (m2, lambda) to the model: field nodes get [phi, m2, lambda] and the
    # readout is conditioned on them (task P1-3). False preserves the original
    # paper architecture, which cannot condition on couplings.
    condition_on_couplings: bool = False


@dataclass
class TrainingConfig:
    """Training loop configuration."""

    epochs: int = 150
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

    lattice: LatticeConfig = dataclass_field(default_factory=LatticeConfig)
    scalar_field: ScalarFieldConfig = dataclass_field(default_factory=ScalarFieldConfig)
    mc: MCConfig = dataclass_field(default_factory=MCConfig)
    model: ModelConfig = dataclass_field(default_factory=ModelConfig)
    training: TrainingConfig = dataclass_field(default_factory=TrainingConfig)
    device: str = "auto"  # "auto" | "cuda" | "cpu"; resolve via resolve_device()
    experiment_name: str = "default"


def resolve_device(device: str) -> str:
    """Resolve a device string, auto-detecting CUDA availability.

    "auto" (and "cuda" when CUDA is unavailable) resolve to the best
    available device, so configs never hard-require a GPU.
    """
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


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
        # Convert dotted keys like "mc.n_configs" into nested dicts
        # so OmegaConf.merge works with structured configs
        nested: dict = {}
        for dotted_key, value in overrides.items():
            parts = dotted_key.split(".")
            d = nested
            for part in parts[:-1]:
                d = d.setdefault(part, {})
            d[parts[-1]] = value
        schema = OmegaConf.merge(schema, nested)

    return OmegaConf.to_object(schema)
