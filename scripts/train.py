"""Train the heterogeneous GNN on Monte Carlo configurations.

Usage:
    python scripts/train.py --data data/mc_configs/phi4_16x16_m2=-0.5_lam=0.5/mc_data.pt
        [--config configs/defaults.yaml] [--experiment_name run1]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from qft_graph.config import (
    ExperimentConfig,
    LatticeConfig,
    ModelConfig,
    TrainingConfig,
    load_config,
)
from qft_graph.fields.scalar import ScalarField
from qft_graph.graphs.builder import HeteroGraphBuilder
from qft_graph.lattice.hypercubic import HypercubicLattice
from qft_graph.models.hetero_gnn import HeteroGNN
from qft_graph.training.trainer import Trainer
from qft_graph.utils.logging import setup_logging
from qft_graph.utils.reproducibility import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HeteroGNN on MC data")
    parser.add_argument("--data", type=str, required=True, help="Path to mc_data.pt")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument("--experiment_name", type=str, default="default")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    logger = setup_logging()

    overrides = {"experiment_name": args.experiment_name}
    if args.device:
        overrides["device"] = args.device

    config = load_config(args.config, overrides)
    set_seed(config.training.seed)

    device = config.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        logger.warning("CUDA not available, falling back to CPU")

    # Load MC data
    logger.info("Loading MC data from %s", args.data)
    mc_data = torch.load(args.data, weights_only=False)
    configurations = mc_data["configurations"]
    actions = mc_data["actions"]

    # Setup lattice and graph builder
    lattice = HypercubicLattice(config.lattice)
    scalar_field = ScalarField()
    builder = HeteroGraphBuilder(lattice, [scalar_field])

    # Build graph dataset
    logger.info("Building graph dataset from %d configurations...", len(configurations))
    dataset = builder.build_dataset(
        configurations={"scalar": configurations},
        actions=actions,
    )

    # Train/val split (80/20)
    n_train = int(0.8 * len(dataset))
    train_dataset = dataset[:n_train]
    val_dataset = dataset[n_train:]
    logger.info("Train: %d, Val: %d", len(train_dataset), len(val_dataset))

    # Create model
    model = HeteroGNN(
        config=config.model,
        lattice_dim=lattice.dimension(),
        field_types={"scalar": scalar_field.dof_per_site()},
        lattice_spacing=lattice.lattice_spacing(),
    )
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %d", n_params)

    # Train
    training_config = config.training
    experiment_dir = Path("experiments/runs") / config.experiment_name
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=training_config,
        experiment_dir=experiment_dir,
        device=device,
    )

    history = trainer.train()
    logger.info("Final val loss: %.6f, correlation: %.4f",
                history["val_loss"][-1], history["val_corr"][-1])


if __name__ == "__main__":
    main()
