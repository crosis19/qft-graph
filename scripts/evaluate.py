"""Evaluate trained model: extract critical exponents and correlations.

Usage:
    python scripts/evaluate.py --checkpoint experiments/runs/default/checkpoint_final.pt
        --data data/mc_configs/phi4_16x16_m2=-0.5_lam=0.5/mc_data.pt
"""

from __future__ import annotations

import argparse
import logging

import torch

from qft_graph.config import LatticeConfig, ModelConfig, load_config
from qft_graph.fields.scalar import ScalarField
from qft_graph.graphs.builder import HeteroGraphBuilder
from qft_graph.lattice.hypercubic import HypercubicLattice
from qft_graph.mc.observables import ObservableSet
from qft_graph.models.hetero_gnn import HeteroGNN
from qft_graph.training.metrics import energy_correlation, relative_error
from qft_graph.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained HeteroGNN")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    logger = setup_logging()
    config = load_config(args.config)

    # Load data
    mc_data = torch.load(args.data, weights_only=False)
    configurations = mc_data["configurations"]
    actions = mc_data["actions"]

    # Setup
    lattice = HypercubicLattice(config.lattice)
    scalar_field = ScalarField()
    builder = HeteroGraphBuilder(lattice, [scalar_field])
    obs = ObservableSet(lattice)

    # Build test graphs
    dataset = builder.build_dataset(
        configurations={"scalar": configurations},
        actions=actions,
    )

    # Load model
    model = HeteroGNN(
        config=config.model,
        lattice_dim=lattice.dimension(),
        field_types={"scalar": scalar_field.dof_per_site()},
        lattice_spacing=lattice.lattice_spacing(),
    )

    ckpt = torch.load(args.checkpoint, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Evaluate energy prediction
    all_pred = []
    all_true = []
    with torch.no_grad():
        for data in dataset[:100]:  # Evaluate on subset
            output = model(data)
            all_pred.append(output["energy"])
            all_true.append(data.y)

    pred = torch.cat(all_pred)
    true = torch.cat(all_true)

    corr = energy_correlation(pred, true)
    rel_err = relative_error(pred, true)
    logger.info("Energy correlation: %.4f", corr)
    logger.info("Relative error: %.4f", rel_err)

    # Compute two-point function from MC data
    G_r = obs.two_point_function(configurations[0])
    xi = ObservableSet.correlation_length(G_r, lattice.lattice_spacing())
    logger.info("Correlation length xi: %.3f", xi)
    logger.info("xi/L ratio: %.4f", xi / lattice.shape[0])


if __name__ == "__main__":
    main()
