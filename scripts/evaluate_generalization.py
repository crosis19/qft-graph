"""Evaluate trained model generalization across coupling values.

Tests whether the HeteroGNN has learned a transferable energy functional
by evaluating a model trained at one (m^2, lambda) on data generated at
different m^2 values (same lambda).

Usage:
    python scripts/evaluate_generalization.py \
        --checkpoint experiments/runs/colab_run/checkpoint_final.pt \
        --data_dir data/mc_configs \
        [--m2_values -0.3 -0.7 -1.0 -1.5 -2.0] \
        [--lambda_val 0.5] [--lattice_size 16]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from qft_graph.actions.phi4 import Phi4Action
from qft_graph.config import LatticeConfig, MCConfig, ScalarFieldConfig, load_config
from qft_graph.fields.scalar import ScalarField
from qft_graph.graphs.builder import HeteroGraphBuilder
from qft_graph.lattice.hypercubic import HypercubicLattice
from qft_graph.mc.metropolis import create_sampler
from qft_graph.models.hetero_gnn import HeteroGNN
from qft_graph.training.metrics import energy_correlation, relative_error
from qft_graph.utils.checkpointing import remap_legacy_state_dict
from qft_graph.utils.logging import setup_logging
from qft_graph.utils.reproducibility import set_seed


def generate_or_load_data(
    m2: float,
    lam: float,
    lattice_size: int,
    data_dir: Path,
    n_configs: int = 2000,
    logger: logging.Logger | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate or load MC data at a given coupling point."""
    dirname = f"phi4_{lattice_size}x{lattice_size}_m2={m2}_lam={lam}"
    data_path = data_dir / dirname / "mc_data.pt"

    if data_path.exists():
        if logger:
            logger.info("Loading existing data from %s", data_path)
        mc_data = torch.load(data_path, weights_only=False)
        return mc_data["configurations"][:n_configs], mc_data["actions"][:n_configs]

    if logger:
        logger.info("Generating %d configs at m2=%.2f, lambda=%.2f, L=%d",
                     n_configs, m2, lam, lattice_size)

    lattice_config = LatticeConfig(dimensions=(lattice_size, lattice_size))
    lattice = HypercubicLattice(lattice_config)
    field_config = ScalarFieldConfig(mass_squared=m2, coupling=lam)
    action = Phi4Action(lattice, field_config)

    mc_config = MCConfig(
        n_configs=n_configs,
        n_thermalization=1000,
        n_sweeps_between=10,
        seed=42,
    )

    sampler = create_sampler(action, mc_config)
    result = sampler.generate(n_configs)
    configs, actions_list = result.configurations, result.actions

    # Save for reuse (same layout as scripts/generate_mc_data.py)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "configurations": configs,
            "actions": actions_list,
            "acceptance_rate": result.acceptance_rate,
        },
        data_path,
    )

    return configs[:n_configs], actions_list[:n_configs]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate coupling generalization")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="data/mc_configs")
    parser.add_argument("--m2_values", type=float, nargs="+",
                        default=[-0.3, -0.5, -0.7, -1.0, -1.5, -2.0])
    parser.add_argument("--lambda_val", type=float, default=0.5)
    parser.add_argument("--lattice_size", type=int, default=16)
    parser.add_argument("--n_configs", type=int, default=2000)
    parser.add_argument("--output", type=str,
                        default="experiments/generalization_results.json")
    args = parser.parse_args()

    logger = setup_logging()
    set_seed(42)

    config = load_config(args.config)
    config.lattice.dimensions = (args.lattice_size, args.lattice_size)

    # Setup lattice and model
    lattice = HypercubicLattice(config.lattice)
    scalar_field = ScalarField()
    builder = HeteroGraphBuilder(lattice, [scalar_field], a_in_edges=config.model.a_in_edges)

    model = HeteroGNN(
        config=config.model,
        lattice_dim=lattice.dimension(),
        field_types={"scalar": scalar_field.dof_per_site()},
        lattice_spacing=lattice.lattice_spacing(),
    )

    ckpt = torch.load(args.checkpoint, weights_only=False)
    model.load_state_dict(remap_legacy_state_dict(ckpt["model_state_dict"]))
    model.eval()
    logger.info("Loaded checkpoint from %s", args.checkpoint)

    data_dir = Path(args.data_dir)
    results = []

    for m2 in args.m2_values:
        configs, actions = generate_or_load_data(
            m2, args.lambda_val, args.lattice_size,
            data_dir, args.n_configs, logger,
        )

        # Build graphs
        dataset = builder.build_dataset(
            configurations={"scalar": configs},
            actions=actions,
        )

        # Evaluate
        all_pred, all_true = [], []
        with torch.no_grad():
            for data in dataset:
                output = model(data)
                all_pred.append(output["energy"])
                all_true.append(data.y)

        preds = torch.cat(all_pred)
        trues = torch.cat(all_true)
        corr = energy_correlation(preds, trues)
        rel_err = relative_error(preds, trues)

        is_training = (abs(m2 - (-0.5)) < 1e-6)
        result = {
            "m2": m2,
            "lambda": args.lambda_val,
            "pearson_r": round(corr, 6),
            "relative_error": round(rel_err, 6),
            "n_configs": len(dataset),
            "is_training_point": is_training,
        }
        results.append(result)

        marker = " (training point)" if is_training else ""
        logger.info("m2=%.2f: r=%.4f, rel_err=%.4f%s", m2, corr, rel_err, marker)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", output_path)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("GENERALIZATION RESULTS")
    logger.info("=" * 60)
    logger.info("%-10s %12s %12s %s", "m^2", "Pearson r", "Rel. Error", "")
    logger.info("-" * 60)
    for r in results:
        marker = " *" if r["is_training_point"] else ""
        logger.info("%-10.2f %12.4f %12.4f%s",
                    r["m2"], r["pearson_r"], r["relative_error"], marker)
    logger.info("* = training coupling point")


if __name__ == "__main__":
    main()
