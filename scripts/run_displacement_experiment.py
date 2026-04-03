"""Experiment: lattice spacing in edge features vs node features.

Tests 4 architecture variants across multiple lattice sizes in two phases:
  Phase A (spacing=1.0): structural test using existing MC data
  Phase B (spacing=1/L): scaling test with rescaled targets

Usage:
    python scripts/run_displacement_experiment.py --phase a --device cpu
    python scripts/run_displacement_experiment.py --phase b --sizes 8 16
    python scripts/run_displacement_experiment.py --phase a --variants baseline a_edges_no_vol
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch

from qft_graph.config import LatticeConfig, ModelConfig, TrainingConfig
from qft_graph.fields.scalar import ScalarField
from qft_graph.graphs.builder import HeteroGraphBuilder
from qft_graph.lattice.hypercubic import HypercubicLattice
from qft_graph.models.hetero_gnn import HeteroGNN
from qft_graph.training.trainer import Trainer
from qft_graph.utils.logging import setup_logging
from qft_graph.utils.reproducibility import set_seed

logger = logging.getLogger("qft_graph.experiment")

VARIANTS = {
    "baseline":       {"a_in_edges": False, "volume_scaling": True},
    "a_edges_vol":    {"a_in_edges": True,  "volume_scaling": True},
    "a_edges_no_vol": {"a_in_edges": True,  "volume_scaling": False},
    "a_nodes_no_vol": {"a_in_edges": False, "volume_scaling": False},
}

DATA_DIR = Path("data/mc_configs")
OUTPUT_BASE = Path("experiments/runs/displacement_edges")


def run_single(
    variant_name: str,
    variant_flags: dict,
    lattice_size: int,
    spacing: float,
    action_scale: float,
    training_config: TrainingConfig,
    device: str,
    output_dir: Path,
) -> dict:
    """Train one variant at one lattice size and return metrics."""
    set_seed(training_config.seed)

    # Load MC data
    data_path = DATA_DIR / f"phi4_{lattice_size}x{lattice_size}_m2=-0.5_lam=0.5" / "mc_data.pt"
    if not data_path.exists():
        logger.warning("MC data not found at %s, skipping", data_path)
        return {"status": "skipped", "reason": "data not found"}

    mc_data = torch.load(data_path, weights_only=False)
    configurations = mc_data["configurations"]
    actions = mc_data["actions"] * action_scale

    # Build lattice and graphs
    lattice_config = LatticeConfig(dimensions=(lattice_size, lattice_size), spacing=spacing)
    lattice = HypercubicLattice(lattice_config)
    scalar_field = ScalarField()
    builder = HeteroGraphBuilder(
        lattice, [scalar_field], a_in_edges=variant_flags["a_in_edges"]
    )

    dataset = builder.build_dataset(
        configurations={"scalar": configurations},
        actions=actions,
    )

    # Train/val split (80/20)
    n_train = int(0.8 * len(dataset))
    train_dataset = dataset[:n_train]
    val_dataset = dataset[n_train:]

    # Create model
    model_config = ModelConfig(
        a_in_edges=variant_flags["a_in_edges"],
        volume_scaling=variant_flags["volume_scaling"],
    )
    model = HeteroGNN(
        config=model_config,
        lattice_dim=lattice.dimension(),
        field_types={"scalar": scalar_field.dof_per_site()},
        lattice_spacing=lattice.lattice_spacing(),
    )
    n_params = sum(p.numel() for p in model.parameters())

    # Train
    run_dir = output_dir / f"{variant_name}_{lattice_size}x{lattice_size}"
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=training_config,
        experiment_dir=run_dir,
        device=device,
    )

    logger.info(
        "=== %s | %dx%d | spacing=%.4f | params=%d ===",
        variant_name, lattice_size, lattice_size, spacing, n_params,
    )
    start = time.time()
    history = trainer.train()
    elapsed = time.time() - start

    result = {
        "variant": variant_name,
        "lattice_size": lattice_size,
        "spacing": spacing,
        "action_scale": action_scale,
        "n_params": n_params,
        "final_val_loss": history["val_loss"][-1],
        "final_val_corr": history["val_corr"][-1],
        "best_val_corr": max(history["val_corr"]),
        "training_time_s": round(elapsed, 1),
        "status": "completed",
        **variant_flags,
    }
    logger.info(
        "  -> corr=%.6f  rel_err=%.6f  time=%.1fs",
        result["final_val_corr"], result["final_val_loss"], elapsed,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Displacement edge experiment")
    parser.add_argument("--phase", choices=["a", "b", "both"], default="both")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS.keys()))
    parser.add_argument("--sizes", nargs="+", type=int, default=[8, 16, 32, 64])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    training_config = TrainingConfig(epochs=args.epochs, seed=args.seed)

    # Validate variant names
    for v in args.variants:
        if v not in VARIANTS:
            raise ValueError(f"Unknown variant '{v}'. Choose from: {list(VARIANTS.keys())}")

    phases = []
    if args.phase in ("a", "both"):
        phases.append("a")
    if args.phase in ("b", "both"):
        phases.append("b")

    all_results = []

    for phase in phases:
        phase_dir = OUTPUT_BASE / f"phase_{phase}"
        logger.info("===== Phase %s =====", phase.upper())

        for variant_name in args.variants:
            for size in args.sizes:
                if phase == "a":
                    spacing = 1.0
                    action_scale = 1.0
                else:
                    # Phase B: spacing = 1/L, rescale actions by (1/L)^d
                    spacing = 1.0 / size
                    d = 2  # 2D lattice
                    action_scale = spacing**d

                result = run_single(
                    variant_name=variant_name,
                    variant_flags=VARIANTS[variant_name],
                    lattice_size=size,
                    spacing=spacing,
                    action_scale=action_scale,
                    training_config=training_config,
                    device=device,
                    output_dir=phase_dir,
                )
                result["phase"] = phase
                all_results.append(result)

                # Save incrementally
                summary_path = OUTPUT_BASE / "summary_results.json"
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                with open(summary_path, "w") as f:
                    json.dump(all_results, f, indent=2)

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Phase':<6} {'Variant':<18} {'Size':<8} {'Spacing':<10} {'Corr':<12} {'Loss':<12} {'Time':<8}")
    print("-" * 80)
    for r in all_results:
        if r["status"] == "completed":
            print(
                f"{r['phase']:<6} {r['variant']:<18} {r['lattice_size']}x{r['lattice_size']:<4} "
                f"{r['spacing']:<10.4f} {r['final_val_corr']:<12.6f} "
                f"{r['final_val_loss']:<12.6f} {r['training_time_s']:<8.1f}"
            )
        else:
            print(f"{r.get('phase', '?'):<6} {r['variant']:<18} {r['lattice_size']}x{r['lattice_size']:<4} SKIPPED")
    print("=" * 80)

    logger.info("Results saved to %s", OUTPUT_BASE / "summary_results.json")


if __name__ == "__main__":
    main()
