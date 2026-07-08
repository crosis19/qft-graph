"""Depth ablation: over-smoothing study (task P1-5).

For B in {1, 2, 3, 4, 6} message-passing blocks, trains
  (a) Homogeneous GNN WITHOUT readout skip connection,
  (b) Homogeneous GNN WITH skip,
  (c) HeteroGNN,
3 seeds each, on one coupling at L=16, and plots Pearson r vs depth.
This measures the paper's central claim: homogeneous message passing
destroys the field signal without an ad hoc skip, while the bipartite
design is immune by construction. If variant (a) does NOT collapse
(r < 0.05 claimed in the manuscript), that discrepancy must be flagged.

Usage:
    python scripts/run_depth_ablation.py \
        --data data/mc_configs/phi4_16x16_m2=-0.5_lam=0.5/mc_data.pt \
        --config configs/paper/phase1_train_16x16.yaml
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from qft_graph.config import load_config
from qft_graph.fields.scalar import ScalarField
from qft_graph.graphs.builder import HeteroGraphBuilder
from qft_graph.lattice.hypercubic import HypercubicLattice
from qft_graph.models.baselines import HomogeneousGNN
from qft_graph.models.hetero_gnn import HeteroGNN
from qft_graph.utils.logging import setup_logging
from qft_graph.utils.reproducibility import set_seed
from qft_graph.utils.run_logging import log_run

from train_baselines import train_model  # shared training loop

DEPTHS = [1, 2, 3, 4, 6]
VARIANTS = ["homogeneous_no_skip", "homogeneous_skip", "hetero"]


def make_model(variant: str, depth: int, config, lattice, scalar_field):
    st_dim = lattice.dimension() if config.model.a_in_edges else lattice.dimension() + 1
    if variant == "hetero":
        model_config = copy.deepcopy(config.model)
        model_config.n_mp_blocks = depth
        return HeteroGNN(
            config=model_config,
            lattice_dim=lattice.dimension(),
            field_types={"scalar": scalar_field.node_feature_dim()},
            lattice_spacing=lattice.lattice_spacing(),
        )
    return HomogeneousGNN(
        lattice_dim=lattice.dimension(),
        hidden_dim=config.model.hidden_dim,
        n_mp_blocks=depth,
        n_encoder_layers=config.model.encoder_layers,
        lattice_spacing=lattice.lattice_spacing(),
        skip_connection=(variant == "homogeneous_skip"),
        st_feature_dim=st_dim,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Depth ablation (P1-5)")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--depths", type=int, nargs="+", default=DEPTHS)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str, default="results/depth_ablation.json")
    args = parser.parse_args()

    logger = setup_logging()
    config = load_config(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    mc_data = torch.load(args.data, weights_only=False)
    lattice = HypercubicLattice(config.lattice)
    scalar_field = ScalarField()
    builder = HeteroGraphBuilder(lattice, [scalar_field], a_in_edges=config.model.a_in_edges)
    dataset = builder.build_dataset(
        configurations={"scalar": mc_data["configurations"]},
        actions=mc_data["actions"],
    )
    n_train = int(0.8 * len(dataset))
    train_loader = DataLoader(dataset[:n_train], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset[n_train:], batch_size=args.batch_size, shuffle=False)

    results = []
    for depth in args.depths:
        for variant in VARIANTS:
            for seed in args.seeds:
                set_seed(seed)
                name = f"{variant}/B{depth}/s{seed}"
                logger.info("=" * 60)
                model = make_model(variant, depth, config, lattice, scalar_field)
                r = train_model(model, train_loader, val_loader,
                                args.epochs, args.lr, device, logger, name)
                r.update({"variant": variant, "depth": depth, "seed": seed})
                results.append(r)
                _save(args, results, logger)

    _save(args, results, logger, final=True)


def _save(args, results, logger, final=False):
    summary = []
    for depth in sorted({r["depth"] for r in results}):
        for variant in VARIANTS:
            rows = [r for r in results if r["depth"] == depth and r["variant"] == variant]
            if not rows:
                continue
            rs = [r["pearson_r"] for r in rows]
            summary.append({
                "variant": variant, "depth": depth, "n_seeds": len(rows),
                "r_mean": float(np.mean(rs)), "r_std": float(np.std(rs)),
                "rel_err_mean": float(np.mean([r["relative_error"] for r in rows])),
            })
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_run": results}, f, indent=2)
    if final:
        logger.info("\n%-22s %6s %20s", "Variant", "B", "r (mean+/-std)")
        for s in summary:
            logger.info("%-22s %6d   %.5f +/- %.5f",
                        s["variant"], s["depth"], s["r_mean"], s["r_std"])
        collapsed = [s for s in summary
                     if s["variant"] == "homogeneous_no_skip" and s["r_mean"] < 0.05]
        logger.info(
            "no-skip collapse (r<0.05) at depths: %s",
            [s["depth"] for s in collapsed] or "NONE — flag to Josh vs. manuscript claim",
        )
        log_run("depth_ablation_provenance",
                config={"epochs": args.epochs, "seeds": args.seeds,
                        "depths": args.depths, "data": args.data},
                metrics={"summary": summary})


if __name__ == "__main__":
    main()
