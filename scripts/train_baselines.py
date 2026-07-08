"""Train baseline models for architecture comparison (Tables I & II).

Trains HeteroGNN, Homogeneous GNN, Lattice CNN, parameter-matched CNN, and
MLP on the same MC data with the same protocol, over multiple seeds
(task P1-4: mean +/- std columns).

Usage:
    python scripts/train_baselines.py \
        --data data/mc_configs/phi4_16x16_m2=-0.5_lam=0.5/mc_data.pt \
        --config configs/paper/phase1_train_16x16.yaml \
        --seeds 0 1 2 3 4 [--epochs 150]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from qft_graph.config import load_config
from qft_graph.fields.scalar import ScalarField
from qft_graph.graphs.builder import HeteroGraphBuilder
from qft_graph.lattice.hypercubic import HypercubicLattice
from qft_graph.models.baselines import HomogeneousGNN, LatticeCNN, MLPBaseline
from qft_graph.models.hetero_gnn import HeteroGNN
from qft_graph.training.losses import EnergyMatchingLoss
from qft_graph.training.metrics import energy_correlation, relative_error
from qft_graph.utils.logging import setup_logging
from qft_graph.utils.reproducibility import set_seed
from qft_graph.utils.run_logging import log_run


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: str,
    logger: logging.Logger,
    model_name: str,
) -> dict:
    """Train a model and return final metrics."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = EnergyMatchingLoss()

    n_params = sum(p.numel() for p in model.parameters())

    best_corr = -1.0
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output["energy"], batch.y.to(device))
            loss.backward()
            optimizer.step()

        model.eval()
        all_pred, all_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                all_pred.append(model(batch)["energy"].cpu())
                all_true.append(batch.y.cpu())
        preds, trues = torch.cat(all_pred), torch.cat(all_true)
        corr = energy_correlation(preds, trues)
        rel_err = relative_error(preds, trues)
        best_corr = max(best_corr, corr)
        scheduler.step()

        if epoch % 50 == 0 or epoch == 1:
            logger.info("[%s] Epoch %d/%d | r=%.5f | RelErr=%.4f",
                        model_name, epoch, epochs, corr, rel_err)

    elapsed = time.time() - start
    logger.info("[%s] Done in %.1fs. Final r=%.5f, rel_err=%.4f",
                model_name, elapsed, corr, rel_err)
    return {
        "model": model_name,
        "n_params": n_params,
        "pearson_r": round(corr, 6),
        "relative_error": round(rel_err, 6),
        "best_pearson_r": round(best_corr, 6),
        "train_time_s": round(elapsed, 1),
    }


def matched_cnn_channels(lattice_dims: tuple, target_params: int) -> int:
    """Channel count giving a LatticeCNN closest to target_params."""
    best_c, best_diff = 32, float("inf")
    for c in range(32, 161, 4):
        model = LatticeCNN(lattice_dims=lattice_dims, hidden_channels=c, n_conv_layers=4)
        n = sum(p.numel() for p in model.parameters())
        if abs(n - target_params) < best_diff:
            best_diff, best_c = abs(n - target_params), c
    return best_c


def model_factories(config, lattice, scalar_field, dims, n_sites):
    """Fresh-model factories, one per architecture (new weights per seed)."""
    st_dim = lattice.dimension() if config.model.a_in_edges else lattice.dimension() + 1

    def hetero():
        return HeteroGNN(
            config=config.model,
            lattice_dim=lattice.dimension(),
            field_types={"scalar": scalar_field.node_feature_dim()},
            lattice_spacing=lattice.lattice_spacing(),
        )

    hetero_params = sum(p.numel() for p in hetero().parameters())
    cnn_matched_c = matched_cnn_channels(tuple(dims), hetero_params)

    return {
        "HeteroGNN": hetero,
        "HomogeneousGNN": lambda: HomogeneousGNN(
            lattice_dim=lattice.dimension(),
            hidden_dim=config.model.hidden_dim,
            n_mp_blocks=config.model.n_mp_blocks,
            n_encoder_layers=config.model.encoder_layers,
            lattice_spacing=lattice.lattice_spacing(),
            st_feature_dim=st_dim,
        ),
        "LatticeCNN": lambda: LatticeCNN(
            lattice_dims=tuple(dims), hidden_channels=32, n_conv_layers=4,
            lattice_spacing=lattice.lattice_spacing(),
        ),
        "LatticeCNN-matched": lambda: LatticeCNN(
            lattice_dims=tuple(dims), hidden_channels=cnn_matched_c, n_conv_layers=4,
            lattice_spacing=lattice.lattice_spacing(),
        ),
        "MLP": lambda: MLPBaseline(n_sites=n_sites, hidden_dim=256, n_layers=3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline models (multi-seed)")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Subset of models to train (default: all)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str, default="results/baseline_results_v2.json")
    args = parser.parse_args()

    logger = setup_logging()
    config = load_config(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading MC data from %s", args.data)
    mc_data = torch.load(args.data, weights_only=False)
    configurations = mc_data["configurations"]
    actions = mc_data["actions"]

    lattice = HypercubicLattice(config.lattice)
    scalar_field = ScalarField()
    builder = HeteroGraphBuilder(lattice, [scalar_field], a_in_edges=config.model.a_in_edges)
    dataset = builder.build_dataset(
        configurations={"scalar": configurations}, actions=actions
    )
    n_train = int(0.8 * len(dataset))
    train_loader = DataLoader(dataset[:n_train], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset[n_train:], batch_size=args.batch_size, shuffle=False)
    logger.info("Train: %d, Val: %d", n_train, len(dataset) - n_train)

    dims = config.lattice.dimensions
    factories = model_factories(config, lattice, scalar_field, dims, dims[0] * dims[1])
    if args.models:
        factories = {k: v for k, v in factories.items() if k in args.models}

    per_seed: list[dict] = []
    for seed in args.seeds:
        for name, factory in factories.items():
            set_seed(seed)
            logger.info("=" * 60)
            logger.info("Training %s (seed %d)", name, seed)
            r = train_model(factory(), train_loader, val_loader,
                            args.epochs, args.lr, device, logger, f"{name}/s{seed}")
            r["seed"] = seed
            r["model"] = name
            per_seed.append(r)
            _save(args, per_seed, logger)  # incremental

    _save(args, per_seed, logger, final=True)


def _save(args, per_seed, logger, final=False):
    models = sorted({r["model"] for r in per_seed})
    summary = []
    for name in models:
        rows = [r for r in per_seed if r["model"] == name]
        rs = [r["pearson_r"] for r in rows]
        res = [r["relative_error"] for r in rows]
        summary.append({
            "model": name,
            "n_params": rows[0]["n_params"],
            "n_seeds": len(rows),
            "r_mean": float(np.mean(rs)),
            "r_std": float(np.std(rs)),
            "rel_err_mean": float(np.mean(res)),
            "rel_err_std": float(np.std(res)),
        })

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_seed": per_seed,
                   "protocol": {"epochs": args.epochs, "batch_size": args.batch_size,
                                "lr": args.lr, "seeds": args.seeds,
                                "data": args.data}}, f, indent=2)

    if final:
        logger.info("\n%-20s %10s %22s %s", "Model", "Params", "r (mean+/-std)", "rel err")
        for s in summary:
            logger.info("%-20s %10d   %.5f +/- %.5f   %.4f +/- %.4f",
                        s["model"], s["n_params"], s["r_mean"], s["r_std"],
                        s["rel_err_mean"], s["rel_err_std"])
        log_run("baselines_v2_provenance",
                config={"epochs": args.epochs, "seeds": args.seeds, "data": args.data},
                metrics={"summary": summary},
                extra={"output_json": str(out_path)})


if __name__ == "__main__":
    main()
