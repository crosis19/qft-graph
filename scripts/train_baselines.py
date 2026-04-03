"""Train baseline models for architecture comparison.

Trains the Homogeneous GNN, Lattice CNN, and MLP baselines on the same
MC data used for the HeteroGNN, enabling a controlled comparison.

Usage:
    python scripts/train_baselines.py \
        --data data/mc_configs/phi4_16x16_m2=-0.5_lam=0.5/mc_data.pt \
        [--epochs 150] [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from qft_graph.config import LatticeConfig, load_config
from qft_graph.fields.scalar import ScalarField
from qft_graph.graphs.builder import HeteroGraphBuilder
from qft_graph.lattice.hypercubic import HypercubicLattice
from qft_graph.models.baselines import HomogeneousGNN, LatticeCNN, MLPBaseline
from qft_graph.models.hetero_gnn import HeteroGNN
from qft_graph.training.losses import EnergyMatchingLoss
from qft_graph.training.metrics import energy_correlation, relative_error
from qft_graph.utils.logging import setup_logging
from qft_graph.utils.reproducibility import set_seed


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
    logger.info("[%s] Parameters: %d", model_name, n_params)

    best_corr = -1.0
    start = time.time()

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output["energy"], batch.y.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        train_loss = total_loss / max(n_batches, 1)

        # Validate
        model.eval()
        all_pred, all_true = [], []
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                output = model(batch)
                pred = output["energy"]
                target = batch.y.to(device)
                val_loss_sum += criterion(pred, target).item()
                val_batches += 1
                all_pred.append(pred.cpu())
                all_true.append(target.cpu())

        preds = torch.cat(all_pred)
        trues = torch.cat(all_true)
        corr = energy_correlation(preds, trues)
        rel_err = relative_error(preds, trues)
        val_loss = val_loss_sum / max(val_batches, 1)

        if corr > best_corr:
            best_corr = corr

        scheduler.step()

        if epoch % 25 == 0 or epoch == 1:
            logger.info(
                "[%s] Epoch %d/%d | Train: %.6f | Val: %.6f | r=%.4f | RelErr=%.4f",
                model_name, epoch, epochs, train_loss, val_loss, corr, rel_err,
            )

    elapsed = time.time() - start
    logger.info("[%s] Done in %.1fs. Final r=%.4f, rel_err=%.4f",
                model_name, elapsed, corr, rel_err)

    return {
        "model": model_name,
        "n_params": n_params,
        "pearson_r": round(corr, 6),
        "relative_error": round(rel_err, 6),
        "best_pearson_r": round(best_corr, 6),
        "train_time_s": round(elapsed, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline models")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str, default="experiments/baseline_results.json")
    args = parser.parse_args()

    logger = setup_logging()
    set_seed(42)

    config = load_config(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load MC data
    logger.info("Loading MC data from %s", args.data)
    mc_data = torch.load(args.data, weights_only=False)
    configurations = mc_data["configurations"]
    actions = mc_data["actions"]

    # Build graph dataset (shared by all models)
    lattice = HypercubicLattice(config.lattice)
    scalar_field = ScalarField()
    builder = HeteroGraphBuilder(lattice, [scalar_field], a_in_edges=config.model.a_in_edges)
    dataset = builder.build_dataset(
        configurations={"scalar": configurations},
        actions=actions,
    )

    # Train/val split
    n_train = int(0.8 * len(dataset))
    train_dataset = dataset[:n_train]
    val_dataset = dataset[n_train:]
    logger.info("Train: %d, Val: %d", len(train_dataset), len(val_dataset))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    dims = config.lattice.dimensions
    n_sites = dims[0] * dims[1]
    results = []

    # --- 1. HeteroGNN (our model) ---
    logger.info("=" * 60)
    logger.info("Training HeteroGNN (ours)")
    hetero_model = HeteroGNN(
        config=config.model,
        lattice_dim=lattice.dimension(),
        field_types={"scalar": scalar_field.dof_per_site()},
        lattice_spacing=lattice.lattice_spacing(),
    )
    r = train_model(hetero_model, train_loader, val_loader,
                    args.epochs, args.lr, device, logger, "HeteroGNN")
    results.append(r)

    # --- 2. Homogeneous GNN baseline ---
    logger.info("=" * 60)
    logger.info("Training HomogeneousGNN baseline")
    homo_model = HomogeneousGNN(
        lattice_dim=lattice.dimension(),
        hidden_dim=config.model.hidden_dim,
        n_mp_blocks=config.model.n_mp_blocks,
        n_encoder_layers=config.model.encoder_layers,
        lattice_spacing=lattice.lattice_spacing(),
    )
    r = train_model(homo_model, train_loader, val_loader,
                    args.epochs, args.lr, device, logger, "HomogeneousGNN")
    results.append(r)

    # --- 3. Lattice CNN baseline ---
    logger.info("=" * 60)
    logger.info("Training LatticeCNN baseline")
    cnn_model = LatticeCNN(
        lattice_dims=tuple(dims),
        hidden_channels=32,
        n_conv_layers=4,
        lattice_spacing=lattice.lattice_spacing(),
    )
    r = train_model(cnn_model, train_loader, val_loader,
                    args.epochs, args.lr, device, logger, "LatticeCNN")
    results.append(r)

    # --- 4. MLP baseline ---
    logger.info("=" * 60)
    logger.info("Training MLP baseline")
    mlp_model = MLPBaseline(
        n_sites=n_sites,
        hidden_dim=256,
        n_layers=3,
    )
    r = train_model(mlp_model, train_loader, val_loader,
                    args.epochs, args.lr, device, logger, "MLP")
    results.append(r)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", output_path)

    # Print summary table
    logger.info("\n" + "=" * 70)
    logger.info("BASELINE COMPARISON RESULTS")
    logger.info("=" * 70)
    logger.info("%-20s %10s %10s %12s", "Model", "Params", "Pearson r", "Rel. Error")
    logger.info("-" * 70)
    for r in results:
        logger.info("%-20s %10d %10.4f %12.4f",
                    r["model"], r["n_params"], r["pearson_r"], r["relative_error"])


if __name__ == "__main__":
    main()
