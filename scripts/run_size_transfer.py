"""Size-transfer experiment (task P1-6, bounded scope).

Trains at L=16 only and evaluates at L in {8, 32, 64} WITHOUT retraining,
for exactly two variants:
  (i)  "coords":   spacetime nodes carry absolute coordinates (as published);
  (ii) "constant": coordinate-free — spacetime features are a constant 1.0,
       geometry carried entirely by displacement edge features (restores
       translation invariance).
3 seeds each. One table; no further variants (plan: do not iterate).

Usage:
    python scripts/run_size_transfer.py [--epochs 150] [--seeds 0 1 2]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from qft_graph.config import LatticeConfig, ModelConfig
from qft_graph.fields.scalar import ScalarField
from qft_graph.graphs.builder import HeteroGraphBuilder
from qft_graph.lattice.hypercubic import HypercubicLattice
from qft_graph.models.hetero_gnn import HeteroGNN
from qft_graph.training.metrics import energy_correlation, relative_error
from qft_graph.utils.logging import setup_logging
from qft_graph.utils.reproducibility import set_seed
from qft_graph.utils.run_logging import log_run

from train_baselines import train_model  # shared training loop

TRAIN_L = 16
EVAL_SIZES = [8, 16, 32, 64]
VARIANTS = ["coords", "constant"]


def build_dataset(L: int, variant: str, data_dir: Path, n_configs: int | None = None):
    path = data_dir / f"phi4_{L}x{L}_m2=-0.5_lam=0.5" / "mc_data.pt"
    if not path.exists():
        return None
    mc = torch.load(path, weights_only=False)
    configs, actions = mc["configurations"], mc["actions"]
    if n_configs:
        configs, actions = configs[:n_configs], actions[:n_configs]
    lattice = HypercubicLattice(LatticeConfig(dimensions=(L, L)))
    builder = HeteroGraphBuilder(
        lattice, [ScalarField()], a_in_edges=True, spacetime_features=variant
    )
    return builder.build_dataset({"scalar": configs}, actions)


def main() -> None:
    parser = argparse.ArgumentParser(description="Size transfer (P1-6)")
    parser.add_argument("--data_dir", type=str, default="data/mc_configs")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str, default="results/size_transfer.json")
    args = parser.parse_args()

    logger = setup_logging()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)

    results = []
    for variant in VARIANTS:
        # Training data at L=16
        train_set = build_dataset(TRAIN_L, variant, data_dir)
        if train_set is None:
            raise SystemExit(f"Missing training data for L={TRAIN_L}")
        n_train = int(0.8 * len(train_set))
        train_loader = DataLoader(train_set[:n_train], batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(train_set[n_train:], batch_size=args.batch_size, shuffle=False)

        # Eval datasets at all sizes (validation slice only at L=16)
        eval_sets = {}
        for L in EVAL_SIZES:
            ds = build_dataset(L, variant, data_dir, n_configs=1000 if L != TRAIN_L else None)
            if ds is None:
                logger.warning("No data for L=%d, skipping", L)
                continue
            eval_sets[L] = ds[-1000:] if L == TRAIN_L else ds

        for seed in args.seeds:
            set_seed(seed)
            model_config = ModelConfig(a_in_edges=True, spacetime_features=variant)
            model = HeteroGNN(
                config=model_config,
                lattice_dim=2,
                field_types={"scalar": 1},
                lattice_spacing=1.0,
            )
            name = f"{variant}/s{seed}"
            logger.info("=" * 60)
            train_model(model, train_loader, val_loader,
                        args.epochs, args.lr, device, logger, name)

            model.eval()
            for L, ds in eval_sets.items():
                preds, trues = [], []
                with torch.no_grad():
                    for batch in DataLoader(ds, batch_size=16):
                        batch = batch.to(device)
                        preds.append(model(batch)["energy"].cpu().reshape(-1))
                        trues.append(batch.y.cpu().reshape(-1))
                preds, trues = torch.cat(preds), torch.cat(trues)
                r = energy_correlation(preds, trues)
                rel = relative_error(preds, trues)
                results.append({
                    "variant": variant, "seed": seed, "eval_L": L,
                    "pearson_r": round(r, 6), "relative_error": round(rel, 6),
                })
                logger.info("[%s] eval L=%d: r=%.5f rel_err=%.4f", name, L, r, rel)
            _save(args, results, logger)

    _save(args, results, logger, final=True)


def _save(args, results, logger, final=False):
    summary = []
    for variant in VARIANTS:
        for L in EVAL_SIZES:
            rows = [r for r in results if r["variant"] == variant and r["eval_L"] == L]
            if not rows:
                continue
            rs = [r["pearson_r"] for r in rows]
            res = [r["relative_error"] for r in rows]
            summary.append({
                "variant": variant, "eval_L": L, "n_seeds": len(rows),
                "r_mean": float(np.mean(rs)), "r_std": float(np.std(rs)),
                "rel_err_mean": float(np.mean(res)), "rel_err_std": float(np.std(res)),
            })
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_run": results,
                   "protocol": {"train_L": TRAIN_L, "epochs": args.epochs,
                                "seeds": args.seeds}}, f, indent=2)
    if final:
        logger.info("\n%-10s %8s %20s %s", "Variant", "eval L", "r (mean+/-std)", "rel err")
        for s in summary:
            logger.info("%-10s %8d   %.5f +/- %.5f   %.4f +/- %.4f",
                        s["variant"], s["eval_L"], s["r_mean"], s["r_std"],
                        s["rel_err_mean"], s["rel_err_std"])
        log_run("size_transfer_provenance",
                config={"train_L": TRAIN_L, "epochs": args.epochs, "seeds": args.seeds},
                metrics={"summary": summary})


if __name__ == "__main__":
    main()
