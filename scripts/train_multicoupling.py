"""Multi-coupling training and Table III' evaluation (task P1-3).

Trains a coupling-conditioned HeteroGNN (field nodes carry [phi, m2, lambda],
readout conditioned on (m2, lambda)) jointly on alternating m^2 grid points
and evaluates on the held-out couplings:

    grid indices 1,3,5,7,9,11   -> training couplings
    grid indices 2,4,6,8,10     -> interpolation holdouts
    grid indices 0,12           -> extrapolation holdouts (both endpoints)

Produces Table III' — Pearson r and relative error, mean +/- std over seeds,
per held-out coupling — as JSON plus copy-paste LaTeX rows.

Usage:
    python scripts/train_multicoupling.py --seeds 0 1 2 3 4 [--epochs 150]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from qft_graph.config import LatticeConfig, ModelConfig, TrainingConfig
from qft_graph.fields.scalar import ScalarField
from qft_graph.graphs.builder import HeteroGraphBuilder
from qft_graph.lattice.hypercubic import HypercubicLattice
from qft_graph.models.hetero_gnn import HeteroGNN
from qft_graph.training.metrics import energy_correlation, relative_error
from qft_graph.training.trainer import Trainer
from qft_graph.utils.logging import setup_logging
from qft_graph.utils.reproducibility import set_seed
from qft_graph.utils.run_logging import log_run

M2_GRID = np.round(np.linspace(-2.9, -0.3, 13), 4)
TRAIN_IDX = [1, 3, 5, 7, 9, 11]
INTERP_IDX = [2, 4, 6, 8, 10]
EXTRAP_IDX = [0, 12]


def load_point(data_dir: Path, L: int, m2: float, lam: float):
    dirname = f"phi4_{L}x{L}_m2={m2:g}_lam={lam:g}"
    path = data_dir / dirname / "mc_data.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run scripts/generate_multicoupling_data.py first"
        )
    return torch.load(path, weights_only=False, map_location="cpu")


def build_graphs(lattice, m2, lam, configs, actions):
    field = ScalarField(couplings=(float(m2), float(lam)))
    builder = HeteroGraphBuilder(lattice, [field])
    return builder.build_dataset({"scalar": configs}, actions)


def evaluate(model, graphs, batch_size=64):
    from torch_geometric.loader import DataLoader

    model.eval()
    device = next(model.parameters()).device
    preds, trues = [], []
    with torch.no_grad():
        for batch in DataLoader(graphs, batch_size=batch_size):
            batch = batch.to(device)
            out = model(batch)
            preds.append(out["energy"].reshape(-1).cpu())
            trues.append(batch.y.reshape(-1).cpu())
    preds, trues = torch.cat(preds), torch.cat(trues)
    return energy_correlation(preds, trues), relative_error(preds, trues)


def main() -> None:
    parser = argparse.ArgumentParser(description="Table III' multi-coupling training")
    parser.add_argument("--lattice_size", type=int, default=16)
    parser.add_argument("--coupling", type=float, default=0.5)
    parser.add_argument("--data_dir", type=str, default="data/mc_configs")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_per_coupling", type=int, default=3000,
                        help="Configs used per training coupling")
    parser.add_argument("--n_eval", type=int, default=1000,
                        help="Held-out configs evaluated per coupling")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str, default="results/table3prime.json")
    args = parser.parse_args()

    logger = setup_logging()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    L, lam = args.lattice_size, args.coupling
    lattice = HypercubicLattice(LatticeConfig(dimensions=(L, L)))
    data_dir = Path(args.data_dir)

    # --- Load all points once ---
    raw = {float(m2): load_point(data_dir, L, m2, lam) for m2 in M2_GRID}

    # Evaluation graphs: last n_eval configs of every coupling (never trained on)
    eval_graphs = {
        m2: build_graphs(
            lattice, m2, lam,
            raw[m2]["configurations"][-args.n_eval:],
            raw[m2]["actions"][-args.n_eval:],
        )
        for m2 in map(float, M2_GRID)
    }

    # Resume: seeds already in the output JSON are skipped entirely; a seed
    # whose training finished but whose evaluation crashed is re-evaluated
    # from its saved checkpoint instead of retrained.
    per_seed: dict[int, dict] = {}
    out_path = Path(args.output)
    if out_path.exists():
        with open(out_path) as f:
            prev = json.load(f).get("per_seed", {})
        per_seed = {int(k): v for k, v in prev.items()}
        if per_seed:
            logger.info("Resuming: seeds %s already complete", sorted(per_seed))

    n_params = None
    for seed in args.seeds:
        if seed in per_seed:
            logger.info("Skipping seed %d (already in %s)", seed, out_path)
            continue
        t0 = time.time()
        set_seed(seed)

        # Training set: first n_per_coupling configs of each training coupling
        train_graphs, val_graphs = [], []
        for idx in TRAIN_IDX:
            m2 = float(M2_GRID[idx])
            configs = raw[m2]["configurations"][: args.n_per_coupling]
            actions = raw[m2]["actions"][: args.n_per_coupling]
            graphs = build_graphs(lattice, m2, lam, configs, actions)
            n_train = int(0.9 * len(graphs))
            train_graphs.extend(graphs[:n_train])
            val_graphs.extend(graphs[n_train:])

        model_config = ModelConfig(condition_on_couplings=True)
        model = HeteroGNN(
            config=model_config,
            lattice_dim=lattice.dimension(),
            field_types={"scalar": 3},
            lattice_spacing=lattice.lattice_spacing(),
            global_dim=2,
        )
        n_params = sum(p.numel() for p in model.parameters())

        experiment_dir = Path(f"experiments/runs/multicoupling/seed{seed}")
        final_ckpt = experiment_dir / "checkpoint_final.pt"
        if final_ckpt.exists():
            # Training finished previously but evaluation didn't get saved:
            # reuse the checkpoint instead of retraining.
            logger.info("seed %d: loading finished checkpoint %s", seed, final_ckpt)
            ckpt = torch.load(final_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            model = model.to(device)
        else:
            training_config = TrainingConfig(
                epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=seed
            )
            trainer = Trainer(
                model, train_graphs, val_graphs, training_config,
                experiment_dir=str(experiment_dir),
                device=device,
            )
            logger.info("seed %d: training on %d graphs (%d params)...",
                        seed, len(train_graphs), n_params)
            trainer.train()

        results = {}
        for m2 in map(float, M2_GRID):
            r, rel = evaluate(model, eval_graphs[m2])
            results[f"{m2:g}"] = {"pearson_r": r, "relative_error": rel}
        per_seed[seed] = results
        logger.info("seed %d done in %.1f min", seed, (time.time() - t0) / 60)

        # Incremental save so partial runs are never lost
        _write_output(args, per_seed, n_params, logger)

    _write_output(args, per_seed, n_params, logger, final=True)


def _write_output(args, per_seed, n_params, logger, final=False):
    # Aggregate mean +/- std over seeds
    table = []
    for i, m2 in enumerate(map(float, M2_GRID)):
        key = f"{m2:g}"
        rs = [per_seed[s][key]["pearson_r"] for s in per_seed]
        res = [per_seed[s][key]["relative_error"] for s in per_seed]
        role = ("train" if i in TRAIN_IDX
                else "interpolation" if i in INTERP_IDX else "extrapolation")
        table.append({
            "m2": m2,
            "role": role,
            "r_mean": float(np.mean(rs)),
            "r_std": float(np.std(rs)),
            "rel_err_mean": float(np.mean(res)),
            "rel_err_std": float(np.std(res)),
        })

    out = {
        "protocol": {
            "m2_grid": [float(x) for x in M2_GRID],
            "train_idx": TRAIN_IDX,
            "interp_idx": INTERP_IDX,
            "extrap_idx": EXTRAP_IDX,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "n_per_coupling": args.n_per_coupling,
            "n_eval": args.n_eval,
            "seeds": sorted(per_seed.keys()),
            "n_params": n_params,
        },
        "table": table,
        "per_seed": {str(s): per_seed[s] for s in per_seed},
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    if final:
        logger.info("\n%-10s %-14s %-22s %s", "m^2", "role", "r", "rel. err")
        for row in table:
            logger.info(
                "%-10g %-14s %.4f +/- %.4f      %.4f +/- %.4f",
                row["m2"], row["role"], row["r_mean"], row["r_std"],
                row["rel_err_mean"], row["rel_err_std"],
            )
        # LaTeX rows for Table III'
        logger.info("\nLaTeX rows (Table III'):")
        for row in table:
            mark = {"train": r"$^\dagger$", "interpolation": "", "extrapolation": r"$^\ast$"}[row["role"]]
            logger.info(
                "$%g$%s & $%.4f \\pm %.4f$ & $%.2f \\pm %.2f\\%%$ \\\\",
                row["m2"], mark, row["r_mean"], row["r_std"],
                100 * row["rel_err_mean"], 100 * row["rel_err_std"],
            )
        log_run(
            "table3prime_provenance",
            config=out["protocol"],
            metrics={"table": table},
            extra={"output_json": str(out_path)},
        )


if __name__ == "__main__":
    main()
