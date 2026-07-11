"""A-4 training: one U(1) graph variant on one (beta, L) ensemble.

Joint multi-task training of action + Wilson loops + Q per plan §3
Task A-4, protocol mirroring Phase I: AdamW, cosine, 150 epochs,
batch 32, H=64. Targets follow the decision record: Wilson targets are
raw per-config W, z-scored per (beta, size) for training stability
(decision 3); action targets are z-scored the same way for loss-scale
balance; Q is trained on its natural integer scale. All standardization
stats are saved and inverted for metrics on the raw scale.

Runs both comparison arms (decision 4):
  fixed H:            --hidden_dim 64
  parameter-matched:  --match_params_to link_nodes:64   (or an integer)

Usage (pilot, laptop CPU):
  python scripts/train_u1.py --data data/u1_configs/u1_L8_beta2.h5 \
      --variant link_nodes --seeds 0 --epochs 150

Usage (Colab, full protocol):
  python scripts/train_u1.py --data ... --variant A --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from qft_graph.config import ModelConfig, resolve_device
from qft_graph.graphs.u1_dataset import build_u1_dataset, standardize_wilson_targets
from qft_graph.models.u1_gnn import (
    DEFAULT_WILSON_LOOPS,
    U1GaugeGNN,
    matched_hidden_dim,
    u1_param_count,
)
from qft_graph.training.metrics import (
    energy_correlation,
    q_rounded_accuracy,
    relative_error,
)
from qft_graph.utils.logging import setup_logging
from qft_graph.utils.reproducibility import set_seed
from qft_graph.utils.run_logging import log_run


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A-4 U(1) variant training")
    p.add_argument("--data", type=str, required=True, help="u1_configs .h5 file")
    p.add_argument("--variant", type=str, required=True,
                   help="link_nodes/edge_features/invariant_oracle or A/B/C")
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--match_params_to", type=str, default=None,
                   help="Parameter-matched arm: 'variant:H' (e.g. link_nodes:64) "
                        "or an integer budget; overrides --hidden_dim")
    p.add_argument("--n_mp_blocks", type=int, default=3,
                   help="Message-passing blocks B (receptive-field study knob)")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--w_action", type=float, default=1.0)
    p.add_argument("--w_wilson", type=float, default=1.0)
    p.add_argument("--w_q", type=float, default=1.0)
    p.add_argument("--wilson_loops", type=str, nargs="+",
                   default=list(DEFAULT_WILSON_LOOPS))
    p.add_argument("--n_configs", type=int, default=None)
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--run_prefix", type=str, default="u1a4",
                   help="results/<prefix>_... run id prefix")
    return p.parse_args()


def resolve_hidden_dim(args, config: ModelConfig, model_kwargs: dict) -> tuple[int, dict]:
    """Fixed-H or parameter-matched H, plus provenance for the run record."""
    if args.match_params_to is None:
        return args.hidden_dim, {"arm": "fixed_h"}
    spec = args.match_params_to
    if ":" in spec:
        ref_variant, ref_h = spec.split(":")
        from dataclasses import replace
        target = u1_param_count(
            replace(config, hidden_dim=int(ref_h)), ref_variant, **model_kwargs
        )
    else:
        target = int(spec)
    h = matched_hidden_dim(target, config, args.variant, **model_kwargs)
    return h, {"arm": "param_matched", "match_spec": spec, "target_params": target}


def split_dataset(dataset: list, train_frac: float, val_frac: float):
    n = len(dataset)
    n_train, n_val = int(train_frac * n), int(val_frac * n)
    return (
        dataset[:n_train],
        dataset[n_train : n_train + n_val],
        dataset[n_train + n_val :],
    )


def standardize_actions(split: list, stats: tuple[float, float] | None = None):
    """Z-score data.y in place (train stats reused on val/test); returns stats."""
    y = torch.cat([d.y for d in split])
    if stats is None:
        stats = (y.mean().item(), y.std().item())
    mean, std = stats
    if not (np.isfinite(std) and std > 0):
        raise ValueError(f"Degenerate action std {std}")
    for d in split:
        d.y = (d.y - mean) / std
    return stats


@torch.no_grad()
def evaluate(model, split, loops, a_stats, batch_size, device) -> dict:
    """Metrics on the raw scale (standardization inverted where needed)."""
    model.eval()
    preds: dict[str, list] = {}
    trues: dict[str, list] = {}
    for batch in DataLoader(split, batch_size=batch_size):
        batch = batch.to(device)
        out = model(batch)
        for i, name in enumerate(loops):
            preds.setdefault(f"w_{name}", []).append(out[f"wilson_{name}"].cpu())
            trues.setdefault(f"w_{name}", []).append(batch.y_wilson[:, i].cpu())
        preds.setdefault("action", []).append(out["energy"].cpu())
        trues.setdefault("action", []).append(batch.y.cpu())
        preds.setdefault("q", []).append(out["q"].cpu())
        trues.setdefault("q", []).append(batch.y_q.cpu())
    P = {k: torch.cat(v) for k, v in preds.items()}
    T = {k: torch.cat(v) for k, v in trues.items()}

    a_mean, a_std = a_stats
    action_pred_raw = P["action"] * a_std + a_mean
    action_true_raw = T["action"] * a_std + a_mean
    metrics = {
        "action_r": energy_correlation(P["action"], T["action"]),
        "action_rel_err": relative_error(action_pred_raw, action_true_raw),
        "q_r": energy_correlation(P["q"], T["q"]),
        "q_acc": q_rounded_accuracy(P["q"], T["q"]),
    }
    for name in loops:
        metrics[f"wilson_{name}_r"] = energy_correlation(P[f"w_{name}"], T[f"w_{name}"])
    return metrics


def train_one_seed(args, seed, logger) -> dict:
    set_seed(seed)
    device = resolve_device(args.device)
    loops = tuple(args.wilson_loops)

    dataset = build_u1_dataset(
        args.data, args.variant, wilson_loops=loops, n_configs=args.n_configs
    )
    train, val, test = split_dataset(dataset, args.train_frac, args.val_frac)

    # Target standardization: train stats, reused on val/test (decision 3)
    w_stats = standardize_wilson_targets(train)
    standardize_wilson_targets(val, stats=w_stats)
    standardize_wilson_targets(test, stats=w_stats)
    a_stats = standardize_actions(train)
    standardize_actions(val, stats=a_stats)
    standardize_actions(test, stats=a_stats)

    config = ModelConfig(hidden_dim=args.hidden_dim, n_mp_blocks=args.n_mp_blocks)
    model_kwargs = dict(wilson_loops=loops, predict_q=True)
    hidden_dim, arm_info = resolve_hidden_dim(args, config, model_kwargs)
    config = ModelConfig(hidden_dim=hidden_dim, n_mp_blocks=args.n_mp_blocks)

    model = U1GaugeGNN(config, variant=args.variant, **model_kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "seed %d: %s H=%d (%s) %d params, %d/%d/%d train/val/test on %s",
        seed, args.variant, hidden_dim, arm_info["arm"], n_params,
        len(train), len(val), len(test), device,
    )

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)

    history = []
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            out = model(batch)
            loss = args.w_action * (out["energy"] - batch.y).pow(2).mean()
            w_loss = sum(
                (out[f"wilson_{name}"] - batch.y_wilson[:, i]).pow(2).mean()
                for i, name in enumerate(loops)
            ) / len(loops)
            loss = loss + args.w_wilson * w_loss
            loss = loss + args.w_q * (out["q"] - batch.y_q).pow(2).mean()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        sched.step()

        if epoch % 5 == 4 or epoch == 0 or epoch == args.epochs - 1:
            m = evaluate(model, val, loops, a_stats, args.batch_size, device)
            m["epoch"] = epoch + 1
            m["train_loss"] = epoch_loss / max(1, len(loader))
            m["elapsed_s"] = time.time() - t0
            history.append(m)
            logger.info(
                "  epoch %3d: loss %.4f | val action r=%.4f | W1x1 r=%.4f | q acc=%.2f",
                epoch + 1, m["train_loss"], m["action_r"],
                m.get("wilson_1x1_r", float("nan")), m["q_acc"],
            )

    test_metrics = evaluate(model, test, loops, a_stats, args.batch_size, device)
    run_id = (
        f"{args.run_prefix}_{args.variant}_{Path(args.data).stem}"
        f"_H{hidden_dim}_B{args.n_mp_blocks}_seed{seed}"
    )

    ckpt_dir = Path("experiments/runs/u1") / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": config,
            "variant": args.variant,
            "wilson_loops": loops,
            "hidden_dim": hidden_dim,
            "n_mp_blocks": args.n_mp_blocks,
            "a_stats": a_stats,
            "w_stats": w_stats,
            "seed": seed,
        },
        ckpt_dir / "checkpoint_final.pt",
    )

    log_run(
        run_id,
        config={**vars(args), "hidden_dim_resolved": hidden_dim, **arm_info},
        metrics={"test": test_metrics, "history": history},
        extra={
            "n_params": n_params,
            "a_stats": a_stats,
            "w_stats": w_stats,
            "checkpoint": str(ckpt_dir / "checkpoint_final.pt"),
            "wall_time_s": time.time() - t0,
        },
    )
    logger.info(
        "seed %d done in %.1f min -> results/%s.json | test action r=%.4f",
        seed, (time.time() - t0) / 60, run_id, test_metrics["action_r"],
    )
    return test_metrics


def main() -> None:
    args = parse_args()
    logger = setup_logging()
    for seed in args.seeds:
        train_one_seed(args, seed, logger)


if __name__ == "__main__":
    main()
