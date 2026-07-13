"""Over-smoothing diagnostic for the beta=4 depth failure (decision 5 note).

The A-4 v2 receptive-field study found depth HELPS at beta=2 (W4x4 r
monotone in B) but DEGRADES at beta=4 (peak at B=3; Q collapses to base
rate at B=6 for every seed) — protocol-independent. Working hypothesis:
over-smoothing — the ordered phase halves the input feature spread
(std cos theta_P: 0.40 -> 0.19) and deep stacks wash out what remains.

This script runs the ARCHITECTURE.md over-smoothing diagnostic on the
saved a4rf checkpoints: after each message-passing block, measure over
held-out configs (the test split the runs never trained on)
  (a) mean pairwise cosine similarity between spacetime-node embeddings
      within a graph (over-smoothing signature: -> 1 with depth), and
  (b) mean per-feature std across nodes (embedding spread).

Outputs: results/a4_oversmoothing.json, figures/a4_oversmoothing.{pdf,png}.

Usage:
    python scripts/diagnose_oversmoothing.py [--n_configs 50]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from qft_graph.graphs.edge_types import ADJACENT
from qft_graph.graphs.u1_dataset import build_u1_dataset
from qft_graph.models.u1_gnn import U1GaugeGNN
from qft_graph.utils.run_logging import log_run

BETAS = [2.0, 4.0]
DEPTHS = [2, 3, 4, 6]
SEEDS = [0, 1, 2]
CKPT_ROOT = Path("experiments/runs/u1")


def load_model(beta: float, B: int, seed: int) -> U1GaugeGNN:
    run_id = f"a4rf_invariant_oracle_u1_L16_beta{beta:g}_H64_B{B}_seed{seed}"
    ckpt = torch.load(
        CKPT_ROOT / run_id / "checkpoint_final.pt",
        map_location="cpu", weights_only=False,
    )
    model = U1GaugeGNN(
        ckpt["model_config"],
        variant=ckpt["variant"],
        wilson_loops=ckpt["wilson_loops"],
        predict_q=True,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model.eval()


@torch.no_grad()
def per_block_stats(model: U1GaugeGNN, graphs: list) -> tuple[np.ndarray, np.ndarray]:
    """Mean pairwise cosine similarity and feature spread after each block.

    Mirrors U1GaugeGNN.forward's encode stage, then steps through
    mp_blocks manually, recording spacetime embeddings block by block.
    Returns arrays of shape (n_blocks + 1,), index 0 = post-encoder.
    """
    n_blocks = len(model.mp_blocks)
    cos = np.zeros(n_blocks + 1)
    spread = np.zeros(n_blocks + 1)
    for g in graphs:
        data = g.clone()
        data["spacetime"].x = model.st_encoder(data["spacetime"].x)
        data[ADJACENT].edge_attr = model.edge_encoder(data[ADJACENT].edge_attr)

        def record(idx: int) -> None:
            x = data["spacetime"].x
            xn = torch.nn.functional.normalize(x, dim=-1)
            sim = xn @ xn.T  # (N, N)
            n = sim.shape[0]
            cos[idx] += ((sim.sum() - n) / (n * (n - 1))).item()
            spread[idx] += x.std(dim=0).mean().item()

        record(0)
        for b, block in enumerate(model.mp_blocks, start=1):
            data = block(data)
            record(b)
    return cos / len(graphs), spread / len(graphs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Over-smoothing diagnostic")
    parser.add_argument("--n_configs", type=int, default=50)
    args = parser.parse_args()

    results: dict[str, dict] = {}
    for beta in BETAS:
        # Same held-out region the training runs used (last 10% is test;
        # take configs from the tail, never seen in training)
        dataset = build_u1_dataset(
            f"data/u1_configs/u1_L16_beta{beta:g}.h5", "invariant_oracle"
        )
        graphs = dataset[-args.n_configs:]
        for B in DEPTHS:
            per_seed_cos, per_seed_spread = [], []
            for seed in SEEDS:
                model = load_model(beta, B, seed)
                cos, spread = per_block_stats(model, graphs)
                per_seed_cos.append(cos)
                per_seed_spread.append(spread)
            key = f"beta{beta:g}_B{B}"
            results[key] = {
                "beta": beta, "B": B,
                "cos_mean": np.mean(per_seed_cos, axis=0).tolist(),
                "cos_std": np.std(per_seed_cos, axis=0).tolist(),
                "cos_per_seed": [c.tolist() for c in per_seed_cos],
                "spread_mean": np.mean(per_seed_spread, axis=0).tolist(),
                "spread_per_seed": [s.tolist() for s in per_seed_spread],
            }
            final = results[key]["cos_mean"][-1]
            print(f"beta={beta:g} B={B}: post-encoder cos="
                  f"{results[key]['cos_mean'][0]:.3f} -> final block cos={final:.3f} "
                  f"(per seed: {[round(c[-1], 3) for c in per_seed_cos]})")

    # --- Figure: similarity trajectories ---
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), constrained_layout=True)
    colors = {2: "C0", 3: "C1", 4: "C2", 6: "C3"}
    for ax, beta in zip(axes, BETAS):
        for B in DEPTHS:
            r = results[f"beta{beta:g}_B{B}"]
            x = np.arange(len(r["cos_mean"]))
            ax.errorbar(x, r["cos_mean"], yerr=r["cos_std"],
                        marker="o", ms=3, capsize=2, color=colors[B],
                        label=f"B={B}")
        ax.set_xlabel("message-passing block (0 = encoder output)")
        ax.set_ylabel("mean pairwise cosine similarity")
        ax.set_ylim(0, 1.05)
        ax.axhline(1.0, color="gray", lw=0.5, ls="--")
        ax.set_title(f"Variant C spacetime embeddings, L=16, β={beta:g}")
        ax.legend(fontsize=8)

    out_dir = Path("figures")
    out_dir.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"a4_oversmoothing.{ext}", dpi=200)

    out_path = Path("results/a4_oversmoothing.json")
    with open(out_path, "w") as f:
        json.dump({"n_configs": args.n_configs, "results": results}, f, indent=2)

    log_run(
        "a4_oversmoothing_provenance",
        config={"n_configs": args.n_configs, "betas": BETAS, "depths": DEPTHS,
                "seeds": SEEDS, "protocol_version": 2},
        metrics={k: {"cos_final": r["cos_mean"][-1],
                     "cos_encoder": r["cos_mean"][0]}
                 for k, r in results.items()},
        extra={"figure": str(out_dir / "a4_oversmoothing.pdf"),
               "output_json": str(out_path)},
    )
    print(f"\n-> {out_path}, figures/a4_oversmoothing.pdf|png")


if __name__ == "__main__":
    main()
