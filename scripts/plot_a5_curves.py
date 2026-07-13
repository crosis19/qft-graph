"""A-5 deliverable: augmentation curves (plan §3, Task A-5 done-when).

Test accuracy (top row) and eps_gauge (bottom row) vs training-set size
at the L=8 beta=2 cell: Variants A and B with and without on-the-fly
gauge augmentation, plus the Variant C ceiling (no augmentation). The
question being answered (plan A-5): does augmentation drive learned
invariance to the Variant-C ceiling cheaply?

Data sources (all committed run records):
- training runs: results/a5aug_*.json / a5base_*.json / a5C_*.json from
  notebook 09. The full-size (n_train=3200) no-augmentation points reuse
  results/a4null_*.json (A/B) and a4C_*.json (C) — the protocols
  coincide there, so those runs are not repeated.
- eps_gauge: results/a5eps_<source_run_id>.json from
  scripts/measure_gauge_invariance.py.

Writes figures/a5_augmentation.{pdf,png} + results/a5_curves.json.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qft_graph.utils.run_logging import log_run

# The experiment cell (notebook 09 F1) and its full train-split size
CELL = "u1_L8_beta2"
FULL_N_TRAIN = 3200

# arm label -> (training-record glob patterns, plot style)
ARMS = {
    "A + aug": (["a5aug_link_nodes_*"], dict(color="#d62728", marker="o", ls="-")),
    "A": (
        ["a5base_link_nodes_*", "a4null_link_nodes_*"],
        dict(color="#d62728", marker="o", ls="--", alpha=0.6),
    ),
    "B + aug": (["a5aug_edge_features_*"], dict(color="#1f77b4", marker="s", ls="-")),
    "B": (
        ["a5base_edge_features_*", "a4null_edge_features_*"],
        dict(color="#1f77b4", marker="s", ls="--", alpha=0.6),
    ),
    "C (oracle)": (
        ["a5C_invariant_oracle_*", "a4C_invariant_oracle_*"],
        dict(color="#2ca02c", marker="^", ls=":"),
    ),
}

# (panel title, test-metric key, eps-metric key); "energy" is the action
# head (CLAUDE.md terminology note)
PANELS = [
    ("action", "action_r", "energy"),
    ("W 1x1", "wilson_1x1_r", "wilson_1x1"),
    ("W 4x4", "wilson_4x4_r", "wilson_4x4"),
    ("Q", "q_r", "q"),
]
N_TRAIN_RE = re.compile(r"_n(\d+)_seed\d+$")
EPS_FLOOR = 1e-4  # exact zeros drawn at the axis floor on the log scale


def n_train_of(run_id: str) -> int:
    m = N_TRAIN_RE.search(run_id)
    return int(m.group(1)) if m else FULL_N_TRAIN


def collect_arm(patterns: list[str]) -> dict[int, dict[str, list]]:
    """n_train -> {metric_key: [per-seed values]} for one arm, joining the
    training record (test metrics) with its a5eps record (eps metrics)."""
    by_n: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for pat in patterns:
        for path in sorted(Path("results").glob(f"{pat}.json")):
            rec = json.loads(path.read_text())
            if "run_id" not in rec or "test" not in rec.get("metrics", {}):
                continue
            if CELL not in rec["run_id"]:
                continue
            n = n_train_of(rec["run_id"])
            for _, r_key, _ in PANELS:
                by_n[n][r_key].append(rec["metrics"]["test"][r_key])
            eps_path = Path("results") / f"a5eps_{rec['run_id']}.json"
            if eps_path.exists():
                eps_rec = json.loads(eps_path.read_text())
                for _, _, e_key in PANELS:
                    by_n[n][f"eps_{e_key}"].append(
                        eps_rec["metrics"][e_key]["eps_gauge"]
                    )
    return by_n


def series(by_n: dict, key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ns = sorted(n for n in by_n if by_n[n].get(key))
    mean = np.array([np.mean(by_n[n][key]) for n in ns])
    std = np.array([np.std(by_n[n][key]) for n in ns])
    return np.array(ns), mean, std


def main() -> None:
    arms = {label: collect_arm(pats) for label, (pats, _) in ARMS.items()}
    if not any(n != FULL_N_TRAIN for by_n in arms.values() for n in by_n):
        raise SystemExit(
            "No a5aug/a5base/a5C size-scan records found — run notebook 09 first"
        )

    fig, axes = plt.subplots(2, len(PANELS), figsize=(4 * len(PANELS), 7),
                             sharex=True)
    for col, (title, r_key, e_key) in enumerate(PANELS):
        ax_r, ax_e = axes[0, col], axes[1, col]
        for label, (_, style) in ARMS.items():
            ns, mean, std = series(arms[label], r_key)
            if len(ns):
                ax_r.errorbar(ns, mean, yerr=std, label=label, capsize=2, **style)
            ns, mean, std = series(arms[label], f"eps_{e_key}")
            if len(ns):
                ax_e.errorbar(ns, np.maximum(mean, EPS_FLOOR), yerr=std,
                              label=label, capsize=2, **style)
        ax_r.set_title(title)
        ax_r.axhline(0.0, color="gray", lw=0.5)
        ax_r.set_ylim(-0.15, 1.05)
        ax_e.set_xscale("log")
        ax_e.set_yscale("log")
        ax_e.set_ylim(EPS_FLOOR / 2, 5)
        ax_e.set_xlabel("training configs")
        if col == 0:
            ax_r.set_ylabel("test r")
            ax_e.set_ylabel(
                f"eps_gauge (floor {EPS_FLOOR:g}: C is exactly 0)"
            )
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        f"A-5 gauge augmentation: accuracy and eps_gauge vs training size "
        f"({CELL}, K=32 copies)"
    )
    fig.tight_layout()

    Path("figures").mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"figures/a5_augmentation.{ext}", dpi=200)

    points = {
        label: {
            str(n): {k: v for k, v in metrics.items()}
            for n, metrics in sorted(by_n.items())
        }
        for label, by_n in arms.items()
    }
    out_path = Path("results/a5_curves.json")
    with open(out_path, "w") as f:
        json.dump({"cell": CELL, "arms": points}, f, indent=2)
    log_run(
        "a5_curves_provenance",
        config={"cell": CELL, "protocol_version": 2},
        metrics={"n_points_per_arm": {k: len(v) for k, v in points.items()}},
        extra={"figure": "figures/a5_augmentation.pdf", "output_json": str(out_path)},
    )
    print(f"curves -> figures/a5_augmentation.{{pdf,png}} + {out_path}")


if __name__ == "__main__":
    main()
