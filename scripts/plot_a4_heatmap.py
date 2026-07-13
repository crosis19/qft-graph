"""A-4 deliverable: receptive-field heatmap (plan §3, Task A-4 done-when).

Prediction r vs (loop area, message-passing depth B) for Variant C at
L=16, one panel per beta in {2, 4} (protocol-v2 runs, a4rf_*). The two
panels carry the paper's two competing effects:
- beta=2: r rises monotonically with B for large loops — the
  receptive-field gain the plan predicted;
- beta=4: r peaks at B=3 and INVERTS for deeper stacks — the ordered
  phase halves the input feature spread and deep stacks degrade
  (over-smoothing hypothesis; decision record 5 outcome correction).

Writes figures/a4_receptive_field.{pdf,png} and the cell matrix to
results/a4_heatmap_data.json.
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

LOOPS = [("1x1", 1), ("2x2", 4), ("2x4", 8), ("3x3", 9), ("4x4", 16)]
DEPTHS = [2, 3, 4, 6]
BETAS = [2.0, 4.0]


def collect() -> dict[tuple, list[dict]]:
    groups = defaultdict(list)
    for path in sorted(Path("results").glob("a4rf_*.json")):
        d = json.loads(path.read_text())
        if "run_id" not in d or "test" not in d.get("metrics", {}):
            continue
        if d["config"].get("protocol_version") != 2:
            raise ValueError(f"Non-v2 run in results/: {path.name}")
        m = re.match(r".*_beta([\d.]+)_H\d+_B(\d+)_seed\d+$", d["run_id"])
        groups[(float(m.group(1)), int(m.group(2)))].append(d["metrics"]["test"])
    return groups


def main() -> None:
    groups = collect()
    if not groups:
        raise SystemExit("No v2 a4rf results found")

    data = {}
    for beta in BETAS:
        mean = np.full((len(DEPTHS), len(LOOPS)), np.nan)
        std = np.full_like(mean, np.nan)
        for i, B in enumerate(DEPTHS):
            tests = groups.get((beta, B), [])
            for j, (name, _) in enumerate(LOOPS):
                vals = [t[f"wilson_{name}_r"] for t in tests]
                if vals:
                    mean[i, j] = np.mean(vals)
                    std[i, j] = np.std(vals)
        data[beta] = (mean, std)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), constrained_layout=True)
    for ax, beta in zip(axes, BETAS):
        mean, std = data[beta]
        im = ax.imshow(mean, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
        for i in range(len(DEPTHS)):
            for j in range(len(LOOPS)):
                if np.isfinite(mean[i, j]):
                    ax.text(
                        j, i, f"{mean[i, j]:.2f}\n±{std[i, j]:.2f}",
                        ha="center", va="center", fontsize=7,
                        color="white" if mean[i, j] < 0.6 else "black",
                    )
        ax.set_xticks(range(len(LOOPS)),
                      [f"{n}\n(area {a})" for n, a in LOOPS], fontsize=8)
        ax.set_yticks(range(len(DEPTHS)), [str(b) for b in DEPTHS])
        ax.set_ylabel("message-passing blocks B")
        ax.set_xlabel("Wilson loop (area)")
        ax.set_title(f"Variant C, L=16, β={beta:g}")
    fig.colorbar(im, ax=axes, label="test Pearson r", shrink=0.9)

    out_dir = Path("figures")
    out_dir.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"a4_receptive_field.{ext}", dpi=200)

    matrix_path = Path("results/a4_heatmap_data.json")
    with open(matrix_path, "w") as f:
        json.dump(
            {
                "depths": DEPTHS,
                "loops": [n for n, _ in LOOPS],
                "areas": [a for _, a in LOOPS],
                "betas": BETAS,
                "r_mean": {str(b): data[b][0].tolist() for b in BETAS},
                "r_std": {str(b): data[b][1].tolist() for b in BETAS},
            },
            f, indent=2,
        )

    log_run(
        "a4_heatmap_provenance",
        config={"protocol_version": 2, "depths": DEPTHS, "betas": BETAS},
        metrics={"r_mean": {str(b): data[b][0].tolist() for b in BETAS}},
        extra={"figure": str(out_dir / "a4_receptive_field.pdf"),
               "matrix_json": str(matrix_path)},
    )
    print(f"heatmap -> figures/a4_receptive_field.pdf|png, matrix -> {matrix_path}")


if __name__ == "__main__":
    main()
