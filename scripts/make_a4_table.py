"""A-4 deliverable: the A/B/C comparison table (plan §3, Task A-4 done-when).

Rows: (variant, L, beta) from the protocol-v2 runs — Variants A/B from the
null-robustness matrix (a4null_*), Variant C from the full protocol
(a4C_*). Columns: action r, Wilson r by loop size, Q regression r and
exact-integer accuracy, parameter count; mean +/- std over seeds.

q_acc context (decision record 5, outcome correction): exact-integer
accuracy obeys the rounding floor acc ~ P(|N(0, sigma_Q*sqrt(1-q_r^2))| <
0.5) and hardens as sigma_Q grows with volume — read it alongside q_r.

Writes results/a4_table.json and prints copy-paste LaTeX rows.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from qft_graph.utils.run_logging import log_run

LOOPS = ("1x1", "2x2", "2x4", "3x3", "4x4")
VARIANT_LABEL = {
    "link_nodes": "A",
    "edge_features": "B",
    "invariant_oracle": "C",
}


def collect() -> dict[tuple, list[dict]]:
    groups = defaultdict(list)
    for prefix in ("a4null", "a4C"):
        for path in sorted(Path("results").glob(f"{prefix}_*.json")):
            d = json.loads(path.read_text())
            if "run_id" not in d or "test" not in d.get("metrics", {}):
                continue
            if d["config"].get("protocol_version") != 2:
                raise ValueError(f"Non-v2 run in results/: {path.name}")
            m = re.match(
                r".*_(link_nodes|edge_features|invariant_oracle)_u1_L(\d+)_beta([\d.]+)_H\d+_B3_seed\d+$",
                d["run_id"],
            )
            if not m:
                continue
            variant, L, beta = m.group(1), int(m.group(2)), float(m.group(3))
            groups[(variant, L, beta)].append(d)
    return groups


def main() -> None:
    groups = collect()
    if not groups:
        raise SystemExit("No v2 a4null/a4C results found")

    rows = []
    for (variant, L, beta) in sorted(
        groups, key=lambda k: (VARIANT_LABEL[k[0]], k[1], k[2])
    ):
        runs = groups[(variant, L, beta)]
        tests = [r["metrics"]["test"] for r in runs]
        row = {
            "variant": VARIANT_LABEL[variant],
            "variant_name": variant,
            "L": L,
            "beta": beta,
            "n_seeds": len(runs),
            "n_params": runs[0]["extra"]["n_params"],
        }
        for key in ("action_r", "q_r", "q_acc", *(f"wilson_{n}_r" for n in LOOPS)):
            vals = [t[key] for t in tests]
            row[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        rows.append(row)

    out_path = Path("results/a4_table.json")
    with open(out_path, "w") as f:
        json.dump({"loops": LOOPS, "rows": rows}, f, indent=2)

    def fmt(m):
        return f"{m['mean']:+.3f}({int(round(m['std'] * 1000)):02d})"

    print(f"{'V':>2} {'L':>3} {'beta':>4} {'params':>7} {'action r':>12} "
          + " ".join(f"{'W' + n:>12}" for n in LOOPS)
          + f" {'q r':>12} {'q acc':>12}")
    for row in rows:
        print(f"{row['variant']:>2} {row['L']:>3} {row['beta']:>4g} "
              f"{row['n_params']:>7} {fmt(row['action_r']):>12} "
              + " ".join(f"{fmt(row[f'wilson_{n}_r']):>12}" for n in LOOPS)
              + f" {fmt(row['q_r']):>12} {fmt(row['q_acc']):>12}")

    print("\nLaTeX rows (variant & L & beta & params & action r & W1x1 & W4x4 & Q acc):")
    for row in rows:
        print(
            f"{row['variant']} & {row['L']} & {row['beta']:g} & "
            f"{row['n_params'] / 1000:.0f}k & "
            f"${row['action_r']['mean']:.3f} \\pm {row['action_r']['std']:.3f}$ & "
            f"${row['wilson_1x1_r']['mean']:.3f} \\pm {row['wilson_1x1_r']['std']:.3f}$ & "
            f"${row['wilson_4x4_r']['mean']:.3f} \\pm {row['wilson_4x4_r']['std']:.3f}$ & "
            f"${row['q_acc']['mean']:.3f} \\pm {row['q_acc']['std']:.3f}$ \\\\"
        )

    log_run(
        "a4_table_provenance",
        config={"protocol_version": 2, "n_rows": len(rows)},
        metrics={"rows": rows},
        extra={"output_json": str(out_path)},
    )
    print(f"\n{len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
