"""A-5 deliverable: the eps_gauge table (plan §3, Task A-5 done-when).

Aggregates results/a5eps_*.json (from scripts/measure_gauge_invariance.py)
to seed mean +/- std per (experiment, variant, L, beta, H, B[, n_train])
cell and per target. Variant C rows are the sanity anchor — exactly 0 by
construction (bit-identical inputs + batch-1 forwards). The source runs'
test action r rides along for context: eps_gauge says how gauge-dependent
a prediction is; r says whether it predicts anything at all.

Writes results/a5_table.json and prints copy-paste LaTeX rows for the
headline comparison (a4null A/B vs a4C at the shared (L, beta) cells).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from qft_graph.utils.run_logging import log_run

RUN_RE = re.compile(
    r"^(?P<prefix>[A-Za-z0-9]+)_(?P<variant>link_nodes|edge_features|invariant_oracle)"
    r"_u1_L(?P<L>\d+)_beta(?P<beta>[\d.]+)_H(?P<H>\d+)_B(?P<B>\d+)"
    r"(?:_n(?P<n_train>\d+))?_seed(?P<seed>\d+)$"
)
VARIANT_LABEL = {
    "link_nodes": "A",
    "edge_features": "B",
    "invariant_oracle": "C",
}
# eps targets in presentation order; "energy" is the action head (see
# CLAUDE.md terminology note)
TARGETS = ("energy", "wilson_1x1", "wilson_2x2", "wilson_2x4", "wilson_3x3",
           "wilson_4x4", "q")
HEADLINE_PREFIXES = ("a4null", "a4C")


def collect() -> dict[tuple, list[dict]]:
    groups = defaultdict(list)
    for path in sorted(Path("results").glob("a5eps_*.json")):
        d = json.loads(path.read_text())
        if d.get("config", {}).get("protocol_version") != 2:
            raise ValueError(f"Non-v2 eps record: {path.name}")
        m = RUN_RE.match(d["config"]["source_run_id"])
        if not m:
            raise ValueError(f"Unparseable source_run_id in {path.name}")
        g = m.groupdict()
        key = (g["prefix"], g["variant"], int(g["L"]), float(g["beta"]),
               int(g["H"]), int(g["B"]),
               int(g["n_train"]) if g["n_train"] else None)
        groups[key].append(d)
    return groups


def aggregate(groups: dict[tuple, list[dict]]) -> list[dict]:
    rows = []
    order = {"link_nodes": 0, "edge_features": 1, "invariant_oracle": 2}
    for key in sorted(groups, key=lambda k: (k[0], order[k[1]], k[2:])):
        prefix, variant, L, beta, H, B, n_train = key
        runs = groups[key]
        row = {
            "experiment": prefix,
            "variant": VARIANT_LABEL[variant],
            "variant_name": variant,
            "L": L, "beta": beta, "H": H, "B": B, "n_train": n_train,
            "n_seeds": len(runs),
            "k_copies": runs[0]["config"]["k_copies"],
            "gauge_seed": runs[0]["config"]["gauge_seed"],
        }
        for t in TARGETS:
            eps = [r["metrics"][t]["eps_gauge"] for r in runs]
            row[f"eps_{t}"] = {"mean": float(np.mean(eps)), "std": float(np.std(eps))}
        action_r = [r["extra"]["source_test_metrics"]["action_r"] for r in runs]
        row["source_action_r"] = {
            "mean": float(np.mean(action_r)), "std": float(np.std(action_r))
        }
        rows.append(row)
    return rows


def fmt(m: dict) -> str:
    if m["mean"] == 0.0 and m["std"] == 0.0:
        return "0 (exact)"
    return f"{m['mean']:.3f}({int(round(m['std'] * 1000)):02d})"


def main() -> None:
    groups = collect()
    if not groups:
        raise SystemExit("No results/a5eps_*.json found — "
                         "run scripts/measure_gauge_invariance.py first")
    rows = aggregate(groups)

    out_path = Path("results/a5_table.json")
    with open(out_path, "w") as f:
        json.dump({"targets": TARGETS, "rows": rows}, f, indent=2)

    hdr = (f"{'exp':>10} {'V':>2} {'L':>3} {'beta':>4} {'H':>4} {'B':>2} "
           f"{'n_tr':>5} {'sd':>2} "
           + " ".join(f'{"e_" + t.replace("wilson_", "W"):>11}' for t in TARGETS)
           + f" {'src act r':>10}")
    print(hdr)
    for row in rows:
        n_tr = row["n_train"] if row["n_train"] is not None else "-"
        print(f"{row['experiment']:>10} {row['variant']:>2} {row['L']:>3} "
              f"{row['beta']:>4g} {row['H']:>4} {row['B']:>2} {n_tr:>5} "
              f"{row['n_seeds']:>2} "
              + " ".join(f"{fmt(row[f'eps_{t}']):>11}" for t in TARGETS)
              + f" {row['source_action_r']['mean']:>10.3f}")

    print("\nLaTeX rows (variant & L & beta & eps_S & eps_W1x1 & eps_W4x4 & eps_Q):")
    for row in rows:
        if row["experiment"] not in HEADLINE_PREFIXES:
            continue

        def ltx(t):
            m = row[f"eps_{t}"]
            if m["mean"] == 0.0 and m["std"] == 0.0:
                return "$0$ (exact)"
            return f"${m['mean']:.3f} \\pm {m['std']:.3f}$"

        print(f"{row['variant']} & {row['L']} & {row['beta']:g} & "
              f"{ltx('energy')} & {ltx('wilson_1x1')} & {ltx('wilson_4x4')} & "
              f"{ltx('q')} \\\\")

    log_run(
        "a5_table_provenance",
        config={"protocol_version": 2, "n_rows": len(rows)},
        metrics={"rows": rows},
        extra={"output_json": str(out_path)},
    )
    print(f"\n{len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
