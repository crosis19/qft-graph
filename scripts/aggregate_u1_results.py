"""Aggregate A-4 U(1) run results into the comparison tables (task A-4).

Scans results/<prefix>*.json produced by scripts/train_u1.py, groups runs
that differ only by seed, and writes seed mean/std per test metric to
results/a4_summary.json plus a readable console table. Also reports the
tail of each run group's validation history (was anything still climbing
when the budget ran out?) — the escape-hatch verdicts depend on it.

Usage:
    python scripts/aggregate_u1_results.py [--glob "a4*"] [--output results/a4_summary.json]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from qft_graph.utils.run_logging import log_run

METRIC_ORDER = [
    "action_r", "action_rel_err",
    "wilson_1x1_r", "wilson_2x2_r", "wilson_2x4_r", "wilson_3x3_r", "wilson_4x4_r",
    "q_r", "q_acc",
]


def collect(glob: str) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(Path("results").glob(f"{glob}.json")):
        with open(path) as f:
            d = json.load(f)
        if "run_id" not in d or "metrics" not in d or "test" not in d["metrics"]:
            continue  # not a train_u1.py run record (e.g. the summary itself)
        gid = re.sub(r"_seed\d+$", "", d["run_id"])
        groups[gid].append(d)
    return groups


def summarize(groups: dict[str, list[dict]]) -> dict:
    summary = {}
    for gid, runs in sorted(groups.items()):
        tests = [r["metrics"]["test"] for r in runs]
        stats = {
            k: {
                "mean": float(np.mean([t[k] for t in tests])),
                "std": float(np.std([t[k] for t in tests])),
            }
            for k in tests[0]
        }
        # Convergence check: last two logged val action-r points per run
        tails = []
        for r in runs:
            hist = r["metrics"].get("history", [])
            if len(hist) >= 2:
                tails.append((hist[-2]["action_r"], hist[-1]["action_r"]))
        summary[gid] = {
            "n_seeds": len(runs),
            "seeds": sorted(r["config"]["seeds"][0] for r in runs),
            "n_params": runs[0].get("extra", {}).get("n_params"),
            "test": stats,
            "val_action_r_tail": tails,
        }
    return summary


def print_table(summary: dict) -> None:
    for gid, s in summary.items():
        cols = "  ".join(
            f"{k.replace('wilson_', 'W').replace('_r', ' r').replace('_acc', ' acc')}"
            f"={s['test'][k]['mean']:+.3f}±{s['test'][k]['std']:.3f}"
            for k in METRIC_ORDER
            if k in s["test"] and k != "action_rel_err"
        )
        print(f"{gid}  (n={s['n_seeds']}, params={s['n_params']})\n    {cols}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate A-4 results")
    parser.add_argument("--glob", type=str, default="a4*")
    parser.add_argument("--output", type=str, default="results/a4_summary.json")
    args = parser.parse_args()

    groups = collect(args.glob)
    if not groups:
        raise SystemExit(f"No results match results/{args.glob}.json")
    summary = summarize(groups)
    print_table(summary)

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n{sum(s['n_seeds'] for s in summary.values())} runs "
          f"in {len(summary)} groups -> {args.output}")

    log_run(
        "a4_summary_provenance",
        config={"glob": args.glob, "n_groups": len(summary)},
        metrics={"groups": sorted(summary)},
        extra={"output_json": args.output},
    )


if __name__ == "__main__":
    main()
