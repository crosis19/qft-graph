"""Generate LaTeX table bodies for the paper from results/ JSONs.

Writes paper/tables/*.tex fragments that main.tex \\input's, so no table
number is ever hand-typed (plan ground rule 4). Skips any table whose
results JSON is missing or incomplete, and says so.

Usage:
    python scripts/make_paper_tables.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
OUT = PROJECT_ROOT / "paper" / "tables"

MODEL_LABELS = {
    "HeteroGNN": r"\textbf{HeteroGNN (ours)}",
    "HomogeneousGNN": "Homogeneous GNN",
    "LatticeCNN": "Lattice CNN",
    "LatticeCNN-matched": "Lattice CNN (matched)",
    "MLP": "MLP",
}


def fmt_pm(mean: float, std: float, digits: int = 5) -> str:
    if 0 < std < 0.5 * 10**-digits:
        # Std would print as 0.0...0 — show its magnitude instead
        exp = int(np.floor(np.log10(std)))
        return rf"${mean:.{digits}f} \pm 10^{{{exp}}}$"
    return rf"${mean:.{digits}f} \pm {std:.{digits}f}$"


def fmt_pct_pm(mean: float, std: float) -> str:
    return rf"${100 * mean:.2f} \pm {100 * std:.2f}\%$"


def fmt_one_minus_r(r_mean: float, r_std: float) -> str:
    """Report 1 - r in scientific notation (external-expert suggestion: with many
    r values near 1, the deviation is the informative quantity)."""
    v = 1.0 - r_mean
    if v <= 0:
        v = 1e-16
    e = int(np.floor(np.log10(v)))
    mv = v / 10**e
    ms = r_std / 10**e
    if r_std <= 0:
        return rf"${mv:.1f} \times 10^{{{e}}}$"
    return rf"${mv:.1f}({ms:.1f}) \times 10^{{{e}}}$"


def load(name: str) -> dict | None:
    path = RESULTS / f"{name}.json"
    if not path.exists():
        print(f"  {name}.json missing — skipped")
        return None
    with open(path) as f:
        return json.load(f)


def complete(data: dict, expected_seeds: int = 5) -> bool:
    """Every model must have all expected seeds (incremental saves mean a
    file can exist mid-run with a partial last seed)."""
    runs = data.get("per_seed", data.get("per_run", []))
    models = {r["model"] for r in runs if "model" in r}
    if not models:
        seeds = {r["seed"] for r in runs}
        return len(seeds) >= expected_seeds
    return all(
        len({r["seed"] for r in runs if r.get("model") == m}) >= expected_seeds
        for m in models
    )


def table1_action() -> bool:
    """Table I: action prediction across lattice sizes (HeteroGNN rows).

    Emits every size whose 5-seed result file is complete; requires at
    least the published 16x16 and 64x64 rows before writing anything.
    """
    sources = [
        ("baseline_8x8_v2", "8x8", 64, False),
        ("baseline_results_v2", "16x16", 256, True),
        ("baseline_32x32_v2", "32x32", 1024, False),
        ("baseline_64x64_v2", "64x64", 4096, True),
    ]
    rows = []
    for name, size, n_sites, required in sources:
        data = load(name)
        if data is None or not complete(data):
            if required:
                print(f"  table1: required {name} incomplete — skipped")
                return False
            continue
        s = next(x for x in data["summary"] if x["model"] == "HeteroGNN")
        size_label = size.replace("x", r" \times ")
        rows.append(
            rf"${size_label}$ & {n_sites} & "
            + fmt_one_minus_r(s["r_mean"], s["r_std"])
            + " & "
            + fmt_pct_pm(s["rel_err_mean"], s["rel_err_std"])
            + r" \\"
        )
    _write(
        "table1_body.tex", rows, "@{}lccc@{}",
        r"Lattice & Sites & $1 - r$ & Rel.\ Error",
    )
    return True


def table2_baselines() -> bool:
    """Table II: architecture comparison at L=16, mean +/- std over seeds."""
    data = load("baseline_results_v2")
    if data is None or not complete(data):
        print("  table2: baseline_results_v2 incomplete — skipped")
        return False
    order = ["HeteroGNN", "HomogeneousGNN", "LatticeCNN", "LatticeCNN-matched", "MLP"]
    rows = []
    for model in order:
        s = next((x for x in data["summary"] if x["model"] == model), None)
        if s is None:
            continue
        r_str = fmt_one_minus_r(s["r_mean"], s["r_std"])
        e_str = fmt_pct_pm(s["rel_err_mean"], s["rel_err_std"])
        rows.append(
            rf"{MODEL_LABELS[model]} & {s['n_params']:,} & {r_str} & {e_str} \\"
        )
    _write(
        "table2_body.tex", rows, "@{}lccc@{}",
        r"Model & Params & $1 - r$ & Rel.\ Error",
    )
    return True


def table3_prime() -> bool:
    """Table III': multi-coupling generalization, mean +/- std over seeds."""
    data = load("table3prime")
    if data is None:
        return False
    if len(data.get("protocol", {}).get("seeds", [])) < 5:
        print("  table3': fewer than 5 seeds — skipped")
        return False
    mark = {"train": r"^\dagger", "interpolation": "", "extrapolation": r"^\ast"}
    rows = []
    for row in data["table"]:
        rows.append(
            rf"${row['m2']:g}{mark[row['role']]}$ & "
            + fmt_one_minus_r(row["r_mean"], row["r_std"])
            + " & "
            + fmt_pct_pm(row["rel_err_mean"], row["rel_err_std"])
            + r" \\"
        )
    _write(
        "table3prime_body.tex", rows, "@{}lcc@{}",
        r"$\msq$ & $1 - r$ & Rel.\ Error",
    )
    return True


def table_size_transfer() -> bool:
    """Size-transfer table (P1-6): variant x eval size."""
    data = load("size_transfer")
    if data is None:
        return False
    seeds = {r["seed"] for r in data["per_run"]}
    if len(seeds) < 3:
        print("  size_transfer: fewer than 3 seeds — skipped")
        return False
    rows = []
    for variant, label in [("coords", "Coordinates"), ("constant", "Coordinate-free")]:
        variant_rows = sorted(
            (x for x in data["summary"] if x["variant"] == variant),
            key=lambda x: x["eval_L"],
        )
        for i, s in enumerate(variant_rows):
            vlabel = label if i == 0 else ""
            train_mark = r"^\dagger" if s["eval_L"] == 16 else ""
            rows.append(
                rf"{vlabel} & ${s['eval_L']} \times {s['eval_L']}{train_mark}$ & "
                + fmt_one_minus_r(s["r_mean"], s["r_std"])
                + " & "
                + fmt_pct_pm(s["rel_err_mean"], s["rel_err_std"])
                + r" \\"
            )
    _write(
        "table_size_transfer_body.tex", rows, "@{}lccc@{}",
        r"Variant & Eval.\ lattice & $1 - r$ & Rel.\ Error",
    )
    return True


def _write(name: str, rows: list[str], colspec: str, header: str) -> None:
    """Write a complete tabular environment.

    The fragment contains the whole {tabular} (TeX's \\input cannot be
    expanded safely mid-alignment, so row-only fragments break after
    booktabs rules); main.tex \\input's it inside the table float.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    lines = [
        "% AUTO-GENERATED by scripts/make_paper_tables.py -- do not edit",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
    ]
    with open(path, "w", encoding="ascii") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  wrote {path.relative_to(PROJECT_ROOT)} ({len(rows)} rows)")


if __name__ == "__main__":
    print("Generating paper table bodies from results/:")
    table1_action()
    table2_baselines()
    table3_prime()
    table_size_transfer()
