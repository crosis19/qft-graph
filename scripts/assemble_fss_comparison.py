"""Assemble the local-vs-cluster FSS comparison for the paper (task V2-2).

Combines the local-Metropolis analysis (results/fss_analysis_v5.json), the
cluster analysis (results/fss_analysis_cluster.json), and the matched
single-sweep tau_int run (results/tau_int_comparison.json) into:

  Table A  chi_max and integrated autocorrelation, per L, local vs cluster;
  Table B  the critical exponents gamma/nu, nu, m2_c, local vs cluster vs exact.

Writes results/fss_local_vs_cluster.json and prints GitHub-flavoured Markdown
for the Sec. IV.D-E rewrite (task V2-3). Every number traces to a committed
analysis JSON (ground rule 4).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from qft_graph.analysis.critical import fit_nu_from_pseudocritical_shifts


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def chi_by_L(report: dict) -> dict[int, tuple[float, float]]:
    return {int(p["L"]): (p["chi_max"], p["chi_max_err"]) for p in report.get("chi_peaks", [])}


def tau_local_by_L(report: dict) -> dict[int, float]:
    """Per-L max tau_int from a sweep analysis' tau_int table (thinned units)."""
    return {int(L): d["max"] for L, d in report.get("tau_int", {}).items()}


def fmt(v: float | None, e: float | None = None, nd: int = 2) -> str:
    if v is None:
        return "--"
    if e is None:
        return f"{v:.{nd}f}"
    return f"{v:.{nd}f}({e:.{nd}f})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", default="results/fss_analysis_v5.json")
    ap.add_argument("--cluster", default="results/fss_analysis_cluster.json")
    ap.add_argument("--tau", default="results/tau_int_comparison.json")
    ap.add_argument("--output", default="results/fss_local_vs_cluster.json")
    args = ap.parse_args()

    local = load(args.local)
    cluster = load(args.cluster)
    tau_cmp = load(args.tau) if Path(args.tau).exists() else {"rows": []}

    chi_loc = chi_by_L(local)
    chi_clu = chi_by_L(cluster)
    tau_rows = {int(r["L"]): r for r in tau_cmp.get("rows", [])}
    # Include sizes that appear only in the tau_int run (e.g. L=128, whose sharp
    # peak the production m^2 grid under-resolves so it is excluded from the
    # precision chi_max fit, but which the sampler still reaches at tau_int ~ 9).
    sizes = sorted(set(chi_loc) | set(chi_clu) | set(tau_rows))

    # --- Table A: chi_max + tau_int(|M|) single-sweep, local vs cluster ---
    table_a = []
    for L in sizes:
        cl = chi_loc.get(L)
        cc = chi_clu.get(L)
        tr = tau_rows.get(L, {})
        table_a.append({
            "L": L,
            "chi_max_local": cl[0] if cl else None,
            "chi_max_local_err": cl[1] if cl else None,
            "chi_max_cluster": cc[0] if cc else None,
            "chi_max_cluster_err": cc[1] if cc else None,
            "tau_int_local": tr.get("tau_int_local"),
            "tau_int_cluster": tr.get("tau_int_cluster"),
            "decorrelation_speedup": tr.get("decorrelation_speedup"),
        })

    # --- Table B: exponents ---
    def gn(rep: dict, key: str = "gamma_over_nu") -> tuple[float, float]:
        g = rep.get(key, {})
        return g.get("value"), g.get("err")

    cl_corr = cluster.get("gamma_over_nu_corrections", {}).get("omega_fixed_2", {})
    cl_aic = cluster.get("gamma_over_nu_aic", {})
    # nu and m2_c from the pseudo-critical-shift fit (robust; the xi/L collapse
    # metric is unreliable). Compute it for BOTH samplers from their chi-peak
    # locations so the comparison is apples-to-apples: both give nu ~ 1, but the
    # local peak locations are CSD-scattered (huge error, poor chi2/dof) while
    # the cluster's are clean.
    def peak_shift_nu(report: dict) -> tuple:
        pk = sorted(report.get("chi_peaks", []), key=lambda p: p["L"])
        if len(pk) < 4:
            return (None, None), (None, None)
        L = np.array([p["L"] for p in pk], float)
        m2p = np.array([p["m2_peak"] for p in pk])
        err = np.array([max(p["m2_peak_err"], 1e-4) for p in pk])
        r = fit_nu_from_pseudocritical_shifts(L, m2p, err)
        if not r.get("fit_ok"):
            return (None, None), (None, None)
        return (r["nu"], r["nu_err"]), (r["m2_c"], r["m2_c_err"])

    local_nu, _ = peak_shift_nu(local)
    cluster_nu, cluster_m2c = peak_shift_nu(cluster)
    table_b = {
        "gamma_over_nu": {
            "local_naive": gn(local),
            "cluster_naive": gn(cluster),
            "cluster_corrections_omega2": (
                cl_corr.get("gamma_over_nu"), cl_corr.get("gamma_over_nu_err")),
            "cluster_aic": (cl_aic.get("gamma_over_nu"), cl_aic.get("gamma_over_nu_err")),
            "exact_2d_ising": 1.75,
        },
        "nu": {
            "local_pseudocritical": local_nu,
            "cluster_pseudocritical": cluster_nu,
            "exact_2d_ising": 1.0,
        },
        "m2_c": {
            "cluster_pseudocritical": cluster_m2c,
            "cluster_2mom_crossing": gn(cluster, "m2_c_2mom"),
        },
    }

    out = {"table_a": table_a, "table_b": table_b}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    # --- Markdown for the paper draft ---
    print("\n### Table A -- chi_max and autocorrelation (local vs cluster)\n")
    print("| L | chi_max (local) | chi_max (cluster) | tau_int local | "
          "tau_int cluster | speedup |")
    print("|---|---|---|---|---|---|")
    for r in table_a:
        sp = r["decorrelation_speedup"]
        print(f"| {r['L']} | {fmt(r['chi_max_local'], r['chi_max_local_err'])} | "
              f"{fmt(r['chi_max_cluster'], r['chi_max_cluster_err'])} | "
              f"{fmt(r['tau_int_local'])} | {fmt(r['tau_int_cluster'])} | "
              f"{('%.1fx' % sp) if sp else '--'} |")

    print("\n### Table B -- critical exponents\n")
    b = table_b
    print("| quantity | local | cluster | exact (2D Ising) |")
    print("|---|---|---|---|")
    print(f"| gamma/nu (naive) | {fmt(*b['gamma_over_nu']['local_naive'])} | "
          f"{fmt(*b['gamma_over_nu']['cluster_naive'])} | 1.75 |")
    print(f"| gamma/nu (corr., omega=2) | -- | "
          f"{fmt(*b['gamma_over_nu']['cluster_corrections_omega2'])} | 1.75 |")
    print(f"| gamma/nu (AIC avg) | -- | "
          f"{fmt(*b['gamma_over_nu']['cluster_aic'])} | 1.75 |")
    print(f"| nu (pseudo-crit. shift) | {fmt(*b['nu']['local_pseudocritical'])} | "
          f"{fmt(*b['nu']['cluster_pseudocritical'])} | 1.0 |")
    print(f"| m2_c (pseudo-crit. shift) | -- | "
          f"{fmt(*b['m2_c']['cluster_pseudocritical'], nd=4)} | -- |")
    print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
