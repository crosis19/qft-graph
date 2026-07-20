"""Matched single-sweep tau_int comparison: local vs cluster (task V2-2).

At each lattice size's pseudo-critical m^2 (read from the cluster FSS analysis),
run the local (Checkerboard-Metropolis) and cluster (Wolff/Brower-Tamayo)
samplers with n_sweeps_between=1 and measure tau_int(|M|) in single-sweep units
-- the apples-to-apples number for the paper's critical-slowing-down table.
Because the production sweeps thin by different n_sweeps_between per sampler,
their reported tau_int are NOT directly comparable; this dedicated run is.

The local tau_int grows ~ L^z with z ~ 2 (the bias the paper indicts); the
cluster tau_int stays O(1) (z ~ 0.25), so the local sampler is run only up to
--local_max_L (it is prohibitively autocorrelated beyond that -- itself the
point), while the cluster is measured at every size.

    python scripts/tau_int_comparison.py --analysis results/fss_analysis_cluster.json \
        --sizes 16 24 32 48 64 96 128 --output results/tau_int_comparison.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qft_graph.actions.phi4 import Phi4Action
from qft_graph.config import LatticeConfig, MCConfig, ScalarFieldConfig
from qft_graph.lattice.hypercubic import HypercubicLattice
from qft_graph.mc.analysis import integrated_autocorrelation_time_wolff
from qft_graph.mc.cluster import ClusterSampler
from qft_graph.mc.metropolis import CheckerboardSampler
from qft_graph.utils.logging import setup_logging
from qft_graph.utils.run_logging import log_run

# Fallback pseudo-critical m^2(L) if no analysis file is given (v5 + extrapolation).
DEFAULT_PEAK_M2 = {
    16: -2.116, 24: -2.157, 32: -2.154, 48: -2.180,
    64: -2.183, 96: -2.187, 128: -2.190,
}


def peak_m2_from_analysis(path: Path) -> dict[int, float]:
    with open(path) as f:
        report = json.load(f)
    return {int(p["L"]): float(p["m2_peak"]) for p in report.get("chi_peaks", [])}


def tau_int_abs_M(configs) -> float:
    series = configs.mean(dim=1).abs()
    tau, _, _ = integrated_autocorrelation_time_wolff(series)
    return float(tau)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", type=str, default="results/fss_analysis_cluster.json")
    ap.add_argument("--sizes", type=int, nargs="+", default=[16, 24, 32, 48, 64, 96, 128])
    ap.add_argument("--coupling", type=float, default=0.5)
    ap.add_argument("--n_configs", type=int, default=4000,
                    help="Single-sweep measurements per (sampler, L).")
    ap.add_argument("--n_therm", type=int, default=1000)
    ap.add_argument("--local_max_L", type=int, default=64,
                    help="Skip the (very autocorrelated) local sampler above this L.")
    ap.add_argument("--n_cluster_per_sweep", type=int, default=2)
    ap.add_argument("--output", type=str, default="results/tau_int_comparison.json")
    ap.add_argument("--seed", type=int, default=20)
    args = ap.parse_args()

    log = setup_logging()
    peaks = dict(DEFAULT_PEAK_M2)
    if args.analysis and Path(args.analysis).exists():
        peaks.update(peak_m2_from_analysis(Path(args.analysis)))
        log.info("using pseudo-critical m^2 from %s", args.analysis)

    rows = []
    for L in args.sizes:
        m2 = peaks[L]
        action = Phi4Action(
            HypercubicLattice(LatticeConfig(dimensions=(L, L))),
            ScalarFieldConfig(mass_squared=m2, coupling=args.coupling),
        )
        row = {"L": L, "m2": m2}

        # Cluster (every size).
        ccfg = MCConfig(
            n_configs=args.n_configs, n_thermalization=args.n_therm,
            n_sweeps_between=1, step_size=1.0, seed=args.seed,
            n_cluster_per_sweep=args.n_cluster_per_sweep, n_local_per_sweep=1,
        )
        t0 = time.time()
        rc = ClusterSampler(action, ccfg).generate(args.n_configs)
        row["tau_int_cluster"] = tau_int_abs_M(rc.configurations)
        row["cluster_fraction"] = float(rc.observables["cluster_fraction"].mean())
        t_cluster = time.time() - t0

        # Local (only up to local_max_L; above it CSD makes this impractical).
        if L <= args.local_max_L:
            lcfg = MCConfig(
                n_configs=args.n_configs, n_thermalization=args.n_therm,
                n_sweeps_between=1, step_size=1.0, seed=args.seed + 1,
            )
            t1 = time.time()
            rl = CheckerboardSampler(action, lcfg).generate(args.n_configs)
            row["tau_int_local"] = tau_int_abs_M(rl.configurations)
            t_local = time.time() - t1
        else:
            row["tau_int_local"] = None
            t_local = 0.0

        speedup = (row["tau_int_local"] / row["tau_int_cluster"]
                   if row["tau_int_local"] else None)
        row["decorrelation_speedup"] = speedup
        rows.append(row)
        log.info(
            "L=%3d m^2=%.3f: tau_int(|M|) cluster=%.2f local=%s  speedup=%s "
            "(frac=%.2f, %.0fs+%.0fs)",
            L, m2, row["tau_int_cluster"],
            f"{row['tau_int_local']:.2f}" if row["tau_int_local"] else "n/a",
            f"{speedup:.1f}x" if speedup else "n/a",
            row["cluster_fraction"], t_cluster, t_local,
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "coupling": args.coupling,
        "n_configs": args.n_configs,
        "n_sweeps_between": 1,
        "note": "tau_int(|M|) in single-sweep units at the pseudo-critical m^2(L)",
        "rows": rows,
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("tau_int comparison written to %s", out)
    log_run(
        "tau_int_comparison_provenance",
        config={"sizes": args.sizes, "coupling": args.coupling,
                "n_configs": args.n_configs, "seed": args.seed},
        metrics={"rows": rows},
        extra={"output_json": str(out)},
    )


if __name__ == "__main__":
    main()
