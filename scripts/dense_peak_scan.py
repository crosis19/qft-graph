"""Dense large-L susceptibility-peak re-scan, parallelised across m^2 points.

The coarse production peak grid (step ~0.019) under-resolves the sharp large-L
peaks, so their quadratic fits have chi^2/dof ~ 3-15 and chi_max is biased low
(task V2-2 follow-up). A narrow grid (step ~0.004) centred on each known peak
fixes that. Precise chi_max at large L still needs many decorrelated samples
(cluster tau_int ~ 9 => ~18 sweeps/sample) and each near-critical cluster flip
is O(fraction x V), so a serial narrow scan is hours -- but the m^2 points are
independent, so this fans ALL (L, m^2) points across cores at once (own
thermalisation per point, no warm-start). Wall time ~ (n_pts x n_sizes /
n_cores) x per-point cost instead of the serial sum.

Peak centres are read from an existing analysis JSON (chi_peaks). Per size it
writes <out_dir>/sweep_LxL_lam=<c>.json (sorted point list, same schema as
scripts/sweep.py) so scripts/analyze_fss.py --peak_dir consumes it unchanged.

    python scripts/dense_peak_scan.py --sizes 64 96 128 \
        --analysis results/fss_analysis_cluster.json \
        --out_dir data/sweep_cluster_peakdense
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def compute_point(job: tuple) -> tuple:
    """Worker: generate one (L, m^2) cluster ensemble and measure observables."""
    L, m2, coupling, n_configs, n_therm, n_between, n_cluster, seed = job
    import torch

    torch.set_num_threads(1)
    sys.path.insert(0, str(REPO / "scripts"))
    sys.path.insert(0, str(REPO / "src"))
    from qft_graph.actions.phi4 import Phi4Action
    from qft_graph.config import LatticeConfig, MCConfig, ScalarFieldConfig
    from qft_graph.lattice.hypercubic import HypercubicLattice
    from qft_graph.mc.cluster import ClusterSampler
    from sweep import measure_point  # reuse the exact production measurement

    action = Phi4Action(
        HypercubicLattice(LatticeConfig(dimensions=(L, L))),
        ScalarFieldConfig(mass_squared=float(m2), coupling=coupling),
    )
    cfg = MCConfig(
        n_configs=n_configs, n_thermalization=n_therm, n_sweeps_between=n_between,
        step_size=1.0, seed=seed, n_cluster_per_sweep=n_cluster, n_local_per_sweep=1,
    )
    t0 = time.time()
    res = ClusterSampler(action, cfg).generate(n_configs)
    point = measure_point(res.configurations, L)
    point.update({
        "m2": float(m2), "lambda": coupling, "dimensions": [L, L],
        "acceptance_rate": res.acceptance_rate,
        "mean_action": float(res.actions.mean()),
        "cluster_fraction": float(res.observables["cluster_fraction"].mean()),
        "seed": seed,
    })
    return L, float(m2), point, time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[64, 96, 128])
    ap.add_argument("--analysis", default="results/fss_analysis_cluster.json",
                    help="JSON with chi_peaks to read each size's peak centre from.")
    ap.add_argument("--coupling", type=float, default=0.5)
    ap.add_argument("--half_width", type=float, default=0.016,
                    help="Fixed m^2 half-window around each peak centre "
                         "(used when --hw_scale is 0).")
    ap.add_argument("--hw_scale", type=float, default=0.0,
                    help="If > 0, use an L-dependent half-window hw_scale / L so "
                         "the grid step scales with the peak width (~L^-1/nu, "
                         "nu=1) -> a fixed fit window is the same relative slice "
                         "of every peak (uniform chi^2/dof across sizes).")
    ap.add_argument("--n_points", type=int, default=9)
    ap.add_argument("--n_configs", type=int, default=2500)
    ap.add_argument("--n_therm", type=int, default=1500)
    ap.add_argument("--n_sweeps_between", type=int, default=6)
    ap.add_argument("--n_cluster_per_sweep", type=int, default=2)
    ap.add_argument("--out_dir", default="data/sweep_cluster_peakdense")
    ap.add_argument("--max_workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=101)
    args = ap.parse_args()

    with open(args.analysis) as f:
        peaks = {int(p["L"]): float(p["m2_peak"])
                 for p in json.load(f).get("chi_peaks", [])}
    # Fallback centres (v5 + extrapolation) if a size is absent from the analysis.
    fallback = {64: -2.189, 96: -2.196, 128: -2.197}

    jobs = []
    for L in args.sizes:
        centre = peaks.get(L, fallback.get(L))
        if centre is None:
            raise SystemExit(f"no peak centre for L={L} (not in analysis or fallback)")
        hw = args.hw_scale / L if args.hw_scale > 0 else args.half_width
        grid = np.linspace(centre - hw, centre + hw, args.n_points)
        for i, m2 in enumerate(grid):
            seed = args.seed + 1000 * L + i
            jobs.append((L, float(m2), args.coupling, args.n_configs, args.n_therm,
                         args.n_sweeps_between, args.n_cluster_per_sweep, seed))
    # Heaviest (largest L) first so they don't tail.
    jobs.sort(key=lambda j: -j[0])
    print(f"{len(jobs)} points across sizes {args.sizes}, max_workers={args.max_workers}",
          flush=True)

    # Pin BLAS/torch threads so the many workers don't oversubscribe.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    os.environ.setdefault("PYTHONPATH", str(REPO / "src"))

    per_L: dict[int, list] = {L: [] for L in args.sizes}
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        futs = [ex.submit(compute_point, j) for j in jobs]
        for fut in as_completed(futs):
            L, m2, point, dt = fut.result()
            per_L[L].append(point)
            done += 1
            print(f"[{done}/{len(jobs)}] L={L} m2={m2:.4f} chi={point['susceptibility']:.2f}"
                  f"({point['susceptibility_err']:.2f}) tau={point['tau_int']:.1f} "
                  f"frac={point['cluster_fraction']:.2f} {dt:.0f}s", flush=True)

    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for L in args.sizes:
        pts = sorted(per_L[L], key=lambda p: p["m2"])
        with open(out_dir / f"sweep_{L}x{L}_lam={args.coupling}.json", "w") as f:
            json.dump(pts, f, indent=2)
    print(f"ALL_DONE {len(jobs)} points in {(time.time()-t0)/60:.1f} min -> {out_dir}",
          flush=True)


if __name__ == "__main__":
    main()
