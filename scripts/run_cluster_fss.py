"""Orchestrate the cluster FSS production sweeps (task V2-2).

Runs, in a bounded process pool, a base (crossings/collapse) sweep and a
dense peak (chi_max) sweep with the Wolff/Brower-Tamayo cluster sampler for
each lattice size, mirroring the local-Metropolis production
(data/sweep_results_v2 + data/sweep_peakrefine -> results/fss_analysis_v5.json)
so scripts/analyze_fss.py consumes the cluster data unchanged:

    python scripts/analyze_fss.py --sweep_dir data/sweep_cluster \
        --peak_dir data/sweep_cluster_peak --sizes 16 24 32 48 64 96 128 \
        --coupling 0.5 --output results/fss_analysis_cluster.json

Cluster autocorrelation is O(1), so fewer configs than the local run give
smaller errors. Each sweep is single-threaded (BLAS/torch pinned to 1 thread);
--max_workers controls how many run at once. Per-sweep logs go to logs_cluster/.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable

# Common critical-region physics for lam=0.5 (results/fss_analysis_v5.json):
# chi peaks near m^2 ~ -2.12..-2.19, xi/L crossings near ~ -2.4.
BASE = dict(m2_min=-1.6, m2_max=-2.7, m2_steps=23, n_therm=800, n_between=6,
            out="data/sweep_cluster")
PEAK = dict(m2_min=-2.02, m2_max=-2.30, m2_steps=15, n_therm=1000, n_between=8,
            out="data/sweep_cluster_peak")

# n_configs per (size); larger lattices use fewer configs (cost ~ V) — still
# far more *independent* samples than the local run thanks to low tau_int.
N_CONFIGS = {16: 2500, 24: 2500, 32: 2500, 48: 2000, 64: 2000, 96: 1500, 128: 1500}


def sweep_cmd(kind: str, L: int, n_configs: int) -> list[str]:
    g = BASE if kind == "base" else PEAK
    return [
        PY, "scripts/sweep.py", "--sampler", "cluster",
        "--dimensions", str(L), str(L), "--coupling", "0.5",
        "--m2_min", str(g["m2_min"]), "--m2_max", str(g["m2_max"]),
        "--m2_steps", str(g["m2_steps"]),
        "--n_configs", str(n_configs),
        "--n_thermalization", str(g["n_therm"]),
        "--n_sweeps_between", str(g["n_between"]),
        "--n_cluster_per_sweep", "2", "--n_local_per_sweep", "1",
        "--store_series", "--warm_start", "--seed", "42",
        "--output", g["out"], "--run_id", f"cluster_{kind}_{L}x{L}",
    ]


def run_job(job: tuple[str, int]) -> tuple[str, int, int, float]:
    kind, L = job
    env = {
        **os.environ,
        "PYTHONPATH": "src",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    }
    log_dir = REPO / "logs_cluster"
    log_dir.mkdir(exist_ok=True)
    t0 = time.time()
    with open(log_dir / f"{kind}_{L}.log", "w") as log:
        proc = subprocess.run(
            sweep_cmd(kind, L, N_CONFIGS[L]),
            cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
        )
    return kind, L, proc.returncode, time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[16, 24, 32, 48, 64, 96, 128])
    ap.add_argument("--kinds", nargs="+", default=["base", "peak"],
                    choices=["base", "peak"])
    ap.add_argument("--max_workers", type=int, default=10)
    args = ap.parse_args()

    jobs = [(k, L) for L in args.sizes for k in args.kinds]
    # Launch the largest lattices first so they don't become the tail.
    jobs.sort(key=lambda j: (-j[1], j[0]))
    print(f"launching {len(jobs)} sweeps, max_workers={args.max_workers}", flush=True)

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = {ex.submit(run_job, j): j for j in jobs}
        for fut in as_completed(futs):
            kind, L, rc, dt = fut.result()
            done += 1
            status = "OK" if rc == 0 else f"FAILED(rc={rc})"
            print(f"[{done}/{len(jobs)}] {kind} L={L}: {status} in {dt/60:.1f} min",
                  flush=True)
    print(f"ALL_DONE in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
