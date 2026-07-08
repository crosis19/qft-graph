"""Precompute training labels for U(1) ensembles (task A-2).

For each configuration in each HDF5 ensemble produced by
scripts/generate_u1_data.py:
  - total Wilson action S = beta * sum(1 - cos theta_P);
  - per-config Wilson loops W(R,T) for (R,T) in {(1,1),(2,2),(2,4),(3,3),(4,4)}
    (RAW signed values — at large area / small beta per-config W fluctuates
    around a tiny mean and can be negative; -ln<W> is computed only at the
    ensemble level for validation against the exact area law);
  - topological charge Q (verified integer).

Labels are stored as datasets alongside theta in the same HDF5 file.

Usage:
    python scripts/label_u1_data.py [--data_dir data/u1_configs]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from qft_graph.fields.gauge import (
    plaquette_angles,
    topological_charge,
    wilson_loop_angles,
)
from qft_graph.mc.analysis import binned_jackknife
from qft_graph.mc.exact_u1 import exact_wilson_loop_infinite_volume
from qft_graph.utils.logging import setup_logging
from qft_graph.utils.run_logging import log_run

LOOP_SIZES = [(1, 1), (2, 2), (2, 4), (3, 3), (4, 4)]


def label_file(path: Path, logger) -> dict:
    with h5py.File(path, "r+") as f:
        theta = f["theta"][:].astype(np.float64)  # (N, 2, L, L)
        beta = float(f.attrs["beta"])
        L = int(f.attrs["L"])
        n = theta.shape[0]

        actions = np.empty(n)
        qs = np.empty(n)
        loops = {rt: np.empty(n) for rt in LOOP_SIZES}

        for i in range(n):
            th = theta[i]
            theta_p = plaquette_angles(th)
            actions[i] = beta * np.sum(1.0 - np.cos(theta_p))
            qs[i] = topological_charge(th)
            for R, T in LOOP_SIZES:
                loops[(R, T)][i] = float(np.cos(wilson_loop_angles(th, R, T)).mean())

        # Q must be integer on every config (oracle 3)
        max_q_dev = float(np.max(np.abs(qs - np.round(qs))))
        if max_q_dev > 1e-10:
            raise RuntimeError(f"{path}: non-integer Q, max dev {max_q_dev}")

        for name, data in [("action", actions), ("q", np.round(qs))]:
            if name in f:
                del f[name]
            f.create_dataset(name, data=data)
        for (R, T), data in loops.items():
            name = f"wilson/{R}x{T}"
            if name in f:
                del f[name]
            f.create_dataset(name, data=data)

        # Ensemble-level validation vs area law (RT << L^2 only)
        validation = {}
        for R, T in LOOP_SIZES:
            if R * T >= L * L / 4:
                continue
            w_mean, w_err = binned_jackknife(
                loops[(R, T)], lambda s: float(np.mean(s)), n_bins=20
            )
            exact = exact_wilson_loop_infinite_volume(beta, R, T)
            validation[f"W({R},{T})"] = {
                "mean": w_mean,
                "err": w_err,
                "exact_inf_volume": exact,
                "neg_log_mean": float(-np.log(w_mean)) if w_mean > 0 else None,
            }

        logger.info(
            "%s: labeled %d configs (S in [%.1f, %.1f], |Q| max %d)",
            path.name, n, actions.min(), actions.max(), int(np.abs(qs).max()),
        )
        return {
            "file": path.name,
            "beta": beta,
            "L": L,
            "n_configs": n,
            "q_abs_max": int(np.abs(np.round(qs)).max()),
            "validation": validation,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Label U(1) ensembles (A-2)")
    parser.add_argument("--data_dir", type=str, default="data/u1_configs")
    args = parser.parse_args()

    logger = setup_logging()
    files = sorted(Path(args.data_dir).glob("u1_L*_beta*.h5"))
    if not files:
        raise SystemExit(f"No ensembles found in {args.data_dir}")

    summaries = [label_file(p, logger) for p in files]
    log_run(
        "u1_labels",
        config={"data_dir": args.data_dir, "loop_sizes": LOOP_SIZES},
        metrics={"files": summaries},
    )
    logger.info("Labeled %d files. Validation table in results/u1_labels.json", len(files))


if __name__ == "__main__":
    main()
