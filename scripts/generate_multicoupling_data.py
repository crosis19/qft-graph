"""Generate the multi-coupling training grid for Table III' (task P1-3).

Protocol (frozen in the plan): L=16, lambda=0.5, 13 m^2 points from -2.9 to
-0.3 (includes the critical region ~ -2.45), 3000 configs per point,
thermalization >= 2000 sweeps, separation 10 sweeps increased near
criticality according to the measured tau_int. Actual separations are
recorded in the dataset metadata.

Usage:
    python scripts/generate_multicoupling_data.py [--pilot_configs 300]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from qft_graph.actions.phi4 import Phi4Action
from qft_graph.config import LatticeConfig, MCConfig, ScalarFieldConfig
from qft_graph.lattice.hypercubic import HypercubicLattice
from qft_graph.mc.analysis import integrated_autocorrelation_time_wolff
from qft_graph.mc.metropolis import create_sampler
from qft_graph.utils.logging import setup_logging
from qft_graph.utils.run_logging import log_run


def adaptive_separation(
    lattice, field_config, base_sep: int, pilot_configs: int, seed: int
) -> tuple[int, float]:
    """Measure tau_int in a pilot run and scale the sweep separation.

    Returns (separation, pilot_tau_int_in_config_units). tau_int ~ 1 in
    stored-config units means base_sep sweeps already decorrelate; larger
    values scale the separation proportionally (capped at 80).
    """
    action = Phi4Action(lattice, field_config)
    mc = MCConfig(
        n_configs=pilot_configs,
        n_thermalization=1000,
        n_sweeps_between=base_sep,
        seed=seed,
    )
    sampler = create_sampler(action, mc)
    result = sampler.generate(pilot_configs)
    absM = result.configurations.mean(dim=1).abs()
    tau, _, _ = integrated_autocorrelation_time_wolff(absM)
    sep = int(min(80, np.ceil(base_sep * max(1.0, tau))))
    return sep, tau


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-coupling data grid (P1-3)")
    parser.add_argument("--lattice_size", type=int, default=16)
    parser.add_argument("--coupling", type=float, default=0.5)
    parser.add_argument("--m2_min", type=float, default=-2.9)
    parser.add_argument("--m2_max", type=float, default=-0.3)
    parser.add_argument("--m2_steps", type=int, default=13)
    parser.add_argument("--n_configs", type=int, default=3000)
    parser.add_argument("--n_thermalization", type=int, default=2000)
    parser.add_argument("--base_separation", type=int, default=10)
    parser.add_argument("--pilot_configs", type=int, default=300)
    parser.add_argument("--data_dir", type=str, default="data/mc_configs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger = setup_logging()
    L = args.lattice_size
    lattice = HypercubicLattice(LatticeConfig(dimensions=(L, L)))
    m2_values = np.round(np.linspace(args.m2_min, args.m2_max, args.m2_steps), 4)

    summary = []
    for i, m2 in enumerate(m2_values):
        dirname = f"phi4_{L}x{L}_m2={m2:g}_lam={args.coupling:g}"
        out_path = Path(args.data_dir) / dirname / "mc_data.pt"
        point_seed = args.seed + 100 * i

        if out_path.exists():
            existing = torch.load(out_path, weights_only=False, map_location="cpu")
            if len(existing["configurations"]) >= args.n_configs and "separation" in existing:
                logger.info("[%d/%d] m2=%g: exists with metadata, skipping", i + 1, len(m2_values), m2)
                summary.append({"m2": float(m2), "path": str(out_path), "skipped": True})
                continue

        field_config = ScalarFieldConfig(mass_squared=float(m2), coupling=args.coupling)
        sep, pilot_tau = adaptive_separation(
            lattice, field_config, args.base_separation, args.pilot_configs, point_seed
        )
        logger.info(
            "[%d/%d] m2=%g: pilot tau_int=%.2f -> separation %d sweeps",
            i + 1, len(m2_values), m2, pilot_tau, sep,
        )

        action = Phi4Action(lattice, field_config)
        mc = MCConfig(
            n_configs=args.n_configs,
            n_thermalization=args.n_thermalization,
            n_sweeps_between=sep,
            seed=point_seed + 1,
        )
        sampler = create_sampler(action, mc)
        result = sampler.generate(args.n_configs)

        absM = result.configurations.mean(dim=1).abs()
        tau_final, tau_err, _ = integrated_autocorrelation_time_wolff(absM)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "configurations": result.configurations,
                "actions": result.actions,
                "acceptance_rate": result.acceptance_rate,
                "config": mc,
                "m2": float(m2),
                "lambda": args.coupling,
                "separation": sep,
                "thermalization": args.n_thermalization,
                "pilot_tau_int": pilot_tau,
                "final_tau_int": tau_final,
                "seed": point_seed + 1,
            },
            out_path,
        )
        logger.info(
            "  saved %d configs (acc %.3f, final tau_int %.2f) -> %s",
            args.n_configs, result.acceptance_rate, tau_final, out_path,
        )
        summary.append(
            {
                "m2": float(m2),
                "separation": sep,
                "pilot_tau_int": pilot_tau,
                "final_tau_int": tau_final,
                "acceptance": result.acceptance_rate,
                "path": str(out_path),
            }
        )

    log_run(
        f"multicoupling_data_{L}x{L}",
        config=vars(args),
        metrics={"n_points": len(m2_values)},
        extra={"points": summary},
    )
    logger.info("Done: %d points. Provenance in results/multicoupling_data_%dx%d.json",
                len(m2_values), L, L)


if __name__ == "__main__":
    main()
