"""Coupling constant sweep for phase transition mapping.

Generates MC data and optionally trains models at multiple coupling values.
Used for finite-size scaling analysis to extract critical exponents.

Usage:
    python scripts/sweep.py --dimensions 8 8 --m2_min -1.0 --m2_max 0.0 --m2_steps 20
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from qft_graph.actions.phi4 import Phi4Action
from qft_graph.config import LatticeConfig, MCConfig, ScalarFieldConfig
from qft_graph.lattice.hypercubic import HypercubicLattice
from qft_graph.mc.analysis import jackknife_mean_error
from qft_graph.mc.metropolis import MetropolisSampler
from qft_graph.mc.observables import ObservableSet
from qft_graph.utils.logging import setup_logging
from qft_graph.utils.reproducibility import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Coupling sweep for phi^4 theory")
    parser.add_argument("--dimensions", type=int, nargs="+", default=[16, 16])
    parser.add_argument("--m2_min", type=float, default=-1.0)
    parser.add_argument("--m2_max", type=float, default=0.0)
    parser.add_argument("--m2_steps", type=int, default=20)
    parser.add_argument("--coupling", type=float, default=0.5)
    parser.add_argument("--n_configs", type=int, default=5000)
    parser.add_argument("--output", type=str, default="data/sweep_results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger = setup_logging()
    set_seed(args.seed)

    dims = tuple(args.dimensions)
    m2_values = np.linspace(args.m2_min, args.m2_max, args.m2_steps)

    lattice_config = LatticeConfig(dimensions=dims)
    lattice = HypercubicLattice(lattice_config)
    obs = ObservableSet(lattice)

    results = []

    for i, m2 in enumerate(m2_values):
        logger.info("Sweep %d/%d: m^2 = %.4f", i + 1, len(m2_values), m2)

        field_config = ScalarFieldConfig(mass_squared=m2, coupling=args.coupling)
        action = Phi4Action(lattice, field_config)
        mc_config = MCConfig(
            n_configs=args.n_configs,
            n_thermalization=500,
            n_sweeps_between=5,
            seed=args.seed + i,
        )
        sampler = MetropolisSampler(action, mc_config)
        mc_result = sampler.generate(args.n_configs)

        # Compute observables
        mags = torch.tensor([
            obs.abs_magnetization(mc_result.configurations[j])
            for j in range(args.n_configs)
        ])
        susc_terms = torch.tensor([
            obs.susceptibility_term(mc_result.configurations[j])
            for j in range(args.n_configs)
        ])

        mag_mean, mag_err = jackknife_mean_error(mags)
        chi = lattice.num_sites() * (susc_terms.mean().item() - mags.mean().item() ** 2)

        # Average correlation length
        xi_samples = []
        for j in range(min(100, args.n_configs)):
            G_r = obs.two_point_function(mc_result.configurations[j])
            xi = ObservableSet.correlation_length(G_r)
            if xi < float("inf"):
                xi_samples.append(xi)
        xi_mean = np.mean(xi_samples) if xi_samples else 0.0

        results.append({
            "m2": float(m2),
            "lambda": args.coupling,
            "dimensions": list(dims),
            "magnetization": mag_mean,
            "magnetization_err": mag_err,
            "susceptibility": float(chi),
            "correlation_length": float(xi_mean),
            "xi_over_L": float(xi_mean / dims[0]),
            "acceptance_rate": mc_result.acceptance_rate,
            "mean_action": mc_result.actions.mean().item(),
        })

        logger.info("  |M| = %.4f +/- %.4f, chi = %.2f, xi = %.2f, xi/L = %.4f",
                     mag_mean, mag_err, chi, xi_mean, xi_mean / dims[0])

    # Save results
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    dims_str = "x".join(str(d) for d in dims)
    out_path = out_dir / f"sweep_{dims_str}_lam={args.coupling}.json"

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Sweep results saved to %s", out_path)


if __name__ == "__main__":
    main()
