"""Generate Monte Carlo configurations for scalar phi^4 theory.

Usage:
    python scripts/generate_mc_data.py [--config configs/defaults.yaml]
        [--output data/mc_configs/] [--n_configs 10000]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from qft_graph.actions.phi4 import Phi4Action
from qft_graph.config import load_config
from qft_graph.lattice.hypercubic import HypercubicLattice
from qft_graph.mc.metropolis import MetropolisSampler, create_sampler
from qft_graph.mc.observables import ObservableSet
from qft_graph.utils.logging import setup_logging
from qft_graph.utils.reproducibility import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MC data for phi^4 theory")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument("--output", type=str, default="data/mc_configs", help="Output directory")
    parser.add_argument("--n_configs", type=int, default=None, help="Override n_configs")
    parser.add_argument("--dimensions", type=int, nargs="+", default=None, help="Lattice dims")
    parser.add_argument("--mass_squared", type=float, default=None, help="m^2 value")
    parser.add_argument("--coupling", type=float, default=None, help="lambda value")
    args = parser.parse_args()

    logger = setup_logging()

    # Build config with overrides
    overrides = {}
    if args.n_configs is not None:
        overrides["mc.n_configs"] = args.n_configs
    if args.dimensions is not None:
        overrides["lattice.dimensions"] = args.dimensions
    if args.mass_squared is not None:
        overrides["scalar_field.mass_squared"] = args.mass_squared
    if args.coupling is not None:
        overrides["scalar_field.coupling"] = args.coupling

    config = load_config(args.config, overrides if overrides else None)

    set_seed(config.mc.seed)
    logger.info("Config: lattice=%s, m2=%.3f, lam=%.3f",
                config.lattice.dimensions, config.scalar_field.mass_squared, config.scalar_field.coupling)

    # Setup
    lattice = HypercubicLattice(config.lattice)
    action = Phi4Action(lattice, config.scalar_field)
    sampler = create_sampler(action, config.mc)

    # Generate
    n = config.mc.n_configs
    result = sampler.generate(n)

    # Compute observables
    obs = ObservableSet(lattice)
    mags = torch.tensor([obs.abs_magnetization(result.configurations[i]) for i in range(n)])
    logger.info("Acceptance rate: %.3f", result.acceptance_rate)
    logger.info("Mean |M|: %.4f +/- %.4f", mags.mean().item(), mags.std().item())
    logger.info("Mean S_E: %.4f +/- %.4f", result.actions.mean().item(), result.actions.std().item())

    # Save
    dims_str = "x".join(str(d) for d in config.lattice.dimensions)
    name = f"phi4_{dims_str}_m2={config.scalar_field.mass_squared}_lam={config.scalar_field.coupling}"
    out_dir = Path(args.output) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        "configurations": result.configurations,
        "actions": result.actions,
        "acceptance_rate": result.acceptance_rate,
        "config": config,
    }, out_dir / "mc_data.pt")

    logger.info("Saved %d configurations to %s", n, out_dir)


if __name__ == "__main__":
    main()
