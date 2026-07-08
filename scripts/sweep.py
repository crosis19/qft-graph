"""Coupling constant sweep for phase transition mapping.

Generates MC data at a grid of m^2 values and computes ensemble
observables — |M|, chi (frozen convention), xi/L — with binned jackknife
errors, bin sizes set by the measured integrated autocorrelation time
(plan tasks P1-1/P1-2). The paper protocol lives in
configs/paper/phase1_fss_sweep.yaml.

Usage (paper Fig. 2 rerun, one lattice size per invocation):
    python scripts/sweep.py --dimensions 16 16 \
        --m2_min -1.5 --m2_max -2.8 --m2_steps 25 \
        --n_configs 2000 --n_thermalization 1500 --n_sweeps_between 10 \
        --warm_start --seed 42

Output JSON schema (per m^2 point):
    m2, lambda, dimensions, magnetization(+_err), susceptibility(+_err),
    susceptibility_var(+_err)          # Var(M) convention, sensitivity check
    xi(+_err), xi_over_L(+_err),       # frozen-convention chi inside xi
    xi_var(+_err), xi_var_over_L(+_err),
    tau_int, tau_int_err, tau_int_window, n_bins, acceptance_rate, seed
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
from qft_graph.mc.analysis import (
    binned_jackknife,
    integrated_autocorrelation_time_wolff,
)
from qft_graph.mc.metropolis import create_sampler
from qft_graph.mc.observables import ObservableSet
from qft_graph.utils.logging import setup_logging
from qft_graph.utils.reproducibility import set_seed
from qft_graph.utils.run_logging import log_run


def choose_n_bins(n_configs: int, tau_int: float) -> int:
    """Number of jackknife bins so that bin size >= 2*tau_int (in config units)."""
    bin_size = max(2, int(np.ceil(2.0 * tau_int)))
    return int(np.clip(n_configs // bin_size, 8, 25))


def measure_point(
    configs: torch.Tensor, L: int
) -> dict[str, float]:
    """All ensemble observables with binned jackknife errors for one m^2 point."""
    M_series = configs.mean(dim=1)
    tau, tau_err, window = integrated_autocorrelation_time_wolff(M_series.abs())
    n_bins = choose_n_bins(len(configs), tau)

    mag, mag_err = binned_jackknife(
        configs, lambda s: float(s.mean(dim=1).abs().mean()), n_bins=n_bins
    )
    chi, chi_err = binned_jackknife(
        configs,
        lambda s: ObservableSet.susceptibility(s, convention="abs"),
        n_bins=n_bins,
    )
    chi_var, chi_var_err = binned_jackknife(
        configs,
        lambda s: ObservableSet.susceptibility(s, convention="var"),
        n_bins=n_bins,
    )
    xi, xi_err = binned_jackknife(
        configs,
        lambda s: ObservableSet.correlation_length_fft(s, L, chi_convention="abs"),
        n_bins=n_bins,
    )
    xi_var, xi_var_err = binned_jackknife(
        configs,
        lambda s: ObservableSet.correlation_length_fft(s, L, chi_convention="var"),
        n_bins=n_bins,
    )

    return {
        "magnetization": mag,
        "magnetization_err": mag_err,
        "susceptibility": chi,
        "susceptibility_err": chi_err,
        "susceptibility_var": chi_var,
        "susceptibility_var_err": chi_var_err,
        "xi": xi,
        "xi_err": xi_err,
        "xi_over_L": xi / L,
        "xi_over_L_err": xi_err / L,
        "xi_var": xi_var,
        "xi_var_err": xi_var_err,
        "xi_var_over_L": xi_var / L,
        "xi_var_over_L_err": xi_var_err / L,
        "tau_int": tau,
        "tau_int_err": tau_err,
        "tau_int_window": window,
        "n_bins": n_bins,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Coupling sweep for phi^4 theory")
    parser.add_argument("--dimensions", type=int, nargs="+", default=[16, 16])
    parser.add_argument("--m2_min", type=float, default=-1.5)
    parser.add_argument("--m2_max", type=float, default=-2.8)
    parser.add_argument("--m2_steps", type=int, default=25)
    parser.add_argument("--coupling", type=float, default=0.5)
    parser.add_argument("--n_configs", type=int, default=2000)
    parser.add_argument("--n_thermalization", type=int, default=1500)
    parser.add_argument("--n_sweeps_between", type=int, default=10)
    parser.add_argument("--warm_start", action="store_true",
                        help="Seed each m^2 point with the last config of the previous one")
    parser.add_argument("--output", type=str, default="data/sweep_results")
    parser.add_argument("--run_id", type=str, default=None,
                        help="results/<run_id>.json provenance record name")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger = setup_logging()
    set_seed(args.seed)

    dims = tuple(args.dimensions)
    L = dims[0]
    # Sweep from m2_min toward m2_max in the given order (paper runs walk
    # from the symmetric phase into the broken phase: -1.5 -> -2.8)
    m2_values = np.linspace(args.m2_min, args.m2_max, args.m2_steps)

    lattice_config = LatticeConfig(dimensions=dims)
    lattice = HypercubicLattice(lattice_config)

    results = []
    warm_phi = None

    for i, m2 in enumerate(m2_values):
        point_seed = args.seed + 1000 * L + i
        field_config = ScalarFieldConfig(mass_squared=float(m2), coupling=args.coupling)
        action = Phi4Action(lattice, field_config)
        mc_config = MCConfig(
            n_configs=args.n_configs,
            n_thermalization=args.n_thermalization,
            n_sweeps_between=args.n_sweeps_between,
            seed=point_seed,
        )
        sampler = create_sampler(action, mc_config)
        mc_result = sampler.generate(args.n_configs, initial_phi=warm_phi)
        if args.warm_start:
            warm_phi = mc_result.configurations[-1].clone()

        point = measure_point(mc_result.configurations, L)
        point.update({
            "m2": float(m2),
            "lambda": args.coupling,
            "dimensions": list(dims),
            "acceptance_rate": mc_result.acceptance_rate,
            "mean_action": mc_result.actions.mean().item(),
            "seed": point_seed,
        })
        results.append(point)

        logger.info(
            "[%d/%d] m^2=%.4f: |M|=%.4f(%.4f) chi=%.2f(%.2f) xi/L=%.4f(%.4f) "
            "tau_int=%.2f acc=%.3f",
            i + 1, len(m2_values), m2,
            point["magnetization"], point["magnetization_err"],
            point["susceptibility"], point["susceptibility_err"],
            point["xi_over_L"], point["xi_over_L_err"],
            point["tau_int"], mc_result.acceptance_rate,
        )

    # Save results
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    dims_str = "x".join(str(d) for d in dims)
    out_path = out_dir / f"sweep_{dims_str}_lam={args.coupling}.json"

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Sweep results saved to %s", out_path)

    run_id = args.run_id or f"fss_sweep_{dims_str}_lam={args.coupling}"
    log_run(
        run_id,
        config={
            "dimensions": list(dims),
            "coupling": args.coupling,
            "m2_min": args.m2_min,
            "m2_max": args.m2_max,
            "m2_steps": args.m2_steps,
            "n_configs": args.n_configs,
            "n_thermalization": args.n_thermalization,
            "n_sweeps_between": args.n_sweeps_between,
            "warm_start": args.warm_start,
            "seed": args.seed,
        },
        metrics={
            "tau_int_max": max(p["tau_int"] for p in results),
            "acceptance_min": min(p["acceptance_rate"] for p in results),
        },
        extra={"output_json": str(out_path)},
    )
    logger.info("Provenance logged to results/%s.json", run_id)


if __name__ == "__main__":
    main()
