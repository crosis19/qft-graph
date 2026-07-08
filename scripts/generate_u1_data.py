"""Generate quenched U(1) gauge ensembles via heatbath (task A-1).

Protocol (configs/mc/heatbath_u1.yaml): beta in {0.5, 1, 2, 3, 4},
L in {8, 16, 32}, 4000 configs per (beta, L), thermalization 1000 sweeps,
separation 5 sweeps. tau_int(plaquette) is measured on each ensemble and
recorded in the HDF5 attrs and the results/ tau_int table.

Storage: HDF5, float32 theta[N_cfg, 2, L, L];
attrs: beta, L, seed, separation, thermalization, tau_int_plaquette.

Usage:
    python scripts/generate_u1_data.py [--config configs/mc/heatbath_u1.yaml]
        [--betas 1.0 2.0] [--sizes 8 16] [--n_configs 4000]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

from qft_graph.actions.wilson import WilsonGaugeAction
from qft_graph.config import LatticeConfig, MCConfig
from qft_graph.fields.gauge import plaquette_angles
from qft_graph.lattice.hypercubic import HypercubicLattice
from qft_graph.mc.analysis import integrated_autocorrelation_time_wolff
from qft_graph.mc.exact_u1 import exact_mean_plaquette
from qft_graph.mc.heatbath import U1HeatbathSampler
from qft_graph.utils.logging import setup_logging
from qft_graph.utils.run_logging import log_run


def main() -> None:
    parser = argparse.ArgumentParser(description="U(1) heatbath data generation")
    parser.add_argument("--config", type=str, default="configs/mc/heatbath_u1.yaml")
    parser.add_argument("--betas", type=float, nargs="+", default=None)
    parser.add_argument("--sizes", type=int, nargs="+", default=None)
    parser.add_argument("--n_configs", type=int, default=None)
    parser.add_argument("--n_thermalization", type=int, default=None)
    parser.add_argument("--n_sweeps_between", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    logger = setup_logging()
    cfg = OmegaConf.load(args.config)
    for key in ("betas", "sizes", "n_configs", "n_thermalization",
                "n_sweeps_between", "output", "seed"):
        val = getattr(args, key)
        if val is not None:
            cfg[key] = val

    out_dir = Path(str(cfg.output))
    out_dir.mkdir(parents=True, exist_ok=True)

    tau_table = []
    for L in cfg.sizes:
        lattice = HypercubicLattice(LatticeConfig(dimensions=(int(L), int(L))))
        for beta in cfg.betas:
            run_seed = int(cfg.seed) + int(1000 * float(beta)) + int(L)
            out_path = out_dir / f"u1_L{int(L)}_beta{float(beta):g}.h5"
            if out_path.exists():
                logger.info("exists, skipping: %s", out_path)
                continue

            action = WilsonGaugeAction(lattice, float(beta))
            sampler = U1HeatbathSampler(
                action,
                MCConfig(
                    n_configs=int(cfg.n_configs),
                    n_thermalization=int(cfg.n_thermalization),
                    n_sweeps_between=int(cfg.n_sweeps_between),
                    seed=run_seed,
                ),
            )
            result = sampler.generate(int(cfg.n_configs))

            # tau_int of the mean-plaquette series (stored-config units)
            cos_p = torch.tensor([
                float(np.cos(plaquette_angles(c.numpy().astype(np.float64))).mean())
                for c in result.configurations
            ])
            tau, tau_err, _ = integrated_autocorrelation_time_wolff(cos_p)
            mean_p = float(cos_p.mean())
            exact_p = exact_mean_plaquette(float(beta), int(L))
            logger.info(
                "L=%d beta=%g: <cosP>=%.5f (exact %.5f), tau_int=%.2f -> %s",
                L, beta, mean_p, exact_p, tau, out_path,
            )

            with h5py.File(out_path, "w") as f:
                f.create_dataset(
                    "theta",
                    data=result.configurations.numpy().astype(np.float32),
                    compression="gzip",
                )
                f.attrs.update({
                    "beta": float(beta),
                    "L": int(L),
                    "seed": run_seed,
                    "separation": int(cfg.n_sweeps_between),
                    "thermalization": int(cfg.n_thermalization),
                    "tau_int_plaquette": tau,
                    "mean_plaquette": mean_p,
                    "exact_mean_plaquette": exact_p,
                })

            tau_table.append({
                "L": int(L), "beta": float(beta), "tau_int": tau,
                "tau_int_err": tau_err, "mean_plaquette": mean_p,
                "exact_mean_plaquette": exact_p,
                "deviation": mean_p - exact_p,
            })

    log_run(
        "u1_heatbath_data",
        config=OmegaConf.to_container(cfg),
        metrics={"tau_int_table": tau_table},
    )
    logger.info("Done. tau_int table in results/u1_heatbath_data.json")


if __name__ == "__main__":
    main()
