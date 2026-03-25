"""Metropolis-Hastings sampler for scalar phi^4 theory.

Optimized with precomputed neighbor tables and numpy inner loops
to avoid per-site edge scanning. Typical speedup: 50-100x over
the naive torch implementation.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import torch

from qft_graph.actions.phi4 import Phi4Action
from qft_graph.config import MCConfig
from qft_graph.mc.sampler import MCResult, MCSampler

logger = logging.getLogger("qft_graph.mc")


class MetropolisSampler(MCSampler):
    """Metropolis-Hastings sampler with single-site Gaussian proposals.

    Generates field configurations distributed as exp(-S_E[phi]) for
    scalar phi^4 theory. Uses precomputed neighbor lookup tables and
    numpy arrays for fast inner loops.

    Args:
        action: Phi4Action instance defining the theory.
        config: MCConfig with sampling parameters.
    """

    def __init__(self, action: Phi4Action, config: MCConfig) -> None:
        self.action = action
        self.config = config
        self.rng = np.random.RandomState(config.seed)
        self._nsites = action.lattice.num_sites()

        # Precompute per-site neighbor list as a (nsites, 2*d) numpy array.
        # This is the key optimization: avoids the O(num_edges) mask scan
        # that was in delta_action on every single-site update.
        src = action._src.numpy()
        dst = action._dst.numpy()
        n_neighbors = 2 * action.lattice.dimension()
        self._neighbors = np.zeros((self._nsites, n_neighbors), dtype=np.int64)
        count = np.zeros(self._nsites, dtype=np.int64)
        for i in range(len(src)):
            s = src[i]
            self._neighbors[s, count[s]] = dst[i]
            count[s] += 1

        # Cache action parameters for numpy inner loop
        self._m_sq = action.m_sq
        self._lam = action.lam
        self._a = action.a
        self._ad = action._ad
        self._a2_inv = 1.0 / (action.a ** 2)

    def _delta_action_np(
        self, phi: np.ndarray, site: int, new_value: float
    ) -> float:
        """Compute Delta S for a single-site update, pure numpy.

        Delta S = a^d * [ kinetic_new - kinetic_old + mass_new - mass_old
                          + quartic_new - quartic_old ]
        """
        old_val = phi[site]
        neighbor_vals = phi[self._neighbors[site]]

        # Kinetic: 0.5 * sum_mu (phi(x) - phi(x+mu))^2 / a^2
        diff_old = old_val - neighbor_vals
        diff_new = new_value - neighbor_vals
        kinetic_delta = 0.5 * self._a2_inv * (
            np.dot(diff_new, diff_new) - np.dot(diff_old, diff_old)
        )

        # Mass: 0.5 * m^2 * phi^2
        mass_delta = 0.5 * self._m_sq * (new_value * new_value - old_val * old_val)

        # Quartic: lambda * phi^4
        quartic_delta = self._lam * (new_value**4 - old_val**4)

        return self._ad * (kinetic_delta + mass_delta + quartic_delta)

    def sweep(self, phi: torch.Tensor) -> tuple[torch.Tensor, float]:
        """One full Metropolis sweep over all sites.

        Works in numpy for speed, converts back to torch at the end.
        """
        phi_np = phi.numpy().copy()
        n_accepted = 0
        order = self.rng.permutation(self._nsites)
        step = self.config.step_size

        # Pre-draw all random numbers for the full sweep
        proposals = self.rng.normal(0, step, size=self._nsites)
        uniforms = self.rng.random(self._nsites)

        for idx in range(self._nsites):
            site = order[idx]
            new_value = phi_np[site] + proposals[idx]
            ds = self._delta_action_np(phi_np, site, new_value)

            if ds <= 0.0 or uniforms[idx] < np.exp(-ds):
                phi_np[site] = new_value
                n_accepted += 1

        return torch.from_numpy(phi_np), n_accepted / self._nsites

    def generate(self, n_configs: int) -> MCResult:
        """Generate decorrelated field configurations.

        Pipeline:
        1. Initialize from hot start
        2. Thermalize for n_thermalization sweeps
        3. Generate n_configs separated by n_sweeps_between sweeps
        """
        phi = 2.0 * torch.rand(self._nsites) - 1.0  # hot start

        # Thermalization
        t0 = time.time()
        logger.info(
            "Thermalizing for %d sweeps on %d-site lattice...",
            self.config.n_thermalization, self._nsites,
        )
        therm_acceptance = []
        for i in range(self.config.n_thermalization):
            phi, acc = self.sweep(phi)
            therm_acceptance.append(acc)
            if (i + 1) % 200 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                logger.info(
                    "  Thermalization %d/%d (%.1f sweeps/s, acc=%.3f)",
                    i + 1, self.config.n_thermalization, rate,
                    np.mean(therm_acceptance[-50:]),
                )

        avg_therm_acc = np.mean(therm_acceptance)
        logger.info(
            "Thermalization done in %.1fs. Avg acceptance: %.3f",
            time.time() - t0, avg_therm_acc,
        )

        # Generation
        configurations = torch.zeros(n_configs, self._nsites)
        actions = torch.zeros(n_configs)
        gen_acceptance = []

        t1 = time.time()
        logger.info("Generating %d configurations...", n_configs)
        for i in range(n_configs):
            for _ in range(self.config.n_sweeps_between):
                phi, acc = self.sweep(phi)
                gen_acceptance.append(acc)

            configurations[i] = phi.clone()
            with torch.no_grad():
                actions[i] = self.action(phi)

            if (i + 1) % 500 == 0:
                elapsed = time.time() - t1
                rate = (i + 1) / elapsed
                logger.info(
                    "  Generated %d/%d configs (%.1f configs/s)",
                    i + 1, n_configs, rate,
                )

        avg_gen_acc = np.mean(gen_acceptance)
        total_time = time.time() - t0
        logger.info(
            "Generation done in %.1fs. Avg acceptance: %.3f",
            time.time() - t1, avg_gen_acc,
        )
        logger.info("Total MC time: %.1fs", total_time)

        return MCResult(
            configurations=configurations,
            actions=actions,
            acceptance_rate=avg_gen_acc,
        )
