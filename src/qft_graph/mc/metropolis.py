"""Metropolis-Hastings sampler for scalar phi^4 theory.

Provides two implementations:
  - MetropolisSampler: Sequential site-by-site updates with numpy.
    Correct for any lattice, ~400 sweeps/s on 16x16.
  - CheckerboardSampler: Vectorized even/odd sublattice updates.
    Updates half the sites simultaneously. 10-50x faster for large lattices.
    Requires periodic BCs on a hypercubic lattice.

Both produce identical physics; the checkerboard sampler is preferred
for lattices >= 32x32.
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
        """Compute Delta S for a single-site update, pure numpy."""
        old_val = phi[site]
        neighbor_vals = phi[self._neighbors[site]]

        diff_old = old_val - neighbor_vals
        diff_new = new_value - neighbor_vals
        kinetic_delta = 0.5 * self._a2_inv * (
            np.dot(diff_new, diff_new) - np.dot(diff_old, diff_old)
        )
        mass_delta = 0.5 * self._m_sq * (new_value * new_value - old_val * old_val)
        quartic_delta = self._lam * (new_value**4 - old_val**4)

        return self._ad * (kinetic_delta + mass_delta + quartic_delta)

    def sweep(self, phi: torch.Tensor) -> tuple[torch.Tensor, float]:
        """One full Metropolis sweep over all sites."""
        phi_np = phi.numpy().copy()
        n_accepted = 0
        order = self.rng.permutation(self._nsites)
        step = self.config.step_size

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

    def generate(
        self,
        n_configs: int,
        initial_phi: torch.Tensor | None = None,
    ) -> MCResult:
        """Generate decorrelated field configurations.

        Args:
            n_configs: Number of configurations to generate.
            initial_phi: Optional initial field for warm-starting.
        """
        if initial_phi is not None:
            phi = initial_phi.clone()
        else:
            phi = 2.0 * torch.rand(self._nsites) - 1.0

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


class CheckerboardSampler(MCSampler):
    """Vectorized Metropolis sampler using checkerboard decomposition.

    On a hypercubic lattice with periodic BCs, sites can be partitioned
    into two sublattices (even and odd, by parity of coordinate sum)
    such that no two sites on the same sublattice are neighbors.
    This allows updating all even (or odd) sites simultaneously.

    Performance: 10-50x faster than sequential MetropolisSampler for
    large lattices (32x32+). The speedup comes from replacing the
    Python for-loop over sites with vectorized numpy operations.

    Args:
        action: Phi4Action instance defining the theory.
        config: MCConfig with sampling parameters.
    """

    def __init__(self, action: Phi4Action, config: MCConfig) -> None:
        self.action = action
        self.config = config
        self.rng = np.random.RandomState(config.seed)
        self._nsites = action.lattice.num_sites()
        self._dims = action.lattice.shape

        # Cache action parameters
        self._m_sq = action.m_sq
        self._lam = action.lam
        self._a = action.a
        self._ad = action._ad
        self._a2_inv = 1.0 / (action.a ** 2)

        # Build neighbor table: (nsites, 2*d)
        src = action._src.numpy()
        dst = action._dst.numpy()
        n_neighbors = 2 * action.lattice.dimension()
        self._neighbors = np.zeros((self._nsites, n_neighbors), dtype=np.int64)
        count = np.zeros(self._nsites, dtype=np.int64)
        for i in range(len(src)):
            s = src[i]
            self._neighbors[s, count[s]] = dst[i]
            count[s] += 1

        # Build checkerboard masks
        # Even sites: sum of multi-indices is even
        # Odd sites: sum of multi-indices is odd
        mi = action.lattice._multi_indices  # (nsites, ndim)
        parity = mi.sum(axis=1) % 2
        self._even_sites = np.where(parity == 0)[0]
        self._odd_sites = np.where(parity == 1)[0]

        logger.info(
            "CheckerboardSampler: %d even + %d odd sites on %s lattice",
            len(self._even_sites), len(self._odd_sites), self._dims,
        )

    def _vectorized_delta_action(
        self, phi: np.ndarray, sites: np.ndarray, new_values: np.ndarray
    ) -> np.ndarray:
        """Compute Delta S for a batch of site updates simultaneously.

        Args:
            phi: Current field config, shape (nsites,).
            sites: Indices of sites to update, shape (n_batch,).
            new_values: Proposed new values, shape (n_batch,).

        Returns:
            Delta S for each proposed update, shape (n_batch,).
        """
        old_vals = phi[sites]  # (n_batch,)
        # Neighbor values: (n_batch, 2*d)
        neighbor_vals = phi[self._neighbors[sites]]

        # Kinetic: 0.5/a² * Σ_nb [(new - nb)² - (old - nb)²]
        #        = 0.5/a² * Σ_nb [(new² - 2*new*nb) - (old² - 2*old*nb)]
        #        = 0.5/a² * Σ_nb [(new² - old²) - 2*(new - old)*nb]
        #        = 0.5/a² * [2d*(new² - old²) - 2*(new - old)*Σ_nb]
        # But direct approach is clearer:
        diff_old = old_vals[:, None] - neighbor_vals  # (n_batch, 2d)
        diff_new = new_values[:, None] - neighbor_vals  # (n_batch, 2d)
        kinetic_delta = 0.5 * self._a2_inv * (
            (diff_new**2).sum(axis=1) - (diff_old**2).sum(axis=1)
        )

        # Mass: 0.5 * m² * (new² - old²)
        mass_delta = 0.5 * self._m_sq * (new_values**2 - old_vals**2)

        # Quartic: λ * (new⁴ - old⁴)
        quartic_delta = self._lam * (new_values**4 - old_vals**4)

        return self._ad * (kinetic_delta + mass_delta + quartic_delta)

    def _half_sweep(self, phi: np.ndarray, sites: np.ndarray) -> tuple[np.ndarray, int]:
        """Update one sublattice (even or odd) in a single vectorized step.

        Args:
            phi: Current field config (modified in-place).
            sites: Indices of sites in this sublattice.

        Returns:
            (updated phi, number of accepted proposals).
        """
        n = len(sites)
        step = self.config.step_size

        proposals = phi[sites] + self.rng.normal(0, step, size=n)
        ds = self._vectorized_delta_action(phi, sites, proposals)

        # Metropolis accept/reject
        accept = (ds <= 0) | (self.rng.random(n) < np.exp(-np.clip(ds, -500, 500)))
        phi[sites[accept]] = proposals[accept]

        return phi, int(accept.sum())

    def sweep(self, phi: torch.Tensor) -> tuple[torch.Tensor, float]:
        """One full checkerboard sweep: update even sites, then odd sites."""
        phi_np = phi.numpy().copy()

        # Update even sublattice
        phi_np, n_acc_even = self._half_sweep(phi_np, self._even_sites)
        # Update odd sublattice
        phi_np, n_acc_odd = self._half_sweep(phi_np, self._odd_sites)

        total_acc = (n_acc_even + n_acc_odd) / self._nsites
        return torch.from_numpy(phi_np), total_acc

    def generate(
        self,
        n_configs: int,
        initial_phi: torch.Tensor | None = None,
    ) -> MCResult:
        """Generate decorrelated field configurations.

        Args:
            n_configs: Number of configurations to generate.
            initial_phi: Optional initial field for warm-starting.
        """
        if initial_phi is not None:
            phi = initial_phi.clone()
        else:
            phi = 2.0 * torch.rand(self._nsites) - 1.0

        # Thermalization
        t0 = time.time()
        logger.info(
            "Thermalizing for %d sweeps on %d-site lattice (checkerboard)...",
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

        avg_therm_acc = np.mean(therm_acceptance) if therm_acceptance else 0.0
        logger.info(
            "Thermalization done in %.1fs. Avg acceptance: %.3f",
            time.time() - t0, avg_therm_acc,
        )

        # Generation
        configurations = torch.zeros(n_configs, self._nsites)
        actions = torch.zeros(n_configs)
        gen_acceptance = []

        t1 = time.time()
        logger.info("Generating %d configurations (checkerboard)...", n_configs)
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

        avg_gen_acc = np.mean(gen_acceptance) if gen_acceptance else 0.0
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


def create_sampler(action: Phi4Action, config: MCConfig) -> MCSampler:
    """Factory function to create the best sampler for the lattice size.

    Uses CheckerboardSampler for lattices >= 32x32 (where vectorization
    gives a significant speedup), MetropolisSampler otherwise.

    Args:
        action: Phi4Action instance.
        config: MCConfig with sampling parameters.

    Returns:
        MCSampler instance.
    """
    nsites = action.lattice.num_sites()
    if nsites >= 1024:  # 32x32
        logger.info(
            "Using CheckerboardSampler for %d-site lattice (vectorized)",
            nsites,
        )
        return CheckerboardSampler(action, config)
    else:
        logger.info(
            "Using MetropolisSampler for %d-site lattice (sequential)",
            nsites,
        )
        return MetropolisSampler(action, config)
