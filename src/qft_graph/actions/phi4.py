"""Euclidean phi^4 action on a hypercubic lattice."""

from __future__ import annotations

import torch

from qft_graph.actions.base import Action
from qft_graph.config import ScalarFieldConfig
from qft_graph.lattice.hypercubic import HypercubicLattice


class Phi4Action(Action):
    """Euclidean phi^4 action on a hypercubic lattice.

    S_E[phi] = sum_x a^d [ (1/2) sum_mu (phi(x+mu) - phi(x))^2 / a^2
                            + (1/2) m^2 phi(x)^2
                            + lambda phi(x)^4 ]

    This is the standard lattice discretization of the continuum action
    for real scalar field theory with quartic self-interaction.

    In 2D with the right choice of m^2 and lambda, this theory lies in the
    Ising universality class, with critical exponent nu = 1.
    """

    def __init__(self, lattice: HypercubicLattice, config: ScalarFieldConfig) -> None:
        self.lattice = lattice
        self.m_sq = config.mass_squared
        self.lam = config.coupling
        self.a = lattice.lattice_spacing()
        self.d = lattice.dimension()
        self._ad = self.a**self.d

        # Precompute neighbor index structure for vectorized action
        src, dst = lattice.neighbor_pairs()
        self._src = src
        self._dst = dst

        # Number of forward neighbors per site = d (for kinetic term, avoid double-counting)
        self._n_forward_edges = lattice.num_sites() * self.d

    def __call__(self, phi: torch.Tensor) -> torch.Tensor:
        """Compute total Euclidean action S_E[phi].

        Args:
            phi: Field configuration of shape (num_sites,).

        Returns:
            Scalar tensor.
        """
        return self.local_action(phi).sum()

    def local_action(self, phi: torch.Tensor) -> torch.Tensor:
        """Compute per-site action density.

        The kinetic term uses only forward differences to avoid double-counting:
        (1/2) sum_{mu>0} (phi(x+mu) - phi(x))^2 / a^2

        Args:
            phi: Field configuration of shape (num_sites,).

        Returns:
            Per-site action of shape (num_sites,).
        """
        nsites = self.lattice.num_sites()

        # Kinetic term: forward differences only (first half of edge list)
        # neighbor_pairs returns +mu then -mu for each direction
        fwd_src = self._src[: self._n_forward_edges]
        fwd_dst = self._dst[: self._n_forward_edges]
        diff = phi[fwd_dst] - phi[fwd_src]
        # Scatter kinetic energy to source sites
        kinetic = torch.zeros(nsites, dtype=phi.dtype, device=phi.device)
        kinetic.scatter_add_(0, fwd_src, 0.5 * diff**2 / self.a**2)

        # Mass term: (1/2) m^2 phi^2
        mass = 0.5 * self.m_sq * phi**2

        # Quartic term: lambda phi^4
        quartic = self.lam * phi**4

        return self._ad * (kinetic + mass + quartic)

    def force(self, phi: torch.Tensor) -> torch.Tensor:
        """Compute molecular dynamics force -dS/dphi.

        -dS/dphi(x) = a^d [ sum_mu (phi(x+mu) + phi(x-mu) - 2*phi(x)) / a^2
                             - m^2 phi(x) - 4 lambda phi(x)^3 ]

        Args:
            phi: Field configuration of shape (num_sites,).

        Returns:
            Force tensor of shape (num_sites,).
        """
        nsites = self.lattice.num_sites()

        # Laplacian: sum_mu [phi(x+mu) + phi(x-mu) - 2*phi(x)] / a^2
        # Using all edges (both directions), sum phi(neighbor) at each site
        neighbor_sum = torch.zeros(nsites, dtype=phi.dtype, device=phi.device)
        neighbor_sum.scatter_add_(0, self._src, phi[self._dst])
        # Each site has 2d neighbors, so the -2*phi term contributes -2d*phi
        laplacian = (neighbor_sum - 2 * self.d * phi) / self.a**2

        force = self._ad * (laplacian - self.m_sq * phi - 4.0 * self.lam * phi**3)
        return force

    def delta_action(self, phi: torch.Tensor, site: int, new_value: float) -> torch.Tensor:
        """Compute change in action from a single-site update.

        Used by Metropolis sampler for efficient acceptance computation.
        Only recomputes terms involving the changed site.

        Args:
            phi: Current field configuration of shape (num_sites,).
            site: Index of site being updated.
            new_value: Proposed new field value at site.

        Returns:
            Scalar tensor Delta S = S_new - S_old.
        """
        old_val = phi[site]

        # Find neighbors of this site
        mask_src = self._src == site
        neighbor_indices = self._dst[mask_src]
        neighbor_vals = phi[neighbor_indices]
        neighbor_sum = neighbor_vals.sum()

        # Old local action contribution
        s_old = self._ad * (
            0.5 * ((neighbor_sum - 2 * self.d * old_val) * (-old_val)) / self.a**2
            + 0.5 * self.m_sq * old_val**2
            + self.lam * old_val**4
        )

        # Actually, let's compute it cleanly
        # Local action at site x: kinetic involves phi(x) and all its neighbors
        # S_local(x) = a^d * [ sum_mu (phi(x) - phi(x+mu))^2 / (2*a^2) + m^2/2 * phi(x)^2 + lam * phi(x)^4 ]
        # But this double counts kinetic with neighbors. For delta, we need:
        # Delta S = sum over all terms involving phi(x)

        new_val_t = torch.tensor(new_value, dtype=phi.dtype, device=phi.device)

        # Terms involving phi(x): kinetic with each neighbor + mass + quartic
        kinetic_old = 0.5 * ((old_val - neighbor_vals) ** 2).sum() / self.a**2
        kinetic_new = 0.5 * ((new_val_t - neighbor_vals) ** 2).sum() / self.a**2

        mass_old = 0.5 * self.m_sq * old_val**2
        mass_new = 0.5 * self.m_sq * new_val_t**2

        quartic_old = self.lam * old_val**4
        quartic_new = self.lam * new_val_t**4

        delta = self._ad * (
            (kinetic_new - kinetic_old)
            + (mass_new - mass_old)
            + (quartic_new - quartic_old)
        )
        return delta
