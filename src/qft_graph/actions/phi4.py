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

        # Precompute per-site neighbor list: (nsites, 2*d) for fast delta_action
        n_neighbors = 2 * self.d
        self._neighbor_table = torch.zeros(
            lattice.num_sites(), n_neighbors, dtype=torch.long
        )
        count = torch.zeros(lattice.num_sites(), dtype=torch.long)
        for i in range(len(src)):
            s = src[i].item()
            self._neighbor_table[s, count[s]] = dst[i]
            count[s] += 1

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

        Uses symmetric kinetic term: assign half of each link's energy to each end.
        For each edge (x, x+mu): each endpoint gets (1/4)(phi(x+mu)-phi(x))^2/a^2.
        Summing over all directed edges with factor 1/4 = (1/2 for link) * (1/2 split).

        Args:
            phi: Field configuration of shape (num_sites,).

        Returns:
            Per-site action of shape (num_sites,).
        """
        nsites = self.lattice.num_sites()

        # Kinetic: use ALL directed edges, factor 1/4 per edge
        # Each undirected link {x, x+mu} appears as two directed edges.
        # The link energy is (1/2)(dphi)^2/a^2.
        # With two directed copies, each carrying factor 1/4, the source
        # site accumulates (1/4)(dphi)^2/a^2 from each direction.
        # Site x has 2d directed edges as source -> gets d * (1/2)(dphi)^2/a^2 total. Correct.
        diff = phi[self._dst] - phi[self._src]
        kinetic = torch.zeros(nsites, dtype=phi.dtype, device=phi.device)
        kinetic.scatter_add_(0, self._src, 0.25 * diff**2 / self.a**2)

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
        neighbor_vals = phi[self._neighbor_table[site]]

        new_val_t = torch.tensor(new_value, dtype=phi.dtype, device=phi.device)

        # Kinetic: site x "owns" 1/4 of each directed edge where it's the source.
        # With 2d directed edges from x, that's (1/4)*sum_neighbors (phi(x)-phi(nb))^2/a^2.
        # But changing phi(x) also affects edges where x is the *destination*.
        # By symmetry of the undirected links, the total change is:
        # Delta_kinetic = (1/2) * sum_neighbors [(new-nb)^2 - (old-nb)^2] / a^2
        diff_old = old_val - neighbor_vals
        diff_new = new_val_t - neighbor_vals
        kinetic_delta = 0.5 * (diff_new.dot(diff_new) - diff_old.dot(diff_old)) / self.a**2

        # Mass: 0.5 * m^2 * phi^2
        mass_delta = 0.5 * self.m_sq * (new_val_t**2 - old_val**2)

        # Quartic: lambda * phi^4
        quartic_delta = self.lam * (new_val_t**4 - old_val**4)

        return self._ad * (kinetic_delta + mass_delta + quartic_delta)
