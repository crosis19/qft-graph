"""Real scalar field phi(x) for Phase 1 (phi^4 theory)."""

from __future__ import annotations

import torch

from qft_graph.fields.base import Field


class ScalarField(Field):
    """Real scalar field phi(x) in R.

    The simplest quantum field: one real degree of freedom per lattice site.
    Used in Phase 1 for phi^4 theory in 2D (Ising universality class).
    """

    def dof_per_site(self) -> int:
        return 1

    def node_type_name(self) -> str:
        return "scalar"

    def node_features(self, configuration: torch.Tensor) -> torch.Tensor:
        """Convert scalar configuration to node features.

        Args:
            configuration: Field values, shape (num_sites,) or (num_sites, 1).

        Returns:
            Feature tensor of shape (num_sites, 1).
        """
        if configuration.dim() == 1:
            return configuration.unsqueeze(-1)
        return configuration

    def initialize(self, num_sites: int, mode: str = "hot") -> torch.Tensor:
        """Generate initial scalar field configuration.

        Args:
            num_sites: Number of lattice sites.
            mode:
                'hot': Uniform random in [-1, 1] (disordered).
                'cold': All sites set to +1 (ordered).
                'gaussian': Normal(0, 1) samples.

        Returns:
            Configuration tensor of shape (num_sites,).
        """
        if mode == "hot":
            return 2.0 * torch.rand(num_sites) - 1.0
        elif mode == "cold":
            return torch.ones(num_sites)
        elif mode == "gaussian":
            return torch.randn(num_sites)
        else:
            raise ValueError(f"Unknown initialization mode: {mode}")
