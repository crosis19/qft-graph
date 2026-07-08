"""Real scalar field phi(x) for Phase 1 (phi^4 theory)."""

from __future__ import annotations

import torch

from qft_graph.fields.base import Field


class ScalarField(Field):
    """Real scalar field phi(x) in R.

    The simplest quantum field: one real degree of freedom per lattice site.
    Used in Phase 1 for phi^4 theory in 2D (Ising universality class).

    Args:
        couplings: Optional (m_squared, coupling_lambda). When given, node
            features become [phi, m2, lambda] and the couplings are exposed
            as graph-level global features, so one model can train jointly
            across coupling values (task P1-3).
    """

    def __init__(self, couplings: tuple[float, float] | None = None) -> None:
        self._couplings = couplings

    def dof_per_site(self) -> int:
        return 1

    def node_feature_dim(self) -> int:
        return 1 if self._couplings is None else 3

    def global_features(self) -> torch.Tensor | None:
        if self._couplings is None:
            return None
        return torch.tensor([list(self._couplings)], dtype=torch.float32)

    def node_type_name(self) -> str:
        return "scalar"

    def node_features(self, configuration: torch.Tensor) -> torch.Tensor:
        """Convert scalar configuration to node features.

        Args:
            configuration: Field values, shape (num_sites,) or (num_sites, 1).

        Returns:
            Feature tensor of shape (num_sites, 1), or (num_sites, 3) with
            [phi, m2, lambda] when couplings were provided.
        """
        if configuration.dim() == 1:
            configuration = configuration.unsqueeze(-1)
        if self._couplings is None:
            return configuration
        const = configuration.new_tensor(list(self._couplings)).expand(
            configuration.shape[0], 2
        )
        return torch.cat([configuration, const], dim=-1)

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
