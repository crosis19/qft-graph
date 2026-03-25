"""Two-point correlation function head."""

from __future__ import annotations

import torch
import torch.nn as nn


class CorrelatorHead(nn.Module):
    """Computes learned two-point correlation functions from node embeddings.

    Uses the inner product of node embeddings as a proxy for the
    two-point function G(x, y) = <phi(x) phi(y)>.

    This head extracts correlation information directly from the
    learned representations, allowing the model to implicitly encode
    long-range correlations in the embedding space.

    Args:
        field_dim: Dimension of field node embeddings.
        hidden_dim: Projection dimension for computing correlations.
    """

    def __init__(self, field_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(field_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h_field: torch.Tensor) -> torch.Tensor:
        """Compute correlation matrix from field embeddings.

        Args:
            h_field: Field node embeddings (num_nodes, field_dim).

        Returns:
            Correlation matrix (num_nodes, num_nodes).
        """
        z = self.projector(h_field)  # (num_nodes, hidden_dim)
        z = nn.functional.normalize(z, dim=-1)
        return z @ z.T
