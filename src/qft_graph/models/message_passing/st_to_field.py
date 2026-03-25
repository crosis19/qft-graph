"""Stage 3: Spacetime -> Field message passing."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class SpacetimeToField(MessagePassing):
    """Stage 3 of the 3-stage message passing: spacetime nodes send
    updated geometric information back to field nodes via reverse
    'inhabits' edges.

    This enables each field node to 'see' its geometric neighborhood
    through the spacetime propagation, coupling field content to
    the lattice geometry — the key physics of QFT on a lattice.

    Args:
        st_dim: Dimension of spacetime node representations.
        field_dim: Dimension of field node representations.
        hidden_dim: Internal MLP hidden dimension.
    """

    def __init__(self, st_dim: int, field_dim: int, hidden_dim: int) -> None:
        super().__init__(aggr="add", flow="source_to_target")
        self.msg_mlp = nn.Sequential(
            nn.Linear(st_dim + field_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, field_dim),
        )

    def forward(
        self,
        x_st: torch.Tensor,
        x_field: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Compute spacetime-to-field messages.

        Args:
            x_st: Spacetime node features (num_st_nodes, st_dim).
            x_field: Field node features (num_field_nodes, field_dim).
            edge_index: (2, num_edges) with [st_idx, field_idx].

        Returns:
            Updated field features (num_field_nodes, field_dim).
        """
        return self.propagate(edge_index, x=x_st, x_field=x_field)

    def message(self, x_j: torch.Tensor, x_field_i: torch.Tensor) -> torch.Tensor:
        """Construct message from spacetime node j to field node i."""
        return self.msg_mlp(torch.cat([x_j, x_field_i], dim=-1))
