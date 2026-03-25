"""Stage 1: Field -> Spacetime message passing."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class FieldToSpacetime(MessagePassing):
    """Stage 1 of the 3-stage message passing: field nodes send information
    to their host spacetime nodes via 'inhabits' edges.

    In Phase 1 (scalar only), this aggregates the single scalar field
    representation at each spacetime site. In later phases with multiple
    field types (gauge + fermion), this aggregates all field content
    into a combined geometric representation at each site.

    Args:
        field_dim: Dimension of field node representations.
        st_dim: Dimension of spacetime node representations.
        hidden_dim: Internal MLP hidden dimension.
    """

    def __init__(self, field_dim: int, st_dim: int, hidden_dim: int) -> None:
        super().__init__(aggr="add", flow="source_to_target")
        self.msg_mlp = nn.Sequential(
            nn.Linear(field_dim + st_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, st_dim),
        )

    def forward(
        self,
        x_field: torch.Tensor,
        x_st: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Compute field-to-spacetime messages.

        Args:
            x_field: Field node features (num_field_nodes, field_dim).
            x_st: Spacetime node features (num_st_nodes, st_dim).
            edge_index: (2, num_edges) with [field_idx, st_idx].

        Returns:
            Updated spacetime features (num_st_nodes, st_dim).
        """
        return self.propagate(edge_index, x=x_field, x_st=x_st)

    def message(self, x_j: torch.Tensor, x_st_i: torch.Tensor) -> torch.Tensor:
        """Construct message from field node j to spacetime node i."""
        return self.msg_mlp(torch.cat([x_j, x_st_i], dim=-1))
