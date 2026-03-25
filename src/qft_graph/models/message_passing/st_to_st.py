"""Stage 2: Spacetime -> Spacetime message passing."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class SpacetimeToSpacetime(MessagePassing):
    """Stage 2 of the 3-stage message passing: spacetime nodes propagate
    information along lattice edges (nearest-neighbor adjacency).

    Each message passing hop corresponds to one lattice derivative step.
    Multiple hops build up the finite-difference Laplacian needed for
    the kinetic term in the action. Edge features carry direction vectors,
    enabling directional derivatives.

    Args:
        st_dim: Dimension of spacetime node representations.
        edge_dim: Dimension of edge features (direction vectors).
        hidden_dim: Internal MLP hidden dimension.
    """

    def __init__(self, st_dim: int, edge_dim: int, hidden_dim: int) -> None:
        super().__init__(aggr="add", flow="source_to_target")
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * st_dim + edge_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, st_dim),
        )

    def forward(
        self,
        x_st: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Propagate information along lattice edges.

        Args:
            x_st: Spacetime node features (num_nodes, st_dim).
            edge_index: (2, num_edges) adjacency.
            edge_attr: (num_edges, edge_dim) direction vectors.

        Returns:
            Updated spacetime features (num_nodes, st_dim).
        """
        return self.propagate(edge_index, x=x_st, edge_attr=edge_attr)

    def message(
        self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """Construct message from neighbor j to node i, incorporating edge direction."""
        return self.msg_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))
