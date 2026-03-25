"""Three-stage message passing block composing field->ST->ST->field."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from qft_graph.graphs.edge_types import ADJACENT, inhabits_edge, inhabits_inv_edge
from qft_graph.models.message_passing.field_to_st import FieldToSpacetime
from qft_graph.models.message_passing.st_to_field import SpacetimeToField
from qft_graph.models.message_passing.st_to_st import SpacetimeToSpacetime


class ThreeStageBlock(nn.Module):
    """One complete field -> spacetime -> spacetime -> field message passing cycle.

    This is the core architectural unit of the heterogeneous GNN.
    Multiple blocks are stacked to increase the effective receptive field.

    Each block includes:
    - Residual connections to preserve high-frequency (short-range) information
    - Layer normalization per node type to prevent over-smoothing
    - Dropout for regularization

    Args:
        field_dims: Dict mapping field type names to their hidden dimensions.
        st_dim: Spacetime node hidden dimension.
        edge_dim: Edge feature dimension (encoded direction vectors).
        hidden_dim: Internal MLP hidden dimension.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        field_dims: dict[str, int],
        st_dim: int,
        edge_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.field_types = list(field_dims.keys())

        # Stage 1: Field -> Spacetime (one per field type)
        self.field_to_st = nn.ModuleDict(
            {
                fname: FieldToSpacetime(fdim, st_dim, hidden_dim)
                for fname, fdim in field_dims.items()
            }
        )

        # Stage 2: Spacetime -> Spacetime
        self.st_to_st = SpacetimeToSpacetime(st_dim, edge_dim, hidden_dim)

        # Stage 3: Spacetime -> Field (one per field type)
        self.st_to_field = nn.ModuleDict(
            {
                fname: SpacetimeToField(st_dim, fdim, hidden_dim)
                for fname, fdim in field_dims.items()
            }
        )

        # Layer normalization
        self.st_norm = nn.LayerNorm(st_dim)
        self.field_norms = nn.ModuleDict(
            {fname: nn.LayerNorm(fdim) for fname, fdim in field_dims.items()}
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, data: HeteroData) -> HeteroData:
        """Apply one 3-stage message passing cycle.

        Modifies node features in-place on the HeteroData object.

        Args:
            data: HeteroData with encoded node/edge features.

        Returns:
            Same HeteroData with updated node features.
        """
        x_st = data["spacetime"].x
        adj_edge_index = data[ADJACENT].edge_index
        adj_edge_attr = data[ADJACENT].edge_attr

        # --- Stage 1: Field -> Spacetime ---
        st_update = torch.zeros_like(x_st)
        for fname in self.field_types:
            x_field = data[fname].x
            inh_edge = inhabits_edge(fname)
            edge_index = data[inh_edge].edge_index

            msg = self.field_to_st[fname](x_field, x_st, edge_index)
            st_update = st_update + msg

        # Residual + norm
        x_st = self.st_norm(x_st + self.dropout(st_update))

        # --- Stage 2: Spacetime -> Spacetime ---
        st_msg = self.st_to_st(x_st, adj_edge_index, adj_edge_attr)
        x_st = self.st_norm(x_st + self.dropout(st_msg))

        # --- Stage 3: Spacetime -> Field ---
        for fname in self.field_types:
            x_field = data[fname].x
            inv_edge = inhabits_inv_edge(fname)
            edge_index = data[inv_edge].edge_index

            field_msg = self.st_to_field[fname](x_st, x_field, edge_index)
            x_field = self.field_norms[fname](x_field + self.dropout(field_msg))
            data[fname].x = x_field

        data["spacetime"].x = x_st
        return data
