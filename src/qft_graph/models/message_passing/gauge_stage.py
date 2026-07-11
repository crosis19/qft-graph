"""Message-passing blocks for the Phase II U(1) graph variants (task A-4).

Variant A (link-nodes) uses the three-stage analog specified in plan §3:
gauge -> st over BOTH typed endpoint edges, st -> st, st -> gauge over both
reverses, with residual + LayerNorm per stage as in Phase I. Variants B and
C carry all field content on spacetime nodes/edges, so their block is the
st -> st stage alone.

Both blocks reuse the Phase I MessagePassing modules, which are generic
over the edge_index they are given.
"""

from __future__ import annotations

import torch.nn as nn
from torch_geometric.data import HeteroData

from qft_graph.graphs.edge_types import (
    ADJACENT,
    GAUGE_ENDS_AT,
    GAUGE_STARTS_AT,
    ORIGIN_OF,
    TARGET_OF,
)
from qft_graph.models.message_passing.field_to_st import FieldToSpacetime
from qft_graph.models.message_passing.st_to_field import SpacetimeToField
from qft_graph.models.message_passing.st_to_st import SpacetimeToSpacetime


class GaugeThreeStageBlock(nn.Module):
    """One gauge -> spacetime -> spacetime -> gauge message-passing cycle.

    A link attaches to two sites, so stage 1 aggregates over the
    starts_at and ends_at edge types with separate message MLPs (the
    relation itself tells a site whether it is the origin or target of
    the link — the analog of the +mu/-mu distinction), and stage 3
    mirrors this with origin_of and target_of.

    A plaquette is a length-4 link-site alternating cycle, so plaquette
    information becomes available within 2 blocks (plan §3, Variant A).

    Args:
        gauge_dim: Gauge (link) node hidden dimension.
        st_dim: Spacetime node hidden dimension.
        edge_dim: Encoded adjacency edge feature dimension.
        hidden_dim: Internal MLP hidden dimension.
        dropout: Dropout probability.
        update_gauge: Whether to build/apply stage 3. The final block of a
            model with no gauge-consuming head (action-only ablation) sets
            this False — its stage-3 output would be dead, and the unused
            parameters would inflate the A/B/C params comparison.
    """

    def __init__(
        self,
        gauge_dim: int,
        st_dim: int,
        edge_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        update_gauge: bool = True,
    ) -> None:
        super().__init__()
        self.update_gauge = update_gauge

        # Stage 1: gauge -> st, one module per endpoint relation
        self.gauge_to_st = nn.ModuleDict(
            {
                "starts_at": FieldToSpacetime(gauge_dim, st_dim, hidden_dim),
                "ends_at": FieldToSpacetime(gauge_dim, st_dim, hidden_dim),
            }
        )

        # Stage 2: st -> st
        self.st_to_st = SpacetimeToSpacetime(st_dim, edge_dim, hidden_dim)

        # Stage 3: st -> gauge, one module per reverse relation
        if update_gauge:
            self.st_to_gauge = nn.ModuleDict(
                {
                    "origin_of": SpacetimeToField(st_dim, gauge_dim, hidden_dim),
                    "target_of": SpacetimeToField(st_dim, gauge_dim, hidden_dim),
                }
            )
            self.gauge_norm = nn.LayerNorm(gauge_dim)

        self.st_norm = nn.LayerNorm(st_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data: HeteroData) -> HeteroData:
        x_st = data["spacetime"].x
        x_gauge = data["gauge"].x

        # --- Stage 1: gauge -> st (both typed endpoint edges) ---
        st_update = self.gauge_to_st["starts_at"](
            x_gauge, x_st, data[GAUGE_STARTS_AT].edge_index
        ) + self.gauge_to_st["ends_at"](
            x_gauge, x_st, data[GAUGE_ENDS_AT].edge_index
        )
        x_st = self.st_norm(x_st + self.dropout(st_update))

        # --- Stage 2: st -> st ---
        st_msg = self.st_to_st(
            x_st, data[ADJACENT].edge_index, data[ADJACENT].edge_attr
        )
        x_st = self.st_norm(x_st + self.dropout(st_msg))

        # --- Stage 3: st -> gauge (both reverses) ---
        if self.update_gauge:
            gauge_update = self.st_to_gauge["origin_of"](
                x_st, x_gauge, data[ORIGIN_OF].edge_index
            ) + self.st_to_gauge["target_of"](
                x_st, x_gauge, data[TARGET_OF].edge_index
            )
            data["gauge"].x = self.gauge_norm(x_gauge + self.dropout(gauge_update))

        data["spacetime"].x = x_st
        return data


class SpacetimeOnlyBlock(nn.Module):
    """st -> st message passing with residual + LayerNorm, for graphs whose
    field content lives on spacetime nodes or adjacency edges (Variants B, C).

    Args:
        st_dim: Spacetime node hidden dimension.
        edge_dim: Encoded adjacency edge feature dimension.
        hidden_dim: Internal MLP hidden dimension.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        st_dim: int,
        edge_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.st_to_st = SpacetimeToSpacetime(st_dim, edge_dim, hidden_dim)
        self.st_norm = nn.LayerNorm(st_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data: HeteroData) -> HeteroData:
        x_st = data["spacetime"].x
        st_msg = self.st_to_st(
            x_st, data[ADJACENT].edge_index, data[ADJACENT].edge_attr
        )
        data["spacetime"].x = self.st_norm(x_st + self.dropout(st_msg))
        return data
