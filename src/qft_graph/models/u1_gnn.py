"""Heterogeneous GNN for the Phase II quenched U(1) experiment (task A-4).

One model class consumes all three A-3 graph variants:
- link_nodes (A): gauge-node graphs, GaugeThreeStageBlock message passing
- edge_features (B): spacetime-only graphs, links on adjacency edges
- invariant_oracle (C): spacetime-only graphs, plaquette trig on nodes

Heads (plan §3, task A-4):
- action: sum over st nodes of a per-site MLP (Phase I Eq. 8 pattern);
  output key stays "energy" (see CLAUDE.md terminology note)
- wilson: one PooledReadoutHead per loop size, target per-config W
  (decision record #3: raw W uniformly across all (beta, size) cells)
- q: PooledReadoutHead regression; exact-integer accuracy after rounding
  is computed by training.metrics.q_rounded_accuracy
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from qft_graph.config import ModelConfig
from qft_graph.graphs.builder import U1GaugeGraphBuilder
from qft_graph.graphs.edge_types import ADJACENT
from qft_graph.models.encoders.spacetime import (
    EdgeEncoder,
    FieldEncoder,
    SpacetimeEncoder,
)
from qft_graph.models.heads.action import ActionHead
from qft_graph.models.heads.pooled import PooledReadoutHead
from qft_graph.models.message_passing.gauge_stage import (
    GaugeThreeStageBlock,
    SpacetimeOnlyBlock,
)

# Raw input feature dims produced by U1GaugeGraphBuilder, per variant
_ST_INPUT_DIM = {"link_nodes": 1, "edge_features": 2, "invariant_oracle": 3}
_EDGE_INPUT_DIM = {"link_nodes": 2, "edge_features": 4, "invariant_oracle": 2}
_GAUGE_INPUT_DIM = 5  # [cos, sin, onehot(mu), beta]

DEFAULT_WILSON_LOOPS = ("1x1", "2x2", "2x4", "3x3", "4x4")


class U1GaugeGNN(nn.Module):
    """GNN over U(1) gauge graphs, all three A-3 variants.

    Args:
        config: ModelConfig (hidden_dim, n_mp_blocks, encoder_layers,
            dropout, activation are used; readout is ignored — heads are
            controlled by wilson_loops / predict_q here).
        variant: "link_nodes" / "edge_features" / "invariant_oracle"
            (aliases "A"/"B"/"C"), must match the graph builder variant.
        wilson_loops: Loop-size names, one pooled head each. Empty
            disables Wilson heads.
        predict_q: Whether to attach the topological-charge head.
        lattice_spacing: Physical lattice spacing for the action head.
        global_dim: Dimension of data.globals ([[beta]] from the builder).
    """

    def __init__(
        self,
        config: ModelConfig,
        variant: str = "link_nodes",
        wilson_loops: tuple[str, ...] = DEFAULT_WILSON_LOOPS,
        predict_q: bool = True,
        lattice_spacing: float = 1.0,
        global_dim: int = 1,
    ) -> None:
        super().__init__()
        if variant not in U1GaugeGraphBuilder.VARIANT_ALIASES:
            raise ValueError(
                f"Unknown variant: {variant!r}. "
                f"Expected one of {sorted(set(U1GaugeGraphBuilder.VARIANT_ALIASES))}"
            )
        self.variant = U1GaugeGraphBuilder.VARIANT_ALIASES[variant]
        self.config = config
        self.lattice_spacing = lattice_spacing
        self.global_dim = global_dim
        self.wilson_loops = tuple(wilson_loops)
        self.predict_q = predict_q
        self._has_gauge_nodes = self.variant == "link_nodes"
        h = config.hidden_dim

        # --- Encoders ---
        self.st_encoder = SpacetimeEncoder(
            input_dim=_ST_INPUT_DIM[self.variant],
            hidden_dim=h,
            n_layers=config.encoder_layers,
            activation=config.activation,
        )
        self.edge_encoder = EdgeEncoder(
            input_dim=_EDGE_INPUT_DIM[self.variant],
            hidden_dim=h,
            activation=config.activation,
        )
        self.gauge_encoder = (
            FieldEncoder(
                input_dim=_GAUGE_INPUT_DIM,
                hidden_dim=h,
                n_layers=config.encoder_layers,
                activation=config.activation,
            )
            if self._has_gauge_nodes
            else None
        )

        # --- Message passing ---
        if self._has_gauge_nodes:
            # The final block's st->gauge stage is dead unless a pooled
            # head reads the gauge embeddings afterwards.
            needs_gauge_out = bool(wilson_loops) or predict_q
            self.mp_blocks = nn.ModuleList(
                [
                    GaugeThreeStageBlock(
                        gauge_dim=h,
                        st_dim=h,
                        edge_dim=h,
                        hidden_dim=h,
                        dropout=config.dropout,
                        update_gauge=(i < config.n_mp_blocks - 1) or needs_gauge_out,
                    )
                    for i in range(config.n_mp_blocks)
                ]
            )
        else:
            self.mp_blocks = nn.ModuleList(
                [
                    SpacetimeOnlyBlock(
                        st_dim=h,
                        edge_dim=h,
                        hidden_dim=h,
                        dropout=config.dropout,
                    )
                    for _ in range(config.n_mp_blocks)
                ]
            )

        # --- Heads ---
        # Action: per-site MLP over st nodes only (plan A-4 head i); the
        # gauge content reaches st embeddings through message passing.
        self.action_head = ActionHead(
            st_dim=h,
            field_dims={},
            hidden_dim=h,
            volume_scaling=config.volume_scaling,
            global_dim=global_dim,
        )

        pooled_dims = {"spacetime": h}
        if self._has_gauge_nodes:
            pooled_dims["gauge"] = h  # mean-pool over BOTH node types (plan)
        self.wilson_heads = nn.ModuleDict(
            {
                name: PooledReadoutHead(pooled_dims, h, global_dim=global_dim)
                for name in self.wilson_loops
            }
        )
        self.q_head = (
            PooledReadoutHead(pooled_dims, h, global_dim=global_dim)
            if predict_q
            else None
        )

    def forward(self, data: HeteroData) -> dict[str, torch.Tensor]:
        """Encode -> message pass -> readout.

        Clones the input first so raw features are never overwritten
        (same rationale as HeteroGNN.forward).

        Returns:
            Dict with keys:
              - "energy": (batch,) total action prediction
              - "local_energy": (num_st_nodes, 1) per-site density
              - "wilson_<name>": (batch,) per configured loop size
              - "q": (batch,) topological charge regression (if enabled)
        """
        data = data.clone()

        # --- Encode ---
        data["spacetime"].x = self.st_encoder(data["spacetime"].x)
        data[ADJACENT].edge_attr = self.edge_encoder(data[ADJACENT].edge_attr)
        if self._has_gauge_nodes:
            data["gauge"].x = self.gauge_encoder(data["gauge"].x)

        # --- Message passing ---
        for block in self.mp_blocks:
            data = block(data)

        # --- Readout ---
        st_batch = data["spacetime"].get("batch", None)
        globals_ = data.globals if self.global_dim > 0 else None

        output: dict[str, torch.Tensor] = {}
        total_e, local_e = self.action_head(
            data["spacetime"].x,
            {},
            self.lattice_spacing,
            2,
            st_batch,
            globals_=globals_,
        )
        if st_batch is None:
            # ActionHead's unbatched branch returns a 0-dim scalar; keep the
            # documented (batch,) contract (batch = 1) so A-5 gauge-copy
            # loops can torch.cat every output key uniformly.
            total_e = total_e.reshape(1)
        output["energy"] = total_e
        output["local_energy"] = local_e

        h_nodes = {"spacetime": data["spacetime"].x}
        batches: dict[str, torch.Tensor | None] = {"spacetime": st_batch}
        if self._has_gauge_nodes:
            h_nodes["gauge"] = data["gauge"].x
            batches["gauge"] = data["gauge"].get("batch", None)

        for name in self.wilson_loops:
            output[f"wilson_{name}"] = self.wilson_heads[name](
                h_nodes, batches, globals_=globals_
            )
        if self.q_head is not None:
            output["q"] = self.q_head(h_nodes, batches, globals_=globals_)

        return output
