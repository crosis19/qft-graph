"""Homogeneous GNN baseline: field values concatenated onto spacetime nodes.

This is the standard approach where all information lives on a single node type.
It serves as the primary ablation against the heterogeneous bipartite architecture.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter


class HomogeneousGNN(nn.Module):
    """Homogeneous GNN that merges field values into spacetime node features.

    Instead of separating geometry and field content into distinct node types,
    this baseline concatenates [coordinates, spacing, phi] as features on a
    single node type and performs standard message passing over the lattice
    adjacency graph.

    Includes a skip connection from the initial encoding to the readout
    (standard JK-Net / DeepGCN practice) to prevent the phi signal from
    being destroyed by over-smoothing during message passing.

    Args:
        lattice_dim: Spatial dimension (2 for 2D lattice).
        hidden_dim: Hidden dimension for all MLPs.
        n_mp_blocks: Number of message-passing blocks.
        n_encoder_layers: Number of layers in the encoder MLP.
        lattice_spacing: Physical lattice spacing.
    """

    def __init__(
        self,
        lattice_dim: int = 2,
        hidden_dim: int = 64,
        n_mp_blocks: int = 3,
        n_encoder_layers: int = 2,
        lattice_spacing: float = 1.0,
    ) -> None:
        super().__init__()
        self.lattice_dim = lattice_dim
        self.lattice_spacing = lattice_spacing

        # Input: coordinates (lattice_dim) + spacing (1) + phi (1)
        input_dim = lattice_dim + 1 + 1

        # Node encoder
        enc_layers = []
        enc_layers.append(nn.Linear(input_dim, hidden_dim))
        enc_layers.append(nn.GELU())
        for _ in range(n_encoder_layers - 1):
            enc_layers.append(nn.Linear(hidden_dim, hidden_dim))
            enc_layers.append(nn.GELU())
        self.node_encoder = nn.Sequential(*enc_layers)

        # Edge encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(lattice_dim, hidden_dim),
            nn.GELU(),
        )

        # Message passing blocks (using PyG MessagePassing)
        self.mp_blocks = nn.ModuleList()
        for _ in range(n_mp_blocks):
            self.mp_blocks.append(
                HomogeneousMPBlock(hidden_dim=hidden_dim)
            )

        # Action readout head — takes BOTH initial encoding (skip) and
        # final MP'd representation, analogous to how the HeteroGNN's
        # action head sees h_st || h_field.
        self.action_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data: HeteroData) -> dict[str, torch.Tensor]:
        """Forward pass using only the homogeneous lattice graph."""
        # Extract raw features from HeteroData
        st_x = data["spacetime"].x  # (N, lattice_dim + 1)
        phi = data["scalar"].x  # (N, 1)

        # Concatenate into single node feature
        h = torch.cat([st_x, phi], dim=-1)  # (N, lattice_dim + 2)
        h = self.node_encoder(h)

        # Save initial encoding for skip connection
        h0 = h

        # Edge features from adjacency
        adj_edge_index = data[("spacetime", "adjacent", "spacetime")].edge_index
        adj_edge_attr = data[("spacetime", "adjacent", "spacetime")].edge_attr
        e = self.edge_encoder(adj_edge_attr)

        # Message passing
        for block in self.mp_blocks:
            h = block(h, adj_edge_index, e)

        # Energy readout with skip connection: [initial_encoding || mp_output]
        h_combined = torch.cat([h0, h], dim=-1)
        per_site = self.action_head(h_combined) * (self.lattice_spacing ** self.lattice_dim)

        batch = data["spacetime"].get("batch", None)
        if batch is not None:
            total = scatter(per_site.squeeze(-1), batch, dim=0, reduce="sum")
        else:
            total = per_site.sum(dim=0).squeeze(-1)

        return {"energy": total}


class HomogeneousMPBlock(MessagePassing):
    """Message-passing block using PyG's MessagePassing for proper aggregation.

    Uses the same pattern as the HeteroGNN's SpacetimeToSpacetime:
    message = MLP(x_target || x_source || edge_attr), then sum aggregation,
    then residual connection + layer normalization.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__(aggr="add", flow="source_to_target")
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        agg = self.propagate(edge_index, x=h, edge_attr=edge_attr)
        h = self.norm(h + agg)
        return h

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Message from source j to target i, with edge features."""
        return self.msg_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))
