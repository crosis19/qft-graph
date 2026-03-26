"""Top-level Heterogeneous GNN for Quantum Field Theory."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from qft_graph.config import ModelConfig
from qft_graph.graphs.edge_types import ADJACENT
from qft_graph.models.encoders.spacetime import EdgeEncoder, FieldEncoder, SpacetimeEncoder
from qft_graph.models.heads.correlator import CorrelatorHead
from qft_graph.models.heads.energy import EnergyHead
from qft_graph.models.message_passing.stage import ThreeStageBlock


class HeteroGNN(nn.Module):
    """Heterogeneous bipartite GNN for quantum field theory.

    Architecture:
      1. Type-specific encoders for spacetime nodes, field nodes, and edges
      2. N stacked ThreeStageBlocks (field -> spacetime -> spacetime -> field)
      3. Readout head(s): energy prediction and/or correlation function

    The model learns to score field configurations, approximating the
    Boltzmann distribution exp(-S_E[phi]) of the quantum field theory.

    This architecture dynamically adapts to the field content: Phase 1 has
    only scalar fields, while later phases add gauge and fermion fields
    with no changes to this class.

    Args:
        config: ModelConfig with architecture hyperparameters.
        lattice_dim: Spatial dimension of the lattice (2 for Phase 1).
        field_types: Dict mapping field type names to their raw input dimensions.
            E.g., {"scalar": 1} for Phase 1, {"scalar": 1, "fermion": 4} for Phase 2.
        lattice_spacing: Physical lattice spacing for energy head scaling.
    """

    def __init__(
        self,
        config: ModelConfig,
        lattice_dim: int,
        field_types: dict[str, int],
        lattice_spacing: float = 1.0,
    ) -> None:
        super().__init__()
        self.config = config
        self.lattice_dim = lattice_dim
        self.lattice_spacing = lattice_spacing
        self._field_type_names = list(field_types.keys())
        h = config.hidden_dim

        # --- Encoders ---
        # Spacetime: input = coordinates (lattice_dim) + lattice_spacing (1)
        self.st_encoder = SpacetimeEncoder(
            input_dim=lattice_dim + 1,
            hidden_dim=h,
            n_layers=config.encoder_layers,
            activation=config.activation,
        )

        # Field encoders: one per field type
        self.field_encoders = nn.ModuleDict(
            {
                fname: FieldEncoder(
                    input_dim=fdim,
                    hidden_dim=h,
                    n_layers=config.encoder_layers,
                    activation=config.activation,
                )
                for fname, fdim in field_types.items()
            }
        )

        # Edge encoder for adjacency direction vectors
        self.edge_encoder = EdgeEncoder(
            input_dim=lattice_dim,
            hidden_dim=h,
            activation=config.activation,
        )

        # --- Message Passing Blocks ---
        field_dims = {fname: h for fname in field_types}
        self.mp_blocks = nn.ModuleList(
            [
                ThreeStageBlock(
                    field_dims=field_dims,
                    st_dim=h,
                    edge_dim=h,
                    hidden_dim=h,
                    dropout=config.dropout,
                )
                for _ in range(config.n_mp_blocks)
            ]
        )

        # --- Readout Heads ---
        self.energy_head = None
        self.correlator_head = None

        if config.readout in ("energy", "both"):
            self.energy_head = EnergyHead(
                st_dim=h,
                field_dims=field_dims,
                hidden_dim=h,
            )

        if config.readout in ("correlator", "both"):
            self.correlator_head = CorrelatorHead(
                field_dim=h,
                hidden_dim=h,
            )

    def forward(self, data: HeteroData) -> dict[str, torch.Tensor]:
        """Full forward pass: encode -> message pass -> readout.

        Creates a shallow clone of the input data so that raw features
        on the original HeteroData are never overwritten. This avoids
        the in-place mutation bug where re-running eval on the same
        dataset would fail because features had been replaced with
        encoded embeddings.

        Args:
            data: HeteroData with raw node/edge features.

        Returns:
            Dict with available keys:
              - "energy": (batch_size,) predicted total action S_E[phi]
              - "local_energy": (num_nodes, 1) per-site energy density
              - "correlator": (num_nodes, num_nodes) learned correlation matrix
        """
        # Clone to avoid mutating the input data's features in-place
        data = data.clone()

        # --- Encode ---
        data = self.encode(data)

        # --- Message Passing ---
        data = self.message_pass(data)

        # --- Readout ---
        return self.readout(data)

    def encode(self, data: HeteroData) -> HeteroData:
        """Apply type-specific encoders to all node and edge types.

        Note: This method writes encoded features back into data's
        attribute tensors. The forward() method clones data first
        so the original raw features are preserved.
        """
        # Spacetime nodes
        data["spacetime"].x = self.st_encoder(data["spacetime"].x)

        # Field nodes
        for fname in self._field_type_names:
            data[fname].x = self.field_encoders[fname](data[fname].x)

        # Adjacency edge features
        data[ADJACENT].edge_attr = self.edge_encoder(data[ADJACENT].edge_attr)

        return data

    def message_pass(self, data: HeteroData) -> HeteroData:
        """Apply stacked ThreeStageBlocks."""
        for block in self.mp_blocks:
            data = block(data)
        return data

    def readout(self, data: HeteroData) -> dict[str, torch.Tensor]:
        """Apply readout heads to final node embeddings."""
        output: dict[str, torch.Tensor] = {}

        batch = data["spacetime"].get("batch", None)

        if self.energy_head is not None:
            h_st = data["spacetime"].x
            h_fields = {fname: data[fname].x for fname in self._field_type_names}
            total_e, local_e = self.energy_head(
                h_st, h_fields, self.lattice_spacing, self.lattice_dim, batch
            )
            output["energy"] = total_e
            output["local_energy"] = local_e

        if self.correlator_head is not None:
            # Use first field type for correlator (scalar in Phase 1)
            fname = self._field_type_names[0]
            output["correlator"] = self.correlator_head(data[fname].x)

        return output
