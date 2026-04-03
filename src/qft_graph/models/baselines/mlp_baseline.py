"""MLP baseline: flatten field configuration and predict action directly.

The simplest possible baseline — no graph structure, no convolutions,
just a feedforward network on the flattened field values.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.utils import scatter


class MLPBaseline(nn.Module):
    """MLP baseline that operates on flattened field configurations.

    Takes the raw phi values at all lattice sites, flattens them into a
    single vector, and predicts the total action via a feedforward network.

    Args:
        n_sites: Total number of lattice sites (L*L).
        hidden_dim: Hidden layer dimension.
        n_layers: Number of hidden layers.
    """

    def __init__(
        self,
        n_sites: int = 256,
        hidden_dim: int = 256,
        n_layers: int = 3,
    ) -> None:
        super().__init__()
        self.n_sites = n_sites

        layers = []
        layers.append(nn.Linear(n_sites, hidden_dim))
        layers.append(nn.GELU())
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, data: HeteroData) -> dict[str, torch.Tensor]:
        """Forward pass: flatten phi values and predict total action."""
        phi = data["scalar"].x  # (total_nodes, 1)
        batch = data["spacetime"].get("batch", None)

        if batch is not None:
            batch_size = batch.max().item() + 1
            # Use reshape instead of view to handle non-contiguous tensors
            phi_flat = phi.squeeze(-1).reshape(batch_size, self.n_sites)
        else:
            phi_flat = phi.squeeze(-1).unsqueeze(0)  # (1, n_sites)

        energy = self.mlp(phi_flat).squeeze(-1)  # (batch_size,)
        return {"energy": energy}
