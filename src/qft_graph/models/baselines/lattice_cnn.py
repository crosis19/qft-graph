"""Lattice CNN baseline: standard 2D convolution over the field grid.

Treats the scalar field configuration as a 2D image and applies
convolutional layers with periodic padding to respect boundary conditions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.utils import scatter


class PeriodicPad2d(nn.Module):
    """Circular padding for periodic boundary conditions."""

    def __init__(self, padding: int = 1) -> None:
        super().__init__()
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.pad(x, [self.padding] * 4, mode="circular")


class LatticeCNN(nn.Module):
    """CNN baseline operating directly on the 2D field grid.

    The scalar field phi(x) is reshaped into a (1, L, L) image and processed
    with periodic-padded convolutions. The output is summed over sites to
    predict the total action, mirroring the extensivity constraint.

    Args:
        lattice_dims: Tuple (L, L) giving lattice dimensions.
        hidden_channels: Number of convolutional channels.
        n_conv_layers: Number of convolutional layers.
        lattice_spacing: Physical lattice spacing.
    """

    def __init__(
        self,
        lattice_dims: tuple[int, int] = (16, 16),
        hidden_channels: int = 32,
        n_conv_layers: int = 4,
        lattice_spacing: float = 1.0,
    ) -> None:
        super().__init__()
        self.lattice_dims = lattice_dims
        self.lattice_spacing = lattice_spacing
        self.lattice_dim = 2

        layers = []
        in_ch = 1  # single scalar field channel
        for i in range(n_conv_layers):
            out_ch = hidden_channels
            layers.append(PeriodicPad2d(1))
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=0))
            layers.append(nn.GELU())
            in_ch = out_ch

        self.conv_layers = nn.Sequential(*layers)

        # Per-site action density predictor (1x1 conv)
        self.action_head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, 1),
        )

    def forward(self, data: HeteroData) -> dict[str, torch.Tensor]:
        """Forward pass: reshape field to grid, apply conv, sum."""
        phi = data["scalar"].x  # (total_nodes, 1)
        batch = data["spacetime"].get("batch", None)

        L_h, L_w = self.lattice_dims

        if batch is not None:
            # Batched: split into individual graphs
            batch_size = batch.max().item() + 1
            phi_grid = phi.reshape(batch_size, 1, L_h, L_w)
        else:
            phi_grid = phi.reshape(1, 1, L_h, L_w)

        # Convolutional feature extraction
        h = self.conv_layers(phi_grid)

        # Per-site action density
        per_site = self.action_head(h) * (self.lattice_spacing ** self.lattice_dim)

        # Sum over spatial dimensions to get total action
        total = per_site.sum(dim=(1, 2, 3))  # (batch_size,)

        return {"energy": total}
