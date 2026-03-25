"""Spacetime node encoder MLP."""

from __future__ import annotations

import torch
import torch.nn as nn


class SpacetimeEncoder(nn.Module):
    """MLP encoder for spacetime lattice nodes.

    Transforms raw spacetime features [coordinates, lattice_spacing]
    into hidden representations. The encoder is shared across all
    spacetime nodes (translation equivariance by weight sharing).

    Args:
        input_dim: Dimension of raw spacetime features (ndim + 1).
        hidden_dim: Output and intermediate hidden dimension.
        n_layers: Number of MLP layers.
        activation: Activation function name.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_layers: int = 2,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        act = _get_activation(activation)

        layers: list[nn.Module] = []
        in_dim = input_dim
        for i in range(n_layers):
            out_dim = hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < n_layers - 1:
                layers.append(act)
            in_dim = out_dim

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode spacetime node features.

        Args:
            x: Raw features of shape (num_nodes, input_dim).

        Returns:
            Encoded features of shape (num_nodes, hidden_dim).
        """
        return self.mlp(x)


class FieldEncoder(nn.Module):
    """MLP encoder for field nodes (generic over field type).

    Each field type gets its own encoder instance, allowing
    different input dimensions (scalar=1, spinor=4, gauge=18, etc.)

    Args:
        input_dim: Dimension of raw field features.
        hidden_dim: Output hidden dimension.
        n_layers: Number of MLP layers.
        activation: Activation function name.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_layers: int = 2,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        act = _get_activation(activation)

        layers: list[nn.Module] = []
        in_dim = input_dim
        for i in range(n_layers):
            out_dim = hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < n_layers - 1:
                layers.append(act)
            in_dim = out_dim

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class EdgeEncoder(nn.Module):
    """MLP encoder for edge features (direction vectors, link variables).

    Args:
        input_dim: Dimension of raw edge features.
        hidden_dim: Output hidden dimension.
        activation: Activation function name.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        act = _get_activation(activation)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


def _get_activation(name: str) -> nn.Module:
    activations = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(),
        "tanh": nn.Tanh(),
    }
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}. Choose from {list(activations.keys())}")
    return activations[name]
