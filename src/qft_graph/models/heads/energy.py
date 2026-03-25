"""Energy/action readout head for the heterogeneous GNN."""

from __future__ import annotations

import torch
import torch.nn as nn


class EnergyHead(nn.Module):
    """Global readout computing the Euclidean action S_E[phi] from node embeddings.

    Implements: S_E = sum_x f_theta(h_spacetime(x), h_field(x)) * a^d

    where f_theta is a learned per-site energy function. The sum over lattice
    sites is the discrete analog of the spatial integral in the continuum action.

    The per-site decomposition ensures extensivity: doubling the lattice volume
    doubles the predicted action, as required by thermodynamics.

    Args:
        st_dim: Spacetime node embedding dimension.
        field_dims: Dict mapping field type names to their embedding dimensions.
        hidden_dim: Internal MLP hidden dimension.
    """

    def __init__(
        self,
        st_dim: int,
        field_dims: dict[str, int],
        hidden_dim: int,
    ) -> None:
        super().__init__()
        total_input = st_dim + sum(field_dims.values())
        self.energy_mlp = nn.Sequential(
            nn.Linear(total_input, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self._field_types = list(field_dims.keys())

    def forward(
        self,
        h_st: torch.Tensor,
        h_fields: dict[str, torch.Tensor],
        lattice_spacing: float,
        lattice_dim: int,
        batch: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute total and per-site energy.

        Args:
            h_st: Spacetime embeddings (num_nodes, st_dim).
            h_fields: Dict of field embeddings {name: (num_nodes, field_dim)}.
            lattice_spacing: Physical lattice spacing a.
            lattice_dim: Spatial dimension d.
            batch: Optional batch assignment tensor for batched graphs.

        Returns:
            (total_energy, per_site_energy) where total_energy has shape
            (batch_size,) and per_site_energy has shape (num_nodes, 1).
        """
        # Concatenate spacetime + all field embeddings at each site
        features = [h_st] + [h_fields[ft] for ft in self._field_types]
        combined = torch.cat(features, dim=-1)

        # Per-site energy density
        per_site = self.energy_mlp(combined)  # (num_nodes, 1)

        # Scale by lattice volume element a^d
        per_site = per_site * (lattice_spacing**lattice_dim)

        # Sum over sites (respecting batching)
        if batch is not None:
            from torch_geometric.utils import scatter
            total = scatter(per_site.squeeze(-1), batch, dim=0, reduce="sum")
        else:
            total = per_site.sum(dim=0).squeeze(-1)

        return total, per_site
