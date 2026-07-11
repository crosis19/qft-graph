"""Graph-level pooled readout head for Wilson loops and topological charge."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.utils import scatter


class PooledReadoutHead(nn.Module):
    """Graph-level scalar readout: mean-pool each node type, concat, MLP.

    Implements the plan §3 (task A-4) Wilson-loop readout — "graph-level
    readout (mean-pool over both node types -> MLP), one head per loop
    size" — and doubles as the topological-charge regression head. Unlike
    ActionHead there is no per-site decomposition: Wilson loops and Q are
    genuinely global observables of a configuration.

    Args:
        node_dims: Dict mapping pooled node type names to their embedding
            dimensions, e.g. {"spacetime": H, "gauge": H} for Variant A or
            {"spacetime": H} for Variants B/C.
        hidden_dim: Internal MLP hidden dimension.
        global_dim: Dimension of graph-level coupling features appended to
            the pooled representation (0 disables).
    """

    def __init__(
        self,
        node_dims: dict[str, int],
        hidden_dim: int,
        global_dim: int = 0,
    ) -> None:
        super().__init__()
        self._node_types = list(node_dims.keys())
        self._global_dim = global_dim
        total_input = sum(node_dims.values()) + global_dim
        self.mlp = nn.Sequential(
            nn.Linear(total_input, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        h_nodes: dict[str, torch.Tensor],
        batches: dict[str, torch.Tensor | None],
        globals_: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute one scalar per graph.

        Args:
            h_nodes: Node embeddings per type, {name: (num_nodes, dim)}.
            batches: Batch assignment per type, {name: (num_nodes,) or None
                for a single unbatched graph}.
            globals_: Graph-level couplings (batch_size, global_dim),
                required iff global_dim > 0.

        Returns:
            (batch_size,) predictions (batch_size = 1 when unbatched).
        """
        pooled = []
        for ntype in self._node_types:
            h = h_nodes[ntype]
            batch = batches[ntype]
            if batch is not None:
                pooled.append(scatter(h, batch, dim=0, reduce="mean"))
            else:
                pooled.append(h.mean(dim=0, keepdim=True))
        combined = torch.cat(pooled, dim=-1)  # (batch_size, sum dims)

        if self._global_dim > 0:
            if globals_ is None:
                raise ValueError(
                    "PooledReadoutHead built with global_dim > 0 but no globals given"
                )
            combined = torch.cat([combined, globals_], dim=-1)

        return self.mlp(combined).squeeze(-1)
