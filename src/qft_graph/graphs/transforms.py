"""PyG transforms for heterogeneous graph preprocessing."""

from __future__ import annotations

import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform


class NormalizeFieldFeatures(BaseTransform):
    """Normalize field node features to zero mean and unit variance.

    Computed per-feature across all nodes of each field type.
    Statistics can be precomputed from the training set and applied
    to validation/test data.
    """

    def __init__(self, stats: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None):
        """
        Args:
            stats: Optional precomputed {field_name: (mean, std)} dict.
                If None, computes from the data (only suitable for single graphs).
        """
        self.stats = stats or {}

    def forward(self, data: HeteroData) -> HeteroData:
        for node_type in data.node_types:
            if node_type == "spacetime":
                continue  # Don't normalize coordinates
            if node_type in self.stats:
                mean, std = self.stats[node_type]
            else:
                mean = data[node_type].x.mean(dim=0)
                std = data[node_type].x.std(dim=0).clamp(min=1e-8)
            data[node_type].x = (data[node_type].x - mean) / std
        return data

    @staticmethod
    def compute_stats(
        dataset: list[HeteroData],
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Compute normalization statistics from a dataset."""
        accum: dict[str, list[torch.Tensor]] = {}
        for data in dataset:
            for node_type in data.node_types:
                if node_type == "spacetime":
                    continue
                accum.setdefault(node_type, []).append(data[node_type].x)

        stats = {}
        for node_type, tensors in accum.items():
            all_features = torch.cat(tensors, dim=0)
            stats[node_type] = (
                all_features.mean(dim=0),
                all_features.std(dim=0).clamp(min=1e-8),
            )
        return stats
