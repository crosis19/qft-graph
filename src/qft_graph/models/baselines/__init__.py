"""Baseline models for architecture comparison."""

from qft_graph.models.baselines.homogeneous_gnn import HomogeneousGNN
from qft_graph.models.baselines.lattice_cnn import LatticeCNN
from qft_graph.models.baselines.mlp_baseline import MLPBaseline

__all__ = ["HomogeneousGNN", "LatticeCNN", "MLPBaseline"]
