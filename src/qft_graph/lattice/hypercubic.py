"""N-dimensional hypercubic lattice with periodic boundary conditions."""

from __future__ import annotations

import itertools
from functools import cached_property

import numpy as np
import torch

from qft_graph.config import LatticeConfig
from qft_graph.lattice.base import Lattice
from qft_graph.lattice.boundary import BoundaryCondition


class HypercubicLattice(Lattice):
    """N-dimensional hypercubic lattice.

    Supports arbitrary dimension: 2D for Phase 1 (scalar phi^4),
    3D for Phase 2 (Schwinger model), 4D for Phase 3 (lattice QCD).
    Neighbor computation and coordinates are fully vectorized.

    Args:
        config: LatticeConfig specifying dimensions, spacing, and boundary.
    """

    def __init__(self, config: LatticeConfig) -> None:
        self._dims = tuple(config.dimensions)
        self._spacing = config.spacing
        self._bc = BoundaryCondition.from_string(config.boundary)
        self._ndim = len(self._dims)
        self._nsites = int(np.prod(self._dims))

    def num_sites(self) -> int:
        return self._nsites

    def dimension(self) -> int:
        return self._ndim

    def lattice_spacing(self) -> float:
        return self._spacing

    def volume(self) -> float:
        return self._spacing**self._ndim * self._nsites

    @cached_property
    def _multi_indices(self) -> np.ndarray:
        """All multi-indices as (num_sites, ndim) array."""
        ranges = [np.arange(d) for d in self._dims]
        grid = np.meshgrid(*ranges, indexing="ij")
        return np.stack([g.ravel() for g in grid], axis=-1)

    def site_coordinates(self) -> torch.Tensor:
        """Physical coordinates: multi_index * spacing.

        Returns:
            Tensor of shape (num_sites, ndim).
        """
        coords = self._multi_indices.astype(np.float64) * self._spacing
        return torch.from_numpy(coords).float()

    def _flat_index(self, multi_idx: np.ndarray) -> np.ndarray:
        """Convert multi-index to flat index. multi_idx shape: (..., ndim)."""
        strides = np.array([int(np.prod(self._dims[i + 1 :])) for i in range(self._ndim)])
        return (multi_idx * strides).sum(axis=-1)

    def neighbor_pairs(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute all nearest-neighbor directed edges.

        For a d-dimensional lattice, each site has 2d neighbors.
        Returns both +mu and -mu directions.

        Returns:
            (src, dst) each of shape (num_sites * 2 * ndim,).
        """
        mi = self._multi_indices  # (nsites, ndim)
        src_list = []
        dst_list = []

        for mu in range(self._ndim):
            for direction in [+1, -1]:
                shifted = mi.copy()
                shifted[:, mu] += direction

                if self._bc == BoundaryCondition.PERIODIC:
                    shifted[:, mu] %= self._dims[mu]
                elif self._bc == BoundaryCondition.OPEN:
                    # Exclude edges that cross the boundary
                    mask = (shifted[:, mu] >= 0) & (shifted[:, mu] < self._dims[mu])
                    flat_src = self._flat_index(mi[mask])
                    flat_dst = self._flat_index(shifted[mask])
                    src_list.append(flat_src)
                    dst_list.append(flat_dst)
                    continue
                else:
                    # Antiperiodic: same connectivity as periodic (sign handled in fields)
                    shifted[:, mu] %= self._dims[mu]

                flat_src = self._flat_index(mi)
                flat_dst = self._flat_index(shifted)
                src_list.append(flat_src)
                dst_list.append(flat_dst)

        src = np.concatenate(src_list)
        dst = np.concatenate(dst_list)
        return torch.from_numpy(src).long(), torch.from_numpy(dst).long()

    def edge_directions(self) -> torch.Tensor:
        """Unit direction vector for each edge, matching neighbor_pairs ordering.

        Returns:
            Tensor of shape (num_edges, ndim).
        """
        directions = []
        mi = self._multi_indices

        for mu in range(self._ndim):
            for direction in [+1, -1]:
                d = np.zeros(self._ndim, dtype=np.float32)
                d[mu] = float(direction)

                if self._bc == BoundaryCondition.OPEN:
                    shifted = mi.copy()
                    shifted[:, mu] += direction
                    mask = (shifted[:, mu] >= 0) & (shifted[:, mu] < self._dims[mu])
                    n_edges = mask.sum()
                else:
                    n_edges = self._nsites

                directions.append(np.tile(d, (n_edges, 1)))

        return torch.from_numpy(np.concatenate(directions, axis=0))

    @cached_property
    def shape(self) -> tuple[int, ...]:
        """Lattice shape tuple, e.g. (16, 16) for 2D."""
        return self._dims
