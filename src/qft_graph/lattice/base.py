"""Abstract base class for spacetime lattice geometry."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class Lattice(ABC):
    """Abstract lattice providing geometry to the graph builder.

    Subclasses implement specific lattice geometries (hypercubic, triangular, etc.)
    and boundary conditions. The interface is dimension-agnostic: 2D for Phase 1,
    4D for Phase 3, with no code changes to downstream consumers.
    """

    @abstractmethod
    def num_sites(self) -> int:
        """Total number of lattice sites."""

    @abstractmethod
    def dimension(self) -> int:
        """Spatial dimension of the lattice."""

    @abstractmethod
    def site_coordinates(self) -> torch.Tensor:
        """Coordinates of all lattice sites.

        Returns:
            Tensor of shape (num_sites, dimension) with physical coordinates.
        """

    @abstractmethod
    def neighbor_pairs(self) -> tuple[torch.Tensor, torch.Tensor]:
        """All nearest-neighbor directed edges.

        Returns:
            Tuple of (src, dst) index tensors, each of shape (num_edges,).
            Both forward and backward directions are included.
        """

    @abstractmethod
    def edge_directions(self) -> torch.Tensor:
        """Unit direction vector for each edge.

        Returns:
            Tensor of shape (num_edges, dimension).
        """

    @abstractmethod
    def lattice_spacing(self) -> float:
        """Physical lattice spacing a."""

    @abstractmethod
    def volume(self) -> float:
        """Total lattice volume V = a^d * num_sites."""
