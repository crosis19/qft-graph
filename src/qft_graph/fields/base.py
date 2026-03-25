"""Abstract base class for quantum fields living on a lattice."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class Field(ABC):
    """Abstract field content living on lattice sites.

    Each Field subclass represents a specific type of quantum field
    (scalar, gauge, fermion) with its own degrees of freedom and
    transformation properties. The Field interface feeds into the
    HeteroGraphBuilder to create typed field nodes in the bipartite graph.
    """

    @abstractmethod
    def dof_per_site(self) -> int:
        """Number of real degrees of freedom per lattice site.

        Examples:
            - Real scalar: 1
            - Complex scalar: 2
            - Dirac spinor in 2D: 4 (2 complex components)
            - SU(3) gauge link: 18 (3x3 complex matrix, before constraints)
        """

    @abstractmethod
    def node_features(self, configuration: torch.Tensor) -> torch.Tensor:
        """Convert raw field configuration to node feature tensor.

        Args:
            configuration: Raw field values, shape depends on field type.
                Scalar: (num_sites,) or (num_sites, 1)

        Returns:
            Feature tensor of shape (num_sites, feature_dim) for the
            graph builder to attach to field nodes.
        """

    @abstractmethod
    def node_type_name(self) -> str:
        """String identifier for this field type in the heterogeneous graph.

        Used as the node type key in PyG HeteroData, e.g. 'scalar', 'gauge', 'fermion'.
        Must be unique across all field types in a given graph.
        """

    @abstractmethod
    def initialize(self, num_sites: int, mode: str = "hot") -> torch.Tensor:
        """Generate an initial field configuration.

        Args:
            num_sites: Number of lattice sites.
            mode: Initialization strategy ('hot', 'cold', 'gaussian').

        Returns:
            Raw configuration tensor.
        """
