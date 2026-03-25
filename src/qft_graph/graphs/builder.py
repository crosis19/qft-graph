"""Heterogeneous bipartite graph construction from lattice + field content."""

from __future__ import annotations

import torch
from torch_geometric.data import HeteroData

from qft_graph.fields.base import Field
from qft_graph.graphs.edge_types import ADJACENT, inhabits_edge, inhabits_inv_edge
from qft_graph.lattice.base import Lattice


class HeteroGraphBuilder:
    """Constructs heterogeneous bipartite graphs from lattice geometry + field content.

    The core architectural innovation: spacetime nodes and field nodes are
    distinct types connected by typed edges, forming a bipartite structure
    that mirrors the geometry-matter separation fundamental to QFT.

    Graph structure:
        G = (V_spacetime, V_field1, ..., V_fieldN,
             E_adjacent, E_inhabits_1, ..., E_inhabits_N)

    This builder is generic over field types:
    - Phase 1: ScalarField only (one phi per site)
    - Phase 2: + GaugeField (links on edges) + FermionField (spinors on sites)
    - Phase 3+: Additional field types added with zero changes to this builder

    Args:
        lattice: Lattice instance providing geometry.
        fields: List of Field instances to include in the graph.
    """

    def __init__(self, lattice: Lattice, fields: list[Field]) -> None:
        self.lattice = lattice
        self.fields = fields
        self._nsites = lattice.num_sites()

        # Precompute lattice topology (fixed across configurations)
        self._coords = lattice.site_coordinates()
        self._adj_src, self._adj_dst = lattice.neighbor_pairs()
        self._edge_dirs = lattice.edge_directions()

        # Bipartite edges: each field node i connects to spacetime node i
        self._bipartite_idx = torch.arange(self._nsites)

    def build(self, configurations: dict[str, torch.Tensor]) -> HeteroData:
        """Build a single heterogeneous graph from field configurations.

        Args:
            configurations: Mapping from field.node_type_name() to raw
                configuration tensor. E.g., {"scalar": tensor(num_sites,)}.

        Returns:
            PyG HeteroData with typed nodes, edges, and features.
        """
        data = HeteroData()

        # --- Spacetime nodes ---
        # Features: [coordinates, lattice_spacing]
        spacing = torch.full((self._nsites, 1), self.lattice.lattice_spacing())
        st_features = torch.cat([self._coords, spacing], dim=-1)
        data["spacetime"].x = st_features
        data["spacetime"].num_nodes = self._nsites

        # --- Spacetime adjacency edges ---
        data[ADJACENT].edge_index = torch.stack([self._adj_src, self._adj_dst])
        data[ADJACENT].edge_attr = self._edge_dirs

        # --- Field nodes + bipartite edges ---
        for field_obj in self.fields:
            fname = field_obj.node_type_name()

            if fname not in configurations:
                raise ValueError(
                    f"Missing configuration for field '{fname}'. "
                    f"Provided: {list(configurations.keys())}"
                )

            # Field node features
            features = field_obj.node_features(configurations[fname])
            data[fname].x = features
            data[fname].num_nodes = self._nsites

            # Bipartite edges: field -> spacetime (inhabits)
            inh_edge = inhabits_edge(fname)
            data[inh_edge].edge_index = torch.stack(
                [self._bipartite_idx, self._bipartite_idx]
            )

            # Reverse bipartite: spacetime -> field (for message passing back)
            inv_edge = inhabits_inv_edge(fname)
            data[inv_edge].edge_index = torch.stack(
                [self._bipartite_idx, self._bipartite_idx]
            )

        return data

    def build_dataset(
        self, configurations: dict[str, torch.Tensor], actions: torch.Tensor | None = None
    ) -> list[HeteroData]:
        """Build a list of HeteroData graphs from batched configurations.

        Args:
            configurations: Mapping from field name to (n_configs, ...) tensors.
            actions: Optional (n_configs,) tensor of true action values.

        Returns:
            List of HeteroData objects, one per configuration.
        """
        # Determine batch size from first field
        first_key = next(iter(configurations))
        n_configs = configurations[first_key].shape[0]

        dataset = []
        for i in range(n_configs):
            single_config = {
                fname: configurations[fname][i] for fname in configurations
            }
            data = self.build(single_config)

            if actions is not None:
                data.y = actions[i].unsqueeze(0)

            dataset.append(data)

        return dataset
