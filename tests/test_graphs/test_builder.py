"""Tests for the HeteroGraphBuilder."""

import torch

from qft_graph.graphs.builder import HeteroGraphBuilder
from qft_graph.graphs.edge_types import ADJACENT, inhabits_edge, inhabits_inv_edge


class TestHeteroGraphBuilder:
    def test_node_types(self, graph_builder, sample_config):
        data = graph_builder.build({"scalar": sample_config})
        assert "spacetime" in data.node_types
        assert "scalar" in data.node_types

    def test_spacetime_features_shape(self, graph_builder, sample_config, small_lattice):
        data = graph_builder.build({"scalar": sample_config})
        # Default a_in_edges=True: features are coordinates only [x1, x2]
        assert data["spacetime"].x.shape == (small_lattice.num_sites(), 2)

    def test_spacetime_features_a_in_nodes(self, small_lattice, scalar_field, sample_config):
        builder = HeteroGraphBuilder(small_lattice, [scalar_field], a_in_edges=False)
        data = builder.build({"scalar": sample_config})
        # a_in_edges=False: features are [x1, x2, lattice_spacing]
        assert data["spacetime"].x.shape == (small_lattice.num_sites(), 3)

    def test_scalar_features_shape(self, graph_builder, sample_config, small_lattice):
        data = graph_builder.build({"scalar": sample_config})
        assert data["scalar"].x.shape == (small_lattice.num_sites(), 1)

    def test_adjacency_edges(self, graph_builder, sample_config, small_lattice):
        data = graph_builder.build({"scalar": sample_config})
        edge_index = data[ADJACENT].edge_index
        assert edge_index.shape[0] == 2
        # 4 neighbors per site in 2D
        assert edge_index.shape[1] == small_lattice.num_sites() * 4

    def test_adjacency_edge_attr(self, graph_builder, sample_config):
        data = graph_builder.build({"scalar": sample_config})
        edge_attr = data[ADJACENT].edge_attr
        # Direction vectors in 2D
        assert edge_attr.shape[1] == 2

    def test_inhabits_edges(self, graph_builder, sample_config, small_lattice):
        data = graph_builder.build({"scalar": sample_config})
        inh = inhabits_edge("scalar")
        edge_index = data[inh].edge_index
        # Bipartite: each field node connects to exactly one spacetime node
        assert edge_index.shape == (2, small_lattice.num_sites())
        # It's a perfect matching: field_i -> spacetime_i
        assert torch.all(edge_index[0] == edge_index[1])

    def test_inhabits_inv_edges(self, graph_builder, sample_config, small_lattice):
        data = graph_builder.build({"scalar": sample_config})
        inv = inhabits_inv_edge("scalar")
        edge_index = data[inv].edge_index
        assert edge_index.shape == (2, small_lattice.num_sites())

    def test_missing_config_raises(self, graph_builder):
        with __import__("pytest").raises(ValueError, match="Missing configuration"):
            graph_builder.build({})

    def test_build_dataset(self, graph_builder, small_lattice):
        n = 5
        configs = {"scalar": torch.randn(n, small_lattice.num_sites())}
        actions = torch.randn(n)
        dataset = graph_builder.build_dataset(configs, actions)

        assert len(dataset) == n
        assert dataset[0].y.shape == (1,)

    def test_spacetime_features_constant_variant(self, small_lattice, scalar_field, sample_config):
        builder = HeteroGraphBuilder(
            small_lattice, [scalar_field], spacetime_features="constant"
        )
        data = builder.build({"scalar": sample_config})
        # Coordinate-free: single constant feature per spacetime node
        assert data["spacetime"].x.shape == (small_lattice.num_sites(), 1)
        assert torch.all(data["spacetime"].x == 1.0)

    def test_unknown_spacetime_features_raises(self, small_lattice, scalar_field):
        with __import__("pytest").raises(ValueError):
            HeteroGraphBuilder(small_lattice, [scalar_field], spacetime_features="bogus")
