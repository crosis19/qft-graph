"""Tests for the 3-stage message passing blocks."""

import torch

from qft_graph.graphs.edge_types import ADJACENT
from qft_graph.models.message_passing.stage import ThreeStageBlock


class TestThreeStageBlock:
    def test_output_shapes(self, graph_builder, sample_config, small_lattice):
        """Block should preserve node feature dimensions."""
        data = graph_builder.build({"scalar": sample_config})
        h = 16

        # Encode to hidden dim
        data["spacetime"].x = torch.randn(small_lattice.num_sites(), h)
        data["scalar"].x = torch.randn(small_lattice.num_sites(), h)
        data[ADJACENT].edge_attr = torch.randn(data[ADJACENT].edge_index.shape[1], h)

        block = ThreeStageBlock(
            field_dims={"scalar": h},
            st_dim=h,
            edge_dim=h,
            hidden_dim=h,
        )

        data = block(data)
        assert data["spacetime"].x.shape == (small_lattice.num_sites(), h)
        assert data["scalar"].x.shape == (small_lattice.num_sites(), h)

    def test_residual_connection(self, graph_builder, sample_config, small_lattice):
        """Output should differ from input (messages have effect)."""
        data = graph_builder.build({"scalar": sample_config})
        h = 16
        data["spacetime"].x = torch.randn(small_lattice.num_sites(), h)
        data["scalar"].x = torch.randn(small_lattice.num_sites(), h)
        data[ADJACENT].edge_attr = torch.randn(data[ADJACENT].edge_index.shape[1], h)

        st_before = data["spacetime"].x.clone()

        block = ThreeStageBlock(
            field_dims={"scalar": h},
            st_dim=h,
            edge_dim=h,
            hidden_dim=h,
        )
        data = block(data)

        # Should be different from input (messages contribute)
        assert not torch.allclose(data["spacetime"].x, st_before, atol=1e-3)
