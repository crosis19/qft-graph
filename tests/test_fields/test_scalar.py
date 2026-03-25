"""Tests for the ScalarField class."""

import torch

from qft_graph.fields.scalar import ScalarField


class TestScalarField:
    def test_dof_per_site(self, scalar_field):
        assert scalar_field.dof_per_site() == 1

    def test_node_type_name(self, scalar_field):
        assert scalar_field.node_type_name() == "scalar"

    def test_node_features_1d(self, scalar_field):
        phi = torch.randn(16)
        features = scalar_field.node_features(phi)
        assert features.shape == (16, 1)

    def test_node_features_2d(self, scalar_field):
        phi = torch.randn(16, 1)
        features = scalar_field.node_features(phi)
        assert features.shape == (16, 1)

    def test_hot_initialization(self, scalar_field):
        phi = scalar_field.initialize(100, mode="hot")
        assert phi.shape == (100,)
        assert phi.std() > 0.1  # Not constant

    def test_cold_initialization(self, scalar_field):
        phi = scalar_field.initialize(100, mode="cold")
        assert torch.allclose(phi, torch.ones(100))

    def test_gaussian_initialization(self, scalar_field):
        torch.manual_seed(42)
        phi = scalar_field.initialize(10000, mode="gaussian")
        # Should be approximately N(0,1)
        assert abs(phi.mean().item()) < 0.1
        assert abs(phi.std().item() - 1.0) < 0.1
