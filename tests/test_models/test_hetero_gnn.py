"""Tests for the top-level HeteroGNN model."""

import torch

from qft_graph.models.hetero_gnn import HeteroGNN


class TestHeteroGNN:
    def test_forward_pass(self, graph_builder, sample_config, model_config, small_lattice):
        """Model should produce energy output."""
        data = graph_builder.build({"scalar": sample_config})
        model = HeteroGNN(
            config=model_config,
            lattice_dim=small_lattice.dimension(),
            field_types={"scalar": 1},
            lattice_spacing=small_lattice.lattice_spacing(),
        )
        model.eval()
        with torch.no_grad():
            output = model(data)

        assert "energy" in output
        assert output["energy"].dim() == 0 or output["energy"].shape == (1,)

    def test_local_energy_shape(self, graph_builder, sample_config, model_config, small_lattice):
        data = graph_builder.build({"scalar": sample_config})
        model = HeteroGNN(
            config=model_config,
            lattice_dim=small_lattice.dimension(),
            field_types={"scalar": 1},
        )
        model.eval()
        with torch.no_grad():
            output = model(data)

        assert "local_energy" in output
        assert output["local_energy"].shape == (small_lattice.num_sites(), 1)

    def test_gradient_flow(self, graph_builder, sample_config, model_config, small_lattice):
        """All parameters should receive gradients."""
        data = graph_builder.build({"scalar": sample_config})
        model = HeteroGNN(
            config=model_config,
            lattice_dim=small_lattice.dimension(),
            field_types={"scalar": 1},
        )
        model.train()
        output = model(data)
        loss = output["energy"].sum()
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            # At least some parameters should have nonzero gradients
        has_nonzero = any(
            param.grad.abs().sum() > 0 for param in model.parameters()
        )
        assert has_nonzero

    def test_deterministic(self, graph_builder, sample_config, model_config, small_lattice):
        """Same input should produce same output."""
        model = HeteroGNN(
            config=model_config,
            lattice_dim=small_lattice.dimension(),
            field_types={"scalar": 1},
        )
        model.eval()

        # Build fresh data each time since forward() mutates node features in-place
        torch.manual_seed(42)
        data1 = graph_builder.build({"scalar": sample_config})
        with torch.no_grad():
            out1 = model(data1)["energy"]

        torch.manual_seed(42)
        data2 = graph_builder.build({"scalar": sample_config})
        with torch.no_grad():
            out2 = model(data2)["energy"]

        assert torch.allclose(out1, out2)

    def test_parameter_count(self, model_config, small_lattice):
        model = HeteroGNN(
            config=model_config,
            lattice_dim=small_lattice.dimension(),
            field_types={"scalar": 1},
        )
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0
        assert n_params < 1_000_000  # Sanity: shouldn't be enormous for small model
