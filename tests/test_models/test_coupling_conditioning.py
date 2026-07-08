"""Tests for coupling-constant conditioning (task P1-3).

Field nodes carry [phi, m2, lambda] and the readout is conditioned on
(m2, lambda) via graph-level globals, so one model can train jointly
across coupling values.
"""

import torch
from torch_geometric.data import Batch

from qft_graph.config import ModelConfig
from qft_graph.fields.scalar import ScalarField
from qft_graph.graphs.builder import HeteroGraphBuilder
from qft_graph.models.hetero_gnn import HeteroGNN


def make_builder(lattice, m2=-0.5, lam=0.5):
    field = ScalarField(couplings=(m2, lam))
    return HeteroGraphBuilder(lattice, [field]), field


class TestScalarFieldCouplings:
    def test_node_features_include_couplings(self):
        field = ScalarField(couplings=(-1.5, 0.5))
        phi = torch.randn(16)
        feats = field.node_features(phi)
        assert feats.shape == (16, 3)
        assert torch.allclose(feats[:, 0], phi)
        assert torch.all(feats[:, 1] == -1.5)
        assert torch.all(feats[:, 2] == 0.5)
        assert field.node_feature_dim() == 3

    def test_global_features(self):
        field = ScalarField(couplings=(-1.5, 0.5))
        g = field.global_features()
        assert g.shape == (1, 2)
        assert g[0, 0] == -1.5 and g[0, 1] == 0.5

    def test_no_couplings_unchanged(self):
        field = ScalarField()
        assert field.node_feature_dim() == 1
        assert field.global_features() is None
        assert field.node_features(torch.randn(16)).shape == (16, 1)


class TestBuilderGlobals:
    def test_globals_attached(self, small_lattice, sample_config):
        builder, _ = make_builder(small_lattice, m2=-2.0, lam=0.5)
        data = builder.build({"scalar": sample_config})
        assert hasattr(data, "globals")
        assert data.globals.shape == (1, 2)
        assert data.globals[0, 0] == -2.0

    def test_no_globals_without_couplings(self, graph_builder, sample_config):
        data = graph_builder.build({"scalar": sample_config})
        assert "globals" not in data


class TestConditionedModel:
    def _model(self, small_lattice, model_config):
        return HeteroGNN(
            config=model_config,
            lattice_dim=small_lattice.dimension(),
            field_types={"scalar": 3},
            lattice_spacing=small_lattice.lattice_spacing(),
            global_dim=2,
        )

    def test_forward_pass(self, small_lattice, model_config, sample_config):
        builder, _ = make_builder(small_lattice)
        data = builder.build({"scalar": sample_config})
        model = self._model(small_lattice, model_config)
        model.eval()
        with torch.no_grad():
            out = model(data)
        assert "energy" in out
        assert out["energy"].dim() == 0 or out["energy"].shape == (1,)

    def test_output_depends_on_couplings(self, small_lattice, model_config, sample_config):
        """Same phi, different (m2, lambda) -> different predicted action."""
        model = self._model(small_lattice, model_config)
        model.eval()
        outs = []
        for m2 in (-0.5, -2.0):
            builder, _ = make_builder(small_lattice, m2=m2)
            data = builder.build({"scalar": sample_config})
            with torch.no_grad():
                outs.append(model(data)["energy"])
        assert not torch.allclose(outs[0], outs[1])

    def test_batched_globals(self, small_lattice, model_config, sample_config):
        """Batched graphs with different couplings each keep their own."""
        graphs = []
        for m2 in (-0.5, -2.0, -1.0):
            builder, _ = make_builder(small_lattice, m2=m2)
            graphs.append(builder.build({"scalar": sample_config}))
        batch = Batch.from_data_list(graphs)
        assert batch.globals.shape == (3, 2)

        model = self._model(small_lattice, model_config)
        model.eval()
        with torch.no_grad():
            out = model(batch)
        assert out["energy"].shape == (3,)
        # Different couplings -> per-graph energies differ
        assert not torch.allclose(out["energy"][0], out["energy"][1])

    def test_gradient_flow(self, small_lattice, model_config, sample_config):
        builder, _ = make_builder(small_lattice)
        data = builder.build({"scalar": sample_config})
        model = self._model(small_lattice, model_config)
        model.train()
        model(data)["energy"].sum().backward()
        assert all(p.grad is not None for p in model.parameters())
