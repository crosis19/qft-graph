"""Tests for the U(1) GNN model over all three A-3 graph variants (task A-4)."""

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from qft_graph.config import LatticeConfig, ModelConfig
from qft_graph.fields.gauge import random_gauge_transform
from qft_graph.graphs.builder import U1GaugeGraphBuilder
from qft_graph.lattice.hypercubic import HypercubicLattice
from qft_graph.models.u1_gnn import U1GaugeGNN
from qft_graph.training.metrics import q_rounded_accuracy

BETA = 2.0
L = 4
ALL_VARIANTS = ["link_nodes", "edge_features", "invariant_oracle"]
WILSON_LOOPS = ("1x1", "2x2")


@pytest.fixture
def lattice():
    return HypercubicLattice(
        LatticeConfig(dimensions=(L, L), spacing=1.0, boundary="periodic")
    )


@pytest.fixture
def small_model_config():
    return ModelConfig(
        hidden_dim=16, n_mp_blocks=2, encoder_layers=1, dropout=0.0,
        activation="gelu", readout="energy",
    )


def make_graphs(lattice, variant, n=2, seed=7):
    rng = np.random.default_rng(seed)
    builder = U1GaugeGraphBuilder(lattice, beta=BETA, variant=variant)
    graphs = []
    for _ in range(n):
        theta = rng.uniform(-np.pi, np.pi, size=(2, L, L))
        graphs.append(builder.build({"gauge": torch.from_numpy(theta)}))
    return graphs


def make_model(config, variant):
    torch.manual_seed(0)
    return U1GaugeGNN(
        config, variant=variant, wilson_loops=WILSON_LOOPS, predict_q=True
    )


class TestForward:
    @pytest.mark.parametrize("variant", ALL_VARIANTS)
    def test_batched_output_shapes(self, lattice, small_model_config, variant):
        model = make_model(small_model_config, variant).eval()
        batch = Batch.from_data_list(make_graphs(lattice, variant, n=3))
        out = model(batch)
        assert out["energy"].shape == (3,)
        assert out["local_energy"].shape == (3 * L * L, 1)
        for name in WILSON_LOOPS:
            assert out[f"wilson_{name}"].shape == (3,)
        assert out["q"].shape == (3,)
        for v in out.values():
            assert torch.isfinite(v).all()

    @pytest.mark.parametrize("variant", ALL_VARIANTS)
    def test_batch_matches_single(self, lattice, small_model_config, variant):
        """Batched forward must agree with per-graph forwards (catches
        scatter/pooling bugs against the batch vector)."""
        model = make_model(small_model_config, variant).eval()
        graphs = make_graphs(lattice, variant, n=2)
        with torch.no_grad():
            out_b = model(Batch.from_data_list(graphs))
            singles = [model(g) for g in graphs]
        for key in ("energy", "q", *(f"wilson_{n}" for n in WILSON_LOOPS)):
            for s in singles:
                assert s[key].shape == (1,), f"unbatched {key} not (1,)"
            stacked = torch.cat([s[key] for s in singles])
            torch.testing.assert_close(out_b[key], stacked, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("variant", ALL_VARIANTS)
    def test_input_not_mutated(self, lattice, small_model_config, variant):
        model = make_model(small_model_config, variant).eval()
        g = make_graphs(lattice, variant, n=1)[0]
        x_before = g["spacetime"].x.clone()
        model(g)
        assert torch.equal(g["spacetime"].x, x_before)


class TestEdgeWiring:
    """Each typed edge set must individually influence the output.

    Pins the GaugeThreeStageBlock wiring: a copy-paste bug where one
    branch consumes another edge type's edge_index (e.g. ends_at reading
    GAUGE_STARTS_AT) makes the output invariant to rewiring the ignored
    edge type — mutation-verified blind spot of the shape/consistency
    tests. Rewiring here = rolling the site side of one edge type.
    """

    @pytest.mark.parametrize(
        "etype_name", ["starts_at", "ends_at", "origin_of", "target_of"]
    )
    def test_gauge_edge_types_all_consumed(
        self, lattice, small_model_config, etype_name
    ):
        from qft_graph.graphs.edge_types import (
            GAUGE_ENDS_AT,
            GAUGE_STARTS_AT,
            ORIGIN_OF,
            TARGET_OF,
        )

        etype = {
            "starts_at": GAUGE_STARTS_AT,
            "ends_at": GAUGE_ENDS_AT,
            "origin_of": ORIGIN_OF,
            "target_of": TARGET_OF,
        }[etype_name]
        # site side is row 1 for gauge->st edges, row 0 for st->gauge
        site_row = 1 if etype[0] == "gauge" else 0

        model = make_model(small_model_config, "link_nodes").eval()
        g = make_graphs(lattice, "link_nodes", n=1)[0]
        g_rewired = g.clone()
        g_rewired[etype].edge_index = g[etype].edge_index.clone()
        g_rewired[etype].edge_index[site_row] = torch.roll(
            g[etype].edge_index[site_row], 1
        )
        with torch.no_grad():
            out = model(g)
            out_rewired = model(g_rewired)
        changed = any(
            not torch.equal(out[k], out_rewired[k])
            for k in out
        )
        assert changed, f"{etype_name} edges do not influence the output"

    def test_adjacency_edges_consumed(self, lattice, small_model_config):
        from qft_graph.graphs.edge_types import ADJACENT

        for variant in ALL_VARIANTS:
            model = make_model(small_model_config, variant).eval()
            g = make_graphs(lattice, variant, n=1)[0]
            g_rewired = g.clone()
            g_rewired[ADJACENT].edge_index = g[ADJACENT].edge_index.clone()
            g_rewired[ADJACENT].edge_index[1] = torch.roll(
                g[ADJACENT].edge_index[1], 1
            )
            with torch.no_grad():
                out = model(g)
                out_rewired = model(g_rewired)
            changed = any(not torch.equal(out[k], out_rewired[k]) for k in out)
            assert changed, f"{variant}: adjacency edges do not influence output"


class TestGaugeInvariance:
    def test_variant_c_outputs_exactly_invariant(self, lattice, small_model_config):
        """C's inputs are bit-identical under gauge transforms (A-3 oracle),
        so a deterministic eval-mode forward must be bit-identical too."""
        model = make_model(small_model_config, "invariant_oracle").eval()
        builder = U1GaugeGraphBuilder(lattice, beta=BETA, variant="invariant_oracle")
        rng = np.random.default_rng(3)
        theta = rng.uniform(-np.pi, np.pi, size=(2, L, L))
        with torch.no_grad():
            ref = model(builder.build({"gauge": torch.from_numpy(theta)}))
            for _ in range(5):
                theta_g = random_gauge_transform(theta, rng)
                out = model(builder.build({"gauge": torch.from_numpy(theta_g)}))
                for key in ref:
                    assert torch.equal(out[key], ref[key]), key

    @pytest.mark.parametrize("variant", ["link_nodes", "edge_features"])
    def test_variants_ab_not_invariant(self, lattice, small_model_config, variant):
        """A and B must NOT be invariant at init — learning invariance is
        the A-5 experiment."""
        model = make_model(small_model_config, variant).eval()
        builder = U1GaugeGraphBuilder(lattice, beta=BETA, variant=variant)
        rng = np.random.default_rng(3)
        theta = rng.uniform(-np.pi, np.pi, size=(2, L, L))
        theta_g = random_gauge_transform(theta, rng)
        with torch.no_grad():
            ref = model(builder.build({"gauge": torch.from_numpy(theta)}))
            out = model(builder.build({"gauge": torch.from_numpy(theta_g)}))
        assert not torch.equal(out["energy"], ref["energy"])


class TestTraining:
    @pytest.mark.parametrize("variant", ALL_VARIANTS)
    def test_gradients_reach_all_parameters(self, lattice, small_model_config, variant):
        model = make_model(small_model_config, variant).train()
        batch = Batch.from_data_list(make_graphs(lattice, variant, n=2))
        out = model(batch)
        loss = out["energy"].pow(2).mean() + out["q"].pow(2).mean()
        for name in WILSON_LOOPS:
            loss = loss + out[f"wilson_{name}"].pow(2).mean()
        loss.backward()
        missing = [
            n for n, p in model.named_parameters()
            if p.requires_grad and p.grad is None
        ]
        assert not missing, f"parameters with no gradient: {missing}"

    def test_action_only_config_has_no_dead_parameters(
        self, lattice, small_model_config
    ):
        """With no gauge-consuming head, the final block's st->gauge stage
        is dropped rather than carried as dead weight — the params column
        of the A/B/C table must count only trainable-in-practice params."""
        torch.manual_seed(0)
        model = U1GaugeGNN(
            small_model_config, variant="link_nodes",
            wilson_loops=(), predict_q=False,
        ).train()
        batch = Batch.from_data_list(make_graphs(lattice, "link_nodes", n=2))
        out = model(batch)
        out["energy"].pow(2).mean().backward()
        missing = [
            n for n, p in model.named_parameters()
            if p.requires_grad and p.grad is None
        ]
        assert not missing, f"dead parameters in action-only config: {missing}"

    def test_unknown_variant_raises(self, small_model_config):
        with pytest.raises(ValueError, match="Unknown variant"):
            U1GaugeGNN(small_model_config, variant="D")

    def test_variant_aliases(self, small_model_config):
        assert U1GaugeGNN(small_model_config, variant="A").variant == "link_nodes"
        assert U1GaugeGNN(small_model_config, variant="C").variant == "invariant_oracle"


class TestQMetric:
    def test_exact_integer_accuracy(self):
        pred = torch.tensor([1.2, -0.4, 2.6, 0.9])
        true = torch.tensor([1.0, 0.0, 3.0, 0.0])
        assert q_rounded_accuracy(pred, true) == pytest.approx(0.75)
        assert q_rounded_accuracy(true, true) == 1.0

    def test_shape_mismatch_raises(self):
        """A (B,1)-vs-(B,) mismatch would broadcast to (B,B) and return a
        plausible wrong number (0.3125 instead of 0.75 here) — reject it."""
        pred = torch.tensor([[1.2], [-0.4], [2.6], [0.9]])
        true = torch.tensor([1.0, 0.0, 3.0, 0.0])
        with pytest.raises(ValueError, match="Shape mismatch"):
            q_rounded_accuracy(pred, true)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="Empty"):
            q_rounded_accuracy(torch.tensor([]), torch.tensor([]))
