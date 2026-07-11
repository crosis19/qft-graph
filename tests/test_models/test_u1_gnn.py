"""Tests for the U(1) GNN model over all three A-3 graph variants (task A-4)."""

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from qft_graph.config import LatticeConfig, ModelConfig
from qft_graph.fields.gauge import random_gauge_transform
from qft_graph.graphs.builder import U1GaugeGraphBuilder
from qft_graph.lattice.hypercubic import HypercubicLattice
from qft_graph.models.u1_gnn import (
    U1GaugeGNN,
    matched_hidden_dim,
    u1_param_count,
)
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


class TestParameterMatching:
    """The H knob for parameter-matched A/B/C comparisons (Josh, 2026-07-11)."""

    KWARGS = dict(wilson_loops=WILSON_LOOPS, predict_q=True)

    def test_param_count_monotonic_in_h(self, small_model_config):
        """Strict monotonicity is the precondition of the binary search."""
        from dataclasses import replace

        for variant in ALL_VARIANTS:
            counts = [
                u1_param_count(
                    replace(small_model_config, hidden_dim=h), variant, **self.KWARGS
                )
                for h in (8, 12, 16, 24, 32)
            ]
            assert all(a < b for a, b in zip(counts, counts[1:])), variant

    def test_matched_is_bruteforce_optimal(self, small_model_config):
        """Binary-search result equals exhaustive scan over the H range."""
        from dataclasses import replace

        target = u1_param_count(
            replace(small_model_config, hidden_dim=24), "link_nodes", **self.KWARGS
        )
        for variant in ALL_VARIANTS:
            best = matched_hidden_dim(
                target, small_model_config, variant, h_min=4, h_max=96, **self.KWARGS
            )
            counts = {
                h: u1_param_count(
                    replace(small_model_config, hidden_dim=h), variant, **self.KWARGS
                )
                for h in range(4, 97)
            }
            brute = min(counts, key=lambda h: (abs(counts[h] - target), h))
            assert best == brute, variant

    def test_matching_a_budget_closes_the_gap(self, small_model_config):
        """B/C matched to A@32's budget land far closer than fixed H=32."""
        from dataclasses import replace

        cfg32 = replace(small_model_config, hidden_dim=32)
        target = u1_param_count(cfg32, "link_nodes", **self.KWARGS)
        for variant in ("edge_features", "invariant_oracle"):
            fixed_gap = abs(u1_param_count(cfg32, variant, **self.KWARGS) - target)
            h = matched_hidden_dim(target, cfg32, variant, **self.KWARGS)
            matched_gap = abs(
                u1_param_count(replace(cfg32, hidden_dim=h), variant, **self.KWARGS)
                - target
            )
            assert h > 32, variant  # B/C are lighter per H, so matched H is larger
            assert matched_gap < 0.05 * target, variant
            assert matched_gap < fixed_gap, variant

    def test_head_config_changes_the_match(self, small_model_config):
        """model_kwargs must flow into the count — an action-only model
        matches at a different H than a full-heads model."""
        from dataclasses import replace

        cfg = replace(small_model_config, hidden_dim=32)
        target = u1_param_count(cfg, "link_nodes", **self.KWARGS)
        h_full = matched_hidden_dim(target, cfg, "edge_features", **self.KWARGS)
        h_bare = matched_hidden_dim(
            target, cfg, "edge_features", wilson_loops=(), predict_q=False
        )
        assert h_bare > h_full  # fewer heads -> needs wider trunk to match

    def test_unreachable_target_raises(self, small_model_config):
        with pytest.raises(ValueError, match="not reachable"):
            matched_hidden_dim(
                10**9, small_model_config, "edge_features", h_max=64, **self.KWARGS
            )


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
