"""Tests for the A-5 gauge-invariance error eps_gauge.

Exact-value tests of gauge_invariance_error (the arbiter of the A-5
definition: eps = mean_configs[std_over_gauge_copies] / std_over_configs,
sample stds), plus the model-level oracle: through the real
gauge_orbit -> builder -> U1GaugeGNN path, Variant C's eps_gauge is
EXACTLY zero (bit-identical inputs) while A and B are strictly positive.
This pins the core loop of scripts/measure_gauge_invariance.py.
"""

import math

import numpy as np
import pytest
import torch

from qft_graph.config import LatticeConfig, ModelConfig
from qft_graph.fields.gauge import gauge_orbit
from qft_graph.graphs.builder import U1GaugeGraphBuilder
from qft_graph.lattice.hypercubic import HypercubicLattice
from qft_graph.models.u1_gnn import U1GaugeGNN
from qft_graph.training.metrics import gauge_invariance_error

L = 4
BETA = 2.0
WILSON_LOOPS = ("1x1", "2x2")
OUTPUT_KEYS = ("energy", "wilson_1x1", "wilson_2x2", "q")


class TestExactValues:
    def test_hand_computed(self):
        """Orbit rows [0,2] and [2,4] both have sample std sqrt(2); config
        preds [1,3] have sample std sqrt(2) -> eps = 1 exactly."""
        out = gauge_invariance_error(
            torch.tensor([[0.0, 2.0], [2.0, 4.0]]), torch.tensor([1.0, 3.0])
        )
        assert out["mean_orbit_std"] == pytest.approx(math.sqrt(2.0), abs=1e-12)
        assert out["config_std"] == pytest.approx(math.sqrt(2.0), abs=1e-12)
        assert out["eps_gauge"] == pytest.approx(1.0, abs=1e-12)

    def test_asymmetric_hand_computed(self):
        """Row stds 0 and 2*sqrt(2) -> numerator sqrt(2); denominator
        std([0,4]) = 2*sqrt(2) -> eps = 1/2 exactly."""
        out = gauge_invariance_error(
            torch.tensor([[1.0, 1.0], [0.0, 4.0]]), torch.tensor([0.0, 4.0])
        )
        assert out["eps_gauge"] == pytest.approx(0.5, abs=1e-12)

    def test_invariant_predictor_is_exactly_zero(self):
        out = gauge_invariance_error(
            torch.tensor([[3.0, 3.0, 3.0], [7.0, 7.0, 7.0]]),
            torch.tensor([3.0, 7.0]),
        )
        assert out["eps_gauge"] == 0.0
        assert out["mean_orbit_std"] == 0.0

    def test_affine_invariance(self):
        """The standardized-output-scale argument: eps is unchanged under
        y_hat -> a*y_hat + b."""
        rng = np.random.default_rng(4)
        orbit = torch.from_numpy(rng.normal(size=(10, 8)))
        config = torch.from_numpy(rng.normal(size=(10,)))
        ref = gauge_invariance_error(orbit, config)["eps_gauge"]
        scaled = gauge_invariance_error(2.5 * orbit - 7.0, 2.5 * config - 7.0)
        assert scaled["eps_gauge"] == pytest.approx(ref, rel=1e-12)

    def test_constant_predictor_degenerate_cases(self):
        # constant everywhere: a constant IS gauge-invariant
        out = gauge_invariance_error(
            torch.tensor([[5.0, 5.0], [5.0, 5.0]]), torch.tensor([5.0, 5.0])
        )
        assert out["eps_gauge"] == 0.0
        # orbit spread but no config spread: all variability is gauge
        out = gauge_invariance_error(
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([5.0, 5.0])
        )
        assert out["eps_gauge"] == float("inf")

    @pytest.mark.parametrize(
        "orbit,config,match",
        [
            (torch.zeros(4), torch.zeros(4), "n_configs, n_copies"),
            (torch.zeros(3, 4), torch.zeros(2), "does not match"),
            (torch.zeros(1, 4), torch.zeros(1), ">=2"),
            (torch.zeros(4, 1), torch.zeros(4), ">=2"),
        ],
    )
    def test_validation_raises(self, orbit, config, match):
        with pytest.raises(ValueError, match=match):
            gauge_invariance_error(orbit, config)


@pytest.fixture
def lattice():
    return HypercubicLattice(
        LatticeConfig(dimensions=(L, L), spacing=1.0, boundary="periodic")
    )


def orbit_predictions(variant, lattice, n_configs=3, n_copies=5, master_seed=99):
    """The measurement loop of scripts/measure_gauge_invariance.py in
    miniature: float32-stored configs, per-config seeded float64 orbits,
    every graph forwarded UNBATCHED (batch size 1).

    Batch size 1 is load-bearing: batched evaluation is
    position-in-batch dependent at float32 lsb (~1e-8) — bit-identical
    graphs placed at different positions of one Batch come back with
    slightly different outputs (observed on torch 2.x CPU), which would
    put a spurious ~1e-8 floor under Variant C's orbit std. Per-graph
    forwards are deterministic, so C's zero is exact."""
    rng = np.random.default_rng(17)
    thetas32 = rng.uniform(-np.pi, np.pi, size=(n_configs, 2, L, L)).astype(np.float32)
    builder = U1GaugeGraphBuilder(lattice, beta=BETA, variant=variant)
    torch.manual_seed(0)
    model = U1GaugeGNN(
        ModelConfig(hidden_dim=16, n_mp_blocks=2, encoder_layers=1),
        variant=variant,
        wilson_loops=WILSON_LOOPS,
        predict_q=True,
    ).eval()

    orbit_preds = {k: [] for k in OUTPUT_KEYS}
    config_preds = {k: [] for k in OUTPUT_KEYS}
    for i in range(n_configs):
        copies = gauge_orbit(
            thetas32[i], n_copies, np.random.default_rng([master_seed, i])
        )
        graphs = [builder.build({"gauge": torch.from_numpy(thetas32[i])})]
        graphs += [builder.build({"gauge": torch.from_numpy(c)}) for c in copies]
        with torch.no_grad():
            outs = [model(g) for g in graphs]  # batch size 1, see docstring
        for k in OUTPUT_KEYS:
            vals = torch.cat([o[k] for o in outs])
            config_preds[k].append(vals[0])
            orbit_preds[k].append(vals[1:])
    return {
        k: gauge_invariance_error(
            torch.stack(orbit_preds[k]), torch.stack(config_preds[k])
        )
        for k in OUTPUT_KEYS
    }


class TestModelLevelOracle:
    def test_variant_c_eps_exactly_zero(self, lattice):
        """The A-5 sanity anchor: C's inputs are bit-identical across the
        orbit, so every head's eps_gauge is exactly 0.0 — no tolerance."""
        eps = orbit_predictions("invariant_oracle", lattice)
        for k in OUTPUT_KEYS:
            assert eps[k]["mean_orbit_std"] == 0.0, k
            assert eps[k]["eps_gauge"] == 0.0, k
            assert eps[k]["config_std"] > 0.0, k

    @pytest.mark.parametrize("variant", ["link_nodes", "edge_features"])
    def test_variants_ab_eps_positive(self, lattice, variant):
        eps = orbit_predictions(variant, lattice)
        for k in OUTPUT_KEYS:
            assert eps[k]["eps_gauge"] > 0.0, k
