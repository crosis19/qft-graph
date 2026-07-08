"""Unit tests for the Wilson gauge action and U(1) link utilities."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from qft_graph.actions.wilson import WilsonGaugeAction
from qft_graph.config import LatticeConfig
from qft_graph.fields.gauge import (
    U1GaugeField,
    gauge_transform,
    plaquette_angles,
    topological_charge,
    wilson_loop,
    wilson_loop_angles,
    wrap_angle,
)
from qft_graph.lattice.hypercubic import HypercubicLattice


@pytest.fixture
def lattice8():
    return HypercubicLattice(LatticeConfig(dimensions=(8, 8)))


class TestWrapAngle:
    def test_range(self):
        x = np.linspace(-10, 10, 1001)
        w = wrap_angle(x)
        assert np.all(w > -np.pi) and np.all(w <= np.pi)

    def test_identity_inside_range(self):
        x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        assert np.allclose(wrap_angle(x), x)

    def test_wraps_by_2pi(self):
        assert np.allclose(wrap_angle(np.array([2 * np.pi + 0.3])), [0.3])
        assert np.allclose(wrap_angle(np.array([-2 * np.pi - 0.3])), [-0.3])


class TestColdConfiguration:
    def test_zero_action_and_charge(self, lattice8):
        theta = np.zeros((2, 8, 8))
        action = WilsonGaugeAction(lattice8, beta=1.7)
        assert float(action(torch.from_numpy(theta))) == 0.0
        assert topological_charge(theta) == 0.0
        assert wilson_loop(theta, 3, 2) == 1.0

    def test_pure_gauge_is_zero_action(self, lattice8):
        # A gauge transform of the cold config has theta_P = 0 exactly
        rng = np.random.default_rng(1)
        alpha = rng.uniform(-np.pi, np.pi, size=(8, 8))
        theta = gauge_transform(np.zeros((2, 8, 8)), alpha)
        action = WilsonGaugeAction(lattice8, beta=1.7)
        assert abs(float(action(torch.from_numpy(theta)))) < 1e-10
        assert abs(topological_charge(theta)) < 1e-10


class TestWilsonGaugeAction:
    def test_local_action_sums_to_total(self, lattice8):
        rng = np.random.default_rng(2)
        theta = rng.uniform(-np.pi, np.pi, size=(2, 8, 8))
        action = WilsonGaugeAction(lattice8, beta=1.0)
        total = float(action(torch.from_numpy(theta)))
        local = action.local_action(torch.from_numpy(theta))
        assert local.shape == (64,)
        assert abs(local.sum().item() - total) < 1e-10

    def test_action_nonnegative_and_bounded(self, lattice8):
        rng = np.random.default_rng(3)
        theta = rng.uniform(-np.pi, np.pi, size=(2, 8, 8))
        beta = 2.0
        action = WilsonGaugeAction(lattice8, beta)
        s = float(action(torch.from_numpy(theta)))
        assert 0.0 <= s <= 2.0 * beta * 64  # 0 <= 1 - cos <= 2 per plaquette

    def test_force_matches_finite_differences(self, lattice8):
        rng = np.random.default_rng(4)
        theta = rng.uniform(-np.pi, np.pi, size=(2, 8, 8))
        action = WilsonGaugeAction(lattice8, beta=1.3)
        force = action.force(torch.from_numpy(theta)).numpy()

        eps = 1e-6
        for mu, x1, x2 in [(0, 0, 0), (0, 3, 5), (1, 2, 7), (1, 7, 0)]:
            tp = theta.copy()
            tp[mu, x1, x2] += eps
            tm = theta.copy()
            tm[mu, x1, x2] -= eps
            ds = (
                float(action(torch.from_numpy(tp)))
                - float(action(torch.from_numpy(tm)))
            ) / (2 * eps)
            assert abs(-ds - force[mu, x1, x2]) < 1e-5

    def test_requires_2d(self):
        lattice3d = HypercubicLattice(LatticeConfig(dimensions=(4, 4, 4)))
        with pytest.raises(ValueError):
            WilsonGaugeAction(lattice3d, beta=1.0)


class TestWilsonLoops:
    def test_1x1_loop_is_plaquette(self):
        rng = np.random.default_rng(5)
        theta = rng.uniform(-np.pi, np.pi, size=(2, 8, 8))
        assert np.allclose(wilson_loop_angles(theta, 1, 1), plaquette_angles(theta))

    def test_loop_angle_additivity(self):
        # A 2x1 loop equals the sum of its two 1x1 plaquettes
        rng = np.random.default_rng(6)
        theta = rng.uniform(-np.pi, np.pi, size=(2, 8, 8))
        p = plaquette_angles(theta)
        loop21 = wilson_loop_angles(theta, 2, 1)
        assert np.allclose(loop21, p + np.roll(p, -1, axis=0))


class TestU1GaugeFieldNodes:
    def test_node_features_shape_and_values(self):
        field = U1GaugeField(L=4)
        theta = torch.zeros(2, 4, 4)
        theta[0, 0, 0] = np.pi / 2
        feats = field.node_features(theta)
        assert feats.shape == (32, 4)
        # First link: theta = pi/2 -> [cos, sin] = [0, 1]; mu=1 onehot
        assert torch.allclose(feats[0], torch.tensor([0.0, 1.0, 1.0, 0.0]), atol=1e-6)
        # A mu=2 link (index 16..31): theta = 0 -> [1, 0]; mu=2 onehot
        assert torch.allclose(feats[16], torch.tensor([1.0, 0.0, 0.0, 1.0]), atol=1e-6)

    def test_initialize_modes(self):
        field = U1GaugeField(L=6)
        hot = field.initialize(36, mode="hot")
        cold = field.initialize(36, mode="cold")
        assert hot.shape == (2, 6, 6) and cold.shape == (2, 6, 6)
        assert torch.all(cold == 0)
        assert hot.abs().max() <= np.pi
