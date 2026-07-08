"""Exact validation oracles for the U(1) heatbath sampler (plan §3.2).

These tests are the arbiters of the staple/plaquette sign conventions:
if the sampler disagrees with the torus-exact character expansion, fix the
sampler, never the test.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from qft_graph.actions.wilson import WilsonGaugeAction
from qft_graph.config import LatticeConfig, MCConfig
from qft_graph.fields.gauge import (
    plaquette_angles,
    random_gauge_transform,
    topological_charge,
    wilson_loop,
)
from qft_graph.lattice.hypercubic import HypercubicLattice
from qft_graph.mc.analysis import binned_jackknife
from qft_graph.mc.exact_u1 import (
    exact_mean_plaquette,
    exact_wilson_loop_infinite_volume,
    string_tension,
)
from qft_graph.mc.heatbath import U1HeatbathSampler


def run_heatbath(beta: float, L: int, n_configs: int = 800, seed: int = 7):
    lattice = HypercubicLattice(LatticeConfig(dimensions=(L, L)))
    action = WilsonGaugeAction(lattice, beta)
    sampler = U1HeatbathSampler(
        action,
        MCConfig(n_configs=n_configs, n_thermalization=300, n_sweeps_between=2, seed=seed),
    )
    return action, sampler.generate(n_configs)


def mean_plaquette_series(configs: torch.Tensor) -> torch.Tensor:
    return torch.tensor(
        [
            float(np.cos(plaquette_angles(c.numpy().astype(np.float64))).mean())
            for c in configs
        ]
    )


class TestExactPlaquetteOracle:
    """Primary sampler test: torus-exact <cos theta_P> via character expansion."""

    @pytest.mark.parametrize("beta,L", [(0.5, 8), (1.0, 8), (2.0, 8), (1.0, 4)])
    def test_mean_plaquette_matches_character_expansion(self, beta, L):
        _, result = run_heatbath(beta, L)
        series = mean_plaquette_series(result.configurations)
        val, err = binned_jackknife(series, lambda s: float(s.mean()), n_bins=20)
        exact = exact_mean_plaquette(beta, L)
        # 4-sigma statistical window plus a tiny absolute floor
        assert abs(val - exact) < 4 * err + 1e-4, (
            f"<cosP>={val:.5f}+-{err:.5f} vs exact {exact:.5f}"
        )

    def test_infinite_volume_anchor(self):
        # I_1(1)/I_0(1) ~ 0.4464; L=32 is effectively infinite volume
        assert abs(exact_mean_plaquette(1.0, 32) - 0.4464) < 1e-3

    def test_finite_volume_correction_direction(self):
        # Finite-volume value differs measurably from infinite volume at L=2
        assert exact_mean_plaquette(1.0, 2) != pytest.approx(
            exact_mean_plaquette(1.0, 32), abs=1e-6
        )


class TestWilsonLoopAreaLaw:
    def test_area_law(self):
        beta, L = 2.0, 16
        _, result = run_heatbath(beta, L, n_configs=800, seed=11)
        for R, T in [(1, 1), (2, 2), (2, 4)]:
            series = torch.tensor(
                [
                    wilson_loop(c.numpy().astype(np.float64), R, T)
                    for c in result.configurations
                ]
            )
            val, err = binned_jackknife(series, lambda s: float(s.mean()), n_bins=20)
            exact = exact_wilson_loop_infinite_volume(beta, R, T)
            assert abs(val - exact) < 4 * err + 1e-3, (
                f"W({R},{T})={val:.5f}+-{err:.5f} vs exact {exact:.5f}"
            )

    def test_string_tension_consistency(self):
        # sigma = -ln(I1/I0) reproduces the area-law decay per unit area
        beta = 2.0
        w11 = exact_wilson_loop_infinite_volume(beta, 1, 1)
        assert abs(-np.log(w11) - string_tension(beta)) < 1e-12


class TestTopologicalCharge:
    def test_integer_on_all_sampled_configs(self):
        _, result = run_heatbath(1.0, 8, n_configs=100, seed=3)
        for c in result.configurations:
            q = topological_charge(c.numpy().astype(np.float64))
            assert abs(q - round(q)) < 1e-10

    def test_hot_start_configs_integer(self):
        rng = np.random.default_rng(5)
        for _ in range(20):
            theta = rng.uniform(-np.pi, np.pi, size=(2, 8, 8))
            q = topological_charge(theta)
            assert abs(q - round(q)) < 1e-10


class TestGaugeInvariance:
    def test_action_q_loops_invariant_under_100_transforms(self):
        lattice = HypercubicLattice(LatticeConfig(dimensions=(8, 8)))
        action = WilsonGaugeAction(lattice, 1.0)
        rng = np.random.default_rng(0)
        theta = rng.uniform(-np.pi, np.pi, size=(2, 8, 8))
        s0 = float(action(torch.from_numpy(theta)))
        q0 = topological_charge(theta)
        w0 = {rt: wilson_loop(theta, *rt) for rt in [(1, 1), (2, 2), (3, 3)]}

        for _ in range(100):
            theta_g = random_gauge_transform(theta, rng)
            assert abs(float(action(torch.from_numpy(theta_g))) - s0) < 1e-10
            assert abs(topological_charge(theta_g) - q0) < 1e-10
            for rt, w in w0.items():
                assert abs(wilson_loop(theta_g, *rt) - w) < 1e-10
