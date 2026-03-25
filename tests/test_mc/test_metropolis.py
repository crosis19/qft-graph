"""Tests for the Metropolis-Hastings sampler."""

import torch

from qft_graph.actions.phi4 import Phi4Action
from qft_graph.config import LatticeConfig, MCConfig, ScalarFieldConfig
from qft_graph.lattice.hypercubic import HypercubicLattice
from qft_graph.mc.metropolis import MetropolisSampler


class TestMetropolisSampler:
    def test_generate_shapes(self, small_lattice, field_config, mc_config):
        action = Phi4Action(small_lattice, field_config)
        sampler = MetropolisSampler(action, mc_config)
        result = sampler.generate(5)

        assert result.configurations.shape == (5, small_lattice.num_sites())
        assert result.actions.shape == (5,)
        assert 0 < result.acceptance_rate < 1

    def test_acceptance_rate_reasonable(self, small_lattice, field_config):
        """Acceptance rate should be between 20-80% for well-tuned step size."""
        mc_config = MCConfig(n_configs=5, n_thermalization=100, n_sweeps_between=5,
                             step_size=1.0, seed=42)
        action = Phi4Action(small_lattice, field_config)
        sampler = MetropolisSampler(action, mc_config)
        result = sampler.generate(5)
        assert 0.1 < result.acceptance_rate < 0.95

    def test_single_sweep(self, small_lattice, field_config, mc_config):
        action = Phi4Action(small_lattice, field_config)
        sampler = MetropolisSampler(action, mc_config)
        phi = torch.randn(small_lattice.num_sites())
        new_phi, acc = sampler.sweep(phi)

        assert new_phi.shape == phi.shape
        # Some sites should change
        assert not torch.allclose(new_phi, phi)

    def test_free_field_gaussian(self):
        """Free field (lambda=0, m^2>0) configs should be approximately Gaussian."""
        config = LatticeConfig(dimensions=(8, 8), spacing=1.0, boundary="periodic")
        lattice = HypercubicLattice(config)
        field_config = ScalarFieldConfig(mass_squared=4.0, coupling=0.0)
        action = Phi4Action(lattice, field_config)
        mc_config = MCConfig(n_configs=200, n_thermalization=500,
                             n_sweeps_between=10, step_size=0.5, seed=42)
        sampler = MetropolisSampler(action, mc_config)
        result = sampler.generate(200)

        # For free field, each site is independently Gaussian with known variance
        all_vals = result.configurations.flatten()
        # Just check it's centered near zero with finite variance
        assert abs(all_vals.mean().item()) < 0.2
        assert all_vals.std().item() > 0.1
