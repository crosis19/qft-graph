"""Tests for the Phi4Action class."""

import torch

from qft_graph.actions.phi4 import Phi4Action
from qft_graph.config import LatticeConfig, ScalarFieldConfig
from qft_graph.lattice.hypercubic import HypercubicLattice


class TestPhi4Action:
    def test_constant_field_mass_term(self):
        """For constant phi=c, kinetic term vanishes. Action = V * (m^2/2 * c^2 + lam * c^4)."""
        config = LatticeConfig(dimensions=(4, 4), spacing=1.0, boundary="periodic")
        lattice = HypercubicLattice(config)
        field_config = ScalarFieldConfig(mass_squared=1.0, coupling=0.5)
        action = Phi4Action(lattice, field_config)

        c = 2.0
        phi = torch.full((16,), c)
        S = action(phi)

        # Expected: 16 * (0.5 * 1.0 * 4.0 + 0.5 * 16.0) = 16 * (2.0 + 8.0) = 160
        expected = 16 * (0.5 * 1.0 * c**2 + 0.5 * c**4)
        assert abs(S.item() - expected) < 1e-5

    def test_free_field_action(self):
        """Free field (lambda=0): action should be purely kinetic + mass."""
        config = LatticeConfig(dimensions=(4, 4), spacing=1.0, boundary="periodic")
        lattice = HypercubicLattice(config)
        field_config = ScalarFieldConfig(mass_squared=1.0, coupling=0.0)
        action = Phi4Action(lattice, field_config)

        torch.manual_seed(42)
        phi = torch.randn(16)
        S = action(phi)
        assert S.item() > 0  # Should be positive for m^2 > 0

    def test_zero_field(self):
        """Zero field should give zero action."""
        config = LatticeConfig(dimensions=(4, 4), spacing=1.0, boundary="periodic")
        lattice = HypercubicLattice(config)
        field_config = ScalarFieldConfig(mass_squared=1.0, coupling=0.5)
        action = Phi4Action(lattice, field_config)

        phi = torch.zeros(16)
        S = action(phi)
        assert abs(S.item()) < 1e-10

    def test_local_action_sums_to_total(self, phi4_action, sample_config):
        """Sum of local action should equal total action."""
        total = phi4_action(sample_config)
        local_sum = phi4_action.local_action(sample_config).sum()
        assert torch.allclose(total, local_sum, atol=1e-5)

    def test_delta_action_consistency(self, phi4_action, sample_config):
        """delta_action should match actual action difference."""
        site = 5
        new_val = sample_config[site].item() + 0.5

        delta = phi4_action.delta_action(sample_config, site, new_val)

        S_old = phi4_action(sample_config)
        phi_new = sample_config.clone()
        phi_new[site] = new_val
        S_new = phi4_action(phi_new)

        assert torch.allclose(delta, S_new - S_old, atol=1e-5)

    def test_force_shape(self, phi4_action, sample_config):
        """Force should have same shape as phi."""
        force = phi4_action.force(sample_config)
        assert force.shape == sample_config.shape

    def test_force_gradient_consistency(self, phi4_action):
        """Force should match -dS/dphi computed by autograd."""
        phi = torch.randn(16, requires_grad=True)
        S = phi4_action(phi)
        S.backward()
        autograd_force = -phi.grad

        analytic_force = phi4_action.force(phi.detach())
        assert torch.allclose(analytic_force, autograd_force, atol=1e-4)
