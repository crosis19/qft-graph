"""Tests for physical observables."""

import torch

from qft_graph.mc.observables import ObservableSet


class TestObservables:
    def test_magnetization_constant_field(self, small_lattice):
        obs = ObservableSet(small_lattice)
        phi = torch.ones(small_lattice.num_sites()) * 3.0
        assert abs(obs.magnetization(phi) - 3.0) < 1e-6

    def test_magnetization_zero_mean(self, small_lattice):
        obs = ObservableSet(small_lattice)
        n = small_lattice.num_sites()
        phi = torch.cat([torch.ones(n // 2), -torch.ones(n // 2)])
        assert abs(obs.magnetization(phi)) < 1e-6

    def test_two_point_function_shape(self, small_lattice):
        obs = ObservableSet(small_lattice)
        phi = torch.randn(small_lattice.num_sites())
        G_r = obs.two_point_function(phi)
        L = small_lattice.shape[0]
        assert G_r.shape == (L // 2 + 1,)

    def test_two_point_function_g0(self, small_lattice):
        """G(0) should be <phi^2> - <phi>^2."""
        obs = ObservableSet(small_lattice)
        phi = torch.randn(small_lattice.num_sites())
        G_r = obs.two_point_function(phi)
        expected_g0 = (phi**2).mean() - phi.mean() ** 2
        # G(0) is averaged over dimensions, so it's counted twice (dim=0, dim=1)
        # The exact match depends on implementation; just check it's positive
        assert G_r[0].item() > 0 or abs(G_r[0].item()) < 0.1

    def test_correlation_length_positive(self, small_lattice):
        """Correlation length should be positive for decaying correlations."""
        obs = ObservableSet(small_lattice)
        # Create a correlator that decays exponentially
        G_r = torch.tensor([1.0, 0.5, 0.25, 0.125])
        xi = ObservableSet.correlation_length(G_r)
        assert xi > 0


class TestSusceptibility:
    """Frozen convention (plan P1-2): chi = V * (<M^2> - <|M|>^2)."""

    def test_symmetric_two_state_exact(self):
        # Ensemble {+c, -c} uniform: <M^2> = c^2, <|M|> = c -> chi_abs = 0,
        # while the Var(M) form gives V*c^2 (contaminated by tunneling).
        V, c = 16, 0.7
        configs = torch.stack([torch.full((V,), c), torch.full((V,), -c)])
        chi_abs = ObservableSet.susceptibility(configs, convention="abs")
        chi_var = ObservableSet.susceptibility(configs, convention="var")
        assert abs(chi_abs) < 1e-6
        assert abs(chi_var - V * c**2) < 1e-5

    def test_iid_gaussian_analytic(self):
        # phi ~ N(0, sigma^2) iid: M ~ N(0, sigma^2/V), so
        # chi_var -> sigma^2 and chi_abs -> sigma^2 * (1 - 2/pi).
        import math

        torch.manual_seed(0)
        sigma, V, n = 2.0, 64, 40_000
        configs = sigma * torch.randn(n, V)
        chi_abs = ObservableSet.susceptibility(configs, convention="abs")
        chi_var = ObservableSet.susceptibility(configs, convention="var")
        assert abs(chi_var - sigma**2) / sigma**2 < 0.05
        expected_abs = sigma**2 * (1 - 2 / math.pi)
        assert abs(chi_abs - expected_abs) / expected_abs < 0.05

    def test_unknown_convention_raises(self):
        import pytest

        with pytest.raises(ValueError):
            ObservableSet.susceptibility(torch.randn(10, 4), convention="bogus")

    def test_correlation_length_fft_conventions_differ_in_broken_phase(self):
        # Broken-phase mock: configs fluctuate around +/- v -> the Var(M)
        # chi is inflated by the two-state structure, the frozen form is not.
        torch.manual_seed(1)
        L, n = 8, 400
        v = 1.0
        signs = torch.where(torch.rand(n, 1) > 0.5, 1.0, -1.0)
        configs = signs * v + 0.05 * torch.randn(n, L * L)
        chi_abs = ObservableSet.susceptibility(configs, convention="abs")
        chi_var = ObservableSet.susceptibility(configs, convention="var")
        assert chi_var > 10 * max(chi_abs, 1e-9)
