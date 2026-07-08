"""Tests for critical exponent extraction."""

import numpy as np

from qft_graph.analysis.critical import (
    finite_size_scaling_crossing,
    susceptibility_peak,
)


class TestCriticalAnalysis:
    def test_susceptibility_peak_finds_maximum(self):
        m2 = np.array([-1.0, -0.7, -0.5, -0.3, 0.0])
        chi = np.array([1.0, 5.0, 20.0, 4.0, 0.5])
        m2_peak, chi_max = susceptibility_peak(m2, chi)
        assert m2_peak == -0.5
        assert chi_max == 20.0

    def test_crossing_with_synthetic_data(self):
        """Two curves that cross should yield a crossing point."""
        # Simulate xi/L for two lattice sizes
        m2_vals = np.linspace(-1.0, 0.0, 50)
        xi_over_L = {
            8: [(m, 0.5 + 0.3 * np.tanh(5 * (m + 0.5))) for m in m2_vals],
            16: [(m, 0.5 + 0.5 * np.tanh(8 * (m + 0.5))) for m in m2_vals],
        }

        m2_c, _ = finite_size_scaling_crossing([8, 16], xi_over_L)
        # The curves are designed to cross near m2 = -0.5
        assert abs(m2_c - (-0.5)) < 0.1


class TestQuadraticPeak:
    def test_exact_parabola(self):
        import numpy as np
        from qft_graph.analysis.critical import susceptibility_peak_quadratic

        m2 = np.linspace(-2.0, 0.0, 21)
        chi = 10.0 - 5.0 * (m2 + 1.3) ** 2
        res = susceptibility_peak_quadratic(m2, chi)
        assert abs(res["m2_peak"] - (-1.3)) < 1e-9
        assert abs(res["chi_max"] - 10.0) < 1e-9

    def test_with_errors_gives_uncertainties(self):
        import numpy as np
        from qft_graph.analysis.critical import susceptibility_peak_quadratic

        rng = np.random.default_rng(3)
        m2 = np.linspace(-2.0, 0.0, 21)
        chi_true = 10.0 - 5.0 * (m2 + 1.3) ** 2
        noise = 0.05  # realistic: chi errors ~ 0.5% of peak height
        err = noise * np.ones_like(chi_true)
        chi = chi_true + rng.normal(0, noise, len(m2))
        res = susceptibility_peak_quadratic(m2, chi, err, n_points=7, seed=0)
        assert res["m2_peak_err"] > 0 and res["chi_max_err"] > 0
        assert abs(res["m2_peak"] - (-1.3)) < 3 * res["m2_peak_err"] + 0.02


class TestGammaOverNu:
    def test_exact_power_law(self):
        import numpy as np
        from qft_graph.analysis.critical import fit_gamma_over_nu

        L = np.array([16, 32, 64])
        chi = 2.0 * L**1.75
        slope, err = fit_gamma_over_nu(L, chi)
        assert abs(slope - 1.75) < 1e-10

    def test_weighted_fit_with_noise(self):
        import numpy as np
        from qft_graph.analysis.critical import fit_gamma_over_nu

        rng = np.random.default_rng(5)
        L = np.array([16, 32, 64])
        chi_true = 2.0 * L**1.75
        chi_err = 0.05 * chi_true
        chi = chi_true * (1 + rng.normal(0, 0.05, 3))
        slope, err = fit_gamma_over_nu(L, chi, chi_err)
        assert err > 0
        assert abs(slope - 1.75) < 3 * err + 0.1


class TestCrossingWithErrors:
    def test_two_lines_known_crossing(self):
        import numpy as np
        from qft_graph.analysis.critical import crossing_with_errors

        m2 = np.linspace(-3.0, -1.0, 21)
        y1 = 0.5 + 0.3 * (m2 + 2.0)   # crosses y2 at m2 = -2
        y2 = 0.5 - 0.2 * (m2 + 2.0)
        e = 0.005 * np.ones_like(m2)
        center, err = crossing_with_errors(m2, y1, e, y2, e, seed=0)
        assert abs(center - (-2.0)) < 0.02
        assert 0 < err < 0.1

    def test_no_crossing_returns_nan(self):
        import numpy as np
        from qft_graph.analysis.critical import crossing_with_errors

        m2 = np.linspace(-3.0, -1.0, 11)
        y1 = np.full_like(m2, 1.0)
        y2 = np.full_like(m2, 0.5)
        e = 0.01 * np.ones_like(m2)
        center, err = crossing_with_errors(m2, y1, e, y2, e, seed=0)
        assert np.isnan(center) and err == float("inf")
