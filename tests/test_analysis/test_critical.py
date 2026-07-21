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
        res = susceptibility_peak_quadratic(m2, chi, err, n_points=7)
        assert res["m2_peak_err"] > 0 and res["chi_max_err"] > 0
        assert res["fit_ok"] and np.isfinite(res["chi2_dof"])
        assert abs(res["m2_peak"] - (-1.3)) < 3 * res["m2_peak_err"] + 0.02
        # Covariance-propagated peak-height error must be commensurate with
        # the data errors, never wildly larger (the old bootstrap failure)
        assert res["chi_max_err"] < 3 * err.max()


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


class TestCorrectionsToScaling:
    """chi_max = A L^(gamma/nu) (1 + b L^(-omega)) recovers gamma/nu when the
    naive power law is biased by the finite-L correction."""

    L = np.array([16, 24, 32, 48, 64, 96, 128], dtype=float)

    def _synthetic(self, A=1.5, gnu=1.75, b=0.8, omega=1.0):
        chi = A * self.L**gnu * (1.0 + b * self.L ** (-omega))
        return chi, 0.01 * chi  # 1% errors for weighting

    def test_recovers_slope_despite_correction(self):
        from qft_graph.analysis.critical import (
            fit_gamma_over_nu,
            fit_gamma_over_nu_corrections,
        )

        chi, err = self._synthetic()
        naive, _ = fit_gamma_over_nu(self.L, chi, err)
        res = fit_gamma_over_nu_corrections(self.L, chi, err)  # omega floats
        assert res["fit_ok"]
        # The model is exact, so the corrected fit recovers gamma/nu tightly...
        assert abs(res["gamma_over_nu"] - 1.75) < 0.03
        assert abs(res["omega"] - 1.0) < 0.3
        # ...while the naive power law is pulled away from 1.75 by the correction.
        assert abs(naive - 1.75) > abs(res["gamma_over_nu"] - 1.75)

    def test_fixed_omega(self):
        from qft_graph.analysis.critical import fit_gamma_over_nu_corrections

        chi, err = self._synthetic(omega=2.0)
        res = fit_gamma_over_nu_corrections(self.L, chi, err, omega=2.0)
        assert res["fit_ok"] and res["n_params"] == 3
        assert abs(res["gamma_over_nu"] - 1.75) < 0.02
        assert res["omega"] == 2.0

    def test_matches_naive_without_correction(self):
        from qft_graph.analysis.critical import (
            fit_gamma_over_nu,
            fit_gamma_over_nu_corrections,
        )

        chi = 2.0 * self.L**1.75  # b = 0, pure power law
        err = 0.01 * chi
        naive, _ = fit_gamma_over_nu(self.L, chi, err)
        res = fit_gamma_over_nu_corrections(self.L, chi, err)
        assert abs(res["gamma_over_nu"] - naive) < 0.05
        assert abs(res["gamma_over_nu"] - 1.75) < 0.05

    def test_too_few_points(self):
        from qft_graph.analysis.critical import fit_gamma_over_nu_corrections

        L = np.array([16, 32, 64], dtype=float)
        chi = 2.0 * L**1.75
        res = fit_gamma_over_nu_corrections(L, chi, 0.01 * chi)  # needs >=5
        assert not res["fit_ok"]


class TestAICModelAverage:
    """AIC-weighted gamma/nu over varying small-L cuts (arXiv:2008.01069)."""

    L = np.array([16, 24, 32, 48, 64, 96, 128], dtype=float)

    def test_average_less_biased_than_full_fit(self):
        from qft_graph.analysis.critical import (
            aic_model_average_gamma_nu,
            fit_gamma_over_nu,
        )

        chi = 1.5 * self.L**1.75 * (1.0 + 0.8 * self.L ** (-1.0))
        err = 0.01 * chi
        naive_full, _ = fit_gamma_over_nu(self.L, chi, err)
        res = aic_model_average_gamma_nu(self.L, chi, err)
        assert res["gamma_over_nu_err"] > 0
        assert res["sys_err"] >= 0 and res["stat_err"] > 0
        # Dropping the (biased) small-L points via AIC weighting reduces the bias.
        assert abs(res["gamma_over_nu"] - 1.75) < abs(naive_full - 1.75)
        assert len(res["models"]) >= 3

    def test_clean_power_law_prefers_full_data(self):
        from qft_graph.analysis.critical import aic_model_average_gamma_nu

        chi = 2.0 * self.L**1.75  # no correction
        err = 0.01 * chi
        res = aic_model_average_gamma_nu(self.L, chi, err)
        assert abs(res["gamma_over_nu"] - 1.75) < 0.02
        # With nothing to gain from dropping data, the 2*N_cut penalty makes the
        # full-data fit (n_cut=0) the highest-weight model.
        full = next(m for m in res["models"] if m["n_cut"] == 0)
        assert full["weight"] == max(m["weight"] for m in res["models"])


class TestPseudocriticalShift:
    """nu and m2_c from m2_peak(L) = m2_c - a L^{-1/nu}."""

    L = np.array([16, 24, 32, 48, 64, 96, 128], dtype=float)

    def _synthetic(self, m2c=-2.2, a=1.6, nu=1.0):
        m2p = m2c - a * self.L ** (-1.0 / nu)
        return m2p, 0.002 * np.ones_like(m2p)

    def test_recovers_nu_and_m2c(self):
        from qft_graph.analysis.critical import fit_nu_from_pseudocritical_shifts

        m2p, err = self._synthetic(m2c=-2.2, a=1.6, nu=1.0)
        res = fit_nu_from_pseudocritical_shifts(self.L, m2p, err)
        assert res["fit_ok"]
        assert abs(res["nu"] - 1.0) < 0.05
        assert abs(res["m2_c"] - (-2.2)) < 0.01

    def test_recovers_non_ising_nu(self):
        from qft_graph.analysis.critical import fit_nu_from_pseudocritical_shifts

        m2p, err = self._synthetic(m2c=-1.5, a=2.0, nu=0.63)
        res = fit_nu_from_pseudocritical_shifts(self.L, m2p, err)
        assert abs(res["nu"] - 0.63) < 0.05

    def test_fixed_nu(self):
        from qft_graph.analysis.critical import fit_nu_from_pseudocritical_shifts

        m2p, err = self._synthetic(m2c=-2.2, a=1.6, nu=1.0)
        res = fit_nu_from_pseudocritical_shifts(self.L, m2p, err, fix_nu=1.0)
        assert res["fit_ok"] and res["n_params"] == 2
        assert abs(res["m2_c"] - (-2.2)) < 0.01


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
