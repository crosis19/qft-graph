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
