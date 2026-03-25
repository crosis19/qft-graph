"""Critical exponent extraction via finite-size scaling."""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit


def finite_size_scaling_crossing(
    L_values: list[int],
    xi_over_L: dict[int, list[tuple[float, float]]],
) -> tuple[float, float]:
    """Find critical coupling m2_c from xi/L crossing point.

    At the critical point, xi/L is scale-invariant: the curves for
    different L cross at m2_c. This is the standard method for locating
    phase transitions in the Ising universality class.

    Args:
        L_values: List of lattice sizes [8, 16, 32].
        xi_over_L: Dict mapping L -> list of (m2, xi/L) tuples.

    Returns:
        (m2_c, m2_c_error) estimated critical coupling.
    """
    # Find pairwise crossings between consecutive L values
    crossings = []

    sorted_L = sorted(L_values)
    for i in range(len(sorted_L) - 1):
        L1, L2 = sorted_L[i], sorted_L[i + 1]
        data1 = sorted(xi_over_L[L1], key=lambda t: t[0])
        data2 = sorted(xi_over_L[L2], key=lambda t: t[0])

        # Interpolate both to common m2 grid
        m2_1, xi_L_1 = zip(*data1)
        m2_2, xi_L_2 = zip(*data2)
        m2_common = np.linspace(
            max(min(m2_1), min(m2_2)),
            min(max(m2_1), max(m2_2)),
            200,
        )
        interp1 = np.interp(m2_common, m2_1, xi_L_1)
        interp2 = np.interp(m2_common, m2_2, xi_L_2)

        # Find zero crossing of difference
        diff = interp1 - interp2
        sign_changes = np.where(np.diff(np.sign(diff)))[0]

        for idx in sign_changes:
            # Linear interpolation to find exact crossing
            m2_cross = m2_common[idx] - diff[idx] * (
                m2_common[idx + 1] - m2_common[idx]
            ) / (diff[idx + 1] - diff[idx])
            crossings.append(m2_cross)

    if not crossings:
        return 0.0, float("inf")

    m2_c = np.mean(crossings)
    m2_c_err = np.std(crossings) if len(crossings) > 1 else 0.0
    return float(m2_c), float(m2_c_err)


def extract_nu(
    L_values: list[int],
    xi_over_L: dict[int, list[tuple[float, float]]],
    m2_c: float,
) -> tuple[float, float]:
    """Extract critical exponent nu from scaling collapse.

    Near the critical point: xi/L = f((m2 - m2_c) * L^{1/nu})

    We find nu that gives the best data collapse by minimizing
    the scatter in the scaled variable.

    Args:
        L_values: List of lattice sizes.
        xi_over_L: Dict mapping L -> list of (m2, xi/L) tuples.
        m2_c: Critical coupling from crossing analysis.

    Returns:
        (nu, nu_error) estimated critical exponent.
    """
    def scaling_quality(nu: float) -> float:
        """Compute quality of data collapse for a given nu."""
        all_x = []
        all_y = []
        for L in L_values:
            for m2, xi_L_val in xi_over_L[L]:
                x = (m2 - m2_c) * L ** (1.0 / nu)
                all_x.append(x)
                all_y.append(xi_L_val)

        all_x = np.array(all_x)
        all_y = np.array(all_y)

        # Sort by x and compute smoothness of the collapsed curve
        order = np.argsort(all_x)
        all_x = all_x[order]
        all_y = all_y[order]

        # Use sum of squared differences between adjacent points
        # after sorting by the scaling variable
        if len(all_y) < 2:
            return float("inf")
        return np.sum(np.diff(all_y) ** 2)

    # Grid search + refinement
    best_nu = 1.0
    best_quality = float("inf")

    for nu_trial in np.linspace(0.5, 2.0, 100):
        q = scaling_quality(nu_trial)
        if q < best_quality:
            best_quality = q
            best_nu = nu_trial

    # Refine with finer grid around best
    for nu_trial in np.linspace(best_nu - 0.1, best_nu + 0.1, 100):
        if nu_trial <= 0:
            continue
        q = scaling_quality(nu_trial)
        if q < best_quality:
            best_quality = q
            best_nu = nu_trial

    # Estimate error from width of minimum
    threshold = 1.1 * best_quality
    nu_range = [nu for nu in np.linspace(0.5, 2.0, 500)
                if scaling_quality(nu) < threshold]
    nu_err = (max(nu_range) - min(nu_range)) / 2 if len(nu_range) > 1 else 0.1

    return float(best_nu), float(nu_err)


def susceptibility_peak(
    m2_values: np.ndarray,
    chi_values: np.ndarray,
) -> tuple[float, float]:
    """Find critical coupling from susceptibility peak.

    The susceptibility chi diverges at the phase transition.
    On a finite lattice, it has a peak at the pseudo-critical coupling.

    Args:
        m2_values: Array of m^2 values.
        chi_values: Array of susceptibility values.

    Returns:
        (m2_peak, chi_max) location and height of the peak.
    """
    idx = np.argmax(chi_values)
    return float(m2_values[idx]), float(chi_values[idx])
