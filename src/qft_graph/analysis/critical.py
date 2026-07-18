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
    """Find critical coupling from susceptibility peak (raw grid max).

    Prefer susceptibility_peak_quadratic, which interpolates between
    grid points and propagates errors.

    Args:
        m2_values: Array of m^2 values.
        chi_values: Array of susceptibility values.

    Returns:
        (m2_peak, chi_max) location and height of the peak.
    """
    idx = np.argmax(chi_values)
    return float(m2_values[idx]), float(chi_values[idx])


def susceptibility_peak_quadratic(
    m2_values: np.ndarray,
    chi_values: np.ndarray,
    chi_errors: np.ndarray | None = None,
    n_points: int = 5,
) -> dict[str, float]:
    """Locate the susceptibility peak by quadratic fit around the maximum.

    Fits chi(m2) = a*m2^2 + b*m2 + c to the n_points grid points nearest
    the raw maximum, weighted by 1/chi_errors when given. Peak location
    and height come from the vertex; their uncertainties are propagated
    from the (unscaled) parameter covariance of the weighted fit via the
    delta method, and the fit's chi^2/dof is reported so poorly-described
    windows are visible. This replaces an earlier parametric bootstrap,
    whose resampled fits included quadratics that badly misdescribed the
    data and produced error bars inconsistent with the underlying points.

    Args:
        m2_values: Array of m^2 values (any order).
        chi_values: Susceptibility at each m^2.
        chi_errors: Optional 1-sigma errors on chi (enables weighting and
            error propagation).
        n_points: Number of grid points in the fit window (must exceed 3
            for a meaningful chi^2/dof).

    Returns:
        Dict with m2_peak, m2_peak_err, chi_max, chi_max_err, chi2_dof
        (errors are 0.0 and chi2_dof is nan when chi_errors is None), and
        fit_ok (False when the fit had no interior maximum and the raw
        grid maximum was used instead, with its data error).
    """
    order = np.argsort(m2_values)
    m2 = np.asarray(m2_values, dtype=float)[order]
    chi = np.asarray(chi_values, dtype=float)[order]
    err = None if chi_errors is None else np.asarray(chi_errors, dtype=float)[order]

    if len(m2) < 3:
        raise ValueError("Need at least 3 points for a quadratic peak fit")
    n_points = min(n_points, len(m2))

    idx = int(np.argmax(chi))
    lo = max(0, min(idx - n_points // 2, len(m2) - n_points))
    window = slice(lo, lo + n_points)
    x, y = m2[window], chi[window]

    if err is not None:
        w = 1.0 / np.clip(err[window], 1e-12, None)
        # cov='unscaled' keeps the covariance in the units implied by the
        # supplied 1/sigma weights (no chi^2/dof rescaling)
        coef, cov = np.polyfit(x, y, 2, w=w, cov="unscaled")
        resid = (np.polyval(coef, x) - y) * w
        dof = max(len(x) - 3, 1)
        chi2_dof = float(np.sum(resid**2) / dof)
    else:
        coef = np.polyfit(x, y, 2)
        cov = None
        chi2_dof = float("nan")

    a, b, c = (float(v) for v in coef)

    def _grid_max_fallback() -> dict[str, float]:
        j = int(np.argmax(y))
        return {
            "m2_peak": float(x[j]),
            # Location known only to the grid resolution in this mode
            "m2_peak_err": float(np.diff(x).mean()) if err is not None else 0.0,
            "chi_max": float(y[j]),
            "chi_max_err": float(err[window][j]) if err is not None else 0.0,
            "chi2_dof": chi2_dof,
            "fit_ok": False,
        }

    if a >= 0:
        return _grid_max_fallback()

    m2_peak = -b / (2 * a)
    chi_max = c - b**2 / (4 * a)
    if not (x.min() <= m2_peak <= x.max()):
        # Vertex escaped the fit window: the quadratic does not describe
        # a peak here; report the raw maximum honestly instead
        return _grid_max_fallback()

    m2_peak_err = chi_max_err = 0.0
    if cov is not None:
        # Delta method: gradients of the vertex coordinates wrt (a, b, c)
        g_m2 = np.array([b / (2 * a**2), -1.0 / (2 * a), 0.0])
        g_chi = np.array([b**2 / (4 * a**2), -b / (2 * a), 1.0])
        # PDG-style scale factor: when the quadratic only approximately
        # describes the window (chi^2/dof > 1), inflate the propagated
        # errors by sqrt(chi^2/dof) rather than report model-perfect ones
        scale = float(np.sqrt(max(chi2_dof, 1.0)))
        m2_peak_err = float(np.sqrt(g_m2 @ cov @ g_m2)) * scale
        chi_max_err = float(np.sqrt(g_chi @ cov @ g_chi)) * scale

    return {
        "m2_peak": float(m2_peak),
        "m2_peak_err": m2_peak_err,
        "chi_max": float(chi_max),
        "chi_max_err": chi_max_err,
        "chi2_dof": chi2_dof,
        "fit_ok": True,
    }


def fit_gamma_over_nu(
    L_values: np.ndarray,
    chi_max: np.ndarray,
    chi_max_err: np.ndarray | None = None,
) -> tuple[float, float]:
    """Fit chi_max ~ L^(gamma/nu) via weighted linear fit of ln chi vs ln L.

    Args:
        L_values: Lattice sizes.
        chi_max: Susceptibility peak heights.
        chi_max_err: Optional 1-sigma errors on chi_max.

    Returns:
        (gamma_over_nu, error) — slope and its standard error from the
        weighted least-squares covariance.
    """
    L = np.asarray(L_values, dtype=float)
    chi = np.asarray(chi_max, dtype=float)
    if len(L) < 2:
        raise ValueError("Need at least 2 lattice sizes")

    x = np.log(L)
    y = np.log(chi)
    if chi_max_err is not None:
        sigma = np.asarray(chi_max_err, dtype=float) / chi  # d(ln chi)
        sigma = np.clip(sigma, 1e-12, None)
    else:
        sigma = np.ones_like(y)

    w = 1.0 / sigma**2
    delta = w.sum() * (w * x * x).sum() - (w * x).sum() ** 2
    slope = (w.sum() * (w * x * y).sum() - (w * x).sum() * (w * y).sum()) / delta
    slope_err = np.sqrt(w.sum() / delta)

    if chi_max_err is None and len(L) > 2:
        # Unweighted: estimate error from fit residuals instead
        intercept = ((w * y).sum() - slope * (w * x).sum()) / w.sum()
        resid = y - (slope * x + intercept)
        s2 = (resid**2).sum() / (len(L) - 2)
        slope_err = np.sqrt(s2 * w.sum() / delta)

    return float(slope), float(slope_err)


def crossing_with_errors(
    m2_values: np.ndarray,
    y1: np.ndarray,
    e1: np.ndarray,
    y2: np.ndarray,
    e2: np.ndarray,
    n_boot: int = 500,
    seed: int = 0,
) -> tuple[float, float]:
    """Crossing point of two curves (e.g. xi/L for L1, L2) with errors.

    Interpolates both curves onto a fine common grid, locates the zero of
    their difference, and estimates the error by parametric bootstrap of
    the data points within their 1-sigma errors.

    Args:
        m2_values: Common m^2 grid of both curves (any order).
        y1, e1: First curve and its errors.
        y2, e2: Second curve and its errors.
        n_boot: Bootstrap resamples.
        seed: Bootstrap RNG seed.

    Returns:
        (m2_cross, error). Returns (nan, inf) if the central curves do
        not cross.
    """
    order = np.argsort(m2_values)
    m2 = np.asarray(m2_values, dtype=float)[order]
    y1 = np.asarray(y1, dtype=float)[order]
    y2 = np.asarray(y2, dtype=float)[order]
    e1 = np.asarray(e1, dtype=float)[order]
    e2 = np.asarray(e2, dtype=float)[order]

    fine = np.linspace(m2.min(), m2.max(), 400)

    def find_crossing(a: np.ndarray, b: np.ndarray) -> float:
        diff = np.interp(fine, m2, a) - np.interp(fine, m2, b)
        sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]
        if len(sign_changes) == 0:
            return np.nan
        i = sign_changes[0]
        return fine[i] - diff[i] * (fine[i + 1] - fine[i]) / (diff[i + 1] - diff[i])

    center = find_crossing(y1, y2)
    if np.isnan(center):
        return float("nan"), float("inf")

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        c = find_crossing(y1 + rng.normal(0.0, e1), y2 + rng.normal(0.0, e2))
        if not np.isnan(c):
            boots.append(c)
    err = float(np.std(boots)) if len(boots) > 1 else float("inf")
    return float(center), err
