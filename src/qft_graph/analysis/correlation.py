"""Two-point correlation function analysis."""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit


def fit_exponential_decay(
    G_r: np.ndarray,
    lattice_spacing: float = 1.0,
    r_min: int = 1,
    r_max: int | None = None,
) -> dict[str, float]:
    """Fit G(r) = A * exp(-r/xi) + C to extract correlation length.

    Args:
        G_r: Two-point function values at integer distances.
        lattice_spacing: Physical lattice spacing.
        r_min: Minimum distance to include in fit (skip r=0).
        r_max: Maximum distance. If None, uses all positive G values.

    Returns:
        Dict with keys: 'xi', 'amplitude', 'offset', 'chi_squared'.
    """
    r_values = np.arange(len(G_r)) * lattice_spacing

    # Filter to valid range
    mask = np.arange(len(G_r)) >= r_min
    if r_max is not None:
        mask &= np.arange(len(G_r)) <= r_max
    mask &= G_r > 0

    r_fit = r_values[mask]
    G_fit = G_r[mask]

    if len(r_fit) < 3:
        return {"xi": 0.0, "amplitude": 0.0, "offset": 0.0, "chi_squared": float("inf")}

    def model(r, A, xi, C):
        return A * np.exp(-r / xi) + C

    try:
        popt, pcov = curve_fit(
            model, r_fit, G_fit,
            p0=[G_fit[0], 1.0, 0.0],
            bounds=([0, 0.01, -np.inf], [np.inf, np.inf, np.inf]),
            maxfev=5000,
        )
        residuals = G_fit - model(r_fit, *popt)
        chi_sq = np.sum(residuals**2) / max(len(r_fit) - 3, 1)

        return {
            "xi": float(popt[1]),
            "amplitude": float(popt[0]),
            "offset": float(popt[2]),
            "chi_squared": float(chi_sq),
        }
    except RuntimeError:
        return {"xi": 0.0, "amplitude": 0.0, "offset": 0.0, "chi_squared": float("inf")}


def effective_mass(G_r: np.ndarray, lattice_spacing: float = 1.0) -> np.ndarray:
    """Compute effective mass m_eff(r) = -log(G(r+1)/G(r)) / a.

    The plateau in m_eff gives 1/xi without fitting assumptions.

    Args:
        G_r: Two-point function values.
        lattice_spacing: Physical lattice spacing.

    Returns:
        Effective mass array (one shorter than G_r).
    """
    G_r = np.asarray(G_r, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = G_r[1:] / G_r[:-1]
        ratio = np.where(ratio > 0, ratio, np.nan)
        m_eff = -np.log(ratio) / lattice_spacing
    return m_eff
