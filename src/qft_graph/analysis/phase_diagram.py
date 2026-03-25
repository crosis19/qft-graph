"""Phase transition detection and phase diagram mapping."""

from __future__ import annotations

import numpy as np


def binder_cumulant(m2_samples: np.ndarray) -> float:
    """Compute Binder cumulant U_4 = 1 - <m^4> / (3 <m^2>^2).

    The Binder cumulant is scale-invariant at the critical point:
    U_4 -> 2/3 in the ordered phase, U_4 -> 0 in the disordered phase.

    Args:
        m2_samples: Array of magnetization samples (not squared — we square internally).

    Returns:
        Binder cumulant value.
    """
    m2 = m2_samples**2
    m4 = m2_samples**4
    m2_mean = m2.mean()
    m4_mean = m4.mean()

    if m2_mean == 0:
        return 0.0

    return 1.0 - m4_mean / (3.0 * m2_mean**2)


def locate_phase_transition(
    m2_values: np.ndarray,
    observable_values: np.ndarray,
    method: str = "peak",
) -> float:
    """Locate the phase transition from observable data.

    Args:
        m2_values: Array of coupling values.
        observable_values: Array of observable (susceptibility, derivative, etc.)
        method: 'peak' for maximum, 'inflection' for steepest change.

    Returns:
        Estimated critical coupling m2_c.
    """
    if method == "peak":
        idx = np.argmax(observable_values)
    elif method == "inflection":
        deriv = np.gradient(observable_values, m2_values)
        idx = np.argmax(np.abs(deriv))
    else:
        raise ValueError(f"Unknown method: {method}")

    return float(m2_values[idx])
