"""Exact results for compact U(1) gauge theory in 2D (validation oracles, A-1).

Character expansion on the L x L torus: the partition function factorizes
over plaquette representations n,

    Z = sum_n I_n(beta)^(L^2)

and the mean plaquette is

    <cos theta_P> = sum_n I_n(beta)^(L^2 - 1) * (I_{n-1}(beta) + I_{n+1}(beta))/2 / Z.

All sums are evaluated with the ratios r_n = I_n(beta)/I_0(beta) to avoid
overflow/underflow at large L^2. Sanity anchor: the infinite-volume limit is
I_1(beta)/I_0(beta) ~ 0.4464 at beta = 1.

Wilson loops (infinite volume, valid for R*T << L^2):

    <W(R,T)> = (I_1/I_0)^(R*T),   sigma(beta) = -ln(I_1/I_0).
"""

from __future__ import annotations

import numpy as np
from scipy.special import iv


def _bessel_ratios(beta: float, n_max: int) -> np.ndarray:
    """r_n = I_n(beta)/I_0(beta) for n = 0..n_max+1."""
    n = np.arange(0, n_max + 2)
    return iv(n, beta) / iv(0, beta)


def exact_mean_plaquette(beta: float, L: int, n_max: int = 20) -> float:
    """Torus-exact <cos theta_P> via character expansion.

    Args:
        beta: Inverse coupling.
        L: Linear lattice size (L x L torus).
        n_max: Representation cutoff; terms decay super-exponentially.

    Returns:
        <cos theta_P> exactly (up to the n_max truncation, ~1e-16 for
        beta <= 5 and n_max = 20).
    """
    V = L * L
    r = _bessel_ratios(beta, n_max)  # r[0..n_max+1]

    # Sum over n = -n_max..n_max using I_{-n} = I_n
    z = 0.0
    num = 0.0
    for n in range(-n_max, n_max + 1):
        rn = r[abs(n)]
        rm = r[abs(n - 1)]
        rp = r[abs(n + 1)]
        z += rn**V
        num += rn ** (V - 1) * 0.5 * (rm + rp)
    return float(num / z)


def exact_wilson_loop_infinite_volume(beta: float, R: int, T: int) -> float:
    """<W(R,T)> = (I_1/I_0)^(R*T), valid for R*T << L^2."""
    ratio = iv(1, beta) / iv(0, beta)
    return float(ratio ** (R * T))


def string_tension(beta: float) -> float:
    """sigma(beta) = -ln(I_1(beta)/I_0(beta))."""
    return float(-np.log(iv(1, beta) / iv(0, beta)))
