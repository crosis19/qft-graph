"""Statistical analysis tools for Monte Carlo data."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch


def jackknife_mean_error(data: torch.Tensor | np.ndarray) -> tuple[float, float]:
    """Compute mean and jackknife error estimate.

    WARNING: assumes statistically independent samples. On autocorrelated
    MC series this underestimates the error by ~sqrt(2*tau_int); use
    binned_jackknife with bin size >= 2*tau_int instead.

    Args:
        data: 1D array of measurements.

    Returns:
        (mean, error) tuple.
    """
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    data = np.asarray(data, dtype=np.float64)
    n = len(data)
    mean = data.mean()

    # Jackknife resampling
    jk_means = np.empty(n)
    for i in range(n):
        jk_means[i] = np.delete(data, i).mean()

    jk_var = (n - 1) / n * np.sum((jk_means - jk_means.mean()) ** 2)
    return float(mean), float(np.sqrt(jk_var))


def bootstrap_mean_error(
    data: torch.Tensor | np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute mean and bootstrap error estimate.

    Args:
        data: 1D array of measurements.
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed for reproducibility.

    Returns:
        (mean, error) tuple.
    """
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    data = np.asarray(data, dtype=np.float64)
    rng = np.random.RandomState(seed)

    n = len(data)
    mean = data.mean()

    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = data[rng.randint(0, n, size=n)]
        boot_means[i] = sample.mean()

    return float(mean), float(boot_means.std())


def binned_jackknife(
    samples: torch.Tensor | np.ndarray,
    estimator: Callable[[torch.Tensor | np.ndarray], float],
    n_bins: int = 20,
) -> tuple[float, float]:
    """Binned (blocked) jackknife for arbitrary estimators on MC series.

    Consecutive samples are grouped into n_bins blocks and the estimator
    is recomputed on each leave-one-block-out subset. With bin size
    >= 2*tau_int the blocks are effectively independent, giving correct
    errors for autocorrelated data (unlike the unbinned leave-one-out
    jackknife, which underestimates them).

    Generalizes the block pattern of
    ObservableSet.correlation_length_fft_jackknife to any estimator.

    Args:
        samples: (n_samples, ...) array/tensor in MC-time order; the
            estimator receives subsets with the same type and trailing shape.
        estimator: Callable mapping a samples subset to a scalar.
        n_bins: Number of jackknife blocks; choose so that
            n_samples/n_bins >= 2*tau_int.

    Returns:
        (value, error): estimator on the full (trimmed) sample and the
        jackknife standard error.
    """
    n = len(samples)
    if n_bins < 2 or n < 2 * n_bins:
        raise ValueError(f"Need n_bins >= 2 and n >= 2*n_bins, got n={n}, n_bins={n_bins}")
    block = n // n_bins
    n_use = block * n_bins
    trimmed = samples[:n_use]

    value = float(estimator(trimmed))

    if isinstance(trimmed, torch.Tensor):
        full_mask = torch.ones(n_use, dtype=torch.bool)
    else:
        full_mask = np.ones(n_use, dtype=bool)

    theta = np.empty(n_bins)
    for b in range(n_bins):
        mask = full_mask.clone() if isinstance(trimmed, torch.Tensor) else full_mask.copy()
        mask[b * block : (b + 1) * block] = False
        theta[b] = float(estimator(trimmed[mask]))

    error = np.sqrt((n_bins - 1) / n_bins * np.sum((theta - theta.mean()) ** 2))
    return value, float(error)


def _normalized_autocorrelation(x: np.ndarray) -> np.ndarray:
    """rho(t) for t = 0..n-1 via FFT, with 1/(n-t) normalization."""
    n = len(x)
    x = x - x.mean()
    m = 1 << (2 * n - 1).bit_length()  # zero-pad to avoid circular wrap
    f = np.fft.rfft(x, m)
    acov = np.fft.irfft(f * np.conj(f), m)[:n]
    acov /= np.arange(n, 0, -1)
    if acov[0] == 0:
        return np.zeros(n)
    return acov / acov[0]


def integrated_autocorrelation_time(
    data: torch.Tensor | np.ndarray, s_param: float = 1.5
) -> float:
    """Integrated autocorrelation time via Wolff automatic windowing.

    See integrated_autocorrelation_time_wolff for details; this wrapper
    keeps the original float-returning signature. The previous
    first-negative-crossing estimator is preserved as
    integrated_autocorrelation_time_naive for comparison.
    """
    tau, _, _ = integrated_autocorrelation_time_wolff(data, s_param)
    return tau


def integrated_autocorrelation_time_wolff(
    data: torch.Tensor | np.ndarray, s_param: float = 1.5
) -> tuple[float, float, int]:
    """Estimate tau_int with Wolff's automatic windowing procedure.

    tau_int(W) = 1/2 + sum_{t=1}^{W} rho(t), with the window W chosen at
    the first sign change of g(W) = exp(-W/tau) - tau/sqrt(W*N), where
    tau = S * tau_int(W) estimates the exponential autocorrelation time
    (U. Wolff, Comput. Phys. Commun. 156 (2004) 143, Sec. 3.3).

    Note: the plan's shorthand criterion "W >= S*tau_int(W)" truncates the
    window too early (~30% low on an AR(1) series with tau_int = 4.5) and
    fails the plan's own 10%-recovery test; the full Wolff g-criterion
    above is used instead, per ground rule 1 (tests arbitrate).

    Args:
        data: 1D MC-time-ordered series.
        s_param: Wolff S parameter (frozen at 1.5 in the plan).

    Returns:
        (tau_int, error, window) — error from
        var(tau_int) ~ (4/N)(W + 1/2 - tau_int) * tau_int^2.
    """
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    data = np.asarray(data, dtype=np.float64)
    n = len(data)
    if n < 4 or data.var() == 0:
        return 0.0, 0.0, 0

    rho = _normalized_autocorrelation(data)
    t_max = n // 2

    tau_int = 0.5
    window = t_max - 1
    for w in range(1, t_max):
        tau_int += rho[w]
        if tau_int <= 0.5:
            # Anticorrelated / noise-dominated: effectively uncorrelated
            tau_exp = 1e-8
        else:
            tau_exp = s_param * tau_int
        g = np.exp(-w / tau_exp) - tau_exp / np.sqrt(w * n)
        if g < 0:
            window = w
            break

    tau_int = max(tau_int, 0.5)
    error = tau_int * 2.0 * np.sqrt(max(window + 0.5 - tau_int, 0.0) / n)
    return float(tau_int), float(error), int(window)


def integrated_autocorrelation_time_naive(data: torch.Tensor | np.ndarray) -> float:
    """Original estimator: sum rho(t) until the first negative value.

    Kept for comparison with the Wolff automatic-windowing estimator.
    Truncating at the first negative rho(t) is noise-sensitive and has no
    principled bias/variance trade-off.

    Args:
        data: 1D timeseries of measurements.

    Returns:
        Estimated tau_int.
    """
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    data = np.asarray(data, dtype=np.float64)
    n = len(data)
    mean = data.mean()
    var = data.var()

    if var == 0:
        return 0.0

    # Normalized autocorrelation
    tau_int = 0.5  # C(0)/2 contribution
    for t in range(1, n // 2):
        c_t = np.mean((data[: n - t] - mean) * (data[t:] - mean)) / var
        if c_t < 0:
            break
        tau_int += c_t

    return float(tau_int)
