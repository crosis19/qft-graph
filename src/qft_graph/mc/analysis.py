"""Statistical analysis tools for Monte Carlo data."""

from __future__ import annotations

import numpy as np
import torch


def jackknife_mean_error(data: torch.Tensor | np.ndarray) -> tuple[float, float]:
    """Compute mean and jackknife error estimate.

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


def integrated_autocorrelation_time(data: torch.Tensor | np.ndarray) -> float:
    """Estimate integrated autocorrelation time tau_int.

    Uses the automatic windowing procedure: sum the normalized
    autocorrelation function until it drops below a threshold.

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
