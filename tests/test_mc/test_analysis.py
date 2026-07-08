"""Tests for MC statistical analysis: binned jackknife and Wolff tau_int.

Validation strategy (plan task P1-1): synthetic AR(1) series
    x_{t+1} = a * x_t + eps_t,  eps_t ~ N(0, 1)
has exactly known statistics:
    rho(t)   = a^t
    tau_int  = (1 + a) / (2 (1 - a))
    Var(x)   = 1 / (1 - a^2)
    SE(mean) = sqrt(Var(x) * 2 * tau_int / N)
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.signal import lfilter

from qft_graph.mc.analysis import (
    binned_jackknife,
    integrated_autocorrelation_time,
    integrated_autocorrelation_time_naive,
    integrated_autocorrelation_time_wolff,
    jackknife_mean_error,
)

AR1_A = 0.8
TAU_TRUE = 0.5 * (1 + AR1_A) / (1 - AR1_A)  # = 4.5
VAR_TRUE = 1.0 / (1 - AR1_A**2)


def ar1_series(n: int, a: float = AR1_A, seed: int = 0) -> np.ndarray:
    """Stationary AR(1) series with unit innovation variance."""
    rng = np.random.default_rng(seed)
    burn = 1000
    eps = rng.normal(0.0, 1.0, n + burn)
    x = lfilter([1.0], [1.0, -a], eps)
    return x[burn:]


class TestWolffTauInt:
    @pytest.mark.parametrize("seed", [0, 1, 2, 42])
    def test_ar1_recovery_within_10_percent(self, seed):
        x = ar1_series(100_000, seed=seed)
        tau, err, window = integrated_autocorrelation_time_wolff(x)
        assert abs(tau - TAU_TRUE) / TAU_TRUE < 0.10
        # Window must comfortably cover the exponential tail
        assert window > TAU_TRUE
        assert err > 0

    def test_white_noise_is_uncorrelated(self):
        x = np.random.default_rng(7).normal(size=50_000)
        tau, _, _ = integrated_autocorrelation_time_wolff(x)
        assert 0.4 < tau < 0.7

    def test_wrapper_matches_full_version(self):
        x = ar1_series(20_000, seed=3)
        tau_full, _, _ = integrated_autocorrelation_time_wolff(x)
        assert integrated_autocorrelation_time(x) == tau_full

    def test_torch_input(self):
        x = torch.from_numpy(ar1_series(20_000, seed=4))
        tau, _, _ = integrated_autocorrelation_time_wolff(x)
        assert abs(tau - TAU_TRUE) / TAU_TRUE < 0.25  # looser: short series

    def test_constant_series(self):
        tau, err, window = integrated_autocorrelation_time_wolff(np.ones(1000))
        assert tau == 0.0 and err == 0.0

    def test_naive_estimator_preserved(self):
        x = ar1_series(20_000, seed=0)
        tau_naive = integrated_autocorrelation_time_naive(x)
        # Same ballpark on AR(1); exact agreement not expected
        assert 0.5 < tau_naive < 3 * TAU_TRUE


class TestBinnedJackknife:
    @pytest.mark.parametrize("seed", [0, 42])
    def test_ar1_standard_error_within_10_percent(self, seed):
        n = 100_000
        x = ar1_series(n, seed=seed)
        se_true = np.sqrt(VAR_TRUE * 2 * TAU_TRUE / n)
        value, err = binned_jackknife(x, lambda s: float(np.mean(s)), n_bins=100)
        assert abs(err - se_true) / se_true < 0.10
        assert abs(value - np.mean(x[: (n // 100) * 100])) < 1e-12

    def test_unbinned_jackknife_undershoots_on_ar1(self):
        n = 10_000
        x = ar1_series(n, seed=0)
        se_true = np.sqrt(VAR_TRUE * 2 * TAU_TRUE / n)
        _, err_unbinned = jackknife_mean_error(x)
        _, err_binned = binned_jackknife(x, lambda s: float(np.mean(s)), n_bins=50)
        # Unbinned ignores autocorrelation: low by ~1/sqrt(2*tau_int) ~ 3x
        assert err_unbinned < 0.5 * se_true
        # Binned recovers it (looser tolerance: only 50 bins)
        assert abs(err_binned - se_true) / se_true < 0.25

    def test_iid_matches_analytic(self):
        n = 50_000
        x = np.random.default_rng(11).normal(0.0, 2.0, n)
        se_true = 2.0 / np.sqrt(n)
        _, err = binned_jackknife(x, lambda s: float(np.mean(s)), n_bins=50)
        assert abs(err - se_true) / se_true < 0.20

    def test_torch_tensor_and_2d_samples(self):
        # Estimator over (n_configs, n_sites) tensors, e.g. susceptibility
        torch.manual_seed(0)
        configs = torch.randn(2000, 16)

        def chi(subset: torch.Tensor) -> float:
            m = subset.mean(dim=1)
            return 16 * (m.pow(2).mean() - m.abs().mean().pow(2)).item()

        value, err = binned_jackknife(configs, chi, n_bins=20)
        assert np.isfinite(value) and err > 0
        # Type preserved: estimator received torch subsets without error

    def test_invalid_args_raise(self):
        x = np.arange(10, dtype=float)
        with pytest.raises(ValueError):
            binned_jackknife(x, np.mean, n_bins=1)
        with pytest.raises(ValueError):
            binned_jackknife(x, np.mean, n_bins=8)
