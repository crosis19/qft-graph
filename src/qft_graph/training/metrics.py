"""Observable-based metrics for monitoring training quality."""

from __future__ import annotations

import torch


def energy_correlation(predicted: torch.Tensor, true: torch.Tensor) -> float:
    """Pearson correlation between predicted and true energies.

    Perfect model: r = 1.0. Random: r ~ 0.
    """
    pred_c = predicted - predicted.mean()
    true_c = true - true.mean()
    num = (pred_c * true_c).sum()
    denom = torch.sqrt((pred_c**2).sum() * (true_c**2).sum())
    if denom == 0:
        return 0.0
    return (num / denom).item()


def relative_error(predicted: torch.Tensor, true: torch.Tensor) -> float:
    """Mean absolute relative error: |S_pred - S_true| / |S_true|."""
    return (torch.abs(predicted - true) / torch.abs(true).clamp(min=1e-8)).mean().item()


def energy_std_ratio(predicted: torch.Tensor, true: torch.Tensor) -> float:
    """Ratio of predicted to true energy standard deviations.

    Should be ~1.0 for a well-calibrated model. < 1 means the model
    is under-estimating fluctuations (too smooth).
    """
    pred_std = predicted.std()
    true_std = true.std()
    if true_std == 0:
        return float("inf")
    return (pred_std / true_std).item()
