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


def gauge_invariance_error(
    orbit_preds: torch.Tensor, config_preds: torch.Tensor
) -> dict[str, float]:
    """Per-target gauge-invariance error eps_gauge (plan §3, task A-5).

        eps_gauge = mean_configs[ std_over_gauge_copies(y_hat) ]
                    / std_over_configs(y_hat)

    A gauge-invariant predictor has eps_gauge = 0 (Variant C reaches this
    exactly: its inputs are bit-identical across gauge copies); a predictor
    whose output tracks the gauge representative as much as the physics has
    eps_gauge ~ 1. Both stds are sample stds (ddof=1), computed in float64.
    The ratio is invariant under affine rescaling of y_hat, so predictions
    can be fed on the model's standardized output scale.

    Args:
        orbit_preds: (n_configs, n_copies) predictions on gauge copies.
        config_preds: (n_configs,) predictions on the original configs
            (the config-to-config spread in the denominator).

    Returns:
        {"eps_gauge", "mean_orbit_std", "config_std"}. A degenerate
        denominator (constant predictor across configs) yields
        eps_gauge = 0.0 when the orbit spread is also zero (a constant IS
        invariant) and inf otherwise; the components disambiguate.
    """
    if orbit_preds.ndim != 2:
        raise ValueError(
            f"orbit_preds must be (n_configs, n_copies), got {tuple(orbit_preds.shape)}"
        )
    n_configs, n_copies = orbit_preds.shape
    if config_preds.shape != (n_configs,):
        raise ValueError(
            f"config_preds shape {tuple(config_preds.shape)} does not match "
            f"orbit_preds n_configs={n_configs}"
        )
    if n_configs < 2 or n_copies < 2:
        raise ValueError(
            f"Need >=2 configs and >=2 gauge copies for sample stds, got "
            f"({n_configs}, {n_copies})"
        )
    mean_orbit_std = orbit_preds.double().std(dim=1).mean().item()
    config_std = config_preds.double().std().item()
    if config_std == 0.0:
        eps = 0.0 if mean_orbit_std == 0.0 else float("inf")
    else:
        eps = mean_orbit_std / config_std
    return {
        "eps_gauge": eps,
        "mean_orbit_std": mean_orbit_std,
        "config_std": config_std,
    }


def q_rounded_accuracy(predicted: torch.Tensor, true: torch.Tensor) -> float:
    """Exact-integer accuracy for topological charge regression (task A-4).

    Q is an exact integer on the torus (A-1 oracle 3), so the honest
    metric is fraction of configs where round(prediction) hits the true
    integer. True values are rounded too — they are stored as floats
    within 1e-10 of integers.
    """
    if predicted.shape != true.shape:
        # A (B,1)-vs-(B,) mismatch would silently broadcast the == to a
        # (B,B) matrix and return a plausible-but-wrong accuracy.
        raise ValueError(
            f"Shape mismatch: predicted {tuple(predicted.shape)} vs "
            f"true {tuple(true.shape)}"
        )
    if predicted.numel() == 0:
        raise ValueError("Empty tensors — accuracy undefined")
    return (torch.round(predicted) == torch.round(true)).float().mean().item()
