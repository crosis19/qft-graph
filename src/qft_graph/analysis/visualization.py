"""Plotting utilities for QFT observables and analysis results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_phase_diagram(
    m2_values: np.ndarray,
    magnetization: np.ndarray,
    magnetization_err: np.ndarray | None = None,
    susceptibility: np.ndarray | None = None,
    title: str = "Phase Diagram",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot order parameter and susceptibility vs coupling."""
    fig, axes = plt.subplots(1, 2 if susceptibility is not None else 1, figsize=(12, 5))
    if susceptibility is None:
        axes = [axes]

    # Magnetization
    ax = axes[0]
    if magnetization_err is not None:
        ax.errorbar(m2_values, magnetization, yerr=magnetization_err, fmt="o-", capsize=3)
    else:
        ax.plot(m2_values, magnetization, "o-")
    ax.set_xlabel(r"$m^2$")
    ax.set_ylabel(r"$|\langle\phi\rangle|$")
    ax.set_title("Order Parameter")
    ax.grid(True, alpha=0.3)

    # Susceptibility
    if susceptibility is not None:
        ax = axes[1]
        ax.plot(m2_values, susceptibility, "s-", color="tab:orange")
        ax.set_xlabel(r"$m^2$")
        ax.set_ylabel(r"$\chi$")
        ax.set_title("Susceptibility")
        ax.grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_correlation_function(
    G_r: np.ndarray,
    lattice_spacing: float = 1.0,
    fit_xi: float | None = None,
    title: str = "Two-Point Function",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot two-point correlation function G(r) on log scale."""
    fig, ax = plt.subplots(figsize=(8, 5))

    r = np.arange(len(G_r)) * lattice_spacing
    ax.semilogy(r, np.abs(G_r), "o-", label=r"$G(r)$")

    if fit_xi is not None and fit_xi > 0:
        r_fit = np.linspace(r[1], r[-1], 100)
        ax.semilogy(r_fit, G_r[1] * np.exp(-r_fit / fit_xi), "--",
                     label=rf"$\xi = {fit_xi:.2f}$", color="tab:red")

    ax.set_xlabel(r"$r / a$")
    ax.set_ylabel(r"$|G(r)|$")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_scaling_collapse(
    L_values: list[int],
    xi_over_L: dict[int, list[tuple[float, float]]],
    m2_c: float,
    nu: float,
    title: str = "Finite-Size Scaling Collapse",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot xi/L vs (m2 - m2_c) * L^{1/nu} for scaling collapse."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.viridis(np.linspace(0, 1, len(L_values)))

    # Raw xi/L vs m2
    ax = axes[0]
    for L, color in zip(sorted(L_values), colors):
        data = sorted(xi_over_L[L], key=lambda t: t[0])
        m2_vals, xi_L_vals = zip(*data)
        ax.plot(m2_vals, xi_L_vals, "o-", color=color, label=f"L={L}")
    ax.axvline(m2_c, color="gray", linestyle="--", alpha=0.5, label=rf"$m^2_c = {m2_c:.3f}$")
    ax.set_xlabel(r"$m^2$")
    ax.set_ylabel(r"$\xi / L$")
    ax.set_title(r"$\xi/L$ crossing")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Collapsed data
    ax = axes[1]
    for L, color in zip(sorted(L_values), colors):
        data = sorted(xi_over_L[L], key=lambda t: t[0])
        for m2, xi_L_val in data:
            x = (m2 - m2_c) * L ** (1.0 / nu)
            ax.scatter(x, xi_L_val, color=color, s=30, label=f"L={L}" if m2 == data[0][0] else "")
    ax.set_xlabel(rf"$(m^2 - m^2_c) \cdot L^{{1/\nu}}$, $\nu = {nu:.3f}$")
    ax.set_ylabel(r"$\xi / L$")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_training_curves(
    history: dict[str, list[float]],
    title: str = "Training Progress",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot training loss and validation metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    ax = axes[0]
    ax.semilogy(epochs, history["train_loss"], label="Train")
    if "val_loss" in history:
        ax.semilogy(epochs, history["val_loss"], label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Correlation
    ax = axes[1]
    if "val_corr" in history:
        ax.plot(epochs, history["val_corr"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Correlation")
    ax.set_title("Energy Prediction Correlation")
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
