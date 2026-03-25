"""Loss functions for training the heterogeneous GNN energy model."""

from __future__ import annotations

import torch
import torch.nn as nn


class EnergyMatchingLoss(nn.Module):
    """MSE loss between predicted and true Euclidean action.

    Trains the GNN energy head to reproduce S_E[phi] computed
    by the exact lattice action for each MC configuration.

    L = (1/N) sum_i (S_GNN(phi_i) - S_MC(phi_i))^2
    """

    def forward(
        self, predicted_energy: torch.Tensor, true_energy: torch.Tensor
    ) -> torch.Tensor:
        return nn.functional.mse_loss(predicted_energy, true_energy)


class KLDivergenceLoss(nn.Module):
    """KL divergence loss between GNN and MC energy distributions.

    Since p_MC(phi) ~ exp(-S_MC[phi]) and p_GNN(phi) ~ exp(-S_GNN[phi]),
    the KL divergence reduces to:

    KL(p_MC || p_GNN) = E_MC[S_GNN - S_MC] + log(Z_GNN/Z_MC)

    Since Z terms are constants w.r.t. model parameters, minimizing
    E_MC[S_GNN] is equivalent. We add a variance penalty for stability.
    """

    def __init__(self, variance_weight: float = 0.01) -> None:
        super().__init__()
        self.variance_weight = variance_weight

    def forward(
        self, predicted_energy: torch.Tensor, true_energy: torch.Tensor
    ) -> torch.Tensor:
        # Primary: minimize expected predicted energy over MC samples
        diff = predicted_energy - true_energy
        primary = diff.mean()

        # Variance regularizer for training stability
        variance_penalty = self.variance_weight * diff.var()

        return primary + variance_penalty


class RelativeEnergyLoss(nn.Module):
    """Relative energy loss: focus on energy differences rather than absolute values.

    L = (1/N) sum_i ((S_GNN(phi_i) - S_GNN_mean) - (S_MC(phi_i) - S_MC_mean))^2

    This is useful because the absolute normalization of the action is
    irrelevant for the Boltzmann distribution — only relative energies
    between configurations matter for sampling.
    """

    def forward(
        self, predicted_energy: torch.Tensor, true_energy: torch.Tensor
    ) -> torch.Tensor:
        pred_centered = predicted_energy - predicted_energy.mean()
        true_centered = true_energy - true_energy.mean()
        return nn.functional.mse_loss(pred_centered, true_centered)
