"""Abstract base class for lattice action functionals."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class Action(ABC):
    """Abstract Euclidean action functional S_E[phi].

    Computes the lattice-discretized action for a given field configuration.
    Used both inside MC samplers (for acceptance probability) and as
    ground-truth training targets for the GNN energy head.
    """

    @abstractmethod
    def __call__(self, phi: torch.Tensor) -> torch.Tensor:
        """Compute total Euclidean action.

        Args:
            phi: Field configuration, shape depends on theory.

        Returns:
            Scalar tensor S_E[phi].
        """

    @abstractmethod
    def local_action(self, phi: torch.Tensor) -> torch.Tensor:
        """Compute action density at each site.

        Args:
            phi: Field configuration.

        Returns:
            Per-site action tensor of shape (num_sites,).
        """

    @abstractmethod
    def force(self, phi: torch.Tensor) -> torch.Tensor:
        """Compute -dS/dphi for HMC molecular dynamics.

        Args:
            phi: Field configuration.

        Returns:
            Force tensor, same shape as phi.
        """
