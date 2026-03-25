"""Abstract base class for Monte Carlo samplers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch


@dataclass
class MCResult:
    """Container for Monte Carlo sampling results."""

    configurations: torch.Tensor  # (n_configs, num_sites)
    actions: torch.Tensor  # (n_configs,)
    acceptance_rate: float
    observables: dict[str, torch.Tensor] = field(default_factory=dict)


class MCSampler(ABC):
    """Abstract Monte Carlo sampler interface."""

    @abstractmethod
    def generate(self, n_configs: int) -> MCResult:
        """Generate decorrelated field configurations.

        Args:
            n_configs: Number of independent configurations to produce.

        Returns:
            MCResult containing configurations, actions, and observables.
        """

    @abstractmethod
    def sweep(self, phi: torch.Tensor) -> tuple[torch.Tensor, float]:
        """One full update sweep over all sites.

        Args:
            phi: Current field configuration.

        Returns:
            Tuple of (updated_phi, acceptance_rate).
        """
