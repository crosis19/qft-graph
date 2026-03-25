"""Metropolis-Hastings sampler for scalar phi^4 theory."""

from __future__ import annotations

import logging

import numpy as np
import torch

from qft_graph.actions.phi4 import Phi4Action
from qft_graph.config import MCConfig
from qft_graph.mc.sampler import MCResult, MCSampler

logger = logging.getLogger("qft_graph.mc")


class MetropolisSampler(MCSampler):
    """Metropolis-Hastings sampler with single-site Gaussian proposals.

    Generates field configurations distributed as exp(-S_E[phi]) for
    scalar phi^4 theory. Single-site updates with Gaussian proposals
    satisfy detailed balance by construction.

    Args:
        action: Phi4Action instance defining the theory.
        config: MCConfig with sampling parameters.
    """

    def __init__(self, action: Phi4Action, config: MCConfig) -> None:
        self.action = action
        self.config = config
        self.rng = np.random.RandomState(config.seed)
        self._nsites = action.lattice.num_sites()

    def sweep(self, phi: torch.Tensor) -> tuple[torch.Tensor, float]:
        """One full Metropolis sweep: visit every site once in random order.

        For each site, propose phi(x) -> phi(x) + eta where eta ~ N(0, step_size^2).
        Accept with probability min(1, exp(-Delta S)).

        Args:
            phi: Current configuration of shape (num_sites,).

        Returns:
            (updated_phi, acceptance_rate).
        """
        phi = phi.clone()
        n_accepted = 0
        order = self.rng.permutation(self._nsites)

        for site in order:
            # Gaussian proposal
            eta = self.rng.normal(0, self.config.step_size)
            new_value = phi[site].item() + eta

            # Compute action change
            delta_s = self.action.delta_action(phi, int(site), new_value)

            # Metropolis acceptance
            if delta_s.item() <= 0 or self.rng.random() < np.exp(-delta_s.item()):
                phi[site] = new_value
                n_accepted += 1

        return phi, n_accepted / self._nsites

    def generate(self, n_configs: int) -> MCResult:
        """Generate decorrelated field configurations.

        Pipeline:
        1. Initialize from hot/cold start
        2. Thermalize for n_thermalization sweeps
        3. Generate n_configs configurations separated by n_sweeps_between sweeps

        Args:
            n_configs: Number of configurations to generate.

        Returns:
            MCResult with configurations and actions.
        """
        # Initialize
        phi = 2.0 * torch.rand(self._nsites) - 1.0  # hot start

        # Thermalization
        logger.info(
            "Thermalizing for %d sweeps...", self.config.n_thermalization
        )
        therm_acceptance = []
        for _ in range(self.config.n_thermalization):
            phi, acc = self.sweep(phi)
            therm_acceptance.append(acc)

        avg_therm_acc = np.mean(therm_acceptance)
        logger.info("Thermalization done. Avg acceptance: %.3f", avg_therm_acc)

        # Generation
        configurations = torch.zeros(n_configs, self._nsites)
        actions = torch.zeros(n_configs)
        gen_acceptance = []

        logger.info("Generating %d configurations...", n_configs)
        for i in range(n_configs):
            for _ in range(self.config.n_sweeps_between):
                phi, acc = self.sweep(phi)
                gen_acceptance.append(acc)

            configurations[i] = phi.clone()
            with torch.no_grad():
                actions[i] = self.action(phi)

        avg_gen_acc = np.mean(gen_acceptance)
        logger.info("Generation done. Avg acceptance: %.3f", avg_gen_acc)

        return MCResult(
            configurations=configurations,
            actions=actions,
            acceptance_rate=avg_gen_acc,
        )
