"""U(1) heatbath sampler for the 2D Wilson gauge action (task A-1).

The single-link conditional distribution is exactly von Mises: for link
theta_mu(x) with staple sum z = exp(i*a) + exp(-i*b) (the two plaquettes
containing the link contribute cos(theta + a) and cos(theta - b)),

    p(theta) ~ exp(beta * |z| * cos(theta + arg z))
    =>  theta ~ VonMises(mu = -arg(z), kappa = beta * |z|).

No tuning, no accept/reject, low autocorrelation. Links are updated in
vectorized (direction, parity) blocks: for direction mu, all links whose
orthogonal coordinate x_nu has equal parity have disjoint staples, so they
can be drawn simultaneously (exact Gibbs updates on independent
conditionals). The torus-exact plaquette test in tests/test_mc/test_heatbath.py
is the arbiter of the staple sign conventions (plan ground rule 1).
"""

from __future__ import annotations

import logging
import time

import numpy as np
import torch

from qft_graph.actions.wilson import WilsonGaugeAction
from qft_graph.config import MCConfig
from qft_graph.mc.sampler import MCResult, MCSampler

logger = logging.getLogger("qft_graph.mc")


class U1HeatbathSampler(MCSampler):
    """Heatbath sampler for compact U(1) links on a 2D periodic lattice.

    Args:
        action: WilsonGaugeAction providing beta and lattice geometry.
        config: MCConfig (step_size is unused; heatbath has no proposal width).
    """

    def __init__(self, action: WilsonGaugeAction, config: MCConfig) -> None:
        self.action = action
        self.config = config
        self.beta = action.beta
        self._L = action.lattice.shape[0]
        self._L2 = action.lattice.shape[1]
        self._rng = np.random.default_rng(config.seed)

    # ------------------------------------------------------------------
    def _staple_angles(self, theta: np.ndarray, mu: int) -> tuple[np.ndarray, np.ndarray]:
        """Angles of the forward and backward staples for all mu-links.

        Forward (plaquette theta_P at x, contains +theta):
            a(x) = theta_nu(x+e_mu) - theta_mu(x+e_nu) - theta_nu(x)
        Backward (plaquette theta_P at x - e_nu, contains -theta;
        returns -b so that the contribution is cos(theta + (-b))):
            -b(x) = -theta_nu(x+e_mu-e_nu) - theta_mu(x-e_nu) + theta_nu(x-e_nu)
        """
        nu = 1 - mu
        t_mu, t_nu = theta[mu], theta[nu]
        a = (
            np.roll(t_nu, -1, axis=mu)
            - np.roll(t_mu, -1, axis=nu)
            - t_nu
        )
        minus_b = (
            -np.roll(np.roll(t_nu, -1, axis=mu), 1, axis=nu)
            - np.roll(t_mu, 1, axis=nu)
            + np.roll(t_nu, 1, axis=nu)
        )
        return a, minus_b

    def sweep(self, phi: torch.Tensor) -> tuple[torch.Tensor, float]:
        """One heatbath sweep updating all 2*L^2 links.

        Args:
            phi: Link angles, shape (2, L, L) (torch tensor).

        Returns:
            (updated configuration, acceptance rate = 1.0).
        """
        theta = phi.detach().cpu().numpy().astype(np.float64).reshape(2, self._L, self._L2)

        for mu in range(2):
            nu = 1 - mu
            coord_nu = np.arange(self._L2 if nu == 1 else self._L)
            for parity in range(2):
                a, minus_b = self._staple_angles(theta, mu)
                z = np.exp(1j * a) + np.exp(1j * minus_b)
                mean = -np.angle(z)
                kappa = self.beta * np.abs(z)

                # Mask: links whose orthogonal coordinate has this parity
                if nu == 0:
                    mask = (coord_nu % 2 == parity)[:, None] & np.ones(
                        (1, self._L2), dtype=bool
                    )
                else:
                    mask = np.ones((self._L, 1), dtype=bool) & (
                        coord_nu % 2 == parity
                    )[None, :]

                draws = self._rng.vonmises(mean[mask], np.maximum(kappa[mask], 0.0))
                theta[mu][mask] = draws

        return torch.from_numpy(theta).float(), 1.0

    # ------------------------------------------------------------------
    def generate(
        self, n_configs: int, initial_phi: torch.Tensor | None = None
    ) -> MCResult:
        """Generate decorrelated link configurations.

        Returns:
            MCResult with configurations of shape (n_configs, 2, L, L)
            and per-config total actions.
        """
        if initial_phi is None:
            theta = torch.from_numpy(
                self._rng.uniform(-np.pi, np.pi, size=(2, self._L, self._L2))
            ).float()
        else:
            theta = initial_phi.reshape(2, self._L, self._L2).clone()

        t0 = time.time()
        logger.info(
            "U(1) heatbath: thermalizing %d sweeps (beta=%.2f, L=%d)...",
            self.config.n_thermalization, self.beta, self._L,
        )
        for _ in range(self.config.n_thermalization):
            theta, _ = self.sweep(theta)

        configs = torch.empty(n_configs, 2, self._L, self._L2)
        actions = torch.empty(n_configs)
        logger.info("Generating %d configurations...", n_configs)
        for i in range(n_configs):
            for _ in range(self.config.n_sweeps_between):
                theta, _ = self.sweep(theta)
            configs[i] = theta
            actions[i] = self.action(theta).float()
            if (i + 1) % max(1, n_configs // 4) == 0:
                rate = (i + 1) / (time.time() - t0)
                logger.info("  %d/%d configs (%.1f configs/s)", i + 1, n_configs, rate)

        logger.info("Done in %.1fs.", time.time() - t0)
        return MCResult(
            configurations=configs, actions=actions, acceptance_rate=1.0
        )
