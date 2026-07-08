"""Wilson gauge action for compact U(1) in 2D (task A-1).

Frozen convention: S = beta * sum_x [ 1 - cos(theta_P(x)) ] with the
plaquette orientation defined in fields/gauge.py.
"""

from __future__ import annotations

import numpy as np
import torch

from qft_graph.actions.base import Action
from qft_graph.fields.gauge import plaquette_angles
from qft_graph.lattice.hypercubic import HypercubicLattice


class WilsonGaugeAction(Action):
    """S[theta] = beta * sum_x (1 - cos theta_P(x)) on an L x L torus.

    Args:
        lattice: 2D periodic hypercubic lattice.
        beta: Inverse coupling.
    """

    def __init__(self, lattice: HypercubicLattice, beta: float) -> None:
        if lattice.dimension() != 2:
            raise ValueError("WilsonGaugeAction is 2D-only (Phase 2a)")
        self.lattice = lattice
        self.beta = float(beta)
        self._L = lattice.shape[0]

    def _theta_np(self, theta: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(theta, torch.Tensor):
            theta = theta.detach().cpu().numpy()
        return np.asarray(theta, dtype=np.float64).reshape(2, self._L, -1)

    def __call__(self, phi: torch.Tensor) -> torch.Tensor:
        """Total action for link angles of shape (2, L, L)."""
        return self.local_action(phi).sum()

    def local_action(self, phi: torch.Tensor) -> torch.Tensor:
        """Per-plaquette action density, shape (L^2,)."""
        theta_p = plaquette_angles(self._theta_np(phi))
        local = self.beta * (1.0 - np.cos(theta_p))
        return torch.from_numpy(local.reshape(-1))

    def force(self, phi: torch.Tensor) -> torch.Tensor:
        """-dS/dtheta, shape (2, L, L).

        theta_mu(x) enters theta_P(x) with sign s_mu (+1 for mu=1, -1 for
        mu=2) and theta_P(x - e_nu) with -s_mu, so
            dS/dtheta_1(x) = beta * [sin theta_P(x) - sin theta_P(x - e2)]
            dS/dtheta_2(x) = beta * [sin theta_P(x - e1) - sin theta_P(x)]
        """
        sin_p = np.sin(plaquette_angles(self._theta_np(phi)))
        d1 = self.beta * (sin_p - np.roll(sin_p, 1, axis=1))
        d2 = self.beta * (np.roll(sin_p, 1, axis=0) - sin_p)
        return -torch.from_numpy(np.stack([d1, d2]))
