"""Compact U(1) gauge link field for Phase 2a (quenched 2D lattice gauge theory).

Frozen conventions (implementation plan §3.1):
  - Links theta[mu, x] in (-pi, pi], U = exp(i*theta), mu in {1, 2},
    periodic BCs on an L x L lattice. Array layout: theta.shape == (2, L, L),
    theta[0] = theta_1 (direction e1 = axis 0), theta[1] = theta_2 (e2 = axis 1).
  - Plaquette (single orientation in 2D):
        theta_P(x) = theta_1(x) + theta_2(x+e1) - theta_1(x+e2) - theta_2(x)
  - Gauge transform: theta_mu(x) -> theta_mu(x) + alpha(x) - alpha(x + e_mu).
  - Topological charge: Q = (1/2pi) * sum_x wrap(theta_P(x)), wrap to (-pi, pi];
    exact integer on the torus.
  - Wilson loop W(R,T): cos of the oriented sum of link angles around an
    R x T rectangle.

All array functions operate on numpy float64 arrays of shape (2, L, L);
torch conversion happens at API boundaries (same pattern as mc/metropolis.py).
SU(3) links (Phase 3) will live in a separate class following this template.
"""

from __future__ import annotations

import numpy as np
import torch

from qft_graph.fields.base import Field


def wrap_angle(x: np.ndarray) -> np.ndarray:
    """Map angles to (-pi, pi]."""
    w = np.mod(x + np.pi, 2.0 * np.pi) - np.pi  # [-pi, pi)
    return np.where(w == -np.pi, np.pi, w)


def plaquette_angles(theta: np.ndarray) -> np.ndarray:
    """theta_P(x) for every site, shape (L, L). Frozen orientation."""
    t1, t2 = theta[0], theta[1]
    return (
        t1
        + np.roll(t2, -1, axis=0)  # theta_2(x + e1)
        - np.roll(t1, -1, axis=1)  # theta_1(x + e2)
        - t2
    )


def topological_charge(theta: np.ndarray) -> float:
    """Q = (1/2pi) sum_x wrap(theta_P(x)); exact integer on the torus."""
    return float(wrap_angle(plaquette_angles(theta)).sum() / (2.0 * np.pi))


def gauge_transform(theta: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Apply theta_mu(x) -> theta_mu(x) + alpha(x) - alpha(x + e_mu).

    Args:
        theta: Link angles (2, L, L).
        alpha: Site angles (L, L).

    Returns:
        Transformed link angles, wrapped to (-pi, pi].
    """
    out = np.empty_like(theta)
    for mu in range(2):
        out[mu] = theta[mu] + alpha - np.roll(alpha, -1, axis=mu)
    return wrap_angle(out)


def wilson_loop_angles(theta: np.ndarray, R: int, T: int) -> np.ndarray:
    """Oriented angle sum around R x T rectangles based at every site.

    loop(x) = sum_{i<R} theta_1(x + i e1)
            + sum_{j<T} theta_2(x + R e1 + j e2)
            - sum_{i<R} theta_1(x + i e1 + T e2)
            - sum_{j<T} theta_2(x + j e2)

    Returns:
        (L, L) array of loop angles.
    """
    t1, t2 = theta[0], theta[1]
    s1 = sum(np.roll(t1, -i, axis=0) for i in range(R))  # bottom edge
    s2 = sum(np.roll(t2, -j, axis=1) for j in range(T))  # left edge
    return (
        s1
        + np.roll(s2, -R, axis=0)   # right edge, shifted to x + R e1
        - np.roll(s1, -T, axis=1)   # top edge, shifted to x + T e2
        - s2
    )


def wilson_loop(theta: np.ndarray, R: int, T: int) -> float:
    """<W(R,T)> averaged over all base points of a single configuration."""
    return float(np.cos(wilson_loop_angles(theta, R, T)).mean())


def random_gauge_transform(
    theta: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Gauge transform with iid alpha(x) ~ Uniform(-pi, pi)."""
    alpha = rng.uniform(-np.pi, np.pi, size=theta.shape[1:])
    return gauge_transform(theta, alpha)


class U1GaugeField(Field):
    """U(1) link variables as graph field content.

    Each of the 2*L^2 links becomes one 'gauge' node with features
    [cos theta, sin theta, onehot(mu)] (beta is appended at graph build
    time, task A-3). Link ordering: mu-major, then row-major sites —
    node index = mu * L^2 + x1 * L + x2.
    """

    def __init__(self, L: int | tuple[int, int] | None = None) -> None:
        """L: side length of a square lattice, or (L1, L2) for rectangular."""
        self.L = L

    def dof_per_site(self) -> int:
        return 1  # one angle per link node

    def node_feature_dim(self) -> int:
        return 4  # [cos, sin, onehot_mu1, onehot_mu2]

    def node_type_name(self) -> str:
        return "gauge"

    def node_features(self, configuration: torch.Tensor) -> torch.Tensor:
        """Convert link angles (2, L, L) to node features (2*L^2, 4)."""
        theta = configuration.reshape(2, -1)
        n_links = theta.shape[1]
        cos_sin = torch.stack([theta.cos(), theta.sin()], dim=-1)  # (2, n, 2)
        onehot = torch.eye(2, dtype=theta.dtype).unsqueeze(1).expand(2, n_links, 2)
        return torch.cat([cos_sin, onehot], dim=-1).reshape(2 * n_links, 4)

    def initialize(self, num_sites: int, mode: str = "hot") -> torch.Tensor:
        """Initial link configuration of shape (2, L1, L2).

        num_sites must equal L1*L2 (square assumed unless L was given
        as a tuple at construction).
        """
        if self.L is None:
            side = int(round(num_sites**0.5))
            shape = (side, side)
        elif isinstance(self.L, int):
            shape = (self.L, self.L)
        else:
            shape = tuple(self.L)
        if mode == "hot":
            theta = np.random.uniform(-np.pi, np.pi, size=(2, *shape))
        elif mode == "cold":
            theta = np.zeros((2, *shape))
        else:
            raise ValueError(f"Unknown initialization mode: {mode}")
        return torch.from_numpy(theta).float()
