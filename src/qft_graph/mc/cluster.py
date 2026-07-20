"""Wolff / Brower-Tamayo embedded-cluster sampler for scalar phi^4 theory.

Single-cluster (Wolff) sign flips decorrelate the Z2 (sign) sector without
critical slowing down (dynamic exponent z ~ 0.25), interleaved with local
Metropolis sweeps (Brower-Tamayo hybrid) that update the field magnitudes
|phi| -- which cluster flips leave invariant. This removes the critical-
slowing-down bias that inflates the local-Metropolis autocorrelation
(tau_int ~ 35-100 at the susceptibility peaks) and lets the finite-size
scaling reach large lattices cheaply (task V2-1 / V2-2).

Physics / bond probability
--------------------------
The lattice action (``actions/phi4.py``, a=1 lattice units, 2D) is

    S = sum_x [ 1/2 sum_mu (phi_{x+mu} - phi_x)^2
                + 1/2 m^2 phi_x^2 + lam phi_x^4 ].

The kinetic term expands to a ferromagnetic nearest-neighbour coupling

    S  ⊃  - K sum_<ij> phi_i phi_j ,     K = a^{d-2}   (= 1 in 2D),

plus on-site 1/2 m^2 phi^2 + lam phi^4 pieces that are invariant under the
Z2 reflection phi -> -phi. Reflecting a connected cluster C therefore changes
only the bonds crossing the boundary partial-C, each by

    Delta S_bond = 2 K phi_i phi_j .

The embedded-Wolff single-cluster move builds C by adding a same-sign
neighbour j (phi_i phi_j > 0) of an in-cluster site i with probability

    p_ij = 1 - exp(-2 K phi_i phi_j)          (0 for opposite-sign neighbours),

then flips phi -> -phi on all of C. It is rejection-free by construction and
preserves every |phi_x|. Because the on-site part is Z2-invariant, the flip
leaves S unchanged up to the boundary bonds already accounted for in p_ij, so
the move satisfies detailed balance for exp(-S).

The exact prefactor (the factor ``2 K``) is *validated*, not assumed: the
cross-check against the Metropolis sampler in
``tests/test_mc/test_cluster.py`` is the arbiter of the bond probability and
of every normalisation here (ground rule 1 -- tests arbitrate conventions).

Magnitude updates are mandatory
-------------------------------
Cluster flips never change |phi_x|, so on their own they do not sample the
magnitude distribution at all. Each hybrid sweep therefore interleaves
``n_local_per_sweep >= 1`` local Metropolis sweeps (reusing the single-site
kernel of ``mc/metropolis.py``) with ``n_cluster_per_sweep`` cluster flips;
both counts are exposed via :class:`~qft_graph.config.MCConfig`.
"""

from __future__ import annotations

import logging
import math
import time

import numpy as np
import torch

from qft_graph.actions.phi4 import Phi4Action
from qft_graph.config import MCConfig
from qft_graph.mc.metropolis import CheckerboardSampler, MetropolisSampler
from qft_graph.mc.sampler import MCResult, MCSampler

logger = logging.getLogger("qft_graph.mc")


def _make_local_sampler(action: Phi4Action, config: MCConfig) -> MCSampler:
    """Pick the local (magnitude-update) sampler for the cluster hybrid.

    Uses the vectorized :class:`CheckerboardSampler` whenever the lattice is
    periodic with all-even extents (the only case for which the even/odd
    2-colouring is a valid bipartition), otherwise the sequential
    :class:`MetropolisSampler`. Both implement identical physics; the choice
    is purely a speed optimisation and is transparent to the cluster move.
    """
    lattice = action.lattice
    dims = lattice.shape
    periodic = lattice.boundary() == "periodic"
    if periodic and all(d % 2 == 0 for d in dims):
        return CheckerboardSampler(action, config)
    return MetropolisSampler(action, config)


class ClusterSampler(MCSampler):
    """Wolff/Brower-Tamayo cluster-hybrid sampler for phi^4 theory.

    Args:
        action: :class:`Phi4Action` defining the theory.
        config: :class:`MCConfig`. ``n_cluster_per_sweep`` and
            ``n_local_per_sweep`` set the hybrid ratio; ``step_size`` and
            ``seed`` are forwarded to the local Metropolis sweeps.

    Notes:
        The cluster growth uses its own ``numpy.random.RandomState(seed)``;
        the local sampler is seeded independently but deterministically, so
        runs are reproducible from ``config.seed``.
    """

    def __init__(self, action: Phi4Action, config: MCConfig) -> None:
        if config.n_local_per_sweep < 1:
            raise ValueError(
                "n_local_per_sweep must be >= 1: cluster flips never update "
                "|phi|, so magnitude updates are mandatory (Brower-Tamayo)."
            )
        if config.n_cluster_per_sweep < 1:
            raise ValueError("n_cluster_per_sweep must be >= 1.")

        self.action = action
        self.config = config
        self.n_cluster_per_sweep = config.n_cluster_per_sweep
        self.n_local_per_sweep = config.n_local_per_sweep

        self.rng = np.random.RandomState(config.seed)
        self._nsites = action.lattice.num_sites()

        # Ferromagnetic coupling K = a^{d-2} multiplying phi_i phi_j in the
        # bond action (= 1 in 2D for any spacing). Validated by the Metropolis
        # cross-check test rather than trusted from this derivation.
        self._K = float(action.a ** (action.d - 2))

        # Local (magnitude) sampler: reuse the Metropolis single-site kernel.
        self._local = _make_local_sampler(action, config)

        # Per-site distinct neighbours + integer bond multiplicity. Multiplicity
        # exceeds 1 only on degenerate small lattices (e.g. 2x2, where the +mu
        # and -mu neighbours wrap to the same site and the bond is doubled);
        # the doubled bond then carries coupling 2K, so the bond probability
        # must use K * multiplicity. On all production lattices multiplicity is
        # 1 and this reduces to the textbook rule.
        src = action._src.numpy()
        dst = action._dst.numpy()
        counts: list[dict[int, int]] = [dict() for _ in range(self._nsites)]
        for s, d in zip(src.tolist(), dst.tolist()):
            counts[s][d] = counts[s].get(d, 0) + 1
        self._nbr: list[np.ndarray] = []
        self._mult: list[np.ndarray] = []
        for i in range(self._nsites):
            js = np.fromiter(counts[i].keys(), dtype=np.int64)
            ms = np.fromiter(counts[i].values(), dtype=np.float64)
            self._nbr.append(js)
            self._mult.append(ms)

        self._last_cluster_sizes: list[int] = []

    # ------------------------------------------------------------------ core

    def _cluster_step(self, phi: np.ndarray) -> int:
        """Grow and flip one single-cluster (Wolff) move in place.

        Args:
            phi: Field configuration (numpy, shape (nsites,)); modified in
                place -- the grown cluster has its sign reflected.

        Returns:
            Number of sites in the flipped cluster.
        """
        rng = self.rng
        two_k = 2.0 * self._K
        nbr = self._nbr
        mult = self._mult

        seed = int(rng.randint(self._nsites))
        in_cluster = np.zeros(self._nsites, dtype=bool)
        in_cluster[seed] = True
        stack = [seed]
        size = 1

        while stack:
            i = stack.pop()
            phi_i = phi[i]
            neighbours = nbr[i]
            mults = mult[i]
            for k in range(neighbours.shape[0]):
                j = int(neighbours[k])
                if in_cluster[j]:
                    continue
                prod = phi_i * phi[j]
                if prod > 0.0:  # same-sign bond only
                    # p = 1 - exp(-2 K mult phi_i phi_j); draw before computing
                    # exp when possible. prod>0 => argument<0 => p in (0,1).
                    if rng.random() < 1.0 - math.exp(-two_k * mults[k] * prod):
                        in_cluster[j] = True
                        stack.append(j)
                        size += 1

        # Rejection-free flip of the whole cluster.
        np.negative(phi, out=phi, where=in_cluster)
        return size

    def sweep(self, phi: torch.Tensor) -> tuple[torch.Tensor, float]:
        """One hybrid sweep: cluster flips, then local magnitude sweeps.

        Args:
            phi: Current field configuration, shape (nsites,).

        Returns:
            (updated_phi, local_acceptance_rate). The acceptance rate refers
            to the local Metropolis sweeps; cluster flips are rejection-free.
            The flipped cluster sizes are stashed in ``_last_cluster_sizes``.
        """
        phi_np = phi.detach().numpy().astype(np.float64, copy=True)
        sizes = [self._cluster_step(phi_np) for _ in range(self.n_cluster_per_sweep)]
        self._last_cluster_sizes = sizes

        phi_t = torch.from_numpy(phi_np).to(dtype=phi.dtype)
        accs: list[float] = []
        for _ in range(self.n_local_per_sweep):
            phi_t, acc = self._local.sweep(phi_t)
            accs.append(acc)

        return phi_t, float(np.mean(accs))

    # ------------------------------------------------------------- generation

    def generate(
        self,
        n_configs: int,
        initial_phi: torch.Tensor | None = None,
    ) -> MCResult:
        """Generate decorrelated configurations with the cluster hybrid.

        Args:
            n_configs: Number of configurations to produce.
            initial_phi: Optional initial field for warm-starting.

        Returns:
            :class:`MCResult`. ``observables`` carries ``cluster_fraction``
            (per-config mean flipped-cluster size / V) and ``local_acceptance``
            (per-config mean local acceptance), for the tau_int / diagnostics
            in the FSS analysis.
        """
        if initial_phi is not None:
            phi = initial_phi.clone().float()
        else:
            # Seed the initial config from the sampler's own RNG (not the global
            # torch RNG) so a run is fully reproducible from config.seed
            # (ground rule 5), independent of any surrounding torch state.
            phi = torch.from_numpy(
                self.rng.uniform(-1.0, 1.0, size=self._nsites)
            ).float()

        t0 = time.time()
        logger.info(
            "Thermalizing cluster hybrid for %d sweeps on %d sites "
            "(%d cluster + %d local per sweep)...",
            self.config.n_thermalization, self._nsites,
            self.n_cluster_per_sweep, self.n_local_per_sweep,
        )
        for i in range(self.config.n_thermalization):
            phi, _ = self.sweep(phi)
            if (i + 1) % 200 == 0:
                logger.info(
                    "  Thermalization %d/%d (%.1f sweeps/s)",
                    i + 1, self.config.n_thermalization,
                    (i + 1) / (time.time() - t0),
                )
        logger.info("Thermalization done in %.1fs.", time.time() - t0)

        configurations = torch.zeros(n_configs, self._nsites)
        actions = torch.zeros(n_configs)
        cluster_fraction = torch.zeros(n_configs)
        local_acceptance = torch.zeros(n_configs)

        t1 = time.time()
        logger.info("Generating %d configurations (cluster hybrid)...", n_configs)
        for i in range(n_configs):
            step_sizes: list[int] = []
            accs: list[float] = []
            for _ in range(self.config.n_sweeps_between):
                phi, acc = self.sweep(phi)
                step_sizes.extend(self._last_cluster_sizes)
                accs.append(acc)

            configurations[i] = phi.clone()
            with torch.no_grad():
                actions[i] = self.action(phi)
            cluster_fraction[i] = (
                float(np.mean(step_sizes)) / self._nsites if step_sizes else 0.0
            )
            local_acceptance[i] = float(np.mean(accs)) if accs else 0.0

            if (i + 1) % 500 == 0:
                logger.info(
                    "  Generated %d/%d configs (%.1f configs/s)",
                    i + 1, n_configs, (i + 1) / (time.time() - t1),
                )

        mean_frac = float(cluster_fraction.mean())
        logger.info(
            "Generation done in %.1fs. Mean cluster fraction %.3f, "
            "mean local acceptance %.3f.",
            time.time() - t1, mean_frac, float(local_acceptance.mean()),
        )

        return MCResult(
            configurations=configurations,
            actions=actions,
            acceptance_rate=float(local_acceptance.mean()),
            observables={
                "cluster_fraction": cluster_fraction,
                "local_acceptance": local_acceptance,
            },
        )


def create_cluster_sampler(action: Phi4Action, config: MCConfig) -> ClusterSampler:
    """Factory mirroring ``metropolis.create_sampler`` for the cluster hybrid."""
    return ClusterSampler(action, config)
