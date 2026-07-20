"""Physics-oracle tests for the Wolff/Brower-Tamayo cluster sampler (task V2-1).

These exact-value and cross-sampler oracles gate every cluster production run
(ground rule 1). They arbitrate the bond-probability prefactor ``2 K`` in
``mc/cluster.py``: any error there surfaces as disagreement with the Metropolis
sampler on the interacting theory (:class:`TestCrossCheckMetropolis`) or with
the exact free-field / 2x2-Boltzmann references (:class:`TestFreeFieldExact`,
:class:`TestDetailedBalance`).

Every test is deterministic (fixed seeds), so the reported margins are fixed
rather than flaky. The suite runs in a couple of minutes on a laptop CPU.

Physics reference (lam=0.5): the susceptibility peaks near m^2 ~ -2.15 for this
action normalisation (results/fss_analysis_v5.json); the cross-checks straddle
that pseudo-critical point (symmetric m^2=-1.8, near-critical -2.15, broken
-2.6) and the autocorrelation test sits on it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from qft_graph.actions.phi4 import Phi4Action
from qft_graph.config import LatticeConfig, MCConfig, ScalarFieldConfig
from qft_graph.lattice.hypercubic import HypercubicLattice
from qft_graph.mc.analysis import binned_jackknife, integrated_autocorrelation_time
from qft_graph.mc.cluster import ClusterSampler
from qft_graph.mc.metropolis import CheckerboardSampler
from qft_graph.mc.observables import ObservableSet


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _action(L: int, m_sq: float, lam: float) -> Phi4Action:
    lattice = HypercubicLattice(LatticeConfig(dimensions=(L, L)))
    return Phi4Action(lattice, ScalarFieldConfig(mass_squared=m_sq, coupling=lam))


def cluster_result(
    L: int,
    m_sq: float,
    lam: float,
    *,
    n_configs: int,
    n_therm: int,
    n_between: int,
    n_cluster: int = 1,
    n_local: int = 1,
    seed: int = 1,
):
    """Run the cluster hybrid and return its MCResult."""
    cfg = MCConfig(
        n_configs=n_configs,
        n_thermalization=n_therm,
        n_sweeps_between=n_between,
        step_size=1.0,
        seed=seed,
        n_cluster_per_sweep=n_cluster,
        n_local_per_sweep=n_local,
    )
    return ClusterSampler(_action(L, m_sq, lam), cfg).generate(n_configs)


def checkerboard_result(
    L: int,
    m_sq: float,
    lam: float,
    *,
    n_configs: int,
    n_therm: int,
    n_between: int,
    seed: int = 2,
):
    """Run the (independent) Checkerboard Metropolis sampler for cross-checks."""
    cfg = MCConfig(
        n_configs=n_configs,
        n_thermalization=n_therm,
        n_sweeps_between=n_between,
        step_size=1.0,
        seed=seed,
    )
    return CheckerboardSampler(_action(L, m_sq, lam), cfg).generate(n_configs)


def _abs_M(configs: torch.Tensor) -> float:
    return float(configs.mean(dim=1).abs().mean())


def _phi2(configs: torch.Tensor) -> float:
    return float(configs.pow(2).mean())


# --------------------------------------------------------------------------- #
# Test 1 -- free field (lambda = 0): exact Gaussian oracle
# --------------------------------------------------------------------------- #
class TestFreeFieldExact:
    """At lam=0 the theory is Gaussian with G(k) = 1/(khat^2 + m^2), so the
    cluster+local hybrid must reproduce the exact lattice moments."""

    @staticmethod
    def phi2_exact(L: int, m_sq: float) -> float:
        """<phi^2> = (1/V) sum_k 1/(khat^2 + m^2), khat^2 = sum_mu 4 sin^2(k_mu/2)."""
        k = 2.0 * np.pi * np.fft.fftfreq(L)
        khat_sq = (2.0 * np.sin(k / 2.0)) ** 2
        K2 = khat_sq[:, None] + khat_sq[None, :]
        return float(np.mean(1.0 / (K2 + m_sq)))

    @pytest.mark.parametrize("L, m_sq", [(8, 1.0), (16, 0.5)])
    def test_phi2_matches_gaussian(self, L: int, m_sq: float):
        res = cluster_result(
            L, m_sq, 0.0, n_configs=4000, n_therm=500, n_between=2, n_cluster=2, seed=1
        )
        measured = _phi2(res.configurations)
        exact = self.phi2_exact(L, m_sq)
        assert abs(measured - exact) / exact < 0.03, (
            f"<phi^2>={measured:.4f} vs exact {exact:.4f} (L={L}, m^2={m_sq})"
        )

    def test_momentum_propagator(self):
        """The low-momentum propagator <|phi(k)|^2>/V must equal 1/(khat^2+m^2)."""
        L, m_sq = 16, 0.5
        res = cluster_result(
            L, m_sq, 0.0, n_configs=5000, n_therm=500, n_between=2, n_cluster=2, seed=4
        )
        mom = ObservableSet.momentum_correlators(res.configurations, L)
        k1_sq = (2.0 * np.sin(np.pi / L)) ** 2
        expected = 1.0 / (k1_sq + m_sq)
        measured = float((mom["F_1_0"].mean() + mom["F_0_1"].mean()) / 2.0)
        assert abs(measured - expected) / expected < 0.05, (
            f"G(k_min)={measured:.4f} vs exact {expected:.4f}"
        )


# --------------------------------------------------------------------------- #
# Test 2 -- cross-check vs Metropolis (arbiter of the bond prefactor)
# --------------------------------------------------------------------------- #
class TestCrossCheckMetropolis:
    """Same physics, different algorithm: <|M|>, chi (frozen convention),
    <phi^2> and <S> from the cluster hybrid must agree with the independent
    Checkerboard-Metropolis sampler within combined jackknife errors. Any error
    in the bond probability p = 1-exp(-2 K phi_i phi_j) would break this.

    A wrong prefactor shifts chi by tens of sigma; the 5-sigma gate below
    cleanly separates "correct" from "broken" while tolerating the normal
    scatter of ~20 independent mean comparisons (seeds are fixed, so the
    outcome is deterministic)."""

    SIGMA_GATE = 5.0

    @pytest.mark.parametrize(
        "L, m_sq",
        [(8, -1.8), (8, -2.15), (8, -2.6), (16, -2.6)],
    )
    def test_agrees_with_metropolis(self, L: int, m_sq: float):
        n = 3000
        rclu = cluster_result(
            L, m_sq, 0.5, n_configs=n, n_therm=1000, n_between=2, n_cluster=2, seed=1
        )
        rcb = checkerboard_result(
            L, m_sq, 0.5, n_configs=n, n_therm=3000, n_between=10, seed=2
        )

        estimators = {
            "abs_M": lambda c: _abs_M(c),
            "chi": lambda c: ObservableSet.susceptibility(c, "abs"),
            "phi2": lambda c: _phi2(c),
        }
        for name, est in estimators.items():
            v1, e1 = binned_jackknife(rclu.configurations, est, n_bins=20)
            v2, e2 = binned_jackknife(rcb.configurations, est, n_bins=20)
            sigma = abs(v1 - v2) / np.sqrt(e1**2 + e2**2 + 1e-300)
            assert sigma < self.SIGMA_GATE, (
                f"{name}: cluster {v1:.4f}({e1:.4f}) vs metropolis {v2:.4f}({e2:.4f}) "
                f"= {sigma:.1f} sigma (L={L}, m^2={m_sq})"
            )

        # Mean action <S> as a fourth, independent handle.
        s1, es1 = binned_jackknife(
            rclu.actions, lambda z: float(np.mean(np.asarray(z))), n_bins=20
        )
        s2, es2 = binned_jackknife(
            rcb.actions, lambda z: float(np.mean(np.asarray(z))), n_bins=20
        )
        sigma_S = abs(s1 - s2) / np.sqrt(es1**2 + es2**2 + 1e-300)
        assert sigma_S < self.SIGMA_GATE, (
            f"<S>: cluster {s1:.3f}({es1:.3f}) vs metropolis {s2:.3f}({es2:.3f}) "
            f"= {sigma_S:.1f} sigma (L={L}, m^2={m_sq})"
        )


# --------------------------------------------------------------------------- #
# Test 3 -- detailed balance / Boltzmann weight on a tiny lattice
# --------------------------------------------------------------------------- #
class TestDetailedBalance:
    """On a 2x2 lattice the Boltzmann integral is 4-dimensional and can be done
    by direct quadrature, giving an exact (non-sampler) reference for the
    interacting theory. The cluster hybrid must reproduce it -- this checks the
    full move (including the doubled-bond multiplicity handling that the 2x2
    torus requires) samples exp(-S)."""

    M_SQ, LAM = 1.0, 0.5  # symmetric phase: the 4D integral converges fast

    @staticmethod
    def _S_2x2(P: np.ndarray, m_sq: float, lam: float) -> np.ndarray:
        """Exact action on the 2x2 torus for field array P[..., 4].

        Sites [0,1,2,3] = (0,0),(0,1),(1,0),(1,1). The four nearest-neighbour
        bonds {0,1},{2,3},{0,2},{1,3} are each doubled by the wrap-around, so
        each contributes (phi_i-phi_j)^2 (= 2 x the usual 1/2 (dphi)^2)."""
        p0, p1, p2, p3 = P[..., 0], P[..., 1], P[..., 2], P[..., 3]
        kinetic = (p0 - p1) ** 2 + (p2 - p3) ** 2 + (p0 - p2) ** 2 + (p1 - p3) ** 2
        mass = 0.5 * m_sq * (P**2).sum(-1)
        quartic = lam * (P**4).sum(-1)
        return kinetic + mass + quartic

    def test_action_formula_matches_code(self):
        """The quadrature reference must use exactly the code's action."""
        action = _action(2, self.M_SQ, self.LAM)
        rng = np.random.default_rng(0)
        for _ in range(8):
            phi = rng.normal(size=4)
            s_code = float(action(torch.from_numpy(phi).float()))
            s_form = float(self._S_2x2(phi, self.M_SQ, self.LAM))
            assert abs(s_code - s_form) < 1e-4, f"{s_code} vs {s_form}"

    def _exact_2x2_moments(self):
        N, R = 44, 4.2
        x = np.linspace(-R, R, N)
        grid = np.stack(np.meshgrid(x, x, x, x, indexing="ij"), axis=-1)
        w = np.exp(-self._S_2x2(grid, self.M_SQ, self.LAM))
        Z = w.sum()
        phi2 = ((grid**2).mean(-1) * w).sum() / Z
        M = grid.mean(-1)
        M2 = ((M**2) * w).sum() / Z
        return float(phi2), float(M2)

    @pytest.fixture(scope="class")
    def run_2x2(self):
        """One 2x2 cluster run, shared by the moment and reflection checks."""
        return cluster_result(
            2, self.M_SQ, self.LAM,
            n_configs=20000, n_therm=2000, n_between=1, n_cluster=1, seed=3,
        )

    def test_2x2_boltzmann_moments(self, run_2x2):
        phi2_exact, M2_exact = self._exact_2x2_moments()
        configs = run_2x2.configurations
        phi2 = _phi2(configs)
        M2 = float(configs.mean(dim=1).pow(2).mean())
        assert abs(phi2 - phi2_exact) / phi2_exact < 0.03, (
            f"<phi^2>={phi2:.4f} vs exact {phi2_exact:.4f}"
        )
        assert abs(M2 - M2_exact) / M2_exact < 0.04, (
            f"<M^2>={M2:.4f} vs exact {M2_exact:.4f}"
        )

    def test_reflection_symmetry(self, run_2x2):
        """The Z2-symmetric ensemble must have <M> ~ 0 within error."""
        M = run_2x2.configurations.mean(dim=1)
        mean_M, err_M = binned_jackknife(M, lambda z: float(np.mean(np.asarray(z))), 20)
        assert abs(mean_M) < 5.0 * err_M + 1e-3, f"<M>={mean_M:.4f} +/- {err_M:.4f}"

    def test_seed_independence(self):
        """Two independent seeds must agree on <phi^2> within combined error."""
        kw = dict(n_configs=12000, n_therm=1500, n_between=1, n_cluster=1)
        r1 = cluster_result(2, self.M_SQ, self.LAM, seed=3, **kw)
        r2 = cluster_result(2, self.M_SQ, self.LAM, seed=17, **kw)
        v1, e1 = binned_jackknife(r1.configurations, _phi2, 20)
        v2, e2 = binned_jackknife(r2.configurations, _phi2, 20)
        sigma = abs(v1 - v2) / np.sqrt(e1**2 + e2**2 + 1e-300)
        assert sigma < 5.0, f"seed 3 {v1:.4f}({e1:.4f}) vs seed 17 {v2:.4f}({e2:.4f})"


# --------------------------------------------------------------------------- #
# Test 4 -- autocorrelation reduction (the justification; logged)
# --------------------------------------------------------------------------- #
class TestAutocorrelationReduction:
    """The point of the cluster algorithm: at the pseudo-critical point the
    local sampler's tau_int(|M|) grows with L (the critical-slowing-down bias
    the paper indicts -- reaching ~35-100 at the production peaks), while the
    cluster hybrid keeps it O(1). Measured in single-sweep units at m^2=-2.15.

    Not a correctness gate; the assertions are deliberately loose (cluster
    stays bounded and beats the local sampler at the largest L) and the numbers
    are printed for the paper's tau_int comparison table (task V2-2)."""

    M_SQ = -2.15

    def test_tau_int_stays_bounded(self, capsys):
        sizes = [8, 16, 24]
        rows = []
        for L in sizes:
            rclu = cluster_result(
                L, self.M_SQ, 0.5,
                n_configs=4000, n_therm=800, n_between=1, n_cluster=2, seed=1,
            )
            rcb = checkerboard_result(
                L, self.M_SQ, 0.5, n_configs=4000, n_therm=1500, n_between=1, seed=2
            )
            absM_clu = rclu.configurations.mean(dim=1).abs().numpy()
            absM_cb = rcb.configurations.mean(dim=1).abs().numpy()
            tau_clu = integrated_autocorrelation_time(absM_clu)
            tau_cb = integrated_autocorrelation_time(absM_cb)
            frac = float(rclu.observables["cluster_fraction"].mean())
            rows.append((L, tau_clu, tau_cb, frac))

        with capsys.disabled():
            print("\n  tau_int(|M|) at m^2=-2.15 (single-sweep units):")
            print("   L   cluster   local(checkerboard)   cluster_fraction")
            for L, tc, tb, fr in rows:
                print(f"  {L:3d}   {tc:6.2f}   {tb:16.2f}   {fr:12.3f}")

        taus_clu = [tc for _, tc, _, _ in rows]
        taus_loc = [tb for _, _, tb, _ in rows]

        # (i) The cluster hybrid decorrelates strictly faster than the local
        #     sampler at every L.
        for L, tc, tb, fr in rows:
            assert tc < tb, f"cluster tau_int {tc:.2f} !< local {tb:.2f} at L={L}"

        # (ii) Critical slowing down IS present in the local reference at the
        #      largest, most-critical L -- otherwise the test would not be
        #      exercising the regime the cluster algorithm exists to fix.
        L_max, _, tb_max, _ = rows[-1]
        assert tb_max > 30.0, (
            f"local tau_int {tb_max:.2f} at L={L_max} too small to show CSD"
        )

        # (iii) The cluster hybrid does NOT inherit that blow-up: its worst-case
        #       tau_int stays O(10) and a small fraction of the local sampler's.
        assert max(taus_clu) < 0.5 * max(taus_loc), (
            f"cluster max tau_int {max(taus_clu):.2f} not << local "
            f"{max(taus_loc):.2f}"
        )
        assert max(taus_clu) < 40.0, (
            f"cluster max tau_int {max(taus_clu):.2f} unexpectedly large"
        )
