> **APPROVED** by Josh, 2026-07-09 — the two-momentum estimator is adopted for xi in Fig. 2 and the nu extraction; chi stays as frozen.
> **Update 2026-07-29:** the two-momentum estimator remains the paper's xi definition (xi/L crossings), but the nu extraction changed in V2-3 (commit 30599e4, 2026-07-20): the xi/L data-collapse nu estimator was found to be a method artefact (no interior minimum), and nu/m2_c now come from the pseudocritical-shift fit m2_peak(L) = m2_c - a·L^(-1/nu) (`src/qft_graph/analysis/critical.py::fit_nu_from_pseudocritical_shifts`), giving nu = 1.04(8), m2_c = -2.217(3).

# FLAG for Josh: the frozen ξ recipe degenerates — proposed k≠0 estimator (P1-2)

**Frozen convention change proposal — needs your sign-off before arXiv v1**
(plan ground rule 2: propose in a comment, never change silently).

## Evidence (seeded FSS sweep v2, L=16/32, 2000 configs/point, binned errors)

The plan froze (§ Task P1-2):

```
chi     = V * ( <M^2> - <|M|>^2 )          # susceptibility AND G(k=0)
xi      = 1/(2 sin(pi/L)) * sqrt( G(0)/G(kmin) - 1 )
```

Substituting the ⟨|M|⟩-subtracted χ as G̃(0) **fails identically in the
symmetric phase**: for Gaussian M, ⟨|M|⟩² = (2/π)⟨M²⟩, so
χ_abs ≈ 0.363·χ_var < G̃(k_min) whenever the true ξ/L is small, the ratio
under the square root goes negative, and the estimator returns 0. In the
v2 sweep, **ξ_abs ≡ 0 for every point with m² ≥ −1.99 at L=16 (m² ≥ −2.04
at L=32)** — half the scan. No crossing analysis is possible with it.

The Var(M) variant (what the published Fig. 2 actually used inside
`correlation_length_fft`) is defined in both phases but is
**tunneling-contaminated in the broken phase**: at L=16 it gives
ξ/L ≈ 0.71–0.97 for m² ∈ [−2.31, −2.20] (unphysical, > the theoretical
crossing value), because rare vacuum tunneling inflates Var(M) at small
volume. The pairwise "crossings" it produces with properly decorrelated,
seeded data are noise-dominated.

## Proposal

Use the standard **two-momentum second-moment estimator** for ξ (k=0 mode
never enters):

```
G(k) = A/(khat² + ξ⁻²),  khat = 2 sin(k/2)
ξ⁻²  = (khat₂² G₂ − khat₁² G₁) / (G₁ − G₂)
G₁ = ⟨|φ̃(k)|²⟩/V averaged over k ∈ {(1,0),(0,1)},  G₂ at k = (1,1)
```

- Immune to both failure modes above (no ⟨M⟩/⟨|M|⟩ subtraction, no Var(M)).
- Exact-value oracle: on exactly-sampled free lattice fields it recovers
  ξ = 1/m to <3% (tests/test_mc/test_observables.py::TestTwoMomentumXi).
- Implemented as `ObservableSet.correlation_length_two_momentum`; the sweep
  now records ξ under **all three** definitions plus per-config
  M and |φ̃(k)|² series (`--store_series`), so switching conventions is
  offline post-processing, never an MC rerun.

**χ itself stays exactly as frozen** — V(⟨M²⟩−⟨|M|⟩²) for the
susceptibility panel, peaks, and γ/ν fit. The proposal only concerns which
correlator ratio defines ξ in Fig. 2's third panel and the ν collapse.

## What happens if you decline

Fig. 2's ξ/L panel and the ν extraction revert to the Var(M) estimator
(the published choice), with an honest caveat about broken-phase tunneling
contamination; the frozen-form ξ cannot be used as written. All three are
in the sweep JSONs either way.
