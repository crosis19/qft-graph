# Phase I journal v2 — cluster sampler + precision FSS (handoff spec)

**Owner:** Josh Brehm · **Branch:** `paper-v2` (do all this work here) ·
**Created:** 2026-07-18 · **Status:** approved by Josh, ready to start.

A fresh session can execute this without the originating conversation. Read
`CLAUDE.md` (ground rules) and this file; you do **not** need to read the whole
`qft_graph_implementation_plan.md` (that's WS1/Phase II). Physics-oracle tests
gate every production run (ground rule 1) — the tests are the arbiters of any
sign/normalization convention; if a derivation here disagrees with a passing
test, trust the test and flag it.

---

## 0. Context: what v1 is, what v2 is, and the git layout

- **arXiv v1 is SUBMITTED and FROZEN.** Tag `paper-v1` → commit `8e15ed9` is the
  exact submission. **Never touch v1.** `master`/`origin/master` sit at the v1
  paper content (+ a Makefile-only fix). Do not merge v2 into master or push
  `paper-v2` until Josh says the v2 paper is complete and re-submitted.
- **v2 = the journal version**, developed entirely on `paper-v2`.
- **Already done for v2** (committed on `paper-v2`; do NOT redo — just verify
  present): all of Schaich's July-17 feedback EXCEPT the cluster algorithm —
  chi-peak covariance error propagation (`src/qft_graph/analysis/critical.py`,
  commit de9f277); 1−r tables, full-width/vertically-stacked figures with
  colorblind-distinct per-L markers, modern reference arXiv:2512.16536, and the
  z≈0.25 (not "near zero") correction (commit 6631184); `analyze_fss.py`
  `--peak_dir`/mixed-grid support (d8d4188).
- **The one Schaich item this spec delivers: the cluster algorithm** and the
  precision FSS it enables.

### Why we're doing it (the FRAMING — agreed with Josh, keep to it)

The uniform 5-size **local-Metropolis** FSS (committed:
`results/fss_analysis_v5.json`, refined data in
`results/fss_peakrefine_*.json` + `data/sweep_peakrefine/`) gives a *precise but
critical-slowing-down-BIASED* γ/ν = **1.60(3)** — the effective slope falls
1.64→1.42 with L, χ_max(64)≈85 vs the L^1.75 prediction ≈100, and τ_int reaches
~35–100 at the peaks. **Do not quote 1.60(3).**

Present the cluster result as **methodological completeness, NOT a precision-
exponent claim.** The narrative: *the local-sampler exponent is critical-slowing-
down-biased (show the τ_int and the L-dependent effective slope as evidence); a
cluster algorithm (z≈0.25) removes the bias and confirms the exact Ising values,
illustrating exactly the sampling bottleneck that motivates the ML program — and
which is unavailable for the gauge+fermion theories of Phase II+ (cluster
algorithms do not exist for QCD).* This keeps the paper in its lane (an
architecture paper) instead of competing as a lattice-FSS study. This framing is
also genuinely useful capital for WS3 (delayed-acceptance / determinant surrogate
is fundamentally about beating sampling cost).

---

## Task V2-1: Wolff / Brower–Tamayo cluster sampler for 2D φ⁴

**Files:** new `src/qft_graph/mc/cluster.py` (subclass `mc/sampler.py::MCSampler`,
same interface as `mc/metropolis.py`); tests `tests/test_mc/test_cluster.py`;
config `configs/mc/cluster_phi4.yaml`. Keep numpy/scipy only (ground rule 5);
numba `@njit` is acceptable for the cluster grow loop if pure-Python is too slow,
but correctness first.

### The physics (derive + VALIDATE; do not take the prefactor on faith)

Our action (`actions/phi4.py`, a=1 lattice units, 2D):
```
S = Σ_x [ ½ Σ_μ (φ_{x+μ} − φ_x)²  +  ½ m² φ_x²  +  λ φ_x⁴ ]
```
The kinetic term expands to a nearest-neighbor coupling −Σ_{<ij>} φ_i φ_j (J=1)
plus on-site φ² pieces. Under the Z₂ reflection φ→−φ applied to a cluster C,
only bonds crossing ∂C change the action, each by **ΔS_bond = 2 φ_i φ_j**
(derivation: ½(φ_j+φ_i)² − ½(φ_j−φ_i)² = 2φ_iφ_j). The on-site ½m²φ²+λφ⁴ is
Z₂-invariant, so the cluster move preserves every |φ_x| and touches only signs.

**Embedded-Wolff (single-cluster) step:**
1. Pick a random seed site; reflection is φ→−φ.
2. Grow the cluster: from a site i already in C, add a same-sign neighbor j
   (i.e. φ_i φ_j > 0) with **bond probability p_ij = 1 − exp(−2 φ_i φ_j)**;
   never add opposite-sign neighbors. (This is the standard FK/Wolff form for
   J=1; the factor 2 is the ΔS_bond above. **VALIDATE the exact prefactor via
   the small-L Metropolis cross-check below — that test is the arbiter.**)
3. Flip φ→−φ on all of C. Rejection-free by construction.

**Magnitude updates are mandatory.** Cluster flips never change |φ_x|, so they do
not sample the magnitude distribution at all. Interleave local updates
(Brower–Tamayo hybrid): e.g. one full local-Metropolis sweep (reuse
`mc/metropolis.py`'s single-site kernel) per cluster step, or a tunable ratio.
Expose the ratio and cluster-steps-per-"sweep" as config.

### Exact-value physics tests (write BEFORE any production run)

1. **Free field (λ=0):** cluster+local must reproduce the exact Gaussian
   ⟨φ²⟩ = (1/V) Σ_k 1/(k̂² + m²), k̂²=Σ_μ 4sin²(k_μ/2), and the momentum-space
   propagator, to statistical error at (m², L) ∈ {(1,8),(0.5,16)}. (Phase I has
   the free-field exact form; cross-check `paper/generate_figures.py::fig_free_field`
   context.)
2. **Cross-check vs Metropolis at small L (THE correctness test):** at L=8 and
   L=16, with both samplers, ⟨|M|⟩, χ (frozen conv V(⟨M²⟩−⟨|M|⟩²)), ⟨φ²⟩, and
   ⟨S⟩ must AGREE within jackknife errors across a few m² (symmetric, near-critical,
   broken). Same physics, different algorithm — any disagreement means the bond
   probability / hybrid is wrong. This test arbiters the prefactor.
3. **Detailed balance sanity:** on a tiny lattice (e.g. 2×2 or 4×4) verify the
   sampler reproduces the Boltzmann weight (high-stats histogram of a coarse
   observable vs a direct/nearly-exact reference), or at minimum that observables
   are reflection-symmetric and independent of seed-site choice.
4. **Autocorrelation reduction (the point):** measure τ_int(χ, |M|) vs L near
   criticality; confirm it stays O(1) where local Metropolis blew up to ~35–100.
   Not a correctness test but the justification — log it.

**V2-1 done when:** tests 1–3 green at every listed (m², L); test 4 shows τ_int
does not grow like the local sampler's; sampler committed with config.

---

## Task V2-2: Cluster FSS production + re-analysis

- **Regenerate the FSS sweeps with the cluster sampler.** Because z≈0.25, larger
  lattices are now cheap — consider extending to L ∈ {16,24,32,48,64,96,128}
  (adds real leverage to the fits; the local sampler couldn't reach these
  cleanly). Keep the frozen χ convention V(⟨M²⟩−⟨|M|⟩²) and the k≠0 two-momentum
  ξ estimator (`ObservableSet.correlation_length_two_momentum`, approved
  docs/xi_estimator_issue.md). Dense grids around each peak (reuse the
  `scripts/sweep.py` `--store_series` machinery). New output dir, e.g.
  `data/sweep_cluster/`; provenance via `log_run` to `results/`.
- **Re-extract all three exponents with `scripts/analyze_fss.py`** (already
  supports this data unchanged): γ/ν from the χ_max fit, ν from the ξ/L collapse,
  m²_c from crossings. With unbiased data these should now be consistent with the
  exact Ising values (γ/ν=1.75, ν=1) with honest small errors.
- **Robustness (Schaich's other suggestions):** offer a corrections-to-scaling
  fit χ_max = A L^{γ/ν}(1 + b L^{−ω}) and/or an AIC-weighted model average
  (Schaich cited arXiv:2008.01069) as a cross-check on the naive power law. With
  clean cluster data the naive and corrected fits should agree — report both.
- **Keep the local-Metropolis result** (`results/fss_analysis_v5.json`) as the
  *evidence for the CSD bias* in the paper's discussion — do not discard it.

**V2-2 done when:** cluster FSS analysis JSON in `results/`; γ/ν, ν, m²_c
extracted with the cluster data (+ corrections/AIC cross-check); τ_int comparison
table (local vs cluster) assembled for the paper.

---

## Task V2-3: FSS section rewrite + numbers + figures (paper edits)

- **Sec. IV.D–E rewrite** to the framing in §0: local-sampler exponent is
  CSD-biased (evidence: effective slope 1.64→1.42 with L, τ_int table); the
  cluster algorithm removes the bias and confirms Ising values; note that no such
  algorithm exists for the gauge+fermion theories the program targets.
- **Update the numbers** (script-generated per ground rule 4): abstract still has
  the **stale γ/ν = 1.57(12)** and IV.D/E carry the old χ_max/ν/m²_c — replace
  with the cluster values. Regenerate the FSS + collapse figures (now clean, more
  sizes, per-L colorblind markers already set up).
- **Abstract/conclusion wording changes need Josh's explicit sign-off** (project
  discipline — see the `> APPROVED by Josh` headers in docs/*.md for the pattern).
  Draft, present, wait.
- Rebuild PDF; refresh the arXiv tarball only when Josh is ready to re-submit.

**V2-3 done when:** paper numbers all trace to the cluster analysis; Josh has
approved abstract/conclusion wording; PDF builds clean.

---

## Recommended first step (optional): note to Schaich

Josh may send a short note to Schaich before/while building this — he offered the
exact guidance (AIC fits, corrections-to-scaling, cluster z≈0.25) and would likely
have a view on presentation. Draft (Josh sends; do not send on his behalf):

> Dear Prof. Schaich — thanks again for the detailed feedback. For the journal
> version I'm taking up your cluster-algorithm suggestion: when I refined the
> susceptibility-peak scans the local-Metropolis γ/ν came out ~1.60, biased low
> by critical slowing down (τ_int ~ 35–100 at the peaks, effective slope falling
> with L). I'm implementing a Wolff/Brower–Tamayo embedded-cluster sampler to
> confirm the Ising values, and framing it as a methodological illustration of
> the sampling bottleneck (rather than a precision-exponent study), since the
> paper's contribution is the GNN architecture. Would you present the biased
> local result alongside the cluster result, or only the cluster values? And is
> an AIC-weighted / corrections-to-scaling fit worth including for L ≤ 128?

---

## Ground rules (from CLAUDE.md — apply throughout)

Physics tests gate production runs. Frozen conventions (χ = V(⟨M²⟩−⟨|M|⟩²), the
k≠0 two-momentum ξ estimator, plaquette/gauge signs). Every paper number from a
committed script + `results/<run_id>.json` provenance. Seeds from config. Boring
numpy/scipy/torch. CPU-first (cluster makes even L=128 laptop-feasible). All work
on `paper-v2`; v1 stays frozen.

### Suggested kickoff prompt for the new session
> Start Phase I journal-v2 work on the `paper-v2` branch — read
> `docs/v2_cluster_fss_plan.md` and `CLAUDE.md` first. Begin with Task V2-1:
> implement and validate the Wolff/Brower–Tamayo φ⁴ cluster sampler
> (`src/qft_graph/mc/cluster.py`), physics-oracle tests before any production run.
> Confirm you're on `paper-v2` (not master) before committing.
