# qft-graph: Implementation Plan — Phase I Corrections + Phase II (Gauge Fields & Fermions)

**Owner:** Josh Brehm · **Repo:** github.com/crosis19/qft-graph (plan aligned to `master` as pushed 2026-03-27)
**Purpose:** This document is the working spec for Claude Code. It covers three workstreams:

- **WS1 — Phase I corrections:** fix the φ⁴ paper's statistical and framing issues, post v1 to arXiv (~1–2 weeks).
- **WS2 — Phase IIa:** quenched compact U(1) gauge theory in 2D — the link-node vs. edge-feature architecture experiment and gauge-invariance measurement (~3–5 weeks).
- **WS3 — Phase IIb:** Schwinger model — GNN surrogate for the fermion determinant, closed with exactness-preserving sampling (multi-month flagship).

> **Status note (2026-07-30):** This plan predates two developments it does not cover.
> (1) WS1 is complete, but after submission arXiv v1 was declined by moderation and the
> arXiv route is **abandoned** (commit bfdc2fb) — publication is journal-only, targeting
> *Machine Learning: Science and Technology* (MLST/IOP). The journal-revision tasks
> (V2-1..V2-3: Wolff/Brower–Tamayo cluster sampler, 7-size FSS to L=128) live in
> **`docs/v2_cluster_fss_plan.md`**, and the venue decision + arXiv postmortem in
> **`docs/v2_review_brief.md`** (the MLST/IOP compliance pass landed as commit 0c3c197,
> "V2-4") — not here.
> (2) WS2 (A-1..A-5) is complete; a standalone Phase IIa paper is drafted in `paper_gauge/`
> and Variant C was ratified as the winning graph variant — see `docs/phase2_decisions.md`.

## 0. Ground rules for Claude Code

1. **Physics tests gate everything.** No training run launches until the exact-value unit tests for that module pass. The tests defined in this doc are the arbiters of sign conventions — if a derivation in this doc conflicts with a passing exact-value test, trust the test and flag the discrepancy.
2. **Conventions in this doc are frozen.** (Gamma matrices, plaquette orientation, gauge-transform signs, χ definition.) Propose changes in a comment; never change silently mid-project.
3. **One task ID per session/PR.** Tasks are labeled P1-x, A-x, B-x below, with explicit definitions of done.
4. **Every figure and table is regenerable** by a committed script + config file. No numbers pasted into the paper by hand without a script that produced them.
5. **Reproducibility:** all seeds from config files; log `(git commit, config hash, seeds, metrics)` to `results/<run_id>.json`. Prefer boring numpy/scipy/torch; no new heavy dependencies without approval.
6. **CPU-first.** Everything here is sized for a laptop (model is ~150k params). GPU is optional acceleration, never a requirement (`device: auto` in `configs/defaults.yaml` — done in commit 176ef10).
7. **Read `ARCHITECTURE.md` before touching model or graph code.** Extend the existing registries (`graphs/node_types.py` `NodeType` enum, `graphs/edge_types.py` helpers, `actions/base.py`, `fields/base.py`) rather than inventing parallel structures. Terminology note: after the terminology cleanup the code says **"action"** (`models/heads/action.py`, `ActionHead`) where the paper says action S_E[φ], but the training-loss key and output dict key remain `energy` (`loss: energy_matching`, `output["energy"]`) — map the two in prose/docstrings, do not mass-rename.

## 1. Repo layout — current state and where new code goes (WS0)

**No restructuring needed.** The repo already has a clean `src/qft_graph` package (actions, fields, graphs, lattice, mc, models, training, analysis, utils), a mirrored pytest suite, OmegaConf-composed configs (`configs/defaults.yaml` + `lattice/`, `mc/`, `model/`, `training/`), entry-point scripts (`generate_mc_data.py`, `train.py`, `evaluate.py`, `sweep.py`), and in-repo LaTeX (`paper/` with `Makefile` and `generate_figures.py`). `fields/gauge.py` and `fields/fermion.py` exist as documented stubs *(2026-07 note: `gauge.py` has since been fully implemented in A-1; `fermion.py` remains a stub pending B-1)*, and `NodeType` already registers `GAUGE` and `FERMION` — Phase II was pre-scaffolded. Follow this map:

| New capability | Home |
|---|---|
| Binned jackknife, Wolff τ_int | extend `src/qft_graph/mc/analysis.py` (jackknife/bootstrap/naive τ_int already there; block-jackknife pattern already in `mc/observables.py::correlation_length_fft_jackknife`) |
| Wilson gauge action | new `src/qft_graph/actions/wilson.py`, subclassing `actions/base.py` |
| U(1) link field + gauge transforms | fill in `src/qft_graph/fields/gauge.py` (its stub comments already spec this) |
| U(1) heatbath sampler | new `src/qft_graph/mc/heatbath.py`, alongside `mc/metropolis.py` |
| Gauge graph variants (A/B/C) | extend `graphs/edge_types.py` (add `("gauge","starts_at","spacetime")`, `("gauge","ends_at","spacetime")` + reverses) and `graphs/builder.py`; reuse `NodeType.GAUGE` |
| Wilson–Dirac operator + determinant labels | new subpackage `src/qft_graph/fermions/` (`dirac.py`, `determinant.py`); graph-side spinor representation in `fields/fermion.py` |
| Delayed-acceptance sampler | new `src/qft_graph/mc/delayed_acceptance.py` |
| All new physics tests | mirrored dirs under `tests/` (e.g. `tests/test_mc/test_heatbath.py`, `tests/test_fermions/`) |

Add a `CLAUDE.md` at repo root containing: the Ground Rules above, `pytest` invocation, the energy↔action terminology note, and a pointer to this plan. Optional tidiness: move `gnn-qft-plan.jsx` and `gnn-qft-proposal.docx` from root into `docs/`. *(Done: `CLAUDE.md` exists at root; both files now live in `docs/`.)*

---

## 2. WS1 — Phase I corrections (paper → arXiv v1)

**Priority: complete before any Phase II training jobs.** (Phase IIa data-gen code, task A-1, may proceed in parallel since it's independent.)

### Task P1-0: Repo ↔ manuscript sync audit (new — do first)

The repo (last push 2026-03-27) lags the manuscript (dated 2026-04-02), and several published results have no visible source in the tree:

1. Commit the manuscript state that produced the current PDF: `paper/main.tex` in the repo predates the April 2 draft, and `paper/figures/` (`energy_prediction`, `finite_size_scaling`, `free_field`, `scaling_collapse`) doesn't match the draft's figure set.
2. **Missing reproducibility pieces — locate on local disk and commit:** (a) no 64×64 lattice config exists (`configs/lattice/` has only 8/16/32, but Table I reports 64×64); (b) no baseline model code (homogeneous GNN, lattice CNN, MLP) or Table II comparison script exists anywhere in the tree; (c) no coupling-generalization eval script (Table III). If any of these lives only in notebooks or uncommitted files, promote to `scripts/` + `configs/`.
3. Reconcile config/paper discrepancies: `configs/defaults.yaml` says `epochs: 200`, `n_configs: 10000`; the paper says 150 epochs, 5000 configs. The actual run configs are the truth — commit them and make the paper's hyperparameter appendix (P1-7) read from them.

**Done when:** a fresh clone can regenerate Tables I–III and Fig. 2 from committed code (documentation of the commands lands in P1-7).

### Task P1-1: Statistics upgrades (extend `mc/analysis.py` — most of this exists)

Already present: unbinned `jackknife_mean_error`, `bootstrap_mean_error`, naive `integrated_autocorrelation_time` (first-negative-crossing), and block-jackknife for ξ in `observables.py`. Add:

- `binned_jackknife(samples, estimator, n_bins)` → (value, error) — general version of the block pattern in `correlation_length_fft_jackknife`, for arbitrary estimators. The existing unbinned jackknife **underestimates errors on autocorrelated MC series**; route ensemble observables through the binned version with bin size ≳ 2τ_int.
- Upgrade `integrated_autocorrelation_time` to Wolff automatic windowing (window W where W ≥ S·τ_int(W), S = 1.5); keep the old estimator for comparison.

**Tests (add to `tests/test_mc/`):** (a) synthetic AR(1) series with known τ_int — recover within 10%; (b) binned jackknife on autocorrelated series reproduces the known standard error within 10%, where the unbinned version demonstrably undershoots.

**Done when:** tests pass; every ensemble observable feeding the paper is rerouted through binned errors.

### Task P1-2: Fix the χ / G̃(0) convention and add error bars to all observables

**Verified in code:** `mc/observables.py::correlation_length_fft` computes `chi = V * (M.pow(2).mean() − M.mean().pow(2))` — the Var(M) form — and `susceptibility_term`'s docstring specifies the same. So the *implementation* matches the paper's Eq. (11), and it is the manuscript's Sec. III.A definition (⟨|M|⟩-subtracted) that never ran. The two differ substantially even in the finite-volume symmetric phase (for Gaussian M they differ by a factor ≈ 1 − 2/π) and in the broken phase Var(M) is contaminated by rare tunneling events. First audit how the ensemble χ in Fig. 2 was actually assembled (`scripts/sweep.py`, `analysis/phase_diagram.py`) — then apply the **frozen convention**, the ⟨|M|⟩-subtracted form, everywhere:

```
chi     = V * ( <M^2> - <|M|>^2 )          # susceptibility AND G(k=0)
G(kmin) = < |phi_tilde(kmin, 0)|^2 > / V   # no subtraction needed: <phi_tilde(k≠0)> = 0
xi      = 1/(2 sin(pi/L)) * sqrt( G(0)/G(kmin) - 1 )
```

Work items:
1. Change the χ lines in `mc/observables.py` (both `correlation_length_fft` and the ensemble assembly in `scripts/sweep.py` / `analysis/phase_diagram.py`) to the frozen convention; align Sec. III.A and Eq. (11) in `paper/main.tex`. As a sensitivity check, compute the ξ/L crossing under *both* conventions and note in the paper whether the extracted m²_c moves outside its error bar.
2. Regenerate Fig. 2 with jackknife error bars, binning set by measured τ_int at each (m², L). Report τ_int values in the paper (table or text).
3. Fit `ln χ_max` vs `ln L` (3 points, with errors) → report γ/ν with uncertainty instead of the eyeballed "consistent with 1.75". Locate peaks by quadratic fit around the maximum, not the raw grid max.
4. ξ/L crossing: report crossing points with errors; in the text, attribute the L=64 drift to *both* corrections-to-scaling and critical slowing down (currently only the latter is claimed).

**Done when:** Fig. 2 regenerated with errors; γ/ν fit value in paper; conventions consistent between code and manuscript.

### Task P1-3: m² and λ as model inputs + multi-coupling training

The current model has no coupling input, so the old Table III conflates distribution shift with an unlearnable label shift. *(2026-07 note: done in P1-3 — coupling conditioning shipped as `ModelConfig.condition_on_couplings` with graph-level `(m², λ)` inputs in `graphs/builder.py` (the "task P1-3" block), plus `scripts/generate_multicoupling_data.py` / `scripts/evaluate_generalization.py`.)*

1. **Model change:** field-node features `[phi_i]` → `[phi_i, m2, lambda]`; also append `(m2, lambda)` to the readout MLP input. Keep it this simple (no global node).
2. **Postmortem first:** inspect the original Table III eval script and determine which m² entered the target action (almost certainly the new one). Write a 3-sentence summary for the paper explaining the old protocol honestly.
3. **Data:** L=16, λ=0.5, m² grid of 13 points from −2.9 to −0.3 (step ≈ 0.2, must include the critical region ≈ −2.45). 3000 configs per point. Thermalization ≥ 2000 sweeps; separation 10 sweeps, **increased near criticality according to measured τ_int** (record actual separations in dataset metadata).
4. **Protocol:** train on alternating grid points; hold out interleaved couplings (interpolation test) and both endpoints (extrapolation test). 5 seeds. Produce **Table III′**: r and relative error, mean ± std over seeds, for each held-out coupling.

**Done when:** Table III′ script committed; old-protocol paragraph drafted for Josh's approval.

### Task P1-4: Seed variance for Tables I & II

Rerun HeteroGNN, homogeneous GNN (with skip), CNN, and MLP at L=16 with 5 seeds each; report mean ± std. Add one row: a **parameter-matched CNN** (~150k params, wider/deeper) to blunt the capacity-mismatch critique. Same protocol for the L=64 HeteroGNN row of Table I.

**Done when:** Tables I & II regenerated with ± columns.

### Task P1-5: Depth ablation (the over-smoothing figure)

For B ∈ {1, 2, 3, 4, 6}: train (a) homogeneous GNN *without* skip, (b) homogeneous *with* skip, (c) HeteroGNN. 3 seeds each, L=16, single coupling. Plot Pearson r vs B (log-scale y near 1 if needed; show the collapse of (a) explicitly). This becomes a new figure and armors the paper's central empirical claim.

**Done when:** figure script committed; r < 0.05 collapse of variant (a) is either reproduced and documented, or the discrepancy vs. the manuscript's claim is flagged to Josh.

### Task P1-6: Size-transfer experiment (bounded scope)

Train at L=16 only; evaluate at L ∈ {8, 32, 64} without retraining.

Caveat to test: absolute coordinate features (x1, x2) may not transfer across L. Run two variants: (i) as-is; (ii) **coordinate-free** — replace spacetime node features with a constant, geometry carried entirely by displacement edge features (bonus: restores translation invariance). Report both. If transfer works, it's a headline paragraph; if not, one honest paragraph. Do not iterate beyond these two variants.

**Done when:** one table, two variants, three eval sizes, 3 seeds.

### Task P1-7: Manuscript edits

1. **FSS attribution fix (the blocking issue).** Sections IV.D–E use Monte Carlo configurations only; the abstract and conclusion currently credit the model. Reword abstract along the lines of: *"The model achieves near-perfect action prediction (r > 0.999) across lattice sizes and coupling values; the underlying Monte Carlo pipeline reproduces Ising-class finite-size scaling — susceptibility peak growth, order-parameter S-curves, and correlation-length crossings consistent with ν = 1 — validating the training distributions."* Retitle IV.D–E framing as validation of the data-generation pipeline. Claude Code drafts; **Josh approves all abstract/conclusion wording before commit.**
2. Fix "8×8 to 64×64" claim: either add 8×8 and 32×32 rows to Table I (data exists) or reword to match table coverage.
3. Soften fiber-bundle wording: field nodes are points in fibers; the collection is a *discretized section*.
4. Clarify explicitly whether one model per lattice size was trained (per Sec. III.B, yes) and integrate the P1-6 transfer result.
5. Hyperparameter appendix: pull actual LR, weight decay, schedule, MLP depths/widths, parameter counts, hardware, wall time **from the committed run configs** (after P1-0 resolves the `defaults.yaml` 200-epoch/10k-config vs. paper 150-epoch/5k-config discrepancy), not from memory.
6. References to add: Cranmer–Kanwar–Racanière–Rezende–Shanahan ML-sampling review (Nature Reviews Physics, 2023); Apte et al., Phys. Rev. B 110, 165133 (2024); arXiv:2604.20797 (gauge-equivariant GNNs for lattice gauge theories, 2026) — position Phase II roadmap against it (exact equivariance vs. architectural flexibility, cf. Sec. V.B); Schlichtkrull et al. (R-GCN) and Hu et al. (HGT) for heterogeneous-GNN context; PyTorch Geometric; Schaich & Loinaz for a 2D φ⁴ critical-coupling comparison — **verify normalization convention mapping before quoting numbers** (their λ normalization differs; if the mapping is ambiguous, cite qualitatively).
7. README: exact command per figure/table. Dependencies live in `pyproject.toml` (torch ≥2.1, torch-geometric ≥2.4, omegaconf, scipy, etc.) — pin the exact versions used for the paper runs (`pip freeze` → a committed lockfile or exact-version extras), and note that `scipy` (already a dependency) covers everything Phase II needs (`special.iv`, `linalg.eig`, sparse).

**Done when:** paper builds; every number traces to a script; Josh has signed off on reworded claims; arXiv tarball ready (hep-lat, cross-list cs.LG).

### WS1 exit criteria

All P1 tasks done → tag `paper-v1`, submit to arXiv. Phase IIa training may then begin. *(Outcome, 2026-07: tag `paper-v1` exists and v1 was submitted, but arXiv declined it via moderation and the arXiv route is abandoned (commit bfdc2fb); Phase I publication continues as a journal submission (MLST) — see `docs/v2_cluster_fss_plan.md` and `docs/v2_review_brief.md`.)*

---

## 3. WS2 — Phase IIa: Quenched compact U(1) in 2D

**Scientific goals:** (1) demonstrate the extensibility claim by adding a gauge field as a genuinely new node type; (2) resolve the link-node vs. edge-feature design tension already present in the Phase I paper (Sec. V.A says "separate node type," the metric-placement discussion says links "share the edge representation" — these are different architectures; the ablation between them IS the paper content); (3) quantitatively measure *learned* gauge invariance — the direct engagement with the exact-equivariance school.

### 3.1 Frozen conventions

- Links: `theta[mu, x] ∈ (−π, π]`, `U = exp(i·theta)`, `mu ∈ {1, 2}`, periodic BCs, lattice L×L.
- Plaquette (single orientation in 2D):
  `theta_P(x) = theta_1(x) + theta_2(x + e1) − theta_1(x + e2) − theta_2(x)`
- Wilson action: `S = beta * sum_x [ 1 − cos(theta_P(x)) ]`
- Gauge transform: `theta_mu(x) → theta_mu(x) + alpha(x) − alpha(x + e_mu)` with arbitrary `alpha(x)`.
- Topological charge: `Q = (1/2π) * sum_x wrap(theta_P(x))` where `wrap` maps to (−π, π]. Q is an exact integer on the torus.
- Wilson loop `W(R,T)`: `cos` of the oriented sum of link angles around an R×T rectangle.

### 3.2 Exact validation oracles (implement in `tests/`)

1. **Torus-exact plaquette** via character expansion — this is the primary sampler test:
   ```
   Z            = sum_n  I_n(beta)^(L^2)
   <cos theta_P> = sum_n I_n(beta)^(L^2 − 1) * (I_{n−1}(beta) + I_{n+1}(beta))/2  /  Z
   ```
   Sum n from −20 to 20 using `scipy.special.iv`; normalize by I_0^(L²) internally to avoid overflow/underflow (work with ratios `I_n/I_0`). Sanity anchor: infinite-volume limit is `I_1(beta)/I_0(beta)` ≈ 0.4464 at β=1.
2. **Wilson loop area law** (infinite-volume; valid for RT ≪ L²): `<W(R,T)> = (I_1/I_0)^(R·T)`; string tension `sigma(beta) = −ln(I_1/I_0)`.
3. Q integer to 1e−10 on every sampled config.
4. Action, Q, and all Wilson loops invariant under 100 random gauge transforms to 1e−10.

### Task A-1: U(1) heatbath sampler — *may start immediately, independent of WS1*

Files: `src/qft_graph/mc/heatbath.py` (sampler, following the `mc/metropolis.py` + `mc/sampler.py` patterns), `src/qft_graph/actions/wilson.py` (subclass `actions/base.py`), fill in `src/qft_graph/fields/gauge.py` per its own stub spec (link variables, plaquettes, gauge transforms). New config group `configs/mc/heatbath_u1.yaml`.

Single-link conditional is exactly von Mises. For link `theta_1(x)` (μ=1; μ=2 analogous by symmetry):

- It appears with coefficient +1 in `theta_P(x)` and −1 in `theta_P(x − e2)`. Write the two plaquette angles as `theta + a` and `−(theta − b)` respectively, where
  ```
  a = theta_2(x + e1) − theta_1(x + e2) − theta_2(x)
  b = theta_1(x − e2) + theta_2(x − e2 + e1) − theta_2(x − e2)
  ```
- Local action: `S_link(theta) = −beta [ cos(theta + a) + cos(theta − b) ] + const = −beta |z| cos(theta + arg z)` with `z = exp(i·a) + exp(−i·b)`.
- Therefore sample directly: `theta ~ VonMises(mu = −arg(z), kappa = beta * |z|)` via `np.random.vonmises`. No tuning, no accept/reject, low autocorrelation.
- If the derived signs of a/b prove wrong, the torus-exact plaquette test is the arbiter — fix signs until it passes; do not "fix" the test.

Implementation notes: sweep = update all 2L² links. Sequential loops are acceptable at these sizes (numba `@njit` if slow); vectorized checkerboard-by-(direction, parity) masking is an optional later optimization, correctness first.

**Data generation:** β ∈ {0.5, 1.0, 2.0, 3.0, 4.0}; L ∈ {8, 16, 32}; 4000 configs per (β, L); thermalization 1000 sweeps; separation 5 sweeps (heatbath decorrelates fast — verify by measuring τ_int(plaquette) and record it). Storage: HDF5, `float32 theta[N_cfg, 2, L, L]`, attrs: beta, L, seed, separation, thermalization.

**Done when:** all four oracle tests pass at every (β, L); τ_int table written to results.

### Task A-2: Precompute labels

For each config: total action; `−ln W(R,T)` for loops {(1,1), (2,2), (2,4), (3,3), (4,4)} (store the raw signed W too — at large area/small β, per-config W fluctuates around a tiny mean and can be negative; the trainable target is per-config `W`, and `−ln<W>` is only for ensemble-level validation); Q. Store alongside configs in the HDF5.

### Task A-3: Graph builders — the core experiment

Files: extend `graphs/edge_types.py` (add `("gauge", "starts_at", "spacetime")`, `("gauge", "ends_at", "spacetime")` and reverses — note these intentionally do NOT reuse the site-field `inhabits_edge()` helper, since a link attaches to *two* sites) and `graphs/builder.py`; reuse the already-registered `NodeType.GAUGE`. Three variants, identical training budgets. β is appended as a scalar input feature in all variants (Phase I's m² lesson).

**Variant A — link-nodes (primary; consistent with the paper's fields-as-nodes thesis):**
- Node types: `spacetime` (L² nodes, features per P1-6 winner: coordinates or constant) and `link` (2L² nodes, features `[cos theta, sin theta, onehot(mu), beta]`).
- Edges: `(link, starts_at, st)` to x; `(link, ends_at, st)` to x+e_mu; reverses `(st, origin_of, link)`, `(st, target_of, link)`; keep `(st, adjacent, st)` with displacement features.
- Message-passing block (three-stage analog): link→st (both typed edges), st→st, st→link. Residual + LayerNorm per stage, as in Phase I.
- Note the geometry: a plaquette is a length-4 link–st alternating cycle, so plaquette information becomes available within 2 blocks.

**Variant B — links as edge features:** put `[cos theta, sin theta]` on the existing st–st adjacency edges alongside displacement. Reversed edge carries the inverse link: `[cos theta, −sin theta]`. Spacetime nodes only.

**Variant C — invariant-input oracle (control):** spacetime node features ← `[cos theta_P(x), sin theta_P(x), beta]` (one plaquette per site in 2D). This is the ceiling: how well can the model do when invariance is handed to it? A/B vs C measures the *cost of learning* invariance.

**Done when:** all three builders produce `HeteroData` passing shape/round-trip tests; a gauge transform of the input configs leaves Variant C inputs bit-identical (its own unit test).

### Task A-4: Training + the receptive-field study

- Heads: (i) total action = sum over st nodes of per-site MLP (same pattern as Phase I Eq. 8); (ii) Wilson loops: graph-level readout (mean-pool over both node types → MLP), one head per loop size, target per-config W (or −ln W where W > 0 uniformly; decide per β and document); (iii) Q: regression head + report exact-integer accuracy after rounding.
- Protocol: AdamW, cosine, 150 epochs, batch 32, H=64 — mirror Phase I. 5 seeds for headline numbers.
- **Receptive-field study:** B ∈ {2, 3, 4, 6} × loop areas {1, 4, 8, 9, 16} → heatmap of prediction r vs (area, B). This turns Phase I's Wilsonian/receptive-field *interpretation* into a measured result. Expect degradation once loop diameter exceeds ~B lattice spacings; that expected failure is a finding, not a bug.

**Done when:** A/B/C comparison table (action r, loop r by size, Q accuracy, params) with seed errors; heatmap figure script.

### Task A-5: Gauge-invariance measurement (the centerpiece)

- For each eval config, generate K=32 random gauge copies. Define per-target invariance error:
  `eps_gauge = mean_configs[ std_over_gauge_copies(y_hat) ] / std_over_configs(y_hat)`
- Report eps_gauge for Variants A, B, C (C should be ~0 by construction — sanity check).
- **Augmentation experiment:** retrain A and B with on-the-fly random gauge transforms as data augmentation. Plot eps_gauge and test accuracy vs training set size, with/without augmentation. Question being answered: does augmentation drive learned invariance to the Variant-C ceiling cheaply, supporting the flexibility-over-rigidity argument of Phase I Sec. V.B?

**Done when:** eps_gauge table + augmentation curves; half-page summary written for the paper.

### WS2 exit criteria / deliverables

Sampler oracles green; A vs B vs C decided with evidence (this choice feeds WS3); receptive-field heatmap; gauge-invariance study. Output: either a short standalone paper ("learned vs. exact gauge invariance in heterogeneous GNNs for lattice gauge theory") or Section 1 of the Schwinger paper — Josh decides after seeing results. *(Decided 2026-07-14: standalone paper — first draft in `paper_gauge/`; Variant C ratified as the winning variant. See `docs/phase2_decisions.md`.)*
---

## 4. WS3 — Phase IIb: Schwinger model (flagship)

**Scientific goal:** a GNN surrogate for `log|det D[U]|` — the first target in this project that is genuinely *expensive* — validated end-to-end inside an exactness-preserving sampler. This permanently answers "why learn something you can compute?"

### 4.1 Frozen conventions

- 2D Euclidean gammas: `gamma_1 = sigma_x`, `gamma_2 = sigma_y`, `gamma_5 = sigma_z` (Pauli matrices). Check `{gamma_mu, gamma_nu} = 2 delta 1` in a test.
- Wilson–Dirac operator, r = 1, on an L×L lattice, U(1) links from WS2, **antiperiodic fermion BC in the time direction (mu=2)** implemented by flipping the sign of U_2 on the top time slice *inside D only* (gauge configs untouched):
  ```
  D(x,y) = (m + 2)·1_2 · delta_{x,y}
           − (1/2) sum_mu [ (1 − gamma_mu) U_mu(x)      delta_{y, x+e_mu}
                          + (1 + gamma_mu) conj(U_mu(y)) delta_{y, x−e_mu} ]
  ```
  Sparse complex128, dimension 2L² (2-component spinors).
- Key structural fact — **the mass shift trick**: `D(m) = D(0) + m·1`. Therefore one eigendecomposition of D(0) per config yields the determinant at *every* mass:
  `log|det D(m)| = sum_i log|lambda_i + m|`, lambda_i = eigenvalues of D(0) (non-Hermitian; `scipy.linalg.eig` on the dense matrix).
- Labels to store per (config, m): `sign(det)` conceptually and `log|det D|`; also `logdet(D†D) = 2·log|det D|` for the N_f = 2 framing.

### 4.2 Exactness tests (write before any ensemble labeling)

1. **Free field.** For U ≡ 1, momentum space gives the closed form (k1 = 2πn/L; k2 = 2π(n+½)/L antiperiodic):
   ```
   log|det D| = sum_k log[ ( m + sum_mu (1 − cos k_mu) )^2 + sum_mu sin^2 k_mu ]
   ```
   Must match dense `slogdet` AND the eigenvalue-trick result to 1e−8, for m ∈ {0.1, 0.5, 1.0}, L ∈ {4, 8}.
2. **gamma5-hermiticity:** `gamma5 D gamma5 == D†` to 1e−12 on random gauge configs.
3. **Gauge invariance of the determinant:** under a random gauge transform of the links, `log|det D|` unchanged to 1e−8 (D → G D G†, G unitary).
4. **Trick vs. LU cross-check:** eigenvalue-based log|det| vs `numpy.linalg.slogdet` on 20 random configs at L=16 to 1e−6; flag any config where they disagree (conditioning) and fall back to slogdet for those.

### Task B-1: Dirac operator + label pipeline

Files: new subpackage `src/qft_graph/fermions/` with `dirac.py` (operator construction) and `determinant.py` (eig/slogdet label pipeline); fill in `src/qft_graph/fields/fermion.py` for the graph-side representation (its stub comments already list antiperiodic BCs and the determinant). Tests in `tests/test_fermions/`.

- Sparse construction; dense conversion for eig/slogdet at L ≤ 32 (2048×2048 at L=32 — seconds per config on CPU).
- Label the quenched ensembles from WS2: β ∈ {1, 2, 3}; mass grid m ∈ {0.1, 0.2, 0.3, 0.5, 0.7, 1.0} (thanks to the mass-shift trick, the whole grid costs one eig per config). **Do not go below m = 0.1 initially** — near-zero modes produce heavy-tailed log-det spikes; document the label distribution per (β, m) with histograms before training.
- Volume plan: L ∈ {8, 16} fully labeled (fast); L = 32: 1500 configs per β (est. ~15–40 s/config for `eig` on laptop CPU → budget accordingly, run overnight); L = 64 deferred (needs sparse/stochastic methods — out of scope for v1).

**Done when:** all four exactness tests green; labeled HDF5 datasets with per-(β,m) label histograms in results.

### Task B-2: Determinant surrogate

- Graph: **winning variant from A-3/A-5** + global scalars `(beta, m)` appended to node features.
- Target: `log|det D| / L²` (per-site normalization keeps the scale sane across volumes). Readout: sum of per-st-node MLP outputs × (1/L² handled consistently) — physics rationale to note in the paper: `Tr log D` has a hopping (loop) expansion, so a quasi-local decomposition is the right inductive bias; expect the truncation of that expansion to show up exactly like the Phase I receptive-field story.
- Training: joint across (β, m); holdouts: (i) interleaved m values (interpolation), (ii) smallest m (hard extrapolation — expect and document degradation), (iii) one held-out β.
- Metrics: Pearson r AND MAE in units of the per-(β,m) *physical* std of log|det| (r alone is inflated by the trivial volume/mass dependence — the honest metric is error relative to config-to-config fluctuations at fixed (β, m)).
- 5 seeds for headline numbers; error-vs-m and error-vs-β figures.

**Done when:** metric table + figures; a written half-page on where the surrogate is/isn't trustworthy.

### Task B-3: Exactness-preserving sampling — delayed acceptance

File: `src/qft_graph/mc/delayed_acceptance.py`. Two-level Metropolis (Christen–Fox delayed acceptance; exact regardless of surrogate quality):

- Target: `pi(U) ∝ exp(−S_gauge[U]) · |det D(m)|^{N_f}` with N_f = 2 via `exp(−S_g + N_f·log|det D|)`.
- **Inner level:** a sub-sweep of single-link Metropolis updates accepted with the *surrogate* effective action difference (gauge part exact + surrogate Δlog|det|). Heatbath is NOT valid here (the det term breaks the von Mises form) — plain Metropolis with tuned proposal width.
- **Outer level:** treat the entire inner sub-sweep `U → U'` as one composite proposal; accept with
  `min(1, [pi_exact(U') · pi_surr(U)] / [pi_exact(U) · pi_surr(U')])`
  which needs ONE exact log|det| per outer step (eig or slogdet).
- Knob: inner sub-sweep length trades outer acceptance vs. exact-eval savings. Start with 1 full sweep per outer step; report outer acceptance and tune.
- **Ground truth:** brute-force Metropolis with exact log|det| at every step, L = 8 (cheap) and a short L = 16 run.
- **Validation observables:** average plaquette (dynamical vs quenched shift), chiral condensate `<psibar psi> = Re Tr D^{-1} / (2 L²)` via exact inverse at L ≤ 16. DA ensemble must match ground truth within jackknife errors.
- **Headline deliverable:** exact-determinant evaluations per effective independent sample, DA vs brute force (i.e., the measured speedup), at matched statistical quality.

**Done when:** DA-vs-ground-truth observable table; speedup number with error; short methods writeup.

### Task B-4 (stretch, explicitly optional): Spectroscopy

Pion correlator `C(t) = sum_x <pi(x,t) pi(0,0)>` from point-source propagators (D solve, L ≤ 16); effective mass plateau; compare qualitative trend toward the known continuum mass gap g/√π as m → 0. Do not start until B-3 is done.

---

## 5. Sequencing, estimates, decisions

### Dependency order

```
P1-0                           (first;  repo/manuscript sync — everything else edits these files)
P1-1 → P1-2                    (days;   stats first, then observables)
P1-3, P1-4, P1-5, P1-6         (parallelizable; ~1 week of runs, mostly unattended)
P1-7 → arXiv v1                (writing; Josh approves claims)
A-1, A-2                       (independent — may run in parallel with all P1 tasks)
A-3 → A-4 → A-5                (after arXiv v1 is posted)
B-1                            (after A-1; tests are self-contained)
B-2                            (after A-5 picks the winning variant)
B-3 → B-4                      (last)
```

### Rough effort

| Block | Wall time | Notes |
|---|---|---|
| WS1 total | 1–2 weeks | Long pole: P1-3 data gen near criticality |
| A-1..A-2 | 3–5 days | Von Mises heatbath is simple; oracles do the hard QA |
| A-3..A-5 | 2–4 weeks | Three variants × targets × seeds; runs are small |
| B-1 | ~1 week | Incl. overnight L=32 labeling |
| B-2 | 1–2 weeks | |
| B-3 | 2–4 weeks | Tuning the inner/outer balance is the research part |

### Open decisions for Josh (flag, don't decide silently)

*(Resolutions are recorded in `docs/phase2_decisions.md`. As of 2026-07: decision 2 resolved — standalone Phase IIa paper, drafted in `paper_gauge/`; decision 4 resolved — coordinate-free features adopted everywhere in Phase II. Decisions 1 and 3 remain open.)*

1. N_f = 2 (positive measure, cleanest) vs N_f = 1 (matches the famous g/√π result) as the *paper's* framing — plan assumes N_f = 2 for sampling, N_f = 1 only if B-4 happens.
2. IIa as standalone short paper vs. first section of the Schwinger paper.
3. Whether Phase I v2 (post-referee or post-IIa) absorbs the delayed-acceptance idea or it stays exclusive to the IIb paper (plan assumes the latter).
4. Coordinate features: if P1-6's coordinate-free variant wins, adopt it everywhere in Phase II.

### Standing guardrails (repeat of §0, for emphasis)

Physics tests gate training. Conventions frozen. One task per session. Every number has a script. Boring dependencies. Log everything. CPU-first.
