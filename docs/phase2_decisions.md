# Phase II decision record

Decisions from the plan's §5 "Open decisions for Josh," resolved as they are
made. Sessions working on A-x/B-x tasks: read this before starting.

## Decided

1. **Coordinate-free spacetime features adopted everywhere in Phase II**
   (Josh, 2026-07-09; plan §5 decision 4). Spacetime nodes carry a constant
   feature; all geometry lives in displacement edge features
   (`HeteroGraphBuilder(spacetime_features="constant")`,
   `ModelConfig.spacetime_features="constant"`). Evidence: P1-6 size
   transfer — both variants reach r = 1.0000 at every eval size, the
   coordinate-free variant has slightly lower relative error at all sizes,
   and it restores exact translation invariance
   (results/size_transfer.json; paper Sec. IV.F). Consequence for A-3: all
   three gauge graph variants use constant spacetime node features.

2. **Variant B carries beta on spacetime node features `[1, beta]`**
   (Josh, 2026-07-11). Rationale: mirrors Variant C (plan puts beta on
   st nodes there), so B-vs-C isolates where the gauge information
   lives; matches the Phase I couplings-on-nodes precedent; keeps
   adjacency edge features purely displacement + parallel transport.
   All variants additionally expose `data.globals = [[beta]]`.

3. **Wilson-loop training target is raw per-config W, uniformly across
   all (beta, size) cells** (Josh, 2026-07-11). Evidence: A-2 label
   scan — `-ln W` is admissible for every config only on a ragged,
   beta- and volume-dependent subset (at L=16, frac(W<=0) reaches ~0.5
   for large-area/small-beta cells; even 4x4 at beta=4 has 13% W<=0),
   and d(-ln W)/dW = -1/W makes near-zero cells heavy-tailed. Raw W
   keeps the receptive-field heatmap's r values comparable across
   cells. Per-(beta, size) z-scoring of targets is allowed for
   training stability (invertible, documented, leaves r unchanged);
   -ln<W> is used only at ensemble level against the exact area law.

4. **A-4 comparison runs in two arms: fixed H AND parameter-matched**
   (Josh, 2026-07-11). At fixed H, Variant A carries ~2.5x the params of
   B/C (296k vs 118k at the protocol H=64, full heads), so the A/B/C
   table gets both arms. Knob: `models/u1_gnn.matched_hidden_dim(
   target_params, config, variant, **model_kwargs)` — binary search over
   constructed models; model_kwargs must match the trained heads. At the
   protocol point: A@64 budget -> B/C at H=102 (+0.07%/+0.04%); B@64
   budget -> A at H=40 (-0.79%), C at H=64 (-0.05%). Which budget anchors
   the headline table is decided when results are in (down-matching A to
   ~118k also keeps the laptop-scale ethos).

5. **Protocol v2: ALL A-4 targets standardized; full rerun for
   uniformity** (Josh, 2026-07-12). Q was the only target trained on
   its natural integer scale; its variance grows with volume and the
   resulting loss imbalance caused Q-head collapse and seed instability
   at L=32 (all beta) and at beta=4 with B>=4 blocks (E4). Fix:
   `standardize_scalar_targets` in `graphs/u1_dataset.py` z-scores
   action and Q like the Wilson targets (train stats reused on
   val/test); exact-integer Q accuracy is computed after de-scaling.
   Run records carry `protocol_version: 2`. All 114 v1 runs archived in
   `results/a4_protocol_v1/` (never mix v1/v2 numbers); the full matrix
   is rerun under v2 via notebook 08. Sanity anchor: at L<=16, B<=3 the
   natural Q std is ~1, so v1 and v2 should agree closely there.

   **Outcome correction (2026-07-13, after the v2 rerun):** the original
   diagnosis was wrong about v2 curing the anomalies; v2 remains the
   adopted protocol, but the two phenomena have different roots.
   (a) Exact-integer Q accuracy obeys a rounding floor,
   acc ~ P(|N(0, sigma_Q*sqrt(1-q_r^2))| < 0.5) — verified
   quantitatively in both protocols — so it hardens with volume
   (sigma_Q grows with L) and NO loss weighting fixes it; v1's
   natural-scale Q loss was an accidental sigma_Q^2 upweighting that
   bought higher q_r at L=32 at the cost of instability elsewhere.
   Report q_r alongside q_acc with the floor formula as context.
   (b) The beta=4 B>=4 failures are protocol-independent: the action
   trains fine (r ~ 0.93-0.99) while global/topological readouts
   collapse and W4x4 INVERTS with depth (0.76/0.91/0.62/0.25 for
   B=2/3/4/6) — vs monotone receptive-field gains at beta=2. The
   ordered phase halves the input feature spread (std cos theta_P:
   0.40 -> 0.19). **Over-smoothing CONFIRMED (2026-07-13,
   scripts/diagnose_oversmoothing.py, figures/a4_oversmoothing.pdf):**
   at beta=4 B=6 the spacetime embeddings collapse (mean pairwise
   cosine 0.96-1.00, two seeds at exactly 1.0) while beta=2 stays
   healthy (~0.5-0.65) at every depth; at the seed-scattered beta=4
   B=4 cell, per-seed collapse predicts per-seed failure (cos 0.88 ->
   W4x4 r 0.26 vs cos 0.73/0.78 -> r 0.83/0.77). Refinement: the
   collapse is TRAINED-IN — it is already present at the encoder
   output of the deep beta=4 models — i.e., a training-dynamics
   degeneracy the deep stacks fall into when input feature spread is
   halved, not pure forward diffusion. Phase I's residual+LayerNorm
   mitigations hold at beta=2 but not beta=4. Consistent detail: with
   fully collapsed (identical) node embeddings the pooled readout
   still carries graph-mean information, which is why the action
   (a bulk average) survives at B=6 while exact-integer Q and large
   Wilson loops (needing per-node structure / rare-event precision)
   fail. Treated as a finding in the heatmap, not an exclusion.

7. **Phase IIa is written up as a standalone short paper** (Josh,
   2026-07-14; resolves open decision 7 — "for now", may still be folded
   into the Schwinger paper as its Sec. 1 later). Working title per plan
   §3 WS2 exit: "learned vs. exact gauge invariance in heterogeneous
   GNNs for lattice gauge theory". Seed prose: docs/a5_summary.md;
   evidence base: results/a4_table.json, results/a5_table.json,
   figures/a4_receptive_field.*, figures/a5_augmentation.*.
   **First draft committed 2026-07-14** (paper_gauge/, 7 pp revtex4-2;
   Josh authorized drafting in the A-5 session). Tables + every inline
   number generated by scripts/make_gauge_paper_tables.py; draft passed
   a 27-agent adversarial audit (19 confirmed defects fixed, incl. a
   Bachtis2021 bib error inherited from paper/references.bib — fixed in
   both). Open TODOs in the draft: repository URL, Phase I arXiv id,
   commit the pilot MLP-probe script before quoting its exact numbers.
   **[Update 2026-07-29: the "Phase I arXiv id" TODO is moot — the arXiv
   route was abandoned 2026-07-22 (commit bfdc2fb; Phase I goes
   journal-only to MLST, docs/v2_review_brief.md), so the paper_gauge
   self-citation (references.bib, note "arXiv preprint, submission
   pending") should cite the MLST submission/DOI instead. The
   repository-URL and MLP-probe-script TODOs remain open.]**

9. **Variant C (invariant_oracle) is the WS2 winning graph variant**
   (Josh, 2026-07-14, ratifying the A-4/A-5 evidence: A/B at chance
   with eps_gauge ~ 1 at every scale, unfixable by augmentation; C at
   ceiling with eps exactly 0). B-2 builds on Variant C graphs +
   global (beta, m) features per plan §4.

## Still open (do not decide silently — flag to Josh)

6. N_f = 2 vs N_f = 1 framing for the Schwinger paper (plan assumes N_f = 2
   for sampling; decide at WS3).
8. Whether Phase I v2 absorbs the delayed-acceptance idea or it stays
   exclusive to the IIb paper (plan assumes the latter).
   **[Resolved de facto 2026-07-18: the Josh-approved Phase I v2 scope
   (docs/v2_cluster_fss_plan.md) is cluster sampler + precision FSS
   only, and v2 executed to completion on that scope (V2-1..V2-4,
   commits d5e071e..0c3c197) with no delayed-acceptance content — it
   stays with the IIb paper, as the plan assumed. Move to "Decided"
   once Josh confirms.]**

## Status pointers (as of 2026-07-11; later entries dated inline)

- WS1 complete; `paper-v1` tag pushed; arXiv submission pending hep-lat
  endorsement (requested from D. Schaich). **[Update 2026-07-29:
  superseded — v1 was submitted to arXiv and declined by moderation
  (ticket MOD-97144); the arXiv route is abandoned (2026-07-22, commit
  bfdc2fb) — journal publication only. The journal version (v2:
  Wolff/Brower–Tamayo cluster FSS, gamma/nu = 1.732(8)) lives on branch
  `paper-v2`, targeting *Machine Learning: Science and Technology*
  (IOP); see docs/v2_review_brief.md and docs/v2_cluster_fss_plan.md.]**
- A-1/A-2 complete: labeled U(1) ensembles in `data/u1_configs/*.h5`
  (theta[N,2,L,L] float32 + action/q/wilson datasets; validation tables in
  `results/u1_heatbath_data.json`, `results/u1_labels.json`).
- A-3 complete: `U1GaugeGraphBuilder` in `graphs/builder.py` with variants
  "link_nodes"/"edge_features"/"invariant_oracle" (aliases A/B/C); oracle
  tests in `tests/test_graphs/test_gauge_builder.py` incl. bit-identical
  Variant C gauge invariance (builder computes trig in float64, emits
  float32 — gauge copies must be generated in float64 from stored links,
  never re-rounded to float32 before build()). Implementation choices
  within A-3 scope: Variant B carries beta on spacetime node features
  `[1, beta]`; all variants expose `data.globals = [[beta]]` (P1-3
  pattern). The plan's Variant A message-passing-block bullet
  (link→st, st→st, st→link) is model work — lands with **A-4**, since
  `models/message_passing/stage.py` only routes inhabits edges today.
- A-4 plumbing complete (`models/u1_gnn.py`, `graphs/u1_dataset.py`,
  `scripts/train_u1.py`); training CLI is the Colab entry point.
- **A-4 pilot finding (L=8, beta=2, seed 0, frozen protocol,
  results/u1pilot_*.json):** Variant C reaches action r=0.9994,
  Q acc 1.00, W r degrading with loop size (1x1: 0.998 -> 4x4: 0.47 —
  the receptive-field story, visible already). Variants A and B sit at
  exact chance (r ~ 0) on ALL targets for all 150 epochs. Probes: (i)
  single-link/action correlations consistent with exactly zero (gauge
  symmetry — no first-order gradient signal); (ii) a plain MLP on all
  raw link trig features memorizes train (r=0.999) with zero test
  generalization (r=-0.03), while the same MLP on plaquette features
  reaches r=0.97 — representation hardness, not a GNN bug. Colab
  protocol implications need Josh (see session notes / pilot report).
- **A-4 COMPLETE** (2026-07-13): full protocol-v2 matrix (114 runs,
  notebook 08) in `results/a4*.json`; deliverables per plan done-when:
  `scripts/make_a4_table.py` -> `results/a4_table.json` (19 rows,
  LaTeX), `scripts/plot_a4_heatmap.py` ->
  `figures/a4_receptive_field.{pdf,png}`. Headline results: (i) A/B
  null on all gauge-invariant targets — robust across seeds, beta,
  volume, four escape hatches, and parameter matching (E5: B at the A
  budget and A at the B budget both at chance) — vs C at ceiling;
  learned-vs-given invariance is binary at this scale, except a small
  seed-consistent r ~ 0.08 at beta=1 (both variants, both protocols —
  curriculum lead). (ii) Receptive-field gains at beta=2 vs
  ordered-phase depth degradation at beta=4 (over-smoothing
  hypothesis; diagnostic task spawned on saved checkpoints).
  (iii) Q rounding floor (decision 5 outcome note). v2 checkpoints in
  `experiments/runs/u1/` are the A-5 evaluation inputs.
- **A-5 part 1 COMPLETE (2026-07-13): eps_gauge on the A-4 v2
  checkpoints.** Protocol: K=32 gauge copies per test-split config,
  per-config seeded rng [gauge_seed=20260713, file_config_idx] (copies
  identical across all models on a file — paired comparison), copies
  generated in float64 from float32 storage (A-3 convention), eps on the
  standardized output scale (affine-invariant). Pipeline:
  `scripts/measure_gauge_invariance.py` -> `results/a5eps_*.json`
  (46 runs) -> `scripts/make_a5_table.py` -> `results/a5_table.json`.
  Headline: **A and B have eps_gauge ~ 1.00 on every target** (0.96-1.05
  across all cells, seeds, both parameter budgets, and all four escape
  hatches) — their predictions vary as much along a gauge orbit as
  across physical configs, i.e. zero gauge-invariant content, a sharper
  null than r ~ 0; **C is exactly 0 (bitwise) everywhere** — the sanity
  anchor holds. Detail: at beta=1 (the cell with the small r ~ 0.08
  signal) eps dips seed-consistently a few % below 1 (A 0.96-0.99,
  B 0.97) — the size expected from a small invariant component.
  Robustness: eps stable to ~1-2% under an alternative gauge seed and
  under K=16.
- **A-5 numerics caveat (load-bearing, test-pinned):** batched PyG
  evaluation is position-in-batch dependent at float32 lsb — bit-identical
  graphs at different positions of one Batch return outputs differing by
  ~1e-8, which would put a spurious floor under C's exact zero. The eps
  protocol therefore forwards every graph UNBATCHED (batch size 1).
- **A-5 COMPLETE (2026-07-13): augmentation experiment (part 2) run and
  measured.** Notebook 09 F1a-c + F2 (102 runs: `a5aug`/`a5base`/`a5C`,
  n_train {50..3200} x 3 seeds at L=8 beta=2 + L=16 aug check; full-size
  no-aug arms reuse a4null/a4C). Outcome — the plan's augmentation
  question gets a clean NO at this scale: (i) test r stays at chance
  (|r| <~ 0.06) for A/B at EVERY n_train, augmented or not; (ii)
  eps_gauge stays ~ 1.00 at every n_train (and at L=16) — augmentation
  never produces an invariant component; (iii) what training does
  instead is shrink predictions toward the constant predictor
  (augmented-A action-head config std 1.4e-3 at n=50 -> 1.1e-6 at
  n=3200; unaugmented 2.5e-3 -> 1.6e-6), with the residual fluctuation
  remaining pure gauge noise; (iv) oracle C reaches action r=0.87(6)
  from 50 configs, ceiling by n~200-400 (W4x4 receptive-field-limited
  at r~0.48, B=3). Reading: single links are first-order uninformative
  under the gauge group, so the augmentation-consistency gradient
  vanishes exactly like the supervised one — invariance must be built
  in; this SHARPENS Phase I Sec. V.B flexibility-over-rigidity
  (flexibility suffices for translation, fails for gauge). Deliverables:
  `results/a5_table.json`, `figures/a5_augmentation.{pdf,png}`
  (`scripts/plot_a5_curves.py`), half-page summary `docs/a5_summary.md`.
  Consequence for B-2: **Variant C (invariant inputs) is the winning
  graph variant** feeding WS3.
- Training runs on Colab via the notebook-08/09 pattern (CLI scripts +
  Drive sync); MC and eps_gauge measurement stay on laptop CPU (Colab
  F3 and laptop cooperated via the measurement script's skip logic —
  both mount the same Drive tree).
