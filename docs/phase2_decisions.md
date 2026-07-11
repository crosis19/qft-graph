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

## Still open (do not decide silently — flag to Josh)

2. N_f = 2 vs N_f = 1 framing for the Schwinger paper (plan assumes N_f = 2
   for sampling; decide at WS3).
3. Phase IIa as standalone short paper vs. first section of the Schwinger
   paper (decide after A-5 results).
4. Whether Phase I v2 absorbs the delayed-acceptance idea or it stays
   exclusive to the IIb paper (plan assumes the latter).

## Status pointers (as of 2026-07-09)

- WS1 complete; `paper-v1` tag pushed; arXiv submission pending hep-lat
  endorsement (requested from D. Schaich).
- A-1/A-2 complete: labeled U(1) ensembles in `data/u1_configs/*.h5`
  (theta[N,2,L,L] float32 + action/q/wilson datasets; validation tables in
  `results/u1_heatbath_data.json`, `results/u1_labels.json`).
- Next task: **A-3** (plan §3) — three gauge graph-builder variants
  (A: link-nodes, B: links-as-edge-features, C: invariant-input oracle),
  extending `graphs/edge_types.py` + `graphs/builder.py`, reusing
  `NodeType.GAUGE`; beta appended as input feature in all variants.
- Training runs on Colab via `notebooks/07_ws1_experiments_colab.ipynb`
  pattern (CLI scripts + Drive sync); MC stays on laptop CPU.
