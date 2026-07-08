# Provenance — Phase 1 paper (pre-correction) results

Frozen copies of the result files behind the manuscript PDF built at commit
`330dacd` (paper/main.pdf, 2026-04-03). Recorded here by task P1-0 so a fresh
clone can trace every published number; the WS1 statistics rework
(P1-2..P1-6) supersedes these.

| File | Feeds | Produced by | Run date |
|---|---|---|---|
| `baseline_results.json` | Table II | `notebooks/05_baselines_colab.ipynb` (equivalent CLI: `scripts/train_baselines.py`) | 2026-03-28 |
| `generalization_results.json` | Table III | `notebooks/05_baselines_colab.ipynb` §5 (equivalent CLI: `scripts/evaluate_generalization.py`) | 2026-03-28 |
| `sweep_results.json` | Fig. 2, Secs. IV.D–E (m²_c, ν, γ/ν) | `notebooks/03_train_colab.ipynb` FSS sweep cell; protocol in `configs/paper/phase1_fss_sweep.yaml` | 2026-03-27 |

Training protocol of record: `configs/paper/phase1_train_16x16.yaml` /
`phase1_train_64x64.yaml`. Model checkpoint: `experiments/runs/colab_run/model_final.pt`
(gitignored; state-dict uses pre-rename keys — load via
`qft_graph.utils.checkpointing.remap_legacy_state_dict`).

Known caveats documented at audit time (fixed in later WS1 tasks):

- Table I numbers come from training runs without seed variance (P1-4 adds ±).
- FSS sweep was unseeded (`seed=None`) and its ξ estimator used the
  V·Var(M) susceptibility internally, while the χ panel used the frozen
  V·(⟨M²⟩−⟨|M|⟩²) convention (P1-2 unifies conventions and adds errors).
- Table III conflates distribution shift with an unlearnable label shift
  because the model has no coupling inputs (P1-3 fixes and produces Table III′).
