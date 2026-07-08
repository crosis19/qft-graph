# P1-0: Repo ↔ manuscript sync audit (2026-07-08)

Task P1-0 of [the implementation plan](../qft_graph_implementation_plan.md).
The plan was written against origin/master as pushed 2026-03-27; local master
was already 2 commits ahead (46f808d baselines/generalization/displacement,
330dacd terminology fix), which resolved several suspected gaps before this
audit began.

## Plan item 1 — commit the manuscript state behind the PDF

**Already resolved.** `paper/main.tex` and `paper/main.pdf` in HEAD (330dacd)
match the working tree exactly; the PDF was rebuilt in that commit.
`git diff HEAD -- paper/` is empty. The two commits carrying this state are
local-only: **origin/master is 2 behind — push when ready** (not done here;
Josh's call).

## Plan item 2 — missing reproducibility pieces

| Suspected missing | Status at audit | Action taken |
|---|---|---|
| 64×64 lattice config | Genuinely missing | Added `configs/lattice/64x64.yaml` |
| Baseline model code (Table II) | Present since 46f808d: `src/qft_graph/models/baselines/`, `scripts/train_baselines.py` | Verified; results frozen in `results/phase1_paper/` |
| Coupling-generalization script (Table III) | Present since 46f808d: `scripts/evaluate_generalization.py`, **but** its data-generation fallback used a stale API (`Phi4Action(field_config, lattice)` swapped args, 4-arg `MetropolisSampler`, nonexistent `.sample()`) and would crash on any missing dataset | Fixed to current API (`create_sampler`, `generate`) |

Additional reproducibility repairs:

- **Legacy checkpoints unloadable**: 330dacd renamed `EnergyHead→ActionHead`,
  breaking `model_final.pt` (keys `energy_head.energy_mlp.*`). Added
  `remap_legacy_state_dict()` in `utils/checkpointing.py`, applied in
  `load_checkpoint`, `scripts/evaluate.py`, `scripts/evaluate_generalization.py`,
  and `paper/generate_figures.py` — the published checkpoint loads again
  (verified: r = 1.0000 on 16×16 validation split).
- **Paper-feeding result JSONs were gitignored** (under `experiments/`).
  Frozen copies now committed in `results/phase1_paper/` with `PROVENANCE.md`.
- Established `results/<run_id>.json` convention (`utils/run_logging.py::log_run`)
  recording git commit, config hash, seeds, metrics for all future runs.

## Plan item 3 — config ↔ paper hyperparameter discrepancies

The actual runs (notebooks 03/05, embedded dataset configs) used **150 epochs,
5000 configs, lr 1e-3, batch 32, seed 42, a_in_edges=False** — the paper is
right, `defaults.yaml` was wrong. Actions:

- Committed the protocols of record: `configs/paper/phase1_train_16x16.yaml`,
  `phase1_train_64x64.yaml`, `phase1_fss_sweep.yaml`. P1-7's hyperparameter
  appendix reads from these.
- Aligned `defaults.yaml` and dataclass defaults to the paper protocol
  (epochs 200→150, n_configs 10000→5000).
- `device: cuda` → `device: auto` with `resolve_device()` (ground rule 6).

## Findings for downstream tasks (not fixed here)

1. **χ conventions (P1-2).** Three different formulas coexist:
   - Notebook 03 (the actual Fig. 2 source): `V·(⟨M²⟩−⟨|M|⟩²)` — **the frozen
     convention; Fig. 2's χ panel is already correct.**
   - `mc/observables.py::correlation_length_fft` (used for Fig. 2's ξ/L):
     `V·Var(M)` internally — must switch to the frozen convention.
   - `scripts/sweep.py`: `V·(⟨⟨φ²⟩_sites⟩−⟨|M|⟩²)` — mixes site-averaged φ²
     with magnetization; **not a susceptibility at all**. Worse than the plan
     suspected. `sweep.py` also averages per-config log-slope ξ over 100
     configs (biased estimator) instead of the FFT ensemble estimator.
   - Manuscript Sec. III.A: bullet says frozen form, ξ-implementation align
     block says Var(M) — internally inconsistent; align both to frozen form.
2. **FSS sweep was unseeded** (`seed=None` in notebook 03) — the P1-2 rerun
   must seed from config.
3. **Paper claim "n_sweeps = 10 between configs (40 for L=64)"** (Sec. III.B)
   is wrong for the *training* data: all four training sets (8–64) used 10
   sweeps / 1000 therm / seed 42 (verified from embedded MCConfig in each
   `mc_data.pt`). The {10,20,40} separations apply to the FSS sweep only.
   Fix wording in P1-7.
4. **Orphan figures**: `energy_prediction.pdf`, `free_field.pdf`,
   `scaling_collapse.pdf` exist in `paper/figures/` but are not referenced by
   `main.tex`; `baseline_comparison.pdf` and `generalization.pdf` are generated
   by `generate_figures.py` but were never produced (missing inputs at run
   time). Reconcile in P1-7.
5. **Unpushed commits**: local master is now several commits ahead of origin.
   Push before tagging `paper-v1`.

## Definition of done check

"A fresh clone can regenerate Tables I–III and Fig. 2 from committed code":

- **Table I**: `scripts/generate_mc_data.py` + `scripts/train.py` + `scripts/evaluate.py`
  with `configs/paper/phase1_train_{16x16,64x64}.yaml` (retraining; checkpoints
  are not committed by design).
- **Table II**: `scripts/train_baselines.py --data ... --epochs 150`.
- **Table III**: `scripts/evaluate_generalization.py --checkpoint ...` (fallback
  data generation now works).
- **Fig. 2**: `paper/generate_figures.py` from `results/phase1_paper/sweep_results.json`
  (copied to the expected `experiments/runs/colab_run/` path) — pure data replot.

Exact per-figure/table commands land in the README as part of P1-7.
