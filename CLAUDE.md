# CLAUDE.md — qft-graph working instructions

Working spec for all Claude Code sessions in this repo:
**[qft_graph_implementation_plan.md](qft_graph_implementation_plan.md)** — read the relevant
task section (P1-x / A-x / B-x) before starting work. For the Phase I journal-revision
workstream (V2-x tasks, branch `paper-v2`), the working spec is
**[docs/v2_cluster_fss_plan.md](docs/v2_cluster_fss_plan.md)** instead. Read
`ARCHITECTURE.md` before touching model or graph code.

## Ground rules

1. **Physics tests gate everything.** No training run launches until the exact-value unit
   tests for that module pass. The tests defined in the plan are the arbiters of sign
   conventions — if a derivation in the plan conflicts with a passing exact-value test,
   trust the test and flag the discrepancy.
2. **Conventions in the plan are frozen** (gamma matrices, plaquette orientation,
   gauge-transform signs, χ definition). Propose changes in a comment; never change
   silently mid-project.
3. **One task ID per session/PR.** Tasks are labeled P1-x, A-x, B-x in the plan (and V2-x
   in `docs/v2_cluster_fss_plan.md`), with explicit definitions of done.
4. **Every figure and table is regenerable** by a committed script + config file. No
   numbers pasted into the paper by hand without a script that produced them.
5. **Reproducibility:** all seeds from config files; log `(git commit, config hash, seeds,
   metrics)` to `results/<run_id>.json`. Prefer boring numpy/scipy/torch; no new heavy
   dependencies without approval.
6. **CPU-first.** Everything is sized for a laptop (~150k-param model). GPU is optional
   acceleration, never a requirement (`device: auto` in configs).

## Terminology: "energy" ↔ "action"

The code says **action** (`models/heads/action.py`, `ActionHead`) after the terminology
cleanup, but the training-loss key and output dict key remain `energy`
(`loss: energy_matching`, `output["energy"]`) where the paper says **action** S_E[φ].
Map the two in prose/docstrings; do not mass-rename.

## Running tests

```bash
.venv/Scripts/python.exe -m pytest tests/ -q      # Windows, this repo's venv
```

All tests must pass before committing. Physics-oracle tests (exact values) are never
weakened to make an implementation pass — fix the implementation.

## Phase II decisions

Read **[docs/phase2_decisions.md](docs/phase2_decisions.md)** before starting
any A-x/B-x task — it records which of the plan's open decisions are resolved
(e.g. coordinate-free spacetime features are adopted everywhere in Phase II)
and which still need Josh.

## Layout for new Phase II code

See the table in plan §1: statistics → `src/qft_graph/mc/analysis.py`; Wilson action →
`src/qft_graph/actions/wilson.py`; U(1) links → `src/qft_graph/fields/gauge.py`; heatbath →
`src/qft_graph/mc/heatbath.py`; Dirac/determinant → `src/qft_graph/fermions/`; graph
variants → extend `graphs/edge_types.py` + `graphs/builder.py`. Extend the existing
registries (`NodeType` enum, edge-type helpers, `actions/base.py`, `fields/base.py`)
rather than inventing parallel structures.
