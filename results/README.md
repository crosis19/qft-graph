# results/

Provenance records for every number quoted in the paper. Convention (plan ground
rule 5): each run whose output feeds a figure or table writes
`results/<run_id>.json` containing `(git commit, config hash, seeds, config, metrics)`
via `qft_graph.utils.run_logging.log_run()`.

- `phase1_paper/` — frozen copies of the result JSONs behind the current
  manuscript PDF (Tables II, III and Fig. 2), with provenance notes. These are
  the *pre-correction* numbers; WS1 tasks P1-2..P1-6 regenerate them with
  proper statistics.

Large artifacts (checkpoints, MC datasets) stay in the gitignored
`experiments/` and `data/` trees; only small JSON/metadata records live here.
