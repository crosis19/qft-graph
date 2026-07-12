# A-4 protocol v1 results (superseded)

The 114 runs here were trained with Q on its natural integer scale
(protocol v1). That unbalanced the joint loss where Q's variance is
large — L=32, and beta=4 with deep stacks (B>=4) — causing Q-head
collapse and seed instability (see the anomaly analysis in
docs/phase2_decisions.md, decision 5, and commits 6a38b92 / 523d4f9).

Protocol v2 standardizes ALL targets (Josh, 2026-07-12); every run was
redone under v2 for uniformity. Kept for provenance — do not mix these
numbers with v2 results in any table or figure. At L<=16 with B<=3,
Q's natural std is ~1, so v1 and v2 should agree closely there (a
consistency check for the rerun).
