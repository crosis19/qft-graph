# A-5 half-page summary: learned vs. given gauge invariance

*(Draft prose for the Phase IIa paper — plan §3, Task A-5 done-when.
Every number regenerates from `scripts/measure_gauge_invariance.py`,
`make_a5_table.py`, `plot_a5_curves.py`; records in `results/a5*.json`.)*

**Measurement.** For each test configuration we generate K=32 random
gauge copies and define, per prediction head,
eps_gauge = mean over configs of the std of the prediction across the
orbit, divided by the std of the prediction across configs. A
gauge-invariant predictor has eps_gauge = 0; a predictor that responds
to the gauge representative as strongly as to the physics has
eps_gauge ~ 1. Copies are generated in float64 from the float32-stored
links, so the invariant-oracle inputs (Variant C) are bit-identical
across the orbit and its eps_gauge is exactly zero by construction —
which the measurement reproduces bitwise on all 36 C checkpoints
(a necessary numerics detail: evaluation is unbatched, since batched
message passing is position-in-batch dependent at float32 lsb).

**Result 1 — the trained models.** Across every (L, beta) cell, seed,
both parameter budgets, and all four escape hatches, Variants A
(link nodes) and B (links as edge features) sit at eps_gauge = 0.96-1.05
on all seven heads: their outputs carry *zero* gauge-invariant
component. This is a sharper statement than the A-4 accuracy null
(r ~ 0) — the models did not learn a noisy invariant; they learned
nothing invariant at all. Consistently, at beta=1, the one cell with a
small real signal (r ~ 0.08), eps_gauge dips a few percent below 1.

**Result 2 — augmentation does not rescue A/B.** Retraining A and B
with a fresh random gauge transform of every training configuration at
every access, over training sizes n = 50 to 3200 (3 seeds), changes
nothing: test r stays within seed noise of zero at every n, and
eps_gauge stays at 1.00 at every n (also at L=16). What augmentation
(and, only slightly more slowly, plain training) does instead is shrink
the prediction spread toward the constant predictor — augmented Variant
A's config-to-config std of the standardized action output falls from
1.4e-3 at n=50 to 1.1e-6 at n=3200 (unaugmented: 2.5e-3 to 1.6e-6) —
while the residual fluctuation remains pure gauge noise. Optimization can damp the covariant component; it never rotates
it into the invariant subspace. Meanwhile the oracle C reaches
action r = 0.87(6) with only 50 configurations and is at ceiling by
n ~ 200-400 (large loops remain receptive-field-limited, W4x4
r ~ 0.48 at B=3).

**Reading.** The failure augmentation was meant to fix is not a missing
invariance prior or a data deficit; it is representational: a single
link angle is exactly uninformative under the gauge group, so both the
supervised gradient and the augmentation-consistency gradient vanish at
first order. At this (laptop) scale, gauge invariance must be built
into the inputs or architecture — flexibility does not substitute. This
sharpens, rather than contradicts, Phase I's flexibility-over-rigidity
argument (Sec. V.B): flexibility won for translation symmetry, where
the symmetry leaves per-site features informative; it loses for gauge
symmetry, which annihilates the local signal entirely.
