# P1-7 wording proposals — NEED JOSH'S APPROVAL before commit

Per the plan, all abstract/conclusion wording changes require your sign-off.
Everything below is a *proposal*; the committed manuscript still carries the
old sentences. Reply with approve/edit per item.

## 1. Abstract — FSS attribution fix (the blocking issue) + accuracy claims

**Current:**

> The model achieves near-perfect action prediction ($r > 0.999$) across
> lattice sizes from $8 \times 8$ to $64 \times 64$ and reproduces
> finite-size scaling behavior including susceptibility peak growth, order
> parameter S-curves, and correlation length crossing consistent with the
> exact critical exponent.

Two problems: (a) FSS results come from the Monte Carlo ensembles, not the
model; (b) with 5-seed statistics, "r > 0.999 at 64×64" holds for 4 of 5
seeds (one converged to 0.910 in the fixed budget — final wording depends on
the P1-4d diagnostic).

**Proposed replacement** (plan's template, updated numbers, transfer result
added):

> The model achieves near-perfect action prediction across lattice sizes
> ($r > 0.9999$ at $8 \times 8$ and $16 \times 16$; median $r = 0.9992$ at
> $64 \times 64$), and a single model trained at $16 \times 16$ transfers to
> lattices from $8 \times 8$ to $64 \times 64$ with $r = 1.0000$ and no
> retraining. The underlying Monte Carlo pipeline reproduces Ising-class
> finite-size scaling --- susceptibility peak growth with
> $\gamma/\nu = 1.57(12)$, order-parameter S-curves, and correlation-length
> crossings giving $\nu = 1.10(43)$, consistent with the exact $\nu = 1$ ---
> validating the training distributions.

(If the P1-4d 300-epoch diagnostic shows seed 4 converging, an alternative
is "$r > 0.999$ across lattice sizes, with one of five seeds at
$64 \times 64$ requiring a doubled epoch budget".)

## 2. Conclusion — same attribution fix

**Current:**

> On two-dimensional scalar $\phifour$ theory, the model achieves
> near-perfect action prediction ($r > 0.999$), reproduces the expected
> Ising universality class finite-size scaling, and yields critical exponent
> estimates consistent with exact values.

**Proposed:**

> On two-dimensional scalar $\phifour$ theory, the model achieves
> near-perfect action prediction and transfers across a $64\times$ range of
> lattice volumes without retraining; the underlying Monte Carlo pipeline
> reproduces the expected Ising-class finite-size scaling
> ($\nu = 1.10 \pm 0.43$, $\gamma/\nu = 1.57 \pm 0.12$), validating the
> ensembles the model is trained on.

## 3. Already committed (body text, not abstract/conclusion — flag if you disagree)

- Sec. IV.D now opens by stating the FSS analysis characterizes the Monte
  Carlo training ensembles (pipeline validation), and the subsection titles
  say so.
- Sec. IV.B collapse claim rewritten to the measured depth-dependent story
  (strict $r<0.05$ only at $B=6$).
- $m^2_c$ moved from $-2.45(10)$ to $-2.22(24)$ with the two-momentum
  estimator (see docs/xi_estimator_issue.md — separate approval item).

## 4. Pending

- Table III' + Sec. IV.C rewrite + postmortem paragraph
  (docs/table3_postmortem.md) — lands when the Colab run syncs.
- Final abstract numbers depend on the P1-4d seed-4 diagnostic.
