# Draft email to Prof. Schaich (v2 update)

> **RESOLVED (2026-08-03): permission GRANTED in writing.** The acknowledgment
> now lives in paper/main.tex inside the acknowledgments environment, in the
> non-anonymous (`\else`) branch only — the anonymized review build continues
> to omit it (identity-adjacent under double-anonymous review). It flows into
> the manuscript automatically at the first revision that de-anonymizes, or at
> acceptance when IOP restores acknowledgements. Nothing to send the journal
> before then.

*Draft for Josh to edit/send. Kept deliberately short with two clear asks —
senior readers respond best to that. Attach `paper/main.pdf`.*

---

**Subject:** qft-graph v2 — thank you, and an update on the arXiv submission

Dear Prof. Schaich,

Thank you again for endorsing the submission and for your comments on the
finite-size scaling — they shaped this revision substantially.

A quick update first: arXiv moderation declined the v1 submission. They judged
that it "would benefit from additional review and revision that is outside of
the services we provide," and offered reconsideration on appeal once the work
has a journal DOI. Rather than contest that, I've focused on strengthening the
manuscript for journal submission.

Your suggestion drove the main change. I implemented a Wolff/Brower–Tamayo
embedded-cluster sampler for the φ⁴ ensembles, which removes the
critical-slowing-down bias in the local-Metropolis results: at the pseudocritical
point τ_int falls from ~185 single sweeps at L = 64 to ~8–9 roughly independent
of L, which made L = 96 and 128 reachable at all. With the bias gone the
finite-size scaling now gives γ/ν = 1.732(8) and ν = 1.04(8), consistent with the
exact Ising values — where the local-sampler fit had been biased low
(γ/ν = 1.60(3), with the effective slope drifting from 1.64 to 1.42 as L grew).
Sections IV.D–E are rewritten around that comparison and now include a τ_int
table. The revised manuscript is attached.

Two things I would be grateful for:

1. Would you be comfortable with an acknowledgment in the paper — something like
   *"We thank D. Schaich for helpful comments on the finite-size scaling
   analysis"*? With no implication, of course, that you endorse the conclusions.

2. I'm planning to submit to *Machine Learning: Science and Technology* (IOP),
   which seems the closest fit for a paper whose central contribution is the
   graph representation itself, validated on a lattice benchmark. If you have a
   view on that — or think a computational/statistical-physics venue would serve
   it better — I would value your thoughts.

Thank you again for the time you have already given this.

Best regards,
Joshua Paul Brehm

---

## Notes (not part of the email)
- **Do not** ask him to intervene with arXiv moderation — not appropriate or
  effective. Informing him is right; he is the endorser and may have seen this
  outcome before.
- The technical paragraph exists to show his advice produced a concrete result;
  that is the most substantive form of thanks.
- If he replies suggesting a physics venue instead, the physics-first abstract
  opening drafted in `v2_review_brief.md` becomes the better choice.
