# Journal-v2 review brief (for the Schaich email)

*Prepared 2026-07-22. One-page summary of where v2 stands after two independent
referee-style reviews, plus the open decisions that need your input.*

## Status
- **arXiv moderation declined the v1 submission** (ticket MOD-97144, 2026-07-21).
  The stated reason is fit/framing ("would benefit from review outside our
  services"), not a scientific critique; the appeal path is a conventional-journal
  DOI. We are therefore prioritizing the journal submission.
- **v2 incorporates your FSS/cluster-sampler suggestions in full.** Before bringing
  it back to you we ran it past two LLMs as referee stand-ins (same as for v1).

## Verdict of both reviews
- **ChatGPT — "Minor Revision (borderline Accept)."** All concerns are
  *calibration of claims*, not deficiencies in the science. Praised the physics
  framing, the honest limitations, and the added experiments.
- **Gemini — "publication-ready."** Singles out the cluster sampler (tau_int
  185 -> ~8-9), the precision exponents gamma/nu = 1.732(8), nu = 1.04(8)
  "squarely in the Ising class," and the Sec IV.E argument (cluster algorithms
  exist for the Z2 scalar but *not* for the gauge/fermion theories we target ->
  which justifies the learned architecture) as "exactly what PRD wants."
- Both independently read the arXiv bounce as a **category/framing** issue, not a
  quality one.

## Changes already made (calibration edits from the reviews)
1. Softened "mirrors the structure of QFT" -> "inspired by / built around" the
   geometry-matter distinction (abstract, intro).
2. "the model learns a local effective action" -> "admits an interpretation as a
   local effective action" (Sec. on interpretation).
3. Conclusion now **leads** with the structural-failure-mode result (our strongest
   claim), per ChatGPT.
4. Future-work section lightly trimmed.
Most other items the reviewers raised (e.g. "engineering patch" -> "architectural
workaround"; "to our knowledge, absent from ...") were already fixed in your pass.

## Open decisions — your call
1. **Journal target.** The two reviews diverge: ChatGPT suggests *Phys. Rev. E*,
   *Computer Physics Communications*, or *SciPost Physics*, and is less sure about
   *PRD* (may want direct gauge-theory application); Gemini thinks the physics
   framing now fits *PRD*/*JHEP*. Worth your read on where it lands.
2. **arXiv category for the eventual re-try.** Gemini recommends **physics.comp-ph
   or cs.LG as the primary** category with hep-lat as a cross-list, rather than
   fighting for hep-lat-primary. Low-friction and consistent with why v1 bounced.
3. **Abstract framing.** Whether to keep the current architecture-first opening or
   switch to a physics-first opening (below) that front-loads the sampling problem
   — this depends on decision (2). Note: Gemini's suggested wording included
   "topological freezing," which this paper does **not** address (it is critical
   slowing down in a scalar theory); the draft below drops that to avoid
   over-claiming.

## Alternate physics-first abstract opening (option, if we push hep-lat)
Replaces only the first sentence; the rest of the abstract is unchanged.

> Generating training ensembles for lattice quantum field theory is limited by the
> critical slowing down of local Monte Carlo updates near criticality — and, for
> the gauge and fermion theories of ultimate interest, by the absence of the
> cluster algorithms that cure it, motivating learned surrogates for the action.
> We introduce a heterogeneous bipartite graph neural network architecture that
> separates *spacetime geometry* from *field content* into distinct node types
> connected by typed edges — a design inspired by the geometry-matter distinction
> in continuum quantum field theory. [...unchanged...]

*(Current opening, for comparison: "We introduce a heterogeneous bipartite graph
neural network architecture that separates spacetime geometry from field
content...")*

The manuscript (`paper/main.pdf`) builds clean; every FSS number is
script-generated from the committed cluster analysis.
