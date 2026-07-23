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

## Decisions taken (2026-07-22)
1. **Journal target: *Machine Learning: Science and Technology* (MLST, IOP).**
   Closest topical fit for a paper whose contribution is the graph representation
   itself, and it serves the goal of building credibility in both the ML and
   physics communities. *Practical checks before submitting:* MLST is fully open
   access (article publication charge — verify the current rate and IOP's waiver
   policy as an unaffiliated author), and it uses IOP style rather than the
   current revtex4-2 (likely fine for a format-neutral initial submission).
   - *Deferred:* PRD is a stronger target for the **Phase II Schwinger/gauge
     paper**, once there are gauge results; reviewers reasonably note PRD referees
     would ask "where is the gauge theory?" for the present scalar-only manuscript.
2. **Abstract: keep the architecture-first opening.** Consistent with an ML venue
   (an MLST referee wants the ML contribution first) and with the fact that the
   graph representation is the originating idea. The physics-first alternative
   below is therefore **not** adopted; it is retained only as a record in case the
   venue changes to a physics journal.
3. **arXiv: not pursuing it — decision made 2026-07-22.** The v1 submission was
   declined by moderation and we are not re-submitting, appealing, or posting a
   later version there. The route is journal publication only.
   - *Recorded for context, since the earlier reviewer advice was wrong on this:*
     the moderation email is submission-level ("your *submission* will not be
     accepted"), never names a category, and offered only one way back — a
     conventional-journal DOI. Had moderators judged it merely misfiled, the
     normal action is *reclassification*, not rejection. Re-submitting
     substantially the same work under a different primary category would have
     routed it to different moderators and is generally treated as circumventing
     moderation. So the reviewers' "switch the primary category" suggestion was
     never a viable shortcut. Moot now regardless.
   - **Consequence for venue choice:** with no preprint anywhere, the published
     version is the *only* public copy, so open access matters more than it
     otherwise would. This strengthens the MLST choice (gold OA) and would make
     SciPost Physics (diamond OA, free) the natural fallback; a paywalled
     subscription-route paper (PRE/CPC) would leave the work effectively
     invisible outside subscribing institutions.
   - The manuscript itself contains no arXiv self-references, so nothing in the
     paper needed changing (verified 2026-07-22). The arXiv eprint fields in
     `references.bib` are ordinary citations of others' work and stay.

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
