> **APPROVED** by Josh, 2026-07-09 — paragraph integrated into Sec. IV.C.

# Postmortem: original Table III protocol (task P1-3)

**What the original eval did** (verified in `scripts/evaluate_generalization.py`
and `notebooks/05_baselines_colab.ipynb` §5): the model trained at
m² = −0.5 was evaluated on datasets generated at other m² values, with target
actions computed using **each dataset's own (new) m²** — the labels are
S_{m²_new}[φ], produced by `Phi4Action` at generation time.

**Why that conflates two effects**: the model has no coupling input, so when
the target function changes from S_{m²_train} to S_{m²_new} the task becomes
partially unlearnable — no function of φ alone can match a family of action
functionals indexed by a hidden parameter. The reported degradation therefore
mixes (i) genuine distribution shift in the sampled configurations with
(ii) an unlearnable label shift, and the large relative errors at distant m²
mostly track the changed action scale rather than architectural failure.

**Draft paragraph for the paper (3 sentences, for Josh's approval):**

> In the original protocol, a model trained at a single coupling was evaluated
> against target actions computed at each new coupling; since the architecture
> received no coupling input, this measured a partially unlearnable label
> shift on top of the distribution shift in the sampled configurations.
> Table III′ replaces this protocol: the coupling constants (m², λ) are now
> model inputs — appended to the field-node features and to the readout — and
> a single model is trained jointly across alternating points of a 13-point
> m² grid spanning both phases. Held-out interleaved couplings test
> interpolation and the two grid endpoints test extrapolation, with all
> entries reported as mean ± std over five seeds.
