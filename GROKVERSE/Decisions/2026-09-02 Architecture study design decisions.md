# 2026-09-02 — Architecture study design decisions

Decisions taken when GROKVERSE was turned from a reproduction-plus-explorer project into a controlled
mechanistic comparison. Source of truth: `Claude/grokverse/docs/dev/PREREG_BRIEF.md` (with its addenda)
and `Claude/grokverse/docs/LABBOOK.md`.

Values marked **[HUMAN]** were decided by the project author on 2026-09-02. Everything else is
**[AI-PROPOSED — pending human approval]** and is collected for approval in `docs/PREREGISTRATION.md` §11.

## 1. Fixed 25,000-step budget instead of stopping at the crossing **[HUMAN]**

**Decision.** Every primary run trains for exactly 25,000 steps with no early stop; every structure metric
is reported at two declared points, the generalization crossing and the final step.

**Reason.** All six legacy un-accelerated runs stopped at their own 0.95 crossing, so every legacy
structure number is an at-transition number. Khanh 2026 shows metrics read at the transition overstate the
converged value by 3–5x (MLP) and 1.3–1.5x (transformer), with a lag of order 10,000 steps. A shared fixed
budget also satisfies the master prompt's "same training steps" requirement for a controlled comparison.

**Alternatives considered.** (a) Train to 1.5x each run's own crossing (the earlier research spec's
proposal) — cheaper, but each run's "final" state is then event-relative, not budget-matched.
(b) Nanda's full 40,000 steps — ~1.6x the compute for the same qualitative point.

**Consequence.** ~4 h per transformer run under 8-way parallelism instead of ~1 h; the budget is
defensible only if the settling check is actually run and reported per seed, so "final" is never renamed
"converged" by assumption.

## 2. Ten paired seeds, full control matrix **[HUMAN]**

Seeds 0–9, identical for both architectures (same split, asserted by a split hash). Plus the Grokfast x
train-fraction confound matrix (3 seeds per cell), a parameter-matched MLP (`d_mlp = 572`, +0.02% of the
transformer's parameter count) over 10 seeds, and a two-hot-input MLP control over 3 seeds.

**Reason.** The legacy comparison confounded acceleration with train fraction and left a 9.8% parameter
difference uncontrolled while weight decay 1.0 acted on all parameters.

## 3. Key frequencies from the neuron-to-logit map, not from a fixed top-8 **[AI-PROPOSED]**

**Decision.** The primary rule is Nanda et al.'s: DFT of `W_L = W_U W_out` along the class axis, norm over
neurons, keep every frequency whose norm is at least **0.25x the maximum** (sensitivity 0.10 and 0.50).
The count is measured, never capped.

**Reason.** The legacy rule capped the count at 8 and the cap binds on all 16 legacy runs, so the count
was never data-determined. The source note establishes the published rule; the paper does not publish the
numeric threshold, so the 0.25 cut is a GROKVERSE choice and is declared as one.

**Note.** This substitution was *pre-declared* as a conditional before the sources were read, and
triggered when the note confirmed the published rule.

## 4. The two-hot control keeps `d_mlp = 512`, not the paper's 256 **[AI-PROPOSED]**

Swaroop's model is two-hot -> 256 ReLU -> p at p=97 with a stratified split and early stopping. GROKVERSE
keeps its own width, modulus, split and budget so the control differs from the shared-embedding MLP in the
**input parametrization only**. It is therefore "our two-hot variant", never a replication of that paper's
model.

## 5. Evidence gate before any cross-architecture interpretation **[AI-PROPOSED]**

Each architecture must independently pass four criteria — periodic structure, the phase-addition relation,
an end-to-end logit fit, causal necessity *and* sufficiency — in at least 8 of 10 seeds, before the
metric-validity hypothesis may be interpreted as "the same Fourier principle in a different
representation". The three branches of the decision tree (both pass / one passes / neither passes) were
written down before the runs, so a negative result is a publishable outcome rather than a failure.

## 6. Analysis code frozen after a pilot **[AI-PROPOSED]**

The seed-0 pair validates the pipeline only; no threshold is tuned on it. After the pilot the analysis
code is frozen at a named commit, and any later change is logged with its reason and forces a re-run of
every affected analysis.

## Related

[[Grokking measurement pitfalls]] · [[2026-09-03 - GROKVERSE]] ·
[[Primary architecture run matrix]]
