# Limitations

AI-drafted (Claude), 2026-09-03 — not yet human-reviewed. Two kinds of limitation are separated:
**design limitations**, which are fixed by the study's construction and are known before any result, and
**result limitations**, which can only be written once the analyses have run. The second section is
deliberately empty of content and lists what must be filled in.

---

## A. Design limitations (known now, not removable by more analysis)

### A1. One point in hyperparameter space

The primary comparison is `p = 113`, addition, `train_frac = 0.3`, `weight_decay = 1.0`, AdamW at
`lr = 1e-3`, full batch, no Grokfast. It is not a sweep. Manir & Rupa 2026 (arXiv:2603.25009) show that
the transformer-vs-MLP relationship *moves* with regularization and depth — and report the timing gap in
the **opposite** direction under their protocol (per-architecture optimizers, depth-4 MLP, `p = 97`,
20 % training fraction). No statement of the form "architecture X is faster/better" is licensed by this
study; only "under the conditions tested".

### A2. Ten seeds is a small sample

With 10 paired seeds, confidence intervals on timing quantities are wide — the legacy 3-seed transformer
spread was ±1196 steps. Effect sizes and intervals are reported for everything, p-values never alone, and
no spurious precision is quoted. A paired difference whose interval spans zero is reported as such.

### A3. Two architectures, one task

Only a 1-layer ReLU transformer without LayerNorm and a 2-layer shared-embedding ReLU MLP (plus a two-hot
MLP control) on modular **addition**. Nothing here transfers to deeper models, other activations, other
group operations, or LayerNorm'd transformers without new evidence. Khanh 2026 specifically finds
LayerNorm changes compression timing, and our transformer has none.

### A4. The transformer's effective operand curves are an approximation

The MLP is exactly additive in `(a, b)` up to the ReLU; the transformer is not, because attention depends
on the input. The comparable object is built with *mean* attention, and the share of the pre-activation it
fails to capture is measured per neuron as `1 − additivity_r2`
(`docs/TRANSFORMER_MECHANISM_DERIVATION.md` §4.1). Where that share is large, the effective-curve
comparison explains correspondingly less of the transformer, and this bound is carried with every
cross-architecture number derived from those curves.

### A5. Architecture-specific objects cannot be compared

`W_pos`, `W_Q/W_K/W_V/W_O`, the residual stream, the attention heads and the direct (non-MLP) logit path
exist only in the transformer. They are analysed and reported, but excluded from the headline comparison,
because the MLP has nothing to compare them against.

### A6. "Final" is a budget, not a proof of convergence

Every run stops at a fixed 25 000 steps. Whether a given metric has actually settled is *tested* per
metric and per seed (change below 5 % between the step-20 000 and step-25 000 checkpoints) and reported;
a metric that fails that test is labelled "final (budget), not converged". Khanh 2026 measures compression
lags of order 10 000 steps or more, and for our MLP the post-crossing window is only ~15 000 steps, so
censoring at the strict tolerance is a real possibility that must be reported rather than hidden.

### A7. Transitions live on a discrete evaluation grid

Train accuracy is evaluated every 10 steps and test accuracy every 25. Every crossing is therefore an
interval `(previous evaluated step, first crossing]`, and the phrase "exact transition" is not used.

### A8. The key-frequency threshold is ours, not the paper's

Nanda et al. determine key frequencies from the neuron→logit map and keep those with "nontrivial"
coefficients, but never publish the numeric cut. The 0.25-of-maximum threshold is a GROKVERSE choice,
declared in advance with sensitivity at 0.10 and 0.50.

### A9. `nanda_exact` may remain partly unreconstructable

The restricted-loss data split is not stated in the paper (the released notebook computes it on all and on
train; the TransformerLens demo on test). Where an item cannot be reconstructed from the primary source,
the implementation raises rather than guesses, and the documentation says so.

### A10. The two-hot control is ours, not Swaroop's model

It keeps our width, modulus, split and budget so that only the input parametrization differs from the
shared-embedding MLP. It therefore tests *our* question and is not a replication of arXiv:2603.23784.

### A11. Ablation is causal about *this* checkpoint, not about learning

All ablations run on unmodified trained checkpoints with no retraining. They can show that a component is
necessary or sufficient for the trained model's behaviour; they cannot show that the component *had to*
form, or how it formed.

### A12. Compute and environment

CPU-only, single-threaded per run, float32 training. Bit-exactness holds per environment: torch reductions
differ across thread counts, which over thousands of steps moves final accuracies by ~0.5 %. Thread count
is pinned and recorded so the matrix is reproducible; results from a different machine may differ in the
last digits, though transition step indices have reproduced exactly in every re-train tried.

### A13. Review is AI-only

Code, derivations, source notes and audits were written by AI agents and reviewed by *other AI agents*
with different instructions. That caught real defects — including one wrong pre-registered prediction
(`docs/LABBOOK.md` entry 15) — but it is not independent human review, and this document must not be read
as if it were. See `AI_DISCLOSURE.md`.

### A14. The pre-registration is not yet human-approved

Every threshold, definition and pass rule other than the step budget and the matrix scope is
`[AI-PROPOSED]`. `docs/HUMAN_DECISIONS.md` still reads `STATUS: NOT YET APPROVED BY HUMAN AUTHORS`.
Results computed under these settings must be reported as such until that changes.

### A15. Bounded, not exhaustive, alternative-mechanism search

If an architecture fails the evidence gate, the follow-up is a *bounded* set of tests (decodability of
`(a+b) mod p` from hidden activations, low-rank and symmetry structure, cross-seed consistency, causal
removal). The study does not promise to discover every possible alternative algorithm; a well-founded
negative result stands even if the bounded analysis finds nothing.

---

## B. Result limitations — **to be written after the analyses**

Each of these can only be filled in with the measurement in hand. Leaving them empty is deliberate; they
must not be pre-filled with expectations.

- [ ] Which architectures passed the evidence gate, in how many seeds, and which criterion failed where.
- [ ] Which decision-tree branch was taken, and what that branch permits saying.
- [ ] For every structure metric: whether it had converged at the budget, per seed and per architecture.
- [ ] The measured `additivity_r2` distribution for the transformer, and therefore how much of it the
      effective-curve comparison explains (A4).
- [ ] Whether the confound matrix separated Grokfast from the training fraction, or whether the cells
      were too few to do so.
- [ ] Whether the parameter-matched and two-hot controls changed any conclusion, and by how much.
- [ ] Ablations that did **not** damage the model (these are findings and must be listed).
- [ ] Any hypothesis refuted, with the number that refuted it.
- [ ] Any run that failed or was excluded, with the reason and a sensitivity analysis without it.
- [ ] Any analysis that had to change after the code freeze, with its labbook entry.
