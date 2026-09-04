# Novelty and related work

**AI-drafted (Claude), 2026-09-03 — not yet human-reviewed.** Master prompt §2 and §12: state precisely
what each source already establishes and what it leaves unanswered, and frame H1 and the MLP half of H2
explicitly as replication. Locations and quotes come from the notes under `docs/sources/`; each source
carries its note's verification status (`V` verified, `V*` partially verified, `U` unverified).

**The short version.** Almost everything this study measures has been observed before by somebody. What
has not been done is measuring it **the same way in both architectures, with causal tests, and asking
whether the metric itself is valid**. If the evidence does not support even that, the honest result is a
negative one.

---

## 1. Nanda, Chan, Lieberum, Smith, Steinhardt (2023) — arXiv:2301.05217 `V*`

**What it establishes.** A 1-layer transformer on modular addition mod 113 represents each operand by
Fourier features at a small set of key frequencies and combines them with the angle-addition identity,
reading out `Logit(c) = Σ_k α_k cos(ω_k(a+b−c))` (§4). Key frequencies are read from the DFT of the
neuron→logit map `W_L = W_U W_out`, and their **count is measured** — 5 for their main model, 3–4 for
other seeds (App. C.2). Restricted and excluded loss project the logits onto (or out of) the
`cos/sin(ω_k(a+b))` directions only, with excluded loss measured on the training pairs (§5.1). The three
phases are memorization, circuit formation and cleanup.

**What it leaves unanswered.** Everything about the MLP: the paper studies one architecture. It gives no
comparison, no waveform analysis (sinusoid versus harmonics), and no statement about whether the
concentration of a representation is a *valid* cross-architecture metric. The numeric threshold for a
"non-trivial" key-frequency coefficient is not published, and neither is the data split of its own
restricted-loss figure — both `[NOT FOUND IN SOURCE]`.

**Our relation.** H1 is a **replication** of this mechanism in our two architectures and training
setting. Our restricted/excluded implementation was **wrong** until now and is corrected against this
source (`docs/MASK_PROTOCOL_AUDIT.md`); that correction is a fix, not a contribution.

## 2. Power, Burda, Edwards, Babuschkin, Misra (2022) — arXiv:2201.02177 `V`

**What it establishes.** Grokking itself: generalization long after memorization on small algorithmic
datasets, controlled by weight decay and the training fraction.

**What it leaves unanswered.** Any mechanism. It is a phenomenon paper.

**Our relation.** The phenomenon we measure; the definition we use for "grokked". Not a contribution.

## 3. Lee, Kim, Choi, Kim (2024), Grokfast — arXiv:2405.20233 `V`

**What it establishes.** Amplifying the slow (EMA) component of the gradient accelerates grokking.

**What it leaves unanswered.** Whether the *solution found* under acceleration is the same one.

**Our relation.** Used only inside the confound matrix, never in the primary setting, precisely because
that question is open. The legacy repository conflated acceleration with the training fraction; the
matrix separates them.

## 4. Manir & Rupa (2026) — arXiv:2603.25009 `V`

**This is the source that most constrains what we may call new.**

**What it establishes.** A controlled study of grokking across depth, architecture, activation and
regularization at `p = 97`. Three findings bear directly on us:

1. **The concentration gap is already published.** §5.7 / Table 8: "the Transformer concentrates 98.5 %
   of its embedding energy in just 5 frequencies, compared to 74.7 % for MLP-GELU and 75.6 % for
   MLP-ReLU." We replicate the *direction* under a different protocol, but the **size depends on when
   it is measured**, which is itself worth recording. The 0.73 versus 0.44 quoted in the superseded
   `RESULTS.md` came from early-stopped runs read at the generalization crossing. Measured at the
   25,000-step budget with no early stopping, over 10 seeds, embedding top-8 concentration is
   **0.961** (transformer, range 0.950–0.971) versus **0.891** (MLP, range 0.772–0.960) — the same
   direction, a gap of 0.07 rather than 0.29. This is the Khanh 2026 overstatement measured on our own
   data. Note also that the top-8 *cap* binds on 6 of 10 MLP runs and 0 of 10 transformer runs, so for
   the MLP the metric is partly reporting the cap rather than the network.
2. **The attention interpretation is already published.** §5.7: self-attention "acts as an implicit
   sparsity-promoting mechanism in the frequency domain". The sentence in our `RESULTS.md` saying
   attention steers the transformer toward a cleaner circuit is the same claim.
3. **The timing gap is protocol-dependent, and their direction is the opposite of ours.** They find the
   **MLP faster** in every multi-seed comparison (1.11×, 1.90×, 1.42×) under per-architecture optimizers,
   a depth-4 MLP and `train_frac 0.20`; we find the transformer faster under one optimizer, a depth-2
   MLP and `train_frac 0.30`. They also report the transformer's seed variance to be 4–6× the MLP's,
   which matches our legacy ±1196 versus ±410.

**What it leaves unanswered.** Their Fourier table is **one seed with no variance**, its top-5 metric is
fixed a priori exactly as our top-8 was, and it **includes the DC mode** (frequency 0 appears in the
transformer's top-5), which ours excludes — so the two numbers are not directly comparable. They run no
per-neuron mechanism analysis, no waveform fitting, no causal ablation, and no test of whether the
concentration metric is valid; they list mechanistic interpretability as future work (§6.4). They also
name "fully isolating architecture from optimization" as an open challenge (§6.3).

**Our relation.** We must (a) never present the concentration gap or the timing gap as a discovery,
(b) cite them when reporting either, (c) state that the timing direction is protocol-dependent and that
ours disagrees with theirs, and (d) restrict our timing claim to "under the conditions examined". What
remains ours is the same-optimizer, same-λ, shared-embedding, multi-seed design they did not run.

## 5. Swaroop (2026) — arXiv:2603.23784 `V`

**What it establishes.** In a **two-hot input, 256-hidden ReLU MLP at `p = 97`**: near-binary square-wave
first-layer weights, with intermediate values only near sign changes; the phase-sum relation
`φ_out = φ_a + φ_b` (which the paper itself attributes to Nanda et al. 2023 and Gromov 2023); a named
periodicity score (Eq. 1) with post-hoc bimodal cuts at 12 and 5; and an idealized reconstruction that
reaches 95.5 % accuracy from parameters extracted from a model that itself scores 0.23 %.

**What it leaves unanswered.** It has **no transformer runs** — its "MLP square versus transformer sine"
contrast is a citation of Nanda, not a measurement. It says nothing about a *learned shared embedding*
(its input is raw two-hot), and it **never mentions harmonics** or a `1/j` amplitude law. It states its
own findings "may be regime-dependent" (§5).

**Our relation.** The MLP half of H2 is a **replication** of the square-wave observation, in a different
architecture (shared embedding plus concatenation rather than two-hot) — which is exactly why we also run
a two-hot control. H1's phase-sum relation is a replication with an origin that predates this paper. The
odd-harmonic and `1/j` predictions are **not from this source**; they are textbook Fourier series,
derived in `docs/MLP_MECHANISM_DERIVATION.md` §5 and verified numerically. Attributing them to Swaroop
would be a citation error.

## 6. Doshi, Das, He, Gromov (2023) — arXiv:2310.13061 `U`

**What it establishes.** On corrupted algorithmic datasets: networks can memorize corrupted labels and
generalize simultaneously; memorizing neurons are identifiable and removable by pruning; regularization
pushes neurons toward generalizing representations. It supplies the inverse participation ratio (Eq. 3–4)
as a per-neuron periodicity statistic, used as a **ranking with no threshold**, and a cumulative
pruning protocol in both ranking directions on an unmodified checkpoint.

**What it leaves unanswered.** It has **no size-matched random control** for the pruning sweep, so its
protocol alone does not answer our H5. Its causal evidence is for a 2-layer quadratic MLP with one-hot
inputs, MSE loss and no biases; its transformer and deeper-MLP results are histogram evidence, not
ablations. It does not corrupt-label-free-compare architectures the way we do.

**Our relation.** We adopt the IPR definition and the pruning sweep verbatim, and **add the random
control the paper lacks**. We may not call low-IPR neurons "memorizing" on this paper's authority, since
we do not corrupt labels — only "non-periodic".

## 7. Khanh (2026) — arXiv:2607.06639 `V`

**What it establishes.** Reading a representation metric at the grokking transition overstates its
converged value by 3–5× on an MLP and 1.3–1.5× on a transformer, with a lag between the accuracy
transition and completed compression of order 10⁴ steps. LayerNorm affects compression timing. It
proposes an audit distinguishing onset from compression.

**What it leaves unanswered.** It audits effective rank and related metrics, not Fourier concentration
specifically, and does not compare architectures mechanistically.

**Our relation.** A **method we adopt, not a finding of ours**: it is why every structure metric is
reported at two declared points and why "converged" is a tested claim rather than a synonym for "final".
It also supplies the most serious alternative explanation for the legacy 0.73-versus-0.44 gap — that the
two architectures had done different amounts of cleanup by their own crossings.

## 8. McCracken et al. (NeurIPS 2025) — arXiv:2505.18266 `V`

**What it establishes.** A unifying account of modular-addition mechanisms that trains **exactly our
architecture family**: "concatenated input pairs", one-hot or with a 128-dimensional learned embedding,
1–4 layers. Crucially (§4.4): "adding either depth, or a trainable embedding matrix for inputs, causes
the network to transition from learning ⌊n/2⌋ types of neurons to much fewer types" — so our
shared-embedding MLP is *predicted* to learn few frequencies and our two-hot control many.

**What it leaves unanswered.** Harmonics and square waves are absent from the paper; it reports only
"secondary spikes" in the DFT that fade with width. It uses R² of simple-neuron fits and DFT peak counts,
never examines Fourier-concentration metrics for validity, and works at moduli 59–66 without our
strong-weight-decay grokking regime.

**Our relation.** It removes any claim that a concatenation MLP is an unusual object, and it supplies a
sharp prediction our two controls can check. Its width caveat is also a warning: our 0.44 could partly be
a finite-width secondary-spike effect, which the parameter-matched `d_mlp = 572` control speaks to.

**Related mechanism papers cited by id only** (from the same note, not independently read): Chughtai et
al. 2023 (group-theoretic), Zhong et al. 2023 (clock/pizza), Gromov 2023 (analytic solutions), Morwani
et al. 2023 (Fourier features and margin), Moisescu-Pareja et al. arXiv:2512.25060 (representation
geometry; states MLP-Concat is "strongly separated from all others"), He et al. arXiv:2602.16849.

---

## 9. What GROKVERSE does **not** claim as new

| claim | already established by |
|---|---|
| networks solve modular addition with Fourier features and phase addition | Nanda 2023; Gromov 2023 |
| ReLU MLP input weights look like square waves | Swaroop 2026 |
| the phase-sum relation `φ_out = φ_a + φ_b` holds per neuron | Nanda 2023; Gromov 2023; replicated by Swaroop 2026 |
| a grokked transformer's embedding is more Fourier-concentrated than an MLP's | Manir & Rupa 2026 |
| attention plausibly promotes frequency sparsity | Manir & Rupa 2026 |
| the transformer-vs-MLP timing gap moves with the protocol | Manir & Rupa 2026 |
| concatenation MLPs implement this family of mechanisms; a trainable embedding reduces the frequency count | McCracken et al. 2025 |
| metrics read at the transition overstate the converged value | Khanh 2026 |
| IPR-ranked pruning separates periodic from non-periodic neurons | Doshi et al. 2023 |
| grokking itself | Power et al. 2022 |

## 10. What GROKVERSE contributes, **if** the evidence supports it

Each item is stated with the wording that would be permitted, and each is conditional on the evidence
gate (`docs/PREREGISTRATION.md` §5).

1. **The same mechanism battery in both architectures.** No consulted source runs the
   sinusoid-versus-harmonics comparison on a transformer. Permitted wording if it holds: *"Under the
   conditions examined, the transformer's effective operand curves are better described by a single
   sinusoid than the MLP's, which favour an odd-harmonic model."*
2. **The shared-embedding concatenation MLP**, with a two-hot control that varies only the input
   parametrization. Permitted: *"The square-wave-like structure reported for two-hot ReLU MLPs also
   appears / does not appear when the MLP reads its operands through a learned shared embedding."*
3. **Causal necessity and sufficiency with size-matched controls, in both architectures.** Doshi supplies
   the sweep but not the control; Nanda's restricted loss is a sufficiency test but was never run against
   a matched control here. Permitted: *"Removing the structured components damages the model markedly
   more than a size-matched random ablation"* — or its negation, which is equally reportable.
4. **H3, the metric-validity question.** Whether top-k Fourier concentration misclassifies a harmonic
   representation as less structured. No consulted source asks this. Permitted if supported: *"The
   results are consistent with the same Fourier principle being expressed through different internal
   representations."* Never: "the same circuit in different bases".
5. **The first multi-seed estimate of the concentration gap** — Manir & Rupa's Table 8 is a single seed.
   Permitted: *"first multi-seed estimate"*, never *"first observation"*.

## 11. Delineation table

| claim | in the literature? | GROKVERSE status |
|---|---|---|
| Fourier phase-addition mechanism (transformer) | yes — Nanda 2023 | replication |
| Fourier phase-addition mechanism (shared-embedding MLP) | partly — Gromov/Swaroop for other MLPs | replication in a new architecture |
| square-wave MLP weights | yes — Swaroop 2026 (two-hot) | replication in a new architecture |
| odd harmonics, `1/j` decay | textbook, no paper needed | derived here |
| concentration gap between architectures | yes — Manir & Rupa 2026 (1 seed) | replication; first multi-seed estimate |
| timing gap direction | yes, and opposite under their protocol | replication of protocol-dependence; not a claim of ours |
| sinusoid-vs-harmonic comparison on a **transformer** | no | **extension** |
| causal ablation with size-matched controls in both architectures | no | **extension** |
| is top-k concentration a valid cross-architecture metric | no | **the proposed primary contribution (H3)** |
| a negative result, if the gate fails | — | reportable and pre-committed |
