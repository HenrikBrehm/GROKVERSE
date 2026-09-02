# GROKVERSE — Master Prompt: Mechanistic Comparative Study of Transformer vs. MLP

## 0. Role and Assignment

You are working in the repository `HenrikBrehm/GROKVERSE`.
Act as a critical research engineer for mechanistic interpretability **and** as an adversarial scientific reviewer.

Your task is NOT to extend GROKVERSE with further visualizations or impressive-sounding text.
Your task is to turn the existing reproduction and visualization project into a methodologically sound mechanistic comparative study between Transformer and MLP.

### Research question (neutrally phrased — not mechanism-assuming)

> Both models generalize on modular addition. **Do they use a Fourier-based phase-addition mechanism, and if so, do they represent and compute it differently?**

This formulation explicitly leaves open that one or both architectures use a different mechanism. Never rephrase the research question in a way that presupposes a shared Fourier algorithm.

The focus is no longer primarily on reproducing grokking. The independent scientific contribution should be the controlled architecture comparison — in particular the question of whether the structural metrics used are themselves valid (see Section 12, H3).

---

## 1. Starting Point in the Repository

The repository already contains:

- a reproduction of grokking on modular addition modulo 113,
- a 1-layer ReLU Transformer,
- a 2-layer ReLU MLP with shared embedding,
- multiple seeds,
- accelerated Grokfast runs,
- unaccelerated runs,
- Fourier and PCA analyses of the embeddings,
- attention evaluations,
- restricted-loss and excluded-loss evaluations,
- an interactive explorer,
- an observed difference between Transformer and MLP:
  - Transformer: higher top-8 Fourier concentration,
  - MLP: lower top-8 Fourier concentration,
  - Transformer: earlier generalization in the runs so far.

This observation is interesting, but not yet a sufficient mechanistic explanation.

In particular, the statement "the MLP learns a less structured or more distributed solution" is not yet adequately supported. The current value is determined mainly from the embedding matrix `W_E`. In the MLP, however, the actual neuron-specific input only arises from the combination of `W_E` and `W_in`.

For an MLP neuron `i`, at minimum the following effective functions must therefore be analyzed:

```
u_a[a, i]            = W_E[a] @ W_in[:d_model, i]
u_b[b, i]            = W_E[b] @ W_in[d_model:, i]
preactivation[a,b,i] = u_a[a, i] + u_b[b, i] + b_in[i]
activation[a,b,i]    = ReLU(preactivation[a, b, i])
output_curve[c, i]   = W_out[i, c]
```

The scientifically relevant question is therefore not only "how much Fourier power is contained in `W_E`?", but:

> Which periodic functions are actually implemented by the combination of embedding, input weights, activations, and output weights — and do they causally carry the model output?

---

## 2. Literature and Primary Sources

For methodological statements, use only primary sources. For every method you adopt, document the exact source, section, equation, or figure.

At minimum, the following must be taken into account:

- **Nanda et al., arXiv:2301.05217** — progress measures, restricted/excluded loss, trigonometric identities.
- **Power et al., arXiv:2201.02177** — the grokking phenomenon.
- **Grokfast, arXiv:2405.20233** — acceleration.
- **Swaroop, arXiv:2603.23784** — already reports square-wave-like ReLU MLP input weights and the phase-sum relationship. Remove the `[UNVERIFIED]` label only after the relevant sections and methods have been concretely recorded.
- **Manir and Rupa, arXiv:2603.25009** — already compares Transformer and MLP embedding Fourier concentration as well as grokking times under controlled conditions. **Consequence: merely observing a concentration gap or a timing gap is NOT the new contribution of this study.**
- **Doshi et al., arXiv:2310.13061** — already analyzes periodic MLP representations and causal neuron pruning, including Transformer and deeper-MLP evidence.
- **Khanh, arXiv:2607.06639** — addresses the danger of interpreting representation metrics at the grokking transition rather than after convergence.
- Check and add the closest NeurIPS 2025 work on unifying modular-addition mechanisms, in particular its treatment of concatenation-based MLPs.

The related-work section must explain precisely **what these works already establish** and **what they leave unanswered**. If a reference cannot be verified, mark it as `[UNVERIFIED]` and do not treat it as evidence.

---

## 3. Scientific Integrity Rules (binding)

1. Do not invent results, values, or completed experiments.
2. Always separate clearly between: **planned**, **implemented**, **tested**, **empirically confirmed**, **not confirmed**, **refuted**.
3. Every number in `RESULTS.md`, in tables, or in figures must be traceable to a concrete run, seed, commit, and configuration file.
4. Negative and unexpected results must be preserved.
5. Existing results must not be overwritten or silently reinterpreted.
6. Before major changes, establish a reproducible baseline state: document the git commit, archive current results, do not delete existing figures.
7. Only call an analysis a reproduction of Nanda if the actual protocol matches the primary source.
8. Deviations from the paper must be visibly labeled as your own methodological variant.
9. Do not retroactively tune thresholds or metrics so that the desired result looks stronger.
10. Primary hypotheses, metrics, and thresholds must be fixed before the full new experiments.
11. The final scientific motivation and interpretation must not be falsely presented as original human work. AI-generated drafts must remain labeled as such until the human authors have independently revised them.
12. The explorer is updated only after the scientific analyses have been frozen and reviewed. Visualization must not drive the research.

---

## 5. Phase 1 — Evidence Audit

Examine at minimum:

- `README.md`, `RESULTS.md`, `PLAN.md`, `PROGRESS.md`, `AI_DISCLOSURE.md`, `RESEARCH_SPEC.md` (if present),
- `training/grokverse/models/mlp.py`,
- `training/grokverse/models/transformer.py`,
- `training/grokverse/analysis/fourier.py`,
- `training/grokverse/analysis/progress_measures.py`,
- `training/grokverse/analysis/compare.py`,
- `training/grokverse/train.py`,
- `training/test_core.py`,
- existing run metadata and figures.

Create `docs/CURRENT_EVIDENCE_AUDIT.md`. The file should classify every central claim from `README.md` and `RESULTS.md` in a table:

| Claim | Existing evidence | Seeds | Metric | Alternative explanation | Methodological limitation | Evidence level | Status |

- Evidence level: **descriptive**, **correlational**, **mechanistic**, **causal**.
- Status: **solid**, **preliminary**, **overinterpreted**, **methodologically problematic**.

Treat the following statements especially critically:

- "The Transformer learns a sparser Fourier circuit."
- "The MLP uses a more distributed solution."
- "The attention structure explains the earlier generalization."
- "Restricted and excluded loss reproduce Nanda's method."
- "0.73 versus 0.44 shows different algorithms."
- "50/50 attention is proof of the addition mechanism."
- "Both models learn the same function." (see Section 10 — until measured, only "both generalize on the same task")

A symmetric attention distribution alone is not yet sufficient causal evidence.

---

## 6. Phase 2 — Mask Protocol (Restricted / Excluded Loss)

Examine the current implementation in `training/grokverse/analysis/progress_measures.py`.

The current broad mask

```python
mask_restr = keep_restr[:, None] & keep_restr[None, :]
```

could retain cross-frequency blocks between different key frequencies. The excluded mask could correspondingly remove substantially more components than the original method.

**Treat this initially as a suspected deviation, not automatically as a proven error.**

1. Reconstruct the exact Nanda protocol from the primary source and, if available, the official code.
2. Document in `docs/MASK_PROTOCOL_AUDIT.md`:
   - which axes are transformed,
   - which Fourier components are kept,
   - which components are removed,
   - whether only same-frequency blocks are permitted,
   - how sine and cosine pairs are handled,
   - how the constant term is handled,
   - how key frequencies are determined,
   - on which data split restricted loss is computed,
   - on which data split excluded loss is computed,
   - what differences exist relative to the current implementation.
3. Implement separate, unambiguously named variants:
   - `nanda_exact_restricted_loss`
   - `nanda_exact_excluded_loss`
   - `full_grid_extension_restricted_loss`
   - `full_grid_extension_excluded_loss`
   - optionally the previous broad mask as `legacy_broad_mask_variant`
4. The legacy variant must no longer be described as a direct reproduction.
5. Write tests for:
   - correct Fourier round-trips,
   - the exact number of retained components (mask cardinality),
   - no impermissible cross-frequency blocks, if this is confirmed by the source,
   - correct handling of the constant term,
   - artificially injected single frequencies,
   - artificially injected mixed frequencies,
   - restricted loss of an ideal Fourier circuit,
   - excluded loss of that same ideal circuit,
   - separate evaluation of the train and test splits,
   - agreement between the vectorized and the reference implementation.
6. Until these tests pass, the existing restricted/excluded numbers must be marked in the documentation as **"not directly comparable to Nanda"**.
7. The selection of key frequencies must also be checked: embedding spectrum, logit map, neuron-logit map, exact procedure from Nanda.
8. Do not implement any desired interpretation before this methodological question is resolved.

---

## 7. Phase 3 — MLP Mechanism Analysis

Create a new module, for example `training/grokverse/analysis/mlp_mechanism.py`.

For every hidden neuron, compute:

- the effective weight curve for operand `a`,
- the effective weight curve for operand `b`,
- the bias,
- the output weight curve over classes `c`,
- the 2D preactivation map,
- the 2D activation map,
- the neuron's contribution to each output logit.

Save numerical results, not just figures.

Per neuron, compute:

- the dominant frequency of `u_a`, `u_b`, and the output curve,
- Fourier power per frequency,
- spectral entropy,
- participation ratio (or inverse participation ratio following Doshi et al.),
- top-1, top-3, and top-8 concentration,
- the share of energy in the fundamental,
- the share in odd harmonics,
- the share in even harmonics,
- periodicity score (cf. Swaroop),
- the phases of the dominant frequencies.

Investigate whether the dominant frequencies of the a-side, the b-side, and the output side agree.

### Waveform fitting (`analysis/wave_fitting.py`)

Fit at least the following models to the effective weight curves:

1. Sinusoid model: `alpha * cos(2*pi*k*n/p + phi) + beta`
2. Ideal square wave: `alpha * sign(cos(2*pi*k*n/p + phi)) + beta`
3. Finite odd-harmonic approximation: `beta + sum_j alpha_j * cos(2*pi*(2j+1)*k*n/p + phi_j)`

Compare the models using:

- normalized MSE,
- explained R²,
- AIC or a comparable complexity correction,
- cross-validation across token positions, where meaningful,
- sensitivity analysis across several frequencies.

Do not report only the best fit. Show the distribution across all neurons and seeds.

### Phase relationship

For neurons with a matching dominant frequency, check the relationship:

```
phase_output ≈ phase_a + phase_b   (mod 2*pi)
```

Compute: circular phase error, median, mean, confidence intervals, and a null distribution obtained by permuting the neuron assignments.

### Activation analysis

Analyze the hidden activation of every neuron over the full `(a,b)` grid:

- 2D Fourier spectrum,
- strongest frequency pairs,
- diagonal structure,
- dependence on `a+b`,
- dependence on `a-b`,
- symmetry under exchange of `a` and `b`,
- the share of activation variance explained by the hypothesized periodic function.

Document the derivation in `docs/MLP_MECHANISM_DERIVATION.md`.

---

## 8. Phase 4 — Transformer Mechanism Analysis

Create a module, for example `training/grokverse/analysis/transformer_mechanism.py`.

Do not examine only `W_E` and average attention. Analyze at minimum:

- `W_E`, `W_U`, `W_Q`, `W_K`, `W_V`, `W_O`, `W_in`, `W_out`,
- position-dependent residual streams,
- the hidden activations of the Transformer MLP,
- the contribution of individual heads,
- the contribution of individual MLP neurons to the output logits.

Derive correctly from the concrete model architecture how the contribution of a single MLP neuron to the output logits is computed. Document the derivation in `docs/TRANSFORMER_MECHANISM_DERIVATION.md`.

Compute:

- the Fourier spectrum of the input directions,
- the Fourier spectrum of the output directions,
- activation maps over `(a,b)`,
- the variance explained by the key-frequency subspaces,
- phase relationships,
- agreement with the trigonometric identities described in Nanda.

Analyze attention in a differentiated way: per head, per input, by training phase, including variance and not just the mean, with causal head ablation.
Treat a 50/50 distribution only as a descriptive finding as long as no causal evidence exists.

**Important:** The Transformer must undergo the same mechanism tests as the MLP, in particular the sinusoid-versus-harmonics comparison. It is not permissible to assume the Fourier mechanism as given for the Transformer and to test only the MLP.

---

## 9. Phase 5 — End-to-End Logit Formula Fit (mandatory)

Weight-level periodicity alone does **not** establish that the entire model implements the hypothesized algorithm. The previous work packages fit waves to weights and effective neuron curves, but not to the full model output.

Add an analysis that fits candidate formulas to the actual logits across **all** `(a,b,c)` combinations — for both architectures.

Compare at minimum:

- a single-frequency or sparse-sinusoid Fourier formula,
- an odd-harmonic or square-wave-like formula,
- a suitable control model with comparable flexibility (matched flexibility).

Report explained variance or R², residuals, and results for **every** seed. Without this fit, it must not be claimed that the model implements the proposed formula.

---

## 10. Phase 6 — Functional Equivalence Over the Full Domain (mandatory)

High test accuracy does not mean that two trained models implement exactly the same function. Add a comparison across all 12,769 input pairs:

- prediction agreement between the architectures,
- the share of inputs both get correct,
- inputs on which only one architecture is correct,
- the overlap and structure of the errors,
- logit or margin similarity, where meaningful.

Until this has been measured, use exclusively the formulation **"both architectures generalize on the same task"**, never "both learn the same function".

---

## 11. Phase 7 — Causal Ablations

Descriptive plots are not sufficient. Implement at least **two causal tests per architecture**, each as a necessity **and** a sufficiency test.

### Possible MLP ablations

1. Keep only neurons classified as structured.
2. Keep only unstructured neurons.
3. Remove structured neurons individually or in groups.
4. Remove key frequencies from the effective weight curves.
5. Keep only key frequencies.
6. Replace effective weights with their best sinusoid fits.
7. Replace effective weights with their best square-wave fits.
8. Compare neurons with the correct frequency but an incorrect phase relationship.
9. Compare neurons with the correct phase but weak periodicity.

### Possible Transformer ablations

1. Keep or remove key Fourier components.
2. Keep or remove MLP neurons with high key-frequency explanation.
3. Ablate individual attention heads.
4. Project OV or QK components.
5. Remove key-frequency subspaces from the residual stream.
6. Keep only the reconstructed trigonometric circuit.

### Measurements after each ablation

Train loss, test loss, train accuracy, test accuracy, change in the logits, change in the margin, change per class, change across several seeds. Report absolute **and** relative changes.

### Ablation rules

- Ablations must be run on unmodified checkpoints.
- No retraining steps, unless they are explicitly a separate study.
- Every ablation needs a matching **size-matched random control ablation**.
- Repeat the central ablations across several seeds.
- Also show cases in which a hypothesized structured component is removed **without** a large drop in performance.
- Measure separately at the generalization transition and in the final converged state (cf. Khanh, arXiv:2607.06639).

Document this in `docs/CAUSAL_ABLATION_PLAN.md`.

---

## 12. Hypotheses, Evidence Gate, and Decision Tree

### Positioning: replication vs. original contribution

The existence of Fourier structure, phase addition, and square-wave-like MLP weights must **not** be presented as a new discovery.

- **H1** and the MLP part of **H2** are to be framed as a **replication** in the GROKVERSE architectures and the GROKVERSE training setting.
- The extensions are:
  - applying comparable mechanism tests to **both** architectures,
  - testing the **Transformer side** of the sinusoid-versus-harmonics comparison,
  - analyzing the **shared-embedding concatenation MLP** rather than only a direct two-hot MLP,
  - clarifying whether the identified structure is **causally** responsible for the output.

### H3 as the proposed primary contribution

The central question, simply put:

> **Does top-k Fourier concentration misclassify a structured harmonic representation as less structured?**

The existing metric may favor a sinusoid-like representation whose power concentrates on a few individual frequencies over a square-wave-like representation whose equally organized signal is spread across the fundamental and its associated harmonics.

H3 should test whether the MLP's lower concentration is a **genuine loss of structure** or a **measurement artifact** caused by different waveforms.

**Mandatory controls for H3:** a cardinality-matched top-m control and a random-set control. Without these, the harmonic-family score could rise simply because it includes more frequencies.

### Evidence gate before H3 (binding)

H3 may **only** be interpreted as evidence for "the same Fourier principle in different representations" if **each architecture independently** demonstrates the following:

1. periodic structure in the relevant weights or activations,
2. the predicted phase-addition relationship,
3. a good end-to-end fit between the candidate Fourier formula and the model logits,
4. causal necessity and sufficiency.

State this explicitly in the specification:

> **Dependence of H3 on the mechanism tests:** H3 does not presuppose that either architecture uses a Fourier circuit. Each architecture is first tested independently for periodic internal structure, the predicted phase relationship, end-to-end logit fit, and causal relevance. Harmonic-family concentration may be interpreted as the same Fourier principle in a different form only for an architecture that passes these tests.

### Weakened H3 conclusion

Do **not** use: "The two architectures use the same circuit in different bases."

Use instead: **"The results are consistent with the same Fourier principle being expressed through different internal representations."**

Even if both models pass the tests, "the same circuit" is too strong, because the same mathematical rule can be realized through different internal operations.

### Pre-specified outcome-dependent decision tree

Fix **before** the experiments what happens for each possible outcome:

- **Both architectures pass the Fourier mechanism tests:** investigate H3 and compare how the Fourier principle is represented and computed.
- **Only one architecture passes:** conduct a bounded investigation of the other architecture's representation.
- **Neither architecture passes:** conclude that the Fourier evidence tested does not identify the learned mechanisms, and investigate the validity of the existing metrics.

This prevents the analysis from silently assuming that H3 must hold.

### Bounded alternative-mechanism analysis

If an architecture fails the Fourier tests, investigate one level further instead of ending with "may use something else". The bounded follow-up tests:

- whether hidden activations primarily encode `(a+b) mod 113` even though no obvious Fourier concentration is present,
- whether activation or logit tensors contain low-rank, symmetry-based, or other compact structure,
- whether the identified structure occurs across seeds,
- whether removing this structure causally damages performance.

The project does **not** promise to discover every possible alternative algorithm. A well-founded negative result remains valid if the bounded analysis identifies none.

### Further hypothesis types to be examined

- Both models use Fourier-related representations and ultimately implement a phase addition. *(claimable only after the evidence gate has been passed)*
- The Transformer forms a sparser sinusoidal representation, while the ReLU MLP forms more harmonic or square-wave-like effective weights.
- The lower top-8 concentration of the MLP embedding matrix does not automatically mean less algorithmic structure; a substantial part of the structure may only become visible in `W_E @ W_in`, in the hidden activations, or in the output weights.
- Mechanistic structure metrics rise already during the memorization plateau and predict the later generalization jump.
- Removing the identified structured components damages the solution markedly more than a size-matched random control ablation.

For every hypothesis, also formulate a null hypothesis and clear possible refutation criteria. Do not try to confirm all hypotheses at any cost.

---

## 13. Definition of "Structured Neuron" — Use Published Measures

Do not reinvent the definition of a structured neuron from scratch. Compare and use, where appropriate:

- Swaroop's periodicity score,
- the inverse participation ratio of Doshi et al.,
- the proposed variance-explained and frequency-matching conditions.

The final definition and the thresholds must be fixed **before** the full runs, justified from the source or from the pilot, and accompanied by a sensitivity analysis. The same conceptual standard should be applied across architectures wherever the internal objects are comparable.

---

## 14. Controlled Architecture Comparison

The comparison must be set up as a controlled study. For Transformer and MLP, use:

- the same seeds,
- the same train/test splits,
- the same modular task,
- the same prime,
- the same training fraction,
- the same accuracy thresholds,
- the same evaluation frequency,
- and, where meaningful, the same optimizer and the same training steps.

Analyze the results **paired by seed**.

Recommended standard:

- modulo 113, addition,
- `train_frac = 0.3`,
- `weight_decay = 1.0`,
- no Grokfast,
- at least 8 paired seeds, target 10.

Individual failed runs must not be removed without justification.

### Factorial control matrix (Grokfast × train_frac)

The previous accelerated and unaccelerated conditions differ simultaneously in Grokfast and in the training fraction. Therefore build:

- no Grokfast, `train_frac 0.3`
- no Grokfast, `train_frac 0.5`
- Grokfast, `train_frac 0.3`
- Grokfast, `train_frac 0.5`

For the control matrix, fewer seeds than in the primary setting suffice, but the seeds must be paired across architectures. Do not attribute any change exclusively to Grokfast or exclusively to `train_frac` as long as the two effects have not been separated.

### Parameter and capacity confounds

Compute and document: total parameter count, trainable parameters, parameters per module, initial weight norms, final weight norms, effective strength of the weight decay.

In addition to the comparison with identical dimension values, run a **parameter-matched control comparison** (deviation ideally ≤ 5%).

Distinguish clearly between "same hyperparameter dimensions" and "similar parameter count".

### Input parametrization as a confound

Optionally implement, as a literature control, a direct two-hot-input MLP following the MLP paper (Swaroop). This model is not the main comparison; it answers:

> Does the square-wave structure persist when the MLP uses a learned shared embedding instead of the direct two-hot input?

Compare: two-hot MLP, shared-embedding MLP, Transformer. Do not automatically label differences as an architecture effect if they could be caused by the input parametrization.

---

## 15. Checkpoints and Timing Measurement

The current pipeline mainly saves embeddings and the final model. Optionally extend it so that full model states are saved at **pre-specified** checkpoints. Do not save unnecessarily large files at every step.

Fixed checkpoints: initialization, shortly before memorization, memorization, mid-plateau, shortly before generalization, generalization, after convergence.

Track over time: Fourier concentration, square-wave fit, sinusoid fit, phase error, share of structured neurons, causal contribution of structured neurons, weight norms.

The checkpoints must not be selected after the fact so that they better display the desired narrative.

### Transition measurement

The current logarithmic checkpoint structure may yield only the first observed checkpoint above a threshold. Improve the measurement:

- train accuracy regularly, e.g. every 10 steps,
- test accuracy regularly, e.g. every 25 or 50 steps,
- full checkpoints only at a few planned points in time,
- alternatively, a deterministic re-measurement within a previously identified transition interval.

For every transition, report: the first evaluated crossing, the previous evaluated step, the resulting uncertainty interval, and the evaluation frequency.

Do **not** use the phrase "exact transition" when only a discrete evaluation grid is available.

Primary thresholds: memorization at train accuracy ≥ 0.99, generalization at test accuracy ≥ 0.95. Run a sensitivity analysis with at least one further threshold.

---

## 16. Statistics

Create `training/grokverse/analysis/statistics.py` and `docs/STATISTICAL_ANALYSIS_PLAN.md`.

Primary analyses:

- paired difference in the generalization step,
- paired difference in the grokking gap,
- paired difference in the structure metrics,
- paired difference in the ablation damage,
- bootstrap confidence intervals,
- robust measures of location and spread,
- display of all individual seeds.

P-values may be used, but must not be the only evidence. Additionally report: effect size, direction of the effect, spread, confidence interval, number of successful runs, number of all started runs.

For small samples: no spurious precision, no exaggerated general claims, no statement that "architecture X is fundamentally faster" when only a narrow hyperparameter range has been examined.

---

## 17. Comparability of the Metrics

Where mathematically meaningful, use the same structure metrics for Transformer and MLP:

spectral entropy, participation ratio, top-k concentration, number of dominant frequencies, phase error, explained variance, ablation damage, timing of structure formation.

Avoid unfair comparison of quantities that are defined differently in the two architectures. Document such cases explicitly.

---

## 18. Files to Be Created

Adapt the exact paths to the existing repository structure.

**Documentation:**

- `docs/CURRENT_EVIDENCE_AUDIT.md`
- `docs/PREREGISTRATION.md` — including the evidence gate and decision tree from Section 12
- `docs/METHODS.md`
- `docs/MASK_PROTOCOL_AUDIT.md`
- `docs/MLP_MECHANISM_DERIVATION.md`
- `docs/TRANSFORMER_MECHANISM_DERIVATION.md`
- `docs/STATISTICAL_ANALYSIS_PLAN.md`
- `docs/CAUSAL_ABLATION_PLAN.md`
- `docs/NOVELTY_AND_RELATED_WORK.md` — delineation against Manir/Rupa, Swaroop, Doshi, Khanh
- `docs/LIMITATIONS.md`
- `docs/CLAIM_EVIDENCE_TABLE.md`
- `docs/HUMAN_INTERPRETATION_TEMPLATE.md`
- `docs/LABBOOK_TEMPLATE.md`

**Analysis code:**

- `analysis/mlp_mechanism.py`
- `analysis/transformer_mechanism.py`
- `analysis/wave_fitting.py`
- `analysis/logit_formula_fit.py` — end-to-end fit (Section 9)
- `analysis/function_agreement.py` — full-domain comparison (Section 10)
- `analysis/causal_ablation.py`
- `analysis/statistics.py`
- `analysis/mask_protocols.py`

---

## 19. Tests

Extend the tests to cover at least:

- mask cardinality,
- no impermissible cross-frequency components,
- a synthetic ideal Fourier circuit,
- a synthetic square wave,
- phase recovery,
- neuron-specific effective MLP weights,
- causal ablation of an artificial model,
- end-to-end logit fit on a synthetic model with a known formula,
- full-domain agreement on two artificially identical and two artificially different models,
- parameter counting,
- paired seed assignment,
- run-manifest validation.

---

## 20. Run Manifest and Reproducibility

For all new experiments, create a machine-readable manifest containing:

run ID, architecture, seed, data-split hash, git commit, Python version, Torch version, CPU/GPU details, thread count, full configuration, start time, end time, status, abort reason, checkpoint paths, result files.

Formats: `results/run_manifest.csv` and `results/run_manifest.json`.

For the runs used in the final version, publish at minimum: run metadata, aggregated metrics, analysis results, figure data, model hashes, reproducible commands. Very large checkpoints may be provided as release artifacts, but must be unambiguously referenced.

---

## 21. Reporting

Revise `RESULTS.md` only after the analyses are complete. Structure every claim according to:

1. observation
2. quantitative evidence
3. alternative explanation
4. causal test
5. limitation
6. permissible conclusion

Use graded formulations:

- "We observe …"
- "This is consistent with …"
- "The ablation supports the interpretation …"
- "The data are not sufficient to rule out …"
- "Under the conditions examined …"

Avoid, without sufficient evidence: "proves", "clearly shows", "Transformers fundamentally learn …", "MLPs are worse …", "the same circuit", "the same function".

The main claim may be used only if it is supported by the data and the evidence gate:

> "Transformer and MLP can solve the same modular task through related Fourier-based principles, but may express them in different neural representations."

If the results show that both mechanisms are essentially the same, exactly that must be reported. If neither architecture passes the Fourier tests, exactly that must be reported.

---

## 22. AI Disclosure

Update `AI_DISCLOSURE.md` completely and honestly. Separate:

- research options proposed by AI,
- the research question selected by humans,
- AI-generated code,
- human-written code,
- mathematical derivations checked by humans,
- experiments started and logged by humans,
- AI-generated raw drafts,
- the final interpretation written by humans.

Do not fill in human original work from conjecture. Use placeholders such as `[HUMAN AUTHORS MUST COMPLETE]` when something has not yet actually been carried out by humans.

---

## 23. Definition of Done

The extension counts as scientifically complete only once:

1. the mask methodology has been checked against the primary source,
2. old and new restricted/excluded variants are unambiguously separated,
3. all scientifically load-bearing functions have automated tests,
4. effective MLP weights rather than only `W_E` are analyzed,
5. hidden activations and output weights of both architectures are examined,
6. at least two causal ablations per architecture have been carried out,
7. the end-to-end logit formula fit is available for both architectures and all seeds,
8. the full-domain function comparison has been measured across all 12,769 input pairs,
9. the evidence gate before H3 has been explicitly evaluated and the decision-tree branch named,
10. the primary setting contains multiple paired seeds,
11. the most important effects are reported with confidence intervals,
12. parameter count and input parametrization have been checked as possible confounds,
13. grokking times are reported together with their measurement interval,
14. the delineation against Manir/Rupa, Swaroop, Doshi, and Khanh is documented and H1/H2 are explicitly framed as replication,
15. every figure is reproducible from stored raw data,
16. negative results are documented,
17. the claims in `README.md` match the actual evidence,
18. `AI_DISCLOSURE.md` correctly describes the actual work process,
19. the final scientific conclusions have been checked by the human authors and written in their own words.

---

## 24. Working Method and Order

Work in small, verifiable steps. After each phase:

- run the relevant tests,
- save a short audit,
- name the changed files,
- name the open risks,
- mark unconfirmed statements,
- do not commit invented results.

**Order:**

1. Secure the baseline
2. Audit
3. Preregistration (including evidence gate, decision tree, novelty delineation)
4. Correct the mask methodology
5. Synthetic tests
6. MLP mechanism analysis
7. Transformer mechanism analysis
8. End-to-end logit fit and full-domain function comparison
9. Pilot with one paired seed
10. Freeze the analysis code
11. Full seed runs
12. Causal ablations
13. Statistics
14. Evaluation of the decision tree; if applicable, bounded alternative-mechanism analysis
15. Human interpretation
16. Documentation
17. Update the explorer

Do not start directly with a large run matrix before the analyses and hypotheses have been frozen.

---

## 25. Output Format at the End of Each Run

At the end, output:

- **Concrete changes** (changed and newly created files),
- **Tests** (which run, which pass, which fail),
- **Files created**,
- **Supported statements** — exclusively statements backed by real results,
- **Hypotheses and planned analyses**,
- **Human decision points** — points that must not be decided by the AI,
- **Risks**: methodological, statistical, computational, documentary,
- **Next step**: exactly one concrete command or one clearly delimited work step.
