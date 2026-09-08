# RESULTS.md — GROKVERSE findings

> Every number on this page traces to a real seeded run under `training/runs/` and to a stored
> analysis artifact under `training/results/`. Nothing here is fabricated, hard-coded or
> cherry-picked. Negative and fragile results are reported as results (PROMPT.md §6).
>
> **Status of the writing.** AI-drafted from the stored artifacts on 2026-09-04, under master prompt
> §21. The **final scientific interpretation in the authors' own words has not yet been written** —
> see §14 and `docs/HUMAN_INTERPRETATION_TEMPLATE.md`. Wording follows §21's graded forms; the
> per-claim six-part structure lives in [`docs/CLAIM_EVIDENCE_TABLE.md`](docs/CLAIM_EVIDENCE_TABLE.md)
> and this page does not state anything more strongly than that table's "permissible conclusion" line.

> **This page replaces the version dated 2026-09-03.** That version carried inline `[AUDIT]` notes
> because its claims had been classified as over-interpreted by
> [`docs/CURRENT_EVIDENCE_AUDIT.md`](docs/CURRENT_EVIDENCE_AUDIT.md). Rather than patch it, it has
> been rewritten against the completed architecture study. Four of its headline statements do not
> survive and are listed as withdrawn in §11. The earlier arithmetic was not wrong; what was wrong was
> what the numbers were said to show.

---

## 0. How to read this page

Four rules govern everything below.

1. **The pre-registered evidence gate reports `neither_passes`.** Master prompt §12 therefore forbids
   reading a harmonic-family or "same Fourier principle" interpretation into these numbers for
   *either* architecture. The main claim the study was permitted to make **conditional on the gate** —
   "Transformer and MLP can solve the same modular task through related Fourier-based principles, but
   may express them in different neural representations" — **is not licensed by this data and is not
   made.** §21 requires that if neither architecture passes the Fourier tests, exactly that is
   reported. That is what §3 reports.
2. **Every threshold, definition and pass rule was fixed before the runs** and is recorded in
   [`docs/PREREGISTRATION.md`](docs/PREREGISTRATION.md), frozen at commit `0b55e1d`. Nothing was
   changed after a number was seen; the two places where a *choice* turned out to be load-bearing are
   escalated to the human authors as **D5** and **D6**, not resolved by the AI.
3. **Numbers are medians over 10 paired seeds** unless stated otherwise, with the observed range or a
   percentile-bootstrap 95 % interval. All ten seeds are shown per comparison in
   `training/results/aggregate/TABLES.md`; no seed is averaged away.
4. **`None` means "not evaluable", never "fails".** Where a criterion could not be tested, this page
   says so rather than scoring it.

---

## 1. What was run

**Task.** `(a + b) mod 113`, all 12,769 input pairs, a fixed 30 % train split. Paired seeds share a
split (identical `split_hash`), so every cross-architecture comparison is on the same data.

**Architectures.**

| | transformer | MLP (shared embedding) | MLP (parameter-matched) | MLP (two-hot) |
|---|---|---|---|---|
| description | 1 layer, ReLU, **no LayerNorm**, `d_model=128`, 4 heads, `d_head=32`, `d_mlp=512`; tokens `[a, b, =]`, read out at `=` | same embedding table `W_E[p, d]`, operand embeddings concatenated into a 2-layer MLP, `d_mlp=512` | as MLP, `d_mlp=572` | two-hot input, no shared embedding, `d_mlp=512` |
| parameters | **226,176** | **204,017** (−9.8 %) | **226,217** (+0.02 %) | 174,193 |

**Optimizer.** Full-batch AdamW, `lr = 1e-3`, `weight_decay = 1.0`, 25,000 steps, **no early stopping**
and **no acceleration** in the primary block. One CPU thread, pinned and recorded per run.

**The matrix — 51 runs, all completed, none failed.**

| block | runs | purpose |
|---|---|---|
| **primary** | 20 (10 transformer + 10 MLP seeds) | every claim below unless marked otherwise |
| confound | 18 | Grokfast × `train_frac` as a 2 × 2, 3 paired seeds per cell |
| parameter-matched | 10 | MLP at `d_mlp = 572` against the same 10 transformer seeds |
| two-hot | 3 | input parametrization as an alternative explanation to architecture |

`training/results/run_manifest.csv` lists all 51 with their config hashes and git commits.

---

## 2. Grokking is reproduced, in both architectures, on every seed

**Observation.** Both architectures memorize the training split within ~150 steps and sit at chance on
held-out data for thousands of steps before test accuracy rises sharply.

**Quantitative evidence** (crossings at train ≥ 0.99 / test ≥ 0.95, the pre-registered `primary`
definition; the evaluation grid is 10 steps for train and 25 for test):

| | memorization | generalization | grokking gap | final test accuracy |
|---|---|---|---|---|
| **transformer** | median **140** (140–150) | median **7,588** (5,625–10,275) | median 7,448 | median **0.99966** (0.9971–1.0000) |
| **MLP** | median **160** (160 in all 10) | median **9,250** (8,150–10,075) | median 9,090 | **1.000000 in all 10 seeds** |

Every crossing is reported with its evaluation interval; **the phrase "exact transition" is not used**
anywhere in this study. A crossing recorded at step *c* is known only to lie in `(c − eval_every, c]`.
Two sensitivity definitions (`sens_loose` 0.98/0.90, `sens_strict` 1.00/0.99) are stored alongside the
primary one in each `run.json`; they move the crossings by a few hundred steps without changing the
ordering.

**Permissible conclusion.** Grokking as described by Power et al. 2022 is reproduced un-accelerated in
both architectures, 20 of 20 runs, with the phase transition logged.

### 2.1 Timing: the transformer crosses earlier in 9 of 10 seeds

Paired difference (transformer − MLP), median **−1,562 steps**, bootstrap CI95 **[−2,895, −822]**,
Cliff's δ −0.80, Cohen's *d_z* −1.08. **Seed 4 reverses the direction** (10,275 vs 8,650). All ten
per-seed interval-consistent bounds exclude zero, so the ±25-step evaluation grid never flips a sign;
what limits the claim is the seed spread, not the grid.

> Under the conditions examined, the transformer's generalization crossing precedes the MLP's by a
> median of about 1,560 steps, in 9 of 10 seeds. This is **not** "transformers grok faster": it is one
> hyperparameter point, and the direction is not unanimous.

---

## 3. The pre-registered evidence gate: `neither_passes`

The gate (`docs/PREREGISTRATION.md` §5) asks four questions of each architecture and requires each to
hold on **≥ 8 of 10 seeds**. Evaluated at the final checkpoint
(`training/results/decision_tree_final.json`):

| criterion | what it requires | transformer | MLP |
|---|---|---|---|
| **G1** structure | structured fraction ≥ 0.25 of live neurons **and** median family fraction ≥ 0.50 | **10/10 ✓** | **10/10 ✓** |
| **G2** phase relation | resultant length `R` ≥ 0.5 **and** above the permutation null's 95th percentile | **10/10 ✓** | **10/10 ✓** |
| **G3** end-to-end formula | a sparse-sinusoid or odd-harmonic formula reaching argmax accuracy ≥ 0.90 on test cells, above the size-matched random-frequency control, with the `(a−b)` control below half | **10/10 ✓** | **10/10 ✓** |
| **G4** causality | `remove_structured` **and** `remove_key_freqs` each *necessary*, and `keep_structured` *sufficient* | **0/10 ✗** | **0/10 ✗** |

**Branch: `neither_passes`.** At the memorization crossing the branch is `undetermined`, because G3 is
not evaluable on any MLP seed there (the random-frequency control degenerates).

This is the study's central negative result, and §21 requires that it be reported exactly: **the
Fourier evidence tested does not, by the pre-registered standard, identify the learned mechanism in
either architecture.** Everything in §4–§6 is therefore a description of measured structure, not a
mechanism claim. At 100,000 steps the externally executed S1 block returns the same verdict — G1–G3 10/10,
G4 0/10, `neither_passes` — re-derived here with the frozen code (§16.3).

---

## 4. What the two networks compute

**Observation.** The two architectures land on very nearly the same input–output function, while the
internal quantities behind it do not resemble each other.

**Quantitative evidence.** Over all 12,769 input pairs at the final checkpoint, across the 10 paired
seeds: median agreement **0.99977**; median 3 disagreeing cells (range 0–26); the MLP is correct on
every cell in all 10 seeds; in **4 of 10 seeds the two agree on every input**. But at the level of the
logits themselves, over the same pairs: **Pearson correlation median 0.083** (0.002–0.205), margin
correlation 0.100, and **top-2 agreement 0.0056** — the runner-up class almost never matches.
(`training/results/function_agreement/`.)

**Alternative explanation.** The task is small enough that any sufficiently flexible model may land on
the same function; agreement over inputs need not indicate a shared mechanism.

**Causal test.** None applies — this is a behavioural measurement by construction.

**Limitation.** `error_jaccard` is 0.000 where defined and **undefined in 4 seeds** because neither
model errs (empty union), so error-overlap structure carries almost no information here.

**Permissible conclusion.** Under the conditions examined, the two architectures agree on 99.98 % of
all inputs and are identical on 4 of 10 seeds, while agreeing on the *shape* of the logits hardly at
all. The phrase **"both learn the same function" remains unsupported**: they agree closely; they are
not identical in 6 of 10 seeds.

---

## 5. The structure that is present

**Observation.** Periodic structure, the predicted phase-addition relation, and an end-to-end Fourier
fit are observed in both architectures on every seed.

**Quantitative evidence** (final checkpoint, medians over 10 seeds):

| | transformer | MLP |
|---|---|---|
| live neurons | 512 / 512 | 512 / 512 |
| structured neurons | 503 (494–512) = **0.982** of live | 453 (444–503) = **0.885** of live |
| median family fraction, curve `u_a` | 0.9923 | 0.9973 |
| phase resultant `R` | **0.9921**, null *q*₉₅ 0.098 | **0.9996**, null *q*₉₅ 0.081 |
| best Fourier formula, argmax accuracy on test cells | **1.0000** | **1.0000** |
| best Fourier formula, R² on test cells | **0.546** (0.427–0.731) | **0.974** (0.962–0.977) |
| size-matched random-frequency control, R² *q*₉₅ | 0.420 | 0.868 |
| `(a−b)` difference control, R² | 0.0001 | 0.0002 |

Structured-neuron fraction is reported under six definitions, not one, and the ordering is unchanged
under all of them (transformer 0.978–0.984, MLP 0.863–0.920 across the family-threshold, top-1 and
IPR-rank-matched variants). Swaroop's periodicity criterion selects 0.000 / 0.005 of neurons at its
strict setting and 0.998 / 0.888 at its loose one — a spread that is itself informative about how much
such definitions carry.

**Alternative explanation.** The structured-neuron definition is permissive: it selects 88–98 % of all
neurons. "Most neurons are structured" is compatible with the threshold being loose rather than the
network being organised. This alternative is **not** ruled out; §6 is where it bites.

**Limitation.** The transformer's per-neuron "effective curves" are a **mean-attention approximation**
(`additivity_r2` median 0.922, range 0.871–0.961), while the MLP's are exact. The two are therefore
not on identical footing. The direct path contributes a 0.002 share of the logits — the computation
runs through the MLP block, not around it.

**Permissible conclusion.** Periodic structure, the phase relation and an end-to-end Fourier fit are
observed in both architectures at the final checkpoint, on all 10 seeds, under the pre-registered
thresholds. This is **not** "the models implement a Fourier algorithm" — that requires G4.

### 5.1 A caveat that bounds this whole section: the MLP has not settled at the budget

Every run stops at a fixed 25,000 steps, and whether a metric has actually converged there is tested
per metric and per seed (relative change between the step-20,000 and step-25,000 checkpoints, "settled"
below 5 %):

| metric | transformer | MLP |
|---|---|---|
| **`structured_fraction_of_live`** | **10/10 settled** (median 0.41 %) | **2/10 settled** (median 7.09 %, max 9.00 %) |
| `embedding_top8_concentration` | 10/10 (0.59 %) | 8/10 (2.80 %) |
| `phase_relation_R` | 10/10 (0.37 %) | 10/10 (0.03 %) |
| `median_family_fraction` | 10/10 (0.15 %) | 10/10 (0.09 %) |
| `logit_key_subspace_share` | 9/10 (1.25 %) | 10/10 (0.40 %) |
| `fraction_best_aic_sinusoid` | **0/10** (17.1 %) | 2/10 (15.3 %) |

`structured_fraction_of_live` is the metric behind G1 and behind the headline architecture gap of
+0.085, and for the MLP it is **still rising at the budget in 8 of 10 seeds** — in seed 0 from 0.223 at
step 8,000 to 0.861 at 20,000 to 0.881 at 25,000, while the transformer's has flattened (0.959 to
0.965). A longer budget would therefore be expected to **shrink** the gap this study reports. **Measured at
100,000 steps on external compute (§16.2): the median gap closes to 0.0000 and the MLP's fraction is settled
in 10 of 10 seeds.**

The waveform shares are unsettled by more still — the transformer's sinusoid share moves **0.255 in
absolute terms** between steps 20,000 and 25,000. This does not touch the §8.1 trajectory result, which
is measured from initialization to the last checkpoint *before* generalization and so ends long before
the budget; it does mean the *final* waveform split in §5 must not be read as an endpoint.

This is the censoring Khanh 2026 (arXiv:2607.06639) warns about, measured here rather than assumed. It
does not invalidate the comparison, but it fixes its meaning: **every number on this page is a
comparison at a fixed 25,000-step budget, not a comparison of converged states.** See
`docs/LIMITATIONS.md` §B3.

### 5.2 Restricted and excluded loss (Nanda et al. protocol)

Medians over 10 seeds at step 25,000, **test split**. Unrestricted test loss for reference:
transformer 0.00182, MLP 0.00001.

| protocol | arch | components kept | restricted | excluded |
|---|---|---|---|---|
| `nanda_exact` | transformer | 10 | **0.0000** | **19.16** |
| `nanda_exact` | MLP | 25 | **0.0000** | **5.00** |
| `paper_literal_2x2_block` | transformer | 19 | 0.0000 | 19.13 |
| `paper_literal_2x2_block` | MLP | 49 | 0.0000 | 4.86 |
| `legacy_broad_mask` | transformer | 101 | 0.0001 | 18.92 |
| `legacy_broad_mask` | MLP | 625 | 0.0000 | 4.37 |

Restricted loss near zero under every protocol says the key frequencies alone suffice to reproduce the
correct logits — **for the MLP as much as for the transformer**. Excluded loss far above the
unrestricted loss says removing them destroys the function in both. Both directions of the Nanda
progress measure therefore hold for the MLP too.

**The excluded losses must not be compared across architectures**: the protocols keep different
numbers of components (10 vs 25 under `nanda_exact`) and the logit scales differ, so a larger excluded
loss means "further from the correct logits", not "more Fourier". What is comparable is the
qualitative pattern, and that pattern is the same in both.

The four protocols are separated and audited in
[`docs/MASK_PROTOCOL_AUDIT.md`](docs/MASK_PROTOCOL_AUDIT.md). The earlier version of this page used
only `legacy_broad_mask` — which keeps 101 components where the released Nanda operator keeps 10 — and
reported it as a reproduction. It was not one; see §11.

---

## 6. Causality — where the Fourier reading fails

This is the section that decides the gate, so it is given in full.

**Observation.** Ablating the *structured-neuron set* is not distinguishable from ablating an equally
large random set of neurons. Ablating the *key frequencies* is sharply distinguishable from ablating
an equal number of random frequencies.

**Quantitative evidence** (final checkpoint, medians over 10 seeds; "control" is a size-matched random
set drawn 50 times; *necessary* requires drop ≥ 0.5 **and** control mean drop < 0.1 **and** z ≥ 3):

| ablation | arch | test acc after | drop | control drop | z | necessary | sufficient |
|---|---|---|---|---|---|---|---|
| `remove_structured` | MLP | 0.009 | 0.991 | **0.878** | 3.6 | **0/10** | — |
| `remove_structured_neurons` | transformer | 0.057 | 0.942 | **0.910** | 1.9 | **0/10** | — |
| `keep_structured` | MLP | 1.000 | 0.000 | 0.000 | −0.1 | — | **10/10** |
| `keep_structured_neurons` | transformer | 0.9997 | 0.000 | 0.000 | −0.4 | — | **10/10** |
| `remove_key_freqs_from_curves` | MLP | 0.011 | **0.989** | **0.000** | — | **10/10** | — |
| `remove_key_freqs_from_embedding` | transformer | 0.015 | **0.984** | **0.0005** | 887 | **10/10** | — |
| `remove_key_subspace_from_residual` | transformer | 0.763 | 0.235 | 0.000 | 4172 | **0/10** | 4/10 |
| `keep_key_freqs_in_curves` | MLP | 1.000 | 0.000 | 0.991 | −399 | — | **10/10** |
| `keep_key_freqs_in_embedding` | transformer | 0.995 | 0.004 | 0.990 | −485 | — | **10/10** |
| `restricted_circuit_only` | transformer | **1.0000** | −0.000 | 0.991 | −311 | — | **10/10** |
| `replace_with_sinusoid_fit` | MLP | **1.0000** | 0.000 | 0.991 | −206 | — | **10/10** |
| `replace_with_square_fit` | MLP | **1.0000** | 0.000 | 0.991 | −185 | — | **10/10** |
| `ablate_head_k` (4 heads × 2 modes) | transformer | 0.51–0.56 | 0.44–0.49 | 0.034–0.041 | 19–27 | 3–4/10 | 0/10 |

> **Two control numbers in this table changed on 2026-09-06.** A determinism defect in the frozen code (an unordered set decided the order in which the shared random generator was consumed) made the size-matched control draws irreproducible; the fix moved the MLP control drop from 0.873 to 0.878 and the transformer's from 0.916 to 0.910. No observed value, no necessity or sufficiency label and no gate verdict changed. `PREREGISTRATION.md` §12 logs it as a post-freeze bug fix and `HUMAN_DECISIONS.md` **D8** puts it to the human authors.

Three things follow.

1. **G4 fails in both architectures for the same reason, and the reason is the structured-neuron
   threshold rather than the models.** Removing 88–98 % of any network destroys it, so the
   size-matched control cannot discriminate: the control alone already does 0.878 (MLP) and 0.910
   (transformer) of the damage. `CAUSAL_ABLATION_PLAN.md` §6 named this outcome in advance — "removing
   `C` and the control do comparable damage → `C` is **not** specifically load-bearing". Changing the
   threshold now, because the outcome is inconvenient, is exactly what pre-registration forbids.
   Recorded for the human authors as **D6**.
2. **The key frequencies, by contrast, are load-bearing in both.** Removing them costs ~0.99 accuracy
   where an equal number of random frequencies costs ~0.000; keeping only them costs ~0.000 where
   keeping an equal number of random ones costs ~0.99. For the MLP, replacing every neuron's curves
   with its *fitted sinusoid* or its *fitted square wave* leaves test accuracy at **1.0000** — the
   fitted waveform is sufficient to carry the function.
3. **Which transformer ablation G4's second condition refers to was never specified, and it decides
   that condition.** `remove_key_subspace_from_residual` gives 0/10;
   `remove_key_freqs_from_embedding` gives 10/10. The AI wired the former before any number was seen
   and **did not change it afterwards**; the choice is escalated as **D5**. It does not change the gate
   outcome — G4 still fails through `remove_structured` in both architectures — but it changes what
   may be said about the transformer's key frequencies.

**Alternative explanation.** For the head ablations, the median drop of 0.44–0.49 sits just below the
0.5 necessity threshold, so "3–4 of 10 necessary" reflects a threshold boundary rather than a clean
negative: each head is separated from its control by z ≈ 19–27, and no single head is sufficient.
Fixing attention to its mean costs 0.336 accuracy, so the attention pattern is **not** merely a
constant 50/50 sum — the input-dependent part carries something.

**Limitation.** Cross-architecture comparison of the individual ablation magnitudes is **not permitted**
(master prompt §17): the ablations modify different objects. For transformer seed 5 the structured set
is all 512 neurons, so no size-matched control set exists and the value is `null`, not a failure.

**Permissible conclusion.** Under the pre-registered structured-neuron definition, the ablation cannot
distinguish the structured set from a size-matched random set, so **causal necessity is not established
for either architecture**. In both architectures, removing the key-frequency components damages the
model far more than removing an equal number of random frequencies; the ablation supports the
interpretation that the key frequencies are load-bearing, while the *neuron set* the study used to name
them is not.

---

### 6.1 The graded ablation: there *is* a small load-bearing set — at the transition

§6 point 1 concedes that G4 cannot discriminate because the structured set is 86–100 % of the live
neurons. An external reviewer raised the same objection independently on 2026-09-05: removing 88–98 %
of a network tells you nothing about *which* neurons mattered. That objection is testable, and
`docs/PREREGISTRATION.md` §14 pre-registers the test — written before any graded number was inspected,
human approval open as **D7**.

**Design.** Rank the live neurons by B1's own yardstick, `s = min(u_a family fraction, u_b family
fraction)`, with B1's categorical failures demoted so the top-*n* sets are **nested subsets of the B1
structured set**. Ablate the top 1 / 2 / 5 / 10 / 25 / 50 % against **50 seeded size-matched random
groups of live neurons**, in both directions, over the 10 primary seeds and both checkpoints.
Discriminable at *f* iff, in ≥ 8 of 10 seeds, the observed drop exceeds **every** control and z ≥ 3.

**Observation.** The answer depends on *when* you look, and the two architectures part company.

| arch | checkpoint | smallest discriminating fraction | at that fraction: drop vs control | seeds |
|---|---|---|---|---|
| MLP | crossing | **1 %** (5 neurons) | **0.187** vs 0.010 | 10/10 |
| transformer | crossing | **1 %** (5 neurons) | **0.031** vs 0.004 | 9/10 |
| MLP | final | **50 %** (256 neurons) | 0.412 vs 0.052 | 9/10 |
| transformer | final | **5 %** (26 neurons) | **0.096** vs 0.0002 | 9/10 |

At the **crossing** checkpoint, where test accuracy has just reached ≈ 0.95, removing the five
most-structured neurons already does an order of magnitude more damage than any of fifty random
five-neuron groups — in **both** architectures. The B1 set is also much smaller there (median 142 of
512 neurons in the MLP, 296 in the transformer) than at the end of training (453 and 503).

At the **final** checkpoint the MLP shows nothing: removing the top 1, 2, 5 or 10 % changes test
accuracy by **exactly 0.0000**, and the ranking does not separate from its control until half the
network is gone. The transformer keeps a small load-bearing core, discriminating from 5 % with a drop
of 0.096 against a control mean of 0.0002 — a ratio of roughly 400.

**Alternative explanation, tested.** That this is an artifact of the particular score is checked two
ways. The pre-registered sensitivity score (the mean of the three family fractions, no demotion)
reproduces the pattern. And the **IPR-ranked sweep that already existed** (D2, Doshi arXiv:2310.13061)
— read at matching fractions with no new computation, a different ranking and a different control —
agrees qualitatively: discriminating from 5 % at crossing in both architectures, and at the final
checkpoint from 25 % in the MLP against 5 % in the transformer. Two independent rankings put the MLP's
final state on the redundant side and the transformer's on the concentrated side.

**Limitation.** Statistical separation is not importance. At the MLP's final checkpoint the 25 %
group is separated from its control in 7 of 10 seeds while costing **0.002** of accuracy; the
pre-registered rule tests only whether the observed damage beats every control, not whether it
matters. Absolute drops belong next to every flag, and are given in
`training/results/GRADED_ABLATION.md`.

**Permissible conclusion.** G4's failure was, as D6 suspected, a property of the *set size* and not
evidence that the structured neurons are causally inert: a nested subset of the very same B1 set is
sharply load-bearing at the transition in both architectures. It remains so at convergence only in the
transformer. **This does not reopen the gate.** G4's verdict is fixed by its own pre-registered rule
and stands at 0/10 in both architectures; §14.5 fixed that in advance, before these numbers were seen.
Nor does it show that the B1 *set* is the mechanism — only that the *ranking* carries causal signal at
small group size.

## 7. H3 — the study's proposed primary contribution — is refuted by its own criterion

**H3 as pre-registered.** The MLP's lower structured-neuron fraction is an artifact of top-*k*
concentration: because the MLP's neurons are more square-wave-like, their power spreads over aliased
odd harmonics that a top-*k* metric cannot see. Under a harmonic-family-aware definition the
architecture gap should close.

**Quantitative evidence** (`training/results/h3_report.json`, 10 paired seeds):

| | median | bootstrap CI95 |
|---|---|---|
| architecture gap under the **top-1** definition | **+0.0898** | — |
| architecture gap under the **family** definition | **+0.0850** | [+0.0600, +0.1004] |
| how much the family definition closes | **+0.0049** | [+0.0012, +0.0076] |

The family definition closes **5.4 %** of the gap. Refutation criterion 2 — *is the MLP still lower
under the family definition, with an interval excluding zero?* — is met: yes, unanimously across all
10 seeds (sign test *p* = 0.002, *d_z* = 2.36). **H3 as pre-registered is refuted.**

**The hypothesised artifact is real; it is simply far too small.** Two synthetic populations built from
each checkpoint's own `(k, phase, amplitude)` and differing **only** in waveform separate by **+0.1893**
at top-1, **+0.0501** at top-4 and **+0.0248** at top-8 — identical for both architectures, as it must
be, since it is a property of the metric and not of the models. So top-*k* concentration does score a
square-wave population ≈ 0.19 lower at top-1. That artifact accounts for 5.4 % of the observed gap.

**Alternative explanation.** This particular family definition (own dominant frequency plus aliased odd
harmonics ≤ 7) may be the wrong harmonic-aware measure, or the effect may be real but smaller than
n = 10 can resolve.

**Limitation.** At *p* = 113 the odd-harmonic families of different fundamentals alias onto one another
— those of `k = 6, 19, 51` all contain 18 — which bounds what any family-based measure can identify. An
audit stored with the report confirms that the criterion "the family beats the matched top-*m*" is
**unsatisfiable by construction** (maximum observed excess 1.1e-16 across all runs); it was never used
as a pass rule.

**Permissible conclusion.** By the criterion fixed before the experiments, the MLP's lower
structured-neuron fraction is a **real difference in the measured structure, not an artifact of top-*k*
concentration**.

---

## 8. H4 — structure appears before generalization, in every seed

**Observation.** Structure metrics rise during the memorization plateau, well before the test-accuracy
jump.

**Quantitative evidence** (onset = first step at which the metric exceeds its own baseline; medians
over 10 seeds):

| metric | transformer onset | MLP onset | onset before generalization |
|---|---|---|---|
| logit key-subspace share | **500** (500–1,000) | **500** (500 in all 10) | 10/10 both |
| structured fraction of live | 1,000 (500–5,000) | 1,000 (1,000 in all 10) | 10/10 both |
| embedding top-8 concentration | 1,000 (1,000–2,000) | 1,000 (500–2,000) | 10/10 both |
| median family fraction | 2,000 (1,000–4,000) | 500 (500–1,000) | 10/10 both |
| median odd-minus-even, `u_a` | 4,500 | 3,500 | 2/2 (txf), 8/8 (MLP) — **undefined on the rest** |
| fraction best fit by a **sinusoid** | 2,000 | 750 | 10/10 both |
| fraction best fit by **odd harmonics** | 3,000 | 1,000 | 10/10 both |
| phase relation `R` | **undefined on all 10** | **undefined on all 10** | — |
| fraction best fit by a **square wave** | **undefined on all 10** | **undefined on all 10** | see §8.1 — it *falls* |

For comparison, generalization is at median 7,588 (transformer) and 9,250 (MLP); memorization at 140
and 160.

### 8.1 The waveform composition shifts during the plateau — and the two architectures diverge there

The onset rule of INTERFACES §12 detects only an *upward* crossing of `init + 3 σ`. A metric that
**decreases** can therefore never register an onset, and "undefined" conflates *never moved* with
*moved downward*. For the square-wave share the truth is the second, and the pre-registered
`change` statistic — init to the last checkpoint before generalization — records it:

| share of neuron curves best fit by | transformer: init → pre-gen | MLP: init → pre-gen |
|---|---|---|
| a **square wave** | 0.787 → **0.116** (**−0.656**) | 0.789 → **0.421** (**−0.366**) |
| a **sinusoid** | 0.149 → **0.646** (+0.495) | 0.148 → **0.333** (+0.186) |
| **odd harmonics** | 0.021 → 0.173 (+0.149) | 0.021 → 0.218 (+0.194) |
| odd harmonics with 1/j decay | 0.042 → 0.019 (−0.023) | 0.035 → 0.024 (−0.008) |

Two things are visible. First, the two architectures **start indistinguishable** (0.787 vs 0.789
square; 0.149 vs 0.148 sinusoid) and separate only during the plateau. Second, they separate in
*different directions*: the transformer moves decisively to sinusoid (0.646) and retains almost no
square-like curves (0.116), while the MLP splits between sinusoid (0.333) and odd harmonics (0.218)
and keeps nearly four times as much square (0.421). The waveform difference reported at the final
checkpoint in §5 is therefore already present **before either architecture generalizes**.

**The init level is not a finding.** At initialization the curves are noise, and the square model has
the same parameter count as the sinusoid, so which one wins by AIC on noise is a property of the model
set. That the two architectures agree to within 0.002 at init is the evidence for reading it that way.
What is a finding is the *change*, which is unanimous in sign across all 10 seeds in both architectures
for the square and sinusoid shares.

These two trajectories were **flat zero for the entire study** until a defect was fixed on 2026-09-04
(`docs/LABBOOK.md` 101): the metric compared an integer model index to a model name and silently
returned 0.0. Nothing else consumed them, so no other number on this page moves.

**This is correlational, and is labelled as such in the artifact itself.** Structure preceding
generalization is consistent with H4; it cannot show that the structure *caused* the jump, nor that it
had to form. Causal statements come from §6 alone.

**Limitation.** The onset indicator's baseline standard deviation is taken over two checkpoints, so the
step is sensitive to the checkpoint grid: it is an **indicator, not a measured transition**. It is also
**rise-only** (§8.1), so it is silent about any metric that falls. `phase_relation_R` produces no onset
in either architecture for a different reason: it is `null` at the first checkpoints, where the
structured set is too small for the statistic to exist, so no baseline can be formed. And
`median_odd_minus_even_u_a` yields an onset on only 2 of 10 transformer seeds. All three are reported
here rather than dropped.

**Permissible conclusion.** Under the conditions examined, six structure metrics reach their onset
before the generalization crossing in 10 of 10 seeds in both architectures, at between 5 % and 40 % of the
way to that crossing (earliest 500 of 9,250 in the MLP; latest 3,000 of 7,588 in the transformer), and the waveform composition shifts substantially over the same interval. This is
consistent with H4. It remains correlational.

---

## 9. Bounded alternative-mechanism analysis

`neither_passes` triggers this analysis for both architectures (`PREREGISTRATION.md` §6.3). It asks
what *else* the representations could be, without proposing a new mechanism.

| | transformer | MLP |
|---|---|---|
| linear probe for the target from the hidden layer | **1.0000** | **0.9999** |
| same probe on a size-matched random control | 0.0079 | 0.0091 |
| chance level | 0.0088 | 0.0088 |
| effective rank of the hidden representation | **12.70** (8.6–18.5) | **66.64** (59.1–79.0) |
| directions carrying 90 % of the variance | 11.5 | 56.5 |
| top-1 singular share | 0.239 | 0.034 |
| effective rank of the logits | 5.04 | 17.37 |

**Singular-direction ablation** (removing the top-*r* singular directions of the hidden representation,
against a size-matched random-direction control):

| removed | transformer drop | exceeds all controls | MLP drop | exceeds all controls |
|---|---|---|---|---|
| top-1 | 0.0000 | 3/10 | 0.0000 | 0/10 |
| top-4 | 0.0002 | 5/10 | 0.0000 | 0/10 |
| top-8 | **0.4616** | **10/10** | 0.0000 | 0/10 |
| top-16 | **0.9644** | **10/10** | **0.0000** | **0/10** |

We observe that the transformer's function is destroyed by removing its 16 leading directions, while
the MLP's is completely unaffected by the same operation. This is consistent with the two architectures
spreading a comparable computation over very different numbers of directions.

**Cross-seed linear CKA — and this is the caveat that governs the whole section.** Between two seeds of
the **same** architecture, CKA is median **0.0017** (transformer) and **0.0973** (MLP). Two runs that
differ only in seed, that agree on 99.98 % of all inputs, and that are by construction the same
mechanism, score as *dissimilar*. **A low cross-architecture representation similarity therefore cannot
be read as evidence of different mechanisms** — the measure does not identify sameness even where
sameness is guaranteed.

---

## 10. Controls — what the differences are *not* explained by

Master prompt §14: no change is attributed to one factor while two are unseparated. All three control
blocks are analysed separately from the primary comparison and are never merged into it.

**Grokfast × `train_frac`, as a 2 × 2** (3 paired seeds per cell; with 3 seeds the intervals are wide,
and no bootstrap interval is reported). Effect on the generalization step:

| | main effect of `train_frac` (0.3 → 0.5) | main effect of Grokfast | interaction |
|---|---|---|---|
| transformer | **−6,367 steps** | −350 steps | +883 |
| MLP | **−8,217 steps** | +892 steps | −933 |

The training fraction dominates by an order of magnitude; the Grokfast main effect is the same size as
the interaction and, for the transformer, indistinguishable from zero at this resolution. The earlier
version of this page reported a cross-architecture speed ratio moving from ~2.9× to ~1.3× between two
settings that differed in **both** knobs, and read the change partly as a statement about Grokfast.
This decomposition is what §14 exists to require, and it does not support that reading.

**Parameter-matched** (MLP at `d_mlp = 572`: 226,217 vs 226,176 parameters, a 0.02 % difference,
against the same 10 transformer seeds):

| quantity | median difference (transformer − MLP) | CI95 |
|---|---|---|
| generalization step | −2,212 steps | [−2,945, −1,133] |
| structured fraction of live | **+0.0969** | [+0.0863, +0.1240] |
| phase `R` | −0.0076 | [−0.0095, −0.0054] |
| square/odd-harmonic fraction | −0.2230 | [−0.4348, −0.1082] |

**Every structure difference survives equalizing the parameter count**, with intervals excluding zero
and the same signs as in the primary block. This control does **not** equalize the *shape* of the
budget, so capacity-allocation effects remain.

**Two-hot input** (3 paired seeds, two-hot MLP against the shared-embedding MLP). This one changes what
may be said:

| quantity | median difference (two-hot − shared-embedding MLP) | CI95 |
|---|---|---|
| square/odd-harmonic fraction | **+0.1836** | [+0.1133, +0.2109] |
| structured fraction of live | −0.1953 | [−0.1992, −0.1699] |
| generalization step | +3,325 steps | [+3,225, +3,650] |
| phase `R` | −0.0002 | [−0.00020, +0.00009] — **spans zero** |

The two-hot MLP is **more** square-wave-like than the shared-embedding MLP, by roughly half the size of
the transformer-vs-MLP waveform difference itself. The waveform result of §5 is therefore **at least
partly a property of the input parametrization rather than of the architecture**. With 3 seeds this
block can only detect a large effect, and a null here would be weak evidence — but this is not a null.

---

## 11. Negative results, and claims withdrawn

Master prompt §23 item 16 requires that these be documented. They are the substance of this study, not
its residue.

**Negative results.**

1. **The evidence gate fails in both architectures** (`neither_passes`, §3). The mechanism reading the
   study was designed to test is not licensed.
2. **H3, the proposed primary contribution, is refuted by its own pre-registered criterion** (§7). The
   artifact it postulated is real and quantified at +0.19 on synthetic populations, and explains 5.4 %
   of the gap it was meant to explain.
3. **Causal necessity of the structured-neuron set is not established in either architecture** (§6),
   because the size-matched control does 87–92 % of the same damage.
4. **Two of seven H4 structure metrics never produce a defined onset**, and one produces it on 2 of 10
   transformer seeds (§8).
5. **G3's random-frequency control degenerates on some runs** (it can draw all 56 available frequencies
   at *p* = 113), so that condition is not always testable as specified; at the memorization crossing
   it is untestable on all 10 MLP seeds, which is why that gate point is `undetermined` rather than a
   verdict.
6. **Cross-seed CKA does not identify same-architecture seeds as similar** (§9), which removes
   representation similarity from the set of measures this study can use to compare architectures.
7. **The paired family-minus-top-1 gap does not differ between architectures**: median −0.0025, CI
   [−0.0060, **+0.0035**] — the one comparison of the eight pre-specified whose interval spans zero.
8. **The MLP's structured fraction has not converged at the step budget** in 8 of 10 seeds (§5.1), so
   the headline structure gap is a budget-fixed comparison and its bias has a known direction.

**Claims withdrawn from the 2026-09-03 version of this page.**

| withdrawn claim | why |
|---|---|
| "the eight key frequencies alone solve the task while removing them destroys it", *as a reproduction of Nanda et al.* | it used `legacy_broad_mask`, which keeps 101 components where the released operator keeps 10. The statement is now made under named protocols in §5.2, where it holds for **both** architectures. |
| "attention ~50/50 directly links the learned structure to the computation" | it was a mean without a variance. Fixing attention to its mean costs 0.336 accuracy (§6), so the input-dependent part is not negligible. |
| "the transformer learns a markedly sparser Fourier circuit" (as a discovery) | the direction is a **replication** of Manir & Rupa 2026, and §6 gives no causal warrant for the word "circuit". The word *markedly* also does not survive the change of measurement point: the retired 0.73-vs-0.44 gap came from early-stopped runs read at the crossing, and at the 25,000-step budget the same quantity is **0.961 vs 0.891** — same direction, gap 0.07 rather than 0.29. The top-8 cap additionally binds on 6 of 10 MLP runs and 0 of 10 transformer runs. |
| "transformer groks ~3× faster" | the ratio moved between settings that differed in two knobs at once; §10 separates them. |

**Claims that may not be made from this evidence at all:** "the two architectures use the same
circuit"; "the same Fourier principle in different representations" (conditional on a gate that did not
pass); "both learn the same function" (99.98 % agreement, but not identical in 6 of 10 seeds);
"transformers grok faster"; "the MLP's structure is a measurement artifact" (refuted, §7).

---

## 12. Where this sits in the literature

Full treatment in [`docs/NOVELTY_AND_RELATED_WORK.md`](docs/NOVELTY_AND_RELATED_WORK.md) and
[`docs/METHODS.md`](docs/METHODS.md), which maps every method to its source and to our implementation.
In short:

- **Power et al. 2022** (arXiv:2201.02177) — the phenomenon. §2 reproduces it un-accelerated.
- **Nanda et al. 2023** (arXiv:2301.05217) — the recipe, the Fourier analysis and restricted/excluded
  loss. §5.2 follows the released operator under a named protocol.
- **Liu et al. 2022, Omnigrok** (arXiv:2210.01117) — grokking controlled by weight norm.
- **Lee et al. 2024, Grokfast** (arXiv:2405.20233) — used only in the confound block, never in the
  primary block.
- **Manir & Rupa 2026** (arXiv:2603.25009) — reports the transformer-vs-MLP concentration difference.
  **H1 and H2 of this study are explicitly framed as replication of that direction, not discovery.**
- **Swaroop**, **Doshi** (IPR-ranked pruning) and **Khanh 2026** (arXiv:2607.06639, on metrics read at
  the transition rather than after convergence) — their definitions are implemented alongside ours as
  alternative structured-neuron criteria (§5), so the results can be read under each. Khanh's concern
  is answered here by construction: every primary-block number is read at step 25,000, after
  convergence, with no early stopping.

---

## 13. Self-critical evaluation

**What is solid.** 51 runs, all completed; 10 paired seeds sharing a split hash; every threshold fixed
before the runs and frozen at a named commit; 24 test files and 1,913 automated checks covering every
scientifically load-bearing function; every figure drawn from the stored aggregate tables only; a gate
that was allowed to fail; a primary hypothesis that was allowed to be refuted by its own criterion; and
three separate control blocks, one of which (§10, two-hot) *changed* what may be claimed.

**What is fragile.**

- **n = 10, one hyperparameter point.** Every "under the conditions examined" is load-bearing.
- **The MLP's structured fraction has not converged at the 25,000-step budget** (§5.1), and it is
  still rising. The direction of that bias is known: a longer budget would shrink the reported gap.
- **The structured-neuron threshold is doing more work than any measurement.** It selects 88–98 % of
  neurons, which is what makes G4 untestable (§6). A different threshold would produce a different gate
  outcome, and that is a decision reserved for the human authors (**D6**), not a result.
- **One un-named identifier decides a gate condition** (**D5**, §6).
- **The transformer's structure metrics are approximations.** `additivity_r2` is 0.922, not 1.
- **Three control blocks have 3 seeds, not 10.** The 2 × 2 decomposition carries no interval at all.
- **Several metrics were quietly empty until they were checked by hand.** Four aggregation defects were
  found this way, three of them producing well-formed tables of `null` rather than errors (labbook
  entries 58 and 99). A regression test was added for each; the class of defect is documented because
  it is likelier than not that others of the same kind remain.

**What we would do next.** Sweep the structured-neuron threshold as a pre-registered sensitivity axis
rather than a fixed choice; add seeds to the control blocks; measure the two-hot block at 10 seeds,
since it is currently the finding most likely to change a conclusion; test `p` other than 113, where
odd-harmonic aliasing does not confound family definitions; and run the multiplication task
(`txf_mul_*`), which is reserved for the human authors (**E6**).

---

## 14. What only the human authors can decide

This page is an AI-drafted report of measurements. The following are **not** filled in, and must not be
filled in from conjecture (master prompt §22):

- **The final scientific interpretation, in the authors' own words** (§23 item 19). Template:
  [`docs/HUMAN_INTERPRETATION_TEMPLATE.md`](docs/HUMAN_INTERPRETATION_TEMPLATE.md).
- **Decisions A–E, the status line, section F and the sign-off** in
  [`docs/HUMAN_DECISIONS.md`](docs/HUMAN_DECISIONS.md).
- **D5** — which transformer ablation G4's `remove_key_freqs` refers to (§6).
- **D6** — whether the structured-neuron threshold that makes G4 untestable stands as pre-registered
  (§6). Changing it requires a full re-run and must not be motivated by the outcome.
- **The 11 `[HUMAN AUTHORS MUST COMPLETE]` placeholders** in [`AI_DISCLOSURE.md`](AI_DISCLOSURE.md).
- **The `txf_mul_*` runs** (**E6**) and the explorer update (**E3**), both reserved.
- **D9** — whether the externally executed 100,000-step block and the reviewer's follow-up analyses (§16)
  enter the study, under the reviewer's own conditions (local reproduction of a seed pair first; the
  competition's rules on external compute).

---

## 15. Reproducing every number on this page

```bash
# from training/  (Python 3.12, CPU-only)
python tests/run_all.py                    # 24 test files, 1,913 checks

# the matrix (51 runs; hours, not minutes)
pwsh ./run_matrix_chain.ps1

# analysis over the completed runs
python -m grokverse.analysis.driver --runs "*_arch25k"
python -m grokverse.analysis.aggregate --runs "txf_add_p113_wd1.0_frac0.3_seed?_arch25k" \
                                               "mlp_add_p113_wd1.0_frac0.3_seed?_arch25k" \
                                        --out results/aggregate
python -m grokverse.analysis.decision_tree     --aggregate-dir results/aggregate --point final
python -m grokverse.analysis.statistics_report --aggregate-dir results/aggregate
python -m grokverse.analysis.h3_report         --aggregate-dir results/aggregate
python -m grokverse.analysis.h4_report         --aggregate-dir results/aggregate
python -m grokverse.analysis.controls_report   --aggregate-dir results/aggregate_all
python -m grokverse.analysis.figures_study     --aggregate-dir results/aggregate
```

Every figure in `training/results/figures/` is drawn from the aggregate tables alone and carries a
caption naming its source file, so no figure can drift from the numbers above. This was checked rather
than asserted: re-running `figures_study` into a scratch directory reproduces **all 11 figures
byte-identically**, and each figure's index entry names source files that all exist (master prompt §23
item 15).

**Freeze audit.** Every analysis module stamps the git commit it ran at into its own output envelope.
Reading those back: **all 696 analysis artifacts across all 51 runs record a commit that is a
descendant of the freeze `0b55e1d`**, across 17 distinct analysis commits, with none missing the
field. No pilot-era output survives anywhere in the study. This is checkable in a second by anyone
with the repository, and is the reason the freeze is a property of the artifacts rather than a claim
about what we remember doing.

**The explorer (`web/`) still shows the pre-study runs** and is deliberately untouched; the required
changes are specified in [`docs/dev/EXPLORER_UPDATE_PLAN.md`](docs/dev/EXPLORER_UPDATE_PLAN.md) and are
blocked on human review (**E3**).

## 16. Externally executed follow-up blocks at 100,000 steps — verified, reported, not yet reproduced here

> **Status.** The runs in this section were executed by the study's external reviewer on the reviewer's own
> AWS account — the 100,000-step convergence block **S1** on 2026-09-05 and the **two-hot control at 10 seeds**
> on 2026-09-07 — and handed back on 2026-09-08 as an unmodified bundle (provenance, hashes and what was
> checked: [`docs/external/README.md`](docs/external/README.md)). Nothing above this section was changed by
> it. Each number below is marked **(v)** if it was re-derived in this repository from the bundle's artifacts
> with this branch's frozen code, or **(r)** if it comes from the reviewer's own scripts, which are not in the
> bundle. Under the reviewer's own runbook (§0.3) no S1 number enters a submission before at least one seed
> pair has been reproduced on the authors' hardware; that has not been run. Whether the block enters the
> study at all is **D9** (`docs/HUMAN_DECISIONS.md`).

### 16.1 What ran (v)

| item | S1 convergence block | two-hot control ×10 |
|---|---|---|
| runs | 20: seeds 0–9 × {transformer, MLP}, `--study conv100k` | 10: `mlp_twohot`, seeds 0–9, `arch25k` |
| steps | **100,000** (4× the primary block) | 25,000 (frozen) |
| every other flag | the frozen `arch25k` configuration (§1) | byte-identical to `matrix_twohot_20260904T022540Z.json` |
| code | `42dd79a` on branch `arch-study-convergence` = this branch's `fda066e` + ten `CHECKPOINT_GRID` entries above 25,000 + the runbook; no other change | same |
| environment | AWS EC2 `c7i.16xlarge`, Linux x86_64, Python 3.12.14, torch 2.12.1+cpu, numpy 2.4.6; 20 single-threaded runs in parallel, ~10.75 h wall clock | same instance class |
| outcome | 20/20 completed, 0 failed | 10/10 completed, 0 failed |
| integrity checked here | 30/30 manifests at `42dd79a`, status `completed`; split hash per seed identical to ours (23 compared); **829/829** checkpoint SHA-256 match | (included) |
| reports checked here | the frozen `decision_tree`, `statistics_report`, `h3_report`, `h4_report` of this branch, re-run on the bundle's `aggregate_conv100k`, reproduce the bundle's four reports byte-for-byte apart from timestamps, commit stamps and paths | — |

The grid extension is the one declared deviation from a pre-registered constant (`docs/PREREGISTRATION.md`
§12). It is additive — a 25,000-step run still selects exactly the same 19 checkpoints — and it is **not
merged** into this branch.

### 16.2 The B3 question: the MLP settles, and the gap closes (v)

`structured_fraction_of_live`, median over 10 seeds, at each block's final checkpoint:

| architecture | 25,000 steps (this study, §5) | 100,000 steps (S1) |
|---|---|---|
| transformer | 0.9824 | **1.0000** |
| MLP | 0.8848 | **1.0000** |
| gap, transformer − MLP | **+0.0977** | **0.0000** |

Two statistics of the same 25k metric appear in this document and should not be confused: **+0.0977** is
the difference between the two per-architecture medians; **+0.0850** (§5.1, §7) is the median of the ten
paired per-seed differences under the family definition, from `h3_report.json`. At 100,000 steps both are
zero (`conv100k_reports/h3_report.json`: `gap_under_top1_median = gap_under_family_median = 0.0`).

Per seed, within the S1 runs (the step-25,000 column is the S1 run's own checkpoint, on Linux, and differs
from our Windows runs at the third decimal — see 16.8):

| seed | MLP @25k | MLP @90k | MLP @100k | Δ 90k→100k | transformer @25k | @90k | @100k | Δ 90k→100k |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.8691 | 1.0000 | 1.0000 | 0.00 % | 1.0000 | 1.0000 | 1.0000 | 0.00 % |
| 1 | 0.8828 | 0.9941 | 0.9941 | 0.00 % | 0.9570 | 1.0000 | 1.0000 | 0.00 % |
| 2 | 0.8887 | 0.9219 | 0.9141 | 0.85 % | 0.9844 | 0.8770 | **0.8047** | **8.24 %** |
| 3 | 0.9551 | 0.9883 | 0.9922 | 0.40 % | 0.9707 | 0.9785 | 0.9824 | 0.40 % |
| 4 | 0.9336 | 1.0000 | 1.0000 | 0.00 % | 0.9941 | 1.0000 | 1.0000 | 0.00 % |
| 5 | 0.8633 | 0.9746 | 0.9902 | 1.60 % | 0.9980 | 1.0000 | 1.0000 | 0.00 % |
| 6 | 0.8574 | 1.0000 | 1.0000 | 0.00 % | 1.0000 | 0.9980 | 0.9980 | 0.00 % |
| 7 | 0.8672 | 1.0000 | 1.0000 | 0.00 % | 0.9746 | 1.0000 | 1.0000 | 0.00 % |
| 8 | 0.8828 | 1.0000 | 1.0000 | 0.00 % | 0.9766 | 0.9922 | 0.9980 | 0.59 % |
| 9 | 0.8496 | 1.0000 | 1.0000 | 0.00 % | 1.0000 | 1.0000 | 1.0000 | 0.00 % |

Convergence by the study's own rule (B9: < 5 % change between the last two checkpoints, here
90,000 → 100,000): **MLP 10/10 settled** (2/10 at the 25k budget, B3), **transformer 9/10** — seed 2's
fraction fell from 0.877 to 0.805 in the last 10,000 steps and is still moving. So at 100,000 steps the one
unsettled metric–seed pair in the block sits on the transformer side. All 20 runs stay generalized (minimum
final test accuracy 0.9992; no de-grokking).

What this settles: §5.1's prediction — a longer budget shrinks the gap — holds, and the "fixed-budget
comparison" caveat now has a measured endpoint for this one metric. What it does not settle: every other
number in this document is still a 25,000-step number.

### 16.3 The gate at 100,000 steps: `neither_passes`, as D6 predicted (v)

| criterion | transformer | MLP |
|---|---|---|
| G1 structure | 10/10 | 10/10 |
| G2 phase relation | 10/10 | 10/10 |
| G3 end-to-end formula | 10/10 | 10/10 |
| G4 causality | **0/10** | **0/10** |

Identical pass pattern to §3. D6 said a longer budget would make G4 *less* discriminating, because the
structured set grows; at 100,000 steps that set is ~100 % of the live neurons in both architectures, so the
size-matched control has nothing left to contrast against. Verdict unchanged, reason confirmed.

### 16.4 Key-frequency counts at 100,000 steps (v)

Per selection rule, median (range) over 10 seeds, from the runs' frozen `key_frequencies/step100000.json`;
the 25,000-step values of §5 alongside:

| rule | transformer @100k | transformer @25k | MLP @100k | MLP @25k |
|---|---|---|---|---|
| `nanda` (primary) | **4.5** (3–5) | 4.5 (3–8) | **11** (9–12) | 12 (9–14) |
| `neuron_clusters` | 4 (3–5) | 4 (3–5) | 8 (7–9) | 9 (8–11) |
| `embedding_threshold` | 5 (3–6) | 5 (3–5) | 7 (7–9) | 9 (8–10) |
| `logit_sum_directions` | 4 (3–5) | 3.5 (2–5) | 7 (6–9) | 7 (6–9) |
| `embedding_top8` | 5 (3–6) | 5 (3–5) | 7 (7–8) | 8 (8–8) |

The count difference — roughly 4–5 against 11 — is stable across budgets and across all five rules. *Which*
frequencies: seed-dependent in both architectures at 25k (§5) and, per the reviewer's S3 (16.6), at 100k.

### 16.5 Attention at 100,000 steps (v)

Median Δ test accuracy over the 10 transformer seeds, from the runs' frozen `causal_ablation/step100000.json`,
with §6's 25,000-step values:

| ablation | @100k | @25k (§6) |
|---|---|---|
| `fix_attention_to_mean` | **−0.415** (−0.549 … −0.375) | −0.336 (−0.506 … −0.201) |
| `ablate_head_k__mean`, k = 0…3 | −0.519 / −0.523 / −0.493 / −0.548 | −0.487 / −0.476 / −0.460 / −0.463 |
| `ablate_head_k__zero`, k = 0…3 | −0.537 / −0.532 / −0.562 / −0.598 | −0.471 / −0.464 / −0.443 / −0.491 |

The input-dependent part of attention costs more at 100k than at 25k. As in §6, this shows attention is
load-bearing; it does not show that attention is *why* the transformer uses fewer frequencies.

### 16.6 The reviewer's own analyses on the S1 checkpoints (r)

Reported from [`docs/external/S2_S3_FINDINGS.md`](docs/external/S2_S3_FINDINGS.md); outputs in
`training/results/external/conv100k/`. The scripts are **not** in the bundle, so none of this has been
re-derived here.

**S2 — leave-one-out causal ranking** (no spectral quantity enters the ranking), fixed cardinality grid,
50 random same-size controls. Median over 10 seeds:

| quantity | MLP | transformer |
|---|---|---|
| k\* necessity (remove-top-k test acc < 0.5) | 320 (62 %) | 128 (25 %) |
| random-order necessity at that k | acc ≈ 1.0 | acc ≈ 0.996 |
| k\* sufficiency (keep-top-k acc ≥ 0.9) | 192 (= random control) | 512, never below the full set (random control: 256) |
| neurons holding 90 % of one key frequency's power | ~50 (43–62) | ~93 (81–124) |
| Spearman, causal rank vs IPR rank | 0.37 | 0.52 |

Reported pattern: in the MLP the causal ranking buys nothing over random in either direction; in the
transformer the top quarter of neurons is collectively necessary but not sufficient. No group of tens of
neurons is decisive in either — consistent with §6.1's final-checkpoint result under a different ranking.

**S2b — percentile ablation ranked by each run's own key-frequency power**, against random and against a
non-key-power ranking. Median test accuracy after removal:

| removed | MLP key-ranked | MLP random | transformer key-ranked | transformer non-key-ranked | transformer random |
|---|---|---|---|---|---|
| 1 % (5) | 1.000 | 1.000 | 0.999 | 0.999 | 1.000 |
| 5 % (26) | 1.000 | 1.000 | 0.947 | 0.919 | 1.000 |
| 10 % (51) | 1.000 | 1.000 | 0.874 | 0.873 | 1.000 |
| 25 % (128) | 0.984 | 1.000 | 0.540 | 0.486 | 0.997 |
| 50 % (256) | 0.516 | 0.997 | 0.211 | 0.138 | 0.941 |

Reported diagnostics: in the MLP the key and non-key rankings pick disjoint neurons (overlap 0.00 at
1–10 %), i.e. frequency-specialised clusters; in the transformer the two rankings select nearly the same
neurons (correlation 0.965, overlap ≈ 1.0), i.e. its high-power neurons carry all of its frequencies at
once. This is the same asymmetry §6.1 measured at 25k with the structure-score ranking (MLP discriminable
only from 50 %, transformer from 5 %).

**S3 — frequency-family comparison across the 20 runs**, every family-aware number next to a size-matched
random-set null: within-architecture, cross-architecture and family-aware Jaccard all sit at the null; no
frequency is used by ≥ 8 of 10 seeds in either architecture (maximum 5/10, MLP frequency 4; 3/10,
transformer frequency 52); within a run the five selection rules agree (pairwise Jaccard 0.68–1.00). The
reported architecture difference is the *number* of frequencies (16.4), not their identity.

### 16.7 The two-hot control at 10 seeds (v for the runs and counts, r for S2)

Seeds 0–2 reproduce our three Windows runs behaviourally — identical split hashes; memorization step
230/230/230 against our 230/240/230; generalization step 12,575/14,100/12,125 against our
12,775/13,300/12,500 (v). Key-frequency counts at 25,000 steps from the runs' frozen artifacts (v): `nanda`
selects **all 56 frequencies in 10 of 10 seeds** (our three: 56/56/56), `neuron_clusters` 16 (12–19) against
our 14 (13–15), `logit_sum_directions` 0 in every seed — the rules disagree wildly, unlike the
shared-embedding models, because the two-hot logit map's spectrum is flat. Structured fraction of live
neurons 0.64–0.71, settled 6/10 (r). The reviewer's S2 on these checkpoints (r): k\* necessity 192, k\*
sufficiency 192 against a random control of 384, causal-vs-IPR Spearman 0.87–0.89, identical across all
10 seeds — the one configuration of the three where a ranked group is both necessary and sufficient and
beats its control.

Reported reading, for the authors to weigh (D9): changing only the input parametrization removes the
shared-embedding MLP's pattern, so "architecture" in any causal claim here must include the embedding.

### 16.8 Caveats

- **Platform.** S1 ran on Linux x86_64; our blocks on Windows AMD64, same torch and numpy. The S1 runs' own
  step-25,000 medians are 0.876 (MLP) and 0.989 (transformer) against our 0.885 and 0.982 — same seeds,
  different platform, third-decimal drift. This is why the runbook's reproduction condition exists, and it
  has not been met.
- **Scripts.** S2, S2b and S3 (and the two-hot S2) were produced by scripts that are not in the bundle;
  requested 2026-09-08. Until they arrive, 16.6 and the S2 part of 16.7 are the reviewer's numbers.
- **`run_manifest.csv`** in the bundle does not index the S1 runs; the index is `aggregate_conv100k/index.json`.
- **Nothing here reopens the gate**, and no 25,000-step number in §1–§15 was changed.

---
