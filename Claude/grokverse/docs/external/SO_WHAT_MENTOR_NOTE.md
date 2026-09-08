# The "so what" — mentor's assessment of what the S1/S2/S3 record earns

**Status: mentor-authored assessment, not the students' interpretation.** Every number
below is traceable to an artifact in this repo or the S3 bundle; nothing here changes a
result. RESULTS.md and the submitted narrative remain the students' to write
(`docs/HUMAN_DECISIONS.md` §F). Sources: `CONVERGENCE_FINDINGS_S1.md`,
`S2_S3_FINDINGS.md`, per-run `analysis/causal_ablation/step100000.json`.

## Three claims the existing data already earns, strongest first

### 1. A circuit found in one model is not a claim about the task

The Fourier mechanism in the grokking literature is a transformer result. This record
shows the mechanism *family* transfers to an MLP (G1–G3 pass 10/10 for both
architectures), but the *implementation* does not (S2/S2b: separable per-frequency
clusters vs one entangled high-power population), and the *specific frequencies* are seed
noise even within one architecture — no frequency is used by more than 5 of 10 seeds
(S3, all set overlaps at the size-matched null). So "we found the circuit for modular
addition" is not a statement about modular addition; it is a statement about one trained
network. That sentence is what the whole project buys, and it contradicts a common
implicit assumption in the mechanistic-interpretability framing.

### 2. The two solutions differ in robustness, not in quality

Both architectures reach test accuracy 1.000 (S1, all 20 runs ≥ 0.9992). The difference
is what each solution survives. The MLP can lose its top-10 % key-frequency contributors
— which contain ~2 entire frequency families — and still score 1.000; damage appears
only at 50 % removed (0.516). The transformer loses accuracy from 5 % removed (0.947)
and is at 0.540 by 25 %, because its top-10 % already spans ~3.5 of its ~4.5 families
(S2b). Compactness has a price, and the price is fragility. That reframes "more compact"
from an aesthetic observation into a functional one, at zero additional compute.

**Not to be said:** "the transformer is better." Both reach 1.000. The defensible word
is *more compact* (fewer frequencies: median 4.5 vs 11), and its measured counterpart is
*more fragile under targeted ablation*.

### 3. The controls decided every conclusion — and the record shows it

* G4 fails 0/10 for both architectures because every spectral-shape structured-neuron
  definition selects 86–100 % of live neurons, so the size-matched ablation cannot
  discriminate (D6) — the criterion, not the models, produced that verdict.
* The one apparently significant family result (family-aware Jaccard, permutation
  p = 0.0003) vanished against the size-matched random null: a set-size artifact (S3).
* The transformer finding in S2b exists only because of the key-vs-non-key ranking
  control: the two rankings select nearly the same neurons (correlation 0.965), which is
  itself the measurement that the transformer has no frequency-specialised grouping.

A project whose own record demonstrates "the control decides the conclusion" three times
is showing methodological maturity most student projects cannot document.

## Status of the three-part target claim

Target: *"differ in which frequency families they use, where those frequencies appear,
and how concentrated the causal computation is across neurons."*

| part | status |
|---|---|
| which families | answered — seed-dependent in both (S3) |
| how concentrated | answered — clusters vs entangled population (S2, S2b) |
| where the structure appears | **partially measured** — within-run rule agreement across objects (embedding / neurons / logit map) is high in both architectures (S3, Jaccard medians 0.68–1.00), but no dedicated per-object localisation analysis exists |

## Two honest gaps, and which one to close

**1. The two-hot control ran at 3 seeds — now CLOSED at 10 seeds.**
This control separates "the architecture causes the MLP pattern" from "concatenated
operand embeddings cause it". Extended to 10 seeds at the frozen 25,000-step budget on
EC2 (byte-identical flags to `results/matrix_twohot_20260904T022540Z.json`; seeds 0–2
reproduce the students' runs behaviourally — identical split hashes, memorization steps,
generalization within a few hundred steps). Outcome (`S2_S3_FINDINGS.md`, two-hot
section): the control *does not* reproduce the shared-embedding MLP's pattern — the
two-hot spectrum is flat (all 56 frequencies selected, every seed, consistent with the
students' own 3), and its causal structure is a concentrated necessary-AND-sufficient
top-192 core, unlike both other models. So the shared embedding is load-bearing in the
"MLP pattern", and claims about architecture must name the input parametrization as part
of it. This changed the shape of the conclusion, which is exactly why it was the gap
worth closing.

**2. The attention explanation is a hypothesis with indirect support.**
Measured at step 100000 over the 10 transformer seeds (`causal_ablation`, this bundle —
note these correct earlier informal figures):

* `fix_attention_to_mean`: Δtest-acc median **−0.415** (range −0.549 to −0.375)
* single-head mean-ablation: Δtest-acc median **−0.523** across the four heads
  (per-head medians −0.49 to −0.55; range over seeds −0.299 to −0.757)
* plus S2: the MLP-block causal core is necessary but not sufficient on its own.

So attention is functionally load-bearing, which is *consistent with* — but does not
demonstrate — "attention is why the transformer needs fewer frequencies". A joint
attention-state × ranked-neuron ablation on the existing checkpoints would move it from
plausible to evidenced (analysis-only; hours). It adds a new claim rather than defending
an existing one: a nice-to-have, not a must.

My view: the story is complete without either; the two-hot extension is the one that
protects an existing claim, so it is the one being done.
