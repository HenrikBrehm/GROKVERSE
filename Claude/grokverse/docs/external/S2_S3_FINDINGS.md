# S2 / S3 follow-up analyses — measured findings

**Status: measurements only.** Interpretation stays with the student authors
(`docs/HUMAN_DECISIONS.md` §F). Analysis-only work over the S1 (`conv100k`) final
checkpoints; no training, no threshold or pass rule of the frozen study changed.

Scientific target these analyses serve (stated by the mentor, to be evaluated by the
students against the numbers below): *"Both architectures use Fourier structure, but they
may differ in which frequency families they use, where those frequencies appear, and how
concentrated the causal computation is across neurons."*

| item | value |
|---|---|
| inputs | 20 S1 runs (seeds 0–9 × {transformer, mlp}), final checkpoint step 100000 |
| scripts | `training/run_s2_ranked_ablation.py`, `training/run_s3_frequency_families.py` (this commit) |
| reused, unmodified | `causal_ablation._mlp_pieces` / `evaluate` / `margin`, `transformer_mechanism.forward_decomposition`, per-run mechanism npz (alive masks, IPR, per-neuron per-frequency power), per-run `key_frequencies` artifacts |
| environment | macOS arm64, Python 3.12.13, torch 2.12.1 (plain build; the pinned `+cpu` wheel has no macOS build), numpy 2.4.6 — analysis-only deviation, no training |
| outputs | S3 `grokverse/conv100k/results/s2_ranked_ablation/` and `.../s3_frequency_families/` |

Design provenance: no formal S2/S3 spec existed in the repo; `HUMAN_DECISIONS.md` D6
names the requirement S2 implements — *"a fixed small cardinality, or selection by causal
contribution rather than by spectral shape"* — after measuring that every spectral-shape
definition selects 86–100 % of live neurons and cannot discriminate (G4).

---

## S2 — ranked graded ablation (causal-contribution ranking)

**Method.** Per run: rank all 512 hidden neurons by leave-one-out margin drop (the fall in
mean correct-minus-best-incorrect logit margin when that neuron's output contribution is
removed). No spectral quantity enters the ranking. Then over a fixed cardinality grid
k ∈ {0,1,2,3,4,6,8,12,16,24,32,48,64,96,128,192,256,384,512}: remove-top-k (necessity) and
keep-only-top-k (sufficiency), each against 50 random same-size sets of live neurons.
Transformer neuron ablation edits only the MLP path (attention and direct path intact),
the same convention as the existing `ipr_ranked_pruning`. Thresholds reused from
`causal_ablation.py`: necessity k\* = smallest k with remove-acc < 0.5; sufficiency
k\* = smallest k with keep-acc ≥ 0.9.

**Result (median over 10 seeds; per-seed values in `s2_summary.json`):**

| quantity | MLP | transformer |
|---|---|---|
| k\* necessity (remove-top-k acc < 0.5) | 320 (62 % of neurons) | **128 (25 %)** |
| random-order necessity at same k | acc still ≈ 1.0 at k=320 | acc ≈ 0.996 at k=128 |
| k\* sufficiency (keep-top-k acc ≥ 0.9) | 192 | **512 (never below the full set)** |
| random-control sufficiency k\* | 192 | 256 |
| Gini of LOO margin drops | 0.354 | 0.370 |
| top-16 share of total LOO margin drop | 0.075 | 0.108 |
| Spearman, causal rank vs IPR (spectral) rank | 0.37 | 0.52 |

Consistency: every one of the 10 transformer seeds has k\*-necessity ≤ 256 and
k\*-sufficiency = 512 except seed 2 (384). Every MLP seed has k\*-necessity ≥ 256 and
k\*-sufficiency within one grid step of the random control.

**Measured pattern, stated without interpretation:**

* **MLP:** the causal ranking buys almost nothing over random. Keep-only-top-192 reaches
  0.9 exactly where a random 192 does; removing the top-320 is needed to halve accuracy.
  No small ranked group is privileged — necessity and sufficiency are both distributed.
* **Transformer:** strong asymmetry. Removing the top-96–256 LOO-ranked neurons collapses
  test accuracy (random removal of the same size leaves it ≈ 1.0), so a quarter of the
  neurons are *collectively necessary*; but keeping only those same neurons does **not**
  restore function (keep-acc at k=384 is only ~0.64 median), while a random 256-neuron
  set suffices. The top causal group is necessary and not sufficient.
* A "small" group in the D6 sense (tens of neurons) is decisive in **neither**
  architecture: single-neuron LOO accuracy drops are ≈ 0 everywhere, and the top-16
  neurons carry ≤ 11 % of total margin contribution in both.

**Frequency specificity (links the ranking to the Fourier structure):**

| quantity (median) | MLP | transformer |
|---|---|---|
| neurons holding 90 % of one key frequency's power | ~50 (range 43–62) | ~93 (range 81–124) |
| share of key-freq power in the minimal sufficient group | 0.50 | 1.0 (degenerate: k\*=512 is the full set) |

Each individual key frequency is implemented across tens of neurons in both
architectures — no frequency is owned by a handful of neurons. The transformer spreads
each frequency roughly twice as widely as the MLP.

## S3 — frequency family comparison

**Method.** Per run, the frozen pipeline's own key-frequency artifacts (primary rule
`nanda`, four alternative rules as sensitivity). Pairwise similarity of the 20 runs' key
sets: raw Jaccard, harmonic-family-aware Jaccard (f matches g if one lies in the other's
{alias(j·f mod 113): j = 1..7} family — the mechanism module's own family convention),
and cosine of the L1-normalised 56-frequency score vectors. Every family-aware number is
reported next to a **size-matched random-set null** (mean over 200 draws of uniformly
random sets of the same cardinalities), because with 7-harmonic families an 11-element
set covers most of the spectrum by chance. Permutation test (10 000 draws) on the
within-architecture minus cross-architecture median gap.

**Result:**

| similarity (median) | within MLP | within TXF | cross-arch | size-matched null (w-MLP / w-TXF / cross) |
|---|---|---|---|---|
| Jaccard (raw sets) | 0.105 | 0.000 | 0.067 | 0.119 / 0.052 / 0.059 |
| family-aware Jaccard | 0.917 | 0.538 | 0.667 | 0.930 / 0.545 / 0.593 |
| score cosine | 0.268 | 0.230 | 0.252 | — |

* Observed similarity sits **at the size-matched null in every cell**. The
  family-aware permutation p = 0.0003 is a set-size artifact (MLP sets are larger, so
  their family cover is larger); against the null there is no excess.
* **Core families are empty**: no frequency is used by ≥ 8/10 seeds in either
  architecture. Maximum usage of any single frequency: 5/10 seeds (MLP, frequency 4),
  3/10 (TXF, frequency 52).
* Within-run harmonic organisation is also at chance: families-per-set-size ratio 0.50
  (MLP) vs null 0.53; 0.68 (TXF) vs null 0.73. Key sets are not built as
  fundamental-plus-harmonics stacks.
* The robust architecture difference is **how many** frequencies, not **which**:
  median primary-set size 11 (MLP, range 9–12) vs 4.5 (TXF, range 3–7). This holds
  under all five selection rules (`n_by_rule` in the output JSON).
* Rule agreement within a run (where the frequencies appear): at step 100000 the five
  rules — which read different objects (logit map, neuron clusters, embedding) — agree
  substantially within both architectures (pairwise Jaccard medians 0.68–1.00, all
  pairs, both architectures; per-pair values in `rule_agreement_within_run`), i.e.
  whichever frequencies a run uses, it uses them consistently across embedding, neurons
  and logits.

**Measured answer to the S3 question:** on this evidence the frequency sets are
**seed-dependent** in both architectures — within-architecture agreement is
indistinguishable from chance and from cross-architecture agreement. What separates the
architectures is the *cardinality* of the set (≈ 11 vs ≈ 4.5) and (from S2) how widely
each frequency is spread across neurons (≈ 50 vs ≈ 93 neurons per frequency) and the
necessity/sufficiency structure of the causal core.

## S2b — percentile ablation ranked by key-frequency contribution

**Method** (`training/run_s2b_freq_ranked_ablation.py`). Per run: rank all 512 neurons by
their summed power at the run's OWN primary key frequencies (from the mechanism npz),
remove the top 1 / 5 / 10 / 25 / 50 %, measure test accuracy against (a) 50 random
same-size live sets and (b) a specificity control ranked by power at NON-key frequencies.
Also recorded: the removed group's share of total key-frequency power and its
dominant-frequency composition.

**Result (median test accuracy over 10 seeds; per-seed in `s2b_summary.json`):**

| removed | k | MLP key-ranked | MLP nonkey | MLP random | TXF key-ranked | TXF nonkey | TXF random |
|---|---|---|---|---|---|---|---|
| 1 %  | 5   | 1.000 | 1.000 | 1.000 | 0.999 | 0.999 | 1.000 |
| 5 %  | 26  | 1.000 | 1.000 | 1.000 | 0.947 | 0.919 | 1.000 |
| 10 % | 51  | 1.000 | 1.000 | 1.000 | **0.874** | 0.873 | 1.000 |
| 25 % | 128 | 0.984 | 1.000 | 1.000 | **0.540** | 0.486 | 0.997 |
| 50 % | 256 | **0.516** | 0.900 | 0.997 | **0.211** | 0.138 | 0.941 |

**Diagnostics that decide what this means (seed-0 pair, representative):**

| | MLP | transformer |
|---|---|---|
| corr(key-power, nonkey-power) over live neurons | 0.54 | **0.965** |
| top-set overlap, key vs nonkey ranking (1–10 %) | **0.00** | 0.98–1.00 |
| distinct dominant frequencies in the removed top-10 % | ~2 of 11 key freqs | ~3.5 of ~4.5 |
| distinct dominant frequencies in the removed top-25 % | ~6 of 11 | ~4 of ~4.5 (all) |

**Measured pattern, stated without interpretation:**

* **Small groups are not decisive in either architecture** at 1 % (5 neurons): accuracy
  stays ≥ 0.999 everywhere.
* **MLP:** the top key-frequency contributors are frequency-specialised (the key and
  non-key rankings pick entirely disjoint neurons at 1–10 %), and removing whole
  frequency clusters is *harmless* until half the network is gone — the top-10 % group
  concentrates ~21 % of all key-frequency power inside ~2 of the 11 key frequencies, and
  accuracy stays at 1.000. Damage appears only at 25–50 %, and at 50 % the key-ranking
  hurts far more than the non-key ranking (0.516 vs 0.900) and than random (0.997) —
  the key-frequency ranking is *specific* for the MLP.
* **Transformer:** degradation starts at 5 % and is severe by 25 % (0.540 vs random
  0.997). But the specificity control **cannot dissociate**: key-power and non-key-power
  rankings select nearly the same neurons (correlation 0.965, top-set overlap ≈ 1.0), and
  the non-key ranking damages at the same rate. The transformer's high-power neurons
  participate in *all* its frequency families at once — the removed top-10 % already
  spans ~3.5 of its ~4.5 key frequencies.

**Family comparison in the ablation's terms:** the MLP implements its ~11 frequency
families as separable neuron clusters (disjoint top-contributor sets per family, loss of
2 whole families costs nothing — the remaining families carry the task); the transformer
implements its ~4.5 families non-separably in a shared high-power population (no
frequency-specialised grouping exists for the ranking to find). Combined with S3:
*which* families those are remains seed-dependent in both architectures; the
architectural signature is family **count** (11 vs 4.5) and family **separability**
(clustered vs entangled), not family identity.

## Two-hot control extended to 10 seeds (was 3)

**Why.** `HUMAN_DECISIONS.md` E4: the two-hot MLP keeps `d_mlp = 512`, p, split and budget,
so *only the input parametrization* differs from the shared-embedding MLP. At 3 seeds it
could not support the claim that the MLP's implementation pattern is architectural. Ten
runs (seeds 0–9) at the frozen 25,000-step budget executed on EC2 (Linux x86_64, torch
2.12.1+cpu — same environment as S1), byte-identical flags to the students' own
`results/matrix_twohot_20260904T022540Z.json` invocation, commit `42dd79a`. 10/10
completed, zero failures. Artifacts: `s3://…/grokverse/twohot10/`.

**Reproduction check (seeds 0–2 vs the students' Windows runs).** Split hashes identical;
memorization step identical or one eval-point apart (230 vs 230/240); generalization steps
within 200–800 steps (ours 12575/14100/12125 vs theirs 12775/13300/12500). Checkpoint
file SHA-256 differ from step 0, as expected across torch builds (container metadata);
behavioural quantities reproduce with normal cross-platform float drift.

**Result 1 — the two-hot spectrum is flat, not sparse.** Primary rule (`nanda`)
key-frequency count at the final checkpoint:

| model (25k budget) | n_keys median | range |
|---|---|---|
| shared-embedding MLP (students' 10 seeds) | 12 | 9–14 |
| transformer (students' 10 seeds) | 4.5 | 3–8 |
| **two-hot MLP (10 fresh seeds)** | **56 — all frequencies** | 56–56 |

All 10 fresh seeds and all 3 of the students' own seeds select every one of the 56
frequencies: the two-hot logit map's Fourier norms span only 8.8–13.6 (max/4 threshold
3.4), i.e. the spectrum is flat and the rule's notion of "key frequencies" does not
discriminate at all. (`neuron_clusters` gives 17, `embedding_threshold` 48,
`logit_sum_directions` 0 — the rules disagree wildly, unlike mlp/txf where they agree at
0.68–1.00.) Frequency-set comparisons (S3) and key-frequency-ranked ablation (S2b) are
therefore not meaningful for this model, which is itself the measurement.

**Result 2 — causal concentration is different from BOTH other models.** S2 LOO ranking
at the two-hot final checkpoint (budget caveat: 25k here vs 100k for mlp/txf):

| quantity (median) | shared-emb MLP @100k | transformer @100k | **two-hot @25k** |
|---|---|---|---|
| k\* necessity | 320 | 128 | **192** |
| k\* sufficiency | 192 (= random ctrl) | 512 (random: 256) | **192 (random: 384)** |
| causal-vs-IPR Spearman | 0.37 | 0.52 | **0.87–0.89** |

Identical across all 10 two-hot seeds. The two-hot model has a ranked top-192 group that
is simultaneously *necessary and sufficient* and clearly beats its random control — the
one configuration of the three where the causal ranking finds a genuinely privileged
group. The shared-embedding MLP tracks random; the transformer's core is necessary but
not sufficient.

**Result 3 — structured fraction.** `structured_fraction_of_live` at 25k: 0.64–0.71
(median ≈ 0.68), settled 6/10 under B9 — far below the shared-embedding MLP (0.88) and
transformer (0.98) at the same budget.

**What the control establishes (measurement, not interpretation):** the shared-embedding
MLP's pattern — sparse frequency sets, distributed causal structure — does **not**
survive changing only the input parametrization. Two-hot flips it to a flat spectrum
with a concentrated, ranking-visible core. So "MLP-ness" alone does not produce the
pattern reported for the MLP; the shared embedding is load-bearing in it. Any claim of
the form "the architecture causes the implementation differences" must name the
embedding parametrization as part of "architecture".

## Bearing on the stated scientific target

Point-by-point, measurements only:

* *"Both architectures use Fourier structure"* — supported by the existing pipeline
  (G1–G3 pass 10/10 for both at 100k; S1 findings doc).
* *"differ in which frequency families they use"* — **not supported**: family choice is
  seed-dependent at chance level in both; what differs is the number of frequencies.
* *"where those frequencies appear"* — within a run, the selection rules over different
  objects agree in both architectures; the between-architecture differences are in
  per-frequency neuron spread (≈ 2× wider in the transformer, S2) and in separability
  (frequency-specialised neuron clusters in the MLP vs a shared all-family population in
  the transformer, S2b).
* *"how concentrated the causal computation is across neurons"* — **this is where the
  architectures genuinely differ**, and in a direction with a twist: the transformer has
  a concentrated *necessary* core (quarter of neurons) that is nevertheless not
  sufficient; the MLP is uniformly distributed with no privileged group under either
  test.

## Caveats

* Analysis ran on macOS arm64 / torch 2.12.1 (plain); the S1 artifacts were produced on
  Linux x86_64 / torch 2.12.1+cpu. Baselines reproduced exactly (test acc 1.0, margins
  match the run files), but this is a different BLAS than the S1 instance.
* The k-grid tops out at 512, so "k\*-sufficiency = 512" means "not reached below the
  full set"; the transformer's `top_group_share = 1.0` at that point is degenerate and
  flagged in the tables.
* LOO ranking measures marginal contribution against the intact network; under
  redundancy it can rank poorly for *joint* sufficiency. That the transformer's keep
  curve lags its own random control is itself a finding about redundancy structure, but
  an interaction-aware ranking (greedy forward selection) would be the natural S2b.
* These scripts are new, unreviewed code (this commit); they reuse the frozen helpers
  for every load, forward pass, split and metric.
