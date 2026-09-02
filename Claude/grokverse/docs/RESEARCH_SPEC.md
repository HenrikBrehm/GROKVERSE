# Research Spec — Transformer vs. MLP on Modular Addition

**Research question**

> Do a Transformer and an MLP learn the *same* rule for modular addition in *different* ways?

| | |
|---|---|
| Repository | `HenrikBrehm/GROKVERSE` |
| Baseline commit | `d9434c1` (to be pinned as `baseline/pre-arch-study` before any change) |
| Status | **SPEC — FOR REVIEW. Nothing in this document has been implemented or measured.** |
| Authors | Henrik Brehm, Ali Kandora |
| External review | Dr. Jana Stucke |
| Drafted by | Claude Opus 5, from a human-written brief; **not yet human-reworded** (see §12). Re-verified against the code and all 16 run artifacts and corrected on 2026-09-01 — changelog in §14 |

---

## 0. How to read this document

This is a **plan, not a result**. Every claim is tagged:

| Tag | Meaning |
|---|---|
| **[MEASURED]** | Already in the repo, traceable to a run + figure. Verified while writing this spec. |
| **[AUDIT]** | A defect or gap found by reading the existing code for this spec. Verified against the source. |
| **[PLANNED]** | Not implemented. Nothing has been run. |
| **[UNVERIFIED]** | Claimed by an external source we have not yet read in the primary. |
| **[HUMAN]** | A decision the AI must not make. See §9. |

The point of the review is to catch a wrong plan *before* ~30–40 CPU-hours are spent on it.

---

## 1. Why this question, and why it is the actual contribution

The current README sells GROKVERSE as a reproduction of Nanda et al. 2023 plus a 3D explorer. That undersells it. A reproduction has a known answer; the interesting result in the repo is one nobody has told us the answer to:

**[MEASURED]** Under an identical, un-accelerated protocol (`p=113`, `train_frac=0.3`, `wd=1.0`, 3 seeds each), the Transformer concentrates **0.73 ± 0.07** of its embedding Fourier power in its top-8 frequencies. The MLP concentrates **0.44 ± 0.01**. Per-seed ranges do not overlap (Transformer 0.66–0.79, MLP 0.43–0.45). All six runs generalize — test accuracy crosses 0.95 in every run; final test accuracies are 0.965–0.994 (Transformer) and 0.975–0.994 (MLP). **Caveat:** these runs early-stop at the generalization crossing, so both concentrations are measured *at* the transition, before the cleanup phase — see §3.9.

Two models, same task, comparable accuracy, measurably different internal structure — measured, so far, at each model's own generalization crossing (§3.9). The current repo **observes** this and then interprets it in one sentence ("the MLP solves the same task with a more distributed frequency representation"). That interpretation is not tested anywhere.

The scientific contribution of this study is to replace that sentence with evidence. The question decomposes into three sub-questions that need different kinds of evidence:

1. **Do both implement a Fourier/phase-addition algorithm at all?** — mechanistic + causal evidence.
2. **If so, in what basis?** — is 0.44 *less structure*, or the *same* structure spread over harmonics that a top-8 metric cannot see?
3. **Does the structural difference explain the timing difference?** — the Transformer generalizes earlier in every seed, but only by ~1.3× un-accelerated.

Sub-question 2 is the one we think is actually load-bearing, and §4/H3 gives it a sharp, falsifiable form.

**Scope change vs. the current repo.** Grokking reproduction moves from headline to method section. The architecture comparison becomes the headline. The 3D explorer stays as the communication layer and is **frozen** until the analyses are final (§11) — the visualization must not steer the science.

---

## 2. What already exists (evidence inventory)

Verified against the code and run artifacts while writing this spec.

| Claim (from `README.md` / `RESULTS.md`) | Evidence | Seeds | Evidence level | Status |
|---|---|---|---|---|
| Transformer groks: memorize @145, generalize @8367 | `runs/txf_..._frac0.3_seed0/run.json` | 1 (+2 more seeds) | descriptive | **solid** |
| Grokking is seed-robust | 3 seeds/arch × 2 settings, all 12 grokked | 3 | descriptive | **solid** |
| Embedding concentrates into few frequencies after grokking | `analysis/fourier.py`; canonical 0.76 vs 0.32 for the single non-grokked run (`txf_add_p113_wd1.0_frac0.3_gf2.0_seed0`: Grokfast at `frac=0.3`, 4 000 steps, never generalized, final test acc 0.106) | 3 | correlational | **solid** |
| Restricted/excluded loss reproduces Nanda's progress measures | `analysis/progress_measures.py` | 1 | mechanistic | **methodologically broken — see §3.1** |
| The 8 key frequencies alone solve the task (restricted 0.002 vs full 0.031) | same | 1 | causal-ish | **invalid until §3.1 is fixed** |
| Transformer learns a sparser Fourier circuit than the MLP | `dominant_fraction` on `W_E` | 3 | correlational | **over-interpreted — see §3.3** |
| The MLP uses a "more distributed" solution | none beyond the above | 3 | none | **unsupported claim** |
| Attention explains earlier generalization | mean attention ≈ 0.50/0.50/0.001 | 2 runs (canonical + one Grokfast) | descriptive | **over-interpreted — see §3.4** |
| Transformer groks earlier in every seed | 3 seeds, no overlap (8367 < 9646) | 3 | descriptive | **solid in direction, weak in magnitude** |

**Never analysed at all** (this is the gap Jana identified): the MLP's `W_in`, `W_out`, `b_in`, its hidden activations; the Transformer's `W_Q/W_K/W_V/W_O/W_in/W_out/W_U`, its residual stream, its MLP neurons. Every structural claim in the repo rests on the **shared embedding matrix `W_E` alone**. Nanda et al. analyse weights *and* activations *and* the neuron→logit map; we reproduce only the embedding half.

---

## 3. Defects found in the pre-study audit

These are problems in the *existing* code. They must be fixed before the new study, because two of them feed the new study's metrics.

### 3.1 The restricted/excluded Fourier masks are too wide **[AUDIT]**

`training/grokverse/analysis/progress_measures.py:compute()`:

```python
keep_restr, is_key = _mode_indices(p, key_freqs)
mask_restr = keep_restr[:, None] & keep_restr[None, :]   # outer product of a 1D mask
mask_excl  = ~(is_key[:, None] | is_key[None, :])
```

`keep_restr` marks the constant row plus the `cos_k`/`sin_k` rows for the 8 key frequencies (17 of 113 rows). The outer product therefore keeps **every** pair of key rows, including *cross-frequency* blocks such as `cos(ω₁₈ a)·cos(ω₁₅ b)`.

The trig-identity circuit only ever uses the **same** frequency on both axes:

```
cos(ω_k(a+b)) = cos(ω_k a)cos(ω_k b) − sin(ω_k a)sin(ω_k b)
sin(ω_k(a+b)) = sin(ω_k a)cos(ω_k b) + cos(ω_k a)sin(ω_k b)
```

Cross-frequency products `k₁ ≠ k₂` are not part of that algorithm. Counted concretely for `p=113` and 8 key frequencies:

| Variant | 2D components **kept** by restricted | 2D components **removed** by excluded |
|---|---|---|
| Current implementation (`legacy_broad_mask`) | **289** of 12 769 | **3 360** of 12 769 (26.3%) |
| Same-frequency 2×2 blocks + constant | **33** | 32 |
| Sum-directions only (`cos/sin(ω_k(a+b))`) + constant | **17** | 16 |

So the current restricted loss gives the "circuit" ~9–17× more freedom than the hypothesis allows, and the current excluded loss deletes ~100× more of the logit tensor than the hypothesis targets. Both biases push in the direction of the reported conclusion: restricted looks *too good*, excluded looks *too destroyed*. `RESULTS.md §2` reports restricted 0.0020 vs full 0.031 and excluded 10.42 — those specific numbers are **not currently interpretable as evidence for the trig-identity circuit**.

**Answering Jana's question directly:** this is **not** a documented, deliberate deviation. The repo documents a *different* deviation — that losses are measured on the full `(a,b)` grid instead of Nanda's per-split protocol (`RESULTS.md` §2 and §5, `IDEAS_BACKLOG.md`). The mask width is documented nowhere. We are treating it as an implementation error until the primary source says otherwise, and it will be labelled as such.

**Fix [PLANNED]:** implement `analysis/mask_protocols.py` with named, separately tested variants — `nanda_exact`, `same_frequency_block`, `sum_directions_only`, `legacy_broad_mask` — reconstruct the exact protocol from arXiv:2301.05217 and Nanda's released notebook, report all variants side by side, and stop calling the legacy numbers a reproduction.

### 3.2 The number of key frequencies is fixed, not measured **[AUDIT]**

`analysis/fourier.py:dominant_frequencies` takes the smallest set reaching 90% of power, **capped at `max_k=8`**. Checked on all 16 existing runs: the cap binds in **every single one** — `n_keep == 8` always, for Transformer and MLP alike, including the multiplication runs at 0.16 concentration. The 90% threshold never fires.

So "top-8 concentration" is an honest name, but the key-frequency *count* has never been determined from data. (Nanda et al. report a handful — we recall 5 for their model, to be confirmed against the primary in WP-0.) Fixing k=8 for both architectures is exactly the assumption that H3 (§4) says may be wrong.

### 3.3 Structure is measured on `W_E` only **[AUDIT]**

For the MLP this is the wrong object. `models/mlp.py` concatenates the two operand embeddings *before* `W_in`, so what a hidden neuron `i` actually sees is:

```
u_a[a,i] = W_E[a] · W_in[:d_model, i]        # effective a-curve of neuron i
u_b[b,i] = W_E[b] · W_in[d_model:, i]        # effective b-curve of neuron i
pre[a,b,i] = u_a[a,i] + u_b[b,i] + b_in[i]
act[a,b,i] = ReLU(pre[a,b,i])
out[i,c]   = W_out[i,c]                       # effective output curve of neuron i
```

`W_E` is a *shared* table that both operand paths read through different halves of `W_in`. A per-neuron circuit can be perfectly periodic while the raw `W_E` spectrum looks diffuse — `W_in` is free to select and rotate. Measuring only `W_E` and concluding "distributed solution" is a non-sequitur.

Note the structural fact this makes visible: **the MLP is exactly additive in `(a,b)` up to the ReLU.** There is no path by which `a` and `b` interact before the nonlinearity. So in this architecture, ReLU is provably the *only* possible source of the multiplication in the trig identity. That makes the MLP the cleaner test case of the mechanism, not the messier one.

### 3.4 Attention evidence is a mean without a variance **[AUDIT]**

`analysis/attention.py` reports the mean over all 12 769 inputs of how much `=` attends to `a` / `b` / `=`. A 50/50 mean is consistent with (i) a genuinely constant 50/50 split and (ii) input-dependent attention that averages to 50/50. Those are different circuits: (i) makes attention a fixed sum — the Transformer becomes additive-then-ReLU, structurally like the MLP; (ii) gives the Transformer a multiplicative path the MLP does not have, which would be a real architectural asymmetry and a candidate explanation for the timing gap. **[PLANNED]** report per-input variance, per-head, plus a causal head ablation.

### 3.5 The two settings differ in two knobs at once **[AUDIT]**

Already disclosed in `RESULTS.md` §3, restated because it constrains the new design: the accelerated setting is (Grokfast, `frac=0.5`) and the un-accelerated is (no Grokfast, `frac=0.3`). The change in speed ratio from ~2.9× to ~1.3× cannot be attributed to either knob. §6/WP-5 decouples this.

### 3.6 Parameter count is an uncontrolled confound **[AUDIT]**

Computed from the actual model classes:

| | total params | per module |
|---|---|---|
| Transformer | **226 176** | `W_E` 14 592, `W_pos` 384, `W_Q/K/V` 16 384 each, `W_O` 16 384, `W_in` 65 536, `W_out` 65 536, `W_U` 14 592 |
| MLP (`d_mlp=512`) | **204 017** | `W_E` 14 464, `W_in` 131 072, `W_out` 57 856, `b_in` 512, `b_out` 113 |

The MLP has **9.8% fewer parameters**, and `weight_decay=1.0` acts on all of them, so the two models are also under different effective regularization pressure. **[PLANNED]** a parameter-matched control: `d_mlp = 572` gives 226 217 parameters, **+0.02%** vs the Transformer.

### 3.7 Grokking-time resolution is coarse and unreported **[AUDIT]**

`utils.log_step_schedule` places ~150 checkpoints geometrically. `detect_transition` returns the first *logged* step above threshold. At step ~8 000 the log grid spacing is ~600 steps (`steps=40000`, `n=150`), so "generalizes at step 8367" carries an unstated uncertainty of that order. The artifact is visible in the data: Transformer seeds 0 and 2 both report generalization at *exactly* step 8367, and MLP seeds 0 and 2 both at *exactly* step 9646 — different trajectories snapping to the same coarse grid points. **[PLANNED]** dense accuracy-only evaluation (train every 10, test every 25 steps), report every transition as an interval with its evaluation grid.

### 3.8 Thread count is logged but not pinned **[AUDIT]**

`run.json` records `torch_num_threads` because CPU reductions differ across thread counts, and `RESULTS.md` §5 reports final accuracies wobbling ~0.5% because of it. If the new run matrix is executed in parallel, thread counts vary per run and the matrix is not reproducible. **[PLANNED]** add `threads: int = 1` to `Config`, call `torch.set_num_threads(cfg.threads)`, parallelize across processes instead of threads.

### 3.9 "Final" states are early-stopped at the generalization crossing **[AUDIT]**

The un-accelerated cross-architecture runs were launched with `--early-stop-acc 0.95` (`PROGRESS.md`), and the artifacts confirm it: in **all six** `frac=0.3` runs the last logged step *equals* the detected generalization step (e.g. Transformer seed 0 stops at step 8367 of a 40 000-step budget). Consequences:

1. Every "final" structure number in this repo — including the headline **0.73 vs 0.44** — is measured **at the moment of generalization**, not after convergence. Nanda et al.'s *cleanup* phase, which further sharpens the circuit, has not yet happened in these snapshots.
2. The comparison is event-matched (each architecture at its own crossing) — defensible, but it is not the "post-convergence" state WP-6 plans checkpoints for, and no one has decided which of the two the headline claim is about.
3. This adds a mundane alternative explanation for the 0.73/0.44 gap alongside H3: **a difference in how much cleanup each architecture has done by its crossing, or would do afterwards**. §10 carries the corresponding risk row.

**Fix [PLANNED]:** the new protocol trains past the crossing under a pre-declared stopping rule (proposal: to `1.5×` the generalization step, capped at the step budget — §9) and reports every structure metric at **two declared points**: the generalization crossing (event-matched) and the final state (fixed post-generalization budget). The legacy at-crossing numbers remain valid *as* at-crossing numbers and are labelled as such.

---

## 4. Pre-registered hypotheses

Fixed **before** any new run. Each has a null hypothesis and a stated falsification criterion, and each is answerable by an experiment in §6. We expect some of these to fail; a failed H2 or H3 is a reportable result, not a problem.

### H1 — Shared mathematical principle

Both architectures implement phase addition: they represent each operand by Fourier features at a small set of frequencies and combine them via the angle-addition identity, reading out the class whose phase matches.

- **Prediction:** for structured neurons in both models, the dominant frequency of the `a`-side curve, the `b`-side curve, and the output curve agree, and the phases satisfy `φ_out ≈ φ_a + φ_b (mod 2π)`.
- **H1₀:** phase errors are indistinguishable from the permutation null (neuron identities shuffled).
- **Falsified if:** the circular phase error distribution overlaps the null at the 95% bootstrap level in either architecture.

### H2 — Different parametrization of that principle

The Transformer forms a near-sinusoidal representation; the ReLU MLP forms effective weight curves better described by an odd-harmonic stack (square-wave-like) than by a single sinusoid.

- **Note on provenance:** the square-wave framing is attributed to arXiv:2603.23784 **[UNVERIFIED]** — we have not read the primary source yet. WP-0 includes reading it. If we cannot verify the claim in the primary, H2 is restated without the attribution as a plain model-comparison question (sine vs. odd-harmonic stack), which is testable on its own.
- **Prediction:** per-neuron model comparison (§5) favours the odd-harmonic model for a significantly larger fraction of MLP neurons than Transformer neurons; amplitudes decay approximately as `1/j` over odd harmonics `j = 1,3,5,…`.
- **H2₀:** the same model wins for the same fraction of neurons in both architectures.
- **Falsified if:** the paired per-seed difference in "fraction of neurons best fit by the harmonic model" has a bootstrap CI containing 0.

### H3 — 0.73 vs 0.44 is a measurement artifact, not less structure *(the key hypothesis)*

The MLP's lower top-8 concentration does **not** mean it is less algorithmically structured. If the MLP's effective weights are square-wave-like at a few fundamentals `k`, their energy necessarily sits at the **odd harmonics** `3k, 5k, 7k, …` — and in `ℤ₁₁₃` those alias to *distinct, scattered* frequency indices. Concretely, for the key set of the canonical run:

| fundamental k | 3k | 5k | 7k |
|---|---|---|---|
| 18 | 54 | 23 | 13 |
| 15 | 45 | 38 | 8 |
| 11 | 33 | 55 | 36 |
| 1 | 3 | 5 | 7 |

(using `k' = min(jk mod p, p − (jk mod p))`, since `cos` identifies `k` with `p−k`).

A single square-wave neuron at fundamental 18 therefore *spreads* its power across k = 18, 54, 23, 13, … A metric that keeps the top 8 *individual* frequencies will score it as unstructured. This is a mechanism by which 0.44 is fully compatible with a *perfectly clean* algorithm.

- **Prediction:** define **harmonic-family concentration** = fraction of power in `{alias(jk) : k ∈ fundamentals, j ∈ {1,3,5,7}}`, with `alias(x) = min(x mod p, p − (x mod p))`. Under H3 this closes most of the 0.73/0.44 gap, while plain top-8 does not.
- **The harmonic cap `j ≤ 7` is load-bearing, not cosmetic.** Without it the metric is degenerate: for odd `p`, the odd multiples of *any* single fundamental hit every residue class of ℤ_p, so the uncapped family covers **all 56 frequencies** and the concentration is identically 1 for every model. `j ≤ 7` retains ~95% of an ideal square wave's power (`1 + 1/9 + 1/25 + 1/49 ≈ 0.95 · π²/8`) and caps the family at ≤ `4·n_f` frequencies.
- **Degrees-of-freedom control (mandatory):** a family of `n_f` fundamentals contains up to `m = 4·n_f > 8` individual frequencies, so it could close the gap merely by keeping more components. Every harmonic-family number is therefore reported next to (i) the plain **top-m** concentration at the same cardinality `m = |family|` and (ii) a null over random frequency sets of size `m`. H3 is supported only if the harmonic family closes the gap *beyond* the matched top-m control.
- **Selection rule, fixed in advance:** per model, fundamentals are chosen greedily — repeatedly add the fundamental whose capped family adds the most power — with the same `n_f` for both architectures (`n_f` set in §9).
- **H3₀:** harmonic-family concentration shows the same architecture gap as the cardinality-matched top-m concentration.
- **Falsified if:** the paired per-seed gap reduction of the harmonic-family metric relative to the matched top-m control has a bootstrap CI containing 0.
- **Consequence if confirmed:** the current headline claim ("Transformer learns a sparser circuit") must be retracted and replaced with "the two architectures use the same circuit in different bases", and `RESULTS.md` §3 is rewritten. We commit to this in advance.

### H4 — Structure precedes generalization

Mechanistic structure metrics rise during the memorization plateau, before test accuracy moves.

- **Prediction:** on paired checkpoints, structure metrics rise measurably before the generalization step in both architectures.
- **H4₀:** structure metrics only rise after the test-accuracy transition.
- **Falsified if:** no structure metric separates from its initialization value before the generalization step.

### H5 — Causal relevance

Removing the identified structured components damages the model far more than a size-matched random control ablation.

- **Prediction:** loss increase from structured-component ablation ≫ loss increase from control ablation of equal size, in both architectures.
- **H5₀:** structured and control ablations do equal damage.
- **Falsified if:** the paired difference in damage has a bootstrap CI containing 0. *(This is the hypothesis most likely to fail, because §3.1 shows our current ablation machinery is not yet trustworthy.)*

---

## 5. Metrics

**Primary** (pre-declared; these decide H1–H5):

| Metric | Definition | Applies to |
|---|---|---|
| Top-8 concentration | fraction of Fourier power in the 8 strongest individual frequencies | both, all weight objects |
| Harmonic-family concentration | fraction of power in odd-harmonic families (`j ≤ 7`) of `n_f` fundamentals, always next to the cardinality-matched top-m control (§4/H3) | both |
| Phase error | circular `φ_out − (φ_a + φ_b)`, median + bootstrap CI, vs permutation null | both |
| Model-comparison win rate | fraction of neurons where the odd-harmonic model beats the single sinusoid by AIC | both |
| Causal ablation damage | `Δ` test loss, structured ablation minus size-matched random control | both |
| Generalization step | first step with test acc ≥ 0.95, reported as an interval | both |

**Secondary:** spectral entropy, participation ratio, top-1/top-3 concentration, cardinality-matched top-m concentration and the random-set null (the H3 controls), explained variance of the key-frequency subspace, per-class logit shift, margin change, weight norms.

**"Structured neuron" — definition must be fixed before running [HUMAN, §9].** Proposal for approval: a neuron is *structured* iff (i) its `a`-curve and `b`-curve share a dominant frequency `k`, (ii) that frequency explains ≥ 50% of the curve's variance (or a harmonic family of `k` does), and (iii) its output curve's dominant frequency is also `k`. Thresholds are declared now and not tuned afterwards. A sensitivity analysis over ≥2 alternative thresholds is mandatory.

**Fairness constraint.** Every metric used to compare architectures must be defined on objects that mean the same thing in both. Where that fails (e.g. the Transformer has `W_pos`, `W_Q/K/V/O`; the MLP does not), the comparison is reported as architecture-specific and excluded from the headline claim. This is stated per metric in `docs/METHODS.md`.

---

## 6. Work packages

Ordered. Each ends with a checkpoint: tests run, files listed, open risks named.

### WP-0 — Baseline freeze and source reading
- Tag `baseline/pre-arch-study` at `d9434c1`; archive current `runs/`, `figures/`, `RESULTS.md`. `training/runs/` is **gitignored**, so the tag preserves neither the run artifacts nor the checkpoints — "archive" means copying `runs/` to an `archive/` directory or a GitHub release *before* anything re-runs (`figures/` and `RESULTS.md` are tracked). Nothing existing is deleted or overwritten.
- Read the primary sources and record section/equation numbers for every method reused: Nanda et al. arXiv:2301.05217 (+ released notebook), Power et al. arXiv:2201.02177, Lee et al. arXiv:2405.20233, and **arXiv:2603.23784** (Jana's reference — read before H2 is finalized).
- Pin `threads=1` (§3.8).
- **Output:** `docs/METHODS.md` (source → method → our implementation, one row per method).

### WP-1 — Mask protocol audit and correction
- Reconstruct Nanda's exact restricted/excluded protocol from the primary. Document: which axes are transformed, which components are kept/removed, whether cross-frequency blocks are permitted, how cos/sin pairs and the constant term are handled, how key frequencies are chosen, and which split each loss is evaluated on.
- Implement `analysis/mask_protocols.py` with the four named variants of §3.1; keep `legacy_broad_mask` runnable for comparison but never label it a reproduction.
- **Tests (must pass before any conclusion uses these numbers):** 2D Fourier round-trip; exact kept/removed component counts per variant (289/33/17 and 3360/32/16 as computed in §3.1); no cross-frequency blocks in the exact variants; correct constant-term handling; synthetic ideal Fourier circuit → restricted ≈ full and excluded ≫ full; injected single frequency recovered; injected mixed frequencies separated; train- and test-split evaluation separated; vectorized implementation matches a slow reference loop.
- **Output:** `docs/MASK_PROTOCOL_AUDIT.md`; existing restricted/excluded numbers in `RESULTS.md` relabelled "not directly comparable to Nanda et al." until re-measured.

### WP-2 — MLP mechanism analysis
- `analysis/mlp_mechanism.py`: per-neuron effective curves `u_a`, `u_b`, bias, output curve, 2D pre-activation and activation maps, per-neuron logit contribution. **Numerical results saved, not only figures.**
- Per neuron: dominant frequency of each of the three curves, power spectrum, spectral entropy, participation ratio, top-1/3/8, fundamental vs odd-harmonic vs even-harmonic energy share, phases.
- `analysis/wave_fitting.py`: fit (1) single sinusoid, (2) ideal square wave, (3) odd-harmonic series truncated at `j = 7` (the same cap as §4/H3); compare by normalized MSE, R², AIC. **Report the full distribution over neurons and seeds, not the best fit.**
- Phase relation `φ_out ≈ φ_a + φ_b` with circular statistics and a permutation null.
- 2D activation analysis: `(a+b)` vs `(a−b)` dependence, symmetry under `a ↔ b`, variance explained by the hypothesized periodic function.
- **Output:** `docs/MLP_MECHANISM_DERIVATION.md`.

### WP-3 — Transformer mechanism analysis
- `analysis/transformer_mechanism.py`. Derive the neuron→logit map from the actual architecture and document the derivation. Because `models/transformer.py` has **no LayerNorm**, the map is exactly linear:

  ```
  neuron_logit_map = W_out @ W_U          # [d_mlp, vocab]
  ```

  i.e. neuron `f`'s contribution to the logit for class `c` is `act_f · (W_out[f,:] @ W_U[:,c])`, added to the residual stream. This is the Transformer's counterpart to the MLP's `W_out[i,c]` and makes the two architectures comparable on the same footing. **The absence of LayerNorm is what makes this exact — state it, do not silently rely on it.**
- Analyse `W_E`, `W_U`, `W_Q/K/V/O`, `W_in`, `W_out`, position-dependent residual streams, hidden activations, per-head and per-neuron logit contributions.
- Attention properly: per head, per input, with **variance** (§3.4), by training phase, plus causal head ablation.
- Apply the same structure metrics as WP-2 wherever mathematically meaningful; document where they are not.
- **Output:** `docs/TRANSFORMER_MECHANISM_DERIVATION.md`.

### WP-4 — Causal ablations
`analysis/causal_ablation.py`. All ablations run on **unmodified checkpoints**; no re-training. Every ablation gets a **size-matched random control**.

*MLP:* keep only structured neurons; keep only unstructured neurons; remove key frequencies from effective weight curves; keep only key frequencies; replace effective weights by their best sinusoid fit; replace by their best harmonic fit; compare right-frequency/wrong-phase neurons against right-phase/weak-periodicity neurons.

*Transformer:* keep/remove key Fourier components; keep/remove high-explanation MLP neurons; ablate individual attention heads; project out key-frequency subspaces of the residual stream; keep only the reconstructed trigonometric circuit.

Measure after each: train/test loss and accuracy, logit change, margin change, per-class change, across seeds. **Report ablations that fail to damage the model** — a structured component whose removal does nothing is a finding about H5.

### WP-5 — Controlled paired comparison
- **Primary setting** (pending §9 approval): `p=113`, addition, `train_frac=0.3`, `wd=1.0`, **no Grokfast**, ≥8 paired seeds (target 10). Identical seeds, splits, thresholds, evaluation frequency, optimizer, and **stopping rule** (§3.9/§9 — *not* the legacy early-stop at the 0.95 crossing) for both architectures. Analysis is **paired by seed**.
- **Confound decoupling** (§3.5), 3 paired seeds per cell: {no Grokfast, Grokfast} × {`frac=0.3`, `frac=0.5`}. No effect is attributed to Grokfast or to `train_frac` alone until they are separated.
- **Parameter control** (§3.6): repeat the primary comparison with `d_mlp=572` (+0.02%). Report "same hyperparameter dimensions" and "matched parameter count" as two distinct comparisons.
- **Input-parametrization control:** a direct two-hot-input MLP (per arXiv:2603.23784, once verified) alongside the shared-embedding MLP, to separate *architecture* effects from *input parametrization* effects. This is a literature control, not the main comparison.
- Failed runs are never silently dropped; the manifest records started vs. completed.

### WP-6 — Grokking-time measurement
Dense accuracy evaluation (train every 10 steps, test every 25); full checkpoints only at pre-declared points — init, pre-memorization, memorization, mid-plateau, pre-generalization, generalization, post-convergence. Two of these (mid-plateau, pre-generalization) exist only *relative to a finished run*, so runs that need them are measured in **two passes**: pass 1 trains with dense accuracy logging and locates the transition; pass 2 re-trains deterministically and captures states at the positions computed from pass 1 — the machinery of `progress_measures.train_capturing_states`, which demonstrably recovers recorded transitions step-exactly. Live-detectable positions (init, memorization crossing, generalization crossing, final state) are saved in pass 1 directly; only the ~6 structure-over-time runs (§8) need pass 2. "Fixed in advance" means the *rule* mapping a measured transition to checkpoint positions is pre-registered — the absolute step numbers cannot be. Every transition reported as `[previous evaluated step, first crossing]` with its evaluation grid. The phrase "exact transition" is banned. Sensitivity analysis at ≥1 alternative threshold besides 0.99/0.95.

### WP-7 — Statistics
`analysis/statistics.py` + `docs/STATISTICAL_ANALYSIS_PLAN.md`, written before the runs. Paired-by-seed differences for generalization step, grok gap, structure metrics, ablation damage. Bootstrap CIs, robust location/spread, **all individual seeds plotted**. Effect size and direction always reported. p-values permitted but never the sole evidence. With n≈10: no false precision, no "architecture X is fundamentally faster" from one hyperparameter region.

### WP-8 — Write-up
`RESULTS.md` rewritten only after analyses are frozen. Each claim structured as: observation → quantitative evidence → alternative explanation → causal test → limitation → permitted conclusion. Graded language ("we observe", "this is consistent with", "the data do not rule out", "under the conditions tested"). Banned without sufficient evidence: "proves", "clearly shows", "Transformers fundamentally …".

---

## 7. The mathematical derivation (answering Jana's question directly)

Jana asked which of two things we want. **Both** — and the spec keeps them separate, because they are different kinds of statement.

### 7.1 Part A — Why a Fourier representation solves `(a+b) mod p` *(pure mathematics, no network involved)*

Deliverable `docs/MATH_DERIVATION.md`. The argument in five steps:

**(1) Represent.** For frequency `k`, write `ω_k = 2πk/p` and map `n ↦ (cos(ω_k n), sin(ω_k n))`. Because `ω_k(n+p) = ω_k n + 2πk`, this map is *automatically* periodic mod `p`. The modular reduction is not computed — it is built into the representation. This is the real reason Fourier features are the natural basis for a cyclic group.

**(2) Combine.** The angle-addition identities

```
cos(ω_k(a+b)) = cos(ω_k a)cos(ω_k b) − sin(ω_k a)sin(ω_k b)
sin(ω_k(a+b)) = sin(ω_k a)cos(ω_k b) + cos(ω_k a)sin(ω_k b)
```

turn *addition of numbers* into a *bilinear operation on features*. Note what this requires: a **product** of an `a`-term and a `b`-term. A purely linear network cannot do it.

**(3) Read out.** For each candidate class `c`, compute

```
Logit(c) = Σ_{k ∈ K} α_k · cos(ω_k (a + b − c))
```

Expanding `cos(ω_k(a+b−c)) = cos(ω_k(a+b))cos(ω_k c) + sin(ω_k(a+b))sin(ω_k c)` shows this is an inner product between the features of `a+b` and the features of `c` — i.e. exactly what an unembedding matrix whose rows are Fourier features of `c` computes. Every cosine attains its maximum simultaneously iff `c ≡ a+b (mod p)`; elsewhere the phases disagree and partially cancel. In the limit of all frequencies with equal weights the readout is exactly a delta function, since

```
Σ_{k=1}^{(p−1)/2} cos(2πkm/p) = (p·1[m ≡ 0 mod p] − 1) / 2
```

With a sparse `K` this becomes an approximation that is still argmax-correct with a margin — which is why a handful of frequencies suffices.

**(4) Where the product comes from in a ReLU net.** Step (2) needs a multiplication, and neither of our models has one. What they have is `ReLU(u_a(a) + u_b(b) + β)` — an **additive** pre-activation followed by a nonlinearity. Expanding the rectification of a sum of two sinusoids of the same frequency produces cross terms in `(a+b)`. So the ReLU *is* the multiplier. This yields the concrete, testable prediction the whole empirical study turns on:

> A neuron whose `a`-curve has frequency `k` and phase `φ_a`, and whose `b`-curve has frequency `k` and phase `φ_b`, should have an output curve of frequency `k` and phase `φ_out ≈ φ_a + φ_b`.

That is H1, and it is measurable per neuron.

**(5) Why square waves, if they appear.** A square wave of fundamental `k` has the Fourier series `(4/π)·Σ_{j odd} sin(jω_k n)/j` — **odd harmonics only, amplitudes ~1/j**. Both facts are directly checkable on a fitted neuron, and they are what make H2 and H3 falsifiable rather than decorative. §4/H3 spells out why this changes the interpretation of 0.73 vs 0.44.

### 7.2 Part B — Showing *our* trained networks actually do this

Part A says a Fourier circuit *can* solve the task. It says nothing about what our networks learned — a plausible story we cannot refute is exactly what the project set out to avoid. Part B is the empirical half: WP-2, WP-3 (does the structure exist?) and WP-4 (is it load-bearing?). The evidence chain we commit to:

1. The structure is present (spectra, fits) — *descriptive*.
2. It has the predicted internal relations (phases add) — *mechanistic*.
3. Removing it breaks the model far more than removing an equal amount of anything else — *causal*.

Only step 3 licenses the word "algorithm". §3.1 is why we do not currently have step 3.

---

## 8. Compute budget

CPU-only, as the whole project has been. **The previous anchor was optimistic:** the measured ~24 min canonical Transformer run had early-stopped at the generalization crossing (~8 400 of 40 000 steps, §3.9), and the legacy cross-arch re-run block took ~4 h for 5 such early-stopped runs (`PROGRESS.md`). The new protocol trains past the crossing (`1.5×` rule, §9) and pins `threads=1` (§3.8). Estimates assume the `1.5×` rule and carry roughly ±50% uncertainty:

| Block | Runs | Estimate |
|---|---|---|
| Primary paired setting (10 seeds × 2 arch, `1.5×`-crossing rule) | 20 | ~10–14 h |
| Confound matrix (4 cells × 2 arch × 3 seeds) | 24 | ~6–9 h — the {no Grokfast, `frac=0.5`} cells have never been run; their grokking time is a guess |
| Parameter-matched control (10 seeds, MLP `d_mlp=572`) | 10 | ~4–6 h |
| Two-hot MLP control (3 seeds) | 3 | ~1.5 h |
| Checkpoint-capturing re-trains for structure-over-time (pass 2, WP-6) | ~6 | ~3–4 h |
| Ablations + analysis (no training) | — | ~2 h |
| **Total** | ~63 runs | **~30–40 h summed single-run time** |

Parallelize across **processes** with `threads=1` each (§3.8), never across threads; the totals above are sums of single-run times, so K parallel workers divide the wall clock by ~K. Full checkpoints: ~0.9 MB per Transformer state; 7 pre-declared checkpoints per run keeps storage well under 1 GB. Large artifacts go to a GitHub release, referenced by hash from the manifest, not committed.

Every run writes a row to `results/run_manifest.{csv,json}`: run ID, architecture, seed, split hash, git commit, Python/torch versions, device, thread count, full config, start/end time, status, abort reason, checkpoint paths, result files.

---

## 9. Decisions the human authors must make **[HUMAN]**

Implementation does not start until `docs/HUMAN_DECISIONS.md` reads `STATUS: APPROVED BY HUMAN AUTHORS`. That file is filled in by Henrik and Ali, not by the model. Open items:

1. Final wording of the research question.
2. Primary hypothesis and null hypothesis (H3 is proposed as primary — confirm or change).
3. Primary vs. secondary metrics (§5), including the harmonic-family parameters: `n_f`, the greedy selection rule, and the `j ≤ 7` cap (§4/H3).
4. Number of seeds (proposed: 10, minimum 8).
5. Primary setting: `p=113`, addition, `frac=0.3`, `wd=1.0`, no Grokfast (confirm).
6. **Definition of a "structured neuron"** and its thresholds (§5) — this decides several results and must be fixed in advance.
7. Which causal ablations are in scope given the compute budget.
8. How arXiv:2603.23784 is handled: replicate its setup, cite it as related work, or use it only as a control (§6/WP-5).
9. Whether the legacy broad-mask numbers stay in `RESULTS.md` as a documented variant or are removed.
10. Division of labour between Henrik and Ali, and **which code and analysis the two of you write yourselves** (BWKI cares; so does §12).
11. Whether the explorer is updated for the new results before submission, or frozen at its current state.
12. **Stopping rule and measurement points** (§3.9): approve the proposal — train to `1.5×` the generalization crossing (capped at the step budget) and report every structure metric at both the crossing and the final state — or set a different rule. The legacy early-stop-at-the-crossing protocol must not be carried forward silently.

---

## 10. Risks

| Risk | Why it matters | Mitigation |
|---|---|---|
| H3 confirmed → current headline claim is wrong | `README.md` and `RESULTS.md` §3 both assert the sparsity difference | Pre-committed in §4 to retract and rewrite; the retraction is itself a reportable result |
| The 0.73/0.44 gap is partly a stopping-time artifact — legacy runs stop at the 0.95 crossing (§3.9); cleanup continues afterwards | A mundane timing difference could masquerade as an architectural one | Re-measure at two pre-declared points (crossing + final state); report the gap at each |
| Nanda's exact protocol turns out to permit cross-frequency blocks | Then §3.1 is not a defect and WP-1 partly dissolves | WP-1 reconstructs from the primary *before* labelling anything an error |
| arXiv:2603.23784 unavailable or says something different | H2's square-wave framing is currently unverified | H2 is restated as a plain model comparison; §4 already provides that fallback |
| n≈10 too small for the timing effect (Transformer seed std was ±1196) | Timing conclusions could be noise | Report CIs and all seeds; if the CI spans 0, say so — the *structural* result does not depend on timing |
| Ablation damage similar for structured and control components | H5 fails | Report it; a negative causal result is more valuable than a decorative plot |
| Compute overruns before the BWKI deadline | ~30–40 h of runs plus analysis (§8) | Run WP-5's primary setting first; the confound matrix and controls are separable increments |
| Analysis code changes after seeing results | Silent p-hacking | Freeze analysis code after the single-seed pilot (WP-5 ordering); record the freeze commit |

---

## 11. Deliverables

**Documentation:** `docs/CURRENT_EVIDENCE_AUDIT.md`, `HUMAN_DECISIONS.md`, `PREREGISTRATION.md`, `METHODS.md`, `MASK_PROTOCOL_AUDIT.md`, `MATH_DERIVATION.md`, `MLP_MECHANISM_DERIVATION.md`, `TRANSFORMER_MECHANISM_DERIVATION.md`, `STATISTICAL_ANALYSIS_PLAN.md`, `CAUSAL_ABLATION_PLAN.md`, `LIMITATIONS.md`, `CLAIM_EVIDENCE_TABLE.md`.

**Code** (under `training/grokverse/analysis/`): `mask_protocols.py`, `mlp_mechanism.py`, `transformer_mechanism.py`, `wave_fitting.py`, `causal_ablation.py`, `statistics.py`.

**Tests** added to `training/test_core.py` (currently 23 checks): mask cardinality per variant; no cross-frequency components in exact variants; synthetic ideal Fourier circuit; synthetic square wave; phase recovery; per-neuron effective MLP weights against a hand-computed reference; causal ablation on a hand-built toy model; parameter counts; paired-seed assignment; run-manifest validation.

**Execution order:** freeze baseline → audit → **human gate** → preregistration → fix masks → synthetic tests → MLP mechanism → Transformer mechanism → **1-seed pilot** → **freeze analysis code** → full seed runs → causal ablations → statistics → human interpretation → documentation → explorer.

The full run matrix does not start before the analysis code is frozen.

---

## 12. Acceptance criteria

1. Mask methodology checked against the primary source; old and new variants clearly separated.
2. Every scientifically load-bearing function has an automated test.
3. Effective MLP weights analysed, not just `W_E`.
4. Hidden activations and output weights analysed in **both** architectures.
5. ≥2 causal ablations per architecture, each with a size-matched control.
6. Primary setting has ≥8 paired seeds.
7. All main effects reported with confidence intervals and all individual seeds shown.
8. Parameter count and input parametrization tested as confounds.
9. Every grokking time reported with its measurement interval.
10. Every figure reproducible from stored raw data.
11. Negative and null results documented.
12. `README.md` claims match the actual evidence level.
13. `AI_DISCLOSURE.md` describes the real process, with `[HUMAN AUTHORS MUST COMPLETE]` placeholders wherever humans have not yet done the work.
14. **The final scientific interpretation is written by Henrik and Ali in their own words.** This document is an AI-drafted plan; it is not human-authored interpretation and must not be presented as such.
15. Every structure metric states the training-time point it was measured at; at-crossing and final-state values are reported separately and never mixed in one comparison.

---

## 13. Non-goals

Not in scope: new architectures beyond the three named (KAN etc.), a full hyperparameter sweep, multiplication mod p as a second study, prime sweeps, explorer features, and any post-hoc threshold tuning that makes a result look stronger.

---

## 14. Revision notes — 2026-09-01 verification pass

Re-verified against the code and all 16 run artifacts: every §3 audit item was re-checked against the source, and the §1/§2 numbers were re-computed from `embeddings.npy` / `run.json` (top-8 fractions, per-seed ranges, transition steps, the §3.2 cap, the ~0.16 multiplication concentration). Corrections:

1. **§1:** "both reach ≥98% test accuracy" was false (Transformer seed 1: 0.965; MLP seed 1: 0.975) — replaced with the measured ranges.
2. **§3.9 (new):** all un-accelerated runs early-stop at the 0.95 crossing (`PROGRESS.md`; last logged step = generalization step in all six runs), so every legacy structure number — including 0.73/0.44 — is measured *at* the transition, pre-cleanup. Stopping rule and dual measurement points propagated into §1, §6 (WP-5/WP-6), §8, §9, §10, §12.
3. **§4/H3:** the harmonic-family metric was degenerate as written — without a harmonic cap, the odd family of any single fundamental covers all 56 frequencies of ℤ₁₁₃ and the metric is identically 1. Pre-registered the `j ≤ 7` cap, a greedy fundamental-selection rule, and a mandatory cardinality-matched top-m control; H3's null and falsification criterion now reference the matched control.
4. **§3.7:** added the concrete grid artifact — Transformer seeds 0 and 2 report the *identical* crossing step 8367, MLP seeds 0 and 2 the identical 9646.
5. **§8:** budget re-anchored — the ~24-min reference run had early-stopped at ~8 400 of 40 000 steps; totals now assume the `1.5×`-crossing rule (~30–40 h).
6. **§2:** the 0.32 non-grokked baseline is now named (`txf_add_p113_wd1.0_frac0.3_gf2.0_seed0`, 4 000 steps, never generalized); the attention row's evidence corrected from 3 seeds to 2 runs.
7. **WP-0:** `training/runs/` is gitignored — the baseline tag preserves no run artifacts; archiving means copying.

Confirmed correct as written (re-derived or re-measured, no change needed): §3.1 mask counts (289/33/17 kept, 3 360 removed, 26.3%), §3.2 (the `max_k=8` cap binds on all 16 runs; multiplication ≈ 0.16), the §3.3 effective-weight formulas, §3.4 (means only, no variance, in `attention.py`), the §3.6 parameter counts (226 176 / 204 017 / `d_mlp=572` → 226 217), the H3 aliasing table, the §7 identities and the delta-function sum, WP-3's `W_out @ W_U` shapes and the no-LayerNorm claim, and the "23 checks" count in §11.
