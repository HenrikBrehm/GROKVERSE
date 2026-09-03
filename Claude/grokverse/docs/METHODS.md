# Methods — source, method, our implementation

**AI-drafted (Claude), 2026-09-03 — not yet human-reviewed.** Master prompt §2: *for every method you
adopt, document the exact source, section, equation, or figure.* One row per method. Locations come from
the notes under `docs/sources/`; the note's own verification status is carried in the last column, so a
method resting on an unverified quote is visible as such.

**Verification legend.** `V` = the note was adversarially re-verified against the re-fetched source.
`V*` = partial (the verification pass was interrupted; some corrections landed). `U` = written from the
source but not independently re-checked. A row that depends on a `V*` or `U` note must be re-checked
before the final write-up.

**Status legend.** `planned` / `implemented` / `tested` (has automated checks that pass).

---

## 1. Training and reproduction

| method | source (id, location) | what the source does | our implementation | deviation | status | note |
|---|---|---|---|---|---|---|
| grokking, definition and phenomenon | Power et al. 2201.02177 | generalization long after memorization on small algorithmic datasets; weight decay and train fraction control it | the phenomenon we measure; crossings from dense evaluation | none | tested | V |
| 1-layer transformer recipe | Nanda 2301.05217 §2 | `p=113`, `d_model 128`, 4 heads, `d_mlp 512`, no LayerNorm, tokens `[a,b,=]`, read-out at `=`, full-batch AdamW `lr 1e-3`, `betas (0.9,0.98)`, `wd 1.0`, `train_frac 0.3` | `models/transformer.py`, `config.py` preset `arch25k` — the same values | fixed 25,000-step budget instead of their stopping rule; **ours** | tested | V* |
| 2-layer MLP with shared embedding | — | no source: the comparison architecture of this study | `models/mlp.py`: shared `W_E [p,d]`, operand concatenation, one ReLU layer | own design; McCracken 2505.18266 §3 confirms concatenation MLPs are a studied family | tested | U |
| two-hot MLP control | Swaroop 2603.23784 §2 | two-hot `2p` → 256 ReLU → `p` at `p=97`, split stratified by `c`, early stopping | `models/mlp_twohot.py` at `d_mlp 512`, `p=113`, our split and budget | **own variant**: only the input parametrization differs from our MLP | tested | V |
| Grokfast acceleration | Lee et al. 2405.20233 | EMA of the gradient amplified by `lambda`; `alpha`, `lambda` | `train.py: apply_grokfast`, `alpha 0.98`, `lambda 2.0` | used only in the confound block, never in the primary setting | tested | V |
| deterministic seeding | — | — | `seed.py`; split from a seeded permutation; `split_hash` asserted equal across paired seeds | own | tested | — |

## 2. Transition measurement

| method | source | what the source does | our implementation | deviation | status | note |
|---|---|---|---|---|---|---|
| dense accuracy evaluation | master prompt §15 | train every 10 steps, test every 25 | `train.py`, under `no_grad`; asserted not to perturb the parameter trajectory | none | tested | — |
| crossings as intervals | master prompt §15 | report the previous evaluated step and the evaluation frequency; never "exact" | `detect_transition_dense`, three threshold sets | none | tested | — |
| at-grok vs converged | Khanh 2607.06639 | metrics read at the transition overstate the converged value by 3–5× (MLP) and 1.3–1.5× (transformer); compression lag of order 10⁴ steps | every structure metric reported at the crossing **and** at the final step; a 5 % change test between steps 20,000 and 25,000 decides whether "converged" may be said | the convergence tolerance is **ours** | planned | V |

## 3. Fourier analysis

| method | source | what the source does | our implementation | deviation | status | note |
|---|---|---|---|---|---|---|
| real Fourier basis over `Z_p` | Nanda 2301.05217 §3; released Colab | row 0 constant, then `(cos_k, sin_k)` pairs, normalized | `fourier.fourier_basis`; orthonormality asserted | none | tested | V* |
| 2D transform of the logits over `(a,b)` | Nanda §5.1; Colab `fft2d` | `einsum('xyz,fx,Fy->fFz')` over the two input axes | `progress_measures.fwd2d` / `inv2d`; round trip asserted to 1e-9 | none | tested | V* |
| embedding power spectrum, top-k concentration | Nanda §4.1 (6 non-negligible frequencies, 5 used) | power per frequency of `W_E` | `fourier.embedding_power_spectrum`, `metrics.topk_concentration` | our count was **capped at 8** and the cap binds on all 16 legacy runs — reported, and superseded as the primary rule | tested | V* |
| key frequencies | Nanda App. C.2 | DFT of the neuron→logit map `W_L = W_U W_out` along the logit axis, norm over neurons, keep the non-trivial ones; **count measured**, threshold not published | `key_frequencies.select(rule="nanda")`, threshold **0.25 × max** with sensitivity 0.10 / 0.50 | the numeric threshold is **ours**, declared in advance | implemented | V* |
| restricted / excluded loss | Nanda §5.1; Colab `get_component_cos_xpy`, `trig_loss`, `excl_loss`; blog | project onto `cos/sin(w_k(a+b))` only; restricted re-adds the constant; **excluded measured on the training pairs**; the restricted split is not stated | `mask_protocols`: `nanda_exact_*`, `full_grid_extension_*`, `legacy_broad_mask_variant`; all three splits reported for restricted | the legacy outer-product mask is an **error**, not a deviation (`docs/MASK_PROTOCOL_AUDIT.md`) | implemented | V* |
| harmonic families and aliasing | textbook Fourier series | a square wave has odd harmonics with amplitudes `1/j`; `alias(jk) = min(jk mod p, p − jk mod p)` | `fourier.harmonic_family`, `metrics.harmonic_shares`, `discrete_square_reference` | **not attributed to any consulted paper** — derived in `docs/MLP_MECHANISM_DERIVATION.md` §5 and verified at `p=113` | tested | — |
| odd-vs-even harmonic shape | — | — | `metrics.harmonic_shares`, per curve on its **own** fundamental (collision-free at prime `p`) | own; supersedes the set-based `fourier.harmonic_shape`, which is corrupted by in-set collisions (even share 0.083 against a true 0.0005) | tested | — |

## 4. Mechanism

| method | source | what the source does | our implementation | deviation | status | note |
|---|---|---|---|---|---|---|
| trig identities and the logit formula | Nanda §4 | `Logit(c) = Σ_k α_k cos(ω_k(a+b−c))`; the readout is an inner product of Fourier features | `docs/MLP_MECHANISM_DERIVATION.md` §3; fitted end-to-end by `logit_formula_fit` | none | tested | V* |
| effective per-neuron MLP curves | — (the repo's own audit) | — | `mlp_mechanism.effective_curves`: `u_a = W_E @ W_in[:d]`, `u_b = W_E @ W_in[d:]`, `out = W_out.T`; reproduces the forward pass to 1.1e-7 | own; the point is that `W_E` alone is the wrong object for this MLP | tested | — |
| ReLU as the multiplier; cross-term amplitude and phase | derived here | — | `docs/MLP_MECHANISM_DERIVATION.md` §4: amplitude `8/(3π²)`, phase `φ_a + φ_b`; **and §4.1: the `(a−b)` term is equally large**, so addition is selected by the population and readout, not by one neuron | own derivation, numerically verified | tested | — |
| phase-sum relation `φ_out ≈ φ_a + φ_b` | Swaroop 2603.23784 §1, §3.1 (attributing it to Nanda 2023 and Gromov 2023) | per neuron, three length-`p` DFTs, dominant non-DC bin, phase of that bin; agreement by circular correlation | `mlp_mechanism.phase_relation`: resultant length with bootstrap CI against a **permutation null** | we use a permutation null and resultant length; the source uses a circular correlation whose definition it does not give | tested | V |
| periodicity score | Swaroop 2603.23784 Eq. 1 | `max_k |ŵ_k| / mean_k |ŵ_k|`, cuts > 12 structured, < 5 unstructured, chosen post hoc from a bimodal histogram | `metrics.periodicity_score`; his cuts exposed as a labelled constant | used only as a **sensitivity** variant, never primary; the post-hoc origin of his cuts is stated | tested | V |
| inverse participation ratio | Doshi 2310.13061 Eq. 3 (with footnote 4, `r=2`) and Eq. 4 | `IPR(x) = Σ_j P_j² / (Σ_j P_j)²` on the DFT power; per neuron the mean over the `a`, `b` and output curves; used as a **ranking**, no threshold | `metrics.inverse_participation_ratio`, with `_ours` exposed separately; `IPR_DEFINITION_STATUS` records the provenance | we adopt the ranking, not a threshold, as the source does | tested | U |
| transformer neuron→logit map | derived here from the architecture | — | `W_out @ W_U`; exact because there is **no LayerNorm**; verified to 2.2e-16 | own derivation (`docs/TRANSFORMER_MECHANISM_DERIVATION.md` §3) | tested | — |
| transformer effective operand curves | derived here | — | mean-attention construction with a measured `additivity_r2`; exact when attention is uniform (4.4e-16) | own; the approximation quality is **measured, never assumed** | tested | — |
| attention analysis | RESEARCH_SPEC §3.4 audit | the legacy analysis reported a mean without a variance | per head: mean, std, quantiles, dependence on `a`, `b`, `(a+b)`; causal head ablation; `fix_attention_to_mean` | own | planned | — |

## 5. Causal tests

| method | source | what the source does | our implementation | deviation | status | note |
|---|---|---|---|---|---|---|
| neuron pruning sweep | Doshi 2310.13061 §2.1, Fig. 6; App. I | rank hidden neurons by IPR, prune cumulatively in both directions on an unmodified checkpoint, measure train/test loss and accuracy at every pruning size | `causal_ablation.ipr_ranked_pruning` | **we add the size-matched random control the paper does not have** | planned | U |
| structured / key-frequency ablations | master prompt §11 | necessity and sufficiency, size-matched random controls | `causal_ablation_mlp.py`, `causal_ablation_transformer.py`; tables in `docs/CAUSAL_ABLATION_PLAN.md` | own | planned | — |
| restricted circuit as a sufficiency test | Nanda §5.1 | restricted loss | reported as `restricted_circuit_only` in the ablation table as well as a progress measure | own framing | planned | V* |

## 6. Comparison and statistics

| method | source | what the source does | our implementation | deviation | status | note |
|---|---|---|---|---|---|---|
| parameter-matched control | RESEARCH_SPEC §3.6 audit | — | `d_mlp = 572` → 226,217 parameters, +0.02 % of the transformer's | own | tested | — |
| paired-by-seed statistics | master prompt §16 | paired differences, bootstrap CIs, effect sizes, all seeds shown | `statistics.py`: percentile bootstrap (10,000, seed 0), exact sign test, exact Wilcoxon, Cohen's `d_z`, Cliff's delta | own; p-values never sole evidence | tested | — |
| full-domain function comparison | master prompt §10 | agreement over all `p²` inputs | `function_agreement.py` | own | tested | — |
| end-to-end logit formula fit | master prompt §9 | fit candidate formulas to the actual logits | `logit_formula_fit.py`, with a matched-flexibility control and an `(a−b)` control | own | tested | — |

## 7. Fairness of metrics across architectures (master prompt §17)

| object | MLP | transformer | comparable? |
|---|---|---|---|
| effective `a` / `b` curves | exact (`W_E @ W_in` halves) | mean-attention approximation, `additivity_r2` reported | **yes**, with the bound attached |
| output curve | `W_out[i,:]` | `(W_out @ W_U)[f,:]` | **yes**, both exact |
| hidden activation over `(a,b)` | `ReLU(u_a+u_b+b_in)` | `ReLU(r_pre @ W_in)` | **yes** |
| per-neuron logit contribution | `act · W_out[i,c]` | `hidden · (W_out W_U)[f,c]` | **yes** |
| embedding `W_E` | `[p,d]` | `[p+1,d]`, `=` row excluded | yes, but decides nothing for either |
| spectral entropy, participation ratio, IPR, top-k, harmonic shape | on the objects above | same | **yes**, same definitions |
| `W_pos`, `W_Q/K/V/O`, residual stream, heads, direct logit path | absent | present | **no** — architecture-specific, excluded from the headline |
| `additivity_r2` | ≡ 1 by construction | measured | **no** — reported as a bound, not as a comparison |

## 8. What is not yet done

`planned` rows above are the honest state on 2026-09-03: the mechanism modules for the transformer, the
attention program, and every causal ablation exist as specifications and tests-to-be, not as measured
results. Rows resting on `V*` or `U` notes — Nanda's key-frequency rule and restricted-loss split, and
Doshi's IPR object — must be re-checked against the primary before the final write-up; the interrupted
verification pass is recorded in `docs/LABBOOK.md`.
