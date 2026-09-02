# Derivation — how a transformer MLP neuron reaches the logits, and what is comparable to the MLP

AI-drafted (Claude), 2026-09-02/03 — not yet human-reviewed. Mathematics and code structure only; it
contains **no measurement of any trained model**. Every algebraic step was verified numerically on random
models (`p = 23`, `d_model = 32`, 2 heads, seed 0) — output in §7.

Companion: `docs/MLP_MECHANISM_DERIVATION.md`, `docs/dev/INTERFACES.md` §6.

The point of this document is the master prompt's §8 requirement: *the transformer must undergo the same
mechanism tests as the MLP.* It is not permissible to assume the Fourier circuit for the transformer and
test only the MLP. To do that, the transformer's objects must first be put on the same footing as the
MLP's `u_a`, `u_b`, `out` — which is what §3 and §4 construct, and §5 measures the cost of.

---

## 1. The model, and why every map below is exact

`training/grokverse/models/transformer.py`, `OneLayerTransformer`: token + learned positional embeddings,
one multi-head attention layer, one ReLU MLP, unembedding, prediction read at the `=` position (index 2).
`p = 113`, `d_model = 128`, 4 heads × `d_head` 32, `d_mlp = 512`, `n_ctx = 3`.

**There is no LayerNorm** (`use_ln` exists in the config but the module never applies one). Every
composition below is therefore exactly linear where it looks linear — no normalization constant depends on
the input. This is what makes the neuron→logit map a fixed matrix rather than an input-dependent one.
State it; do not silently rely on it.

## 2. Forward decomposition

Write `t = (a, b, =)` for the token triple, `s ∈ {0, 1, 2}` for positions.

```
x_s      = W_E[t_s] + W_pos[s]                             embedding at position s
A[h,s]   = softmax_s( (x_2 W_Q^h)·(x_s W_K^h)^T / sqrt(d_head) )      attention FROM '=' (row 2)
z_h      = Σ_s A[h,s] · (x_s W_V^h)                         head h output at the read-out position
r_pre    = x_2 + Σ_h z_h W_O^h                              residual stream before the MLP
hidden_f = ReLU( r_pre · W_in[:, f] )                       hidden activation of neuron f
r_post   = r_pre + Σ_f hidden_f · W_out[f, :]
logits   = r_post · W_U
```

Causal masking makes rows 0 and 1 irrelevant to the read-out; only row 2 of the attention pattern enters.

**Verified:** this decomposition reproduces `model.logits_last` to `2.7e-7` (float32 model evaluated in
float64).

## 3. The neuron→logit map — the transformer's counterpart of `W_out[i, c]`

Because there is no LayerNorm, the path from a hidden activation to the logits is a fixed linear map:

```
neuron_logit_map = W_out @ W_U          [d_mlp, vocab];  restrict to the p number classes
```

Neuron `f` contributes `hidden_f · neuron_logit_map[f, c]` to the logit of class `c`. The logits split
into two additive paths:

```
logits = r_pre @ W_U           (direct path: embeddings + attention, no MLP)
       + hidden @ (W_out @ W_U)  (MLP path)
```

**Verified:** `direct + mlp = logits` to `2.7e-7`, and `Σ_f contribution_f = mlp path` to `2.2e-16`.

So `neuron_logit_map[f, :]` **is** the transformer's output curve, exactly comparable to the MLP's
`out[:, i] = W_out[i, :]`. Both are length-`p` curves over the class index; both are the object whose phase
H1 predicts. The `direct_path_share` of logit variance is reported alongside, because the MLP has no
counterpart to it and a large direct share limits how much of the transformer's behaviour the neuron-level
analysis can explain.

## 4. Effective operand curves — the counterpart of `u_a`, `u_b`

The MLP's pre-activation is exactly `u_a[a] + u_b[b] + β`. The transformer's is not, because `A[h,s]`
depends on the input. Define the **mean-attention** effective curves using the grid-averaged attention
`Ā[h,s] = mean over all p² inputs of A[h,s]`, and the OV circuit `OV_h = W_V^h W_O^h` (`[d, d]`):

```
u_a[n, f] = Σ_h Ā[h,0] · ((W_E[n] + W_pos[0]) OV_h) · W_in[:, f]
u_b[n, f] = Σ_h Ā[h,1] · ((W_E[n] + W_pos[1]) OV_h) · W_in[:, f]
c[f]      = (W_E[=] + W_pos[2]) · W_in[:, f]                      (residual skip at the read-out)
          + Σ_h Ā[h,2] · ((W_E[=] + W_pos[2]) OV_h) · W_in[:, f]  (the '=' token via attention)
```

giving the approximation `r_pre · W_in[:, f] ≈ u_a[a, f] + u_b[b, f] + c[f]`, of exactly the MLP's form.

### 4.1 The approximation is not free — measure it, never assume it

Define, per neuron, the **additivity R²** of the true pre-activation against that additive reconstruction
over the full grid:

```
additivity_r2[f] = 1 − Σ_{a,b} (true_pre[a,b,f] − (u_a[a,f] + u_b[b,f] + c[f]))²
                       / Σ_{a,b} (true_pre[a,b,f] − mean)²
```

`1 − additivity_r2` is exactly the share of the pre-activation carried by **input-dependent attention** —
the quantity `docs/RESEARCH_SPEC.md` §3.4 asks for and the earlier attention analysis (a mean without a
variance) could not provide. Two readings, and they are different circuits:

* `additivity_r2 ≈ 1` — attention is effectively a fixed weighted sum. The transformer is then
  *additive-then-ReLU*, structurally like the MLP, and the two architectures are directly comparable on
  `u_a`, `u_b`, `out`.
* `additivity_r2` well below 1 — attention gives the transformer a multiplicative path the MLP does not
  have. That is a real architectural asymmetry, a candidate explanation for a timing difference, and it
  bounds how much of the transformer the effective-curve analysis can explain. In that case the
  transformer's `hidden` activations are analysed directly (they always are) and the effective-curve
  comparison is reported with the bound attached.

**By construction the MLP has `additivity_r2 ≡ 1`.** The comparison is therefore never "which architecture
is more additive" — it is "how much of the transformer's pre-activation is captured by the object we
compare".

**Verified — two limiting cases.** With `W_Q = W_K = 0` the attention row is exactly uniform (`1/3, 1/3,
1/3`) and the reconstruction is **exact** (max absolute error `4.4e-16`), confirming the algebra. On a
random initialized model the median `additivity_r2` is `0.9974` (min `0.9867`) — at initialization
attention is near-uniform, so this is a check of the formula, **not** a prediction for trained models,
where `Ā` may be far from uniform and input-dependent. The per-run value is measured, never assumed.

## 5. Attention, measured properly

The legacy analysis reported the mean of `A[h, 2, s]` over all 12 769 inputs and read "≈ 50/50 to both
operands" as evidence for the addition circuit. A 50/50 *mean* is consistent with a constant 50/50 split
**and** with strongly input-dependent attention that averages to 50/50. The mechanism analysis therefore
reports, per head:

* mean, standard deviation, min, max and the 5/95 quantiles of `A[h,2,s]` over all inputs;
* variance of `A[h,2,0]` explained by `a`, by `b`, and by `(a+b) mod p`;
* the same over training checkpoints, so a claim about *when* attention settles is dated;
* **causal head ablation**: zero-ablate and mean-ablate each head (replace `z_h` by 0 or by its grid mean),
  measuring train/test loss and accuracy;
* **`fix_attention_to_mean`**: replace `A` by `Ā` everywhere. If loss barely changes, attention *is* a
  fixed sum, and the `additivity_r2 ≈ 1` reading is causally confirmed rather than merely fitted.

A symmetric attention distribution alone remains a **descriptive** finding until one of these ablations
moves the loss.

## 6. The same tests as the MLP, and where they stop being comparable

Applied to `u_a`, `u_b` and `neuron_logit_map` with **the same functions** as the MLP (INTERFACES §5/§6:
`neuron_tables`, `structured_neuron_definitions`, `phase_relation`, `activation_analysis` on the true
`hidden`, `logit_contributions`, and the wave-model comparison of `docs/MLP_MECHANISM_DERIVATION.md` §5):
per-curve spectra, dominant frequency agreement, family/top-1 fractions, harmonic shares, the phase-sum
relation, sinusoid-vs-square-vs-odd-harmonic model comparison.

Transformer-only objects — `W_pos`, `W_Q/W_K/W_V/W_O`, the residual stream, per-head OV spectra, the
direct path — are analysed and reported, but **excluded from the headline architecture comparison**,
because the MLP has nothing they could be compared against (master prompt §17 fairness constraint). The
comparable set is fixed in advance:

| object | MLP | transformer | comparable? |
|---|---|---|---|
| effective `a`/`b` curve | `W_E @ W_in[:d]` / `[d:]` | §4 mean-attention curves | yes, with `additivity_r2` attached |
| output curve | `W_out[i, :]` | `(W_out @ W_U)[f, :]` | yes, both exact |
| hidden activation over `(a,b)` | `ReLU(u_a + u_b + b_in)` | `ReLU(r_pre @ W_in)` | yes |
| per-neuron logit contribution | `act · W_out[i, c]` | `hidden · (W_out W_U)[f, c]` | yes |
| embedding `W_E` | `[p, d]` | `[p+1, d]` (the `=` row excluded) | yes, but decides nothing for either |
| attention / heads / residual / `W_pos` | — | present | no — architecture-specific |
| direct (non-MLP) logit path | — | present | no — reported as a bound |

## 7. Verification output

`C:/Users/henri/.claude/jobs/955bf44e/tmp/verify_derivation.py` (mirrored as
`training/tests/test_derivations.py`), project venv, `p = 23`, `d_model = 32`, 2 heads, `d_mlp = 24`:

```
max |decomposition - model|         : 2.71e-07
max |direct+mlp - logits|           : 2.71e-07
max |sum_f contrib_f - mlp path|    : 2.22e-16
additivity_r2 (random model)        : median 0.9974  min 0.9867
uniform attention row (W_Q=W_K=0)   : [0.3333, 0.3333, 0.3333]
max |approx - true| (W_Q=W_K=0)     : 4.44e-16
```

## 8. What this licenses

Licensed: treating `W_out @ W_U` as the transformer's output curve; running the MLP's mechanism battery on
the mean-attention effective curves **with** the additivity bound reported; and the attention program of
§5.

Not licensed: assuming the transformer implements the Fourier circuit because Nanda et al. found one in a
similar model. This study's transformer is tested with the same gate as the MLP — periodic structure,
phase addition, end-to-end logit fit, causal necessity and sufficiency — and the gate is evaluated per
architecture before any cross-architecture interpretation (`docs/PREREGISTRATION.md`).
