# Derivation — what a hidden neuron of the MLP computes, and what that predicts

AI-drafted (Claude), 2026-09-02/03 — not yet human-reviewed. Mathematics and code structure only; it
contains **no measurement of any trained model**. Every algebraic step below was verified numerically on
random models (`p = 23`, seed 0) — the check script and its output are reproduced in §7.

Companion: `docs/TRANSFORMER_MECHANISM_DERIVATION.md` (same objects for the transformer),
`docs/dev/INTERFACES.md` §5 (the module this derivation specifies).

---

## 1. The model

`training/grokverse/models/mlp.py`, `TwoLayerMLP`, with `p = 113`, `d_model = 128`, `d_mlp = 512`:

```
x      = concat(W_E[a], W_E[b])            in R^{2d}      W_E  [p, d]   (shared table)
h      = ReLU(x @ W_in + b_in)             in R^{d_mlp}   W_in [2d, d_mlp]
logits = h @ W_out + b_out                 in R^{p}       W_out[d_mlp, p]
```

The two-hot control (`models/mlp_twohot.py`) is the same with `W_E = I_p` and `d = p`.

## 2. Effective per-neuron curves

Split `W_in` into its `a`-half and `b`-half. For hidden neuron `i`, define three length-`p` curves:

```
u_a[n, i] = W_E[n] · W_in[:d, i]        the a-curve   ("what neuron i sees when a = n")
u_b[n, i] = W_E[n] · W_in[d:, i]        the b-curve
out[c, i] = W_out[i, c]                 the output curve over classes c
```

Then, exactly (no approximation):

```
pre[a, b, i] = u_a[a, i] + u_b[b, i] + b_in[i]
act[a, b, i] = ReLU(pre[a, b, i])
logits[a, b, c] = Σ_i act[a, b, i] · out[c, i] + b_out[c]
```

**Verified:** reconstructing the forward pass from `u_a`, `u_b`, `b_in`, `W_out`, `b_out` reproduces
`model.logits_last` to `1.1e-7` (float32 model evaluated in float64; the residual is float32 round-off).

Two consequences that decide how this architecture must be analysed:

1. **`W_E` alone is the wrong object.** `W_E` is a shared table read through two different halves of
   `W_in`; `W_in` is free to select and rotate directions. A neuron's circuit can be perfectly periodic
   while the raw `W_E` spectrum looks diffuse, and vice versa. Any structural claim about the MLP must be
   made on `u_a`, `u_b`, `out` (or on the activations), not on `W_E`.
2. **The MLP is exactly additive in `(a, b)` up to the ReLU.** No path lets `a` and `b` interact before the
   nonlinearity. In this architecture the ReLU is therefore *provably the only possible source* of the
   multiplication that the angle-addition identity needs. That makes the MLP the cleaner test case of the
   mechanism, not the messier one.

## 3. Why a Fourier representation solves `(a + b) mod p` at all

Pure mathematics, no network involved. For frequency `k` write `ω_k = 2πk/p`.

**(1) Represent.** The map `n ↦ (cos ω_k n, sin ω_k n)` is automatically periodic mod `p`, because
`ω_k(n + p) = ω_k n + 2πk`. The modular reduction is not computed; it is built into the representation.

**(2) Combine.** The angle-addition identities

```
cos(ω_k(a+b)) = cos ω_k a · cos ω_k b − sin ω_k a · sin ω_k b
sin(ω_k(a+b)) = sin ω_k a · cos ω_k b + cos ω_k a · sin ω_k b
```

turn addition of numbers into a bilinear operation on features. Note what this requires: a **product** of
an `a`-term and a `b`-term. A purely linear network cannot do it.

**(3) Read out.** With `Logit(c) = Σ_{k ∈ K} α_k cos(ω_k(a + b − c))`, expanding the cosine shows this is an
inner product between the Fourier features of `a+b` and those of `c` — what an unembedding whose rows are
Fourier features computes. Every cosine peaks simultaneously iff `c ≡ a + b (mod p)`; elsewhere the phases
disagree and partially cancel. In the limit of all frequencies with equal weights the readout is exactly a
delta function, since

```
Σ_{k=1}^{(p−1)/2} cos(2πkm/p) = (p·1[m ≡ 0 mod p] − 1) / 2
```

so a sparse `K` gives an approximation that is still argmax-correct with a margin — which is why a handful
of frequencies suffices.

**Provenance:** this argument is Nanda et al. 2023 (arXiv:2301.05217) §4; see
`docs/sources/nanda2023_progress_measures.md`. It is reproduced here because the study needs its
*predictions*, not because it is new.

## 4. Where the product comes from: the ReLU is the multiplier

Step (2) needs a multiplication; the MLP has none. What it has is `ReLU(u_a(a) + u_b(b) + β)` — an additive
pre-activation followed by a nonlinearity. Rectifying a sum of two same-frequency sinusoids produces a
cross term in `(a + b)`.

**Derivation.** Let `u = ω_k a − φ_a`, `v = ω_k b − φ_b`, and take the idealized neuron
`ReLU(cos u + cos v)` (unit amplitudes, zero bias). Using `ReLU(x) = (x + |x|)/2` and
`cos u + cos v = 2 cos s · cos t` with `s = (u+v)/2`, `t = (u−v)/2`:

```
ReLU(cos u + cos v) = cos s · cos t + |cos s| · |cos t|
```

The first term is `(cos u + cos v)/2` and has no `2s = u+v` component. In the second, the Fourier series of
the rectified cosine is

```
|cos θ| = 2/π + (4/π) Σ_{m≥1} (−1)^{m+1} cos(2mθ) / (4m² − 1)
```

so the `cos 2s` coefficient of `|cos s|` is `4/(3π)`, and the mean of `|cos t|` over the grid is `2/π`.
For frequencies that equidistribute `s` and `t` over the `(a, b)` grid the product's `cos(u+v)` coefficient
is therefore

```
A_cross = (4/(3π)) · (2/π) = 8/(3π²) ≈ 0.27019
```

and, since `u + v = ω_k(a + b) − (φ_a + φ_b)`, the cross term is
`A_cross · cos(ω_k(a+b) − (φ_a + φ_b))`.

**Verified** at `p = 23`, `k = 3`: for `(φ_a, φ_b) = (0.7, −1.3)` the measured cross-term phase is `−0.6002`
against `φ_a + φ_b = −0.6000` (wrapped difference `2.1e-4`, the discrete-sampling residual) and the measured
amplitude `0.2698` against `8/(3π²) = 0.2702`; likewise for `(0, 0)`, `(2.5, 2.5)` and `(1.1, 0.3)`.

### 4.1 The `(a−b)` term is exactly as large — the readout, not the neuron, selects `(a+b)`

`|cos s|·|cos t|` is **symmetric in `s` and `t`**. The coefficient of `cos 2t = cos(u−v)` is therefore the
same `8/(3π²)`, with phase `φ_a − φ_b`. A single rectified neuron produces an `(a+b)` component and an
`(a−b)` component of **equal amplitude**; it does not by itself prefer addition.

**Verified** at `p = 23`, `k = 3`: sum amplitude `0.2698` / difference amplitude `0.2705` for
`(0.7, −1.3)`; the two agree to the discretization residual for every phase pair tested, and the measured
difference-term phase equals `φ_a − φ_b` in every case.

What selects `(a + b)` is the **population together with the readout**. If each neuron's output curve
carries `φ_out = φ_a + φ_b`, then across neurons the `(a+b)` contributions all land at the same phase and
add coherently, while the `(a−b)` contributions arrive at phase `φ_a − φ_b`, which is unrelated to the
readout phase, and cancel.

**Verified** — 400 idealized neurons at one frequency with random `(φ_a, φ_b)` and readout
`cos(ω_k c − (φ_a + φ_b))`, logits centred over classes: projecting on `cos(ω_k(a+b−c))` gives coefficient
`54.03` and `R² = 0.938`; projecting on `cos(ω_k(a−b−c))` gives `−2.94` and `R² = 0.003`. (Argmax accuracy
is 0.456 at a single frequency — several frequencies are needed for a correct readout, as §3(3) says.)

**Consequence for the analysis, fixed before any run was analysed.** The sum-vs-difference contrast is a
prediction about the **logits and the neuron population**, not about one neuron's activation map. An
individual circuit neuron is expected to show `sum_direction_share ≈ diff_direction_share` in its own 2D
activation spectrum, and that is *not* evidence against the mechanism. The falsifiable per-neuron
prediction is the **phase relation** below; the sum-over-difference dominance is tested on the logit
tensor (`analysis/logit_formula_fit`, where `control_difference` is exactly the `(a−b)` formula) and on
the population reconstruction. `docs/dev/INTERFACES.md` §5 was corrected accordingly on 2026-09-03.

### 4.2 The per-neuron prediction

For the readout to peak at `c = a + b`, the output curve must carry the sum phase. Hence the per-neuron,
falsifiable prediction the whole empirical study turns on:

> **H1 (per neuron):** if the `a`-curve has frequency `k` and phase `φ_a`, and the `b`-curve has the same
> frequency `k` and phase `φ_b`, then the output curve has frequency `k` and phase
> `φ_out ≈ φ_a + φ_b (mod 2π)`.

The null is the permutation null: shuffle neuron identity between the operand side and the output side,
leaving both marginal phase distributions intact (`analysis/mlp_mechanism.phase_relation`).

**Provenance.** The phase-sum relation for ReLU MLPs is *not* new: Swaroop (arXiv:2603.23784, §1, §3.1)
reports it and attributes it in turn to Nanda et al. 2023 and Gromov 2023. GROKVERSE's H1 is a replication
in this architecture and training setting. What is not in any fetched source is the relation for a
**shared-embedding concatenation** MLP (Swaroop's model has a two-hot input and no embedding).

### 4.3 What the derivation does *not* say

* It gives no reason for the amplitudes or the bias `β`; with `β ≠ 0` and unequal amplitudes the cross-term
  coefficient changes (the phase does not). The measured statistic is therefore the **phase**, and
  amplitudes are reported descriptively.
* It says nothing about whether the trained network actually uses this route. That is the causal question
  (`docs/CAUSAL_ABLATION_PLAN.md`), not an algebraic one.
* Two curves at *different* dominant frequencies produce no `(a+b)` term at either frequency; such neurons
  are excluded by the structured-neuron definition rather than being evidence against it.

## 5. Square waves and odd harmonics

If a neuron's effective curve is not a sinusoid but a **square wave** at fundamental `k`, its Fourier
series contains the odd harmonics only, with amplitudes decaying as `1/j`:

```
sign(cos θ) = (4/π) Σ_{j odd} (−1)^{(j−1)/2} cos(jθ) / j        power ∝ 1/j²
```

so the power at `3k, 5k, 7k` relative to the fundamental is `1/9 + 1/25 + 1/49 = 0.1715`.

**This is a textbook Fourier fact, not a claim of any source consulted.** Swaroop reports square-wave-like
*input weights*; the paper never mentions harmonics or a `1/j` law. Attributing the harmonic prediction to
that paper would be wrong; it is derived here.

**Two properties of the discrete case matter for the metric** (verified at `p = 113`, for `k = 1, 5, 18`,
`sign(cos(2πkn/p))` sampled at `n = 0..p−1`):

| quantity | measured | continuous ideal |
|---|---|---|
| fundamental / total non-constant power | 0.8107 | 8/π² = 0.8106 |
| odd-harmonic (j = 3,5,7) power / fundamental | 0.1717 | 0.1715 |
| even-harmonic (j = 2,4,6) power / fundamental | 5.8e-4 | 0 |

so a discrete square wave carries a small but nonzero even-harmonic power — the reference against which a
measured `even_share` must be read (`analysis/metrics.discrete_square_reference`).

**Aliasing.** In `Z_p` the harmonic `jk` folds to `alias(jk) = min(jk mod p, p − (jk mod p))`. For a
*single* fundamental at prime `p` the aliased odd harmonics `3k, 5k, 7k` are distinct from each other and
from `k` (verified for every `k` at `p = 113`). Across a *set* of fundamentals they can collide — e.g. in
the legacy top-8 set `{18,15,11,1,13,22,56,36}`, `2·18 → 36`, `7·18 → 13`, `2·11 → 22`, `7·11 → 36`,
`7·13 → 22`, `2·56 → 1` all land on another member of the set. Any harmonic-shape statistic defined on a
*set* must therefore state how collisions are handled; the per-neuron statistic used here is defined on
each curve's **own** fundamental, where the collision count is zero.

**This is the mechanism behind H3:** a top-`k` metric counting *individual* frequencies scores a clean
square-wave circuit as unstructured, because a single square-wave neuron spreads its power over
`k, 3k, 5k, 7k, …` which alias to scattered indices.

## 6. Activation-level predictions

For a neuron implementing the circuit, `act[a, b]` is (to first order) a function of `a + b` alone. The
measurable consequences, all computed on the full `(a, b)` grid:

| quantity | prediction for a circuit neuron | control |
|---|---|---|
| 2D spectrum of `act` | mass in same-frequency `(cos_k, sin_k)²` blocks | cross-frequency blocks near zero |
| sum vs difference directions **of one neuron** | **approximately equal** (§4.1) — this is *not* a discriminating test | — |
| `Var(E[act \| a+b]) / Var(act)` | **not** near 1 for a single neuron, because the `(a−b)` term is equally large; reported descriptively next to the same quantity for `a − b` | the `a − b` value |
| swap symmetry `corr(act[a,b], act[b,a])` | 1 iff `φ_a = φ_b`; below 1 otherwise | — |
| `ev_hypothesized` | R² of `act` against `ReLU(fit_a + fit_b + b_in)` near 1 | fit from the wave models of §5 |
| sum vs difference **in the logits / population** | `cos/sin(ω_k(a+b))` share ≫ `cos/sin(ω_k(a−b))` share | `control_difference` in `analysis/logit_formula_fit` |

The row that carries the inferential load at the neuron level is the **phase relation** of §4.2, not the
sum-vs-difference contrast; the latter is tested where it is actually predicted, on the logits.

## 7. Verification script and output

`C:/Users/henri/.claude/jobs/955bf44e/tmp/verify_derivation.py` (also reproduced as
`training/tests/test_derivations.py`), run with the project venv:

```
max |decomposition - model|      : 2.7e-07     (transformer, see the companion doc)
max |effective-curve MLP - model|: 1.1e-07
ReLU cross-term phase -0.6002 vs phi_a+phi_b -0.6000 (wrapped diff 2.1e-04)
cross-term amplitude sqrt(c^2+s^2) = 0.2698 (theory 8/(3 pi^2) = 0.2702)
square k=  1: fund/total 0.8107  odd/fund 0.1717 (ideal 0.1715)  even/fund 5.8e-04
square k=  5: fund/total 0.8107  odd/fund 0.1717 (ideal 0.1715)  even/fund 5.8e-04
square k= 18: fund/total 0.8107  odd/fund 0.1717 (ideal 0.1715)  even/fund 5.8e-04
```

## 8. What this derivation licenses, and what it does not

Licensed: the *predictions* tested in `docs/PREREGISTRATION.md` (H1 phase sum, H2 waveform, H3 aliasing
mechanism, the activation battery), and the claim that for this architecture the ReLU is the only possible
multiplier.

Not licensed: any statement that the trained MLP *does* implement this. Structure present in the weights is
descriptive; phases adding is mechanistic; only ablation (necessity **and** sufficiency, against a
size-matched random control) licenses the word *algorithm*. That is the evidence gate.
