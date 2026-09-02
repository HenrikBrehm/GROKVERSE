# ReLU as the multiplier in modular addition

Reusable mechanistic knowledge, derived and numerically verified on 2026-09-03. Source of truth:
`Claude/grokverse/docs/MLP_MECHANISM_DERIVATION.md` §4, pinned by
`Claude/grokverse/training/tests/test_derivations.py`. Nothing here comes from a trained model — it is
algebra plus synthetic checks.

## The problem the network has to solve

Computing `(a + b) mod p` with Fourier features needs the angle-addition identity

```
cos(w(a+b)) = cos(wa)cos(wb) - sin(wa)sin(wb)
```

which requires a **product** of an `a`-term and a `b`-term. Neither GROKVERSE architecture has a
multiplication. The MLP in particular is *exactly additive* in `(a, b)` up to the nonlinearity:
`pre[a,b,i] = u_a[a,i] + u_b[b,i] + b_in[i]`. So in that architecture the ReLU is provably the only
possible source of the product.

## How rectification produces the product

Write `u = wa - phi_a`, `v = wb - phi_b`. Using `ReLU(x) = (x + |x|)/2` and
`cos u + cos v = 2 cos(s) cos(t)` with `s = (u+v)/2`, `t = (u-v)/2`:

```
ReLU(cos u + cos v) = cos(s)cos(t) + |cos(s)| |cos(t)|
```

The Fourier series `|cos x| = 2/pi + (4/pi) * sum_m (-1)^(m+1) cos(2mx)/(4m^2 - 1)` gives the `cos(2s)`
coefficient `4/(3pi)` and the mean of `|cos t|` as `2/pi`, so the `(a+b)` term has amplitude

```
8 / (3 pi^2) = 0.27019
```

and, because `u + v = w(a+b) - (phi_a + phi_b)`, **phase `phi_a + phi_b`**. Measured at p=23, k=3:
amplitude 0.2698, phase error 2.1e-4.

## The trap: the (a-b) term is exactly as large

`|cos s| |cos t|` is **symmetric in s and t**. The `(a-b)` term therefore has the *same* amplitude
`8/(3 pi^2)` and phase `phi_a - phi_b`. Measured: 0.2698 (sum) vs 0.2705 (difference).

**A single rectified neuron does not prefer addition.** What selects `(a+b)` is the neuron *population*
together with the readout: if each neuron's output curve carries `phi_out = phi_a + phi_b`, the `(a+b)`
contributions all land at the same phase and add coherently, while the `(a-b)` contributions arrive at
`phi_a - phi_b`, which is unrelated to the readout, and cancel.

Verified with 400 synthetic neurons at one frequency (random phases, readout `cos(wc - (phi_a+phi_b))`):

| projection | R^2 of the centred logits |
|---|---|
| `cos(w(a+b-c))` | 0.938 |
| `cos(w(a-b-c))` | 0.003 |
| scrambled readout phase, on `cos(w(a+b-c))` | 0.113 |
| readout `phi_a - phi_b`, on `cos(w(a-b-c))` | 0.909 |

## Why this matters for measurement

A per-neuron test of "does the activation depend on `a+b` more than on `a-b`" is **not** a test of the
mechanism — an ideal circuit neuron scores about equal on both. The falsifiable per-neuron prediction is
the **phase relation** `phi_out ~= phi_a + phi_b`; sum-over-difference dominance must be tested on the
**logits** (against an `(a-b)` formula of matched flexibility) or on the population reconstruction.

This corrected a prediction that had already been written into the study's interface contract, before any
run was analysed. See [[Grokking measurement pitfalls]].

## Related

[[Fourier circuit for modular addition]] · [[Grokking measurement pitfalls]] ·
[[Transformer vs MLP mechanism comparison]]
