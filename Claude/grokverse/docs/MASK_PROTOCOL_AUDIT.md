# Mask protocol audit — restricted and excluded loss

**AI-drafted (Claude), 2026-09-03 — not yet human-reviewed.** Implements master prompt §6. Every
statement about the published protocol comes from `docs/sources/nanda2023_progress_measures.md`, which
quotes Nanda et al. 2023 (arXiv:2301.05217), the authors' released Colab notebook and the
TransformerLens demo verbatim with locations. Statements about *our* code were checked against
`training/grokverse/analysis/mask_protocols.py` and `progress_measures.py`.

**Headline.** The mask GROKVERSE has been using (`legacy_broad_mask`) is **not** the published operator
and never was. It is treated here as an **implementation error**, not a documented deviation, because
the repository documents a *different* deviation (full-grid evaluation) and says nothing about the mask
width. Every restricted/excluded number currently in `RESULTS.md` is therefore labelled
**"not directly comparable to Nanda et al."** until the corrected protocols are measured.

---

## 1. The nine questions of master prompt §6.2

| # | question | published protocol | `legacy_broad_mask` | `same_frequency_block` | `sum_directions_only` |
|---|---|---|---|---|---|
| 1 | which axes are transformed | the two **input** axes `(a, b)` of the logit tensor `L[a,b,c]`, orthonormal real Fourier basis over `Z_p`; the class axis is untouched | same | same | same |
| 2 | which components are kept by restricted | the constant plus, per key `k`, the **two** unit directions `cos(ω_k(a+b))` and `sin(ω_k(a+b))` | the constant plus **every pair** of key rows, cross-frequency included | the constant plus the full 2×2 `(cos,sin)²` block of each key `k` (4 components) | **matches the published operator** |
| 3 | which components are removed by excluded | exactly those same sum directions; everything else, constant included, is kept | every component touching *any* key row or column | the same four-component blocks | the two sum directions per key |
| 4 | are cross-frequency blocks permitted | **no** — no source ever keeps or deletes a `k₁ ≠ k₂` product | **yes — this is the defect** | no | no |
| 5 | how cos/sin pairs are handled | as an orthogonal **projection** inside each 4-D block, not a boolean mask: the sum directions are a 2-D *subspace* | boolean mask over rows | boolean mask over the block | orthogonal projection (`(A−B)/2`, `(C+D)/2`) |
| 6 | how the constant is handled | restricted **re-adds** the constant (the Colab's `bias_correction=True`, equivalent to adding the grid mean); excluded leaves it untouched | kept by both masks | kept by both | kept by restrict; untouched by exclude |
| 7 | how key frequencies are chosen | DFT of the neuron→logit map `W_L = W_U W_out` along the logit axis, norm over neurons; keep those with "nontrivial" coefficients. The **count is measured** (5 for their main model, 3–4 for other seeds). The numeric threshold is **[NOT FOUND IN SOURCE]** | top-8 of the `W_E` power spectrum, **capped at 8**; the cap binds on all 16 legacy runs so the count was never data-determined | same as legacy | same as legacy |
| 8 | split for restricted loss | **[NOT FOUND IN SOURCE]** in the paper. Released Colab: `mode='all'` *and* `mode='train'`, reported as identical. TransformerLens demo: **test** | full grid | full grid | full grid |
| 9 | split for excluded loss | **training pairs only** — paper §5.1 "We measure this on the training data"; Colab `mode='train'` | full grid | full grid | full grid |

## 2. Component counts, computed

For `p = 113` and the canonical 8 key frequencies `{18, 15, 1, 11, 13, 22, 56, 36}`, from
`mask_protocols.build_protocol` (asserted in `test_core.py`):

| protocol | kept by restricted | removed by excluded | share of the 12,769-cell tensor removed | keeps cross-frequency |
|---|---|---|---|---|
| `legacy_broad_mask` | **289** | **3,360** | **26.3 %** | **yes** |
| `same_frequency_block` | 33 | 32 | 0.3 % | no |
| `sum_directions_only` | 17 | 16 | 0.1 % | no |
| published operator, our 8 key frequencies | 17 | 16 | 0.1 % | no |
| published operator, Nanda's own 5 | 11 | 10 | 0.08 % | no |

So the legacy restricted loss gives the "circuit" **17×** more freedom than the hypothesis allows, and
the legacy excluded loss deletes **210×** more of the logit tensor than the hypothesis targets. Both
biases push in the direction of the reported conclusion: restricted looks too good, excluded looks too
destroyed. The blog's own description of the published operator — "deleting 2 directions among 113²" per
frequency — matches `sum_directions_only` (2 per key) and is irreconcilable with 3,360.

## 3. Verdict per variant

| variant | verdict |
|---|---|
| `legacy_broad_mask` | **Not a reproduction.** Nothing in the paper, the blog or the code keeps or deletes cross-frequency components. Its restricted and excluded masks are not even complements of one another (289 vs 3,360). Keep it runnable for comparison; never label it a reproduction. |
| `same_frequency_block` | **Partial — matches one reading of the paper's text, not the released code.** §5.1 says "the 20 terms" and §4.1 identifies them as five same-frequency 2×2 blocks, so keeping four components per key is defensible *as a reading of the prose*. But no source keeps the **difference** directions `cos/sin(ω_k(a−b))`, which this variant does. Report it under the name **`paper_literal_2x2_block`**; never call it exact. |
| `sum_directions_only` | **Matches the operator the released code implements**, up to the split and the constant. The Colab's `get_component_cos_xpy` projects `(A,B)` onto `(1,−1)/√2` and `get_component_sin_xpy` projects `(C,D)` onto `(1,1)/√2`; our `_project` computes `u=(A−B)/2 → (u,−u)` and `v=(C+D)/2 → (v,v)`, the same orthogonal projection. `restrict` re-inserting `Lhat[0,0]` equals `bias_correction=True`. |
| `nanda_exact` | **Currently raises `NotImplementedError`, and that is correct** as long as item 8 is unresolved. See §4. |

## 4. Specification of the four required functions

Master prompt §6.3 names four functions. Their contracts:

### `nanda_exact_restricted_loss(logits, key_freqs, cfg)`
Keep the constant plus, per key `k`, the projection onto `span{cos(ω_k(a+b)), sin(ω_k(a+b))}`; inverse
transform; cross-entropy in float64. Because the paper does not state its split, this function
**returns all three** (`test`, `train`, `all`) with the split labelled, and the quoted "Nanda-style"
number is the **test** split, following the TransformerLens demo — with that provenance stated at every
use. Key frequencies come from the `nanda` rule (§1 row 7) with our declared 0.25-of-maximum threshold,
never from a fixed top-8.

### `nanda_exact_excluded_loss(logits, key_freqs, cfg)`
`L − P_S(L)` with `S` the same sum-direction subspace over all key frequencies at once; constant and
every other component untouched; evaluated on the **training pairs**, which the source does state. A
per-frequency variant (`L − P_{S_k}(L)` for each `k`) is also provided, because that is what the released
notebook actually plots.

### `full_grid_extension_restricted_loss` / `full_grid_extension_excluded_loss`
The **same component rule** as the two above, evaluated over the full `(a,b)` grid. This is *our*
methodological variant and is labelled as such wherever it appears. It exists so the new numbers can be
compared against the legacy full-grid ones without confounding the mask change with the split change.

### `legacy_broad_mask_variant(logits, key_freqs, cfg, which)`
The existing outer-product mask, kept runnable and reproduced bit-for-bit (pinned in `test_core.py`
against the original `_mode_indices` masks) so the old numbers can be regenerated and compared. Never a
reproduction.

**One item blocks `nanda_exact` from being called "exact" without qualification:** the paper's own
restricted-loss split (§1 row 8). It is `[NOT FOUND IN SOURCE]`. The implementation therefore reports
all three splits rather than guessing one, and the documentation says which convention any single quoted
number follows. Guessing here would manufacture exactly the false reproduction claim this module exists
to remove.

## 5. Required tests and their status

Master prompt §6.5 lists ten tests. Status as of 2026-09-03 (`test_core.py`, 125 checks, passing):

| test | status |
|---|---|
| correct Fourier round-trips | **passing** — `inv2d(fwd2d(L)) == L` to 1e-9 |
| exact number of retained components per variant | **passing** — 289/33/17 and 3,360/32/16 pinned |
| no impermissible cross-frequency blocks | **passing** — probed by construction; only the legacy variant keeps them |
| correct handling of the constant term | **passing** — for every implemented variant |
| artificially injected single frequencies | **passing** — an ideal `cos(ω_k(a+b))` survives restriction and is annihilated by exclusion |
| artificially injected mixed frequencies | **passing** — a `cos(ω₃a)cos(ω₅b)` term leaks through the legacy mask and through neither exact variant |
| restricted loss of an ideal Fourier circuit | **passing** (as the survival test above) |
| excluded loss of that same circuit | **passing** (as the annihilation test above) |
| separate evaluation of train and test splits | **NOT YET** — the current implementation evaluates the full grid only; this is what the four new functions add |
| vectorized vs reference implementation | **passing** — analytic `n_kept`/`n_removed` verified against the operator trace, which equals the rank of a projection |

Two further checks added on 2026-09-03: `DiffDirectionsOnly` must annihilate `cos(ω_k(a+b))` and keep
`cos(ω_k(a−b))` (the mirror control), and `per_frequency_shares` must sum to the total non-constant power.

## 6. Consequences already applied

* `RESULTS.md` §2 carries an inline `[AUDIT]` note stating that the restricted/excluded numbers use the
  legacy mask, giving the 289-vs-17 and 3,360-vs-16 counts, and withdrawing the sentence that claimed the
  eight key frequencies alone solve the task.
* `README.md` carries a matching note.
* `progress_measures.TOP_LEVEL_PROTOCOL` remains `legacy_broad_mask` **only** because the frozen explorer
  reads those keys; the file records which protocol filled them, and every variant is stored alongside.
* The pre-registration's primary key-frequency rule is the published one, not top-8
  (`docs/PREREGISTRATION.md` §4.3).

## 7. Open items

1. The paper's restricted-loss split — `[NOT FOUND IN SOURCE]`, handled by reporting all three.
2. The published numeric threshold for "nontrivial" key-frequency coefficients — `[NOT FOUND IN SOURCE]`;
   ours is 0.25 of the maximum, declared in advance with sensitivity at 0.10 and 0.50.
3. The verification pass over `docs/sources/nanda2023_progress_measures.md` was interrupted and is
   **partial** (four verifier corrections landed). Items 1, 2 and §1 rows 7–9 should be re-checked
   against the primary before the final write-up.
4. The paper's §4.4 "113·113−40" ablation keeps a 3×3 block `{const, cos_k, sin_k}²` per key. That is a
   **different** ablation from restricted loss and, if ever implemented, must be reported under its own
   name (`paper_3x3_block`), never as restricted loss.
