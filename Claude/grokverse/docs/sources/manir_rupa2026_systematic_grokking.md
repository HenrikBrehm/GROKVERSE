# Manir & Rupa (2026) — "A Systematic Empirical Study of Grokking: Depth, Architecture, Activation, and Regularization"

| | |
|---|---|
| arXiv | **2603.25009v1** [cs.LG], submitted **26 Mar 2026** (only v1 exists as of 2026-09-02) |
| Authors | Shalima Binta Manir (University of Maryland, Baltimore County), Anamika Paul Rupa (Howard University); footnote "Equal contribution." |
| Venue | none stated (preprint) |
| Code release | **[NOT FOUND IN SOURCE]** — no repository/URL anywhere in the paper; §5.3 refers only to "the original notebook runs" |
| Fetched | `https://arxiv.org/html/2603.25009`, `https://arxiv.org/abs/2603.25009`, `https://arxiv.org/pdf/2603.25009` |
| Read by | Claude (Fable 5.1) for GROKVERSE WP-0, 2026-09-02 |

**Provenance of the quotes.** The PDF was downloaded and converted with `pdftotext -layout`; every quote below was taken from that text and located by section / table / figure number. `pdftotext` drops or garbles some glyphs (`λ`, `×`, `≥`, `≈`, `∼`, `ℓ2`, `ΔT`); these were restored from the arXiv HTML rendering of the same sentence. Per-seed table cells were re-summed to check that they reproduce the printed means (they do — see the ✓ marks). Tags used: **[NOT FOUND IN SOURCE]** = looked for, not present; **[OBSERVATION]** = my own inference from the printed numbers, not a claim the paper makes; **[FROM MEMORY - UNVERIFIED]** = not from the fetched text.

**Verification pass [verifier 2026-09-02].** Independently re-fetched `https://arxiv.org/abs/2603.25009` and `https://arxiv.org/html/2603.25009` (eight targeted extractions) and re-downloaded `https://arxiv.org/pdf/2603.25009` (v1, 18 pages, converted with `pdftotext -layout`). Every quoted passage, table cell, section/table/figure number and every **[NOT FOUND IN SOURCE]** label below was checked against that text; all printed means/stds/ratios were recomputed from the per-seed values. Corrections and additions are marked "[verifier 2026-09-02]"; the writer's caveats are left intact. Summary of the pass is in §7 at the end.

**One-line summary.** A p=97 modular-arithmetic study on Google-Colab T4s (~10–12 GPU-hours total, A.1) whose headline is that architecture matters less than regularization. It **does** contain a Fourier-concentration comparison of Transformer vs MLP embeddings (§5.7, Table 8: Transformer 98.5 % vs MLP ≈75 % top-5 concentration, single seed), and it finds the **MLP grokking faster than the Transformer** in every multi-seed comparison (1.11×, 1.90×, 1.42×) — the *opposite direction* to GROKVERSE's timing result.

---

## 1. Setup: task, modulus, split, architectures, optimizer, weight-decay grid, steps, seeds

### 1.1 Task and modulus (§3.1, Eq. 1)

> "We study the modular addition task: f(a, b) = (a + b) mod 97, a, b ∈ {0, 1, . . . , 96}, (1) which defines a classification problem with 97² = 9,409 total input pairs and 97 output classes. Following Power et al. (2022), each integer is represented via a learned embedding. For Transformers, (a, b) is provided as a length-2 input sequence; for MLPs, the embeddings are summed before being passed through the network." — §3.1

Second task (§5.8): "We replace the addition task (a + b) mod 97 with multiplication (a × b) mod 97, excluding pairs where a = 0 or b = 0 (which always map to class 0), leaving 96² = 9,216 total pairs."

### 1.2 Train fraction (§3.2)

> "We use a fixed 20%/80% train/test split, sampling 1,882 pairs for training and holding out the remaining 7,527 for testing. The same split is used across all experiments to ensure comparability. This small training fraction is critical for inducing the memorization phase necessary for grokking; with larger fractions, models generalize without the characteristic delay." — §3.2

**[OBSERVATION]** §5.1 says "a 20% training fraction (1,883 pairs)"; §3.2 says 1,882. 0.2 × 9,409 = 1,881.8. One-pair inconsistency in the paper itself. Whether the *same* split is used for the multiplication task (9,216 pairs) is **[NOT FOUND IN SOURCE]**. [verifier 2026-09-02] Confirmed: §5.8 says only "We replicate H2 (MLP vs. Transformer, 5 seeds each, 600,000 steps, optimal λ from H4) and H3 (GELU/ReLU/Tanh, 5 seeds each, 400,000 steps) on this task using otherwise identical hyperparameters." — the train *fraction* is therefore presumably still 20 %, but the pair set is different (9,216 ≠ 9,409), so it cannot literally be "the same split"; the paper does not restate the fraction or the split for multiplication.

### 1.3 Training protocol (§3.3, Table 1)

> "Unless otherwise stated, we use the following default hyperparameters. All models are trained using full-batch gradient descent (i.e., batch size equals the full training set). We use SGD with momentum for MLPs and AdamW Loshchilov & Hutter (2017) for Transformers; the relatively large weight decay for AdamW follows prior work demonstrating its necessity for inducing grokking in this setting Liu et al. (2022)." — §3.3

**Table 1: Default training hyperparameters** (verbatim rows)

| Hyperparameter | Value |
|---|---|
| Optimizer | SGD (MLP); AdamW (Transformer) |
| Learning rate | 10⁻² (SGD); 10⁻³ (AdamW) |
| Momentum | 0.9 (SGD only) |
| Weight decay (default) | 2 × 10⁻³ (SGD); 1.0 (AdamW) |
| Gradient clip norm | 1.0 (depth ≥ 8) |
| Batch size | Full train set |
| Training steps | 400,000–800,000 |
| Loss function | Cross-entropy |
| Seeds per config | 3–5 |

Not specified anywhere: AdamW β₁/β₂ **[NOT FOUND IN SOURCE]**; learning-rate schedule / warm-up **[NOT FOUND IN SOURCE]**; initialization scheme **[NOT FOUND IN SOURCE]**; evaluation frequency for the grokking-delay tables **[NOT FOUND IN SOURCE]** (only Exp. 6 states "log RMS parameter norm, train accuracy, and test accuracy every 2,000 steps"; **[OBSERVATION]** every T_train / T_grok in Tables 2, 3, 9 is a multiple of 2,000, so the resolution of every reported delay is plausibly 2,000 steps — the paper never says so). Note that the MLP learning rate actually used in H2–H4 is **3 × 10⁻²**, not the Table 1 default (Table 3/4/5 captions).

### 1.4 Grokking-delay definition (§3.4)

> "We define T_train as the first step at which training accuracy ≥ 99%, and T_grok as the first step at which validation (test) accuracy ≥ 99%. The grokking delay is: ΔT = T_grok − T_train." — §3.4 (displayed, unnumbered equation)

> "If a configuration does not reach 99% test accuracy within the step budget, it is labeled did not grok (DNF). DNF seeds are excluded from mean delay computations but reported explicitly alongside the number of successful seeds. We report the mean and standard deviation of ΔT over non-DNF seeds." — §3.4

Table 5 caption adds two DNF classes: "DNF_tr = failed to memorize within 60,000 steps. DNF_te = memorized but did not grok within budget."

### 1.5 Transformer definition (§4.1)

> "Our Transformer Vaswani et al. (2017) is an encoder-only model with embedding dimension d = 512, h = 4 attention heads, and depth 1 (one encoder layer). Inputs a and b are each mapped to learned embeddings (97 × 512) and combined with learned positional embeddings (2 × 512), forming a sequence of length 2. The encoder uses a pre-norm configuration (norm_first=True), a feedforward dimension of 4d = 2,048, GELU activations Hendrycks & Gimpel (2016), and no dropout. The final representation is obtained by mean-pooling across the sequence dimension and projecting to 97 output logits. The model is trained with AdamW Loshchilov & Hutter (2017) (lr = 10⁻³, weight decay = 1.0, gradient clipping norm = 1.0). The relatively large weight decay (λ = 1.0) follows Liu et al. (2022) and is required to reliably induce grokking with AdamW." — §4.1

So: **no `=` / operator token, sequence length 2, LayerNorm (pre-norm), GELU, readout by mean-pooling** — i.e. a `torch.nn.TransformerEncoderLayer`-style block, *not* the Nanda et al. recipe. Whether the input embedding is shared between the two positions ("each mapped to learned embeddings (97 × 512)") is ambiguous in the text; Exp. 7 treats "the learned embedding matrix W_E ∈ ℝ^{97×d}" as a single matrix. Parameter count **[NOT FOUND IN SOURCE]**. Whether the unembedding is tied **[NOT FOUND IN SOURCE]**.

### 1.6 MLP definition (§4.2, §4.3)

> "The baseline MLP takes as input two indices a, b ∈ {0, . . . , 96}, maps them to embeddings (97 × width), and sums the embeddings to form the hidden state h₀ = Emb(a) + Emb(b). This is passed through d feedforward layers: h_i = GELU(W_i h_{i−1} + b_i), followed by a linear projection to 97 logits Hendrycks & Gimpel (2016). The baseline uses width w = 256 and depth d = 2." — §4.2

> "For the depth 8 variant (H1), we adopt a residual MLP architecture He et al. (2016) to ensure optimization stability. Each block is a pre-norm residual unit: h ← h + 0.5 · FC₂(GELU(FC₁(LN(h)))), where LN denotes layer normalization and 0.5 is a residual scaling factor. The width is increased to w = 512, and gradient clipping (max norm = 1.0) is applied to prevent instability." — §4.2

> "We evaluate three depths: d ∈ {2, 4, 8}. Depths 2 and 4 use the standard (non-residual) MLP with width 256. Depth 8 uses the residual MLP with width 512 and LayerNorm, which is necessary for stable training at this depth. We use 5 seeds for depths 2 and 4, and 3 seeds for depth 8." — §4.3

**Input parametrization is *summed* embeddings** (`Emb(a) + Emb(b)`), one shared table of shape 97 × width. The MLP used in the architecture comparison (H2, H3 Sweep B, H4, H2-mul) is **depth 4, width 512, GELU, SGD lr = 3 × 10⁻²** (Table 3/4/5/10 captions), *not* the depth-2 width-256 baseline. Parameter counts **[NOT FOUND IN SOURCE]**.

### 1.7 Activations (§4.4)

> "We evaluate three activation functions applied to all feedforward sublayers (Transformer FFN and MLP hidden layers): activation ∈ {ReLU, GELU, Tanh}." — §4.4

### 1.8 Weight-decay grids (§5.5)

> "For the MLP (depth-4 GELU, width 512, SGD, lr = 3 × 10⁻²) we sweep: λ_MLP ∈ {10⁻⁵, 5 × 10⁻⁵, 10⁻⁴, 5 × 10⁻⁴, 10⁻³, 2 × 10⁻³, 5 × 10⁻³}. For the Transformer (width 512, 4 heads, AdamW, lr = 10⁻³) we sweep: λ_TF ∈ {0.01, 0.1, 0.5, 1.0, 2.0, 5.0}. We use 3 screening seeds per λ (up to 400,000 steps each), then confirm the optimal λ for each architecture with 5 seeds for a final fair comparison (Part C)." — §5.5 Setup

### 1.9 Steps and seeds per experiment (collected)

| Exp. | Section | Models | Steps | Seeds |
|---|---|---|---|---|
| 1 canonical | §5.1 | depth-2 GELU MLP, w=256, SGD lr 10⁻², λ=2×10⁻³ | 400,000 | "seed 0" |
| 2 depth (H1) | §5.2 | MLP d ∈ {2,4,8} | 400,000 | 5 per depth ("depth 8 results use 3 previously completed seeds plus 2 additional seeds run here") |
| 3 architecture (H2) | §5.3 | 1-layer TF (λ=1.0) vs depth-4 MLP w=512 (λ=5×10⁻⁴) | "up to 600,000" | 5 each |
| 4 activation (H3) | §5.4 | depth-4 MLP, Sweep A (w=256) / Sweep B (w=512) | 400,000 | 5 per cell |
| 5 weight decay (H4) | §5.5 | MLP and TF sweeps; Part C | 400,000 (screen) | 3 per λ; 5 for Part C |
| 6 weight norm | §5.6 | six configs | logged every 2,000 | **1 seed each** |
| 7 Fourier | §5.7 | MLP-GELU, MLP-ReLU, TF at optimal λ | "to full grokking" | **seed 0 only** |
| 8 multiplication | §5.8 | H2-mul (600,000 steps), H3-mul (400,000) | | 5 each |

Hardware (A.1): "All experiments were run on an NVIDIA T4 GPU (Google Colab) using CUDA graph acceleration (torch.cuda.CUDAGraph). ... Total compute across all reported experiments was approximately 10–12 GPU-hours." Exp. 1 ran "611 seconds on an NVIDIA GeForce RTX 3060 Laptop GPU" (§5.1).

### 1.10 Canonical run numbers (§5.1, Figure 1) [verifier 2026-09-02 — missing from the writer's note]

> "Setup. We train a 2-layer MLP (width 256, GELU activations) with SGD (lr = 10⁻², momentum = 0.9, weight decay = 2 × 10⁻³) on a 20% training fraction (1,883 pairs) for 400,000 steps. We report results for seed 0." — §5.1

> "Results. Figure 1 shows the canonical grokking pattern. Training accuracy reaches 100% rapidly at step T_train = 1,000, while test accuracy stagnates near chance (19.9%) for thousands of subsequent steps. A sharp generalization transition then occurs, with test accuracy crossing 99% at step T_grok = 33,000, yielding a grokking delay of ΔT = 32,000 steps. After grokking, test accuracy stabilizes at 99.35% and continues to improve slowly, reaching 99.35% by step 400,000." — §5.1

Figure 1 caption: "Train accuracy (blue) reaches 100% at step T_train = 1,000, while test accuracy (orange) remains near chance until a sharp transition at step T_grok = 33,000 (grokking delay = 32,000 steps). Test accuracy stabilizes at 99.35% and continues to slowly improve through step 400,000. The dashed line marks the 99% threshold."

**[OBSERVATION — verifier]** Chance level for a 97-class problem is ≈ 1.0 %, not 19.9 %; a test accuracy of 19.9 % during the "memorization" plateau is *well above* chance and is not commented on. Also, Table 2 reports this same configuration's seed 0 with T_train = 2,000 / ΔT = 32,000 (not 1,000 / 32,000 as in §5.1); the two accounts of seed 0 differ in T_train and T_grok by 1,000 steps each while agreeing on ΔT = 32,000 (which also matches Table 7's "Depth-2 GELU (baseline)" delay). Consistent with a 2,000-step evaluation grid in Table 2 vs. a finer one in §5.1, but the paper does not say so.

---

## 2. Everything the paper reports about Transformer vs MLP

### 2.1 Headline statements

Abstract: "(2) the apparent gap between Transformers and MLPs largely disappears (1.11× delay) under matched hyperparameters, indicating that previously reported differences are largely due to optimizer and regularization confounds".

Contribution 2 (§1): "Architecture differences are largely confounded. Under matched hyperparameters, the gap between Transformers and MLPs is substantially reduced, and only becomes moderate when each is evaluated at its optimal regularization, indicating that previously reported differences are largely driven by optimizer and regularization choices."

§6.2: "Our H2 results nuance the common claim that Transformers grok more slowly than MLPs: the gap is mostly an optimizer and regularization confound, not an architectural one. The Transformer's residual variance advantage is consistent with the slingshot mechanism of Thilak et al. (2022)."

Conclusion (§7): "Our results show that grokking dynamics are not primarily determined by architecture, but by the interaction between optimization stability and regularization."

Note the phrase "largely vanishes" does not occur; the paper's wording is "largely disappears" (abstract), "nearly vanishes" (§6.1 item 2), "near-parity" (§5.3).

[verifier 2026-09-02] Confirmed by grep of the full PDF text: "largely vanishes" occurs nowhere. The §6.1 item 2 sentence, verbatim — it is the one place the paper spells out what "matched" means: "H2 – Architecture: At matched hyperparameter configurations (both using λ = 1.0 for Transformer, λ = 5 × 10⁻⁴ for MLP), the gap nearly vanishes: MLP 45,600 ± 5,550 vs. Transformer 50,800 ± 22,565 (ratio 1.11×). However, H4 reveals that λ = 1.0 is suboptimal for the Transformer—its true optimum is λ = 5.0." — §6.1. I.e. "matched" = each architecture at its own canonical (λ, lr, optimizer), exactly as the writer says in §2.2 below.

### 2.2 Experiment 3 — H2 at "canonical" configs (§5.3, Table 3, Figure 3)

> "Setup. We compare a 1-layer Transformer (width 512, 4 heads, AdamW, λ = 1.0) against a 4-layer MLP (width 512, GELU, SGD, λ = 5 × 10⁻⁴), both trained on 20% of the modular addition data for up to 600,000 steps on an NVIDIA T4 GPU, over 5 random seeds. These hyperparameters match the exact configurations used in the original notebook runs. The architectures use different optimizers by design: Transformers require AdamW with strong weight decay for reliable grokking, while MLPs grok reliably with SGD and lighter regularization." — §5.3

> "Results. ... When both architectures are evaluated at their exact canonical hyperparameter configurations, the grokking gap is far smaller than initial estimates suggested: the MLP achieves a mean delay of 45,600 ± 5,550 steps while the Transformer achieves 50,800 ± 22,565 steps—a ratio of 1.11× with 4.1× higher variance. All 10 seeds grokked. This is a substantial revision of a prior estimate of 2.18×, which arose from inconsistent hyperparameter configurations between the two architectures; in particular, the Transformer in the earlier run used a suboptimal weight decay and learning rate relative to its final canonical config. One Transformer seed (seed 2) exhibited an elevated delay of 76,000 steps, consistent with the 'slingshot' dynamics reported by Thilak et al. (2022)." — §5.3

**Table 3** ("H2 Architecture comparison over 5 seeds (corrected configs). Transformer uses AdamW (λ = 1.0, lr = 10⁻³); MLP uses SGD (λ = 5 × 10⁻⁴, lr = 3 × 10⁻²).")

| Architecture | Seed | T_train | T_grok | ΔT |
|---|---|---|---|---|
| Transformer | 0 | 4,000 | 46,000 | 42,000 |
| Transformer | 1 | 4,000 | 36,000 | 32,000 |
| Transformer | 2 | 4,000 | 80,000 | 76,000 |
| Transformer | 3 | 2,000 | 32,000 | 30,000 |
| Transformer | 4 | 2,000 | 76,000 | 74,000 |
| Transformer mean | | | | **50,800 ± 22,565** ✓ (254,000/5) |
| MLP | 0 | 2,000 | 56,000 | 54,000 |
| MLP | 1 | 2,000 | 42,000 | 40,000 |
| MLP | 2 | 2,000 | 44,000 | 42,000 |
| MLP | 3 | 2,000 | 50,000 | 48,000 |
| MLP | 4 | 2,000 | 46,000 | 44,000 |
| MLP mean | | | | **45,600 ± 5,550** ✓ (228,000/5) |

> "Analysis. The near-parity between architectures (1.11×) at matched configurations is the key finding of H2. The earlier apparent gap of 2.18× was substantially an artifact of hyperparameter imbalance: in those initial runs the Transformer was paired with a suboptimal weight decay relative to the MLP, artificially inflating its grokking delay. When both architectures are evaluated at their respective canonical configurations, the Transformer grokks only marginally more slowly on average. The Transformer's remaining 4.1× higher variance—even at matched configs—is consistent with the 'slingshot' mechanism of Thilak et al. (2022): the generalization transition in attention-based models is more sensitive to oscillatory weight-norm dynamics, producing a wider spread of grokking steps across random seeds. This result cautions against attributing architecture-level differences in grokking to inductive biases alone when optimizer and regularization choices differ." — §5.3

Figure 3 caption: "(b) Mean delay ± std: MLP achieves 45.6k ± 5.6k vs. Transformer 50.8k ± 22.6k, a 1.11× difference with 4.1× higher variance. (c) T_train vs. T_test scatter: both architectures memorize rapidly but diverge in generalization onset."

The runs behind the "prior estimate of 2.18×" are not tabulated; their hyperparameters are **[NOT FOUND IN SOURCE]** beyond "suboptimal weight decay and learning rate". What "matched hyperparameters" means here: each architecture at *its own* "canonical" config (different optimizer, lr, λ, depth) — not identical hyperparameters.

### 2.3 Part C — "definitive" comparison at each optimum (§5.5, Figure 5c)

> "Part C: fair comparison at optimal λ each (5 seeds). At their respective optimal regularization, the MLP achieves mean ΔT = 26,800 ± 6,419 steps and the Transformer achieves 50,800 ± 38,745 steps—a ratio of 1.90×. This is our definitive architecture comparison, superseding H2's 1.11× (which used a suboptimal λ = 1.0 for the Transformer)." — §5.5

> "Critically, H2 used λ = 1.0 for the Transformer, which is suboptimal by a factor of 1.83× in delay explaining why the H2 gap of 1.11× was so small. The true architecture gap at optimal regularization is 1.90× (Part C), confirming that architecture does have a real, if moderate, effect on grokking speed beyond optimizer choice." — §5.5 Analysis

Figure 5 caption (c): "Fair comparison at optimal λ each (5 seeds): MLP achieves 26.8k ± 6.4k vs. Transformer 50.8k ± 38.7k, a 1.90× gap—the definitive architecture comparison superseding H2's 1.11×."

§6.1 item 2: "At optimal λ each (Part C of H4), the gap is 1.90× (MLP: 26,800 ± 6,419; Transformer: 50,800 ± 38,745, 5 seeds each). Architecture does have a real effect on grokking speed, but it is moderate and substantially entangled with regularization choices. The Transformer retains 6× higher variance at its own optimum, consistent with slingshot dynamics Thilak et al. (2022)."

Per-seed Part C values: **[NOT FOUND IN SOURCE]** (no table; Figure 5c is a bar chart).

**[OBSERVATION — internal tension in the Transformer-at-λ=5.0 numbers.]** The paper prints three different delays for the Transformer at its optimum λ = 5.0: Table 6 (3 screening seeds) **24,000 ± 10,583**; Table 7 / Table 8 (seed 0) **12,000**; Part C (5 seeds) **50,800 ± 38,745** — which is more than double the screening mean and *identical to the digit* to the H2 mean at λ = 1.0 (50,800). The paper does not comment on this. Anyone citing "1.90×" should note that it rests on five unlisted seeds whose mean disagrees with the same authors' three-seed screen at the same λ.

### 2.4 Numbers from the sweeps that feed the comparison (Tables 5, 6 — full tables in §4 below)

Transformer λ = 1.0: 44,000 ± 12,166 (3/3); λ = 5.0: 24,000 ± 10,583 (3/3). 44,000/24,000 = 1.83 ✓. MLP λ = 5 × 10⁻⁴: 45,333 ± 7,572; λ = 10⁻³: 25,333 ± 5,033.

### 2.5 Weight-norm runs (Table 7, 1 seed each)

Delay column: Transformer (λ = 1.0) 58,000; Transformer (λ = 5.0) 12,000; MLP GELU (H2 config, λ = 5 × 10⁻⁴) 54,000; MLP GELU (optimal λ = 10⁻³) 26,000. **[OBSERVATION]** In this single-seed table the Transformer at λ = 5.0 (12,000) is *faster* than the MLP at its optimum (26,000) — the only place in the paper where the Transformer leads.

### 2.6 Modular multiplication (§5.8, Table 9, Figure 8)

> "Architecture (H2-mul). All 10 seeds grokked. MLP achieves mean ΔT = 24,800 ± 3,347 steps; Transformer achieves 35,200 ± 17,470—a ratio of 1.42×. Both architectures grokk faster on multiplication than addition (MLP: 24,800 vs. 26,800; Transformer: 35,200 vs. 50,800), suggesting that at optimal regularisation multiplication is not harder than addition for these models. The architecture gap narrows from 1.90× to 1.42×, but the direction is preserved: MLPs grokk faster in both tasks." — §5.8

**Table 9** ("H2-mul: Architecture comparison on (a × b) mod 97. Both architectures use optimal λ from H4 (MLP: λ = 10⁻³; Transformer: λ = 5.0).")

| Arch. | Seed | T_train | T_grok | ΔT |
|---|---|---|---|---|
| MLP | 0 | 2,000 | 24,000 | 22,000 |
| MLP | 1 | 2,000 | 28,000 | 26,000 |
| MLP | 2 | 2,000 | 26,000 | 24,000 |
| MLP | 3 | 2,000 | 24,000 | 22,000 |
| MLP | 4 | 2,000 | 32,000 | 30,000 |
| MLP mean | | | | **24,800 ± 3,347** ✓ |
| Transformer | 0 | 2,000 | 32,000 | 30,000 |
| Transformer | 1 | 2,000 | 66,000 | 64,000 |
| Transformer | 2 | 2,000 | 26,000 | 24,000 |
| Transformer | 3 | 2,000 | 22,000 | 20,000 |
| Transformer | 4 | 2,000 | 40,000 | 38,000 |
| Transformer mean | | | | **35,200 ± 17,470** ✓ |

§5.8 Analysis: "The narrowing of the architecture gap from 1.90× to 1.42× suggests that the Transformer's self-attention is relatively better suited to multiplication than addition—consistent with the observation that both architectures are actually faster on multiplication, but the Transformer benefits more (0.69× of its addition time vs. 0.91× for the MLP)."

**[OBSERVATION — verifier 2026-09-02]** 35,200 / 50,800 = 0.693 ✓, but 24,800 / 26,800 = 0.925, not the printed "0.91×". Small, but it is the paper's own arithmetic, so do not copy "0.91×" without this note. Also §5.8 opens its Analysis with a stronger claim than the writer quoted: "The key finding is that both the architecture effect and the GELU advantage generalize to modular multiplication, confirming they are properties of the models and training dynamics rather than artifacts of the addition task structure." — §5.8.

### 2.7 Direction and magnitude, collected

| Comparison | MLP ΔT | Transformer ΔT | ratio TF/MLP | seeds |
|---|---|---|---|---|
| "prior estimate" (untabulated) | — | — | 2.18× | ? |
| H2, canonical configs (Table 3) | 45,600 ± 5,550 | 50,800 ± 22,565 | 1.11× | 5 + 5 |
| Part C, optimal λ each (Fig. 5c) | 26,800 ± 6,419 | 50,800 ± 38,745 | 1.90× | 5 + 5 |
| H2-mul, optimal λ (Table 9) | 24,800 ± 3,347 | 35,200 ± 17,470 | 1.42× | 5 + 5 |
| Table 7 single seed, optimal λ | 26,000 | 12,000 | 0.46× | 1 + 1 |

**In every multi-seed comparison the MLP groks faster than the Transformer**, and the Transformer's seed-to-seed spread is 4–6× larger. The paper's own message is that the *magnitude* is a hyperparameter artifact (2.18 → 1.11 → 1.90 → 1.42), not that any particular ratio is a property of the architectures.

---

## 3. Representation / Fourier metrics — PRESENT (§5.7, Table 8, Figure 7)

This is the section that decides whether GROKVERSE's "Transformer has higher top-k Fourier concentration than the MLP" is already published. **It is.**

### 3.1 Full text of Experiment 7

> "Goal. Test whether all three model types converge to a sparse Fourier representation after grokking Nanda et al. (2023), and whether activation functions or architecture affect the structure of the learned solution." — §5.7

> "Setup. We train MLP (GELU), MLP (ReLU), and Transformer to full grokking (optimal λ each, seed 0), then compute the discrete Fourier transform of the learned embedding matrix W_E ∈ ℝ^{97×d} along the residue dimension. We report the top-5 frequency concentration (fraction of total Fourier energy in the five highest-energy frequencies) and visualize each model's embeddings projected onto their top-2 Fourier directions." — §5.7

> "Results. Table 8 and Figure 7 show the results. All three models converge to highly sparse Fourier representations, confirming Nanda et al. (2023)'s finding for a broader set of architectures and activations. However, the degree of sparsity differs substantially: the Transformer concentrates 98.5% of its embedding energy in just 5 frequencies, compared to 74.7% for MLP-GELU and 75.6% for MLP-ReLU." — §5.7

> "The MLP models share two dominant frequencies (6 and 21), suggesting a common algorithmic solution, while the Transformer uses an entirely different set of frequencies ([0, 16, 29, 32, 34] vs. [6, 12, 21, 37, 46]). The DC component (frequency 0) appearing in the Transformer's solution has no analogue in the MLP solutions, consistent with the Transformer's mean-pooling operation creating a bias toward position-invariant representations." — §5.7

> "MLP-ReLU took 3.1× longer to grok (80,000 vs. 26,000 steps) but arrived at an equally sparse representation (75.6% vs. 74.7%), consistent with the weight norm analysis: the same generalizing solution is reached by both activations, but GELU traverses the weight norm trajectory faster." — §5.7

> "Analysis. Three conclusions follow from these results. First, sparse Fourier representations are a universal property of grokked models on modular addition, appearing in both MLPs and Transformers regardless of activation function. Second, the Transformer's dramatically higher concentration (98.5% vs. ∼75%) suggests that self-attention acts as an implicit sparsity-promoting mechanism in the frequency domain, enforcing a cleaner algorithmic solution. This may also explain the Transformer's higher seed-to-seed variance (H2): a more concentrated, brittle solution is more sensitive to initialisation. Third, the fact that GELU and ReLU converge to representations of equal sparsity—despite very different training timelines—confirms the weight norm analysis from Experiment 6: activation functions do not change what solution is learned, only how quickly the network gets there." — §5.7

**Table 8** ("Fourier concentration of learned embeddings post-grokking. Top-5 concentration = fraction of total Fourier energy in the five highest-energy frequency components.")

| Model | Top-5 frequencies | Top-5 conc. | Grokking delay |
|---|---|---|---|
| MLP (GELU) | 6, 12, 21, 37, 46 | 74.7% | 26,000 |
| MLP (ReLU) | 6, 7, 11, 16, 21 | 75.6% | 80,000 |
| Transformer | 0, 16, 29, 32, 34 | 98.5% | 12,000 |

Figure 7 caption: "Fourier analysis of learned embeddings post-grokking. Top row (a–c): Normalised Fourier energy spectrum of the embedding matrix W_E for each model. Highlighted bars (colour) show the top-5 frequencies. The Transformer (c) concentrates 98.5% of energy in 5 frequencies; MLPs concentrate ∼75%. Bottom row (d–f): Each residue's embedding projected onto the two highest-energy Fourier directions, coloured by residue value. The circular/elliptical structure in all three panels confirms that the learned representations encode modular arithmetic via sinusoidal patterns Nanda et al. (2023)."

§6.1 item 6: "The Transformer achieves 98.5% top-5 concentration vs. ∼75% for both MLPs, using a completely different set of frequencies."

### 3.2 What the metric definition does and does not specify

| Property | Status |
|---|---|
| Object analysed | "the learned embedding matrix W_E ∈ ℝ^{97×d}" (§5.7) — embedding only, no other weight |
| Transform | "discrete Fourier transform ... along the residue dimension" (§5.7) |
| Statistic | "top-5 frequency concentration (fraction of total Fourier energy in the five highest-energy frequencies)" (§5.7, Table 8 caption) |
| k fixed a priori | yes, k = 5; no data-driven count, no other k reported. "top-8"/"top-k" for k ≠ 5: **[NOT FOUND IN SOURCE]** |
| How "energy" is computed (|F|² summed over the d columns?) | **[NOT FOUND IN SOURCE]** |
| Whether k and 97−k are merged / one-sided spectrum | **[NOT FOUND IN SOURCE]**. **[OBSERVATION]** all listed frequencies are ≤ 48 = (97−1)/2, consistent with a one-sided spectrum, but the paper never says so |
| DC / constant mode | **included** in both numerator and denominator — frequency 0 is one of the Transformer's top-5 (Table 8) and the paper discusses it as a "DC component" |
| Normalisation of Figure 7 spectra | "Normalised Fourier energy spectrum" — how: **[NOT FOUND IN SOURCE]** |
| Checkpoint analysed | "to full grokking" / "post-grokking" — whether at T_grok or at end of budget: **[NOT FOUND IN SOURCE]** |
| Seeds | "seed 0" only; no variance, no per-seed range |
| λ used | "optimal λ each": Transformer λ = 5.0, MLP-GELU λ = 10⁻³ (from Table 6/5, and Table 8's delays 12,000 / 26,000 match Table 7's λ = 5.0 / λ = 10⁻³ rows). The λ used for MLP-ReLU: **[NOT FOUND IN SOURCE]** (Table 8 delay 80,000 ≠ Table 7's Sweep-B ReLU 180,000, so it is a different run). [verifier 2026-09-02] Confirmed not stated. The most plausible reading of "optimal λ each" is λ = 10⁻³ for *both* MLPs — the only MLP optimum the paper ever determines (Table 5, GELU), and the value the Table 10 caption applies to all three activations on multiplication ("depth-4 MLP, width 512, SGD, λ = 10⁻³") — but for the addition Fourier run this is an inference, not a statement in the paper |
| Architecture of "Transformer"/"MLP" | the §4.1 mean-pooled encoder (d = 512) and the depth-4, width-512 summed-embedding MLP |
| Time course of concentration | none — a single post-grokking snapshot |

### 3.3 Other representation / mechanistic measurements

| Measurement | In paper? |
|---|---|
| Weight norm (RMS over all parameters) at init and at grokking step | **yes** — §5.6, Table 7, Figure 6 (see §3.4) |
| Projection of embeddings onto top-2 Fourier directions (ring plots) | **yes**, qualitative only — Figure 7 d–f, no number |
| Attention patterns / attention maps | **[NOT FOUND IN SOURCE]** |
| Neuron-level frequency analysis, MLP neuron periodicity | **[NOT FOUND IN SOURCE]** |
| Restricted / excluded loss, any Nanda-style progress measure | **[NOT FOUND IN SOURCE]** |
| Causal ablation of frequencies or heads | **[NOT FOUND IN SOURCE]** |
| Effective rank, spectral entropy, participation ratio | **[NOT FOUND IN SOURCE]** |
| Logit / unembedding Fourier analysis | **[NOT FOUND IN SOURCE]** |
| Structure metric over training time | **[NOT FOUND IN SOURCE]** |
| Fourier analysis for the multiplication task | **[NOT FOUND IN SOURCE]** (Exp. 8 reports timing only) |

### 3.4 Weight-norm section (§5.6, Table 7, Figure 6) — verbatim

> "Setup. We train 6 configurations (1 seed each, CUDA graph accelerated) and log RMS parameter norm, train accuracy, and test accuracy every 2,000 steps: depth-2 GELU baseline; MLP GELU at H2 config (λ = 5 × 10⁻⁴) and at optimal (λ = 10⁻³); Transformer at H2 (λ = 1.0) and optimal (λ = 5.0); and MLP ReLU at Sweep-B (λ = 5 × 10⁻⁴)." — §5.6

> "Results. ... All six configurations follow the same qualitative weight norm trajectory: a rapid decay from initialization (‖W‖_RMS ≈ 0.1–0.4) toward a low plateau, with grokking occurring near the bottom of this decay. The weight norm at the grokking step is highly consistent across the five width-512 models: mean 0.0219 ± 0.0032 (CV = 14.5%). The depth-2 baseline (width 256) grokks at a higher norm (0.0695), consistent with its smaller parameter count producing a higher per-parameter norm at equivalent representational capacity." — §5.6

> "The most striking result concerns ReLU vs. GELU: both grokk at nearly identical weight norms (0.0219 vs. 0.0225), yet ReLU requires 6.9× more steps (180,000 vs. 26,000). This cleanly dissociates two mechanisms: (i) the weight norm threshold at which generalisation occurs is activation-independent; (ii) the activation function controls the rate at which weight decay drives the norm to that threshold, not the threshold itself." — §5.6

**Table 7** ("Weight norm at grokking step across six configurations. RMS norm is computed over all parameters. Depth-2 uses width 256; all others use width 512.")

| Configuration | Delay (steps) | ‖W‖ at init | ‖W‖ at grokking |
|---|---|---|---|
| Depth-2 GELU (baseline) | 32,000 | 0.372 | 0.0695 |
| MLP GELU (H2 config) | 54,000 | 0.209 | 0.0236 |
| MLP GELU (optimal λ) | 26,000 | 0.209 | 0.0225 |
| Transformer (λ = 1.0) | 58,000 | 0.126 | 0.0254 |
| Transformer (λ = 5.0) | 12,000 | 0.120 | 0.0160 |
| MLP ReLU (Sweep-B) | 180,000 | 0.209 | 0.0219 |
| Mean ± std (width-512 only) | | | 0.0219 ± 0.0032 |

> "Analysis. These results support a weight-norm-threshold account of grokking Kumar et al. (2023); Liu et al. (2022): for a given model capacity, there exists a characteristic weight norm below which the generalizing solution becomes accessible. The threshold appears to be set by model width (and hence parameter count) rather than by architecture, activation, or optimizer." — §5.6

[verifier 2026-09-02] The same paragraph continues: "What varies across conditions is how quickly weight decay can drive the norm to this threshold—a function of the λ–lr product and, crucially, of the activation function. GELU reaches the threshold 6.9× faster than ReLU despite an identical threshold value, consistent with GELU's smoother gradient landscape facilitating faster effective weight decay." — §5.6. Figure 6 caption (b): "All width-512 models converge to ‖W‖ ≈ 0.022 at grokking, regardless of architecture or activation."

---

## 4. The "Goldilocks" weight-decay finding (§5.5, Tables 5–6, Figure 5)

Abstract: "(4) weight decay is the dominant control parameter, exhibiting a narrow 'Goldilocks' regime in which grokking occurs, while too little or too much prevents generalization."

Contribution 4 (§1): "Weight decay is the dominant control parameter. Grokking occurs only within a narrow range of regularization strengths, with both insufficient and excessive weight decay preventing generalization."

> "MLP sweep. The relationship between λ and grokking is sharply non-monotonic. At λ ≤ 10⁻⁵, all seeds memorize but never grok within budget. At λ = 5 × 10⁻⁵, only 1 of 3 seeds grokked (delay 388,000 steps). Reliable fast grokking appears at λ = 10⁻⁴ (3/3 seeds, mean ΔT = 220,000 ± 36,056) and peaks at λ = 10⁻³ (3/3 seeds, mean ΔT = 25,333 ± 5,033). Strikingly, λ = 2 × 10⁻³ causes complete failure to memorize—all seeds fail to reach 99% training accuracy within 60,000 steps, a sharp cliff only one factor-of-two above the optimum." — §5.5

> "Transformer sweep. At λ = 0.01 no seed grokks within 400,000 steps. The optimal is λ = 5.0 (3/3 seeds, mean ΔT = 24,000 ± 10,583)—notably higher than λ = 1.0 used in H2, which yields mean ΔT = 44,000. The Transformer thus requires 5,000× stronger weight decay than the MLP to reach its optimum, reflecting AdamW's fundamentally different gradient scaling." — §5.5

**Table 5** ("H4a: MLP weight decay sweep (depth-4 GELU, width 512, SGD, lr = 3 × 10⁻², 3 seeds, 400,000 steps). DNF_tr = failed to memorize within 60,000 steps. DNF_te = memorized but did not grok within budget.")

| λ | Grokked/3 | Mean ΔT | Std |
|---|---|---|---|
| 10⁻⁵ | 0/3 | DNF_te: memorizes, no grok | |
| 5 × 10⁻⁵ | 1/3 | 388,000 | — |
| 10⁻⁴ | 3/3 | 220,000 | 36,056 |
| 5 × 10⁻⁴ | 3/3 | 45,333 | 7,572 |
| 10⁻³ | 3/3 | 25,333 | 5,033 |
| 2 × 10⁻³ | 0/3 | DNF_tr: cannot memorize | |
| 5 × 10⁻³ | 0/3 | DNF_tr: cannot memorize | |

**Table 6** ("H4b: Transformer weight decay sweep (width 512, 4 heads, AdamW, lr = 10⁻³, 3 seeds, 400,000 steps). DNF_te = memorized but did not grok within budget.")

| λ | Grokked/3 | Mean ΔT | Std |
|---|---|---|---|
| 0.01 | 0/3 | DNF_te: no grok in 400,000 steps | |
| 0.1 | 3/3 | 202,667 | 141,454 |
| 0.5 | 3/3 | 35,333 | 17,010 |
| 1.0 | 3/3 | 44,000 | 12,166 |
| 2.0 | 3/3 | 65,333 | 42,771 |
| 5.0 | 3/3 | 24,000 | 10,583 |

> "Analysis. These results confirm H4 while also providing a retroactive correction to H2. The MLP Goldilocks zone (λ ∈ [5 × 10⁻⁴, 10⁻³]) is narrow a factor-of-two step beyond the optimum collapses training entirely. The Transformer's optimal at λ = 5.0 is 5,000× larger, confirming that weight decay operates very differently under AdamW vs. SGD: AdamW's adaptive learning rates require much stronger ℓ2 pressure to counteract the effective learning rate scaling on large-norm parameters Loshchilov & Hutter (2017)." — §5.5 (punctuation as printed)

Figure 5 caption: "(a) MLP λ sweep: optimal at λ = 10⁻³; λ ≥ 2 × 10⁻³ prevents memorization entirely. (b) Transformer λ sweep: optimal at λ = 5.0, requiring 5,000× stronger regularization than the MLP."

**[OBSERVATION]** The Transformer sweep is non-monotonic between 0.5 (35,333) → 1.0 (44,000) → 2.0 (65,333) → 5.0 (24,000); the stated "Goldilocks" interval is only given for the MLP. No Transformer λ > 5.0 was tested, so the upper edge of the Transformer's zone is not established. Note also that the Transformer's optimum λ = 5.0 is far above GROKVERSE's `wd = 1.0`; in this paper λ = 1.0 with AdamW is a *suboptimal* Transformer setting (44,000 vs 24,000).

### 4.1 Depth (H1) and activation (H3) — numbers, for completeness

**Table 2** (depth; "Mean excludes DNF seeds"): depth 2 (w=256): seeds 0–4 ΔT = 32,000 / 20,000 / 26,000 / DNF / 210,000 → mean (4 seeds) 72,000 ✓ **± 85,536 ✗** [verifier 2026-09-02: the writer's ✓ was wrong for the std — from the four printed ΔT values the sample std (ddof = 1) is 92,130 and the population std (ddof = 0) is 79,787; the printed 85,536 reproduces from neither. The mean does reproduce.]; depth 4 (w=256): "DNF (all)" 0/5; depth 8 (residual, w=512): 48,000 / 24,000 / 28,000 / DNF / DNF → mean (3 of 5) 33,333 ± 12,858 ✓ (sample std). §5.2: "The relationship between depth and grokking is non-monotonic. Depth 2 is our baseline with a mean grokking delay of 72,000 steps over grokking seeds (4 of 5 grokked; one seed exhibited a very late grokking at step 212,000 and one DNF). Depth 4 represents a critical failure regime: all 5 seeds failed to grok within the 400,000-step budget."

**[OBSERVATION — verifier 2026-09-02, std convention.]** Recomputed from per-seed values: Table 3 (22,565 / 5,550), Table 9 (17,470 / 3,347) and Table 2 depth-8 (12,858) are *sample* stds (ddof = 1); Table 7's "0.0219 ± 0.0032" is a *population* std (ddof = 0; the sample std would be 0.0035); Table 2 depth-2 (85,536) matches neither. The stds in Tables 5, 6, 10 and Part C cannot be checked (no per-seed values printed). Quote the ± values as printed, but do not treat them as one consistent estimator.

**Table 10** ("H3-mul: Activation comparison on (a × b) mod 97 (depth-4 MLP, width 512, SGD, λ = 10⁻³, 5 seeds)") [verifier 2026-09-02 — numbers missing from the writer's note]: GELU 5/5, 24,400 ± 3,578; ReLU 5/5, 93,200 ± 47,045; Tanh 3/5, 102,667 ± 19,732. §5.8: "GELU grokks in 5/5 seeds (24,400 ± 3,578 steps); ReLU in 5/5 (93,200 ± 47,045)" — ratio "3.82×—close to the addition ratio of 4.32×" (93,200 / 24,400 = 3.82 ✓).

**Table 4** (activation; "All depth-4 MLPs, SGD, 5 seeds, 400,000 steps"): Sweep A (lr = 10⁻², λ = 2 × 10⁻³, width 256): GELU 0/5 DNF; ReLU 2/5, 266,000 ± 96,167; Tanh 3/5, 242,667 ± 134,288. Sweep B (lr = 3 × 10⁻², λ = 5 × 10⁻⁴, width 512): GELU 5/5, 45,600 ± 5,550; ReLU 5/5, 196,800 ± 89,728; Tanh 2/5, 188,000 ± 36,770. §5.4: "GELU grokks in all 5 seeds at 45,600 ± 5,550 steps, a 4.32× faster mean than ReLU". **[OBSERVATION]** Sweep-B GELU is the *same* run set as the H2 MLP (identical 45,600 ± 5,550). Also: a depth-4 width-256 GELU MLP fails 0/5 (Table 2, Table 4 A) while a depth-4 width-512 GELU MLP at lighter λ groks 5/5 (Table 3/4 B) — "depth-4 MLPs fail" (abstract) is a statement about one hyperparameter cell.

---

## 5. Limitations and open questions the authors name

> "Our study focuses on controlled analysis of grokking in modular arithmetic tasks (modulo 97). While this provides a clean and widely used testbed, extending these findings to broader datasets remains an important direction." — §6.3

> "Architectural comparisons require different optimizers and regularization regimes (e.g., SGD for MLPs, AdamW for Transformers) to ensure stable training. Although we carefully match configurations, fully isolating architecture from optimization is an open challenge." — §6.3

> "Finally, our results are based on finite training budgets and selected hyperparameter regimes. Further exploration of longer training horizons and wider hyperparameter spaces may refine the observed dynamics." — §6.3

> "Future directions include mechanistic interpretability analysis of learned representations Nanda et al. (2023), extending to other algorithmic tasks such as modular multiplication Power et al. (2022), studying the interaction between depth and regularization Liu et al. (2022), and linking grokking dynamics to measures such as loss landscape sharpness or gradient noise Thilak et al. (2022). An additional direction is to investigate whether the depth 4 failure regime generalizes and whether residual connections He et al. (2016) consistently enable stable grokking at depth." — §6.4

Limitations the paper does **not** name but that follow from its own tables **[OBSERVATION]**: Fourier analysis is n = 1 (seed 0); Part C per-seed values are not printed and conflict with Table 6/7 (§2.3 above); the two architectures differ in depth (1 vs 4), optimizer, learning rate, λ, normalisation (pre-LN vs none), activation of the FFN (GELU) and readout (mean-pool vs sum); parameter counts are not reported; the "prior estimate of 2.18×" is unreproducible from the paper; delays are quantised to 2,000 steps; no code is released.

---

## 6. Consequences for GROKVERSE

### 6.1 Side-by-side: what each study actually measures

| | Manir & Rupa 2026 | GROKVERSE (current repo) |
|---|---|---|
| Task, modulus | (a+b) mod **97**, also (a×b) mod 97 | (a+b) mod **113** (+ multiplication runs) |
| Train fraction | 0.20 (1,882 pairs), fixed split | 0.3 (un-accelerated) / 0.5 (Grokfast) |
| Transformer | encoder-only, d = 512, 4 heads, 1 layer, **pre-LN, GELU FFN, length-2 input, mean-pool readout** | Nanda recipe: d = 128, 4 heads, 1 layer, **no LN, ReLU, `a b =` input, readout at `=`** (`models/transformer.py`, `Config`) |
| MLP | depth **4** (H2) or 2 (baseline), width 512/256, **summed** embeddings `Emb(a)+Emb(b)`, GELU | depth **2** (one hidden layer), d_mlp = 512, **concatenated** embeddings `[W_E[a], W_E[b]]`, ReLU, shared `W_E` (`models/mlp.py`) |
| Optimizer | **SGD+momentum (MLP) vs AdamW (TF)** — different by design | **AdamW for both**, lr 10⁻³, β = (0.9, 0.98) |
| Weight decay | MLP 5×10⁻⁴ / 10⁻³; TF 1.0 / 5.0 | **1.0 for both** |
| Batch | full batch | **full batch** [verifier 2026-09-02: `train.py: _train_loop` takes one AdamW step per iteration on `logits_last(run.train_x)`, i.e. the entire training split; the writer left this as "(see train.py)"] |
| Steps / seeds | 400k–600k; 5 seeds (timing), 1 seed (Fourier) | budget 40k (un-acc.) / 8k (Grokfast), but runs **early-stop at the 0.95 test-acc crossing** (`--early-stop-acc 0.95`, `RESEARCH_SPEC` §3.9; 0.98 for the Grokfast README recipe), so the un-accelerated runs actually train ~8–10k steps; 3 seeds (both timing and Fourier) [verifier 2026-09-02: refined from "30k–40k"] |
| Grokking criterion | first step test acc ≥ 99 %, delay = T_grok − T_train (train ≥ 99 %) | test acc crossing 0.95 (0.99 for train), early-stopped at crossing (`RESEARCH_SPEC` §3.9); `config.THRESHOLD_SETS["primary"] = (0.99, 0.95)` ✓ |
| Fourier metric | **top-5** of "total Fourier energy" of W_E, **DC included** (freq 0 in TF's top-5), k/97−k merging unstated, seed 0 | **top-8** of frequency power, cos/sin pairs merged over k = 1..56, **constant mode excluded** from the denominator (`analysis/fourier.py: embedding_power_spectrum`), 3 seeds |
| Checkpoint for Fourier | "to full grokking" (unspecified) | at the 0.95 generalization crossing (early stop) — `analysis/compare.py` calls `dominant_frequencies(embeds[-1], p)` on the *last logged* snapshot of `embeddings.npy`, which for an early-stopped run is the crossing step [verifier 2026-09-02, checked against the code] |
| Result: concentration | TF 98.5 % vs MLP 74.7 % / 75.6 % | TF 0.73 ± 0.07 vs MLP 0.44 ± 0.01 (un-acc.), 0.59 vs 0.35 (Grokfast) |
| Result: timing direction | **MLP faster** (1.11×, 1.90×, 1.42×), TF variance 4–6× higher | **Transformer faster** in every seed (~1.3× un-acc., ~2.9× Grokfast), TF spread larger (±1196 vs ±410) |

### 6.2 What is already published — GROKVERSE must NOT present these as new

1. **"The Transformer's grokked embedding is more Fourier-concentrated than the MLP's."** Published: §5.7 / Table 8 / Figure 7 / §6.1 item 6 — "the Transformer concentrates 98.5% of its embedding energy in just 5 frequencies, compared to 74.7% for MLP-GELU and 75.6% for MLP-ReLU." GROKVERSE's 0.73 vs 0.44 is a *replication of the direction* with a different metric, modulus, models, optimizer, checkpoint and more seeds — it must be introduced as "consistent with Manir & Rupa (2026, Table 8)", never as the first observation. The `README.md` / `RESULTS.md` §3 sentence "the transformer converges to a markedly sparser Fourier circuit than a 2-layer MLP" and `RESEARCH_SPEC.md` §1 "one nobody has told us the answer to" are now factually out of date and need the citation.
2. **The interpretation that attention is what makes the Transformer sparser.** Published: §5.7 Analysis — "suggests that self-attention acts as an implicit sparsity-promoting mechanism in the frequency domain, enforcing a cleaner algorithmic solution." GROKVERSE's `RESULTS.md` §3 sentence "The transformer's attention bias ... steers it toward a cleaner trig-identity-style circuit" is the same untested conjecture; attribute it, and note that neither paper has tested it causally.
3. **"Both MLPs and Transformers converge to sparse Fourier embeddings; sparse Fourier representations are universal across these architectures."** Published: §5.7 Analysis "First, sparse Fourier representations are a universal property of grokked models on modular addition, appearing in both MLPs and Transformers regardless of activation function." Also the ring/circle plots of embeddings projected on top-2 Fourier directions (Figure 7 d–f) — GROKVERSE's PCA ring figures are not novel as a qualitative observation for an MLP.
4. **"The Transformer-vs-MLP grokking-time gap is not a fixed constant; it moves with hyperparameters/regularization."** Published: abstract (2), contribution 2, §5.3 (2.18× → 1.11×), §5.5 Part C (1.90×), §5.8 (1.42×), §6.2. GROKVERSE's own observation that its ratio moves from ~2.9× to ~1.3× between settings is the same message and must be framed as agreeing with Manir & Rupa.
5. **"The Transformer's grokking time has much larger seed-to-seed variance than the MLP's."** Published: §5.3 "4.1× higher variance", §6.1 "6× higher variance at its own optimum", attributed to the slingshot mechanism. GROKVERSE's ±1196 vs ±410 is consistent, not new.
6. **A grokking MLP on modular addition and multiplication, with weight-decay Goldilocks behaviour, a weight-norm threshold at grokking, and GELU ≫ ReLU speed** — all published (Tables 2–7, 9–10). If GROKVERSE reports weight norms at the crossing, cite Table 7.

### 6.3 What the paper does NOT establish — where GROKVERSE's contribution still lives (state it narrowly)

1. **Direction of the timing gap is protocol-dependent, and the paper's direction is the opposite of GROKVERSE's.** Manir & Rupa: MLP faster in every multi-seed comparison (with per-architecture optimizers, depth-4 MLP, p = 97, 20 %). GROKVERSE: Transformer faster in every seed (same optimizer and λ for both, depth-2 MLP, p = 113, 30 %). Therefore GROKVERSE must (a) never claim "Transformers grok faster than MLPs" as an architectural fact, (b) explicitly report that the published comparison found the reverse, and (c) treat the *sign flip between protocols* as the interesting, un-explained datum — it is direct evidence for the paper's own thesis that the gap is a confound.
2. **Same-optimizer / same-λ / shared-embedding comparison.** The authors name it as open: "fully isolating architecture from optimization is an open challenge" (§6.3); they compare SGD-vs-AdamW, depth-1-vs-depth-4. GROKVERSE's design (AdamW, wd = 1.0, identical `W_E` shape, identical split/seeds) is exactly the control they did not run. This is a legitimate but *incremental* contribution; it does not overturn anything.
3. **Seed-robustness of the concentration gap.** Table 8 is one seed per model with no variance. GROKVERSE has 3 seeds with non-overlapping per-seed ranges (0.66–0.79 vs 0.43–0.45); WP-5's ≥ 8 paired seeds would be the first multi-seed estimate. Say "first multi-seed estimate", not "first observation".
4. **Whether the concentration gap is a metric artifact (harmonic aliasing, RESEARCH_SPEC H3), the cardinality-matched top-m control, the random-set null, DC handling.** None of it is in the paper; its top-5 metric is fixed a priori exactly as GROKVERSE's top-8 was (§3.2 audit). Note two comparability traps: (i) the paper's 98.5 % **includes the DC mode** (frequency 0 is in the Transformer's top-5, attributed to mean-pooling), whereas GROKVERSE's metric removes the constant mode from the denominator — the numbers are not on the same scale and must not be placed in one table without saying so; (ii) the paper's Transformer has no `=` token and mean-pools, so its DC component has no counterpart in a Nanda-style model.
5. **Structure over time / at-crossing vs post-convergence (RESEARCH_SPEC H4, §3.9).** The paper has one post-grokking snapshot and no time course; the checkpoint is not even specified. Open.
6. **Per-neuron mechanism, wave-form fitting (H1/H2), causal ablations (H5), attention statistics (§3.4 audit).** All **[NOT FOUND IN SOURCE]**; the authors list "mechanistic interpretability analysis of learned representations" as *future work* (§6.4). GROKVERSE's restricted/excluded-loss reproduction, attention measurements and planned ablations are not pre-empted by this paper.
7. **Parameter-matched control (§3.6 audit).** The paper reports no parameter counts and matches nothing but "width 512". Open.
8. **p = 113 / Nanda-recipe models / Grokfast.** All **[NOT FOUND IN SOURCE]** in the paper. GROKVERSE's numbers are in a different regime; do not import the paper's λ-optimum (5.0) or its ratios as expectations for p = 113 without re-measuring.

### 6.4 Concrete wording obligations

- `README.md` line 31 / `RESULTS.md` §3 finding 1: prefix with "As Manir & Rupa (2026, arXiv:2603.25009, Table 8) report at p = 97 for a single seed (98.5 % vs ≈75 % top-5, DC included), …" and change "markedly sparser" claims from *discovery* to *replication under a different protocol*.
- `RESULTS.md` §3 finding 2 (speed): add "Manir & Rupa (2026) find the opposite ordering — MLP faster by 1.11×–1.90× — under per-architecture optimizers at p = 97; the direction of the timing gap is therefore protocol-dependent, which is itself consistent with their central claim that the gap is an optimizer/regularization confound."
- `RESEARCH_SPEC.md` §1: replace "one nobody has told us the answer to" with the citation; keep sub-questions 2 (basis / artifact) and 3 (timing link) as the open questions — they are genuinely not in the paper. Add the DC-inclusion difference to §5 (metric definitions) so the two metrics are never confused.
- `RESEARCH_SPEC.md` §4 H3: the paper's fixed k = 5 vs GROKVERSE's fixed k = 8 strengthens the case that a data-driven / harmonic-family metric is needed; cite the paper's Table 8 frequency lists ([6, 12, 21, 37, 46] for MLP-GELU vs [0, 16, 29, 32, 34] for the Transformer) as published evidence that the two families do not even share frequencies — but note "seed 0" and "p = 97" every time.
- Any statement about weight decay: the paper's Transformer optimum is λ = 5.0 under AdamW and λ = 1.0 is documented there as 1.83× slower; GROKVERSE's wd = 1.0 (Nanda's value at p = 113) is not the paper's optimum and should not be described as "canonical" without saying whose canon.
- Cite as: Manir, S. B. & Rupa, A. P. (2026). *A Systematic Empirical Study of Grokking: Depth, Architecture, Activation, and Regularization.* arXiv:2603.25009v1.

### 6.5 Things to re-check before quoting the paper's Transformer numbers

The Part C Transformer mean (50,800 ± 38,745, 5 seeds, λ = 5.0) vs the 3-seed screen at the same λ (24,000 ± 10,583) vs the single-seed run (12,000) — §2.3 [OBSERVATION]. If GROKVERSE ever tabulates the paper's "1.90×", carry this caveat with it, or quote the 1.11× (Table 3, per-seed data printed) and the 1.42× (Table 9, per-seed data printed) instead.

---

## 7. Verifier log [verifier 2026-09-02]

**Sources re-fetched.** `arxiv.org/abs/2603.25009` (v1 only, "[v1] Thu, 26 Mar 2026 04:16:01 UTC (749 KB)", cs.LG); `arxiv.org/html/2603.25009` (8 extractions: abstract + contributions + §3; §4; §5.1–5.3 + Tables 2–3 + Fig. 3; §5.4 + §5.8 + Tables 4, 9, 10; §5.5 + Tables 5–6; §5.6–5.7 + Tables 7–8; §6–7 + A.1; author block); `arxiv.org/pdf/2603.25009` (18 pages, `pdftotext -layout`, 55 kB of text, grepped for every phrase the note makes a claim about).

**Checked and found correct (no change).** Title, authors, affiliations ("University of Maryland, Baltimore County" / "Howard University", footnote "Equal contribution."), date, version; absence of any code URL (`github`/`http` absent from the PDF text); §3.1 Eq. (1) and the Power-et-al. sentence; §3.2 (1,882 / 7,527) and the §5.1 "1,883" discrepancy; §3.3 and all nine Table 1 rows; §3.4 definitions; §4.1 verbatim (incl. "97 × 512", "2 × 512", "norm_first=True", "mean-pooling", "λ = 1.0 follows Liu et al. (2022)"); §4.2–4.4 verbatim; §5.3 Setup/Results/Analysis verbatim, all 10 Table 3 rows, Figure 3 caption; §5.5 Setup grids, "MLP sweep", "Transformer sweep", "Part C", "Analysis" verbatim (incl. the unpunctuated "is narrow a factor-of-two step"), all rows of Tables 5 and 6, Figure 5 caption; §5.6 Setup/Results verbatim, all rows of Table 7; §5.7 Goal/Setup/Results/Analysis verbatim, all rows of Table 8 (frequency lists 6,12,21,37,46 / 6,7,11,16,21 / 0,16,29,32,34; 74.7 / 75.6 / 98.5 %; delays 26,000 / 80,000 / 12,000), Figure 7 caption; §5.8 Setup/H2-mul/Analysis verbatim, all 10 Table 9 rows; §6.1 items 2 and 6, §6.2, §6.3, §6.4, §7, A.1 verbatim; Table 4 (all six cells) and the "4.32×"; every ratio the paper prints (1.11, 1.90, 1.42, 1.83, 4.1, 6, 4.32, 3.82, 6.9, 3.1, 0.69, 5,000, CV 14.5 %) reproduces from the printed numbers except 0.91 (see §2.6); the writer's ✓ marks on Table 3 and Table 9 means/stds; every **[NOT FOUND IN SOURCE]** label in §1.3, §1.5, §1.6, §2.2, §2.3, §3.2, §3.3, §6.3 (parameter counts, AdamW betas, schedule, init, eval frequency, tied unembedding, Part C per-seed values, k ≠ 5, energy definition, one-sidedness, checkpoint, ReLU λ, attention/neuron/ablation/time-course analyses, multiplication Fourier analysis) — all confirmed absent. No **[FROM MEMORY - UNVERIFIED]** tag was used by the writer, and none was needed.

**Corrected.** (1) §4.1: the ✓ on Table 2's depth-2 "72,000 ± 85,536" — the std does not reproduce from the printed per-seed values. (2) §6.1 "Batch" row: was "(see `train.py`)", now the verified "full batch". (3) §6.1 "Steps / seeds" row: "30k–40k" was the budget, not the trained length; runs early-stop at ~8–10k.

**Added (load-bearing, was in the source but missing).** §1.10 canonical-run numbers (T_train 1,000 / T_grok 33,000 / ΔT 32,000; "near chance (19.9%)"); §1.2 the §5.8 "otherwise identical hyperparameters" sentence; §2.1 the §6.1 sentence defining "matched"; §2.6 the 0.91× arithmetic slip and the §5.8 "key finding" sentence; §3.2 gloss on the ReLU λ; §3.4 the λ–lr-product continuation and Figure 6(b); §4.1 Table 10 numbers and the std-convention observation; §6.1 the `compare.py` `embeds[-1]` detail.

**Repository claims in §6 checked against code.** `models/transformer.py` (no LayerNorm, ReLU, `[a, b, =]`, readout `[:, -1, :]`), `models/mlp.py` (shared `W_E [p, d]`, `torch.cat([W_E[a], W_E[b]])`, one ReLU hidden layer of `d_mlp = 512`), `config.py` (`p = 113`, `d_model = 128`, `n_heads = 4`, `lr = 1e-3`, `weight_decay = 1.0`, `beta1/beta2 = 0.9/0.98`, `THRESHOLD_SETS["primary"] = (0.99, 0.95)`), `train.py` (`torch.optim.AdamW` for every arch, full-batch loop, `--early-stop-acc`), `analysis/fourier.py` (`embedding_power_spectrum`: cos/sin pairs merged over k = 1..56, `total = freq_power.sum()` excludes the constant mode; `dominant_frequencies(threshold = 0.9, max_k = 8)` with the cap binding), `analysis/compare.py` (`embeds[-1]`), `README.md` line 31, `RESULTS.md` §3 lines 70–71 and the two result tables (0.59 ± 0.06 / 0.35 ± 0.003; 0.73 ± 0.07 / 0.44 ± 0.01; 750 ± 97 / 2163 ± 72; 7676 ± 1196 / 9883 ± 410), `docs/RESEARCH_SPEC.md` §1 line 36 ("one nobody has told us the answer to"), §3.2, §3.4, §3.6, §3.9, §4 H1–H5, §5, WP-5 ("≥8 paired seeds (target 10)"). All references in §6 resolve to the stated places; no discrepancy found beyond the two corrected rows.
