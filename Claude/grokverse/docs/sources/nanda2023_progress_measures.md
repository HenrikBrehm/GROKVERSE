# Source note — Nanda, Chan, Lieberum, Smith, Steinhardt (2023), "Progress measures for grokking via mechanistic interpretability"

arXiv:2301.05217 (v1 2023-01-12, v2 2023-01-13, v3 2023-10-19; comments field: "10 page main body, 2 page references, 24 page appendix"; ICLR 2023). Written 2026-09-02 for GROKVERSE WP-0/WP-1 (docs/RESEARCH_SPEC.md §3.1, §3.2).

## Sources actually fetched (and how)

| Source | URL | Status |
|---|---|---|
| Paper, full text (v3 HTML) | https://arxiv.org/html/2301.05217 | fetched; HTML saved locally and stripped to text — every quote below was grepped from that text |
| Paper PDF | https://arxiv.org/pdf/2301.05217 | fetched (2.9 MB) but the fetch tool could not decode it; no PDF text extractor available locally — **not used** |
| Paper abstract page | https://arxiv.org/abs/2301.05217 | fetched (metadata only) |
| Code link named in the paper | https://neelnanda.io/grokking-paper | fetched; it is a landing page linking OpenReview + the GitHub repo, not a notebook |
| GitHub repo root | https://api.github.com/repos/neelnanda-io/Grokking/contents/ | fetched; files: `.gitignore`, `README.md` (305 B), `_Home.py`, `grokking.py` (8,327 B), `requirements.txt`, `streamlit_experiment.py`; dirs `figs/`, `pages/`, `saved_runs/` |
| Repo README | https://raw.githubusercontent.com/neelnanda-io/Grokking/main/README.md | fetched verbatim: "This is a dump of relevant saved model weights and loss curves for the notebook A Mechanistic Interpretability Analysis of Grokking: https://colab.research.google.com/drive/1F6_1_cWXE5M7WocUcpQWp3v8z4b1jL20#scrollTo=rk8LtmzQfyDC&uniqifier=8" |
| `grokking.py` | https://raw.githubusercontent.com/neelnanda-io/Grokking/main/grokking.py | fetched in full: Streamlit page = Plotly helpers + Fourier basis + loads `saved_runs/mod_addition_frac_train_sweep.pth`. **No model, no training loop, no restricted/excluded loss, no key_freqs.** |
| `pages/3_page_3_grokky.py` | https://raw.githubusercontent.com/neelnanda-io/Grokking/main/pages/3_page_3_grokky.py | fetched in full (8,330 B): same content as `grokking.py` (Fourier basis + loss-curve plot). |
| `Grokking_Analysis.ipynb` (name given in the task) | https://raw.githubusercontent.com/neelnanda-io/Grokking/main/Grokking_Analysis.ipynb | **HTTP 404 — no such file in the repo.** |
| The actual analysis notebook (Colab) | https://colab.research.google.com/drive/1F6_1_cWXE5M7WocUcpQWp3v8z4b1jL20 | Colab page needs a Google login (fetch returned the sign-in page). Retrieved instead through the Drive export `https://drive.google.com/uc?export=download&id=1F6_1_cWXE5M7WocUcpQWp3v8z4b1jL20` → 41.2 MB `.ipynb`, 346 cells **with saved outputs**. Cell numbers below refer to this file (0-based index in `nb['cells']`). `https://bit.ly/neelgrokking` (blog link) redirects to the same Colab id. |
| Blog write-up | https://www.alignmentforum.org/posts/N6WM6hs7RQMKDhYjB/a-mechanistic-interpretability-analysis-of-grokking | fetched (page too long for the fetch tool; full text obtained from the greaterwrong mirror https://www.greaterwrong.com/posts/N6WM6hs7RQMKDhYjB/a-mechanistic-interpretability-analysis-of-grokking) |
| Blog index page | https://www.neelnanda.io/blog/interlude-a-mechanistic-interpretability-analysis-of-grokking | fetched; contains only the link "[Colab notebook with code + technical details](https://bit.ly/neelgrokking)" |
| Secondary code (Neel Nanda's later demo, NOT the paper's code) | https://raw.githubusercontent.com/TransformerLensOrg/TransformerLens/main/demos/Grokking_Demo.ipynb | fetched (5.9 MB, 140 cells). It contains functions literally named `get_restricted_loss` / `get_excluded_loss`. Used only as a cross-check; it is a *different run* (25,000 epochs, `seed = 999`, `DATA_SEED = 598`) and its `main` has been edited since (a cell comment says "Derive the key frequencies from this run's embedding instead of hardcoding values from a past run"). |

Notation: quotes are verbatim; LaTeX in the HTML was transcribed to plain notation (`w_k`, `W_L`, `cos(w_k(a+b))`) without changing words. `[NOT FOUND IN SOURCE]` = looked for and absent. Nothing in this note is from memory.

---

## 1. Model and training setup

**Paper, Section 3 "Setup and Background":**

> "The input to the model is of the form "a b =", where a and b are encoded as P-dimensional one-hot vectors, and = is a special token above which we read the output c. In our mainline experiment, we take P=113 and use a one-layer ReLU transformer, token embeddings with d=128, learned positional embeddings, 4 attention heads of dimension d/4=32, and n=512 hidden units in the MLP. In other experiments, we vary the depth and dimension of the model."

> "We did not use LayerNorm or tie our embed/unembed matrices."

> "Our mainline dataset consists of 30% of the entire set of possible inputs (that is, 30% of the 113·113 pairs of numbers mod P). We use full batch gradient descent using the AdamW optimizer (Loshchilov & Hutter, 2017) with learning rate γ=0.001 and weight decay parameter λ=1. We perform 40,000 epochs of training. As there are only 113·113 possible pairs, we evaluate test loss and accuracy on all pairs of inputs not used for training."

> "After around 10,000 epochs, the network generalizes and test accuracy increases to near 100%."

**Paper, Figure 2 caption:** "The train and test accuracy (left) and train and test loss (right) of one-layer transformers on the modular addition task described in Section 3, over 5 random seeds."

**Paper, Appendix A "Mathematical Structure of the Transformer":**

> "We denote our hyperparameters as follows: d_vocab=113 is the size of the input and output spaces (treating '=' separately), d_model=128 is the width of the residual stream (i.e. embedding size), d_head=32 is the size of query, key and value vectors for a single attention head, and d_mlp=512 is the number of neurons."

> "W_in and b_in for the input linear map of the MLP layer; W_out and b_out for the output linear map of the MLP layer; and W_U (unembedding layer). Note that we do not have biases in our embedding, attention layer or unembedding, and we do not tie the matrices for the embedding/unembedding layers."

> "Note that loss is only calculated from the logits on the final token, and information only moves between tokens during the attention layer, so our variables from the end of the attention layer onwards only refer to the final token."

Equations (Appendix A, unnumbered): `x_i^(0) = W_E t_i + p_i`; `A^j = softmax(x^(0)T W_K^jT W_Q^j x_2^(0))`; `x^(1) = [Σ_j W_O^j W_V^j (x^(0)·A^j)] + x_2^(0)`; `MLP = ReLU(W_in x^(1))`; `x^(2) = W_out N + x^(1)`; `Logits = W_U x^(2)`.

**Appendix A.1 "Empirical Model Simplifications":** "The attention paid from '=' to itself is trivial. In practice, the average attention paid is 0.1% to 0.4% for each head, and ablating this does not affect model performance at all." — "The skip connection around the MLP layer is not important … loss goes from 2.4·10⁻⁷ to 9.12·10⁻⁷ and 7.25·10⁻⁷ respectively" (zero / mean ablation). "zero ablating attention heads increases loss to 24.3, while zero ablating the skip connection around the attention heads increases loss to 19.1". Consequence stated there: "Logits ≈ W_U W_out MLP, which we denote as W_L = W_U W_out".

**Betas, LR schedule, seed values:** `[NOT FOUND IN SOURCE]` in the paper (grep for "beta", "0.98", "schedule", "warm" over the full text: no hits except the attention-fit coefficient β^j).

**Colab notebook, cell 69 (markdown "Architecture"):** "It's a 1 layer transformer, with no layer norm and learned positional embeddings. d_model = 128, n_heads = 4, d_head=32, d_mlp=512. Input format is `x|y|=`, d_vocab=114 (integers from 0 to p−1 and =). It was trained with full batch training, with 0.3 of the total data as training data. It is trained with AdamW, with lr=10⁻³ and very high weight decay (wd=1)".

**Colab cell 71 (hyper-parameters, verbatim):**
```python
lr=1e-3 #@param
weight_decay = 1.0 #@param
p=113 #@param
d_model = 128 #@param
fn_name = 'add' #@param ['add', 'subtract', 'x2xyy2','rand']
frac_train = 0.3 #@param
num_epochs = 50000 #@param
save_models = False #@param
save_every = 100 #@param
# Stop training when test loss is <stopping_thresh
stopping_thresh = -1 #@param
seed = 0 #@param

num_layers = 1
batch_style = 'full'
d_vocab = p+1
n_ctx = 3
d_mlp = 4*d_model
num_heads = 4
assert d_model % num_heads == 0
d_head = d_model//num_heads
act_type = 'ReLU' #@param ['ReLU', 'GeLU']
# batch_size = 512
use_ln = False
```

**Colab cell 74 (split, verbatim) and its saved output:**
```python
def gen_train_test(frac_train, num, seed=0):
    # Generate train and test split
    pairs = [(i, j, num) for i in range(num) for j in range(num)]
    random.seed(seed)
    random.shuffle(pairs)
    div = int(frac_train*len(pairs))
    return pairs[:div], pairs[div:]
```
Output: `3830 8939` (train / test pairs; 3830 = int(0.3·12769)).

**Colab cell 76 (optimizer, verbatim):** `optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98))` and `scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(step/10, 1))` — i.e. a 10-step linear warm-up, one optimizer step per epoch on the whole train set (`train_loss = full_loss(model, train)` … `train_loss.backward(); optimizer.step(); scheduler.step()`).

**Colab cell 10 (model code):** `Embed`: `W_E = randn(d_model, d_vocab)/sqrt(d_model)`; `Unembed`: `W_U = randn(d_model, d_vocab)/sqrt(d_vocab)`; `PosEmbed`: `W_pos = randn(max_ctx, d_model)/sqrt(d_model)`; attention `W_K/W_Q/W_V = randn(num_heads, d_head, d_model)/sqrt(d_model)`, `W_O = randn(d_model, d_head*num_heads)/sqrt(d_model)`, causal `tril` mask, scores divided by `sqrt(d_head)`, **no attention biases**; `MLP` has `W_in, b_in, W_out, b_out`; a `LayerNorm` class exists but `use_ln = False`.

**Read-out position and output classes — Colab cell 12 `full_loss`:** "# Take the final position only / logits = model(data)[:, -1]"; cell 85: `original_logits = model(all_data)[:, -1]` then "# Remove equals sign from output logits / original_logits = original_logits[:, :-1]". Loss is `cross_entropy_high_precision` (log-softmax in float64; cell 12 comment: "log_softmax has a float32 underflow on overly confident data").

**Checkpoints / analysed model — Colab cell 204:** "This is an analysis of the modular addition transformer studied above during training, checkpoints taken every 100 epochs, from epoch 0 to 50K (the model above was taken at 40K)". Cell 91: "The model's training curve exhibits clear grokking, from approximately epoch 8000 to epoch 14000." Cell 85 output: `Original loss: 2.4122030061370614e-07` (matches the paper's 2.4·10⁻⁷ / 2.41·10⁻⁷).

**TransformerLens demo (different run, for comparison):** `p = 113; frac_train = 0.3; lr = 1e-3; wd = 1.; betas = (0.9, 0.98); num_epochs = 25000; checkpoint_every = 100; DATA_SEED = 598`; `HookedTransformerConfig(n_layers=1, n_heads=4, d_model=128, d_head=32, d_mlp=512, act_fn="relu", normalization_type=None, d_vocab=p+1, d_vocab_out=p, n_ctx=3, seed=999)`; all `b_` parameters frozen ("Disable the biases"); no LR scheduler; markdown: "Training the model with full batch training rather than stochastic gradient descent. We do this so to make training smoother and reduce the number of slingshots."

**Summary for item 1:** 1 layer; d_model 128; 4 heads; d_head 32; d_mlp 512; ReLU; no LayerNorm; learned positional embedding; tokens `[a, b, =]` with d_vocab 114 (`=` is token 113); loss/logits read at the last position (`=`), `=` class dropped from the logits; p = 113; train fraction 0.3 (3830 pairs); full-batch AdamW, lr 1e-3, wd 1.0; betas (0.9, 0.98) **from code only**; 40,000 epochs in the paper (Colab run went to 50K, analysed at 40K); 5 seeds in Figure 2 / Appendix C.2.1; Colab `seed = 0`.

---

## 2. Fourier basis and the objects it is applied to

**Paper §3.1:** frequencies "w_k = 2kπ/P, k ∈ ℕ". The basis normalisation is `[NOT FOUND IN SOURCE]` in the paper body; it is in the code and blog:

**Colab cell 121 (verbatim code; identical in `grokking.py` and `pages/3_page_3_grokky.py`):**
```python
fourier_basis = []
fourier_basis.append(torch.ones(p)/np.sqrt(p))
fourier_basis_names = ['Const']
# Note that if p is even, we need to explicitly add a term for cos(kpi), ie
# alternating +1 and -1
for i in range(1, p//2 +1):
    fourier_basis.append(torch.cos(2*torch.pi*torch.arange(p)*i/p))
    fourier_basis.append(torch.sin(2*torch.pi*torch.arange(p)*i/p))
    fourier_basis[-2]/=fourier_basis[-2].norm()
    fourier_basis[-1]/=fourier_basis[-1].norm()
    fourier_basis_names.append(f'cos {i}')
    fourier_basis_names.append(f'sin {i}')
fourier_basis = torch.stack(fourier_basis, dim=0).to('cuda')
```
Row layout: 0 = constant, then `(cos k, sin k)` at rows `2k−1, 2k` for k = 1..56 — the same layout as GROKVERSE `fourier.fourier_basis`. Cell 121 markdown: "Each term is scaled to have norm 1." (its next sentence says "the constant term is scaled by 1/p, the rest by √2/p" — that is the notebook's own wording; the code divides by the actual norm, i.e. 1/√p and √(2/p)). Cell 122: "Crucially, this is an orthonormal basis".

**Blog ("Reverse Engineering the Algorithm" preamble):** "we can also take a basis of p cosine and sine waves, F∈R^{p×p}, where F_0=(1,1,...,1) is the constant vector, and F_{2k−1}=cos(2π/p kx) and F_{2k}=sin(2π/p kx) are the cosine and sine wave of frequency w=2π/p k … for k=1,...,(p−1)/2. … If we normalise these rows, we get an orthonormal basis of cosine and sine waves (so F⁻¹=Fᵀ). We refer to these normalised waves as Fourier Components and this overall basis as the 1D Fourier Basis." — "If we apply this change of basis to both the input space for x and for y we apply a 2D DFT … terms of the form sin(w₁x)cos(w₂y) (ie the outer product of each pair of rows in F) form an orthogonal basis of p² vectors (henceforth referred to as the 2D Fourier Basis)."

**2D transform, Colab cell 16 (verbatim):**
```python
def fft2d(mat):
    # Converts a pxpx... or batch x ... tensor into the 2D Fourier basis.
    # Output has the same shape as the original
    shape = mat.shape
    mat = einops.rearrange(mat, '(x y) ... -> x y (...)', x=p, y=p)
    fourier_mat = torch.einsum('xyz,fx,Fy->fFz', mat, fourier_basis, fourier_basis)
    return fourier_mat.reshape(shape)
```

**Which objects are transformed (paper):**
- `W_E`: §4.1 "We apply a Fourier transform along the input dimension of the embedding matrix W_E then compute the ℓ2-norm along the other dimension; results are shown in Figure 3. We plot only the components for the first 56 frequencies, as the norm of the components for frequencies k and P−k are symmetric." Colab cell 220: `W_E_fourier = (model.embed.W_E[:, :-1] @ fourier_basis.T)` (the `=` column is dropped), squared and summed over the residual dimension.
- Neuron activations over the (a,b) grid (2D): Colab cell 172 "# Center the neurons to remove the constant term … fourier_neuron_acts = fft2d(neuron_acts_centered)"; cell 173: "we first center the activation to have mean zero - these are ReLU neurons and are always non-negative". Paper §4.3 / Figure 5 use the same.
- Logits (2D over inputs): §4.1 "we represent the logits in the 2D Fourier basis over the inputs, then take the ℓ2-norm over the output dimension"; §4.4 "we take a 2D DFT on the 113·113·113 logit matrix over all 113·113 pairs of inputs to get the logits in the Fourier basis, then set various frequencies in this basis to 0."
- Neuron-logit map `W_L = W_U W_out` (1D over the logit axis): §4.2 "We perform a discrete Fourier transform (DFT) on the logit axis of W_L"; Colab cell 199: `W_logit = W_U @ W_out; imshow_div(fourier_basis @ W_logit, …)` with the note (cell 198) "Here we are applying the 1D Fourier transform to the *output* space, not to the input space."

---

## 3. How key frequencies are determined, and how many

**Paper §4.1 (embedding):** "The embedding matrix W_E is sparse in the Fourier basis–it only has significant nonnegligible norm at 6 frequencies. Of these frequencies, only 5 appear to be used significantly in later parts of the model (corresponding to k∈{14,35,41,42,52}). We dub these the key frequencies of the model."

**Paper Appendix C.2 (the rule actually used for the progress measures, verbatim):** "Note that in general, while all models learn to use variants of the modular arithmetic algorithm, they use a varying number of different key frequencies. In order to find the key frequencies to calculate the excluded and restricted loss, we perform a DFT on the neuron-logit map W_L, then take the frequencies with nontrivial coefficients." Footnote 3: "One method for getting a general (model-independent) progress measure for this task is to compute the excluded loss for each of the 56 unique frequencies and then take the max. We omit the plots for this variant of the excluded loss as they are broadly similar."

**Paper Appendix C.2.1 (other seeds):** "The matrices are sparse in the Fourier basis, enabling us to identify 3 or 4 key frequencies for each of the seeds. Again, note that the specific frequencies differ by seed." … "we 'read off' the MLP activations in the 6 or 8 directions corresponding to the key frequencies". Table 3 lists per seed: seed 1 → k ∈ {2, 9, 19, 31}; seed 2 → {40, 44, 53}; seed 3 → {31, 45, 49, 52}; seed 4 → begins with 17, 32 (rest of that table not extracted).

**Numeric threshold for "nontrivial coefficients":** `[NOT FOUND IN SOURCE]` — the paper gives none.

**Count: measured, not fixed.** Mainline 5; other seeds 3–4 (above); Appendix C.2.3 / Table 5 "We list their key frequencies" per model.

**Colab (how the notebook computes `key_freqs` — from neuron activations, not from W_E or W_L), cell 175 markdown + cell 176 code (verbatim):** "For each neuron, we find the frequency where the linear and quadratic terms of that frequency explains the largest amount of the neuron activation's variance."
```python
fourier_neuron_acts_square = fourier_neuron_acts.reshape(p, p, d_mlp)
neuron_freqs = []
neuron_frac_explained = []
for ni in range(d_mlp):
    best_frac_explained = -1e6
    best_freq = -1
    for freq in range(1, p//2):
        # We extract the linear and quadratic fourier terms of frequency freq,
        # and look at how much of the variance of the full vector this explains
        # If neurons specialise into specific frequencies, one frequency should
        # have a large value
        frac_explained = (extract_freq_2d(fourier_neuron_acts_square[:, :, ni], freq).pow(2).sum()/
                          fourier_neuron_acts_square[:, :, ni].pow(2).sum()).item()
        if frac_explained > best_frac_explained:
            best_freq = freq
            best_frac_explained = frac_explained
    neuron_freqs.append(best_freq)
    neuron_frac_explained.append(best_frac_explained)
neuron_freqs = np.array(neuron_freqs)
neuron_frac_explained = np.array(neuron_frac_explained)
key_freqs, neuron_freq_counts = np.unique(neuron_freqs, return_counts=True)
```
with `extract_freq_2d` (cell 14): `index_1d = [0, 2*freq-1, 2*freq]` → the 3×3 block {const, cos_k, sin_k}×{const, cos_k, sin_k}. Note `range(1, p//2)` = 1..55 (k = 56 is never a candidate). Cell 179: `neuron_freqs[neuron_frac_explained < 0.85] = -1.` (cluster membership threshold 0.85; `key_freqs` itself is `np.unique` **before** this threshold). Cell 180 saved output:
```
Cluster 0: freq 14. 44 neurons
Cluster 1: freq 35. 93 neurons
Cluster 2: freq 41. 145 neurons
Cluster 3: freq 42. 87 neurons
Cluster 4: freq 52. 64 neurons
Cluster 5: freq -1. 79 neurons
```
(44+93+145+87+64 = 433 = the paper's "433 (84.6%)"; 79 = the "always firing" cluster.) `key_freqs` is computed once from the 40K-epoch model and reused, unchanged, for every checkpoint in the training-dynamics section (`get_metrics` loops `full_run_data['state_dicts']` with the global `key_freqs`).

**TransformerLens demo (current `main`, cell 70):** `fourier_norms = (fourier_basis @ W_E).norm(dim=-1); key_freq_indices = [i for i, norm in enumerate(fourier_norms) if i > 0 and norm > fourier_norms.max() / 4]; key_freqs = sorted({(i + 1) // 2 for i in key_freq_indices})` — an embedding-norm rule with a max/4 threshold; comment: "Derive the key frequencies from this run's embedding instead of hardcoding values from a past run — they vary with seed and training length." Cell 72 still hardcodes `fourier_basis[[34, 50, 64, 94]]` as "Cos of key freqs"; in that notebook's basis order (Const, Sin1, Cos1, Sin2, Cos2, …) these indices are k = 17, 25, 32, 47 (derived here from the index rule, not stated in the notebook).

**Summary for item 3:** three different rules appear in the sources — (a) W_E Fourier norms (paper §4.1, 6 → 5), (b) DFT of W_L, "nontrivial coefficients" (paper App. C.2, the rule for restricted/excluded), (c) neuron-cluster argmax over the 3×3 block (Colab). Count is measured per model (5 / 3 / 4 / 4 …); no numeric threshold is published.

---

## 4. Trigonometric identities and the logit formula

**Paper §3.1 (verbatim, three bullets):**
> "Given two one-hot encoded tokens a,b map these to sin(w_k a), cos(w_k a), sin(w_k b), and cos(w_k b) using the embedding matrix, for various frequencies w_k=2kπ/P, k∈ℕ."
> "Compute cos(w_k(a+b)) and sin(w_k(a+b)) using the trigonometric identities: cos(w_k(a+b)) = cos(w_k a)cos(w_k a) − sin(w_k a)sin(w_k b); sin(w_k(a+b)) = sin(w_k a)cos(w_k b) + cos(w_k a)sin(w_k b). In our networks, this is computed in the attention and MLP layers."  (the "cos(w_k a)cos(w_k a)" is a typo in the source; §4.2 restates it as cos(w_k a)cos(w_k b))
> "For each output logit c, compute cos(w_k(a+b−c)) using the trigonometric identity: cos(w_k(a+b−c)) = cos(w_k(a+b))cos(w_k c) + sin(w_k(a+b))sin(w_k c).  (1)  This is a linear function of the already-computed values cos(w_k(a+b)), sin(w_k(a+b)) and is implemented in the product of the output and unembedding matrices W_L."
> "The unembedding matrix also adds together cos(w_k(a+b−c)) for the various ks. This causes the cosine waves to constructively interfere at c*=a+b mod p (giving c* a large logit), and destructively interfere everywhere else".

**Paper §4.2, eq. (2) and (3):**
> "W_L = Σ_{k∈{14,35,41,42,52}} cos(w_k) u_kᵀ + sin(w_k) v_kᵀ  (2)  for some u_k, v_k ∈ R^512, where cos(w_k), sin(w_k) ∈ R^113 are vectors whose c-th entry is cos(w_k c) and sin(w_k c)."
> "Logits(a,b) = W_L MLP(a,b) ≈ Σ_k cos(w_k) u_kᵀ MLP(a,b) + sin(w_k) v_kᵀ MLP(a,b)  (3)"
> "We check empirically that the terms u_kᵀ MLP(a,b) and v_kᵀ MLP(a,b) are approximate multiples of cos(w_k(a+b)) and sin(w_k(a+b)) (>90% of variance explained). … As a sanity check, we confirm that the logits are indeed well-approximated by terms of the form cos(w_k(a+b−c)) (95% of variance explained)."
> "We approximate the output logits as the sum Σ_k α_k cos(w_k(a+b−c)) for k∈{14,35,41,42,52} and fit the coefficients α_k via ordinary least squares. This approximation explains 95% of the variance in the original logits. … If we evaluate test loss using this logit approximation, we actually see an improvement in loss, from 2.4·10⁻⁷ to 4.7·10⁻⁸."

So the claimed logit formula is `Logit(a,b,c) ≈ Σ_k α_k cos(w_k(a+b−c))`, with only `cos(w_k(a+b))` and `sin(w_k(a+b))` appearing as MLP-level features (eq. 3). No `cos(w_k(a−b))` term appears in any formula. Blog ("Calculating cos(w(x+y)),sin(w(x+y)) and calculating logits"): "We also see that the 2x2 blocks are uniform, showing that cos(w(x+y)) and sin(w(x+y)) have the same coefficient. Further analysis shows that everything other than cos(w(x+y)),sin(w(x+y)) for these 5 frequencies is essentially zero." Blog figure caption: "Only terms for the products of waves of the same frequency are non-trivial. Note that these are 2x2 cells with very similar colour, not single large pixels."

**Where the multiplication happens.**
- §3.1: "In our networks, this is computed in the attention and MLP layers."
- App. C.1.2: "Attention heads approximately compute degree-2 polynomials of a single frequency or are used to amplify W_E. … As the attention heads are approximately bilinear (product of attention weights and OV circuit), they are a natural place to perform this computation. … For two of the four heads, the corresponding OV circuit is concentrated on that same frequency. … For the remaining two heads, their attention scores approximately sum to one and the OV circuits contain all five key frequencies, suggesting that they are used to increase the magnitude of key frequencies in the residual stream." Table 2: head 0 k=35 (FVE 99.03%), head 1 k=42 (98.49%), head 2 k=52 (99.07%), head 3 k=42 (97.91%).
- §4.3: "out of 512 total neurons, 433 (84.6%) have over 85% of their variance explained with a single frequency" (degree-2 polynomial of one frequency); "the map W_L from neurons to logits has only two non-trivial components, corresponding to sine and cosine at that frequency" (the k = 14 cluster: 44 neurons).
- Blog: "There are three non-linear operations in a 1L transformer—the attention softmax, the element-wise product of attention and the value vectors, and the ReLU activations in the MLP layer. Here, the model uses both ReLU activations and element-wise products with attention to multiply terms". Colab cell 193: "Each neuron's activations are a linear combination of 1, cos(wx), cos(wy), sin(wx), sin(wy), cos(wx)cos(wy), sin(wx)cos(wy), cos(wx)sin(wy), sin(wx)sin(wy). To calculate the logits, the network cancels out all directions apart from cos(w(x+y))=cos(wx)cos(wy)−sin(wx)sin(wy) and sin(w(x+y))=sin(wx)cos(wy)+cos(wx)sin(wy) and then multiplies these by cos(wz),sin(wz)".

---

## 5. Restricted loss — precise definition

**Paper §5.1 (complete paragraph, verbatim):**
> "Restricted loss. Since the final network uses a sparse set of frequencies w_k, it makes sense to check how well intermediate versions of the model can do using only those frequencies. To measure this, we perform a 2D DFT on the logits to write them as a linear combination of waves in a and b, and set all terms besides the constant term and the 20 terms corresponding to cos(w_k(a+b)) and sin(w_k(a+b)) for the five key frequencies to 0. We then measure the loss of the ablated network."

**Paper §1:** "restricted loss, where we ablate every non-key frequency, and excluded loss, where we instead ablate all key frequencies. Both metrics improve continuously prior to when grokking occurs." **Figure 6 caption:** "…and everything except for the five key frequencies (restricted loss)." **Figure 7 caption:** "(Top Right) The restricted loss begins declining before test loss declines, but has an inflection point when grokking begins to occur." **Figure 18 caption (other seeds):** "As with the mainline model, restricted loss consistently declines prior to train loss."

**Components — what "the 20 terms" means.** §4.1: "There are only twenty components with significant norm, corresponding to the products of sines and cosines for the five key frequencies w_k. These show up as five 2×2 blocks in Figure 4." So "20 terms" counts the 20 basis components of the five same-frequency 2×2 blocks {cos_k,sin_k}×{cos_k,sin_k}. The *directions* cos(w_k(a+b)) and sin(w_k(a+b)) are only 2 per frequency (10 in total), each spanning two of those components. The paper text does not say which of the two (block vs. direction) is zeroed. **The released code projects onto the 10 directions**, Colab cell 16 (verbatim):
```python
def get_component_cos_xpy(tensor, freq, collapse_dim=False):
    # Gets the component corresponding to cos(freq*(x+y)) in the 2D Fourier basis
    # This is equivalent to the matrix cos((x+y)*freq*2pi/p)
    cosx_cosy_direction = fourier_2d_basis_term(2*freq-1, 2*freq-1).flatten()
    sinx_siny_direction = fourier_2d_basis_term(2*freq, 2*freq).flatten()
    # Divide by sqrt(2) to ensure it remains normalised
    cos_xpy_direction = (cosx_cosy_direction - sinx_siny_direction)/np.sqrt(2)
    # Collapse_dim says whether to project back into R^(p*p) space or not
    if collapse_dim:
        return (cos_xpy_direction @ tensor)
    else:
        return cos_xpy_direction[:, None] @ (cos_xpy_direction[None, :] @ tensor)

def get_component_sin_xpy(tensor, freq, collapse_dim=False):
    # Gets the component corresponding to sin((x+y)*freq*2pi/p) in the 2D Fourier basis
    sinx_cosy_direction = fourier_2d_basis_term(2*freq, 2*freq-1).flatten()
    cosx_siny_direction = fourier_2d_basis_term(2*freq-1, 2*freq).flatten()
    sin_xpy_direction = (sinx_cosy_direction + cosx_siny_direction)/np.sqrt(2)
    if collapse_dim:
        return (sin_xpy_direction @ tensor)
    else:
        return sin_xpy_direction[:, None] @ (sin_xpy_direction[None, :] @ tensor)
```
and the notebook's restricted-type loss, cell 245 (named **trig loss** there; the paper's word "restricted" does not occur in the Colab):
```python
def trig_loss(model, mode='all'):
    logits = model(all_data)[:, -1, :-1]
    trig_logits = sum([get_component_cos_xpy(logits, freq) +
                   get_component_sin_xpy(logits, freq)
                   for freq in key_freqs])
    return test_logits(trig_logits,
                       bias_correction=True,
                       original_logits=logits,
                       mode=mode)
get_metrics(model, metric_cache, trig_loss, 'trig_loss')

trig_loss_train = partial(trig_loss, mode='train')
get_metrics(model, metric_cache, trig_loss_train, 'trig_loss_train')
```
Cell 244 markdown: "We further define **trig loss** as the loss where we extract out just the directions of the logits corresponding to cos(w(x+y)),sin(w(x+y)) in the key frequencies. We run this on all of the data, and on just the training set … Trig loss on all data and train loss on just the training data are identical … **Aside:** Projecting onto the trig dimensions requires all of the data to be input into the model. To calculate the trig train loss, we first get the logits for *all* of the data, then project onto the trig components, then throw away the test data logits." Blog: "Trig loss: … only allow them to use the directions corresponding to cos(w(x+y)),sin(w(x+y)) for the 5 relevant frequencies. … Here we evaluate performance on both the train and test set—trig loss for the train and test set is almost equal (the lines are identical)".

**Constant term.** Paper: kept ("besides the constant term"). Code: `bias_correction=True` in `test_logits` (cell 12): "# Applies bias correction - we correct for any missing bias terms, independent of the input, by centering the new logits along the batch dimension, and then adding the average original logits across all inputs / logits = einops.reduce(original_logits - logits, 'batch ... -> ...', 'mean') + logits". Since every cos/sin(w_k(a+b)) direction (k ≠ 0) has zero mean over the full grid, adding the per-class mean of the original logits over all p² inputs is exactly keeping the (const, const) 2D-Fourier component.

**Cross-frequency products:** never kept — neither the paper's wording nor either code path touches any component with different frequencies on the two axes.

**Split.** Paper §5.1: `[NOT FOUND IN SOURCE]` — no split is stated for restricted loss (contrast the explicit "training data" for excluded loss). Colab: `mode='all'` and `mode='train'` (both plotted). TransformerLens demo `get_restricted_loss` (cell 131): `return loss_fn(restricted_logits[test_indices], test_labels)` → **test split**; there the projection is done on the MLP activations (`approx_neuron_acts += neuron_acts.mean(dim=0)`, then `+= (neuron_acts * cos_apb_vec).sum(dim=0) * cos_apb_vec` for `cos_apb_vec = cos(freq·2π/p·(a+b))/‖·‖` and likewise `sin_apb_vec`), mapped through `W_out @ W_U`, and re-centred: `restricted_logits += logits.mean(dim=0, keepdim=True) - restricted_logits.mean(dim=0, keepdim=True)`.

**Numbers.** Colab cell 197 saved output (10-direction projection, bias-corrected, all data): `Loss with just x+y components: 5.547432139695308e-08` vs `Original Loss: 2.4122030061370614e-07`. Paper §4.4: "restrict each neuron's activation to just the components of the polynomial corresponding to terms of the form cos(w_k(a+b)) and sin(w_k(a+b)) in the key frequencies. This improves loss by 77% (to 5.54·10⁻⁸)" — the same number to three figures, i.e. the paper's "cos/sin(w_k(a+b)) only" ablation is numerically the Colab 10-direction projection (inference from the coincidence; the paper describes it at neuron level, but the projection over (a,b) commutes with the linear map W_L). A **different** number is reported for the wider keep-set: "We then ablate all 113·113−40 of the Fourier components besides key frequencies; this ablation actually improves performance (loss drops 70% to 7.24·10⁻⁸)" — 40 kept components = 8 per frequency = the 3×3 {const,cos_k,sin_k}² block minus the constant (matches `extract_freq_2d`), so that ablation also keeps the linear terms const×cos_k etc. Both variants exist in the paper; §5.1's wording ("the 20 terms corresponding to cos(w_k(a+b)) and sin(w_k(a+b))") is closest to the 5.54·10⁻⁸ variant.

---

## 6. Excluded loss — precise definition

**Paper §5.1 (complete paragraph, verbatim):**
> "Excluded loss. Instead of keeping the important frequencies w_k, we next remove only those key frequencies from the logits but keep the rest. We measure this on the training data to track how much of the performance comes from Fourier multiplication versus memorization. The idea is that the memorizing solution should be spread out in the Fourier domain, so that ablating a few directions will leave it mostly unaffected, while the generalizing solution will be hurt significantly."

**All-at-once vs per-frequency.** App. C.1.5: "In Figure 15, we plot the excluded loss if we exclude each of the five key frequencies (as opposed to all five key frequencies)." → the Figure 7 excluded loss removes all key frequencies together; Figures 23 and 26 call this "full excluded loss". Footnote 3 (App. C.2): model-independent variant = max over the 56 single-frequency excluded losses.

**Which components are removed.** Paper: "those key frequencies" / §4.4 "ablate the components corresponding to each of the key frequencies … set various frequencies in this basis to 0" — the paper does not enumerate them. Blog (progress-measure section): "Formally, we calculate the logits, use a 2D DFT to write them as a linear combination of waves in x and waves in y, and set the terms corresponding to cos(w(x+y)),sin(w(x+y)) to zero and keep the other terms the same. Then measure the train loss. … If it isn't (ie, the memorising algorithm), then this is just deleting 2 directions among 113², and so should have no effect." Colab cell 215: "**Excluded Loss** for frequency w is the loss on the training set where we delete the components of the logits corresponding to cos(w(x+y)) and sin(w(x+y)). We get a separate metric for each w in the key frequencies." Cell 216 (verbatim):
```python
def excl_loss(model):
    logits = model(all_data)[:, -1, :-1]
    row = []
    for freq in key_freqs:
        row.append(test_logits((logits -
                                get_component_cos_xpy(logits, freq) -
                                get_component_sin_xpy(logits, freq)),
                               bias_correction=False,
                               mode='train').item())
    return row
```
So in the released code the removed set is exactly the 2 sum-directions of one frequency (per-frequency variant; no bias correction; train split). TransformerLens demo `get_excluded_loss` (cell 137): subtracts the cos/sin(w(a+b)) projections for **all** `key_freqs` at once from the MLP activations, adds the attention-path skip back (`residual_stream_final = excluded_neuron_acts @ W_out + cache["resid_mid", 0][:, -1, :]`), unembeds, and returns `loss_fn(excluded_logits[train_indices], train_labels)`.

**Constant term:** untouched (kept) in all versions. **Split:** training data (paper, blog, both codes). Blog footnote: "Note that excluded loss specifically is weird, because we fit the coefficients of cos(w(x+y)) to the logits across all of the data, so it's technically also a function of the test data."

---

## 7. The three phases and the other progress measures

**Paper §5.2 (verbatim):**
> "Memorization (Epochs 0k–1.4k). We first observe a decline of both excluded and train loss, with test and restricted loss both remaining high and the Gini coefficient staying relatively flat. In other words, the model memorizes the data, and the frequencies w_k used by the final model are unused."
> "Circuit formation (Epochs 1.4k–9.4k). In this phase, excluded loss rises, sum of squared weights falls, restricted loss starts to fall, and test and train loss stay flat. This suggests that the model's behavior on the train set transitions smoothly from the memorizing solution to the Fourier multiplication algorithm. The fall in the sum of squared weights suggests that circuit formation likely happens due to weight decay. Notably, the circuit is formed well before grokking occurs."
> "Cleanup (Epochs 9.4k–14k). In this phase, excluded loss plateaus, restricted loss continues to drop, test loss suddenly drops, and sum of squared weights sharply drops. … This is most cleanly shown in the sharp increase in the Gini coefficient for the matices W_E and W_L, which shows that the network is becoming sparser in the Fourier basis."

**Other measures, §5.1:** "(1) the Gini coefficient (Hurley & Rickard, 2009) of the norms of the Fourier components of W_E and W_L, which measures the sparsity of W_E and W_L in the Fourier basis, and (2) the ℓ2-norm of the weights during training, since weight decay should push these down once the train loss is near zero." Figure 7 caption: "(Bottom Left) The Gini coefficient of the norms of the Fourier components of W_E and W_L increase sharply during cleanup. (Bottom Right) The sums of squared weights decreases smoothly during circuit formation and more sharply during cleanup, indicating that both phases are linked to weight decay." App. C.1.5 adds: restricted *accuracy* (Fig. 13), OLS coefficients of cos(w_k(a+b−c)) over training (Fig. 14), per-frequency excluded loss/accuracy (Fig. 15).

**Seeds, App. C.2.1:** "while all of the models complete memorization by around 1400 epochs, circuit formation and cleanup occur at different times."

**Colab cell 248:** `sum_sq_weights(model)` = `param.pow(2).sum()` per named parameter, plus the total. Cell 247 phases for that run: "(0-1K) … (1K - 8K) … (8K-13K) … (13K-43K) Then all weights plateau (43K-) … a small but noticeable kink … (955 to 942)". Blog phases: Memorisation 0–1K, Interpolation 1K–9K, Generalisation 8K–12K ("a phase change in trig loss"), Cleaning Up Noise 9K–13K, Stability 13K–end. TransformerLens demo (its own run): `memorization_end_epoch = 1500; circuit_formation_end_epoch = 13300; cleanup_end_epoch = 16600`.

---

## 8. Reported numbers (all from the sources above)

| Quantity | Value | Where |
|---|---|---|
| Key frequencies, mainline | k ∈ {14, 35, 41, 42, 52} (5); W_E has 6 non-negligible | §4.1, Fig. 3 |
| Key frequencies, other seeds | 3 or 4 per seed; seed 1 {2,9,19,31}, seed 2 {40,44,53}, seed 3 {31,45,49,52} | App. C.2.1, Table 3 |
| Neurons per frequency | 14: 44, 35: 93, 41: 145, 42: 87, 52: 64, unexplained: 79 (of 512) | Colab cell 180 output; §4.3 "433 (84.6%) have over 85% of their variance explained with a single frequency"; "44 neurons in the k=14 cluster" |
| W_L rank / residual | "approximately rank 10"; residual Frobenius norm "under 0.55% of the norm of W_L" | §4.2 |
| Table 1 FVE of u_kᵀMLP, v_kᵀMLP by single cos/sin(w_k(a+b)) | 93.2, 93.5, 96.8, 96.5, 97.0, 97.0, 96.4, 96.4, 97.4, 98.2 % | Table 1 |
| Logits ≈ Σ α_k cos(w_k(a+b−c)) | 95% variance; loss 2.4e-7 → 4.7e-8 | §4.2 |
| Original loss | 2.4122e-7 (Colab), "2.41·10⁻⁷" (§4.4) | cell 85; §4.4 |
| Replace 433 neurons by degree-2 polynomials | +3%: 2.41e-7 → 2.48e-7 | §4.4 |
| Keep only cos/sin(w_k(a+b)) components | −77% → 5.54e-8 (paper); 5.547e-8 (Colab cell 197) | §4.4; cell 197 |
| Keep const + 40 key-freq components (3×3 blocks) | −70% → 7.24e-8 | §4.4 |
| Project MLP acts on the 10 W_L directions / on their nullspace | 1.19e-7 (−50%) / 5.27 ("worse than uniform") | §4.4 |
| Ablate single key freq at neuron level (Colab) | 14: 2.0e-4, 35: 4.6e-4, 41: 1.9e-3, 42: 5.2e-3, 52: 2.4e-2 | cell 191 output |
| Phases | 0–1.4k / 1.4k–9.4k / 9.4k–14k epochs | §5.2 |
| Restricted / excluded loss curve values | `[NOT FOUND IN SOURCE]` as numbers — only curves (Fig. 7, 15, 18) | — |
| Sum of squared weights | qualitative only: rises during memorization ("total weights rise a lot", blog), falls smoothly in circuit formation, sharply in cleanup (Fig. 7); Colab total "955 to 942" kink at 43K | §5.2; blog; cell 247 |
| Weight-decay sweep | λ=0.3 ≈ 3k epochs to grok, λ=1.0 5–10k, λ=3.0 ≈ 20k; λ=0 never groks; dropout p∈{0.2,0.5} groks (3 seeds), ℓ1 never | App. D.1 |
| Data-fraction sweep | 30–50% grok; ≥60% immediate generalization; 10%/20% no grokking in 40k epochs | App. C.2.2, Fig. 20 |
| Other primes | P=53 needs λ=5; P=109 "exactly the same behavior"; P=401 no grokking for λ∈{0.3,0.5,1,3,5,8} | App. C.2.2 |
| Attention heads | frequencies 35, 42, 52, 42; FVE 99.03/98.49/99.07/97.91%; sigmoid→linear improves 2.41e-7 → 2.12e-7 | App. C.1.3, Table 2 |

---

## 9. Code: actual implementation of restricted/excluded loss and key frequencies

- **Repository `neelnanda-io/Grokking`:** contains **no** restricted/excluded-loss code and no key-frequency code. `grokking.py` and `pages/3_page_3_grokky.py` hold only Plotly helpers, the 1D Fourier basis (identical to Colab cell 121) and a loss-curve plot from `saved_runs/mod_addition_frac_train_sweep.pth`. `saved_runs/` holds 14 `.pth` files (loss curves / weights for the blog's tasks: `mod_addition_frac_train_sweep`, `mod_addition_no_wd`, `no_wd_width_scan`, `low_precision_mod_addition`, `wd_10-1_…`, `wd_10-2_…`, 5-digit addition, induction head, skip trigram). `Grokking_Analysis.ipynb` does not exist (404).
- **Colab notebook (the notebook the README and the blog point to; obtained via the Drive export, complete, 346 cells):** the definitions are in cells 12 (`test_logits` with `mode` ∈ {train,test,all} and `bias_correction`), 16 (`fft2d`, `get_component_cos_xpy`, `get_component_sin_xpy`), 176/179 (`neuron_freqs`, `key_freqs`), 216 (`excl_loss`, per frequency, train, no bias correction), 245 (`trig_loss`, 10 directions, bias correction, `mode='all'` and `'train'`), 248 (`sum_sq_weights`), 220 (`fourier_embed`), 226 (`tensor_trig_ratio`: fraction of centred variance in the sum-directions, computed as `((F[2f−1,2f−1] − F[2f,2f])/√2)² + ((F[2f−1,2f] + F[2f,2f−1])/√2)²`). Tensors: logits of **all** p² inputs at the last position with the `=` class dropped, shape `[p*p, p]`; the (a,b) axis is what gets 2D-transformed. Training-set/test-set membership is `is_train`/`is_test` boolean masks over the same p² ordering (cell 75). The word "restricted" never appears; the paper's "restricted loss" corresponds to the Colab's `trig_loss` (same operator; the paper's 5.54·10⁻⁸ equals cell 197's number).
- **Neither the Colab nor the paper defines a numeric threshold for key frequencies**; the Colab derives them from neuron clusters (argmax over k = 1..55 of the 3×3-block explained variance), the paper text says W_E norms (§4.1) and "DFT on W_L … nontrivial coefficients" (App. C.2).
- **TransformerLens `Grokking_Demo.ipynb` (secondary, later, different run):** `get_restricted_loss` — MLP-activation-level projection onto {mean, cos(w(a+b)), sin(w(a+b))} for `key_freqs`, → `W_out @ W_U`, re-add mean logits, CE on `test_indices`; `get_excluded_loss` — subtract the same projections for all key freqs at once, keep the attention skip path, CE on `train_indices`; `key_freqs` from W_E Fourier norms > max/4 (current `main`).

---

## Consequences for GROKVERSE

Reference: `training/grokverse/analysis/mask_protocols.py` (module docstring: rows `(cos_k, sin_k)` at `1+2(k−1), 2+2(k−1)`; block coordinates `A = Lhat[cos_k,cos_k]`, `B = Lhat[sin_k,sin_k]`, `C = Lhat[sin_k,cos_k]`, `D = Lhat[cos_k,sin_k]`; sum directions `(1,−1,0,0)`, `(0,0,1,1)`; difference directions `(1,1,0,0)`, `(0,0,1,−1)`) and `progress_measures.py` (docstring: "the losses here are measured over the FULL (a, b) grid (train and held-out points together)"; `key_freqs = dominant_frequencies(final_W_E, p)["dominant"]` with a fixed `max_k = 8` cap; `TOP_LEVEL_PROTOCOL = "legacy_broad_mask"`).

### Does each existing variant match the published protocol?

| Variant | Verdict | Why (sources) |
|---|---|---|
| `legacy_broad_mask` | **No.** | It keeps every pair of key rows including cross-frequency products (`cos(w_18 a)·cos(w_15 b)`), and its excluded mask deletes every component touching a key row/column. Nothing in the paper, blog or code ever keeps or deletes cross-frequency components: the paper's kept set is "the constant term and the 20 terms corresponding to cos(w_k(a+b)) and sin(w_k(a+b))" (§5.1) and the 20 significant logit components are "five 2×2 blocks" (§4.1); the code removes "just … 2 directions among 113²" per frequency (blog; Colab cell 216). 289 kept / 3360 removed has no counterpart anywhere. Must never be labelled a reproduction (as the module already says). |
| `same_frequency_block` | **Partial — matches the literal count in the paper's text, not the released code.** | §5.1 says "the 20 terms" and §4.1 identifies the 20 significant terms as the five same-frequency 2×2 blocks, so keeping const + the four components {A,B,C,D} per key frequency (21 for 5 freqs) is one defensible reading of the paper. But no source ever keeps the difference directions `cos/sin(w_k(a−b))`; the code (`get_component_cos_xpy`/`sin_xpy`, TL `cos_apb_vec`/`sin_apb_vec`) projects onto the sum directions only, and the blog states "everything other than cos(w(x+y)),sin(w(x+y)) for these 5 frequencies is essentially zero" (so for a *grokked* model the two readings coincide, during training they need not). Keep it, name it `paper_literal_2x2_block` in reports, never call it exact. |
| `sum_directions_only` | **Yes — this is the operator the released code implements** (up to the split and constant handling, see spec). | Colab `get_component_cos_xpy`: direction `(cos_k⊗cos_k − sin_k⊗sin_k)/√2`, i.e. `(A,B) ↦` projection on `(1,−1)/√2`; `get_component_sin_xpy`: `(sin_k⊗cos_k + cos_k⊗sin_k)/√2`, i.e. `(C,D) ↦ (1,1)/√2`. `trig_loss` = sum of these projections over `key_freqs` (restricted); `excl_loss` = logits minus these projections (excluded), constant untouched. GROKVERSE's `_project` (`u=(A−B)/2 → (u,−u)`, `v=(C+D)/2 → (v,v)`) is the same orthogonal projection; `restrict` re-inserting `Lhat[0,0]` equals the Colab's `bias_correction=True` (the projected logits have zero grid-mean, so adding the per-class mean of the original logits is exactly the constant component). `n_kept = 1 + 2·n_freq`, `n_removed = 2·n_freq` ↔ blog "deleting 2 directions among 113²" per frequency. TL demo's neuron-level version is the same operator pushed through the linear map `W_out @ W_U` (projection over (a,b) commutes with a linear map over neurons), differing only in that its excluded loss leaves the attention-skip contribution unablated. |

### What a `nanda_exact` restricted and excluded loss must do

**Components (reconstructed from code; paper wording is compatible but ambiguous — see §5):**
- Logits: run the model on **all p² pairs** (the projection needs the full grid — Colab cell 244 "Projecting onto the trig dimensions requires all of the data"), take the last-position logits, drop the `=` class → `L[a,b,c]`, `c ∈ 0..p−1`.
- 2D transform over `(a,b)` with the orthonormal real basis (row 0 = const/√p; rows 2k−1, 2k = cos_k, sin_k divided by their norms; k = 1..(p−1)/2) — exactly `fourier.fourier_basis` / Colab cell 121; `fft2d` = `einsum('xyz,fx,Fy->fFz')`.
- Circuit subspace S = span over key k of the two unit vectors `(cos_k⊗cos_k − sin_k⊗sin_k)/√2` and `(sin_k⊗cos_k + cos_k⊗sin_k)/√2` (= normalised `cos(w_k(a+b))`, `sin(w_k(a+b))` on the grid). No difference directions, no linear terms `const×cos_k`, no cross-frequency terms.
- **Restricted** = constant component + P_S(L), inverse-transform, cross-entropy in float64 (Colab `cross_entropy_high_precision`). Equivalent implementation: `P_S(L) + mean_{a,b} L` (Colab `bias_correction=True`).
- **Excluded (full)** = `L − P_S(L)` with S over all key frequencies at once (paper Fig. 7, "full excluded loss" Figs. 23/26; TL `get_excluded_loss`); constant and every other component kept. Also provide **per-frequency excluded** (`L − P_{S_k}(L)` for each k; Colab `excl_loss`, paper Fig. 15) — that is what the released notebook actually plots. No bias correction for excluded.
- Optional, paper §4.4 keep-set variant for the "113·113−40" ablation: const + 3×3 blocks {const,cos_k,sin_k}² per key k (8 non-constant components each; Colab `extract_freq_2d`) — this is a *different* ablation (7.24·10⁻⁸ vs 5.54·10⁻⁸) and must be reported under its own name (`paper_3x3_block`), not as restricted loss.

**Split:**
- Excluded: **training pairs only** (paper §5.1 "We measure this on the training data"; blog "Then measure the train loss"; Colab `mode='train'`; TL `train_indices`). Note the projection itself is fit on all data (blog footnote: "technically also a function of the test data").
- Restricted: **paper text gives no split** `[NOT FOUND IN SOURCE]`. Released Colab: `mode='all'` **and** `mode='train'` (both plotted, reported as identical). TL demo: **test** split. Recommendation: compute and store all three (`test`, `train`, `all`) explicitly labelled; when a single "Nanda-style" number is quoted, use **test** and state that this follows the TransformerLens demo, while the paper's own Figure 7 split is unstated. GROKVERSE's current full-grid evaluation of both losses is not the published protocol for excluded loss and must be relabelled.

**Key-frequency rule:**
- Published rule for the progress measures (App. C.2, verbatim): DFT of `W_L = W_U W_out` over the logit axis, norm over the neuron axis; key frequencies = "the frequencies with nontrivial coefficients". The **count is measured** (5 mainline; 3 or 4 for the other seeds), never fixed; the **threshold is not published** `[NOT FOUND IN SOURCE]`. Alternatives in the sources: W_E Fourier norms (paper §4.1: 6 non-negligible, 5 "used significantly in later parts"; TL demo: `> max/4`), and neuron-cluster argmax with a 0.85 explained-variance cut (Colab cells 176/179).
- Frequencies are fixed once from the final (analysed) checkpoint and applied to every earlier checkpoint (Colab `get_metrics` with the global `key_freqs`; TL demo likewise) — GROKVERSE's "fix from the final state" practice matches; its **top-8 cap on W_E power does not**: `nanda_exact` must (i) take the frequency set from the W_L spectrum of the final model (for the MLP architecture, `W_L` = `W_U @ W_out` of the MLP), (ii) let the count be whatever passes a *declared* threshold (the threshold value is a GROKVERSE choice and must be documented as such, e.g. a fraction of the maximum component norm as in the TL demo, or the Colab's neuron-cluster rule as a cross-check), (iii) record the resulting k's and count per run, and (iv) never silently cap at 8.

**Items not reconstructable from the fetched text:** the numeric "nontrivial" threshold for key frequencies; the data split of the paper's own restricted-loss curve; the numeric values of restricted/excluded loss in Figure 7; AdamW betas in the paper (code says (0.9, 0.98)). Everything else above is reconstructable and quoted.
