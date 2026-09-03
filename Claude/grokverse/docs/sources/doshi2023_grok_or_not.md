# Doshi, Das, He, Gromov — "To grok or not to grok: Disentangling generalization and memorization on corrupted algorithmic datasets"

Source note for GROKVERSE (WP-0, "record section/equation numbers for every method reused").

| item | value |
|---|---|
| arXiv id | 2310.13061 (cs.LG) |
| versions (arXiv abs page, submission history) | v1 "Thu, 19 Oct 2023 18:01:10 UTC (3,483 KB)"; v2 "Mon, 4 Mar 2024 21:59:58 UTC (4,391 KB)" |
| arXiv "Comments" field (verbatim) | "9+20 pages, 7+25 figures, 2 tables" |
| venue | **ICLR 2024** — verified via dblp, which lists the paper twice: "To Grok or not to Grok: Disentangling Generalization and Memorization on Corrupted Algorithmic Datasets", venue "ICLR 2024" (2024), and "CoRR abs/2310.13061" (2023) (`https://dblp.org/search?q=To+grok+or+not+to+grok+Disentangling+generalization+and+memorization`, fetched 2026-09-03). The venue is **not** written in the arXiv Comments field, not in the v2 HTML text, and not in the GitHub README; the OpenReview forum `UHjE5v5MB7` (found by web search) returned a challenge page and could not be read. |
| authors (abs page) | Darshil Doshi, Aritra Das, Tianyu He, Andrey Gromov |
| code (paper, footnote 1) | "Code to reproduce our results is available at https://github.com/d-doshi/Grokking.git" |
| version quoted here | **v2** HTML (`https://arxiv.org/html/2310.13061v2`); section, equation, figure, table and appendix numbers below are those of v2 |

**How the quotes were obtained.** The `WebFetch` summarizer refused to return long verbatim passages ("would exceed fair use guidelines"), so the v2 HTML was downloaded with `curl` and converted to plain text locally (LaTeXML `alttext` of every `<math>` element kept as `$...$`). Every quotation below is copied from that text. Equations are reproduced as the LaTeX source of the HTML. Figure read-offs are marked "read off Figure N (approximate)" and are not numbers the paper states. Statements from the released GitHub code are marked **[FROM CODE, not the paper]**. Everything I derive myself is marked **[DERIVED — not in source]**.

Fetched: `https://arxiv.org/html/2310.13061` (WebFetch, 11 prompts), `https://arxiv.org/html/2310.13061v2` (curl), `https://arxiv.org/abs/2310.13061` (WebFetch + curl), `https://github.com/d-doshi/Grokking` and `.../tree/main/utils` (WebFetch), raw files `utils/data_processing.py`, `utils/models.py`, `utils/datasets.py`, `utils/hooks.py`, `Notebooks/Grokking_with_label_corruption.ipynb`, `Notebooks/Grokking_with_transformer.ipynb` (curl from `raw.githubusercontent.com/d-doshi/Grokking/main/...`). Figure PNGs from the arXiv HTML were viewed directly; Figures 2 and 6 are SVG in the HTML and were rasterized with headless Chrome before viewing.

**Re-verification (2026-09-03, second pass).** The v2 HTML was downloaded again (`curl`, 308,927 bytes, 117 `<math>` elements, LaTeXML alttext kept as `$...$`) and every quotation in Sections 0–5 of this note was re-checked against the fresh dump by exact string match: Section 1.1 (paragraphs "Grokking modular arithmetic", "Periodic weights", "Inverse participation ratio", "Label corruption"; footnotes 2–4; Equations 1–4), Section 2 phase definitions, Section 2.1 (Hypothesis 2.1, Figure 4/5/6 captions, Equation 5, pruning paragraph), Section 2.2 (Hypothesis 2.2, Figure 7 caption), Sections 3.2 (footnotes 5–6), 4 and 5 in full, Appendix A (default setting and A.1 in full), Appendix B, Appendix C (in full), Appendix D (Equations 6–8), Appendix E, Appendix G (G.1–G.4 and Figure 12–15 captions), Appendix H, Appendix I (I.1, I.2, Figure 17/18 sub-captions), Appendices M, N, O, P, and Appendix Q (Figure 32 caption). No discrepancies were found. The whole dump contains no occurrence of the string "threshold" and no occurrence of "random" in connection with pruning or ablation. The released code (`utils/data_processing.py`, `utils/models.py`, `utils/datasets.py`, both notebooks) was downloaded again from `raw.githubusercontent.com/d-doshi/Grokking/main/...` and every **[FROM CODE]** statement below was re-checked against it. The **[DERIVED]** numerical IPR check was re-run with the project venv. Figures 4(a), 5 and 7(a) (PNG files `ipr_hist_wd_with_phases.png`, `ipr_hist_scatter_bn_with_phases.png`, `phase_modular_addition_wd_tf_mse_N=128.png` from the arXiv HTML) were re-viewed and the approximate read-offs confirmed; one read-off (Figure 5 BatchNorm-weight range) was widened. The venue row was upgraded from [FROM MEMORY - UNVERIFIED] to verified (dblp).

---

## 0. Abstract (verbatim, v2)

> "Robust generalization is a major challenge in deep learning, particularly when the number of trainable parameters is very large. In general, it is very difficult to know if the network has memorized a particular set of examples or understood the underlying rule (or both). Motivated by this challenge, we study an interpretable model where generalizing representations are understood analytically, and are easily distinguishable from the memorizing ones. Namely, we consider multi-layer perceptron (MLP) and Transformer architectures trained on modular arithmetic tasks, where ( $\xi\cdot 100\%$ ) of labels are corrupted (i.e. some results of the modular operations in the training set are incorrect). We show that (i) it is possible for the network to memorize the corrupted labels and achieve $100\%$ generalization at the same time; (ii) the memorizing neurons can be identified and pruned, lowering the accuracy on corrupted data and improving the accuracy on uncorrupted data; (iii) regularization methods such as weight decay, dropout and BatchNorm force the network to ignore the corrupted data during optimization, and achieve $100\%$ accuracy on the uncorrupted dataset; and (iv) the effect of these regularization methods is ("mechanistically") interpretable: weight decay and dropout force all the neurons to learn generalizing representations, while BatchNorm de-amplifies the output of memorizing neurons and amplifies the output of the generalizing ones. Finally, we show that in the presence of regularization, the training dynamics involves two consecutive stages: first, the network undergoes grokking dynamics reaching high train and test accuracy; second, it unlearns the memorizing representations, where the train accuracy suddenly jumps from $100\%$ to $100(1-\xi)\%$ ."

---

## 1. Models, input parametrization, task, p, optimizer, regularization

### 1.1 Two-layer MLP (Section 1.1 "Preliminaries", paragraph "Grokking modular arithmetic", Equation 1)

> "We consider the modular addition task $(m+n)\,\%\,p$ . It can be learned by a two-layer MLP. Explicitly, the network function takes form"
>
> $${\bm{f}}(m,n)={\bm{W}}\phi\left({\bm{W}}_{in}({\bm{e}}_{m}\oplus{\bm{e}}_{n})\right)={\bm{W}}\phi\left({\bm{U}}\,{\bm{e}}_{m}+{\bm{V}}\,{\bm{e}}_{n}\right)\,. \qquad (1)$$
>
> "Here, ${\bm{e}}_{m},{\bm{e}}_{n}\in\mathbb{R}^{p}$ are one_hot encoded numbers $m,n$ . " $\oplus$ " denotes concatenation of vectors $({\bm{e}}_{m}\oplus{\bm{e}}_{n}\in\mathbb{R}^{2p})$ . ${\bm{W}}_{in}\in\mathbb{R}^{N\times 2p}$ and ${\bm{W}}\in\mathbb{R}^{p\times N}$ are the first and second layer weight matrices, respectively. $\phi$ is the activation function. ${\bm{W}}_{in}$ is decomposed into two $N\times P$ blocks: ${\bm{U}},{\bm{V}}\in\mathbb{R}^{N\times p}$ . ${\bm{U}},{\bm{V}}$ serve as embedding vectors for $m,n$ respectively. ${\bm{f}}(m,n)\in\mathbb{R}^{p}$ is the network-output on one example pair $(m,n)$ . The targets are one_hot encoded answers ${\bm{e}}_{(m+n)\,\%\,p}$ ."

Input parametrization is therefore **one-hot** for both operands, concatenated; there is no separate embedding table and no bias term in Eq. (1). **[FROM CODE, not the paper]** `utils/models.py`, `class fcn`: `self.fc1 = nn.Linear(in_size, h_size, bias=False)`, `self.fc2 = nn.Linear(h_size, out_size, bias=False)`, `self.act = nnPower(2)`; the notebook extracts `U = model.fc1.weight.data[:, :p]`, `V = model.fc1.weight.data[:, p:]`, `W = model.fc2.weight.data`.

Activation (Section 1.1, paragraph "Periodic weights"):

> "Choosing quadratic activation function $\phi(x)=x^{2}$ makes the problem analytically solvable (Gromov, 2023). Upon grokking, the trained network weights are qualitatively similar to the analytical solution presented in Gromov (2023)."

Footnote 3: "Periodicity of weights is approximate in trained networks."

### 1.2 Dataset, p, data fraction, loss, optimizer (Section 1.1; Appendix A)

> "The dataset consists of $p^{2}$ examples pairs; from which we randomly pick $\alpha p^{2}$ examples for training [footnote 2], and use the other $(1-\alpha)p^{2}$ examples as a test set. We will refer to $\alpha$ as the data fraction."

Footnote 2: "Throughout the text, networks are trained with Full-batch AdamW and MSE loss, unless otherwise stated."

Appendix A "Further Experimental Details", "Default setting" (verbatim):

> "A 2-layer fully connected network with quadratic activation function and width $N=500$ , initial weights sampled from $\mathcal{N}(0,(16N)^{-2/3})$ , trained on $p=97$ Modular Addition dataset. The networks are trained for 2000 optimization steps, using full-batch, MSE loss, AdamW optimizer, learning rate $\eta=0.01$ and $\beta=(0.9,0.98)$ ."

Appendix A.1 (per-figure settings, verbatim):

> "Figure 1: Default setting with $\alpha=0.5,\xi=0.35$ and (a) $wd=0$ (b) $wd=5$ and (c) $wd=15$ ."
> "Figure 2: Default setting with $\alpha=0.5,\xi=0.0,wd=5.0$ ."
> "Figure 3: Each plot is scanned over $17$ different data fractions $\alpha$ ranging from $0.1$ to $0.9$ and $19$ noise levels $\xi$ from $0.0$ to $0.9$ using the default setting."
> "Figures 4 and 6: Default setting with $\alpha=0.5,\xi=0.35$ ."
> "Figure 5: 2-layer fully connected network with quadratic activation function and width $N=500$ , with and without BatchNorm. We train for $\sim 8000$ steps, with MSE loss, Adam optimizer with minibatch of size 64, and learning rate $\eta=0.005$ . Data fraction $\alpha=0.65$ , corruption fraction $\xi=0.2$ ."
> "General setting for minibatch training: We train the network with batch size $256$ for $2000$ steps. For batch sizes $8$ and $64$ , we scale the number of total steps so as to keep the number of epochs roughly equal. (In other words, all the networks see the same amount of data.) The learning rate is scaled as $0.01\cdot\sqrt{\text{batch size}/256}$ ."
> "Figure 7: All networks are trained with MSE loss using AdamW optimizer with learning rate $\eta=0.001$ and $(\beta_{1},\beta_{2})=(0.9,0.98)$ . (a) Encoder-only Transformer, embedding dimension $d_{\mathrm{embed}}=128$ with ReLU activation, trained for $3000$ steps with $3$ random seeds; (b) Width $N=500$ trained for $5000$ steps with $3$ random seeds."

Regularization sweeps: weight-decay values named in the text/captions are wd = 0, 5, 15 (Figure 4 / Appendix A.1) and 20 (Appendix E: "weight decay = 20", "right-most panel, wd=20.0"); dropout probabilities in Figure 19 "dp=0.0 ... dp=0.3 ... dp=0.6 ... dp=0.9"; BatchNorm "post-BN" (Equation 5, Section 2.1): ${\bm{f}}(m,n)={\bm{W}}\,\texttt{BN}\left(\phi\left({\bm{U}}\,{\bm{e}}_{m}+{\bm{V}}\,{\bm{e}}_{n}\right)\right)$. The full wd / dp grid of Figure 3: [NOT FOUND IN SOURCE] as text (only visible as panel titles in the images).

### 1.3 Label corruption (Section 1.1, paragraph "Label corruption")

> "To introduce label corruption, we choose a fraction $\xi$ of the training examples, and replace the true labels with random labels. The corrupted labels are generated from a uniform random distribution over all the labels. This is often called symmetric label noise in literature (Song et al., 2022). This type of label corruption does not introduce label asymmetry."

**[FROM CODE, not the paper]** `utils/datasets.py`, `noisy_dataset`: `n_noise = int(noise_level * train_size)`; `random_labels = torch.randint(0, p, (n_noise,))`; `labels_noise[:n_noise] = random_labels` — the first `n_noise` examples of the (deterministically shuffled) training set are corrupted, and the random label may coincide with the true one (the paper allows for this in Appendix C: "we subtract $\xi$ from $1.05$ instead of $1.00$ to account for the $1-2\%$ random guessing accuracy").

### 1.4 The analytical periodic solution (Section 1.1, Equation 2; Appendix D, Equations 6–8)

> "The following analytical expression for the network weights leads to $100\%$ generalization (for sufficiently large width)"
>
> $${U}_{ki}=\left(\frac{2}{N}\right)^{-\frac{1}{3}}\cos{\left(\frac{2\pi}{p}i\sigma(k)+\phi^{(u)}_{k}\right)}\,,\quad {V}_{kj}=\left(\frac{2}{N}\right)^{-\frac{1}{3}}\cos{\left(\frac{2\pi}{p}j\sigma(k)+\phi^{(v)}_{k}\right)}\,,\quad {W}_{qk}=\left(\frac{2}{N}\right)^{-\frac{1}{3}}\cos{\left(-\frac{2\pi}{p}q\sigma(k)-\phi^{(u)}_{k}-\phi^{(v)}_{k}\right)}\,, \qquad (2)$$
>
> "where $\sigma(k)$ denotes a random permutation of $k$ in $S_{N}$ – reflecting the permutation symmetry of the hidden neurons. The phase $\phi_{k}^{(u)}$ and $\phi_{k}^{(v)}$ are uniformly i.i.d. sampled between $(-\pi,\pi]$ ."

Appendix D expands $f_q(m,n)=\sum_k W_{qk}(U_{km}+V_{kn})^2$ into nine cosine terms (Equation 6), of which the boxed one is $\frac{1}{2}\cos\left(\frac{2\pi}{p}\sigma(k)(m+n-q)\right)$; "Note that the desired term is the only one that does not have additive phases in the argument. ... Consequently, as $N$ becomes large, all other terms will vanish due to random phase approximation." Equation 7: $f_{q}(m,n)={\frac{1}{N}\sum_{k=1}^{N}\cos{\left(\frac{2\pi}{p}\sigma(k)(m+n-q)\right)}}\approx\delta^{p}(m+n-q)$, with the modular Kronecker delta of Equation 8. Section 3.2, footnote 5: "More precisely, it only requires $\lceil p/2\rceil$ frequencies for real weights."; footnote 6: "The analytical solution becomes exact only at large $N$ ; with sub-leading corrections of order $1/\sqrt{N}$ ."

### 1.5 Transformer and deeper MLPs (Section 2.2; Appendix A "Figure 7"; Appendix G)

Paper text: "we conduct further experiments on $1$ -layer Transformers and $3$ -layer ReLU MLPs" (Section 2.2); "Encoder-only Transformer, embedding dimension $d_{\mathrm{embed}}=128$ with ReLU activation, trained for $3000$ steps with $3$ random seeds" (Appendix A). Figure 12 sub-captions: "(a) Transformer with $8$ heads and embedding dimension $128$ ." "(b) Transformer with $4$ heads and embedding dimension $128$ ." Number of heads / MLP width / positional embedding / tokenization of the Figure 7 transformer: [NOT FOUND IN SOURCE] in the paper text.

**[FROM CODE, not the paper]** `Notebooks/Grokking_with_transformer.ipynb` config: `n_layer = 1`, `if_ln = False # Use LayerNorm or Not`, `n_embd = 128`, `n_head = 4`, `wide_factor = 4 # MLP width`, `block_size = 2`, `vocab_size = config.p`, `weight_tying = False`; the same notebook cell sets `config.lr = 0.01`, `config.weight_decay = 0.5`, `config.noise_level = 0.0`, i.e. the notebook's default run is **not** the Figure 7 run (Appendix A gives $\eta=0.001$ for Figure 7), and the transformer notebook contains no IPR code. `utils/models.py`, `class EncoderOnlyTransformers`: `wte = nn.Embedding(config.vocab_size, config.n_embd)`, `wpe = nn.Embedding(config.block_size, config.n_embd)`, `x = self.transformer.drop(tok_emb + pos_emb)`, `lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)`. I.e. the sequence is the two number tokens only (no "=" token), learned positional embeddings, no LayerNorm in the notebook config.

---

## 2. The inverse participation ratio (IPR)

### 2.1 Definition (Section 1.1, paragraph "Inverse participation ratio", footnote 4, Equations 3–4)

> "To characterize the generalizing representations quantitatively, we utilize a quantity familiar from the physics of localization: the inverse participation ratio (IPR). Its role is to detect periodicity in the weight matrix."
>
> "Let ${\bm{U}}_{k\cdot},{\bm{V}}_{k\cdot}({\bm{W}}_{k\cdot})$ denote the $k^{th}$ row(column) vectors of the weights ${\bm{U}},{\bm{V}}({\bm{W}})$ . Consider the discrete Fourier transforms of these vectors, denoted by $\widetilde{\bm{U}}_{k\cdot},\widetilde{\bm{V}}_{k\cdot}(\widetilde{\bm{W}}_{k\cdot})$ . The Fourier decompositions of the periodic rows(columns) of these weights are highly localized. We leverage this to quantify the similarity of trained weights to the analytic solution (equation 2). To that end, we define the Inverse Participation Ratio (IPR) (Girvin & Yang, 2019; Pastor-Satorras & Castellano, 2016; Gromov, 2023) for these vectors: [footnote 4]"
>
> Footnote 4: "In general IPR is defined as $(\lVert U_{k\cdot}\rVert_{2r}/\lVert U_{k\cdot}\rVert_{2})^{2r}$ . We set $r=2$ , a common choice."
>
> $$\text{IPR}^{(u)}_{k}\coloneqq\left(\frac{\lVert\widetilde{\bm{U}}_{k\cdot}\rVert_{4}}{\lVert\widetilde{\bm{U}}_{k\cdot}\rVert_{2}}\right)^{4}\,;\qquad \text{IPR}^{(v)}_{k}\coloneqq\left(\frac{\lVert\widetilde{\bm{V}}_{k\cdot}\rVert_{4}}{\lVert\widetilde{\bm{V}}_{k\cdot}\rVert_{2}}\right)^{4}\,;\qquad \text{IPR}^{(w)}_{k}\coloneqq\left(\frac{\lVert\widetilde{\bm{W}}_{\cdot k}\rVert_{4}}{\lVert\widetilde{\bm{W}}_{\cdot k}\rVert_{2}}\right)^{4}\,; \qquad (3)$$
>
> "where $\lVert\cdot\rVert_{P}$ denotes the $L^{P}$ -norm of the vector. One can readily see from equation 3 that $IPR\in[1/p,1]$ , with higher values for periodic vectors and lower values for non-periodic ones. It is useful to quantify the periodicity of each neuron in the hidden layer of the network (indexed by $k$ ). We define per-neuron IPR by averaging over the weight-vectors connected to the neuron:"
>
> $$\text{IPR}_{k}\coloneqq\frac{1}{3}\left(\text{IPR}^{(u)}_{k}+\text{IPR}^{(v)}_{k}+\text{IPR}^{(w)}_{k}\right)\,. \qquad (4)$$
>
> "Since trained networks generalize via the periodic features, a larger population of high-IPR neurons leads to better generalization. To quantify the overall similarity of the trained network to the analytical solution (equation 2) we average $\text{IPR}_{k}$ over all hidden neurons: $\overline{\text{IPR}}\coloneqq\mathbb{E}_{k}\left[\text{IPR}_{k}\right]$ ."

So, answering the questions of the brief:

* **What tensor:** the **weights**, not activations and not a separately fitted Fourier model. Per hidden neuron $k$: row $k$ of $U$ (length $p$, indexed by the first operand), row $k$ of $V$ (length $p$, second operand), column $k$ of $W$ (length $p$, indexed by output class). The DFT is taken along that length-$p$ axis. (For the multi-layer models see §4: "neurons connected to the input (embedding) layers", "row-wise IPR for input layer weight and column-wise IPR for output layer weight".)
* **Formula:** Eq. (3), i.e. $\sum_j |\tilde x_j|^4 / (\sum_j |\tilde x_j|^2)^2$ on the DFT coefficients; footnote 4 fixes $r=2$.
* **Per-neuron aggregate:** Eq. (4), plain mean of the three; **network-level:** $\overline{\text{IPR}}$ = mean over neurons (unnumbered, Section 1.1). Figure 2(a) caption: "$\overline{\text{IPR}}\coloneqq\mathbb{E}_{k}\left[\text{IPR}_{k}\right]$ monotonically increases over time, indicating periodic representations."
* **Range stated:** "$IPR\in[1/p,1]$".
* **DFT normalization / real vs complex FFT:** [NOT FOUND IN SOURCE] in the paper. **[FROM CODE, not the paper]** `utils/data_processing.py`: `def calculate_ipr(array, r): return np.power(array / np.sqrt((array ** 2).sum()), 2*r).sum()`; label-corruption notebook: `iprs['U'][i_d, k] = calculate_ipr( np.absolute(np.fft.rfft(U[k])), r )`, same for `V[k]` and `W[:,k]`, with `r = 2`, and `ipr_k = (iprs['U'][-1] + iprs['V'][-1] + iprs['W'][-1])/3`. So the released code uses the **one-sided real FFT magnitude** (`np.fft.rfft`, DC included, $\lfloor p/2\rfloor+1 = 49$ coefficients for $p=97$). Normalization is irrelevant because Eq. (3) is scale-invariant.

**[DERIVED — not in source]** Numerical check with the project venv (p = 97): with `rfft` a pure real cosine row gives IPR = 1.000, a constant row 1.000, a one-hot row $1/49 = 0.0204$, i.i.d. Gaussian rows mean 0.040 (range 0.028–0.085 over 2000 draws, `numpy.random.default_rng(0)`; an earlier run with a different seed gave 0.028–0.070). With a **full complex** `fft` the same cosine gives exactly 0.5 (its power is split between $\pm k$), Gaussian rows give mean 0.020 ≈ $2/p$, a one-hot row gives $1/p = 0.0103$. The paper's stated range $[1/p,1]$ therefore matches the full-DFT convention, while the released code and the histograms (high mode near 0.85–0.95 in Figures 4/5, see below) match the `rfft` convention. The Eq. (2) solution has IPR$_k$ = 1 for every neuron under `rfft` (and 0.5 under full `fft`, except for the $\sigma(k)\equiv 0 \pmod p$ rows, which are constant and give 1 in both). **Whichever convention GROKVERSE adopts must be stated, because the ideal value differs by a factor 2.**

### 2.2 Thresholds / values

* The paper defines **no IPR threshold** for calling a neuron "periodic" or "memorizing". Neurons are **ranked** by IPR$_k$ and pruned in that order (Section 2.1, Figure 6), and the histograms are read qualitatively as "bi-modal". A search of the full v2 text for "threshold" returns nothing. **[FROM CODE, not the paper]** `data_processing.py` contains `ipr_test(array, r, chi_ipr)` returning `ipr_local >= chi_ipr`, but `chi_ipr` is never used in the label-corruption notebook.
* The only numeric thresholds in the paper are the **phase-classification** thresholds on accuracies (Appendix C, Table 2), verbatim: "Trained networks with $\geq 90\%$ test accuracies get classified into Coexistence, Partial Inversion and Full Inversion based on their training accuracy ... Coexistence phase has $\geq 90\%$ training accuracy ... Partial Inversion phase has $<90\%$ but $\geq 100(1.05-\xi)\%$ training accuracy ... Full Inversion phase has $<100(1.05-\xi)\%$ training accuracy ... Memorization phase has $<90\%$ test accuracy but $\geq 90\%$ training accuracy ... Forgetting or confusion phase has $<90\%$ test as well as training accuracies."
* Read off Figure 4(a) (approximate; N = 500, α = 0.5, ξ = 0.35): Coexistence (wd = 0.0): low mode at IPR ≈ 0.05 with ≈ 325 neurons in the first bin, high mode at IPR ≈ 0.85–0.92 with ≈ 130 neurons, essentially nothing in between. Partial Inversion (wd = 5.0): ≈ 220 neurons at ≈ 0.05, broad high mode 0.6–0.9. Full Inversion (wd = 15.0): ≈ 30 neurons at ≈ 0.05, high mode 0.5–0.8 peaking near 0.7–0.75. Read off Figure 5 (α = 0.65, ξ = 0.2, batch 64): high mode ≈ 0.95 with and without BatchNorm; |BatchNorm weight| mostly ≈ 0.02–0.03 (full spread ≈ 0.015–0.04) for IPR < 0.1 neurons vs ≈ 0.03–0.05 for IPR > 0.9 neurons. Read off Figure 2(a) (ξ = 0, wd = 5): avg IPR rises from ≈ 0.05 to ≈ 0.8 over 100 steps while test accuracy jumps between steps ≈ 25 and 40.

---

## 3. Periodic (generalizing) vs memorizing neurons; the pruning protocol

### 3.1 Hypothesis and histogram evidence (Section 2.1, Figure 4)

> **Hypothesis 2.1.** "For a two-layer MLP with quadratic activation trained on (one-hot encoded) modular addition datasets with label corruption, the high IPR neurons facilitate generalization via feature-learning whereas the low IPR neurons cause memorization of corrupted training data."

> "To test Hypothesis 2.1 we study the distribution of per-neuron IPR in various phases, with different levels of regularization. In the Coexistence phase, the network simultaneously generalizes and memorizes the corrupted train data. From Hypothesis 2.1, we expect the presence of both generalizing and memorizing neurons. Indeed, we find a bi-modal distribution of IPRs in trained networks (left column of Figure 4). Remarkably, the network not only distinguishes between the tasks of feature-learning and memorization, it also assigns distinct and independent sub-networks to each."

> "... we observe a distribution-shift towards higher IPR in the Partial Inversion phase (middle column of Figure 4). In the Full Inversion phase (induced by weight decay or dropout), we see that memorizing neurons are almost entirely eliminated in favour of generalizing ones (right column of Figure 4)."

Figure 4 caption: "Distribution of per-neuron IPR for trained networks in various phases. Coexistence phase has a bimodal IPR distribution, where the high and low IPR neurons facilitate generalization and memorization, respectively. Regularization with weight decay or Dropout shifts the IPR distribution towards higher values – Generalizing neurons get more populous compared to memorizing ones; resulting in more robust generalization and Inversion. All plots are made for networks trained with data-fraction $\alpha=0.5$ and corruption-fraction $\xi=0.35$ ; with various regularization strengths."

BatchNorm (Section 2.1, Figure 5): "BatchNorm layer does does not affect the IPR distribution of the neurons significantly – the low IPR neurons persist even in the Full Inversion phase (Figure 5). Instead, BatchNorm adjusts its own weights to de-amplify the low-IPR (memorizing) neurons, biasing the network towards generalizing representations. This can be seen from the strong correlation between $\text{IPR}_{k}$ and the corresponding BatchNorm weights in Figure 5(right). BatchNorm effectively prunes the network and it learns to do this directly from the data!"

### 3.2 Pruning protocol (Section 2.1, Figure 6; Appendix I; Appendix Q)

Main-text protocol (verbatim):

> "To further isolate the effect of individual neurons on performance, we perform the two complementary pruning experiments (Figure 6) : Starting from a trained network in the Coexistence phase, we gradually prune out neurons, one-at-a-time, (i) starting from the lowest IPR neuron (ii) starting from the highest IPR neuron, while keeping track of training and test accuracies. To distinguish network's ability to memorize, we also track the accuracies on corrupted and uncorrupted parts of the training dataset. Before pruning, the network has near- $100\%$ accuracy on both train and test data. In case (i) (Figure 6(left)), we see a monotonic decrease in accuracy on corrupted examples as more and more low IPR neurons are pruned. The accuracy on test data as well as ucorrupted training data remains high. The resulting (overall) training accuracy plateaus around $100(1-\xi)\%$ , corresponding to the fraction of uncorrupted examples. In case (ii) (Figure 6(right)), as more high IPR neurons are pruned, we see a decline in test accuracy (as well as the accuracy on the uncorrupted training data). The network retains its ability to memorize, so the accuracy on corrupted training data remains high."

> "We emphasize that, although each generalizing neuron learns the useful periodic features, the overall computation in the network is emergent: it is performed collectively by all generalizing neurons."

Figure 6 caption: "Progressively pruning neurons from a trained model (Coexistence phase), based on IPR. (data fraction $\alpha=0.5$ , corruption fraction $\xi=0.35$ ). (Left) Pruning out neurons starting from the lowest IPR. The accuracy on corrupted training data decreases, providing evidence that low IPR neurons are responsible for memorization. Test accuracy as well as accuracy on uncorrupted training data remains high. (Right) Pruning out neurons starting from the highest IPR. Test accuracy decreases providing evidence that high IPR neurons are responsible for generalization. Accuracy on corrupted training data remains high."

Read off Figure 6 (approximate): x-axis "fraction of neurons pruned", left panel 0–0.7, right panel 0–0.3; curves "training set", "test set", "corrupted training data", "uncorrupted training data", dotted line "$100(1-\xi)\%$" = 65 %. Left: corrupted-data accuracy falls from ≈ 94 % to ≈ 0 % by fraction ≈ 0.6; overall train accuracy decays to the 65 % line; test and uncorrupted stay ≈ 97–100 %. Right: test accuracy falls from ≈ 97 % to ≈ 0 % by fraction 0.3; corrupted accuracy stays ≈ 97 %; uncorrupted training accuracy starts falling at ≈ 0.17 and is ≈ 40 % at 0.3.

**What is pruned / how — [FROM CODE, not the paper]** (`Grokking_with_label_corruption.ipynb`, section "# Pruning (from Coexistence phase)", hyperparameters `p = 97`, `data_frac = 0.5`, `noise_level = 0.35`, `N = 500`, `dp = 0.0`, `epochs = 1000`, `lr = 1e-2`, `wd = 0.0`, `r = 2`):

```python
X_crr = X_train[:n_noise]; Y_crr = Y_train[:n_noise]          # corrupted subset
X_noncrr = X_train[n_noise:]; Y_noncrr = Y_train[n_noise:]    # uncorrupted subset
ipr_k = (iprs['U'][-1] + iprs['V'][-1] + iprs['W'][-1])/3
sorted_args = np.argsort(ipr_k)                               # ascending IPR
for n_prune in range(N):
    net = copy.deepcopy(model)
    net.fc1.weight.data[sorted_args[:int(n_prune)], :p] = 0   # U rows of pruned neurons
    net.fc1.weight.data[sorted_args[:int(n_prune)], p:] = 0   # V rows
    net.fc2.weight.data[:, sorted_args[:int(n_prune)]] = 0    # W columns
    ... train/test loss (MSE), train/test acc, acc on X_crr, X_noncrr, and on X_og (all p^2 pairs, true labels)
## Reverse
sorted_args_rev = np.argsort(ipr_k)[::-1].copy()              # descending IPR, same loop
```

I.e. a pruned neuron has its two input rows and its output column set to zero on a **copy of the trained checkpoint, no retraining**; the pruned set grows cumulatively from 0 to N neurons; the plotted range is `n_prune_forward = 350` (= 0.7 N) for the low-IPR direction and `n_prune_rev = 150` (= 0.3 N) for the high-IPR direction, matching the x-ranges of the two panels of Figure 6. The notebook also creates `random_args = np.arange(N); np.random.shuffle(random_args)` but **never uses it** in the pruning loops.

**Controls.** Random-pruning or size-matched controls: [NOT FOUND IN SOURCE] — neither in the paper text nor in the released notebook (the shuffled index array is dead code). The only comparison is low-IPR-first vs high-IPR-first.

Extended pruning (Appendix I.1, verbatim): "In contrast to the plots in Figure 6, In Figure 17 we plot longer x-axes – we show accuracies with fraction of pruned neurons ranging from 0.0 to 1.0. We also perform the pruning experiment for Partial Inversion and Full Inversion phases. Pruning affects generalization and memorization differently in different phases because the proportion of high and low IPR neurons in trained networks depends on the phase." Figure 17 sub-captions: "(a) Coexistence phase (weight decay = 0.0). ... Note that the accuracy on corrupted data is retained longer than that on uncorrupted data." "(b) Partial Inversion phase (weight decay = 5.0). ... (Right) Pruning high IPR neurons decreases performance on test data. It improves the performance on corrupted training data and retains it on uncorrupted training data. In the absence of periodic neurons, the (memorizing) low-IPR neurons have an increased influence on the network predictions, which leads to this increase in memorization." "(c) Full Inversion phase (weight decay = 15.0). ... (Right) Pruning high IPR neurons does not improve memorization significantly, except for the small increase in accuracy on corrupted training data towards the end. This is because the fraction of the neurons that have low IPR is very small in this case." All at "$N=500,\alpha=0.5,\xi=0.35$".

Losses under pruning (Appendix I.2, verbatim): "Train loss is lowered by, both, memorizing (low IPR) and generalizing (high IPR) neurons. This leads to monotonic behaviors in all the cases. Test loss is lowered by periodic neurons, while increased by memorizing neurons. An effect that sits on top of these is the effective cancellation of sub-leading terms in the analytical solution 6. Since the accuracies only depend on the largest logit, these sub-leading terms have little effect on them (at sufficient width). However, losses are significantly affected by them. With zero or small weight decay values, this cancellation is achieved by low IPR neurons. Thus, in these cases, low IPR neurons serve the dual purpose of memorization as well as canceling the sub-leading terms. With large weight decay values, this cancellation is achieved by a different algorithm, where "secondary peaks" appear in the Fourier spectrum of the periodic neurons. We leave an in-depth analysis of these algorithmic biases for future work."

How many neurons can be pruned (Appendix Q, verbatim): "In Figure 32 we show how many neurons can pruned from a trained network at various widths. We train networks with widths between 500 and 10000; with learning rate 0.005 and weight decay 1.0, with AdamW optimizer and MSE loss. We use corrupted modular addition data, with data fraction $\alpha=0.5$ and corruption fraction $\xi=0.35$ . We repeat the experiment 40 times, averaging over different random seeds." Figure 32(a): "Number of neurons required after pruning (from low IPR) to retain generalization. We see a sub-linear increase in the required neurons with width. shaded region denoted the standard error over 40 runs." (b): "$\%$ of neurons that can be pruned (from low IPR) while retaining generalization." The numeric values are only in the figure: [NOT FOUND IN SOURCE] as text.

---

## 4. Transformer and deeper-MLP results on periodic representations

### 4.1 Section 2.2 "General Architectures" (verbatim)

> "In this subsection, we aim to check if (i) we can quantify memorization/generalization capabilities of general architectures using IPR distribution of the neurons; (ii) our understanding of regularization can be generalized to such cases."

> **Hypothesis 2.2.** "For any network that achieves non-trivial generalization accuracy on the modular addition dataset with label corruption, its operation can be decomposed into two steps: (I) The input (embedding) layer with row-wise periodic weight maps the data onto periodic features; (II) The remainder of the network nonlinearly implements trigonometric operations over the periodic neurons and maps them to the output predictions. We hypothesize that step (I) depends only on the dataset characteristics; and hence, is a universal property across various network architectures."

> "To verify Hypothesis 2.2 and better understand the impact of regularization, we conduct further experiments on $1$ -layer Transformers and $3$ -layer ReLU MLPs. Our findings, illustrated in the first two columns of Figure 7 and paralleling the observations in Figure 3(a), indicate that increased weight decay promotes generalization over memorization. IPR histograms in the last two columns of Figure 7 further confirm that weight decay encourages the correct features and corroborates the assertions made in Hypothesis 2.2."

> "For these general models, Dropout mostly leads to an extended confusion phase, while BatchNorm only facilitates coexistence. We observe weight decay to be essential for these models to have the full inversion phase. We refer the reader to Appendix G for an in-depth discussion."

Figure 7 caption: "Modular Addition phase diagrams and IPR histograms with different architectures and weight decay values (denoted by "wd"). The IPR histograms are plotted for neurons connected to the input (embedding) layers." Sub-captions "(a) 1-layer Transformer." "(b) 3-layer ReLU MLP."

Read off Figure 7(a) (approximate): panels "wd = 10.0" and "wd = 50.0"; IPR histogram x-axis runs 0–0.4 only; "init" ≈ 0.03–0.06; Coexistence (wd = 10.0) "final" stays at ≈ 0.03–0.05; Full Inversion (wd = 50.0) "final" is a broad mode at ≈ 0.10–0.28 peaking ≈ 0.17–0.2. Note the scale: the transformer embedding IPRs are far below the two-layer-MLP values of Figure 4 (≈ 0.85–0.95). Which axis of the embedding matrix is Fourier-transformed for these histograms: [NOT FOUND IN SOURCE] (the released transformer notebook contains no IPR code either).

### 4.2 Appendix G "Transformers and Deep MLPs" (verbatim)

> "In this section, we present (i) more weight decay experiments on different configurations of Transformers; (ii) the Effect of Dropout on Transformers; (iii) the Effect of BatchNorm on $3$ -layer ReLU MLPs."

> "For reasons that we do not fully understand, $3$ -layer ReLU MLPs with even $0.05$ Dropout probabilities failed to memorize the dataset for any data fraction $\alpha>0.3$ and noise level $\xi\geq 0.1$ , even after an extensive hyperparameter search. We leave the discussion about this for the future."

G.1 "Effect of Depth / Number of Heads / Weight Tying": "To further support our Hypothesis 2.2, we conducted further experiments by varying the configuration of Transformer models. Figure 12 suggests that our Hypothesis holds irrespective of the Transformer configurations." Figure 12 caption: "The IPR histograms of the Embedding layer for Transformers with different configurations. We see that depth does not affect the IPR distribution, whereas more heads encourages high IPR embeddings and weight tying discourages it. All models are trained with corruption fraction $\alpha=0.8$ and corruption fraction $\xi=0.2$ ." [sic — the first "corruption fraction" is the data fraction.] Sub-captions: "(a) Transformer with $8$ heads and embedding dimension $128$ ." "(b) Transformer with $4$ heads and embedding dimension $128$ ." (The Figure 12 image is not among the HTML image files; not viewed.)

G.2 "Dropout Phase Diagram for Transformers": "From Figure 13, we see that with a small dropout probability, the model managed to have a coexistence phase with an average higher IPR for neurons connected to the embedding layer. Notably, any dropout probability larger than $0.1$ does not help." Figure 13 caption: "Modular Addition phase diagrams and IPR histograms with different dropout probabilities (denoted by "dp"). The IPR histograms are plotted for neurons connected to the embedding layer. We select points trained with data fraction $\alpha=0.9$ and corruption fraction $\xi=0.1$ ."

G.3 "Effect of Parameterization for $3$ -layer MLPs": "As we mentioned in Hypothesis 2.2, the input layer should have high IPR columns irrespective of the details of the network. ... we consider two different parameterizations: (i) The standard parameterization with weights sampled from $\mathcal{N}(0,2/\mathrm{fan\_in})$ ; (ii) The parameterization with weights sampled from $\mathcal{N}(0,2/N^{2/3})$ , which corresponds to a network parameterized in a more lazy training regime (Jacot et al., 2022). Our results in Figure 14 show that as the network gets into a lazier regime, the row-wise IPR distribution of the input layer weight barely shifts, whereas the column-wise IPR distribution of the output layer weight has larger values. We interpret this shift as happening because the network with lazier parameterization fails to utilize the hidden layers to decode the periodic features embedded by input layers." Figure 14 caption: "Phase diagrams and the IPR histograms of the columns of input layer weight and rows of the output layer weights for $3$ -layer MLPs with different parameterizations. Selected models are trained with data fraction $\alpha=0.8$ and corruption fraction $\xi=0.2$ ." (The paper is inconsistent about "rows"/"columns" of the input and output layers between the G.3 text, the Figure 14 caption and the Figure 15 caption.)

G.4 "BatchNorm for $3$ -Layer ReLU MLPs": "We found that BatchNorm for $3$ -layer MLPs only has coexistence phases and can perfectly memorize and generalize for small corruption fractions $\xi$ , we showed a representative training curve in Figure 15. We find that with BatchNorm, the $3$ -layer MLP tends to have high IPR neurons close to the output layer instead of having high IPR for neurons close to the input layer. We believe that failure to remove the very low IPR neurons connected to the input layer is why this network is always in the coexistence phase." Figure 15 caption: "IPR of a $3$ -layer MLP in coexistence phase, trained with data fraction $\alpha=0.8$ and corruption fraction $\xi=0.1$ . We plotted row-wise IPR for input layer weight and column-wise IPR for output layer weight; the correlation between column-wise IPR of output layer weights with the absolute value of BatchNorm weights right before it. The correlation between the BatchNorm weights and IPR values we pointed out in the main text still holds. However, we did not observe a strong correlation between row-wise IPR of the input layer and the BatchNorm weight next to it."

**Pruning on Transformers or 3-layer MLPs:** [NOT FOUND IN SOURCE] — the pruning experiments (Figure 6, Figures 17–18, Figure 32) are all on the two-layer quadratic MLP.

### 4.3 Other architecture/setting variations (Appendices N, O, P, M, H)

* ReLU two-layer MLP (Appendix N): "The requirement for regularization for grokking is higher compared to quadratic activation function. We noticed that the phase diagram for weight decay behaves similarly to quadratic activation functions, except for the disappearance of the Forgetting phase. Dropout does not seem as effective for ReLU networks, with the network entering Confusion phase for even moderate dropout probabilites."
* Width (Appendix O, Figure 29): at $N=5000$ with quadratic activation "the network generalizes marginally poorly. Note: the absence of Partial Inversion phase and smaller Coexistence phase with $wd=0.0$ . This can be attributed to the increased capacity to memorize due to abundance of Neurons." Figure 30 (ReLU, $N=5000$): "the network performance is drastically improved compared to $N=500$ (Figure 27). The confusion phase is largely eliminated".
* Modular multiplication (Appendix P, Figure 31): "we ... obtain almost identical phase diagrams to modular addition".
* Cross-entropy (Appendix M): "The requirement for regularization is much greater for CrossEntropy loss compared to MSE. ... Inversion occuring for a narrow band of weight decay values, beyond which we find Confusion. We do not observe a Forgetting phase in this case."
* MNIST (Appendix H): "The generalizing features are harder to quantify for MNIST – we leave this analysis for future work."

---

## 5. Limitations stated by the authors

Section 5 "Limitations and Future work" (complete, verbatim):

> "Although the trained models in our setup are completely interpretable by means of analytical solutions, the training dynamics that lead to these solutions remains an open question. Solving the dynamics should also shed more light on the effects of regularization and label corruption."
>
> "While our work is limited to the exactly solvable models and algorithmic datasets, it paves the way to quantitative investigation of generalization and memorization in more realistic settings."

Further caveats scattered in the text:

* Footnote 3: "Periodicity of weights is approximate in trained networks."
* Footnote 6: "The analytical solution becomes exact only at large $N$ ; with sub-leading corrections of order $1/\sqrt{N}$ ."
* Appendix G: "For reasons that we do not fully understand, $3$ -layer ReLU MLPs with even $0.05$ Dropout probabilities failed to memorize the dataset ... even after an extensive hyperparameter search. We leave the discussion about this for the future."
* Appendix E (Forgetting phase): "Forgetting exclusively occurs in grokked networks with activation functions of degree higher than 1 (e.g. we use $\phi(x)=x^{2}$ ) and with adaptive optimizers." and "We defer this analysis for future work and welcome an examination by our readers."
* Appendix I.2: "We leave an in-depth analysis of these algorithmic biases for future work."
* Appendix H: "The generalizing features are harder to quantify for MNIST – we leave this analysis for future work."
* Appendix C: phase labels rest on fixed accuracy cut-offs (90 %, $100(1.05-\xi)\%$).
* Not stated by the authors but visible in source/code **[DERIVED — not in source]**: no random or size-matched pruning control; IPR histograms and pruning curves in Figures 4–6 are shown for a single network without seeds/error bars (Figure 32 averages 40 seeds; Figure 7 uses 3 seeds); the IPR convention (`rfft` in code vs the "$[1/p,1]$" range in the text) is not pinned down in the paper.

---

## 6. Consequences for GROKVERSE

What can be adopted verbatim, with the equation numbers to cite in `docs/METHODS.md`:

### 6.1 Metric definitions to adopt

1. **Per-vector IPR — Eq. (3) with footnote 4 ($r=2$).** For a length-$p$ real curve $x$ over the input number (or output class), compute its DFT $\tilde x$ and $\text{IPR}(x)=\left(\lVert\tilde x\rVert_4/\lVert\tilde x\rVert_2\right)^4=\sum_j|\tilde x_j|^4/\big(\sum_j|\tilde x_j|^2\big)^2$. Equivalently, with per-frequency power $P_j=|\tilde x_j|^2$: $\sum_j P_j^2/(\sum_j P_j)^2$. **[DERIVED — not in source]** This is directly computable from the `power` array `[half, n]` that `analysis/mlp_mechanism.curve_spectra` already returns (which excludes the constant row; Doshi's `rfft` includes DC — state the choice). This is the concrete definition for the "participation ratio" that `docs/RESEARCH_SPEC.md` §5 lists as a secondary metric and §6 WP-2 lists per neuron.
2. **Per-neuron IPR — Eq. (4):** $\text{IPR}_k=\tfrac13(\text{IPR}^{(u)}_k+\text{IPR}^{(v)}_k+\text{IPR}^{(w)}_k)$. **[DERIVED — not in source]** Mapping to GROKVERSE's MLP (`models/mlp.py`, one-hot → shared `W_E` → `W_in`): Doshi's $U_{k\cdot}$, $V_{k\cdot}$, $W_{\cdot k}$ are, respectively, the effective curves `u_a[:, k] = W_E @ W_in[:d, k]`, `u_b[:, k] = W_E @ W_in[d:, k]` and `out[:, k] = W_out[k, :]` of `mlp_mechanism.effective_curves` (in Doshi's model the input is one-hot and there is no embedding, so $U_{k\cdot}$ *is* the neuron's input curve). The mapping is a composition of linear maps, so it is exact; it is not something the paper states.
3. **Network-level IPR** $\overline{\text{IPR}}=\mathbb{E}_k[\text{IPR}_k]$ (Section 1.1, unnumbered; Figure 2(a)) as a **training-time progress measure** — usable for H4 ("structure precedes generalization"), since Figure 2(a) shows it rising through the transition.
4. **Reference values:** under the released `rfft` convention the ideal Eq. (2) solution has $\text{IPR}_k=1$ and i.i.d. Gaussian curves have $\approx 0.04$ at $p=97$ **[DERIVED — not in source]**; the observed bimodal split in Figure 4(a) is ≈ 0.05 vs ≈ 0.85–0.9 (read off, approximate). For GROKVERSE's $p=113$ the `rfft` floor is $1/57\approx 0.018$; the full-`fft` floor is $1/113$. Declare the convention in `METHODS.md` and in the metric's docstring; do **not** import the paper's "$[1/p,1]$" without checking it against the convention actually implemented.
5. **No IPR threshold** exists in the paper to copy. If GROKVERSE needs a binary "periodic / non-periodic" label, the threshold is a GROKVERSE decision ([HUMAN], RESEARCH_SPEC §5) and must be declared before the runs; the paper's own usage is a **ranking** (argsort of $\text{IPR}_k$), which needs no threshold and is the safer thing to adopt.

### 6.2 Causal-pruning protocol to adopt (Section 2.1, Figure 6; Appendix I, Figures 17–18; Appendix Q, Figure 32)

Adopt verbatim:

* Operate on the **unmodified trained checkpoint, no retraining** ("Starting from a trained network ... we gradually prune out neurons, one-at-a-time"; code: `copy.deepcopy(model)` per pruning size). This matches RESEARCH_SPEC WP-4 ("All ablations run on unmodified checkpoints; no re-training").
* **Rank hidden neurons by $\text{IPR}_k$ (Eq. 4)** and prune **cumulatively** in two directions: (i) lowest IPR first, (ii) highest IPR first.
* **Pruning a neuron = zeroing its input rows and its output column** (code: `fc1.weight[k, :p] = 0`, `fc1.weight[k, p:] = 0`, `fc2.weight[:, k] = 0`). **[DERIVED — not in source]** In GROKVERSE both architectures have biases and a ReLU; zeroing `W_out[k, :]` (MLP) or `W_out[k, :]` of the transformer MLP alone already removes neuron $k$'s contribution to the logits exactly, so the input-side zeroing is redundant for the forward function but harmless; do it to stay protocol-identical, and say what is done with `b_in[k]`.
* **Measure after every pruning size:** train and test accuracy and train and test loss (Appendix I.2 shows the losses carry information the accuracies hide), plotted against **fraction of neurons pruned** (Figure 6 x-axis), over the full range 0–1 (Figure 17), and — if GROKVERSE ever uses label corruption — separately on the corrupted and uncorrupted training subsets.
* **Report the pruning curves per seed and per phase** (Figure 17 does it per phase; Figure 32 averages 40 seeds with standard error).
* **Two-direction design as the causal contrast:** low-first should leave test accuracy intact for a large fraction of the width; high-first should destroy test accuracy after removing a small fraction (Figure 6: test accuracy ≈ 0 after pruning ≈ 0.3 N from the top; read off, approximate). GROKVERSE's H5 prediction is stated as structured-vs-random damage, whereas Doshi's evidence is low-vs-high ranked damage — both can be reported from the same sweep.

**What GROKVERSE must add, because the paper does not have it:** the **size-matched random control** required by RESEARCH_SPEC H5/WP-4 — [NOT FOUND IN SOURCE] in the paper, and dead code in the notebook (`random_args` shuffled, never applied). Adopting Doshi's protocol therefore does not discharge H5's control requirement; it supplies the ranked-ablation sweep that the control is compared against.

### 6.3 Cautions for the cross-architecture comparison

* IPR values are only comparable across objects of the same kind. Figure 4 (two-layer MLP weight rows, high mode ≈ 0.9) and Figure 7(a) (transformer embedding, everything ≤ 0.4) live on different scales; the paper never compares them numerically and never states which axis of the embedding matrix it transforms. GROKVERSE's RESEARCH_SPEC §5 "fairness constraint" applies: compute IPR on the length-$p$ effective curves over the input number in both architectures (MLP: `u_a`, `u_b`, `out`; transformer: the per-neuron effective input curves derived in WP-3, or the columns of `W_E[:p]`), and flag anything else as architecture-specific.
* The paper's pruning evidence is for the two-layer quadratic MLP only; the transformer/3-layer results are histogram evidence for Hypothesis 2.2, not ablations. Any GROKVERSE claim that "IPR-ranked pruning is causal in the transformer" is new and needs its own control.
* Doshi's models differ from GROKVERSE's: one-hot inputs with no embedding table, quadratic activation and no biases (Eq. 1; code), $p=97$, MSE loss, full-batch AdamW, lr 0.01, $\beta=(0.9,0.98)$, init $\mathcal N(0,(16N)^{-2/3})$; the transformer has two tokens and no LayerNorm (code). Expect quantitative differences (e.g. for the ReLU network "The requirement for regularization for grokking is higher", Appendix N; Appendix I.2 notes that at small weight decay low-IPR neurons also cancel sub-leading terms, so pruning them can move the *loss* even when accuracy is flat).
* Label corruption is the paper's tool for making "memorizing neurons" observable; GROKVERSE does not corrupt labels, so the "accuracy on corrupted training data" readout is unavailable, and low-IPR neurons in GROKVERSE checkpoints cannot be called "memorizing" on the strength of this paper alone — only "non-periodic".
