# Source note — Swaroop (2026), "Latent Algorithmic Structure Precedes Grokking: A Mechanistic Study of ReLU MLPs on Modular Arithmetic"

| field | value (from https://arxiv.org/abs/2603.23784, fetched 2026-09-02) |
|---|---|
| arXiv id | 2603.23784, **v1 only** ("v1 (Tue, 24 Mar 2026 23:36:26 UTC)"); "Submitted on 24 Mar 2026" |
| author | Anand Swaroop (single author; HTML gives a personal e-mail, no affiliation) |
| comments | "9 pages, 5 figures" |
| license | "License: CC BY 4.0" (HTML header) |
| code | **[NOT FOUND IN SOURCE]** — no repository URL anywhere in the text |
| sections | Abstract · 1 Introduction · 2 Setup · 3 Results · 3.1 Noise · 4 Idealized Model · 4.1 Extraction · 5 Discussion · 6 Conclusion · References (5 entries) |

**How this note was made.** The arXiv HTML (`https://arxiv.org/html/2603.23784v1`) was downloaded raw and converted to text with the LaTeX `alttext` of every formula kept, so all quotes below are verbatim from the source (formulas are the author's LaTeX). The abstract page was fetched for metadata. The PDF (`https://arxiv.org/pdf/2603.23784`) was fetched and saved but its text was not extracted (no PDF text tool in the venv); it was not needed because the HTML is generated from the same LaTeX. Figures 4 and 5 (`phase.png`, `phase_under_noise.png`) were read as images to recover the per-α correlation coefficients printed in the panel titles; Figures 1–3 are SVGs with no text labels and were not rendered. Anything not in the paper is marked **[NOT FOUND IN SOURCE]**; anything I computed myself is marked **[OWN COMPUTATION — NOT IN SOURCE]**; nothing is filled from memory.

**Verification [verifier 2026-09-02].** Independently re-fetched `https://arxiv.org/abs/2603.23784` (still v1 only) and the raw HTML `https://arxiv.org/html/2603.23784v1` (LaTeXML, `alttext` kept) and re-checked every quotation, section/equation/figure reference, table cell and number below against that text; the Figure 4/5 PNGs were re-downloaded and re-read. No wrong quote, number or location was found. A keyword sweep of the body text (case-insensitive) gives: "harmonic" 0, "Gibbs" 0, "batch" 0, "loss" 0, "cross-entropy" 0, "initializ" 0, "schedule" 0, "duty" 0, "github" 0; "seed" appears only in the two "40 (different) random seeds" sentences (§4, Table 1 caption); "embedding(s)" only in the two Nanda-related sentences quoted in §6 below; "transformer(s)" only in the abstract, §1 and §3 (plus the Li et al. title). All **[NOT FOUND IN SOURCE]** labels below are therefore confirmed. The writer's **[OWN COMPUTATION]** numbers were re-derived independently (numpy): periodicity score of a cosine = 48.000 at $p=97$, of $\mathrm{sign}\circ\cos$ = 15.815; `angle(fft(cos(2πνj/p−φ))[ν]) = −φ` (−0.7000), for $\mathrm{sign}\circ\cos$ −0.7125; odd-harmonic power share 0.9500, even 0.00086, amplitude ratios 0.333/0.200/0.143, DC mean ±0.0103 — all as stated. Figures 1–3 are external SVG files referenced by `<object data=…>` (`wa_overlay.svg`, `wb_overlay.svg`, `accuracy_noise.svg` = Fig. 3(a), `periodicity_noise.svg` = Fig. 3(b)); all four were downloaded and contain **zero `<text>` elements** (matplotlib path-only output, 142–1272 `<path>`s each), so the writer's statement that no values can be read from them as text is confirmed — the Fig. 3 accuracy/periodicity-vs-α values remain **[NOT FOUND IN SOURCE]** except for the two endpoints quoted in §3.1. Additions by the verifier are marked "[verifier 2026-09-02]"; nothing of the writer's was deleted.

**Terminology used by the paper.** $W_a^{(i)}$, $W_b^{(i)}$ = the two halves of hidden neuron $i$'s input-weight row (over the one-hot of $a$ and of $b$); $W_{\mathrm{out}}^{(i)}$ = its output-weight column; $\phi_a,\phi_b,\phi_{\mathrm{out}}$ = phases of the dominant DFT component of each; "clean model" = $\alpha=0$; "noisy models" = label-noise fraction $\alpha>0$; "structured neurons" = periodicity score $>12$; "saturation checkpoint" = first time train accuracy reaches 99 %.

---

## 1. Exact model and training setup (§2 "Setup")

Verbatim, §2:

> "We train a multi-layer perceptron (MLP) with one hidden layer on the task of adding two integers, $a$ and $b$, modulo 97. The input is a dimension-194 two-hot vector (a one-hot encoding of $a$ concatenated with a one-hot encoding of $b$). The output is a dimension-97 one-hot vector representing $a+b\ (\mathrm{mod}\ 97)$. The hidden layer consists of 256 fully connected neurons and uses the ReLU activation function."

> "We use the AdamW optimizer on all parameters of the model, with a learning rate of $10^{-3}$ and a weight decay coefficient of $1.0$."

> "The dataset used is the full set of $97^{2}$ integer triples $(a,b,c)$, where $a,b,c\in[0,96]$ and $c\equiv a+b\ (\mathrm{mod}\ 97)$. We use a 30/70 train/validation split, stratified by $c$. We introduce a hyperparameter $\alpha\in[0,1]$ for label noise. We corrupt a fraction $\alpha$ of training labels by replacing each with a value chosen uniformly at random among the remaining $96$ classes. The validation set is unaffected by $\alpha$ and always contains ground-truth triples. We train with $\alpha=0$ (no noise, the standard grokking task; we will refer to this as the "clean model"), as well as with 18 non-zero values ranging from $0.01$ to $0.30$"

> footnote 1: "The specific values used are $0.01$, $0.02$, $0.03$, $0.04$, $0.05$, $0.06$, $0.07$, $0.08$, $0.09$, $0.1$, $0.11$, $0.12$, $0.13$, $0.14$, $0.15$, $0.20$, $0.25$, and $0.30$."

> "We utilize early stopping, stopping training when the model's maximum validation accuracy hasn't improved by at least $10^{-4}$ in the past $50{,}000$ steps, or when the model's validation accuracy rises past $0.999$. We set a hard limit of $500{,}000$ steps (in our experiments, this limit is never reached; for $\alpha=0$, the second early stopping condition is reached, and for non-zero $\alpha$ the first early stopping condition is reached). For each model, we save a saturation checkpoint when the model achieves 99% accuracy on its train set, and the final model when it triggers any of the stopping conditions mentioned."

Also relevant, §1 "Contribution": "In our experimental setting (weight decay $1.0$, train fraction $0.3$), input weights are near-binary square waves rather than cosines".

Summary table (every row traceable to the quotes above):

| item | value | status |
|---|---|---|
| input parametrization | **two-hot**: one-hot($a$) ‖ one-hot($b$), dimension 194; no embedding layer | stated |
| depth / width | one hidden layer, 256 ReLU units | stated |
| output | 97-way one-hot (logits over $a+b \bmod 97$) | stated |
| $p$ | 97 | stated |
| train fraction | 0.3 (30/70 split, stratified by $c$) | stated |
| optimizer | AdamW, lr $10^{-3}$, weight decay 1.0, "on all parameters" | stated |
| steps | early stopping (see quote); hard cap 500 000 never reached; actual step counts at which the clean model stopped / grokked | **[NOT FOUND IN SOURCE]** (no step numbers anywhere) |
| batch size | **[NOT FOUND IN SOURCE]** | |
| loss function | **[NOT FOUND IN SOURCE]** | |
| initialization | **[NOT FOUND IN SOURCE]** | |
| LR schedule | **[NOT FOUND IN SOURCE]** | |
| seeds for the *trained* models | **[NOT FOUND IN SOURCE]** — one clean model and 18 noisy models are described; the only seed count in the paper ("40 different random seeds", §4) refers to the hand-constructed models | |
| biases in the trained model | first-layer bias exists implicitly: §4 speaks of "the effective first-layer bias (after accounting for the vertical offset of the input weights)"; output bias **[NOT FOUND IN SOURCE]** | |
| Grokfast / other accelerators | **[NOT FOUND IN SOURCE]** (not used, not mentioned) | |

---

## 2. The square-wave claim (Abstract, §1, §3, Figs. 1–2)

**What is observed.** Abstract: "We find empirically that ReLU MLPs in our experimental setting instead learn near-binary square wave input weights, where intermediate-valued weights appear exclusively near sign-change boundaries".

§1: "input weights are near-binary square waves rather than cosines: Weights take values $\sim\epsilon\pm A$ across most of their domain (empirically, $A$ is approximately constant across neurons and $\epsilon$ is close to zero), with intermediate values appearing exclusively near sign-change boundaries."

§1: "We find that for a given $i$, $W_{a}^{(i)}$ and $W_{b}^{(i)}$ follow square waves with the same frequency and amplitude, but not necessarily the same phase. Output weights $W_{\mathrm{out}}^{(i)}$, which pass through no nonlinearity, do not follow a square wave; however, they retain the same frequency as $W_{a}^{(i)}$ and $W_{b}^{(i)}$, and their dominant Fourier phases satisfy $\phi_{\mathrm{out}}=\phi_{a}+\phi_{b}$."

§3: "We find that in the clean model, $W_{a}^{(i)}$ and $W_{b}^{(i)}$ follow square waves almost perfectly, as opposed to the sinusoidal waves found in the embeddings of grokking transformers in Nanda et al. (2023)."

**How the "ideal square wave" is built (§3).** "For each neuron, we extract the frequency and phase of the dominant Fourier component with DFT and construct a square wave with the same frequency and phase; no fitting is performed on the frequency and phase. We compute the amplitude and vertical offset such that the constructed wave's upper and lower levels match the median values in the positive and negative half-periods of the neuron weights. We call these constructed waves ideal square waves. Several examples are shown in Figures 1 and 2."

Figure 1 caption: "Actual $W_{a}$ for various neurons plotted in blue, ideal square wave in gray. Intermediate values appear only near sign-change boundaries. Figures are scaled horizontally such that no more than 5 periods are visible in order to show fine details."
Figure 2 caption: "Actual $W_{b}$ from various neurons in red, ideal wave in gray. Intermediate values only near sign-change boundaries, and the dominant frequency extracted from $W_{b}$ matches that of $W_{a}$."

**How "near-binary" / square-vs-cosine is quantified (§3) — the only quantitative test:**

> "To provide a quantitative measure of fit, for each neuron, we calculate the mean nearest-point distance between the input weights and the ideal square wave, as well as the mean nearest-point distance between the input weights and an ideal cosine wave with the same phase, frequency, amplitude, and vertical offset. We perform a paired $t$-test on these values and find that the distance to the ideal square waves is lower: $t(211)=-27.0052$ and $p<10^{-6}$ for the structured neurons; $t(255)=-28.29$ and $p<10^{-6}$ over all 256 neurons."

Numbers characterising the binary levels (§4, given as justification for zero biases in the constructed model): "In the trained model, the effective first-layer bias (after accounting for the vertical offset of the input weights) has mean $0.010\pm 0.004$ (SD) across structured neurons, less than 2.5% of the typical weight amplitude $A\approx 0.45$, and all 212 structured neurons satisfy $|b_{\text{eff}}|<0.05$."

Things the paper does **not** provide:
* a per-neuron "binariness" statistic (e.g. fraction of weights within some tolerance of $\pm A$), or a count/fraction of intermediate-valued weights, or a definition of "near" a sign-change boundary — **[NOT FOUND IN SOURCE]**; the boundary statement is supported by the overlays in Figs. 1–2 and by the nearest-point-distance t-test only.
* the definition of "mean nearest-point distance" (whether it is a vertical residual or a 2-D nearest point on the curve) — **[NOT FOUND IN SOURCE]**.
* duty cycle of the square wave — **[NOT FOUND IN SOURCE]** (the constructed waves are $\mathrm{sign}\circ\cos$, i.e. 50 % duty cycle by construction; for the trained weights nothing is stated). [verifier 2026-09-02] Precision: the explicit $\mathrm{sign}\circ\cos$ form appears only in §4/§4.1 (constructed and extracted models). For §3's "ideal square waves" the waveform formula is never written out — the text gives only "a square wave with the same frequency and phase" (from the DFT) with levels set to the half-period medians (quote above); that it is a 50 % duty-cycle $\mathrm{sign}\circ\cos$ is a reasonable reading, not a statement of the paper.
* an $R^2$ or MSE of the square-wave fit — **[NOT FOUND IN SOURCE]**.
* any statement that the square-wave shape appears in *both* halves vs. only one, beyond "we find identical results on $W_{b}$, with the same neurons in each group" (§3, about the periodicity grouping).

Regime caveat by the author (§5): "Our finding that input weights follow square waves rather than the sinusoidal waves reported in prior work may be regime-dependent. […] Our setting uses ReLU activations with AdamW and weight decay 1.0. Whether the square wave structure arises from the ReLU activation, the strength of weight decay, or their interaction remains an open question."

---

## 3. The phase-sum relation $\phi_{\mathrm{out}}=\phi_a+\phi_b$ (§1, §3.1, Figs. 4–5)

**Which vectors, which transform.** §1: "let $W_{a}^{(i)}$ and $W_{b}^{(i)}$ be the input weight vectors over the one-hot encodings of $a$ and $b$, $W_{\mathrm{out}}^{(i)}$ the vector of weights connecting hidden neuron $i$ to each of the output nodes, and $\phi_{a}^{(i)}$, $\phi_{b}^{(i)}$, and $\phi_{\mathrm{out}}^{(i)}$ the phases of the respective dominant Fourier components (extracted using discrete Fourier transform)."
§3.1: "We extract the phase of the dominant Fourier component from each neuron's $W_{\mathrm{out}}$."
So: three separate 1-D DFTs of length 97 (the $a$-half, the $b$-half, the output column), phase of the dominant non-DC bin of each.

**Frequency agreement (precondition).** §3: "We also find that the dominant Fourier frequencies of $W_{a}^{(i)}$, $W_{b}^{(i)}$, and $W_{\mathrm{out}}^{(i)}$ are equal for $92.97\%$ of all neurons and for 100% of structured neurons." §3.1: "across all noise levels, the dominant frequencies of $W_{a}^{(i)}$, $W_{b}^{(i)}$, and $W_{\mathrm{out}}^{(i)}$ match for 100% of neurons with periodicity $>12$ (although with $\alpha=0.30$ there are only 7 such neurons)."

**The relation and how agreement is quantified.** §3.1: "We find that $\phi_{\mathrm{out}}=\phi_{a}+\phi_{b}$ holds for the highest periodicity neurons in both the clean model and the noisy models. As shown in Figure 4, all structured neurons satisfy $\phi_{\mathrm{out}}=\phi_{a}+\phi_{b}$—up to shifts by multiples of $2\pi$—almost perfectly. This matches the empirical results uncovered in Gromov (2023) and Nanda et al. (2023) and the analytical construction in the former. However, both works find this relation in sinusoidal weights rather than square-wave-like weights."

Figure 4 caption: "$\phi_{\mathrm{out}}$ against $\phi_{a}+\phi_{b}$ across all neurons in the clean model, wrapped to $[-\pi,\pi]$. When restricted to only structured neurons, there is a nearly perfect correlation (circular correlation $r=0.9993$)."

Figure 5 caption: "$\phi_{\mathrm{out}}$ against $\phi_{a}+\phi_{b}$ for various $\alpha$ levels. The highest-periodicity neurons in each model lie close to the diagonal, regardless of the spread of the lower-periodicity neurons. Circular correlation coefficients are listed; $r$ is taken over all neurons, $r^{\prime}$ over only structured neurons (periodicity $>12$). $r^{\prime}\geq 0.998$ for all $\alpha$ levels, indicating that structured neurons satisfy the phase-sum relation regardless of noise."

Per-panel values read off the Figure 5 panel titles (image `phase_under_noise.png`; six panels, not all 18 α levels):

| α | $r$ (all neurons) | $r'$ (structured) |
|---|---|---|
| 0.00 | 0.988 | 0.999 |
| 0.05 | 0.992 | 0.999 |
| 0.10 | 0.919 | 0.998 |
| 0.15 | 0.959 | 0.998 |
| 0.20 | 0.924 | 0.999 |
| 0.30 | 0.896 | 1.000 |

(Figure 4 panels are titled "All neurons" and "Structured neurons"; the colourbar "Periodicity" runs to ≈ 19–20 in both figures, i.e. the largest periodicity scores of trained neurons are ≈ 19–20 — read off the colourbar, approximate.)

**Sign convention.** The paper never states whether $\phi$ is the argument of the DFT coefficient or its negative. The only convention visible is the one used to *build* waves (§4, §4.1): $W_a^{(i)}[j]=f\!\left(\tfrac{2\pi\nu j}{97}-\phi_a^{(i)}\right)$ and $W_{\mathrm{out}}^{(i)}[j]=g\!\left(\tfrac{2\pi\nu j}{97}-\phi_{\mathrm{out}}^{(i)}\right)$, i.e. the phase enters with a minus sign, $\cos(\omega j-\phi)$, and in the from-scratch model the output phase is set to $-\phi_{ai}-\phi_{bi}$ inside the argument (i.e. $\phi_{\mathrm{out}}=\phi_{ai}+\phi_{bi}$).
**[OWN COMPUTATION — NOT IN SOURCE]** (numpy, $p=97$, $\nu=5$, $\phi=0.7$): for $w_j=\cos(2\pi\nu j/p-\phi)$, `angle(fft(w)[ν]) = −0.7000 = −φ`; for $w_j=\mathrm{sign}\cos(\cdot)$ it is $-0.7125$. So if $\phi$ is taken as the *negative* of the DFT argument the paper's $\cos(\omega j-\phi)$ convention is reproduced; the relation $\phi_{\mathrm{out}}=\phi_a+\phi_b$ is invariant under flipping the sign of all three phases, so either convention gives the same relation as long as it is applied identically to $W_a$, $W_b$, $W_{\mathrm{out}}$.

Not provided by the paper: a permutation/shuffle null for the phase relation, a mean absolute circular error, a tolerance window and the fraction of neurons inside it, or a per-seed distribution — **[NOT FOUND IN SOURCE]**. The reported agreement statistics are exactly: $r=0.9993$ (clean, structured neurons), and $r'\ge 0.998$ for all α (per-panel values above).

[verifier 2026-09-02] Three additions:
* **Which "circular correlation" is meant — [NOT FOUND IN SOURCE].** The paper writes only "circular correlation $r$" (Fig. 4 and Fig. 5 captions); no formula and no reference (e.g. Jammalamadaka–Sarma vs. Fisher–Lee) is given. Any reimplementation must state its own choice.
* Per-panel values re-read from `phase_under_noise.png` on 2026-09-02: identical to the table above (α=0.00: r=0.988, r′=0.999; 0.05: 0.992/0.999; 0.10: 0.919/0.998; 0.15: 0.959/0.998; 0.20: 0.924/0.999; 0.30: 0.896/1.000). Figure 4 panel titles "All neurons" / "Structured neurons" confirmed; its two colourbars end at ≈19 (left) and ≈19.5 (right), Figure 5's shared colourbar at ≈19 — consistent with the writer's "≈19–20". Note that Fig. 4's caption value $r=0.9993$ and Fig. 5's α=0.00 panel value $r'=0.999$ describe the same quantity at different rounding.
* Sentence in §3.1 the writer did not quote (verbatim): "Although periodicity generally degrades as the label noise increases, we find that the $\phi_{\mathrm{out}}=\phi_{a}+\phi_{b}$ relation is preserved for the highest-periodicity neurons within each tested $\alpha$ level. This is evident in Figure 5. This suggests that latent structure is formed to some degree, even with significant label noise, which raises the question of whether it can be extracted to recover a correct algorithm despite the model's failure to generalize. We find that this is possible."

---

## 4. The "periodicity score" (§3, Eq. 1)

A named per-neuron score exists. §3: "Not all neurons in the hidden layer learn periodic representations. To distinguish between neurons that encode useful frequency structure and those that do not, we quantify the periodicity of each neuron's input weights using the periodicity score: The ratio of the magnitude of the dominant Fourier component to the mean magnitude across all non-DC components,"

$$\mathrm{per}(w)=\frac{\max_{k=1}^{p-1}|\hat{w}_{k}|}{\frac{1}{p-1}\sum_{k=1}^{p-1}|\hat{w}_{k}|}, \qquad (1)$$

"where $\hat{w}_{k}$ denotes the $k$-th DFT coefficient of weight vector $w$. A high periodicity score indicates that the weight vector is dominated by a single frequency; a score near 1 indicates a flat spectrum with no dominant frequency."

Thresholds and counts (§3, verbatim): "Applying this measure to $W_{a}^{(i)}$ across all 256 hidden neurons reveals a bimodal distribution: 212 neurons have high periodicity scores ($\mathrm{per}(W_{a}^{(i)})>12$) and 37 have low periodicity scores ($\mathrm{per}(W_{a}^{(i)})<5$), with a clear gap between the two populations; 7 neurons fall between the two thresholds and are excluded from both groups (we find identical results on $W_{b}$, with the same neurons in each group). These thresholds were chosen to capture the clear gap in the bimodal distribution. We refer to the high-periodicity neurons as structured neurons and the remaining neurons as unstructured neurons. Zeroing the outputs of all unstructured neurons produces an accuracy drop of less than 0.001, confirming that they contribute negligibly to the neural network's output. We find unstructured neurons tend to have significantly lower-norm connections to the output layer."

So in the clean model 212/256 = 82.8 % of neurons are "structured" (fraction computed from the quoted counts). Under noise: "although with $\alpha=0.30$ there are only 7 such neurons" (§3.1); the per-α counts for the other 17 noise levels are **[NOT FOUND IN SOURCE]** (Fig. 3(b) plots only the *mean* periodicity vs α, and its values are not given in the text). §3.1: "periodicity of neurons follows a gradual decrease (fig. 3(b))"; Figure 3 caption: "(a) Validation accuracy against $\alpha$. (b) Mean periodicity over all neurons against $\alpha$."

Dominant frequency: §3 "we extract the frequency and phase of the dominant Fourier component with DFT" — i.e. the arg-max bin of Eq. 1's numerator; no tie-breaking rule between $k$ and $p-k$ stated **[NOT FOUND IN SOURCE]**.

**[OWN COMPUTATION — NOT IN SOURCE]** — properties of Eq. 1 needed before reusing it (numpy, $p=97$, magnitudes over $k=1..p-1$ exactly as in Eq. 1):
* A pure single-frequency cosine scores $(p-1)/2 = 48$ (the maximum), at every $\nu$.
* An ideal $\mathrm{sign}\circ\cos$ square wave scores $15.82$ at every $\nu$ (its energy is spread over harmonics). Hence the ">12" threshold sits *below* the value of a perfect square wave, and the score is not square-wave-specific — a clean cosine scores three times higher. The colourbar maxima ≈ 19–20 in Figs. 4–5 therefore mean the trained rows are *more* spectrally concentrated than an ideal $\mathrm{sign}\circ\cos$ (consistent with rounded transitions, "intermediate values near sign-change boundaries").
* The maximum depends on $p$, so the thresholds 12 / 5 are specific to $p=97$ and cannot be transferred to $p=113$ without re-deriving the gap.

[verifier 2026-09-02, OWN COMPUTATION — NOT IN SOURCE] Re-run independently with the Eq. 1 definition (numpy `fft`, magnitudes over $k=1..p-1$): cosine → 48.000 ($p=97$) / 56.000 ($p=113$); $\mathrm{sign}\circ\cos$ → 15.815 ($p=97$) / **17.991 ($p=113$)**; identical for $\nu\in\{5,18\}$ and $\phi\in\{0,0.3,0.7\}$. Implementation note for reuse: Eq. 1 is defined on DFT **magnitudes** $|\hat w_k|$, not power; because $|\hat w_k|=|\hat w_{p-k}|$ for real $w$, taking max and mean over the half spectrum $k=1..(p-1)/2$ gives exactly the same value as over all $p-1$ bins, so `curve_spectra`'s half-spectrum `power` array can be used — but as $\sqrt{\text{power}}$.

---

## 5. Idealized model and extraction (§4, §4.1, Tables 1–2)

**From-scratch construction (§4).** "We first construct from scratch an MLP that solves the modular addition task. We use an MLP with the same setup (dimension-194 two-hot input, dimension-256 hidden layer, dimension-97 one-hot output). We use weights"

$$W_{a}^{(i)}[j]=f\!\left(\frac{2\pi ij}{97}-\phi_{ai}\right),\quad W_{b}^{(i)}[j]=f\!\left(\frac{2\pi ij}{97}-\phi_{bi}\right),\quad W_{\mathrm{out}}^{(i)}[j]=g\!\left(\frac{2\pi ij}{97}-\phi_{ai}-\phi_{bi}\right),\quad i\in[0,255],\ j\in[0,96]$$

"with phases $\phi_{ai}$ and $\phi_{bi}$ chosen independently per neuron and uniformly at random from the range $[-\pi,\pi]$. We set all biases to zero: In the trained model, the effective first-layer bias (after accounting for the vertical offset of the input weights) has mean $0.010\pm 0.004$ (SD) across structured neurons, less than 2.5% of the typical weight amplitude $A\approx 0.45$, and all 212 structured neurons satisfy $|b_{\text{eff}}|<0.05$."

(Reading note, not interpretation: in these equations the neuron index $i$ itself is the frequency; no amplitude parameter appears — $f,g$ have unit amplitude.)

"$f$ and $g$ determine the type of wave that $W_{a}$, $W_{b}$, $W_{\mathrm{out}}$ follow; $\cos$ for sinusoidal waves and $\mathrm{sign}\circ\cos$ for square waves. We test three different settings: 1. $f=g=\cos$; this is identical to the setup in Gromov (2023) with ReLU activations. 2. $f=\mathrm{sign}\circ\cos$, $g=\cos$. 3. $f=g=\mathrm{sign}\circ\cos$. We will refer to these three as Cos→Cos, Sq→Cos, and Sq→Sq respectively. […] We use ReLU activations across all constructed models."

"We evaluate the constructed models over 40 different random seeds and various hidden layer widths $N$; the results are summarized in Table 1. All three methods show a monotonic increase in accuracy as the hidden layer width increases, and all achieve high ($>0.998$) accuracy with 512 neurons. Square wave input weights and sinusoidal output weights show the greatest accuracy across all hidden widths, by a statistically significant margin for $N\leq 384$ ($p<0.01$). This suggests that not only is the square wave construction for input weights more faithful to the representation our model learns, but also that it is a more accurate solution."

Table 1 ("Accuracy of constructed models for different hidden widths. Values are mean $\pm$ standard deviation over 40 random seeds. All models use ReLU activations."):

| $N$ | Cos→Cos | Sq→Cos | Sq→Sq |
|---|---|---|---|
| 16 | 0.0422 ± 0.0049 | **0.0735 ± 0.0041** | 0.0458 ± 0.0064 |
| 32 | 0.1322 ± 0.0125 | **0.1944 ± 0.0093** | 0.1415 ± 0.0107 |
| 64 | 0.3653 ± 0.0313 | **0.4449 ± 0.0234** | 0.3314 ± 0.0178 |
| 96 | 0.5850 ± 0.0379 | **0.6522 ± 0.0263** | 0.5057 ± 0.0245 |
| 128 | 0.7482 ± 0.0372 | **0.7968 ± 0.0243** | 0.6512 ± 0.0258 |
| 192 | 0.9222 ± 0.0178 | **0.9417 ± 0.0100** | 0.8450 ± 0.0196 |
| 256 | 0.9793 ± 0.0067 | **0.9849 ± 0.0042** | 0.9375 ± 0.0107 |
| 384 | 0.9987 ± 0.0008 | **0.9992 ± 0.0005** | 0.9907 ± 0.0030 |
| 512 | 0.9999 ± 0.0001 | **1.0000 ± 0.0001** | 0.9989 ± 0.0006 |

"Note that this constructed model is from scratch, with no relation to the actual models we have trained beyond the square-wave structure and the $\phi_{\mathrm{out}}=\phi_{a}+\phi_{b}$ relation."

**Extraction from trained models (§4.1).** "The process is simple: For each neuron, we extract the frequency from the dominant Fourier component of $W_{a}^{(i)}$, and the phase from the dominant Fourier component of $W_{a}^{(i)}$, $W_{b}^{(i)}$, $W_{\mathrm{out}}^{(i)}$. Let $\nu$ be the extracted frequency and $\phi_{a}^{(i)},\phi_{b}^{(i)},\phi_{\mathrm{out}}^{(i)}$ be the extracted phases. Then, we construct a model whose input and output weights follow the extracted phases and frequencies. More formally,"

$$W_{a}^{(i)}[j]=f\!\left(\frac{2\pi\nu j}{97}-\phi_{a}^{(i)}\right),\quad W_{b}^{(i)}[j]=f\!\left(\frac{2\pi\nu j}{97}-\phi_{b}^{(i)}\right),\quad W_{\mathrm{out}}^{(i)}[j]=g\!\left(\frac{2\pi\nu j}{97}-\phi_{\mathrm{out}}^{(i)}\right).$$

"We test the same three settings for $f$ and $g$ as previously described. We perform the extraction on actual models with various $\alpha$ values, on both the final checkpoint and the saturation checkpoint. We evaluate each extracted model on the validation set; our results are summarized in Table 2."

Note: the frequency $\nu$ is taken from $W_a$ only and reused for all three vectors; the *measured* output phase $\phi_{\mathrm{out}}^{(i)}$ is used (not $\phi_a+\phi_b$). Amplitude: §1 says the idealized model is "parameterized entirely by frequencies and phases extracted via DFT" and the §4.1 equations carry no amplitude, whereas the abstract says "parametrized by the frequencies, phases, and amplitudes extracted" — the paper is internally inconsistent on whether an amplitude is used; how (or whether) unstructured neurons are treated differently in the extraction is **[NOT FOUND IN SOURCE]**. [verifier 2026-09-02] The nearest statement is the opening of §4.1 (verbatim): "we use a procedure to replace all $W_{a}^{(i)}$, $W_{b}^{(i)}$, and $W_{\mathrm{out}}^{(i)}$ with square or cosine waves" — "all" reads as every neuron, structured or not, being replaced by its own extracted wave; no separate handling of unstructured neurons is described, and the writer's label stands for anything beyond that.

Table 2 ("Validation accuracy of real and extracted models across noise levels. Results are shown for the final model checkpoint and the saturation checkpoint (when training accuracy first reaches 99%)."):

| α | Final: Real | Sq→Sq | Sq→Cos | Cos→Cos | Saturation: Real | Sq→Sq | Sq→Cos | Cos→Cos |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.9991 | 0.9719 | 0.9979 | 0.9988 | <0.0001 | 0.6912 | 0.8133 | 0.7661 |
| 0.05 | 0.9358 | 0.9705 | 0.9970 | 0.9979 | <0.0001 | 0.7545 | 0.8407 | 0.7967 |
| 0.10 | 0.5663 | 0.9607 | 0.9954 | 0.9957 | <0.0001 | 0.7586 | 0.8825 | 0.8420 |
| 0.15 | 0.1797 | 0.9505 | 0.9912 | 0.9921 | 0.0005 | 0.6408 | 0.7799 | 0.7289 |
| 0.20 | 0.0436 | 0.9135 | 0.9778 | 0.9766 | 0.0002 | 0.6402 | 0.7571 | 0.6959 |
| 0.25 | 0.0120 | 0.9320 | 0.9813 | 0.9765 | 0.0003 | 0.5086 | 0.6496 | 0.5912 |
| 0.30 | 0.0023 | 0.8735 | 0.9548 | 0.9405 | 0.0011 | 0.5641 | 0.6501 | 0.5944 |

§4.1 text: "The Sq→Cos extraction consistently achieves high accuracy on final checkpoints across all noise levels, reaching 95.5% at $\alpha=0.30$ where the real model achieves only 0.23%. At lower noise levels, Sq→Cos and Cos→Cos perform similarly on the final checkpoint, suggesting that for well-trained models the output phase structure is sufficiently preserved that either input wave type recovers the algorithm. The advantage of Sq→Cos becomes pronounced in the final checkpoints at high noise ($\alpha\geq 0.25$) and is consistent across all saturation checkpoints, where the real model accuracy is near zero throughout. The saturation results confirm that the algorithmic structure—the appropriate phases and frequencies to perform modular addition—is already encoded to a significant degree at memorization, before grokking occurs."

Noise curve (§3.1): "as the noise level $\alpha$ increases, the validation accuracy of the final model follows a decreasing sigmoidal transition (fig. 3(a)) from $0.999$ with $\alpha=0.00$ to $0.0023$ with $\alpha=0.30$, while periodicity of neurons follows a gradual decrease (fig. 3(b))."

So the "0.23 % → 95.5 %" headline is Table 2, row α = 0.30, final checkpoint, Real = 0.0023 vs Sq→Cos = 0.9548. Note that the real model at α = 0.30 is *below* chance (1/97 ≈ 1.03 %) — the paper does not comment on this. Only one model per α; no seed variance for Table 2 — **[NOT FOUND IN SOURCE]**.

---

## 6. Transformers, embeddings, harmonics

**Transformers.** The paper contains **no transformer experiments**. Every transformer statement is a citation of Nanda et al. (2023):
* §1: "Mechanistic interpretability work on transformers has characterized one algorithm underlying grokking as a Fourier multiplication circuit: The network learns input embeddings that are sinusoidal functions of the input token, and computes the sum via a phase-addition mechanism in weight space (Nanda et al., 2023)."
* §1: "Together, these works establish sinusoidal weights as a common signature of grokking on modular addition tasks."
* §3: "as opposed to the sinusoidal waves found in the embeddings of grokking transformers in Nanda et al. (2023)."
* Abstract: "has been characterized in previous works as producing sinusoidal input weight distributions in transformers and multi-layer perceptrons (MLPs)."
No architecture comparison, no attention analysis, no claim about *why* transformers differ — **[NOT FOUND IN SOURCE]**.

**Embeddings.** The paper's own model has no embedding layer (two-hot input, §2). The word "embeddings" appears only in the two Nanda-related sentences quoted above. "Shared embedding", embedding dimension, tied/untied embeddings, or any comparison of embedding-based vs two-hot MLPs: **[NOT FOUND IN SOURCE]**.

**Harmonics.** "odd harmonics", "higher harmonics", "$1/j$" (or $1/k$) amplitude decay, "Fourier series of a square wave", "Gibbs": **[NOT FOUND IN SOURCE]**. The paper analyses only the *dominant* DFT component per vector (frequency + phase) plus the Eq. 1 ratio; it never decomposes the square wave into harmonics, never reports harmonic amplitudes or their decay, and never uses harmonic content as evidence. Its evidence for "square wave" is (i) visual overlays (Figs. 1–2), (ii) the paired t-test on nearest-point distances to ideal square vs. ideal cosine (§3), and (iii) the higher accuracy of Sq→Cos constructions (Table 1).

**[OWN COMPUTATION — NOT IN SOURCE]** (for later use, not attributable to the paper): a discrete $\mathrm{sign}\cos(2\pi\nu j/97-\phi)$ sampled at the 97 integers has 95.0 % of its non-DC power on the aliased odd harmonics $\{1,3,5,7\}\nu$, 0.09 % on $\{2,4,6,8\}\nu$, amplitude ratios $|\hat w_{3\nu}|/|\hat w_{\nu}|,\ |\hat w_{5\nu}|/|\hat w_{\nu}|,\ |\hat w_{7\nu}|/|\hat w_{\nu}| = 0.333, 0.200, 0.143$ (= $1/3, 1/5, 1/7$ to three decimals), and a DC mean of $\pm 0.0103$ (odd $p$ makes the two levels unequal in count). Same values for $\nu\in\{5,18\}$, $\phi\in\{0,0.3\}$.

**[verifier 2026-09-02] Two related-work sentences the writer did not quote, relevant for positioning (what the paper itself concedes was already known):**
* §1 on Gromov (2023): "This picture was placed on analytic footing by Gromov (2023), who showed that cosine input weights with a phase-sum constraint $\phi_{\mathrm{out}}=\phi_{a}+\phi_{b}$ are sufficient for arbitrarily high accuracy modular addition in an MLP with a sufficiently large hidden layer, derived an analytic circuit construction for quadratic activations, and showed empirically that this extends to ReLU." §4 adds: "Gromov (2023) proves that, with a sufficiently large hidden layer, their cosine-only model with quadratic activations achieves arbitrarily high accuracy. They also find empirically that the same occurs with ReLU activations, although more neurons are required to match the accuracy of quadratic activations." — i.e. by the paper's own account the phase-sum relation in ReLU MLPs is *not* new to it; its novelty is the square-wave shape and the extraction-under-noise result.
* §1 on Doshi et al. (2024): "Doshi et al. (2024) study the same ReLU MLP architecture under label corruption, showing that memorizing and generalizing neurons coexist and can be identified via their Fourier localization. While their work characterizes the presence of periodic structure under noise, it does not examine the weight shape, the phase-sum constraint, or whether the algorithmic structure survives in models that fail to grok entirely."

**Main "precedes grokking" claim (for context).** §1: "This result implies that grokking does not discover the correct algorithm. The algorithmic structure (the periodic input weight phases and their sum relation) is already encoded during memorization. Grokking sharpens the latent algorithmic structure until generalization is achieved." Evidence = Table 2 saturation columns (real <0.0001–0.0011 vs Sq→Cos 0.65–0.88). No time-resolved (step-by-step) measurement of structure during training is reported; only two checkpoints per model (saturation, final) — **[NOT FOUND IN SOURCE]** for anything finer.

---

## 7. Explicit limitations and open questions (§5)

> "Our finding that input weights follow square waves rather than the sinusoidal waves reported in prior work may be regime-dependent. Gromov (2023) analytically constructs cosine weights using vanilla gradient descent without regularization; Doshi et al. (2024) use AdamW with weight decay but primarily study quadratic activations, for which the analytic cosine solution is exact. Our setting uses ReLU activations with AdamW and weight decay 1.0. Whether the square wave structure arises from the ReLU activation, the strength of weight decay, or their interaction remains an open question."

> "Several limitations apply. Our experiments are confined to a single task (modular addition), a single architecture (one-hidden-layer ReLU MLP), and a single modulus ($p=97$). Whether the latent structure finding generalizes to other tasks or architectures is an important direction for future work."

Additional gaps visible from the text (my reading, not the author's list): single trained model per α (no seed variance for Tables 2 / Figs. 3–5); no ablation of weight decay or train fraction; no step-resolved dynamics; batch size, loss, init, seeds unstated; no code release.

References in the paper (complete list, 5 entries): Doshi, Das, He, Gromov (2024) "To grok or not to grok: disentangling generalization and memorization on corrupted algorithmic datasets", ICLR; Gromov (2023) "Grokking modular arithmetic", arXiv:2301.02679; Li, Liang, Shi, Song, Zhou (2025) "Fourier circuits in neural networks and transformers: a case study of modular arithmetic with multiple inputs", AISTATS pp. 523–531; Nanda, Chan, Lieberum, Smith, Steinhardt (2023) "Progress measures for grokking via mechanistic interpretability"; Power, Burda, Edwards, Babuschkin, Misra (2022) "Grokking: generalization beyond overfitting on small algorithmic datasets", arXiv:2201.02177. Chughtai, Morwani, Liu (Omnigrok), Zhong, Varma, Grokfast: not cited.

---

## Consequences for GROKVERSE

Where the project uses this source (per `docs/RESEARCH_SPEC.md` H2/H3/§6/§7.1 and `docs/dev/RUN_FORMAT_V2.md` §2; [verifier 2026-09-02] also `docs/dev/PREREG_BRIEF.md`, `docs/dev/INTERFACES.md` and the docstring of `training/grokverse/models/mlp_twohot.py` — see item 9), item by item. [verifier 2026-09-02] Repository state checked against: `training/grokverse/config.py`, `data.py`, `train.py`, `models/mlp_twohot.py`, `analysis/mlp_mechanism.py`, and the docs named above, as of 2026-09-02.

1. **"Square-wave input weights" (H2 attribution) — SUPPORTED, with a narrow scope.** The paper does report near-binary square-wave *first-layer weight rows* $W_a^{(i)}, W_b^{(i)}$ in a **two-hot-input, one-hidden-layer, 256-ReLU MLP at $p=97$, AdamW lr $10^{-3}$, wd 1.0, train fraction 0.3**, and states explicitly that this "may be regime-dependent" (§5). It says nothing about MLPs with a learned/shared embedding, nothing about $p=113$, nothing about effective curves $u_a = W_E W_{in}$. So RESEARCH_SPEC's "[UNVERIFIED]" tag on H2 can be replaced by: *attributed to Swaroop (2026) for raw two-hot input rows in that regime; its transfer to our embedding-MLP's effective curves is our own hypothesis.* H2 should not say the paper found this "in ReLU MLPs" in general.

2. **Phase-sum relation $\phi_{\mathrm{out}}\approx\phi_a+\phi_b$ (H1) — SUPPORTED.** Exact method to reuse: per neuron, three length-$p$ DFTs ($a$-half, $b$-half, output column), dominant non-DC bin, phase of that bin; check equal dominant frequencies first (paper: 92.97 % of all, 100 % of structured neurons); agreement reported as circular correlation ($r=0.9993$ structured, clean; $r'\ge0.998$ under noise) on a $\phi_{\mathrm{out}}$-vs-$(\phi_a+\phi_b)$ scatter wrapped to $[-\pi,\pi]$. The paper gives **no** permutation null, tolerance window or error distribution — H1's "permutation null / bootstrap" design is *stronger* than the source and must be described as ours. Sign convention: the paper builds waves as $\cos(2\pi\nu j/p-\phi)$; with numpy's DFT that $\phi$ equals *minus* the coefficient argument [own computation]; the relation is convention-invariant if one convention is used for all three vectors — record which one `analysis/mlp_mechanism.py` uses. [verifier 2026-09-02] Resolved from the code: `analysis/mlp_mechanism.py::curve_spectra` writes each column as `A cos(w_k n - phi)` with `phi = atan2(coeff_sin, coeff_cos)` (same statement in `docs/dev/INTERFACES.md` §1: "`A cos(2πkn/p − φ)`; `φ = atan2(coeff_sin, coeff_cos)`"), i.e. the phase enters with a minus sign exactly as in the paper's §4/§4.1 construction formulas — the two conventions agree, no sign flip is needed. Two further differences to keep in mind when comparing numbers: (i) `phase_relation` reports the mean resultant length $R$ of $\phi_{\mathrm{out}}-(\phi_a+\phi_b)$ with a bootstrap CI and a permutation null — **not** a circular correlation — so our statistic is not numerically comparable with the paper's $r$; (ii) the paper never defines which circular-correlation coefficient it uses (see §3 above), so even a deliberate re-implementation of "$r$" would be our choice. `RESEARCH_SPEC.md` H1 (permutation null, 95 % bootstrap) confirmed as written.

3. **"Periodicity score" — SUPPORTED as a named, exactly defined metric (Eq. 1)**: $\max_{k=1}^{p-1}|\hat w_k| \big/ \tfrac{1}{p-1}\sum_{k=1}^{p-1}|\hat w_k|$ on $W_a^{(i)}$ (and identically on $W_b^{(i)}$), thresholds $>12$ structured / $<5$ unstructured / gap excluded, "chosen to capture the clear gap in the bimodal distribution" — i.e. post hoc from the histogram, not principled. Counts 212 / 37 / 7 at $p=97$, clean. Caveats for reuse: the score's maximum is $(p-1)/2$ (48 at $p=97$, 56 at $p=113$) so the thresholds are $p$-specific; an ideal cosine scores 48 and an ideal $\mathrm{sign}\circ\cos$ only 15.8 [own computation], so a *high* score is not evidence of square-wave shape — it is a single-frequency-dominance measure and must not be used as a square-vs-sine discriminator. If GROKVERSE reuses it, re-derive the gap at $p=113$ and report it alongside the spectral-entropy / participation-ratio measures already planned in WP-2. [verifier 2026-09-02] Repository status: no `periodicity_score` (or `wave_fitting.py` / `binarization_score`) exists anywhere under `training/` yet (grep 2026-09-02). `docs/dev/INTERFACES.md` §2 specifies `periodicity_score(curve)` as "Swaroop 2603.23784's per-neuron score **if the source note defines one**; otherwise `dominant_fraction` under the name `periodicity_score_ours`" — that conditional is now settled: the source defines Eq. 1, so the function must implement Eq. 1 literally (max/mean of DFT *magnitudes* over non-DC bins; see the §4 implementation note above) and may not be silently replaced by `dominant_fraction`, which is a power ratio with a different scale (maximum 1, not $(p-1)/2$). `docs/dev/PREREG_BRIEF.md` ("Swaroop-style periodicity definitions with the thresholds stated in `docs/sources/`") therefore gets thresholds 12 / 5 — valid at $p=97$ only; at $p=113$ the ideal-square-wave value is 17.99 and the cosine value 56, so a "$>12$" cut would sit even further below an ideal square wave than at $p=97$. Note also that Eq. 1 is applied by the paper to *raw* two-hot rows $W_a^{(i)}$; applying it to our embedding-MLP's effective curves $u_a=W_E W_{in}$ is an extension, not a reuse.

4. **Two-hot input MLP as literature control (WP-5, `mlp_twohot`) — architecture SUPPORTED and now confirmed:** input $2p$ two-hot → 256 ReLU → $p$ logits; AdamW lr $10^{-3}$, wd 1.0 on all parameters; 30/70 split stratified by $c$; early stop when max val-acc has not improved by $10^{-4}$ in 50 000 steps or val-acc > 0.999; hard cap 500 000; "saturation checkpoint" at 99 % train accuracy. `RUN_FORMAT_V2.md`'s "exact layer sizes to be confirmed" is resolved: `d_mlp = 256`, input `2p = 194` at `p = 97`. **Not** derivable from the paper (so must be labelled as our choices): batch size, loss, init scale, LR schedule, seed count, output bias. Because GROKVERSE runs $p=113$, a `mlp_twohot` run at $p=113$ is a *control in the paper's spirit*, not a replication; a replication requires $p=97$ (and the stratified-by-$c$ split, which the current `train.py` split rule should be checked against).

   [verifier 2026-09-02] **Checked against the code — the current `arch25k_twohot` preset differs from the paper in more than $p$:**
   * **Width:** `config.py` default `d_mlp = 512`, and the `arch25k_twohot` preset does not override it; `docs/dev/PREREG_BRIEF.md` §Setting also writes "`concat(onehot(a), onehot(b)) → ReLU(512) → 113`". The paper uses **256**. So `RUN_FORMAT_V2.md`'s "exact layer sizes to be confirmed" is resolved *by the source* (256) but **not yet reflected in the preset** — a decision is needed: keep 512 (matches our shared-embedding MLP's `d_mlp`) or add a `d_mlp=256` variant that matches the paper; either way the run must be labelled accordingly.
   * **Split:** `data.py::make_dataset` uses `torch.randperm(p*p)` seeded by `cfg.seed` with `n_train = round(train_frac·p²)` — a plain random permutation, **not** stratified by $c$ as in the paper.
   * **Stopping rule:** fixed `steps = 25000` (master-prompt §14 budget) — the paper uses early stopping (val-acc > 0.999, or no $10^{-4}$ improvement in 50 000 steps; cap 500 000) and saves a "saturation checkpoint" at 99 % train accuracy, which our checkpoint kinds (`memorization`/`generalization`/`grid`) do not reproduce one-to-one.
   * **Optimizer:** `train.py` builds `torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.98))` — weight decay on *all* parameters including biases, which matches the paper's "on all parameters"; `betas` are ours (paper: unstated).
   * **Batching / loss:** `_train_loop` computes `F.cross_entropy` on the entire training set every step (full-batch). Paper: both unstated, so this is our choice, not a match or a mismatch.
   * **Init / biases:** `W_in ~ N(0, init_scale/√(2p))`, `W_out ~ N(0, init_scale/√d_mlp)`, `b_in = b_out = 0` at init, trainable — paper unstated for the trained model.
   * **Analysis coverage:** `analysis/mlp_mechanism.py::effective_curves` raises `ValueError` for `cfg.arch != "mlp"`, so the per-neuron mechanism analysis (spectra, structured-neuron masks, phase relation) **cannot yet run on `mlp_twohot` checkpoints**; `docs/dev/INTERFACES.md` §5 plans "works unchanged for `arch == 'mlp_twohot'` where `u_a = W_in[:p]`, `u_b = W_in[p:]`", but that branch is not implemented. Without it the literature control produces curves but no mechanism numbers.

5. **Odd harmonics / $1/j$ amplitude decay / harmonic-family concentration (H2 prediction, H3, §7.1(5)) — NOT FROM THIS SOURCE.** The paper never mentions harmonics. These statements are textbook Fourier-series facts and must be cited as such (or derived in `docs/MLP_MECHANISM_DERIVATION.md`), never attributed to arXiv:2603.23784. The one thing the paper does supply that bears on H3 is indirect: its own Eq. 1 score of an ideal square wave is only ≈ 15.8/48 of a cosine's [own computation], which is the same "spread energy" effect H3 is built on — but that inference is ours. [verifier 2026-09-02] Two repository facts: (a) `docs/MLP_MECHANISM_DERIVATION.md` **does not exist yet** — it is a planned output (`RESEARCH_SPEC.md` WP-2 "Output:" line and the §11 documentation list), so until it is written the Fourier-series facts have no in-repo derivation to cite; `RESEARCH_SPEC.md` §7.1(5) already states them as textbook facts without attributing them to Swaroop, which is correct. (b) `docs/dev/PREREG_BRIEF.md` H2 is headed "MLP half: replication of Swaroop" — only the *square-wave-vs-sinusoid* part of H2 replicates this paper; the odd-harmonic amplitude-ratio statistic (`α₃/α₁`, "ideal square wave: 1/3") and the `fraction_neurons_best_aic ∈ {square, odd_harmonics}` model comparison are not in the paper (it does no waveform model comparison beyond the nearest-point-distance $t$-test) and should be labelled as our extension there too.

6. **"Intermediate values only near sign-change boundaries" — SUPPORTED qualitatively only** (figure captions + $t$-test); there is no boundary-localisation statistic to reuse. If GROKVERSE wants this as a measurable claim, the metric must be designed here (e.g. distance-to-nearest-transition of every non-saturated weight) and described as new.

7. **Transformer vs MLP contrast — NOT SUPPORTED by this paper.** It has no transformer runs; the "MLP square vs transformer sine" contrast in it is a citation of Nanda et al. (2023). GROKVERSE's cross-architecture claims must rest on Nanda et al. and on our own paired runs, with Swaroop cited only for the MLP side.

8. **Idealized-model / extraction procedure — SUPPORTED and reusable as a causal check** (Sq→Cos from extracted $\nu,\phi_a,\phi_b,\phi_{\mathrm{out}}$, biases zero, unit amplitude). Numbers to quote: Table 1 $N=256$: Cos→Cos 0.9793 ± 0.0067, Sq→Cos 0.9849 ± 0.0042, Sq→Sq 0.9375 ± 0.0107 (40 seeds); Table 2 α = 0.30 final: real 0.0023 vs Sq→Cos 0.9548. Mind the abstract/§4.1 inconsistency on amplitudes if implementing.

9. **[verifier 2026-09-02] Other places in the repository that lean on this source, and what the fetched text says about each:**
   * `docs/dev/INTERFACES.md` §2 `binarization_score(curve)` — "fraction of `|y|/max|y|` above 0.8 (near-binary weights, Swaroop) — labelled ours unless the source defines it": the source defines **no** binariness statistic (see §2 above), so this stays `[ours]`; the 0.8 cut has no counterpart in the paper.
   * `docs/dev/INTERFACES.md` §2 `periodicity_score(curve)` — resolved in item 3: implement Eq. 1.
   * `docs/dev/PREREG_BRIEF.md` §Positioning ("square-wave-like ReLU-MLP weights are **not** new … H1 and the MLP half of H2 are a replication") — consistent with the paper *and* with the paper's own related-work account (§6 above): the phase-sum relation in ReLU MLPs is attributed by Swaroop to Gromov (2023) and Nanda et al. (2023), so H1 should cite those as the origin and Swaroop as a further MLP confirmation, not as the source of the relation.
   * `docs/dev/PREREG_BRIEF.md` §Setting row "input parametrization … our variant of Swaroop's setup" — correct label; see item 4 for the concrete deviations (width 512 vs 256, $p$, split, stopping rule).
   * `training/grokverse/models/mlp_twohot.py` docstring ("until the source note confirms the exact layer sizes this is OUR two-hot variant") — the source now confirms 194 → 256 ReLU → 97; the docstring and `RUN_FORMAT_V2.md` §2 should be updated to say the paper's sizes are known and state whether the preset matches them (it currently does not).
   * `docs/RESEARCH_SPEC.md` §6 WP-5 "two-hot-input MLP (per arXiv:2603.23784, once verified)" and H2 "[UNVERIFIED]" — both conditions are now met by this note; the tags can be replaced as described in items 1 and 4.
