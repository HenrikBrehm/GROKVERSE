# Source note — Power et al. 2022 (Grokking) and Lee et al. 2024 (Grokfast)

**Purpose.** Ground GROKVERSE's definition of grokking, its reproduction recipe, and its use of Grokfast in the two primary sources, verbatim, and check `training/grokverse/train.py::apply_grokfast` against the published rule.

**Protocol.** Every load-bearing statement below is a verbatim quote from the fetched source (arXiv HTML for both papers; arXiv abs pages for bibliographic data; the official Grokfast repository's `README.md` and `main.py` for the reference implementation). Section / equation / figure / table numbers are those of the source. Markers:

- `[VERBATIM]` — copied from the fetched text.
- `[NOT FOUND IN SOURCE]` — looked for, not present in the fetched text; not filled from memory.
- `[EXTRACTED, SENTENCE NOT CAPTURED]` — a number the fetch returned without the surrounding sentence; treat as lower-confidence.
- `[INFERENCE]` — my reasoning, not the source.
- `[FROM MEMORY - UNVERIFIED]` — not used anywhere in this note.
- `[MEASURED IN REPO]` — a number read from a `run.json` under `training/runs/`.

Fetched: `https://arxiv.org/html/2201.02177`, `https://arxiv.org/abs/2201.02177`, `https://arxiv.org/html/2405.20233`, `https://arxiv.org/abs/2405.20233`, `https://raw.githubusercontent.com/ironjr/grokfast/main/README.md`, `https://raw.githubusercontent.com/ironjr/grokfast/main/main.py`. (`https://arxiv.org/pdf/2201.02177` was fetched but returned unreadable binary; the HTML rendering was used instead.)

[verifier 2026-09-02] Independently re-fetched all of the above plus `https://arxiv.org/html/2405.20233v2`, `https://export.arxiv.org/abs/2405.20233`, `https://raw.githubusercontent.com/ironjr/grokfast/main/LICENSE` (`https://arxiv.org/pdf/2405.20233v2` returned unreadable binary). Every quoted passage below was re-checked against the fetched text; corrections and additions are marked "[verifier 2026-09-02]". The repository line numbers in B.7 were re-read from the working tree on 2026-09-02 (`train.py` last committed 25890dc, 2026-09-02) — the writer's line numbers predated a refactor and were all wrong; corrected below. Run-table numbers were re-read from every `training/runs/*/run.json` and all match.

---

## A. Power, Burda, Edwards, Babuschkin, Misra — "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets" (arXiv:2201.02177)

**Bibliographic (arXiv abs page, verbatim fields).** Authors: "Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin, Vedant Misra". Submission history: "[v1] Thursday, 6 January 2022 18:43:37 UTC (1,763 KB)" — only one version. Comments: "Correspondence to alethea@openai.com. Code available at: this https URL" (linking `https://github.com/openai/grok`). Primary category: cs.LG.

### A.1 The definition and the exact wording used for the phenomenon

Abstract `[VERBATIM]`:

> "In this paper we propose to study generalization of neural networks on small algorithmically generated datasets. In this setting, questions about data efficiency, memorization, generalization, and speed of learning can be studied in great detail. In some situations we show that neural networks learn through a process of "grokking" a pattern in the data, improving generalization performance from random chance level to perfect generalization, and that this improvement in generalization can happen well past the point of overfitting. We also study generalization as a function of dataset size and find that smaller datasets require increasing amounts of optimization for generalization. We argue that these datasets provide a fertile ground for studying a poorly understood aspect of deep learning: generalization of overparametrized neural networks beyond memorization of the finite training dataset."

The defining sentence, §1 Introduction `[VERBATIM]`:

> "We show that, long after severely overfitting, validation accuracy sometimes suddenly begins to increase from chance level toward perfect generalization. We call this phenomenon 'grokking'."

Figure 1 caption `[VERBATIM]` (the canonical picture):

> "Left. Grokking: A dramatic example of generalization far after overfitting on an algorithmic dataset. We train on the binary operation of division mod 97 with 50% of the data in the training set. Each of the 97 residues is presented to the network as a separate symbol, similar to the representation in the figure to the right. The red curves show training accuracy and the green ones show validation accuracy. Training accuracy becomes close to perfect at <10³ optimization steps, but it takes close to 10⁶ steps for validation accuracy to reach that level, and we see very little evidence of any generalization until 10⁵ steps. Center. Training time required to reach 99% validation accuracy increases rapidly as the training data fraction decreases. Right. An example of a small binary operation table. We invite the reader to make their guesses as to which elements are missing."

§3.1 "Generalization beyond overfitting" `[VERBATIM]`:

> "Deep learning practitioners are used to seeing small improvements in validation accuracy after validation loss stops decreasing. A double descent of validation loss has been documented in some circumstances, but is considered unusual among practitioners [Nakkiran et al. 2019; Belkin et al. 2018; d'Ascoli et al. 2020]. On the small algorithmic datasets that we study, improved generalization after initial overfitting occurs for a range of models, optimizers, and dataset sizes, and in some cases these effects are extremely pronounced. A typical example is shown for modular division in Figure 1. There we see that validation accuracy starts increasing beyond chance level only after 1000 times more optimization steps than are required for training accuracy to get close to optimal."

> "We found these behaviors to be typical for all the binary operations for dataset sizes that were close to the minimal dataset size for which the network generalized within the allotted optimization budget. For larger dataset sizes, the training and validation curves tend to track each other more closely."

[verifier 2026-09-02] The first §3.1 paragraph has one more sentence that the writer dropped, and it is load-bearing for `RESULTS.md` l. 21 ("test loss … climbs to ~26 during the plateau, then collapses"): `[VERBATIM]` "In Figure 4 the training/validation losses are also plotted and we see the double descent of the validation loss." Figure 4 (Appendix A.2) caption begins `[VERBATIM]` "The loss curves for modular division, train and validation." — i.e. Power documents the validation-*loss* rise-then-fall that GROKVERSE observes; that shape is in the source.

Phrases the paper uses for the phenomenon (all `[VERBATIM]`, section in brackets): "grokking" [§1, title]; "generalization far after overfitting" [Fig. 1]; "generalization beyond overfitting" [title, §3.1 heading]; "improvement in generalization can happen well past the point of overfitting" [abstract]; "long after severely overfitting" [§1]; "late generalization" [§4]; "double descent of validation loss" [§3.1]. Words checked and **not** present in the paper: "phase transition" `[NOT FOUND IN SOURCE]`; "sudden" occurs only in the §1 defining sentence. [verifier 2026-09-02] Both absence claims re-checked: "phase transition" — no occurrence; "sudden"/"suddenly" — exactly one occurrence, the §1 sentence. Section headings as printed (for citation): §3.1 "Generalization beyond overfitting", §3.1.1 "Learning time curves", §3.2 "Grokking on a variety of problems", §3.3 "Ablations and Tricks", §3.4 "Qualitative Visualization of Embeddings", §4 "Discussion", A.1.1 "Binary operations", A.1.2 "Model and optimization".

Operational criterion for "time to generalization": Figure 1 (center) uses "99% validation accuracy" `[VERBATIM]`; §3.1.1 uses "median number of optimization steps until validation performance first reaches 99%" `[VERBATIM]`.

### A.2 The tasks (Appendix A.1.1, §2)

§2 `[VERBATIM]`: "All of our experiments used a small transformer trained on datasets of equations of the form a∘b=c, where each of "a", "∘", "b", "=", and "c" is a separate token."

Appendix A.1.1 `[VERBATIM]`: "For each binary operation we constructed a dataset of equations of the form ⟨x⟩⟨op⟩⟨y⟩⟨=⟩⟨x∘y⟩" … "where ⟨a⟩ stands for the token corresponding to element a." … "The following are the binary operations that we have tried (for a prime number p=97):" — the list as extracted: x∘y = x+y (mod p) for 0≤x,y<p; x−y (mod p); x/y (mod p); "[x/y (mod p) if y is odd, otherwise x−y (mod p)]"; x²+y² (mod p); x²+xy+y² (mod p); x²+xy+y²+x (mod p); x³+xy (mod p); x³+xy²+y (mod p); and on S₅: "x∘y=x⋅y for x,y∈S₅", "x∘y=x⋅y⋅x⁻¹ for x,y∈S₅", "x∘y=x⋅y⋅x for x,y∈S₅".

Split, A.1.1 `[VERBATIM]`: "For each training run, we chose a fraction of all available equations at random and declared them to be the training set, with the rest of equations being the validation set."

### A.3 Architecture and optimizer (Appendix A.1.2)

`[VERBATIM]`: "We trained a standard decoder-only transformer with causal attention masking, and calculated loss and accuracy only on the answer part of the equation."

`[VERBATIM]`: "For all experiments we used a transformer with 2 layers, width 128, and 4 attention heads, with a total of about 4⋅10⁵ non-embedding parameters."

Default optimization `[VERBATIM]`: "AdamW optimizer with learning rate 10⁻³, weight decay 1, β₁=0.9, β₂=0.98, linear learning rate warmup over the first 10 updates, minibatch size 512 or half of training dataset size (whichever was smaller) and optimization budget of 10⁵ gradient updates."

Ablation variants listed in A.1.2 (as extracted): full-batch Adam; standard Adam; full-batch Adam with Gaussian noise; Adam with residual dropout 0.1; AdamW with weight decay 1 (default); AdamW with weight decay toward initialization; Adam with learning rate 3⋅10⁻⁴; Adam with learning rate 3⋅10⁻³; Adam with Gaussian weight noise (σ=0.01). Dropout as a default setting: `[NOT FOUND IN SOURCE]` (residual dropout 0.1 appears only as an ablation).

`[INFERENCE]` The Power architecture is *not* the GROKVERSE architecture: GROKVERSE trains a 1-layer transformer, p=113, full batch, no warmup (Nanda et al. recipe, `config.py` defaults). Power's recipe shares AdamW, lr 1e-3, wd 1, β=(0.9, 0.98) with GROKVERSE.

### A.4 The role of weight decay (§3.3, Figure 2 left)

§3.3 "Ablations and Tricks" `[VERBATIM]`:

> "We've tried various forms of regularization to see what can induce networks to generalize better on our datasets."
> "We find that adding weight decay has a very large effect on data efficiency, more than halving the amount of samples needed compared to most other interventions."
> "Adding some noise to the optimization process (e.g. gradient noise from using minibatches, Gaussian noise applied to weights before or after computing the gradients) is beneficial for generalization."
> "We found that learning rate had to be tuned in a relatively narrow window for the generalization to happen (within 1 order of magnitude)."

[verifier 2026-09-02] Two sentences of §3.3 between the second and third quote above were omitted by the writer and bear on *why* weight decay works `[VERBATIM]`: "We found that weight decay towards the initialization of the network is also effective, but not quite as effective as weight decay towards the origin. This makes us believe that the prior, that approximately zero weights are suitable for small algorithmic tasks, explains part, but not all of the superior performance of weight decay." The noise sentence continues `[VERBATIM]` "…consistent with the idea that such noise might induce the optimization to find flatter minima that generalize better." Also, §3.4 (missed entirely by the writer) ties weight decay to the *visible embedding structure* `[VERBATIM]`: "The structure is more apparent in networks that were optimized with weight decay." — the only sentence in Power linking weight decay to the learned representation rather than to data efficiency.

Figure 2 (left) caption `[VERBATIM]`:

> "Different optimization algorithms lead to different amounts of generalization within an optimization budget of 10^5 steps for the problem of learning the product in the abstract group S_5. Weight decay improves generalization the most, but some generalization happens even with full batch optimizers and models without weight or activation noise at high percentages of training data. Suboptimal choice hyperparameters severely limit generalization. Not shown: training accuracy reaches 100% after 10^3-10^4 updates for all optimization methods."

`[INFERENCE]` Power measures weight decay's effect as **data efficiency** (fraction of data needed to generalize within a fixed 10⁵-step budget), not as a change in the number of steps at fixed data. Any GROKVERSE statement "weight decay speeds up grokking" is not what this source says; the source says weight decay lets the network generalize at smaller training fractions.

### A.5 Train-fraction dependence (§3.1.1, Figure 1 center)

§3.1.1 "Learning time curves" `[VERBATIM]`:

> "In a typical supervised learning problem, decreasing the amount of training data decreases the converged generalization performance of the model when the optimization procedure is capable of interpolating the training data. In our setting, we observe a different phenomenon: while the converged performance stays constant at 100% within a range of training dataset sizes, the optimization time required to achieve that performance grows quicky as the dataset size is decreased."

> "Figure 1 (center) shows median number of optimization steps until validation performance first reaches 99% for the product in abstract group S5. In the vicinity of 25-30% of data, a decrease of 1% of training data leads to an increase of 40-50% in median time to generalization. While the number of steps until validation accuracy > 99% grows quickly as dataset size decreases, the number of steps until the train accuracy first reaches 99% generally trends down as dataset size decreases and stays in the range of 10³-10⁴ optimization steps. We've observed a similar pattern of exponential increase in optimization time until reaching generalization as dataset size decreases on all the algorithmic tasks for which we could get the networks to generalize."

Discussion §4 `[VERBATIM]`: "In addition, we document an interesting phenomenon, where the number of optimization steps needed to reach a given level of performance increases quickly as we reduce the size of the training dataset. Since this represents a way trade compute for performance on smaller amounts of data, it would be useful to investigate in future work whether the effect is also present for other datasets."

Note: the 40–50 %-per-1 % figure is stated for the **S₅ product**, not for modular arithmetic; a corresponding number for modular addition: `[NOT FOUND IN SOURCE]`.

### A.6 Which operations grok (§3.2, Figure 2 right)

§3.2 `[VERBATIM]`:

> "We've measured the mean accuracy across three runs for training datasets consisting of different fractions of all available equations for a variety of binary operations listed in Appendix A.1.1. The results are presented in Figure 2 (right)."
> "Since the operands are presented to the neural network as unrelated abstract symbols, the operations x+y(mod p−1) and x∗y(mod p) with a prime number p and non-zero x,y are indistinguishable from the neural network's perspective (and similarly x−y(mod p−1) and x/y(mod p)). This is because every nonzero residue modulo a prime can be represented as a power of a primitive root. This representation shows the equivalence (up to renaming of symbols) of modular addition modulo p−1 and modular multiplication modulo p. We see in Figure 2 (right) that x−y and x/y indeed take about the same amount of data for generalization to occur."
> "Some of the operations listed in Figure 2 (right) are symmetric with respect to the order of the operands (x+y, x∗y, x2+y2 and x2+xy+y2). Such operations tend to require less data for generalization than closely related non-symmetrical counterparts (x−y, x/y, x2+xy+y2+x). We believe this effect might be partially architecture-dependent, since it's easy for a transformer to learn a symmetric function of the operands by ignoring positional embedding."
> "Some operations (for example x3+xy2+y(mod97)) didn't lead to generalization within the allowed optimization budget at any percentage of data up to 95%."

Figure 2 (right) caption `[VERBATIM]`: "Best validation accuracy achieved after 10^5 steps on a variety of algorithmic datasets, averaged over 3 seeds. Generalization happens at higher percentages of data for intuitively more complicated and less symmetrical operations."

### A.7 What they say about the learned structure (§4, Figure 3)

§4 `[VERBATIM]`: "We have also seen that visualizing the embedding spaces of these neural networks can show natural kinds of structure, for example in problems of modular arithmetic the topology of the embeddings tends to be circles or cylinders. We also see that the network tends to idiosyncratically organize the embeddings by various residues."

Figure 3 caption `[VERBATIM]`: "Left. t-SNE projection of the output layer weights from a network trained on S5. We see clusters of permutations, and each cluster is a coset of the subgroup ⟨(0,3)(1,4),(1,2)(3,4)⟩ or one of its conjugates. Right. t-SNE projection of the output layer weights from a network trained on modular addition. The lines show the result of adding 8 to each element. The colors show the residue of each element modulo 8."

`[INFERENCE]` Power does **not** perform a Fourier analysis and does not claim a trigonometric algorithm; the "circles or cylinders" observation is a t-SNE of the *output* layer weights. Fourier/trig claims belong to Nanda et al. 2023 (separate note), not to this source.

[verifier 2026-09-02] Figure 3 lives in §3.4 "Qualitative Visualization of Embeddings" (not §4; the §4 sentence quoted above is the Discussion's recap). Verified sentences of §3.4 `[VERBATIM]`: first sentence "In order to gain some insight into networks that generalize, we visualized the matrix of the output layer for the case of modular addition and S5." ; "For example the circular topology of modular addition is shown with a 'number line' formed by adding 8 to each element." ; "The structure is more apparent in networks that were optimized with weight decay." The full §3.4 text could not be extracted verbatim (fetch tool declined the whole section); nothing beyond these three sentences is asserted. The `[INFERENCE]` above stands: the object visualised is "the matrix of the output layer", and no Fourier transform is mentioned.

---

## B. Lee, Kang, Kim, Lee — "Grokfast: Accelerated Grokking by Amplifying Slow Gradients" (arXiv:2405.20233)

**Bibliographic (arXiv abs page, verbatim fields).** Authors: "Jaerin Lee, Bong Gyun Kang, Kihoon Kim, Kyoung Mu Lee". Submission history: "[v1] Thursday, 30 May 2024 16:35:30 UTC", "[v2] Wednesday, 5 Jun 2024 15:12:00 UTC". Comments: "17 pages, 13 figures. Typo fixed. Project page: this https URL" (`https://jaerinlee.com/research/grokfast`); code: `https://github.com/ironjr/grokfast`. Primary category cs.LG. **Note:** the author list given in the task brief ("Lee, Kim, Choi, Kim") does not match the arXiv record; the correct list is Lee, Kang, Kim, Lee. The HTML fetched is the latest version (v2).

### B.1 Their definition of grokking and their operational criterion

Abstract `[VERBATIM]`:

> "One puzzling artifact in machine learning dubbed *grokking* is where delayed generalization is achieved tenfolds of iterations after near perfect overfitting to the training data. Focusing on the long delay itself on behalf of machine learning practitioners, our goal is to accelerate generalization of a model under grokking phenomenon. By regarding a series of gradients of a parameter over training iterations as a random signal over time, we can spectrally decompose the parameter trajectories under gradient descent into two components: the fast-varying, overfitting-yielding component and the slow-varying, generalization-inducing component. This analysis allows us to accelerate the grokking phenomenon more than ×50 with only a few lines of code that amplifies the slow-varying components of gradients."

§1 `[VERBATIM]`: "Grokking is a recently discovered phenomenon where generalization is achieved long after a model overfits to the training data." … "Our goal is, to this end, to accelerate the grokking phenomenon."

§2.1 hypothesis `[VERBATIM]`: "The fast-varying component of the parameter updates contributes to the rapid overfitting, and the slow-varying component contributes to the slow generalization." Also `[VERBATIM]`: "the grokking phenomenon is directly related to the low-frequency part of the dual representation U(ω)".

[verifier 2026-09-02] **Location corrected.** The "fast-varying component … slow generalization" sentence is in **§1 Introduction**, not §2.1, and is the second half of a longer sentence `[VERBATIM]`: "…the parameter updates under grokking takes effect in two different timescales: the fast-varying component of the parameter updates contributes to the rapid overfitting, and the slow-varying component contributes to the slow generalization." §2.1 ("Filter Design") restates the premise in the "low-frequency part of the dual representation U(ω)" sentence (correctly located) and states the design goal `[VERBATIM]` "Our goal is therefore to design a filter h(t) with low-pass characteristics". Section headings as printed: §2.2 "Experiment", §2.3 "Discussion", §2.4 "Limitations", §5 "More Discussion", §5.1 "Difference between Algorithm 2 and the Momentum in Typical Optimizers", §5.2 "Visualizing Trajectories".

Operational criterion, §2.2 `[VERBATIM]`: "Comparing the time to reach the accuracy of 0.95, the generalization, i.e., the late saturation of the validation accuracy, happens after ×97.3 iterations after the rapid saturation of the training accuracy (the overfitting)." (baseline, modular multiplication) [verifier 2026-09-02: writer dropped the closing "(the overfitting)"; restored]. The 0.95 validation-accuracy crossing is the quantity the algorithmic-task speedups are measured on (Table 1 column header `[VERBATIM]` "Iterations at acc≥0.95"; Table 3 column header `[VERBATIM]` "Iterations @ 95% Val. Acc."). [verifier 2026-09-02] Note that §2.2 phrases the same criterion as "the number of iterations taken to make the validation accuracy reach 95% of the training accuracy" (quoted in B.5) — the paper itself uses both wordings; "every speedup in the paper" was the writer's overreach, since MNIST/QM9/IMDb (§4.2–4.4) report different quantities (delay factor, minimum validation loss, accuracy), so the sentence is restricted to the algorithmic task.

### B.2 The method: equations and both algorithms

Filter view (§2.1) `[VERBATIM]`: Eq. (3) "ĝ(t)=g(t)+h(t)∗g(t)"; Eq. (4) "û(t)=u(ĝ(t),t)=u(g(t)+h(t)∗g(t),t)"; Eq. (5) "Ĝ(ω)=G(ω)+H(ω)G(ω)=(1+H(ω))G(ω)"; Eq. (6) (windowed moving average) "h(t)=(λ/w)Π(t/w−1/2)={λ/w, if 0≤t<w; 0, otherwise}".

Section 3 (the EMA variant), complete `[VERBATIM]`:

> "In the previous section, we empirically prove that using an LPF to the sequence of model parameter updates leads to faster generalization under the grokking phenomenon. However, for practical purposes, we require an LPF design with a smaller memory footprint. To this end, we modify Algorithm 1 by replacing the windowed moving average with an exponential moving average (EMA) filter. The impulse response of the filter becomes: h(t)=λ(1−α)∑τ=0..t α^τ δ(t−τ)=λα^t(1−α), [Eq. 7] where δ(t) is the discrete unit impulse at the origin. This filter also has two hyperparameters: the scalar factor λ and the scalar momentum α. The corresponding Algorithm 2 only requires additional memory with the same size of the model itself, reducing ×50 amount of required memory compared to Algorithm 1."

Recommended ranges `[VERBATIM]`: "From our empirical studies, we recommend λ∈[0.1,5] and α∈[0.8,0.99]."

Algorithm 1 (Grokfast-MA) `[VERBATIM]`, line by line:

```
Algorithm 1 Grokfast-MA
1: Param: window size w, scalar factor λ.
2: Input: initial parameters θ0, stochastic objective function f(θ), optimizer's parameter update u(g,t)
3: begin: t←0; Q←Queue(capacity=w)
4: while θt not converged do
5:   t←t+1
6:   gt←∇θf(θt−1): Calculate gradients.
7:   Insert(Q,gt): Insert gradients to Q.
8:   ĝt←gt+λ⋅Avg(Q): Filter gradients.
9:   ût←u(ĝt,t): Calculate update.
10:  θt←θt−1+ût: Update parameters.
11: end while
```

Algorithm 2 (Grokfast-EMA, "Grokfast") `[VERBATIM]`, line by line:

```
Algorithm 2 Grokfast-EMA (Grokfast).
1: Param: scalar momentum α, factor λ.
2: Input: initial parameters θ0, stochastic objective function f(θ), optimizer's parameter update u(g,t) from gradient g at timestep t.
3: begin: t←0; μ←θ0: EMA of gradients.
4: while θt not converged do
5:   t←t+1
6:   gt←∇θf(θt−1): Calculate gradients.
7:   μ←αμ+(1−α)gt: Calculate EMA.
8:   ĝt←gt+λμ: Filter gradients.
9:   ût←u(ĝt,t): Calculate update.
10:  θt←θt−1+ût: Update parameters.
11: end while
```

Line 3 was re-fetched specifically; the HTML source reads `\mu\leftarrow\theta_{0}` — the paper literally initialises the gradient EMA to the **initial parameters θ₀**. The official code does not do this (see B.3). `[INFERENCE]` This is most plausibly a typo in the paper, but the source itself does not say so, and no erratum text was found `[NOT FOUND IN SOURCE]`.

[verifier 2026-09-02] Confirmed independently on the v2 HTML: Algorithm 2 line 3 reads "begin: t←0; μ←θ₀: EMA of gradients."; lines 7–8 read "μ←αμ+(1−α)gₜ" and "ĝₜ←gₜ+λμ"; Algorithm 1 line 8 reads "ĝₜ←gₜ+λ⋅Avg(Q)"; Eq. (7) reads "h(t)=λ(1−α)∑_{τ=0}^{t} α^τ δ(t−τ)=λα^t(1−α)"; the recommended-range sentence and the "×50" memory sentence are as quoted. The Section 3 paragraph was verified sentence by sentence; the writer's "complete" quote is accurate. Line 2 of Algorithm 2 was missing its tail "from gradient g at timestep t." — restored above. The v2 "Typo fixed" note in the arXiv Comments field does not identify which typo was fixed; whether line 3 is the one is `[NOT FOUND IN SOURCE]`.

§5.1 on relation to optimizer momentum `[VERBATIM]`: "Instead of using the scaled momentum as a parameter update, we use the smoothened gradient as a *residual*, which is added to the gradient before it is fed into the optimizer." … "The line 7-8 is applied to the gradients independently to the underlying optimizer. The optimizer can be of any type unless it is of the first-order gradient descent-based."

### B.3 The reference implementation (official repository)

`README.md`, `gradfilter_ema` `[VERBATIM]`:

```python
def gradfilter_ema(
    m: nn.Module,
    grads: Optional[Dict[str, torch.Tensor]] = None,
    alpha: float = 0.99,
    lamb: float = 5.0,
) -> Dict[str, torch.Tensor]:
    if grads is None:
        grads = {n: p.grad.data.detach() for n, p in m.named_parameters() if p.requires_grad}

    for n, p in m.named_parameters():
        if p.requires_grad:
            grads[n] = grads[n] * alpha + p.grad.data.detach() * (1 - alpha)
            p.grad.data = p.grad.data + grads[n] * lamb

    return grads
```

Placement `[VERBATIM]`: "Between `loss.backward()` and `optimizer.step()`, insert one of the following line." with `grads = None` "before the training loop". README reproduction commands for the algorithmic task `[VERBATIM]`: baseline `python main.py --label test`; MA `python main.py --label test --filter ma --window_size 100 --lamb 5.0 --weight_decay 0.01`; EMA `python main.py --label test --filter ema --alpha 0.98 --lamb 2.0 --weight_decay 0.005`. README hyperparameter table (as extracted): Algorithmic/EMA α=0.98, λ=2.0, wd=0.005; Algorithmic/MA w=100, λ=5.0, wd=0.01; MNIST α=0.8, λ=0.1, wd=2.0; IMDb α=0.98, λ=2.0, wd=10.0; QM9 α=0.9, λ=1.0, wd=0.01.

`main.py` `[VERBATIM]`: optimizer `optimizer = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2))` with `parser.add_argument("--optimizer", default="Adam")`, `--weight_decay default=0`, `--alpha default=0.99`, `--lamb default=5.0`, `--window_size default=100`, `--p default=97`, `--budget default=3e5`, `--lr default=1e-3`, `--batch_size default=512`; task `result = x * y % p`; split `train_idx, valid_idx = torch.randperm(data.shape[1]).split(data.shape[1] // 2)` (i.e. a 50/50 split; no `--fraction` argument exists).

[verifier 2026-09-02] All of the above re-verified against `README.md` and `main.py`. Additions: `--beta1 default=0.9`, `--beta2 default=0.98`, `--filter default="none"` (so the README's baseline command `python main.py --label test` is Adam, wd 0, no filter). Repository licence (`LICENSE` file, verbatim first line and copyright line): "MIT License", "Copyright (c) 2024 Jaerin Lee"; the README carries the badge "license-MIT". This closes the "not verified here" gap in C.9.

`[INFERENCE]` Three facts follow from the code, none of which the paper text states: (i) the EMA state is initialised to the **first gradient** (first call: `grads[n] = g`, then the update yields `α·g + (1−α)·g = g`); (ii) the reference optimizer is `torch.optim.Adam` with its **coupled** L2 `weight_decay` (added to the gradient inside `step()`, i.e. *after* the Grokfast filter), not decoupled AdamW; (iii) the function-signature defaults (α=0.99, λ=5.0) are not the values the paper reports for the algorithmic task (α=0.98, λ=2.0, see B.4).

### B.4 Experimental setup for the modular-arithmetic task

Appendix B.1 `[VERBATIM]`: "binary operation x⋅y (mod p), with p=97. The network is a two-layer decoder-only Transformer with hidden dimension of 128 and 4 heads in its attention. The positional embedding has length of 5, and GELU and layer normalization is used throughout the network." … "We use cross entropy loss to train the network and an Adam with betas (β₁,β₂)=(0.9,0.98), a constant learning rate of 10⁻³, batch size of 512, and linear learning rate warmup schedule over the first 10 iterations."

§4.1 `[VERBATIM]`: "This is the same modular multiplication task devised to report the grokking phenomenon [Power et al., 2022]."

Training fraction in the paper text: `[NOT FOUND IN SOURCE]` (code: 50/50, B.3). Baseline weight decay in the paper text: `[NOT FOUND IN SOURCE]` (code default `--weight_decay 0`, B.3).

Grokfast-MA setting, §2.2 `[VERBATIM]`: "Choosing the hyperparameters from a simple grid search over λ∈{1,2,5,10} and w∈{2,5,10,20,50,100,200}, we found that the filter works best when λ=5 and w=100."

Grokfast-EMA setting, Figure 8 caption `[VERBATIM]` (figure cross-references lost in the HTML extraction): "Acceleration of delayed generation with Grokfast-EMA (Grokfast). The task is the same modular multiplication as in Figure 4. The amount of acceleration relies on three hyperparameters, the amplifier gain λ, the window size w, and the weight decay wd. Figures [..] and [..] use α=0.98, λ=2.0, and wd=0.005. Figures [..] and [..] show acceleration results when wd=0. Figures [..], [..], and [..] use the same set of hyperparameters unless specified otherwise."

[verifier 2026-09-02] A second extraction resolved the cross-references as sub-panels: "Figures (a) and (b) use α=0.98, λ=2.0, and wd=0.005. Figures (c) and (d) show acceleration results when wd=0. Figures (a), (b), (c), and (d) use the same set of hyperparameters unless specified otherwise." (Panel letters come from the HTML link text and are less certain than the words.) Note the caption says "window size w" although the EMA filter has no window — the paper's own wording. §4.1's third sentence, omitted by the writer `[VERBATIM]`: "Consuming much smaller computational resources (×50 less) compared to Algorithm 1, exponential moving average effectively captures the slow variation of the gradients necessary for accelerating the delayed generalization."

Figure 7 caption `[VERBATIM]`: "The acceleration effect of Grokfast-MA is greatly enhanced when accompanied with appropriate value of weight decay. However, the weight decay alone not always yield beneficial results."

### B.5 Reported speedups

Table 1 (§2.3; iterations to the 0.95 criterion), as extracted: Baseline 39,890 its (×1); 1-Stage Grokfast-MA 2,940 its (×13.57); 2-Stage 1,920 its (×20.78); 2-Stage Slow-only: "Not converged".

[verifier 2026-09-02] Table 1 re-verified. Caption `[VERBATIM]`: "Summary of results of Figure 6." Column headers `[VERBATIM]`: "Name | ĝₜ at (A→B) | ĝₜ at (B→C) | Iterations at acc≥0.95"; the filter columns read gₜ / gₜ (Baseline), gₜ+λ⋅Avg(Q) / gₜ+λ⋅Avg(Q) (1-Stage), gₜ / gₜ+λ⋅Avg(Q) (2-Stage), gₜ / λ⋅Avg(Q) (2-Stage Slow-only).

§2.2 `[VERBATIM]`: "As shown in Figure 4, the number of iterations taken to make the validation accuracy reach 95% of the training accuracy is reduced by ×13.57, which is a remarkable reduction of training iterations."

§2.3 `[VERBATIM]`: "we can accelerate the grokking effect further by ×1.53 by separating the training stage of the model and applying Grokfast-MA only after the model becomes overfitted" … "The results clearly reveal that removing the original gradients leads to much slower and unstable training" … "both the fast and the slow components of the gradients are necessary for faster grokking". [verifier 2026-09-02: the ×1.53 sentence in full reads "As the results show, we can accelerate the grokking effect further by ×1.53 by separating the training stage of the model and applying Grokfast-MA only after the model becomes overfitted, suggesting an adaptive optimizer."]

Table 3 (§4.1 region), caption `[VERBATIM]`: "Quantitative results of Grokfast with a Transformer decoder trained for the algorithmic data (modular multiplication). The experiments corresponds to that of Figure 2 and [..]." Rows as extracted (columns: Iterations @ 95% Val. Acc. | Wall Clock Time @ 95% Val. Acc. (s) | VRAM (MB) | Latency Per Iteration (s)):

| Algorithm | Iterations | Wall clock (s) | VRAM (MB) | Latency/it (s) |
|---|---|---|---|---|
| Baseline | 39890 | 5984 | 290 | 0.15 |
| Grokfast-MA | 790 (×50.49↓) | 292 (×20.49↓) | 458 | 0.37 |
| Grokfast (EMA) | 910 (×43.84↓) | 137 (×43.79↓) | 294 | 0.15 |

The "×50" of the abstract and the Conclusion's "reducing the number of required training iterations by up to ×50" `[VERBATIM]` correspond to the **Grokfast-MA** row of Table 3 (×50.49), not to the EMA variant (×43.84). The Table 3 caption does not state which weight decay each row used `[NOT FOUND IN SOURCE]`; `[INFERENCE]` the gap between Table 1's ×13.57 and Table 3's ×50.49 for the same MA filter is consistent with Figure 7's statement that weight decay "greatly enhance[s]" the MA acceleration and with the README's `--weight_decay 0.01` MA command, but the paper text does not spell this out.

§2.4 `[VERBATIM]` (overhead of the MA variant): "using w=100, the training time per iteration is increased by ×2.4, measured with a single 1080 Ti GPU. Still, the reduction of wall clock time before the delayed generalization of the results in Figure 7 is ×20.5, which is also a notable reduction of time."

Other tasks: §4.2 MNIST `[VERBATIM]`: "With α=0.8, λ=0.1, and wd=2.0, the delay until grokking is reduced by ×22.0. Moreover, the final evaluation accuracy becomes higher from 89.8% to 91.2%." §4.3 QM9 `[VERBATIM]`: "The validation loss drops faster and by a larger margin under Grokfast." (0.00659 → 0.00348 `[EXTRACTED, SENTENCE NOT CAPTURED]`). §4.4 IMDb: "with α=0.98, λ=2.0, and wd=10.0" `[VERBATIM]`; accuracy 0.84 → 0.90 `[EXTRACTED, SENTENCE NOT CAPTURED]`.

[verifier 2026-09-02] The two `[EXTRACTED, SENTENCE NOT CAPTURED]` numbers are located: 0.00659 / 0.00348 are the "Minimum Val. Loss" column of **Table 5 (Appendix D)**, Baseline / Grokfast rows (VRAM 216 / 216 MB; latency 40.2 / 41.4 ms per iteration); 0.84 / 0.90 are the validation accuracies in **Table 6 (Appendix D)**, Baseline / Grokfast. They are table cells, not sentences, so the label was correct but the location was missing. The IMDb sentence in full `[VERBATIM]`: "Figure 11 compares the baseline with the model trained with an optimizer modified by Algorithm 2 with α=0.98, λ=2.0, and wd=10.0."

### B.6 Does Grokfast preserve the phenomenon, or change it? Is the solution the same?

What the paper claims — acceleration of the same event: Figure 2 caption `[VERBATIM]`: "Accelerating generalization of a model under grokking phenomenon. Our Grokfast is a simple algorithmic modification upon existing optimizers to pull forward the time of event of sudden generalization after overfitting, also known as the grokking phenomenon." [verifier 2026-09-02: the writer cut the sentence after "sudden generalization"; the tail "after overfitting, also known as the grokking phenomenon." is restored — it matters because it is the paper's own equation of "the event" with "the grokking phenomenon".] §4.1 `[VERBATIM]`: "Under the grokking phenomenon, the validation loss of the model first increases before it decreases again later during the late generalization stage as depicted in Figure [..] (baseline). … In contrast, models under our Grokfast training algorithms show significantly smaller peak in the validation loss as shown in Figures [..] and [..]. This implies that Grokfast effectively keeps the generalization route B→C close to the global optimum at state C."

What the paper says about the trajectory and the final state, §5.2 `[VERBATIM]`:

> "the model drifts through a significantly longer pathway from the overfitting (state B, 500 steps) to its full generalization (state C, 300k steps), compared to the initial state (state A, 0 steps) to the overfitting state (state B)." [baseline]
> "the ratio between the two distances AB and BC in the parameter space becomes more even." [Grokfast]
> "the model under our algorithm deviates ×8 further from the initial point than the baseline does, with ×5 smaller standard deviation." [state B]
> "Although the generalization accuracy, the training accuracy, the training loss, and the validation loss at the final state (state C) are similar in both the baseline and Grokfast as showcased in Figure 2, we cannot simply say that the states C of baseline and of Grokfast belong to the same network state. Likewise the state B of the baseline and of Grokfast are different."
> "These observations support our interpretation to regard the grokking phenomenon as a state transition between at least three distinct states."

Table 2 caption `[VERBATIM]`: "Parameter space distances between the intermediate models of experiments in Figure 2. Each state corresponds to each of the markers … Average and standard deviations of five instances are shown. Grokfast converges to a nearer point in the parameter space, and the baseline model travels longer." Rows as extracted (Baseline | Grokfast): AB 0.97±2.2 | 7.7±0.42; BC 563.7±199.4 | 15.3±0.84; AC 570.7±199.8 | 35.8±0.98. Text `[VERBATIM]`: "distances AC [between initial and final state] becomes much (×16) shorter with our Grokfast algorithm".

[verifier 2026-09-02] Two corrections to Table 2. (i) The writer's table has **no unit**: the column header reads `[VERBATIM]` "ℓ₂ Distances (×1000)", so the entries are ℓ₂ distances scaled by 10³ (two independent extractions agree on "(×1000)"). (ii) The caption ellipsis reads "Each state corresponds to each of the markers of Figure 12." The ×16 sentence in full `[VERBATIM]`: "the distances AC̄ between the initial and the final state becomes much (×16) shorter with our Grokfast algorithm" (the bracketed gloss in the writer's version was the paper's own words, not an insertion; the overlines AB̄, BC̄, AC̄ are in the source). The load-bearing §5.2 sentence was verified in full from its first word: "Although the generalization accuracy, the training accuracy, the training loss, and the validation loss at the final state (state C) are similar in both the baseline and Grokfast as showcased in Figure 2, we cannot simply say that the states C of baseline and of Grokfast belong to the same network state. Likewise the state B of the baseline and of Grokfast are different." The 500-steps/300k-steps sentence begins "in the baseline setup, the model drifts through…" — writer's quote is the tail of that sentence.

Figure 12 caption `[VERBATIM]`: "Trajectories of model parameters from experiments … projected onto two principal axes of the PCA of the intermediate model parameters of the baseline. The two models travel along distinct pathways in the parameter space with different pace."

A statement that the **learned solution / representation / algorithm is the same** with and without Grokfast: `[NOT FOUND IN SOURCE]`. The paper does not analyse the learned circuit (no Fourier or mechanistic analysis of the accelerated model was found in the fetched text) and explicitly declines to identify the two end states ("we cannot simply say … belong to the same network state"). What it does claim is that the *metrics* at the end state are "similar" and that the phenomenon (overfit first, generalize later) still occurs, earlier.

Sensitivity: the abstract's "with only a few lines of code" `[VERBATIM]`; Figure 4 caption `[VERBATIM]`: "The amount of acceleration relies on the two hyperparameters, amplifier gain λ and window size w."; Figure 8 caption (above) adds weight decay as a third; Figure 7 (above): "the weight decay alone not always yield beneficial results." A limitations paragraph about side effects beyond compute overhead (§2.4): `[NOT FOUND IN SOURCE]`.

[verifier 2026-09-02] Figure 4's caption has a third sentence the writer omitted `[VERBATIM]`: "Each hyperparameter has a sweet spot; increasing one arbitrarily does not guarantee faster acceleration." — direct support for treating (α, λ) as task-tuned, not universal. The `[NOT FOUND IN SOURCE]` above is confirmed and sharpened: §2.4 is literally headed "Limitations" and its content is memory ("it requires w times more memory to store all the previous gradients, limiting its utilization" `[VERBATIM]`) and per-iteration time (×2.4); no other limitation is stated anywhere in the fetched text. The Conclusion sentence in full `[VERBATIM]`: "By amplifying the latter with low-pass filtering, we can bring forward the moment of sudden late generalization, i.e., grokking, reducing the number of required training iterations by up to ×50." Figure 12 caption verified as quoted.

### B.7 GROKVERSE's implementation vs the published rule

`training/grokverse/train.py` lines 37–52 (`apply_grokfast`), called at line 155–156 between `loss.backward()` (l. 154) and `opt.step()` (l. 157); the same function is called from `training/grokverse/analysis/progress_measures.py` lines 108–112 in the same position. Defaults in `training/grokverse/config.py` lines 64–66: `grokfast: bool = False`, `grokfast_alpha: float = 0.98`, `grokfast_lambda: float = 2.0`; preset `"grokfast": {"grokfast": True, "steps": 4000}` (l. 174). Optimizer: `torch.optim.AdamW(..., weight_decay=cfg.weight_decay)` with `weight_decay = 1.0` default (train.py l. 119–122, config.py l. 55).

[verifier 2026-09-02] **Line numbers corrected against the working tree (train.py at commit 25890dc):** `apply_grokfast` is `train.py` l. 56–71 (docstring l. 57–62; skip-if-no-grad l. 64–65; `prev = gf_ema.get(name)` l. 66; EMA/first-gradient branch l. 67–69; `gf_ema[name] = ema` l. 70; `prm.grad.add_(ema, alpha=cfg.grokfast_lambda)` l. 71). It is called in `_train_loop` at l. 338–339, between `loss.backward()` (l. 337) and `opt.step()` (l. 340); `gf_ema: dict = {}` is created at l. 331. The optimizer is built in `train()` at l. 295–298. In `progress_measures.py` the call is at l. 111–112, between `loss.backward()` (l. 110) and `opt.step()` (l. 113), with `gf_ema` at l. 104. `config.py`: `weight_decay: float = 1.0` is l. 56 (not 55); l. 64–66 and l. 174 are correct as stated. The table's inner references "(l. 47–48)", "(l. 48–50)", "(l. 52)" below map to l. 67–68, 67–69, 71 respectively. The function body itself is exactly as the writer describes.

| Aspect | Paper (Alg. 2) | Official code (`gradfilter_ema`) | GROKVERSE `apply_grokfast` | Match? |
|---|---|---|---|---|
| EMA update | line 7: μ←αμ+(1−α)gₜ | `grads[n] = grads[n]*alpha + g*(1-alpha)` | `ema = α·prev + (1−α)·grad` (l. 48–50) | yes |
| Filtered gradient | line 8: ĝₜ←gₜ+λμ | `p.grad.data = p.grad.data + grads[n]*lamb` | `prm.grad.add_(ema, alpha=λ)` (l. 52) | yes |
| Placement | before the optimizer update u(ĝ,t) | between `backward()` and `step()` | between `backward()` and `step()` | yes |
| EMA initialisation | line 3: μ←θ₀ (literal) | first gradient (μ₁ = g₁) | `prev is None → ema = grad.clone()` (l. 47–48), i.e. μ₁ = g₁ | matches the **code**, not the paper's literal line 3 |
| Filtered-out step | — | none | none | — |
| α, λ | recommended ranges λ∈[0.1,5], α∈[0.8,0.99]; algorithmic task α=0.98, λ=2.0 (Fig. 8) | signature defaults α=0.99, λ=5.0; README algorithmic command α=0.98, λ=2.0 | α=0.98, λ=2.0 | matches the paper's algorithmic-task setting and the README command; inside the recommended ranges; does not match the bare function-signature defaults (which the paper never reports as results) |
| Weight decay | wd=0.005 with `torch.optim.Adam` (coupled L2, applied after the filter) | `--weight_decay 0.005`, `--optimizer Adam` | wd=1.0 with `AdamW` (decoupled; unaffected by the filter) | **no** — a different regulariser at a 200× different value |
| Task / model | x·y mod 97, 2-layer, GELU, LayerNorm, batch 512, 50 % split, warmup | same | (a+b) mod 113, 1-layer ReLU, no LN, full batch, frac 0.5 (accelerated runs), no warmup | **no** |
| Parameter coverage | all θ | `p.requires_grad` | `prm.grad is not None` | equivalent for GROKVERSE's models |

Numerical note `[INFERENCE]`: on the first step the reference computes `α·g + (1−α)·g`, GROKVERSE uses `g` exactly; the two can differ in the last floating-point bit, otherwise the recurrences are identical. GROKVERSE's `gf_ema` dict is mutated in place, mirroring the reference; the EMA state is not checkpointed (no resume path exists, so this has no consequence today).

Verdict: **the rule matches the published Grokfast-EMA recurrence (lines 7–8) and the official implementation, including its first-gradient initialisation; it departs from the paper's literal line 3 (μ←θ₀), which the official code also does not follow. The hyperparameters α=0.98, λ=2.0 are the paper's own algorithmic-task values, but they were tuned on a different task, model, batch regime and — importantly — a different weight-decay regime (Adam-coupled 0.005 vs AdamW-decoupled 1.0). No published result covers GROKVERSE's combination.**

Measured in this repository `[MEASURED IN REPO]` (`training/runs/<run_id>/run.json`, `transition` block; thresholds train ≥0.99 / test ≥0.95):

| run_id | grokfast | frac | steps | train_sat | test_gen | final test acc |
|---|---|---|---|---|---|---|
| txf_add_p113_wd1.0_frac0.3_seed0 | no | 0.3 | 40000 | 145 | 8367 | 0.981 |
| txf_add_p113_wd1.0_frac0.3_seed1 | no | 0.3 | 40000 | 145 | 6295 | 0.965 |
| txf_add_p113_wd1.0_frac0.3_seed2 | no | 0.3 | 40000 | 145 | 8367 | 0.994 |
| txf_add_p113_wd1.0_frac0.5_gf2.0_seed0 | yes | 0.5 | 8000 | 179 | 675 | 0.983 |
| txf_add_p113_wd1.0_frac0.5_gf2.0_seed1 | yes | 0.5 | 8000 | 202 | 859 | 0.990 |
| txf_add_p113_wd1.0_frac0.5_gf2.0_seed2 | yes | 0.5 | 5000 | 192 | 716 | 0.985 |
| txf_add_p113_wd1.0_frac0.3_gf2.0_seed0 | yes | 0.3 | 4000 | 150 | **none** | **0.106** |
| mlp_add_p113_wd1.0_frac0.3_seed{0,1,2} | no | 0.3 | 40000 | 156/156/167 | 9646/10357/9646 | 0.994/0.975/0.983 |
| mlp_add_p113_wd1.0_frac0.5_gf2.0_seed{0,1,2} | yes | 0.5 | 5000 | 204/204/204 | 2121/2121/2246 | 0.985/0.987/0.995 |

`[INFERENCE]` The only Grokfast run at the canonical fraction 0.3 (`txf_add_p113_wd1.0_frac0.3_gf2.0_seed0`) did not generalise within its 4,000-step budget — a budget shorter than the un-accelerated crossings (6,295–8,367). Every Grokfast run that grokked was at frac 0.5. So the repository currently contains **no** evidence that Grokfast accelerates grokking at frac 0.3/wd 1.0; the apparent ×9–12 compression (8367→675 etc.) is confounded with the change of `train_frac`, exactly as `docs/RESEARCH_SPEC.md` §3.5 already records.

[verifier 2026-09-02] Run table re-read from all 16 `run.json` files: every number matches. The "×9–12" is loose: pairing seeds gives 8367/675 = 12.4, 6295/859 = 7.3, 8367/716 = 11.7 (so ×7–12), and mean-to-mean 7676/750 = 10.2 (`[MEASURED IN REPO]`, legacy `transition` block). Seed pairing across two different settings has no scientific meaning anyway; the confound argument does not depend on the exact factor. `RESEARCH_SPEC.md` §3.5 is at l. 137–139 and says what the writer reports. Three `txf_mul_p113_wd1.0_frac0.5_gf2.0_seed{0,1,2}` runs also exist (train_sat 190/202/202, test_gen 912/912/1029, final test 0.985/0.989/0.983) — used in C.8.

---

## C. Consequences for GROKVERSE

1. **Definition to cite.** Use Power §1 verbatim as the definition ("long after severely overfitting, validation accuracy sometimes suddenly begins to increase from chance level toward perfect generalization. We call this phenomenon 'grokking'."). Do not attribute "phase transition" to Power — the phrase is absent from that paper; "sudden" is Power's own word, "phase transition" needs a different citation or must be presented as GROKVERSE's description.

2. **Thresholds are a hybrid of the two sources; say so.** GROKVERSE's primary pair (train ≥0.99, test ≥0.95; `config.py` `THRESHOLD_SETS["primary"]`) uses Grokfast's 0.95 validation criterion ("the time to reach the accuracy of 0.95", §2.2; Table 3 "Iterations @ 95% Val. Acc."), whereas Power uses 99 % validation accuracy (Fig. 1 center, §3.1.1). Neither source uses a train-accuracy threshold for the *start* of the gap; GROKVERSE's `grok_gap` (test_gen − train_sat) is its own construct. `RESULTS.md` should state that the 0.95 choice follows Grokfast and that the strict sensitivity set (`sens_strict`, 1.0/0.99) is the one closest to Power.

3. **The train-fraction confound is not a technicality — Power quantifies it.** "In the vicinity of 25-30% of data, a decrease of 1% of training data leads to an increase of 40-50% in median time to generalization" (§3.1.1, S₅ product). Moving from frac 0.3 to 0.5 is therefore expected, from the literature alone, to shorten the delay by a large factor independent of any optimizer trick. The Grokfast-vs-un-accelerated timing comparison in `RESULTS.md` §1/§3 must not be read as a Grokfast speedup until the {Grokfast} × {frac} cells of `RESEARCH_SPEC.md` §3.5 have been run — and the failed frac-0.3 Grokfast run (B.7) makes that decoupling the first thing to run, with a budget of at least the un-accelerated crossing (≥ 8,400 steps), before any speedup is claimed.

4. **Do not import the paper's speedup numbers.** ×13.57 / ×20.78 / ×43.84 / ×50.49 were measured on x·y mod 97, a 2-layer GELU/LayerNorm transformer, batch 512, Adam with coupled L2 weight decay 0.005 (or 0), 50/50 split. GROKVERSE's setting (1-layer ReLU, full batch, AdamW wd 1.0, p 113, addition) shares none of the decisive knobs; Figures 7–8 make the acceleration explicitly dependent on weight decay. Only in-repo measurements may be quoted, and they must carry the confound caveat of item 3.

5. **Grokfast runs are timing/robustness evidence, not mechanism evidence.** The source itself refuses to identify the accelerated and un-accelerated end states ("we cannot simply say that the states C of baseline and of Grokfast belong to the same network state", §5.2; Table 2: the Grokfast model ends ×16 closer to initialisation). No statement that the learned solution is the same exists in the paper `[NOT FOUND IN SOURCE]`. Consequently: (a) the sentence in `RESULTS.md` line 27, "Grokfast compresses the *timing* of the same phase transition … it does not manufacture it", is defensible for the *curve shape* (Grokfast Fig. 2, §4.1) but should be softened where it implies the same *solution*; (b) Fourier sparsity, key frequencies, attention splits and progress-measure signatures reported from `*_gf2.0_*` runs must be labelled "accelerated run" and never pooled with un-accelerated runs in a mean; (c) the observed sparsity difference (0.52–0.64 Grokfast vs 0.76 un-accelerated, `RESULTS.md` l. 33–35) is currently explained by "more training"; the §5.2 result offers a second, untested explanation (different end state), which the write-up should mention. The primary mechanistic claims should rest on the three un-accelerated transformer seeds and the three un-accelerated MLP seeds already in `training/runs/`.

6. **Document the paper-vs-code discrepancy in the code.** `apply_grokfast`'s docstring should record: implements Alg. 2 lines 7–8 and the official `gradfilter_ema` initialisation (μ₁ = g₁); the paper's line 3 literally reads μ←θ₀ and is not followed (nor is it followed by the official code). This is the kind of detail a reviewer will check. [verifier 2026-09-02: still open — the current docstring (`train.py` l. 57–62) says only "Grokfast (arXiv:2405.20233): amplify the slow (EMA) gradient component" and mentions neither the initialisation nor the (α, λ) provenance.]

7. **Weight decay: two different objects.** GROKVERSE's wd 1.0 is Power's (and Nanda's) AdamW value ("AdamW optimizer with learning rate 10⁻³, weight decay 1", A.1.2) — keep it. But the Grokfast α/λ were tuned under `torch.optim.Adam(weight_decay=0.005)`, i.e. coupled L2 added to the gradient after filtering. `RESULTS.md`/`THIRD_PARTY.md` should say that GROKVERSE transfers only (α, λ) and keeps its own decoupled wd 1.0, and that this combination is untested in the source. Power's claim about weight decay is about **data efficiency** ("more than halving the amount of samples needed", §3.3), not about time-to-grok at fixed data; phrase any weight-decay statement accordingly. [verifier 2026-09-02: one Power sentence *does* connect weight decay to the learned representation and may be cited for the embedding-structure story — §3.4: "The structure is more apparent in networks that were optimized with weight decay." It is qualitative (t-SNE of the output layer) and says nothing about timing.]

8. **The `mul` runs have a known algebraic twin.** Power §3.2: modular multiplication mod p on non-zero residues is "indistinguishable from the neural network's perspective" from addition mod p−1 (primitive-root relabelling). GROKVERSE's `data.py` builds all p² pairs including a=0 or b=0 (225 of 12,769 rows map to 0 for `mul`), so the `txf_mul_*` task is "addition mod 112 on a cyclic group of order 112, plus an absorbing zero". Any add-vs-mul comparison should be framed with this equivalence (it predicts similar behaviour, which is what the three `txf_mul_p113_wd1.0_frac0.5_gf2.0_seed*` runs show: test_gen 912/912/1029 vs 675/859/716 for add, same setting) rather than as two unrelated operations.

9. **Attribution gap.** `THIRD_PARTY.md` currently has no Grokfast entry (grep: no match), although `apply_grokfast` is functionally the official `gradfilter_ema` with the same defaults as the README's algorithmic command. Add an entry (repository `https://github.com/ironjr/grokfast`, paper arXiv:2405.20233 v2) per `PROMPT.md` §6 "Respect licenses"; record the repository licence after checking it (not verified here). [verifier 2026-09-02: grep re-run on 2026-09-02, still no match. Licence now verified from the repository's `LICENSE` file: "MIT License", "Copyright (c) 2024 Jaerin Lee" — an MIT entry can be written directly. Also note `THIRD_PARTY.md` l. 36 says the implementation is "written from the papers' descriptions rather than copied"; for `apply_grokfast` that is accurate only if the first-gradient initialisation is credited to the official code, since the paper's Alg. 2 line 3 says something else (B.2).]

10. **Correct the author list wherever "Lee, Kim, Choi, Kim" appears.** arXiv lists Jaerin Lee, Bong Gyun Kang, Kihoon Kim, Kyoung Mu Lee. `RESULTS.md` l. 82 currently says "Lee et al. 2024", which is fine; do not expand it to the wrong names.

11. **Architecture provenance.** Power's model is a 2-layer decoder-only transformer with loss "only on the answer part of the equation" (A.1.2) trained with minibatches and warmup; GROKVERSE's 1-layer full-batch model is Nanda et al.'s, and `PLAN.md` correctly names Nanda as the reproduction target. The literature paragraph in `RESULTS.md` should keep Power as the *phenomenon* citation and Nanda as the *recipe* citation, and cite Grokfast only for the accelerated runs.

12. [verifier 2026-09-02] **Repository cross-checks of items 1–11.** All file references were re-read on 2026-09-02: `RESULTS.md` l. 27 (the "compresses the *timing* of the same phase transition … does not manufacture it" sentence), l. 33/35 (0.52–0.64 vs 0.76 sparsity), l. 82 ("Lee et al. 2024, Grokfast"); `config.py` l. 26–30 (`THRESHOLD_SETS`: primary (0.99, 0.95), sens_strict (1.0, 0.99)); `data.py` l. 29–34 (all p² pairs, `mul` includes a=0/b=0; 2·113−1 = 225 zero-product rows of 113² = 12,769); `RESEARCH_SPEC.md` §3.5 l. 137–139; `THIRD_PARTY.md` (no Grokfast entry). Every claim in items 1–11 about repository content is accurate. One addition to item 1: `RESULTS.md` l. 23 and l. 27 use "suddenly"/"sharp jump" — both are within Power's own vocabulary ("suddenly", §1); only "phase transition" (l. 27) is not. One addition to item 5: `RESULTS.md` l. 21 reports the test loss climbing to ~26 during the plateau; Power §3.1 + Figure 4 document exactly this validation-loss double descent, so that observation *can* be cited to Power (A.1, verifier addition).
