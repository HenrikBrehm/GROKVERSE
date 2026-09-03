# Khanh 2026 — "At-Grok Is Not Converged: A Measurement-Validity Audit for Grokking Representation Metrics"

| item | value |
|---|---|
| arXiv | [2607.06639](https://arxiv.org/abs/2607.06639), v1, "Tue, 7 Jul 2026 14:44:18 UTC (193 KB)" |
| author | Truong Xuan Khanh (submission history: "From: Xuan Khanh Truong") |
| listing comments | "26 pages, 7 figs, 3 tables" |
| code | "https://github.com/ClevixLab/grokking-compression-clock" (Sec. 11, "Code and data") |
| sources read | `https://arxiv.org/abs/2607.06639` (listing), `https://arxiv.org/html/2607.06639` (full LaTeXML HTML, downloaded and converted to text; all quotes below are from it unless marked), `https://arxiv.org/pdf/2607.06639` (downloaded, not used — the HTML was complete) |
| note written | 2026-09-02, for the GROKVERSE architecture study (`docs/dev/RUN_FORMAT_V2.md`, `docs/RESEARCH_SPEC.md`) |
| verified | [verifier 2026-09-02] source re-fetched independently: `abs` listing via WebFetch; `html` via curl (324,240 bytes, LaTeXML) converted to text with math `alttext` kept, plus section-by-section WebFetch prompts as a second reading. Every quote, number and location below was re-checked; corrections and additions are marked "[verifier 2026-09-02]". |
| updated | [pass 2026-09-03] third independent reading for the run-format v2 / 25k-budget decision: `abs` and `html` re-fetched with ten section-specific WebFetch prompts; raw HTML re-downloaded with curl (324,240 bytes, same size as the verifier's copy) to `docs/sources/_raw/khanh_2607.06639.html` and converted to text with math `alttext` kept (`docs/sources/_raw/khanh_2607.06639.txt`). Every number and quote below was re-checked against that text; no correction to the 2026-09-02 content was needed. Two errors appeared only in the WebFetch summaries, not in this note — "1/d initialization" (raw HTML: `$1/\sqrt{d}$`, Sec. 6) and Table 3 ε=0.05 "1.46 / 1.42" (raw HTML: add 1.36, mult 1.46) — so summaries must not be quoted without the raw text. Additions are marked "[pass 2026-09-03]": §6.2b (budget check against the *measured* `arch25k` crossings), §6.3 item 2 (the PREREG_BRIEF convergence rule vs. the paper's criterion), §7 ledger. |

Conventions in this note: every load-bearing statement is a verbatim quote with its section / table / figure
number. `[NOT FOUND IN SOURCE]` marks a requested item the paper does not contain. `[OWN COMPUTATION]` marks
arithmetic done here (Python, see the Consequences section). Nothing below is filled from memory; the one
place where memory would have been tempting is flagged `[FROM MEMORY - UNVERIFIED]`.

Section numbers: the HTML numbers only the eleven top-level sections and Appendices A–C; every sub-number
below (1.1–1.5, 3.1–3.4, …) is this note's own enumeration of the paper's 42 *unnumbered* run-in paragraph
headings (`ltx_title_paragraph`, e.g. "Task and model.", "Clocks."), in order of appearance [verifier 2026-09-02:
checked against the raw HTML; the writer's original wording "follow the HTML" was inexact]. 1 Introduction (1.1–1.5), 2 Related Work (2.1–2.5), 3 Setup and definitions
(3.1 Task and model, 3.2 Norm budget, 3.3 Effective rank, 3.4 Clocks), 4 At-grok is not converged (4.1, 4.2),
5 Generalization precedes compression (5.1, 5.2), 6 What modulates the lag (6.1–6.6), 7 A transformer
dose-response (7.1–7.4), 8 Why a tested audit (8.1–8.3), 9 Takeaways (9.1, 9.2), 10 Limitations (10.1–10.5),
11 Conclusion, Appendix A (depth law), Appendix B (sensitivity, Table 3), Appendix C (toolkit catalog).

---

## 0. Abstract (arXiv listing, verbatim)

> On modular arithmetic, a network's embedding keeps compressing for tens of thousands of steps after it has
> already generalized. Reading effective rank at the grokking transition overstates the converged value by
> 3-5x on an MLP, and by 1.3-1.5x on a transformer trained to convergence; on the MLP it also erases which
> cells compress at all. Compression lags the accuracy transition by an amount on the order of the
> time-to-grok, at least 10,000 steps, rather than coinciding with it. A one-variable ablation shows what sets
> the lag size: adding LayerNorm to an otherwise identical transformer moves the fraction of compression done
> by the grok step from 0.87 to 0.25, and a pre-registered control rules out scale invariance as the
> mechanism. We package this as an audit that separates onset from compression, flags censoring, excludes
> boundary cells that never fully generalize, and checks that the reference floor has plateaued, with an
> adversarial suite that caught a false-confidence bug in our own branch. A secondary, MLP-specific depth law
> linking norm budget to converged floor fails a generality test on a transformer and flips sign under free
> weight decay. Code and the toolkit are released.

The in-paper abstract (HTML) is longer and adds the phrasing that matters for us: "a value read at the
grokking transition overstates the converged circuit by $3$–$5\times$ and, on the modular MLP, inverts the
very trend it is used to report" and "Contra readings in which low-rank compression coincides with the
accuracy transition, compression lags it by a large amount (of order $T_{\mathrm{grok}}$, $\geq 10^{4}$
steps)."

---

## 1. Setup

### 1.1 Primary MLP audit (Sec. 3, Sec. 4, Table 2)

- Task and model (Sec. 3.1, verbatim): "We study modular addition and multiplication mod $p$ ($p=59$) with a
  two-layer MLP over learned token embeddings ($d=128$, hidden $H=256$, GELU), trained full-batch with AdamW
  and decoupled weight decay. This is the standard grokking setting; our analysis is about when
  representation structure is measured, not about the architecture."
- Control knob = a norm clamp, not free weight decay (Sec. 3.2, verbatim): "We use this knob in a matched
  form, holding the global parameter norm at $\rho\lVert W\rVert_{c}$ throughout training, where
  $\lVert W\rVert_{c}$ is the norm the free dynamics reach at grokking. Sweeping $\rho$ gives a family of runs
  that grok at different speeds; we ask how the post-grok representation evolves in each."
- Training length (Sec. 4, verbatim): "Training each budget to $2\times 10^{5}$ steps refutes that reading".
  Table 2 caption: "($p=59$, 6 seeds per cell, $2\times 10^{5}$-step budget)".
- Grid (Table 2): "cells (norm budgets $\rho$)" — modular multiplication "7 ($1.05$–$1.35$)", modular
  addition "9 ($1.00$–$1.40$)".
- Seeds: "6 seeds per cell" (Table 2 caption; Fig. 1 and Fig. 3 captions: "6 seeds, real data").
- Train fraction for the clamp MLP runs: `[NOT FOUND IN SOURCE]` (a train fraction, 0.3, is stated only for
  the Sec. 6 harness, see 1.3).
- Learning rate, batch details beyond "full-batch", logging cadence for the MLP runs: `[NOT FOUND IN SOURCE]`.

### 1.2 Free weight-decay replication (Sec. 4.2)

Verbatim: "On a free weight-decay sweep at $p=59$, with no norm clamp at all, the at-grok embedding rank
exceeds the converged floor in every fully-generalizing cell on modular addition (median $37\rightarrow 18$),
modular multiplication ($35\rightarrow 18$), and parity, a non-arithmetic task ($23\rightarrow 3$): a
$2$–$7\times$ overstatement, with parity showing the largest transient (transient_replication.py)."
Weight-decay values for this sweep, seeds, and steps: `[NOT FOUND IN SOURCE]` in Sec. 4.2 itself (Sec. 6
gives MLP decays {0.5, 1, 2, 4} for the harness; whether Sec. 4.2 uses the same sweep is not stated).

### 1.3 One-variable harness: MLP vs. canonical transformer vs. +LayerNorm (Sec. 6)

Verbatim: "we run a single free-weight-decay harness that changes one thing at a time: arm A is the MLP
anchor; arm B is a canonical attention model that is LayerNorm-free (ReLU MLP block, no biases, $1/\sqrt{d}$
initialization); arm C is arm B with LayerNorm added and nothing else. A-vs-B isolates architecture; B-vs-C
isolates LayerNorm. Each arm is swept under free weight decay (MLP $\{0.5,1,2,4\}$; transformer
$\{1,1.5,2,3\}$), $p=59$, train fraction $0.3$, up to $6\times 10^{4}$ steps, every quantity computed by the
same per-seed clock as the MLP audit."

Power (Sec. 6.3, verbatim): "This harness changes one variable cleanly but is not high-powered: it uses few
seeds and a $6\times 10^{4}$-step budget". Exact seed count: `[NOT FOUND IN SOURCE]`.

Pre-registered control arm (Sec. 6.6, verbatim): "a control arm tf_rms that adds RMS-normalization on the
embedding input only — per-token scale invariance, with no mean-centering, affine gain, or per-sublayer norm
— run under the same harness"; acceptance "iff both PD1 (deferral: $\mathrm{frac\text{-}pre}<0.40$) and PD2
(norm-driven: post-grok $\mathrm{Spearman}(\lVert W\rVert,\mathrm{rank})<-0.30$)" (thresholds recovered from
the HTML math `alttext`; the surrounding sentence was garbled by my HTML-to-text conversion, the two
inequalities and the "iff both" structure are verbatim). [verifier 2026-09-02: both `alttext` strings confirmed in
the raw HTML, `\mathrm{frac\text{-}pre}<0.40` and `\mathrm{Spearman}(\lVert W\rVert,\mathrm{rank})<-0.30`; the Fig. 5
caption states the same thresholds in plain text: "frac-pre for tf_rms is $0.90$, above the $0.40$ threshold and
indistinguishable from the canonical transformer ($0.88$)" and "($-0.56$, below the $-0.30$ threshold)". Note the
paper-internal wobble: the canonical transformer's frac-pre is $0.87$ in Sec. 6.2 / Fig. 4B and $0.88$ in Sec. 6.6 /
Fig. 5A; do not "correct" one to the other.] Result: "tf_rms satisfies PD2 (post-grok Spearman
$=-0.56$) but decisively fails PD1, with $\mathrm{frac\text{-}pre}=0.90$" (Sec. 6.6).

### 1.4 Well-powered transformer sweep (Sec. 7)

Verbatim: "We therefore run a well-powered transformer sweep to true convergence, a one-layer un-normalized
attention model on modular addition ($p=59$), under both the clamp protocol (seven budgets
$\rho\in[0.9,1.4]$) and a free weight-decay protocol (six decays), three seeds each, with dense post-grok
logging of five spectral measures at four weight loci and the activation representation."

Seed fragility (Sec. 7.4, verbatim): "The un-normalized transformer groks erratically across seeds at this
configuration: only 16 of 21 clamp cells and 14 of 18 free-decay cells reach $0.90$ test accuracy, with grok
times for the same budget ranging from $\sim 10^{3}$ to $>10^{5}$ steps and one seed failing to grok at all.
We therefore restrict every transformer statistic above to cells that fully generalize and whose floor has
plateaued".

Transformer width, heads, MLP width, learning rate, total step budget: `[NOT FOUND IN SOURCE]` (only "trained
to true convergence" and grok times "$>10^{5}$ steps", which implies a budget above $10^{5}$ but no number is
given). Note that arm B of Sec. 6 "is" the Sec. 7 model: "that section's un-normalized model is arm B here"
(Sec. 6.5).

---

## 2. Metrics audited, definitions, and the numbers

### 2.1 Definitions

Effective rank (Sec. 3.3, verbatim): "For a weight matrix $M$ with singular values $\sigma_{i}$ we report the
variance-normalized spectral-entropy effective rank
$\exp\!\big(-\sum_{i}\bar{\sigma}_{i}\log\bar{\sigma}_{i}\big)$, $\bar{\sigma}_{i}=\sigma_{i}^{2}/\sum_{j}\sigma_{j}^{2}$
— the squared (explained-variance) normalization used in the complexity-of-grokking work we build on [6].
The effective-rank construction is due to Roy and Vetterli [17], who normalize the singular values directly
($\bar{\sigma}_{i}=\sigma_{i}/\sum_{j}\sigma_{j}$); the form we use squares them (equivalently it is the
spectral entropy of the eigenvalues of $M^{\top}M$). Both are maximized by a flat spectrum and equal one at
rank one". The transformer figures use the un-squared form: "Effective rank (Roy–Vetterli)" (Fig. 6 caption). [verifier
2026-09-02: the Sec. 6 harness, by contrast, uses the squared form — Fig. 4 caption: "Embedding squared-normalized
effective rank under free weight decay, $p=59$" — so the 0.87 / 0.66 / 0.25 frac-pre values and the 1.5× / 1.5× /
3.2× harness transients are squared-form numbers, while the Sec. 7 1.34× / 1.48× medians are Roy–Vetterli.]

Locus: the embedding matrix throughout the MLP audit ("embedding effective rank", Sec. 4); on the
transformer "the embedding $W_{E}$ (left) and unembedding $W_{U}$ (right)" (Fig. 6) plus "$W_{\mathrm{out}}$"
and "the activation representation" (Sec. 7.1, Sec. 7.3).

Participation ratio and stable rank: used as "metric-agnostic" cross-checks (Sec. 4.2, Sec. 7.1); their
formulas: `[NOT FOUND IN SOURCE]`.

frac-pre (Sec. 6.2, verbatim): "Define
$\mathrm{frac\text{-}pre}=(r_{\mathrm{init}}-r_{\mathrm{grok}})/(r_{\mathrm{init}}-r_{\mathrm{floor}})$, the
fraction of the total embedding-rank compression already completed by the grok step."

### 2.2 Overstatement at the grokking transition vs. converged

MLP, clamp protocol (Sec. 4 and Sec. 4.1, verbatim):
- "On modular addition, effective rank falls from $\approx 50$ at grok to $\approx 7$ at convergence in the
  fast-grokking cells, while the slowest budget still falls from $\approx 10$ to $\approx 4$."
- "at every budget that compresses it sits $3$–$5\times$ above the converged value (median $3.3\times$)."
- "the boundary cell ($\rho=1.00$, converged rank $\approx 42$) and its neighbour ($\rho=1.05$, converged rank
  $\approx 12$) read $46$ and $48$ at grok, a $4\%$ difference that hides a $3.6\times$ difference in the
  converged solution."
- Fig. 2 caption: "the at-grok curve overstates it by $3$–$5\times$ and is nearly flat where the converged
  curve falls steepest."

MLP, free weight decay (Sec. 4.2): "$2$–$7\times$ overstatement" (medians 37→18, 35→18, 23→3 as quoted in
1.2). Metric-agnostic: "On the same free-decay addition cells the at-grok overstatement appears in
participation ratio ($2.2\times$) and stable rank ($1.4$–$1.7\times$) just as in effective rank
($2.5$–$2.6\times$), and the one decay setting with no effective-rank transient shows none in the other two
either".

[verifier 2026-09-02, addition] Sec. 4.2 also cites an independent replication of the transient on a different
metric (verbatim): "Brown et al. [18] report the same qualitative signature while tracking the last-layer intrinsic
dimension of activations across grokking: on their slower-grokking modular-division runs the intrinsic dimension
shows a transient rise at the onset of generalization, followed by a descent to a value below the earlier plateau".

Harness, all three arms (Sec. 6.1, verbatim): "The at-grok transient exceeds the converged floor on all three
arms: $\approx 1.5\times$ on the MLP, $\approx 1.5\times$ on the canonical transformer, and $\approx 3.2\times$
on the LayerNorm transformer."

Transformer, well-powered sweep (Sec. 7.1, verbatim): "On the clamp sweep, the embedding effective rank read
at grokking sits above its converged floor in $92\%$ of generalizing, non-censored cells (median
$1.34\times$); the unembedding overstates by $1.48\times$ in every such cell, and the free-decay sweep shows
the same ($1.36\times$ and $1.38\times$; Fig. 6). The overstatement is metric-agnostic, appearing in
Roy–Vetterli and variance-normalized effective rank, participation ratio, and stable rank alike, and present
at every locus we log: it is largest on the MLP-block output weights ($W_{\mathrm{out}}$, up to $3.6\times$ at
the smallest budget)". "The magnitude is more modest than the $3$–$5\times$ overstatement of the modular MLP:
on the transformer the embedding transient is $\approx 1.3$–$1.5\times$."

Which coefficient is the reliable one (Sec. 6.3, verbatim): "the most dramatic magnitude (the
$\approx 3.2\times$ LayerNorm transient) comes from it, whereas the well-powered transformer sweep of
Section 7, trained to true convergence, gives a more modest $1.3$–$1.5\times$ embedding transient."

### 2.3 The lag $T_{\mathrm{compress}} - T_{\mathrm{grok}}$

MLP (Sec. 5, verbatim): "Compression follows grokking by a lag comparable to the time-to-grok itself: median
lag $1.7\times 10^{4}$ steps on multiplication ($\mathrm{lag}/T_{\mathrm{grok}}=1.04$) and $1.8\times 10^{4}$
on addition ($\mathrm{lag}/T_{\mathrm{grok}}=1.00$)". Table 2: "median gap $T_{\mathrm{compress}}-T_{\mathrm{grok}}$" = 17,000 (mult) / 18,000 (add).

The tolerance-independent form (Sec. 5, verbatim): "Its magnitude is of order $T_{\mathrm{grok}}$ and always
$\geq 10^{4}$ steps; the exact ratio $\mathrm{lag}/T_{\mathrm{grok}}$ depends on the compression tolerance
$\varepsilon$ (which sets how close to the floor counts as "compressed") and ranges over $[0.65,1.46]$ as
$\varepsilon$ varies from $0.20$ to $0.05$ (Appendix B, Table 3). We report $\varepsilon=0.10$, where
$\mathrm{lag}/T_{\mathrm{grok}}\approx 1.0$, as the default, and treat "$\geq 10^{4}$ steps" rather than
"$=T_{\mathrm{grok}}$" as the claim that does not depend on the tolerance."

Ordering (Sec. 5, verbatim): "The norm budget strongly orders $T_{\mathrm{grok}}$
($\rho_{S}(\rho,T_{\mathrm{grok}})=+0.90,+0.85$) but does not order $T_{\mathrm{compress}}$
($\rho_{S}(\rho,T_{\mathrm{compress}})=+0.18,+0.29$)".

Transformer (Sec. 7.2, verbatim): "across the clean transformer cells the median lag is $\approx 2.7\times 10^{4}$
steps ($\mathrm{lag}/T_{\mathrm{grok}}\approx 0.63$), and as on the MLP the compression time is not cleanly
ordered by the budget ($\rho_{S}(\rho,\mathrm{lag})=+0.37,\,p=0.47,\,n=6$ cells — a low-power estimate".

What sets the lag (Sec. 6.2, verbatim): frac-pre "is $0.87$ on the canonical transformer (most compression
precedes grokking, leaving a small residual lag and a small transient), $0.66$ on the MLP, and $0.25$ on the
LayerNorm transformer (most compression follows grokking, producing both the large lag and the
$\approx 3\times$ transient)." And: "the grok-to-compression lag is smallest for the canonical transformer and
large with LayerNorm or on the MLP — but present, not zero, in every arm (the canonical transformer still
settles within $\approx 0.63\,T_{\mathrm{grok}}$ of grokking, Sec. 7)."

[verifier 2026-09-02, additions from Sec. 6.2 and Sec. 6.5 that the writer left out]
- The harness MLP lag is *not* given a number: "We read the lag qualitatively here, since its precise coefficient is
  seed-sensitive when a high frac-pre leaves only a narrow band to the floor" (Sec. 6.2). The only numerical MLP lag
  coefficients in the paper are the clamp-audit ones of Sec. 5 / Table 2.
- Post-grok norm–rank coupling (Sec. 6.2, verbatim): "the post-grok coupling between the global weight norm and the
  embedding rank is strongly negative only under LayerNorm (Spearman $-0.80$) and positive for the canonical model
  ($+0.70$): the LayerNorm arm compresses as its norm grows after grokking, whereas the canonical arm has already
  compressed."
- Cross-architecture timing, Sec. 6.5 (verbatim): "Manir and Rupa [21] find, in a controlled study on modular
  addition, that the apparent Transformer–MLP grokking gap largely dissolves (a $1.11\times$ delay) under matched
  hyperparameters, and they attribute previously reported architecture differences to optimization and
  regularization confounds. Our one-variable ablation localizes one such factor — normalization — as a specific,
  controllable variable inside that picture." (GROKVERSE's own note on that paper,
  `docs/sources/manir_rupa2026_systematic_grokking.md`, records the same $1.11\times$ and that the MLP was the
  *faster* architecture there.)

Cross-paper comparison the author draws (Sec. 2.2): Tang et al. find "test accuracy precedes the topological
transition by $\sim 10^{3}$ steps. Our rank-compression lag is $\sim 2\times 10^{4}$ steps."

---

## 3. The audit methodology

### 3.1 Clocks: onset vs. compression complete

Sec. 3.4, verbatim: "We define $T_{\mathrm{grok}}$ as the first step with median test accuracy $\geq 0.9$,
and $T_{\mathrm{compress}}$ as the first step after $T_{\mathrm{grok}}$ at which the median effective rank
falls within $\varepsilon$ of its final-plateau floor ($\varepsilon=0.1$ of the at-grok-to-floor drop). The
lag is $T_{\mathrm{compress}}-T_{\mathrm{grok}}$."

Note the median is across seeds (per cell) for the headline clocks; Sec. 6 and Sec. 7 use "the same per-seed
clock" (Sec. 6, Sec. 7.2). [verifier 2026-09-02: Sec. 3.4 does not say over what the median is taken; "across seeds"
is an inference from "6 seeds per cell" (Table 2), "lines are per-budget medians" (Fig. 6) and the analyzer's
"per-seed CSV" (Appendix C) — read it as `[OWN INFERENCE]`, not verbatim.]

The floor (Sec. 6.4, verbatim): "The converged floor, computed over the final tenth of training"; Appendix B
default "floor frac $0.10$" (Table 3 caption).

### 3.2 The five checks (Sec. 1.5 and Sec. 8)

Sec. 1.5, verbatim: "Given per-step logs of a representation metric and test accuracy across norm budgets,
the audit: (i) separates the onset clock $T_{\mathrm{grok}}$ (first step at the grokking accuracy threshold)
from the compression clock $T_{\mathrm{compress}}$ (first post-onset step at which the metric settles within
a tolerance of its floor); (ii) gates boundary cells that never fully generalize or never compress before
computing any ordering statistic; (iii) verifies the floor itself has plateaued, since the same transient
hazard can recur on the denominator; (iv) returns low-power / descriptive only when fewer than two
non-boundary cells remain; and (v) declines a clock-type verdict whenever the cross-budget order statistic is
undefined. A nine-case adversarial suite fixes each verdict and its reason in advance."

Sec. 8, verbatim (same list, with the censoring rule made explicit): "It (i) flags censoring when cells do
not reach the compression threshold within budget; (ii) excludes boundary/incomplete-grok cells before any
ordering statistic is computed; (iii) emits low-power / descriptive only when fewer than two non-boundary
cells remain; (iv) verifies that the compression floor has itself plateaued before using it as a reference;
and (v) returns large lag, ordering undetermined — rather than asserting a shared clock — whenever the order
statistic across budgets is undefined (too few cells or tied values)."

### 3.3 Boundary gate

Sec. 5.2, verbatim: "On modular addition this is concrete: the $\rho=1.00$ cell sits at effective rank $42.1$
at convergence (versus a typical floor of $7.5$ for the cells that complete), a drop of only $0.09$ relative
to its at-grok value, since it groks only partially and never compresses. The audit detects such cells by
their tiny drop-to-at-grok ratio and high floor, excludes them from the dose-response, and reports them
separately as boundary cells." The partial-generalization symptom (Sec. 4): "accuracy plateaus near $0.99$,
never reaching $1.0$". Defaults (Table 3 caption): "min drop $1.0$, gate thr $0.25$"; the exact formula in
which these two thresholds enter: `[NOT FOUND IN SOURCE]` (only "drop/at-grok ratio below threshold",
Fig. 1D caption).

### 3.4 Censoring and floor plateau

Censoring: cells that "do not reach the compression threshold within budget" (Sec. 8 (i)). On the
transformer: "high-budget cells grok late and were excluded if their floor was still falling at the step cap,
and the result holds on the cells that remain" (Appendix A, generality test 1). Property test (Appendix C):
"censoring monotonic under truncation (less data never yields more confidence)".

Floor plateau — why (Sec. 1.2, verbatim): "the "converged floor" against which we judge the at-grok value can
itself be transient. On the lowest-budget MLP cells the embedding plateaus after grokking and then, well
after it, undergoes a second collapse as the global weight norm roughly doubles (Sec. 6, Fig. 4C). A floor
read over the final tenth of training would sit inside this second regime. [...] a careful audit verifies
that the floor has plateaued, not merely that training has run long." The concrete numbers (Sec. 6.4): "the
embedding groks at rank $\approx 22$, holds a plateau, and then, well after grokking, undergoes a second
transition in which the global weight norm roughly doubles and the rank collapses to $\approx 4$". The
numerical plateau test (slope / window / tolerance): `[NOT FOUND IN SOURCE]` — the paper states that the
check exists and is "part of the verdict" (Sec. 10.2) but does not print its formula.

### 3.5 Verdict vocabulary and free parameters

Analyzer verdicts (Appendix C, `analyze_compression_clock_v1_5.py`): "one-clock / two-clock /
partially-separated / large-lag-ordering-undetermined / low-power / censoring-inconclusive".

Free parameters and defaults (Appendix B, verbatim): "The audit exposes five thresholds: the compression
fraction $\varepsilon$, the floor-averaging window, the grokking accuracy threshold, the minimum at-grok drop,
and the boundary drop-threshold." Defaults (Table 3 caption): "$\varepsilon=0.10$, floor frac $0.10$, grok thr
$0.90$, min drop $1.0$, gate thr $0.25$". Robustness: "In 28 of the 30 settings the verdict is partially
separated + large lag"; "The lag magnitude stays of order $T_{\mathrm{grok}}$ ($\geq 10^{4}$ steps)
throughout; only its calibration moves." [verifier 2026-09-02, the two exceptions, verbatim: "in 2 ($\varepsilon=0.20$
on multiplication and grok thr $=0.99$ on addition) the same-signed ordering is strong enough that the analyzer labels
it one clock + large lag, i.e. a single ordering with the lag still present, not a coincident clock" (Appendix B).]

Table 3 (Appendix B), lag in steps and lag/$T_{\mathrm{grok}}$, the rows that vary $\varepsilon$ and the grok
threshold (transcribed from the HTML table):

| parameter | task | lag | lag/T_grok | ρ_S(T_grok) | ρ_S(T_compress) | ρ_S(floor) |
|---|---|---|---|---|---|---|
| ε=0.05 | add | 24750 | 1.36 | +0.85 | +0.17 | −1.00 |
| ε=0.05 | mult | 25500 | 1.46 | +0.90 | +0.39 | −1.00 |
| ε=0.10 | add | 18000 | 1.03 | +0.85 | +0.29 | −1.00 |
| ε=0.10 | mult | 17000 | 1.03 | +0.90 | +0.18 | −1.00 |
| ε=0.20 | add | 11500 | 0.65 | +0.85 | +0.40 | −1.00 |
| ε=0.20 | mult | 10500 | 0.67 | +0.90 | +0.20 | −1.00 |
| floor frac=0.05 / 0.20 | add | 18000 | 1.03 | +0.85 | +0.29 | −1.00 |
| floor frac=0.05 / 0.20 | mult | 17000 | 1.03 | +0.90 | +0.21 / +0.18 | −1.00 |
| grok thr=0.95 | add | 18750 | 0.97 | +0.91 | +0.31 | −1.00 |
| grok thr=0.95 | mult | 17500 | 0.97 | +0.92 | +0.32 | −1.00 |
| grok thr=0.99 | add | 20000 | 0.93 | +0.91 | +0.30 | −1.00 |
| grok thr=0.99 | mult | 19500 | 0.97 | +0.98 | +0.54 | −1.00 |
| min drop=0.5 / 2.0 | both | unchanged from default | | | | |
| gate thr=0.20 / 0.30 | both | unchanged from default | | | | |

Footnote 1 of Appendix B: the lag/$T_{\mathrm{grok}}$ column of Table 3 "is computed as the ratio of medians
(median gap over median $T_{\mathrm{grok}}$ across cells), giving $1.03$ for both tasks at
$\varepsilon=0.10$; Table 2 instead reports the canonical analyzer's median of the per-cell ratios ($1.00$ on
addition, $1.04$ on multiplication). The two aggregations agree to within $0.04$".

### 3.6 Recommended measurement points

The paper does not publish a step-count or a logging-interval recommendation: `[NOT FOUND IN SOURCE]`. What
it does prescribe is (a) which two points must both be reported — the at-grok value and the converged floor,
as in Fig. 2 ("Embedding effective rank read at grokking (red) vs. the converged floor after long training
(blue)"), (b) that the metric trajectory must be logged densely enough after onset to date
$T_{\mathrm{compress}}$ ("dense post-grok logging", Sec. 7), and (c) that the logging-density question is open:
"Checkpoint-density guidance: a planned ablation that subsamples post-grok checkpoints to report how many are
needed for a reliable $T_{\mathrm{compress}}$, turning the transformer censoring we observed into a concrete
logging recommendation." (Sec. 9.2). The "final" label discipline is stated in Sec. 1.2 (quoted in 3.4) and
Sec. 10.2: "the tool does not solve convergence: it detects one species of non-convergence (a metric still
falling at the read step) and flags censoring, and it requires a floor-plateau check that we make part of
the verdict. We do not claim a general certificate that a representation has converged".

### 3.7 Adversarial suite

Sec. 8.3, verbatim: "The suite ships nine pre-registered cases: clean one-clock, all-censored, high-floor
boundary, true two-clock, single-valid-cell, non-monotone (rank rebounds after falling),
compression-before-grok, tied orderings, and heavy noise. Each case fixes the expected verdict and the reason
before running; a case that passes only by avoiding a crash is treated as a failure." The bug it caught
(Sec. 8.2): a branch "re-introduced a verdict that asserted "one clock" while the underlying rank correlation
was undefined ($\mathrm{NaN}$ Spearman). The nine-case adversarial suite caught this, not because a verdict
string was missing, but because the suite requires a clock-type verdict to be backed by a finite order
statistic." Data-level self-correction (Sec. 5.1): a "smaller grid (four and five budget cells) showed
same-signed $T_{\mathrm{compress}}$ correlations ($+0.40$ and $+0.95$) that invited a "one clock" reading.
Enlarging the grid to seven and nine cells weakened that ordering sharply".

---

## 4. Fourier concentration and top-k metrics

The paper audits effective rank only; it does not measure any Fourier-concentration or top-k metric.
Everything it says about Fourier structure:

- Motivation (Sec. 1): grokking "is increasingly studied through the representation: the Fourier structure of
  embeddings [5], effective rank or spectral-entropy complexity [6], intrinsic dimension, and persistent
  homology [7]."
- Opposite timing of a Fourier measure (Sec. 2.1, verbatim): "Sivasankar [19] show that a permutation-tested
  Fourier circuit-synchronization measure reaches its post-grok level roughly $500$–$3{,}000$ steps before
  grokking on modular addition (mean lead $\approx 1{,}700$ steps) — the opposite timing to the embedding-rank
  compression we date after it. That two representational quantities move in opposite directions across the
  same transition is a direct illustration of why which quantity one reads, and when, determines the
  picture."
- Gini concentration (Sec. 2.1): Wang [15] "report a transient peak in the Gini concentration of the full
  parameter vector that coincides with grokking."
- The hazard applies to Fourier readings too (Sec. 8.1, verbatim): "The same transition-time checkpoint at
  which one reads embedding rank is also where a Fourier decomposition, a persistent-homology diagram, or an
  intrinsic-dimension estimate of the representation would be read — and each of those is a quantity still
  in motion at $T_{\mathrm{grok}}$; indeed a Fourier circuit-synchronization measure and $H_{1}$ persistence
  move in the opposite direction to rank across the transition [19, 7]. A mechanistic-interpretability
  pipeline that characterizes the grokked circuit from a transition-time snapshot therefore inherits the
  transient on whatever metric it uses."
- [verifier 2026-09-02, addition] The same paragraph names the three axes on which such a reading varies (Sec. 8.1,
  verbatim): "three choices determine what such a reading reports, and each varies independently across the
  literature: which quantity is measured (rank falls, while circuit synchronization and topological persistence
  rise), at which locus (the embedding, the first layer, and their product compress on different schedules [9]), and
  when it is read (the onset-to-compression lag we date). Our audit fixes the last axis — it decides whether a reading
  has settled in time — and its metric-agnostic input schema is built to range over the first two."
- Relation to Nanda-style progress measures (Sec. 2.5): "Fourier-feature circuits and restricted-loss progress
  measures [5] [...] measures structure that is meaningful at and around the transition; our audit is
  complementary and prior to it, providing a check on when a transition-time measurement of any such metric
  describes the converged circuit rather than a passing state."
- Clock generality and future Fourier clock (Sec. 9.2): "since the clock definition needs only a quantity that
  falls as the representation compresses, running the full lag measurement under each of these measures is a
  direct extension; the analyzer's input schema accepts any such quantity." "Same-run multi-clock measurement:
  add persistent-homology, LID, and Fourier-Gini clocks to the analyzer". "A metric-timing atlas: because
  different representational quantities cross grokking in opposite directions — a Fourier
  circuit-synchronization measure leads it [19] while embedding effective rank lags it (this work) — placing
  several such clocks on one timeline from a single run would turn "compression relative to grokking" from a
  single number into a per-metric map".
- top-k of anything (top-k Fourier power, top-k singular values): `[NOT FOUND IN SOURCE]`. [verifier 2026-09-02:
  confirmed — the string "top-" does not occur anywhere in the HTML.]
- What "Fourier-Gini" denotes and how Sivasankar's synchronization measure is defined: `[NOT FOUND IN SOURCE]`
  (only cited).

---

## 5. Limitations (Sec. 10, verbatim excerpts)

- Scope of the dose-response (Sec. 10 opening): "The full compression-lag dose-response is established on the
  MLP (two tasks, seven and nine norm budgets); on the enlarged grid the compression time is not ordered by
  the norm budget ($\rho_{S}(\rho,T_{\mathrm{compress}})=+0.18,+0.29$), so we report the large lag (robust
  and replicated across the architectures we tested) and the monotone floor law (robust on the MLP, but
  architecture-specific) as the two MLP findings, and explicitly do not claim a single shared clock."
- No prior work is accused (Sec. 10.1): "We do not exhibit a published study that read at-grok representation
  structure as converged and thereby reached a reversed conclusion, and we attribute no such error to
  specific prior work".
- The denominator hazard (Sec. 10.2): "The reference floor against which the at-grok value is judged can
  itself be a transient [...] it means the tool does not solve convergence".
- Power of the mechanism claim (Sec. 10.3): "The one-variable harness (Sec. 6) isolates LayerNorm cleanly but
  uses few seeds and a $6\times 10^{4}$-step budget; its most dramatic magnitude (the $\approx 3.2\times$
  LayerNorm transient) is the least well-powered number in the paper." Also Sec. 6.3: "the precise
  coefficients ($\mathrm{lag}/T_{\mathrm{grok}}$, the transient multiple) are not pinned down by the present
  data."
- Depth law (Sec. 10.4): "established only on a small modular MLP at one width [...] it does not generalize
  across architecture [...] and its direction is protocol-specific".
- Not measured (Sec. 10.5): "We do not measure persistent homology or intrinsic dimension on the same runs
  [...] The effective-rank metric and the norm-clamp protocol are borrowed, not introduced [...] We make no
  mechanistic claim about why compression lags grokking beyond identifying normalization as the controlling
  variable".
- Transformer seed fragility (Sec. 7.4, quoted in 1.4): 16/21 and 14/18 cells reach 0.90; several clean cells
  "rest on a single generalizing seed" (Sec. 7.2).
- Setting specificity not listed by the author but visible in the setup: one prime ($p=59$), one MLP width
  ($d=128$, $H=256$), grok threshold 0.9, and (for the headline numbers) a norm-clamp intervention rather
  than free weight decay. `[OWN OBSERVATION]`.

---

## 6. Consequences for GROKVERSE

GROKVERSE facts used here (from the repo, not the paper): primary setting `p=113`, addition, `train_frac=0.3`,
`weight_decay=1.0`, `lr=1e-3`, full batch, no Grokfast, fixed `steps=25000`, no early stop, paired seeds 0–9,
transformer without LayerNorm and a 2-layer shared-embedding MLP (`docs/dev/RUN_FORMAT_V2.md`; `config.py`
preset `arch25k`; `RESEARCH_SPEC.md` WP-3 "models/transformer.py has no LayerNorm"). Legacy un-accelerated
generalization crossings (test acc ≥ 0.95, log-grid resolution): transformer 8367 / 6295 / 8367, MLP
9646 / 10357 / 9646 (`docs/BASELINE.md`); all six legacy runs stopped at their crossing, so every legacy
structure number is an at-transition number (`RESEARCH_SPEC.md` §3.9). Headline structure metric: top-8
Fourier power concentration of $W_E$ (0.73 vs 0.44 un-accelerated) — a quantity that *rises* as the circuit
forms (`analysis/fourier.py: frequency_concentration_over_time`), plus spectral entropy / participation ratio
as secondaries (`RESEARCH_SPEC.md` §5 "Metrics" [verifier 2026-09-02: the writer cited §4, which is the hypotheses
section H1–H5; the "Secondary:" metric list is in §5]).

### 6.1 What the paper says about our situation, directly

1. Our legacy headline is exactly the "snapshot" reading the audit targets. The paper's representative
   hazard case (Sec. 1.1) is "sweep weight decay on a modular task, take each run's grokking step as
   the first step test accuracy crosses $0.9$, and report the embedding effective rank (or the representation's
   intrinsic dimension) there as the complexity of the generalizing solution" [verifier 2026-09-02: restored to
   verbatim; the writer had replaced "weight decay" by "[a control parameter]"]. Substitute "architecture" for
   weight decay and "top-8 Fourier concentration" for effective rank and that is the 0.73-vs-0.44 comparison. Sec. 8.1 makes the
   substitution explicit: a Fourier decomposition read at the transition "inherits the transient on whatever
   metric it uses." GROKVERSE's own data already show the transient on our metric: "the longer un-accelerated
   run (0.76) is markedly sparser than the early-stopped Grokfast runs (0.52–0.64)" (`RESULTS.md` §2,
   "Convergence").
2. The two architectures are predicted to differ in *how much of their structure is already formed at the
   crossing*. Our transformer is LayerNorm-free like the paper's arm B (frac-pre 0.87, lag $\approx 0.63\,T_{\mathrm{grok}}$,
   Sec. 6.2 / 7.2); our MLP is closest to arm A (frac-pre 0.66, Sec. 6.2; the lag $\approx 1.0\,T_{\mathrm{grok}}$ is the *clamp-audit*
   MLP number of Sec. 5 / Table 2, not a harness number — the harness MLP lag is read "qualitatively" only, Sec. 6.2
   [verifier 2026-09-02]). Transfer caveat [verifier 2026-09-02]: the paper's MLP is a GELU net with $d=128$, $H=256$
   at $p=59$ (Sec. 3.1) under a norm clamp; GROKVERSE's is a ReLU net with `d_mlp=512` at $p=113$ under free weight
   decay (`docs/dev/PREREG_BRIEF.md`), so 0.66 and 1.0 are orderings to test, not values to expect.
   If that ordering transfers, an event-matched comparison at the crossing compares a transformer that has
   done most of its compression against an MLP that has done two thirds — which is precisely the "mundane
   alternative explanation" `RESEARCH_SPEC.md` §3.9 already lists. The paper's lesson is that the
   architecture difference in *timing* (frac-pre) and the architecture difference in *converged structure*
   are two different findings and must be reported as two numbers.
3. A rising metric is fine for the clock: "the clock definition needs only a quantity that falls as the
   representation compresses" (Sec. 9.2) — apply it to $1-c(t)$ where $c$ is the top-8 concentration, or
   define "settled" symmetrically as $|c(t)-c_{\mathrm{floor}}| \le \varepsilon\,|c(T_{\mathrm{gen}})-c_{\mathrm{floor}}|$.
   `[OWN ADAPTATION]` — the paper does not spell this out.

### 6.2 The 25k-step budget against the paper's lag numbers `[OWN COMPUTATION]`

Ratios, not absolute steps, are the transferable form (the paper: "$\geq 10^{4}$ steps, of order
$T_{\mathrm{grok}}$"; coefficients "not pinned down"). Using the legacy crossings as $T_{\mathrm{grok}}$:

| | transformer seeds (8367, 6295, 8367) | MLP seeds (9646, 10357, 9646) |
|---|---|---|
| budget / $T_{\mathrm{grok}}$ | 2.99, 3.97, 2.99 | 2.59, 2.41, 2.59 |
| post-grok window = 25000 − $T_{\mathrm{grok}}$ | 16633, 18705, 16633 | 15354, 14643, 15354 |
| predicted $T_{\mathrm{compress}}$ at lag/$T_{\mathrm{grok}}$ = 0.63 (un-normalized transformer, Sec. 7.2) | 13638, 10261, 13638 | — |
| predicted $T_{\mathrm{compress}}$ at 1.00 (MLP default ε=0.10, Table 2) | 16734, 12590, 16734 | 19292, 20714, 19292 |
| predicted $T_{\mathrm{compress}}$ at 1.46 (MLP, ε=0.05, Table 3) | 20583, 15486, 20583 | 23729, **25478**, 23729 |

[verifier 2026-09-02: every cell above recomputed with the project Python (`.venv`), all confirmed; the MLP row at
the 0.63 coefficient, which the writer left blank because 0.63 is a transformer-only number, would read 15723 /
16882 / 15723 — still inside 25k.]

Reading: (a) for the LayerNorm-free transformer the budget is comfortable at every coefficient the paper
reports; (b) for the MLP at the paper's default tolerance the predicted settling step is 19–21k, i.e. inside
25k with a 4–6k-step margin, which is thin against the seed-to-seed spread of $T_{\mathrm{grok}}$ (±410 on
n=3 in the legacy runs, and 10 seeds will widen it); (c) at the strict tolerance ε=0.05 one legacy MLP seed
would already be censored. The absolute MLP lags of Table 2 (17,000–18,000 steps) exceed the MLP post-grok
window (14.6–15.4k) — those absolute numbers come from $p=59$, a norm clamp and a different width, so they
are not directly transferable, but they are the paper's most robust statement ("always $\geq 10^{4}$ steps")
and our window clears $10^{4}$ by only ~5k. Conclusion: 25k is a defensible fixed budget *provided the
settling test is actually run and reported per seed*; it is not a budget at which "final" may be renamed
"converged" by assumption. The paper's own Sec. 6 harness at the same train fraction (0.3) and free weight
decay used "up to $6\times 10^{4}$ steps" and still called itself under-powered. If any primary-block seed is
censored under ε=0.10, the fallback is a longer-budget replicate for that architecture (e.g. 50k) rather than
a silently early-stopped comparison.

### 6.2b The same check against the measured `arch25k` crossings `[OWN COMPUTATION, pass 2026-09-03]`

Input: `training/runs/{mlp,txf}_add_p113_wd1.0_frac0.3_seed{0..9}_arch25k/run.json` →
`transitions.primary.generalization.first_crossing_step` (test acc ≥ 0.95, dense evaluation every 25 steps, so each
crossing is an interval `(step−25, step]`). Run metadata only — used here for budget adequacy, not as a result; the
paired timing comparison belongs to the statistics plan and is deliberately not made here. At the time of this
computation all 10 MLP runs and transformer seeds 0–7 had `steps_completed = 25000`; transformer seeds 8–9 were at
step 14,515 and still running (their crossings, at 8,125 and 5,825, are already fixed). Coefficients are the
paper's lag/$T_{\mathrm{grok}}$ values (0.63 un-normalized transformer, Sec. 7.2; 1.00 MLP default ε=0.10, Table 2;
1.46 the largest value in Table 3, ε=0.05 on multiplication), applied as $T_{\mathrm{compress}} = (1+c)\,T_{\mathrm{gen}}$.

| | MLP, seeds 0–9 | transformer (no LayerNorm), seeds 0–9 |
|---|---|---|
| $T_{\mathrm{gen}}$ (0.95), per seed | 9125, 10075, 9175, 8150, 8650, 9375, 9900, 9200, 9800, 9300 | 7975, 6250, 8125, 7425, 10275, 5625, 6325, 7750, 8125, 5825 |
| median / min / max | 9250 / 8150 / 10075 | 7587.5 / 5625 / 10275 |
| 25000 / $T_{\mathrm{gen}}$, median (min) | 2.70 (2.48) | 3.30 (2.43) |
| post-generalization window 25000 − $T_{\mathrm{gen}}$, min / median / max | 14925 / 15750 / 16850 | 14725 / 17412 / 19375 |
| predicted $T_{\mathrm{compress}}$ at c = 0.63, median / max | 15077 / 16422 | 12367 / 16748 |
| predicted $T_{\mathrm{compress}}$ at c = 1.00, median / max (seeds > 20000) | 18500 / 20150 (1: seed 1) | 15175 / 20550 (1: seed 4) |
| predicted $T_{\mathrm{compress}}$ at c = 1.46, median / max (seeds > 25000) | 22755 / 24784 (0; seed 1 has 216 steps of margin) | 18665 / 25276 (1: seed 4, censored) |
| windows shorter than the paper's absolute MLP lag (17k / 18k, Table 2) | 10 of 10 / 10 of 10 | 3 of 10 / 6 of 10 |
| windows shorter than the tolerance-free bound $10^{4}$ | 0 of 10 | 0 of 10 |

Reading, restricted to what the paper licenses (ratios, direction, and the $\geq 10^{4}$ bound; Sec. 5, Sec. 6.3):

1. The tolerance-free claim is cleared by every seed of both architectures: no post-generalization window is
   shorter than $10^{4}$ steps (minimum 14,725).
2. At the paper's default tolerance (c = 1.00, the MLP coefficient) all 20 seeds are predicted to settle inside
   the budget, but the slowest seed of each architecture lands between 20k and 25k — i.e. inside the last fifth of
   training that the PREREG_BRIEF convergence rule (§6.3 item 2) uses as its "already flat" window. For those seeds
   the step-20000 checkpoint would be read while the metric is, by this prediction, still moving.
3. At the strict tolerance (c = 1.46) one transformer seed (seed 4, $T_{\mathrm{gen}}$ = 10,275, the latest crossing
   in the matrix) is predicted to settle 276 steps *after* the budget and would be censored; the slowest MLP seed
   clears it by 216 steps. The strict tolerance is therefore the one at which the censoring flag (§6.3 item 3)
   must be expected to fire, and the sensitivity pair ε = 0.05 / 0.20 is not optional.
4. The paper's absolute MLP lags (17,000–18,000 steps at $p=59$, norm clamp, $d=128$, $H=256$, GELU) exceed every MLP
   post-generalization window in the matrix (max 16,850). Absolute transfer is not expected (different $p$, width,
   activation and protocol), but this is exactly why the study cannot borrow the paper's numbers and must date
   $T_{\mathrm{settle}}$ on its own trajectories, per seed and per metric.
5. The 0.63 coefficient — the only transformer number — leaves ≥ 8,250 steps of margin for every transformer seed.
   If the paper's ordering transfers (un-normalized transformer settles earlier than the MLP relative to its own
   $T_{\mathrm{gen}}$), the event-matched comparison at the crossing is the one most exposed to the timing
   confound (§6.1 item 2), and the budget-matched comparison at 25k is the safer of the two declared points for
   the transformer and the more marginal of the two for the MLP.

Conclusion unchanged from §6.2, now with the real crossings: 25k is adequate for the primary block at the default
tolerance provided the per-seed settling test is run and reported; it is marginal at the strict tolerance for the
slowest seed of each architecture, and the "converged at budget" label must come from the test, never from the
budget.

### 6.3 What a GROKVERSE study must report so a structure metric is not read at the transition

Adapted one-to-one from Sec. 1.5 / Sec. 8 / Appendix B; items marked (new) are not yet in
`RUN_FORMAT_V2.md`.

1. **Two clocks per run and per metric.** $T_{\mathrm{gen}}$ (already: first dense crossing, reported as an
   interval) and $T_{\mathrm{settle}}(m,\varepsilon)$ = first logged step after $T_{\mathrm{gen}}$ at which
   $|m(t)-m_{\mathrm{floor}}| \le \varepsilon\,|m(T_{\mathrm{gen}})-m_{\mathrm{floor}}|$, with ε=0.10 as the
   default and ε=0.05, 0.20 as the mandatory sensitivity pair (Table 3). Report the lag and lag/$T_{\mathrm{gen}}$
   per seed, paired by seed across architectures. (new)
2. **Floor definition and floor-plateau check.** $m_{\mathrm{floor}}$ = median over the final tenth of the
   trajectory (floor frac 0.10; sensitivity 0.05, 0.20). Pre-register a plateau test before the runs — the
   paper requires one but prints no formula — e.g. the final-tenth floor and the preceding-tenth window must
   agree within the same ε, otherwise the run is "floor not plateaued". (new)
   [pass 2026-09-03] `docs/dev/PREREG_BRIEF.md` ("Setting", row "measurement points") has since fixed a rule: "A metric is called "converged at budget" only if its change between the step-20000 and step-25000 checkpoints is below 5 % of its value (Khanh 2607.06639); otherwise "final (budget), not converged"". Two things the citation should not be read to imply `[OWN OBSERVATION]`: (a) the paper prints no plateau formula (`[NOT FOUND IN SOURCE]`, §3.4 above), so the 20k→25k / 5 % rule is GROKVERSE's own and should be cited as *motivated by* the paper, not *taken from* it; (b) the paper's tolerance is normalised by the at-grok-to-floor **drop** ("ε=0.1 of the at-grok-to-floor drop", Sec. 3.4) and its floor is a window median ("computed over the final tenth of training", Sec. 6.4; floor frac 0.05 / 0.20 in Table 3), whereas the PREREG rule is normalised by the metric's **value** and compares two single checkpoints one fifth of the budget apart. The two criteria diverge in opposite directions depending on the metric: for a concentration that moves, say, 0.70→0.76 between the crossing and the floor, 5 % of value (≈0.038) is ≈60 % of the drop — six times looser than the paper's ε; for an effective rank falling 50→7, 5 % of the floor value (≈0.35) is <1 % of the drop — ten times stricter. Recommendation: keep the PREREG rule as the pre-registered *label* rule (it is fixed and must not be changed post hoc), but report the paper-style quantities next to it for every metric — $T_{\mathrm{settle}}(\varepsilon)$ with ε ∈ {0.05, 0.10, 0.20} of the drop, floor = median over the final tenth (sensitivity 0.05 / 0.20) — so that a reader can compare GROKVERSE to the paper on the paper's own definition. The 20k checkpoint is also inside the predicted settling region of the slowest seeds at the default tolerance (§6.2b item 2), so a "converged at budget" verdict for those seeds is the case where the two definitions are most likely to disagree; if they do, say so rather than pick one.
3. **Censoring flag.** A run whose metric has not settled by step 25000 is reported as *censored* for that
   metric; its step-25000 value is a lower/upper bound, never a converged value. Censored seeds stay in the
   manifest and in every table with the flag (Sec. 8 (i); `RUN_FORMAT_V2.md` already bans the word
   "converged" for `final`). (new)
4. **Boundary gate.** Seeds that never reach the generalization threshold, or whose metric moves by less than
   a pre-declared minimum between $T_{\mathrm{gen}}$ and the floor, are reported separately and excluded from
   the paired comparison before any statistic is computed (Sec. 5.2). GROKVERSE has no "did-not-grok" label [verifier 2026-09-02: the string occurs nowhere in the code or
   docs]; what exists is the v2 rule that a threshold crossing which never happens is stored as `null` with the last
   evaluated step (`RUN_FORMAT_V2.md` §3: "A crossing that never happens is `null` with the last evaluated step
   recorded"; `train.py` returns `first_crossing_step: None`), which covers the first case once the paired analysis
   treats a `null` generalization crossing as a boundary seed; the minimum-drop rule (paper defaults: min drop 1.0 rank units, gate thr 0.25 of
   at-grok) needs a concentration-scale analogue fixed in advance. (new)
5. **Both readings, side by side.** Every structure metric at (a) the generalization event checkpoint and
   (b) the final / plateau checkpoint, plus the ratio (b)/(a) per seed — the Fig. 2 layout, one panel per
   architecture. The at-crossing numbers remain valid *as* at-crossing numbers (already the plan in
   `RESEARCH_SPEC.md` §3.9); the audit adds that the headline claim must name which of the two it is about.
6. **frac-pre per architecture.** $(m(T_{\mathrm{gen}})-m(0))/(m_{\mathrm{floor}}-m(0))$ for the top-8
   concentration and for the effective rank of $W_E$ (Sec. 6.2 definition, sign-adapted). This is the single
   quantity the paper found to differ between an un-normalized transformer (0.87) and an MLP (0.66); if our
   two architectures differ in it, part of the at-crossing sparsity gap is timing, not endpoint. (new)
7. **Dense post-generalization logging of the metric, not only of checkpoints.** The v2 grid saves full
   state at 10k, 12k, 14k, 16k, 18k, 20k, 22.5k, 25k — 2–2.5k spacing after grokking — and the legacy
   `embeddings.npy` schedule is geometric (`n_logged_steps=150`: with `steps=25000` the ratio is
   $25000^{1/149}\approx 1.070$, i.e. ≈700-step spacing at step 10k and ≈1400-step spacing at step 20k
   `[OWN COMPUTATION]`; [verifier 2026-09-02: measured directly with `utils.log_step_schedule(25000, 150)`, which
   yields 125 unique logged steps after rounding/dedup — consecutive spacings are 593–679 steps between 9k and 11k,
   954–1021 between 14k and 16k, and 1252–1340 between 19k and 21k, i.e. slightly below the writer's ≈700 / ≈1400
   estimates but the same order]). A $T_{\mathrm{settle}}$ dated on that grid is known only to within one grid
   interval, ≈1.3k steps at the MLP's predicted settling region. Since a $[113,128]$ (MLP `W_E`) or $[114,128]$ (transformer `W_E`: `vocab_size = p + 1` for the `=` token,
   `config.py` [verifier 2026-09-02]) embedding snapshot is cheap, log $W_E$ (and for the
   transformer $W_U$, Sec. 7.1 finds the unembedding transient larger) every 100–250 steps after
   $T_{\mathrm{gen}}$, or log the scalar metrics at that cadence. The paper calls its transformer logging
   "dense post-grok logging" without a number, so the cadence is our choice and must be pre-registered. (new)
8. **Low-power rule.** Any cell with fewer than two non-boundary, non-censored seeds gets a descriptive
   verdict only (Sec. 1.5 (iv)); with paired seeds 0–9 this should not bind on the primary block but will on
   the 3-seed confound cells.
9. **Threshold sensitivity already covers the grok-threshold axis.** GROKVERSE's `THRESHOLD_SETS` (0.95 primary,
   0.90 loose, 0.99 strict) maps onto Table 3's "grok thr" rows, where lag/$T_{\mathrm{grok}}$ moved only
   0.93–1.03; report the clocks under all three.
10. **Effective rank of $W_E$ as a secondary metric.** Adding the paper's exact quantity (squared-normalized
    spectral-entropy effective rank, Sec. 3.3; and the Roy–Vetterli form for the transformer, Fig. 6) costs
    one SVD per snapshot and lets GROKVERSE's numbers be compared to the paper's 1.3–1.5× (transformer) and
    3–5× (MLP) transients directly, at $p=113$ instead of 59. (new, optional)
11. **Language.** "final" ≠ "converged" unless item 2 passes and item 3 is clear for that metric
    (`RUN_FORMAT_V2.md` §3 role table already says this); "compression complete" / "settled" is a per-metric,
    per-ε statement, never a property of the run.

### 6.4 One caution the paper adds against over-reading its own numbers

The transient multiples and lag coefficients above are stated by the author as direction-only outside the
MLP audit: "the direction is consistent across the well-powered sweep and this one-variable harness, but the
precise coefficients [...] are not pinned down by the present data" (Sec. 6.3). GROKVERSE should therefore
cite this paper for (i) the *existence* of a post-grok transient on representation metrics including
Fourier readings (Sec. 4, Sec. 7.1, Sec. 8.1), (ii) the *requirement* to separate onset from settling and to
check the floor (Sec. 1.5), and (iii) the *ordering* MLP-lag > un-normalized-transformer-lag (Sec. 6.2) — not
for a specific step count that a 25k budget can be said to satisfy.

---

## 7. Verification ledger

Verified verbatim in the fetched HTML (section given): all quotes in Secs. 0–5 above. Recovered from the HTML
math `alttext` after my text conversion garbled them: PD1 "$\mathrm{frac\text{-}pre}<0.40$", PD2
"$\mathrm{Spearman}(\lVert W\rVert,\mathrm{rank})<-0.30$" (Sec. 6.6), and "$p<0.01$" for the activation-locus
floor ordering (Sec. 7.3 / Appendix A).

[verifier 2026-09-02] Independent re-verification: all three `alttext` strings above confirmed in the raw HTML
(`p<0.01` occurs exactly twice, Sec. 7.3 and Appendix A generality test 1); every quotation in Secs. 0–5 re-read
against the text; Table 3 re-transcribed cell by cell (no discrepancy); the §6.2 arithmetic recomputed (no
discrepancy). Corrections made in place: the sub-section numbering convention (header), `RESEARCH_SPEC.md` §4 → §5
(§6 intro), the Sec. 1.1 quote restored to verbatim (§6.1 item 1), the arm-A lag provenance (§6.1 item 2), the
non-existent "did-not-grok" label (§6.3 item 4), the transformer `W_E` row count and the measured log-grid spacing
(§6.3 item 7). Additions: Fig. 4 squared-form note (§2.1), Brown et al. transient (§2.2), harness-MLP-lag-is-
qualitative / norm–rank coupling / Manir & Rupa 1.11× (§2.3), the two Appendix B exceptions (§3.5), the Sec. 8.1
three-axes quote (§4), the PD1/PD2 plain-text source and the 0.87-vs-0.88 wobble (§1.3), and the median-over-what
inference flag (§3.1).

Not found in the source (requested by the task brief): learning rate; transformer width / heads / depth beyond
"one-layer"; total step budget of the Sec. 7 transformer sweep; logging cadence for any run; train fraction
of the clamp MLP audit; seed count of the Sec. 6 harness; formulas for participation ratio, stable rank, the
boundary-gate ratio and the floor-plateau test; any top-k metric; any explicit "measure at these steps"
recommendation.

From memory: nothing. `[FROM MEMORY - UNVERIFIED]` would have applied to the definition of Sivasankar's
Fourier circuit-synchronization measure and to Nanda et al.'s "cleanup" phase; both are left uncited here
rather than filled in.

[pass 2026-09-03] Third reading, independent of the two above: `abs` and `html` re-fetched (ten section-specific
prompts), raw HTML re-downloaded with curl (324,240 bytes) and converted with `alttext` kept
(`docs/sources/_raw/khanh_2607.06639.{html,txt}`). Re-confirmed in the raw text: every number in §0–§5, the full
Table 3 (ε rows: add 24750/1.36, 18000/1.03, 11500/0.65; mult 25500/1.46, 17000/1.03, 10500/0.67), the arm-B
initialization `$1/\sqrt{d}$` (Sec. 6), "final tenth of training" (Sec. 1.2 and Sec. 6.4), "dense post-grok logging"
without a cadence (Sec. 7), "few seeds" without a count (Sec. 6.3, Sec. 10.3), the absence of the string "top-"
anywhere in the paper. Two errors were produced by the WebFetch summariser and rejected against the raw text
("1/d initialization"; Table 3 ε=0.05 "1.46 / 1.42"). No correction to the 2026-09-02 text. Additions: header row,
§6.2b (measured `arch25k` crossings vs. the paper's lag coefficients, computed with the project Python from
`run.json`), §6.3 item 2 (PREREG_BRIEF convergence rule vs. the paper's criterion, flagged as `[OWN OBSERVATION]`).
Nothing from memory.
