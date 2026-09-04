# Grokking measurement pitfalls

Collected while auditing GROKVERSE's existing evidence and reading the primary sources (2026-09-02/03).
Each entry is a way a plausible-looking measurement can be wrong. Sources of truth:
`Claude/grokverse/docs/CURRENT_EVIDENCE_AUDIT.md`, `docs/LEGACY_METRIC_AUDIT.md`,
`docs/sources/*.md`.

## 1. Reading a structure metric at the transition instead of at convergence

Khanh 2026 (arXiv:2607.06639) audits exactly this: taking each run's grokking step as the first step test
accuracy crosses a threshold and reporting a representation metric *there*. The measured overstatement is
**3–5x on an MLP and 1.3–1.5x on a transformer** for effective rank, with a compression lag of order
10,000 steps or more after the accuracy transition.

GROKVERSE's legacy runs all used `--early-stop-acc 0.95`, so in all six un-accelerated runs the last
logged step *equals* the generalization step. Every legacy structure number, including the headline
0.73 vs 0.44 top-8 concentration, is therefore an **at-transition** number, not a converged one. Two
architectures that compress at different speeds can produce a "structural difference" that is really a
timing difference.

Fix used: a fixed step budget for every run, plus every structure metric reported at **two declared
points** (the generalization crossing and the final budget step), never mixed in one comparison, and a
convergence check between the last two checkpoints.

## 2. A threshold that never fires, so the "measured" count is a constant

`dominant_frequencies` took the smallest frequency set reaching 90% of power, **capped at 8**. On all 16
legacy runs the cap binds and the 90% threshold never fires — so "top-8 concentration" is an honest name
but the *count* was never determined from data. Fixing k for both architectures is precisely the
assumption that the metric-validity hypothesis questions.

## 3. Top-k concentration punishes a square wave for being a square wave

A square wave at fundamental k puts its power at the odd harmonics 3k, 5k, 7k, which in Z_p alias to
scattered indices. A perfectly clean square-wave circuit therefore spreads its power and scores *low* on
a top-k metric that counts individual frequencies. Numbers at p=113: the discrete square wave holds
0.8107 of its non-constant power in the fundamental and 0.1717 of that in the odd harmonics 3/5/7
(continuous ideal 0.8106 / 0.1715), with a small nonzero even share 5.8e-4.

## 4. "Family concentration beats top-m" is unsatisfiable by construction

Top-m is the argmax over sets of size m, so a harmonic family of the same cardinality can never exceed
it. Any criterion of that form is guaranteed to fail and proves nothing. A harmonic-aware metric must be
compared against a **cardinality-matched** top-m control and a random-set null, and the discriminating
statistic must be the *shape* (odd vs even harmonic energy), not the concentration.

## 5. A mean without a variance is not a mechanism

Averaged over all 12,769 inputs, the read-out position's attention splits ~50/50 between the two operands.
That is equally consistent with a genuinely constant 50/50 split (attention is a fixed sum, so the
transformer is additive-then-ReLU like the MLP) and with strongly input-dependent attention that averages
to 50/50 (a multiplicative path the MLP does not have). Those are different circuits. The fix is to report
per-input variance and quantiles, and to run causal head ablation plus a "fix attention to its mean"
ablation.

## 6. A coarse evaluation grid invents precision

With ~150 logarithmically spaced checkpoints, the spacing near step 8,000 is several hundred steps.
Two transformer seeds reporting the *identical* crossing step 8367 and two MLP seeds the identical 9646
is the signature of different trajectories snapping to the same grid point. Every transition must be
reported as an interval `(previous evaluated step, first crossing]` with its evaluation frequency, and
the phrase "exact transition" must not be used.

## 7. A mask that is wider than the hypothesis

Building a 2D Fourier mask as the outer product of a 1D key-frequency mask keeps *cross-frequency* blocks
such as `cos(w_18 a) cos(w_15 b)`, which the trig-identity circuit never uses. For p=113 and 8 key
frequencies that keeps 289 components where the published operator keeps 17, and the matching excluded
mask deletes 3,360 components (26.3% of the logit tensor) where the published one deletes 16. Both biases
push toward the desired conclusion: restricted loss looks too good, excluded loss looks too destroyed.

## 8. Publishing a replication as a discovery

Manir & Rupa 2026 (arXiv:2603.25009) already report that a grokked transformer's embedding is more
Fourier-concentrated than an MLP's (98.5% vs ~75% top-5 at p=97, one seed), *and* that the
transformer-vs-MLP timing gap moves with hyperparameters — with the timing direction opposite to
GROKVERSE's under their protocol. A concentration gap or a timing gap alone is therefore not a new
contribution.

## Related

[[ReLU as the multiplier in modular addition]] · [[Fourier circuit for modular addition]] ·
[[Transformer vs MLP mechanism comparison]]


---

## Measured outcomes for pitfalls 1 and 2 (added 2026-09-04, after the study ran)

Both were predictions when written. Both are now measured on our own 51-run matrix, at a fixed
25,000-step budget with no early stopping.

**Pitfall 1 — reading at the transition.** The legacy headline was 0.73 (transformer) vs 0.44 (MLP)
top-8 embedding concentration, read at each run's generalization crossing. The same quantity measured
after convergence, over 10 seeds each: **0.961** (transformer, 0.950–0.971) vs **0.891** (MLP,
0.772–0.960). Same direction, **gap 0.07 instead of 0.29**. Khanh's overstatement effect, confirmed on
our data rather than cited from theirs.

**Pitfall 2 — a cap that always binds.** At convergence the top-8 cap binds on **0 of 10 transformer
runs and 6 of 10 MLP runs**, where in the legacy runs it bound on all 16. So the metric is no longer a
constant-k count for the transformer, and still partly one for the MLP.

**A new one, harder to see than either.** The convergence check that pitfall 1's fix promised was not
actually run until 2026-09-04. When it was, it found the MLP's `structured_fraction_of_live` — the
metric behind the study's headline structure gap — settled at the budget in only **2 of 10 seeds** and
still rising, while the transformer's had flattened. A fixed budget removes the *at-transition* bias and
replaces it with a *censoring* bias whose direction has to be measured, not assumed. Writing the fix
into the protocol is not the same as executing it.

## 9. An extractor that returns a plausible number instead of an error

Five defects in this study shared one shape: **well-formed output containing no information**. None
crashed; none failed a test; each was found only because someone went looking for a specific number and
it was not there.

- `aggregate._wave_fitting` guessed flat key names → every waveform column `null`; the H2 figure would
  have been blank.
- `aggregate._progress_measures` read one nesting level too shallow → every restricted/excluded-loss
  column `null`.
- `structure_over_time` compared an **integer model index** to a **model name** (`best_by_aic` has two
  conventions in the same module) → both waveform trajectories were constant **0.0** for the entire
  study.

The third is the dangerous one, and the lesson. A defect that returns `null` announces itself as an
absence. A defect that returns `0.0` looks like a measurement — and downstream, an onset detector
reporting "no onset" for a flat-zero series looks like a finding. In a codebase where `null` legitimately
means "not evaluable", a silent `null` is invisible; a silent `0.0` is worse than invisible, because it
is quietly *persuasive*.

**What actually catches this class**, as opposed to more unit tests of the extractor:

1. **An invariant the empty case cannot satisfy.** The four waveform shares must sum to exactly 1. No
   table of zeros can pass that. The old test asserted only `square + sinusoid <= 1`, which zeros pass.
2. **Cross-module agreement at a shared checkpoint.** Two modules computing the same quantity at step
   25,000 must agree. That is what exposed 0.0 against 0.4023.
3. **Suspicion of exact zeros across independent seeds.** The trigger here was a convergence table
   showing *exactly* 0.0000 % change on all 20 runs. Independent seeds do not agree exactly.
4. **Cross-checking the prose against the artifacts.** `tests/check_results_numbers.py` re-derives every
   number quoted in `RESULTS.md` from `results/`. The document gets the same treatment as the code.

Corollary learned the hard way: **check the checker on a case whose answer you already know.** The first
version of the figure-provenance audit reported 11 spurious failures because it resolved recorded paths
against the wrong base directory. A verification script that reports a failure it cannot substantiate is
worse than no script.
