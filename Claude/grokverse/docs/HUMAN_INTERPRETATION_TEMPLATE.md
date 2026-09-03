# Human interpretation — template

```
STATUS: EMPTY — NO HUMAN TEXT HAS BEEN WRITTEN IN THIS FILE
```

This file exists so the study's conclusions can be written **by the human authors in their own words**,
after the analyses are frozen. The AI fills in nothing here except this structure. Every field marked
`[HUMAN AUTHORS MUST COMPLETE]` stays as it is until a human replaces it.

**Why this is separate from `RESULTS.md`.** `RESULTS.md` reports measurements. This file is where the
authors say what they think the measurements *mean*. BWKI requires a clearly identifiable portion of own
work, and the master prompt (§3 rule 11, §23 item 19) requires that the final scientific interpretation be
human-written and not presented as such until it is. An AI-drafted interpretation, if one is ever produced
as raw material, must stay labelled AI-drafted until a human has independently rewritten it.

**Before writing, read:** `docs/PREREGISTRATION.md` (what was promised), `docs/LIMITATIONS.md` (what cannot
be concluded), the decision-tree branch that was actually taken, and the per-seed tables — not only the
summary numbers.

---

## 0. What the analyses returned

Filled in from `results/decision_tree.json` and the aggregate tables. Copy the numbers; do not restate
them from memory.

| item | value |
|---|---|
| Gate result, transformer | `[HUMAN AUTHORS MUST COMPLETE]` — passed / failed, in how many of 10 seeds, which criterion failed where |
| Gate result, MLP | `[HUMAN AUTHORS MUST COMPLETE]` |
| Decision-tree branch taken | `[HUMAN AUTHORS MUST COMPLETE]` — both pass / one passes / neither passes |
| Analysis freeze commit | `[HUMAN AUTHORS MUST COMPLETE]` |
| Was the pre-registration human-approved before the analyses? | `[HUMAN AUTHORS MUST COMPLETE]` — yes/no, and if no, say so in the write-up |

---

## 1. Per-hypothesis interpretation

Repeat this block for **H1, H2, H3, H4, H5** and for the main claim. Do not skip a hypothesis because it
was refuted — a refuted hypothesis gets the same six steps.

### H_ : `[title]`

**1. Observation.** What was seen, in one or two sentences, without interpretation.
`[HUMAN AUTHORS MUST COMPLETE]`

**2. Quantitative evidence.** The specific numbers, with their intervals, the number of seeds, and where
they come from (file and function). Not "clearly higher" — the number.
`[HUMAN AUTHORS MUST COMPLETE]`

**3. Alternative explanation.** What else could produce this observation? Consider at least: stopping
time and convergence state, parameter count, input parametrization, the evaluation grid, the width of the
Fourier mask, measuring on `W_E` only, the fixed frequency count, the training fraction, Grokfast, and
seed noise. Say which of these the design rules out and which it does not.
`[HUMAN AUTHORS MUST COMPLETE]`

**4. Causal test.** What ablation or intervention was run, what the size-matched control did, and what
that licenses. If no causal test applies to this hypothesis, say so explicitly.
`[HUMAN AUTHORS MUST COMPLETE]`

**5. Limitation.** What this result cannot show. Point to the relevant entry in `docs/LIMITATIONS.md`.
`[HUMAN AUTHORS MUST COMPLETE]`

**6. Permissible conclusion.** One sentence, in graded language, that the evidence above actually
supports.
`[HUMAN AUTHORS MUST COMPLETE]`

---

## 2. The main claim

The main claim may be used **only** if the data and the evidence gate support it:

> "Transformer and MLP can solve the same modular task through related Fourier-based principles, but may
> express them in different neural representations."

**Does the evidence support it?** `[HUMAN AUTHORS MUST COMPLETE]` — yes / no / partly, and why.

**If not, what is the honest headline instead?** `[HUMAN AUTHORS MUST COMPLETE]`

If the results show the two mechanisms are essentially the same, say exactly that. If neither architecture
passes the Fourier tests, say exactly that. A well-founded negative result is the deliverable in those
branches, not a fallback.

---

## 3. What was retracted

The pre-registration commits in advance to retracting the current claim that the transformer "learns a
sparser Fourier circuit" if H3 is supported. Record here what was actually retracted and what replaced it.

| old claim | where it appeared | retracted? | replaced by |
|---|---|---|---|
| "The Transformer learns a sparser Fourier circuit" | README, RESULTS §3 | `[HUMAN]` | `[HUMAN]` |
| "The MLP uses a more distributed solution" | RESULTS §3 | `[HUMAN]` | `[HUMAN]` |
| "Restricted and excluded loss reproduce Nanda's method" | RESULTS §2 | `[HUMAN]` | `[HUMAN]` |
| "50/50 attention … directly links the learned structure to the computation" | RESULTS §2 | `[HUMAN]` | `[HUMAN]` |

---

## 4. Positioning against the literature

For each, one sentence saying what this study adds **beyond** it, or that it adds nothing:

| source | what it already establishes | what this study adds |
|---|---|---|
| Nanda et al. 2023 (2301.05217) | the Fourier/trig circuit and the progress measures | `[HUMAN]` |
| Manir & Rupa 2026 (2603.25009) | the concentration gap and the protocol-dependence of the timing gap | `[HUMAN]` |
| Swaroop 2026 (2603.23784) | square-wave input weights and the phase-sum relation in a two-hot ReLU MLP | `[HUMAN]` |
| Doshi et al. 2023 (2310.13061) | IPR-ranked neurons and pruning | `[HUMAN]` |
| Khanh 2026 (2607.06639) | metrics read at the transition overstate the converged value | `[HUMAN]` |
| McCracken et al. 2025 (2505.18266) | concatenation MLPs; embeddings reduce the number of learned frequencies | `[HUMAN]` |

---

## 5. Banned-phrase check

Search the finished write-up for each of these and confirm none survives without the evidence to back it:

- [ ] "proves"
- [ ] "clearly shows"
- [ ] "Transformers fundamentally learn …"
- [ ] "MLPs are worse …"
- [ ] "the same circuit"
- [ ] "the same function" (only "both generalize on the same task" is permitted until §6.2 is measured)
- [ ] "exact transition"
- [ ] any claim of novelty for the concentration gap or the timing gap

Graded language to use instead: "We observe …", "This is consistent with …", "The ablation supports the
interpretation …", "The data are not sufficient to rule out …", "Under the conditions examined …".

---

## 6. Sign-off

| | |
|---|---|
| Author | `[HUMAN AUTHORS MUST COMPLETE]` |
| Date | |
| I have read the per-seed tables, not only the summaries | yes / no |
| I have checked the derivations in `docs/*_MECHANISM_DERIVATION.md` myself | yes / no |
| I have read the primary sources cited above in the original | yes / no |
| This interpretation is written in my own words, not edited from an AI draft | yes / no |
| Parts I did not write myself, and who did | |
