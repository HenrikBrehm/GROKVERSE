# Nondeterministic control draws from a Python set

Found 2026-09-06 while adding the graded structured-neuron ablation. Not introduced by that work — the defect had been in the frozen analysis code since `ac5bec4`, i.e. from before the analysis freeze at `0b55e1d`.

## Problem

`causal_ablation.py` produced **different random-control statistics on every run**, from identical code, identical inputs and the same seed. The project constitution names determinism as a non-negotiable ("same config+seed => same curve"), so this was a direct violation that nothing in the pipeline noticed.

## Symptoms

Re-running `causal_ablation` on the 20 primary runs and diffing against the artifacts already on disk:

- every **observed** ablation value was identical (or differed only in the last floating-point bit);
- but 560 of 660 pre-existing ablation blocks differed, all in the **control** statistics;
- `remove_structured` control mean drop moved by up to **0.0034**, `remove_structured_neurons` by **0.0015**;
- the key-frequency ablations were untouched.

The giveaway was the `n_structured` dictionary, whose **key order differed** between the old and the new file:

```
old: {'sensitivity_family_0.70': 434, 'primary_family_0.50': 445, 'sensitivity_family_0.30': 455}
new: {'sensitivity_family_0.70': 434, 'sensitivity_family_0.30': 455, 'primary_family_0.50': 445}
```

A dict only reorders like that if it was built from an unordered source.

## Cause

`resolve_structured_masks` built the available definitions as a **Python set**:

```python
have = {d for d in wanted if f"mask__{d}" in z.files}
return {"sets": {d: ... for d in have}, ...}
```

Python randomises `str` hashing per process, so the iteration order of a set of strings changes from process to process. Both `mlp_ablations` and `transformer_ablations` then iterate `masks["sets"].items()` while drawing from **one shared `rng`**. Which 50 control draws went to which structured definition therefore depended on the process's hash seed.

Confirmed directly: the same three-element set printed in six fresh interpreters gave four different orders.

## Solution

`wanted` is already an ordered tuple, so iterate that instead:

```python
have = tuple(d for d in wanted if f"mask__{d}" in z.files)
```

One line. `_recompute_structured_masks` already iterated `wanted` and was correct. `MODULE_VERSION` 1.1 → 1.2 and every affected checkpoint re-run, as `PREREGISTRATION.md` §9 requires for a logged bug fix.

## Why it mattered

The two moved numbers are quoted in `RESULTS.md` §6 and asserted by `tests/check_results_numbers.py` at a tolerance of ±0.001, which the drift exceeded. Nothing scientific changes — the controls stay at ≈0.87 and ≈0.91, and G4 still reads 0/10 in both architectures — but the claim that every number is regenerable was false until this was fixed. Escalated to the human authors as `HUMAN_DECISIONS.md` **D8**.

## How to prevent it in the future

- Never let a `set` (or anything hash-ordered) determine the order in which a shared random generator is consumed. Sort it, or iterate the ordered source it was filtered from.
- A control that is not reproducible is not a control. Treat "re-run the module and diff the artifact" as a routine check, not an afterthought — this defect was invisible for three days precisely because nothing ever re-ran the module in a fresh process and compared.
- The same shape as [[Silent extractors that return a plausible number]]: well-formed output, correct shape, plausible value, quietly wrong.

## Related

[[Silent extractors that return a plausible number]]
