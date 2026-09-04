"""Checks for analysis/h4_report.py (PREREGISTRATION §3 H4).

Run: ``python tests/test_h4_report.py`` from ``training/``.

Synthetic aggregate rows only.

WHAT IS PINNED HERE
--------------------
H4 is an ordering claim over three step counts — memorization, structure onset, generalization — so
the checks build rows whose ordering is known by construction and assert the counts come out right in
both directions. The other property pinned is the **denominator**: a metric whose onset is undefined
on some seeds must be reported with that count rather than quietly dropped, because "10 of 10" and
"2 of 2" are very different claims and only the denominator distinguishes them.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grokverse.analysis import h4_report as H4  # noqa: E402


def check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        raise SystemExit(1)


def _row(arch, seed, onset, gen=9000, mem=150, change=0.3, metric="structured_fraction_of_live"):
    row = {"run_id": f"{arch}_s{seed}", "arch": arch, "seed": seed,
           "memorization_step": mem, "generalization_step": gen,
           "tag": "all_checkpoints", "step": None}
    for m in H4.METRICS:
        row[f"h4__{m}__onset_step"] = onset if m == metric else None
        row[f"h4__{m}__change"] = change if m == metric else None
        row[f"h4__{m}__init"] = 0.0
        row[f"h4__{m}__pre_gen"] = change
    return row


def _agg(tmp: Path, rows, name="a") -> Path:
    d = tmp / name / "aggregate"
    d.mkdir(parents=True, exist_ok=True)
    (d / "structure_over_time.json").write_text(
        json.dumps({"module": "structure_over_time", "rows": rows}), encoding="utf-8")
    return d


def check_ordering(tmp: Path):
    print("\n-- the ordering H4 asserts --")
    rows = [_row("transformer", s, onset=1000) for s in range(6)]
    rep = H4.compute(_agg(tmp, rows, "o1"))
    m = rep["per_architecture"]["transformer"]["metrics"]["structured_fraction_of_live"]
    check("an onset between memorization and generalization counts on both sides",
          m["n_onset_before_generalization"] == 6 and m["n_onset_after_memorization"] == 6)
    check("the onset median is reported", m["onset_step"]["median"] == 1000)
    check("the change is summarized across seeds",
          abs(m["change_init_to_pre_generalization"]["median"] - 0.3) < 1e-9)

    late = [_row("transformer", s, onset=9500) for s in range(6)]     # AFTER generalization
    m2 = H4.compute(_agg(tmp, late, "o2"))["per_architecture"]["transformer"]["metrics"][
        "structured_fraction_of_live"]
    check("CONTROL: an onset after the generalization crossing counts 0 before it",
          m2["n_onset_before_generalization"] == 0)
    early = [_row("transformer", s, onset=50) for s in range(6)]      # BEFORE memorization
    m3 = H4.compute(_agg(tmp, early, "o3"))["per_architecture"]["transformer"]["metrics"][
        "structured_fraction_of_live"]
    check("CONTROL: an onset before memorization counts 0 after it",
          m3["n_onset_after_memorization"] == 0)
    check("...though it is still before generalization",
          m3["n_onset_before_generalization"] == 6)


def check_denominator(tmp: Path):
    print("\n-- the denominator is never quietly reduced --")
    rows = ([_row("mlp", s, onset=1000) for s in range(4)]
            + [_row("mlp", s, onset=None) for s in range(4, 10)])
    rep = H4.compute(_agg(tmp, rows, "d1"))
    m = rep["per_architecture"]["mlp"]["metrics"]["structured_fraction_of_live"]
    check("seeds with no defined onset are counted, not dropped",
          m["n_seeds"] == 10 and m["n_onset_defined"] == 4)
    check("the 'before generalization' count is over the DEFINED onsets",
          m["n_onset_before_generalization"] == 4)
    check("the reading states how many seeds had no onset at all",
          "6 seed(s) have no defined onset" in m["reading"])
    check("every seed appears in the per-seed detail", len(m["per_seed"]) == 10)
    check("a seed with no onset has None, not a fabricated step",
          any(d["onset_step"] is None and d["onset_before_generalization"] is None
              for d in m["per_seed"]))


def check_caveats(tmp: Path):
    print("\n-- the caveats travel with the numbers --")
    rep = H4.compute(_agg(tmp, [_row("mlp", s, onset=1000) for s in range(3)], "c1"))
    check("the analysis is labelled correlational in the artifact itself",
          "CORRELATIONAL" in rep["reading"])
    check("...and points at causal_ablation for causal claims",
          "causal_ablation" in rep["reading"])
    check("the onset's two-checkpoint fragility is recorded",
          "two" in rep["onset_caveat"] and "indicator" in rep["onset_caveat"])
    check("the hypothesis is quoted so the artifact is self-describing",
          "memorization plateau" in rep["hypothesis"])
    check("both step counts are summarized for context",
          {"memorization_step", "generalization_step"} <= set(rep["per_architecture"]["mlp"]))
    check("the summary block gives the per-metric ratio as a string",
          "/" in rep["summary"]["mlp"]["onset_before_generalization_by_metric"][
              "structured_fraction_of_live"])
    raised = False
    try:
        H4.compute(tmp / "does_not_exist")
    except ValueError:
        raised = True
    check("a missing aggregate table raises, naming what to run first", raised)


def main():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        check_ordering(tmp)
        check_denominator(tmp)
        check_caveats(tmp)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
