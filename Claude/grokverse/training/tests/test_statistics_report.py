"""Checks for analysis/statistics_report.py (STATISTICAL_ANALYSIS_PLAN §2–§6, §9).

Run: ``python tests/test_statistics_report.py`` from ``training/``.

Synthetic aggregate tables only.

THE TWO THINGS THE PLAN MAKES EASY TO GET WRONG
------------------------------------------------
1. **Conflating grid uncertainty with seed spread.** §6 asks for the interval-consistent bounds that
   follow from test accuracy being evaluated only every 25 steps. That is a *different* question from
   whether the seeds agree on a direction, and an earlier version of this module reported per-seed
   unanimity under the name "survives interval bounds". On the real data those answers differ — every
   seed's interval excludes zero, yet one seed reverses the direction — so the two are asserted
   separately here.
2. **Letting a p-value carry a claim.** §4 makes p-values descriptive companions. The Holm adjustment
   is tested as arithmetic and as a *labelled aid*, never as a decision rule.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grokverse.analysis import statistics_report as SR  # noqa: E402
from grokverse.analysis.figures_study import COMPARISONS  # noqa: E402


def check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        raise SystemExit(1)


def _row(arch, seed, gen, **over):
    row = {"run_id": f"{arch}_s{seed}", "arch": arch, "seed": seed, "p": 113,
           "tag": "step025000", "step": 25000, "steps_completed": 25000,
           "generalization_step": gen, "memorization_step": 150, "eval_every_test": 25,
           "structured_fraction_of_live": 0.4 if arch == "transformer" else 0.3,
           "phase_R": 0.99, "sparse_sinusoid__r2_test": 0.8, "odd_harmonics__r2_test": 0.7,
           "u_a__fraction_best_aic_square": 0.4, "u_a__fraction_best_aic_odd_harmonics": 0.2,
           "h3b__u_a__family_median": 0.6, "h3b__u_a__top1_median": 0.4,
           "remove_structured__drop": 0.9, "remove_structured__control_mean_drop": 0.1}
    row.update(over)
    return row


def _agg(tmp: Path, rows) -> Path:
    d = tmp / "aggregate"
    d.mkdir(parents=True, exist_ok=True)
    for module in ("mlp_mechanism", "transformer_mechanism", "wave_fitting", "h3_validity",
                   "causal_ablation", "logit_formula_fit"):
        keep = [r for r in rows
                if module != "mlp_mechanism" or r["arch"] == "mlp"]
        keep = [r for r in keep
                if module != "transformer_mechanism" or r["arch"] == "transformer"]
        (d / f"{module}.json").write_text(json.dumps({"module": module, "rows": keep}),
                                          encoding="utf-8")
    return d


# --------------------------------------------------------------------------- #
def check_holm():
    print("\n-- Holm adjustment (a descriptive aid, never a decision rule) --")
    out = SR.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    check("the smallest p is multiplied by n", abs(out["a"] - 0.03) < 1e-12)
    check("the adjustment is monotone", out["a"] <= out["c"] <= out["b"])
    check("values are capped at 1", all(v <= 1.0 for v in SR.holm({"a": 0.6, "b": 0.9}).values()))
    check("a None p-value stays None", SR.holm({"a": 0.01, "b": None})["b"] is None)
    check("an empty input is handled", SR.holm({}) == {})


def check_timing_bounds():
    print("\n-- timing: the evaluation grid vs the seed spread (plan 6) --")
    rows = ([_row("transformer", s, 5000 + 100 * s) for s in range(4)]
            + [_row("mlp", s, 9000) for s in range(4)])
    t = SR.timing_bounds(rows)
    check("one entry per paired seed", t["n_pairs"] == 4 and len(t["per_seed"]) == 4)
    first = t["per_seed"][0]
    check("the bound is [(a - e) - b, a - (b - e)] exactly",
          first["interval_consistent_low"] == (5000 - 25) - 9000
          and first["interval_consistent_high"] == 5000 - (9000 - 25))
    check("with a large, consistent gap the direction is unanimous",
          t["direction_unanimous_on_point_estimates"] is True
          and t["seeds_reversing_the_median_direction"] == [])
    check("...and every seed's interval excludes zero",
          t["n_seed_intervals_excluding_zero"] == 4)
    check("the phrase 'exact transition' is forbidden in the output",
          "not used" in t["wording_rule"])

    # one seed reversing the direction: the grid still excludes zero everywhere, the SPREAD does not
    rows2 = ([_row("transformer", s, 5000 if s < 3 else 12000) for s in range(4)]
             + [_row("mlp", s, 9000) for s in range(4)])
    t2 = SR.timing_bounds(rows2)
    check("a reversing seed makes the direction non-unanimous",
          t2["direction_unanimous_on_point_estimates"] is False
          and t2["seeds_reversing_the_median_direction"] == [3])
    check("...while every per-seed interval still excludes zero (the grid is not the limit)",
          t2["n_seed_intervals_excluding_zero"] == 4)
    check("the reading says the spread is what limits the claim, not the grid",
          "seed spread" in t2["reading"] and "NOT unanimous" in t2["reading"])
    check("...and it refuses a universal claim",
          "under the conditions examined" in t2["reading"])

    # a difference SMALLER than the grid must show up as an interval spanning zero
    rows3 = ([_row("transformer", s, 9010) for s in range(3)]
             + [_row("mlp", s, 9000) for s in range(3)])
    t3 = SR.timing_bounds(rows3)
    check("a difference smaller than the evaluation interval gives seed intervals spanning zero",
          t3["n_seed_intervals_excluding_zero"] == 0)
    check("no pairs at all is reported, not crashed on",
          SR.timing_bounds([_row("mlp", 0, 9000)])["n_pairs"] == 0)


def check_compute(tmp: Path):
    print("\n-- compute(): the eight pre-specified comparisons --")
    rows = ([_row("transformer", s, 6000 + 50 * s) for s in range(10)]
            + [_row("mlp", s, 9000 + 50 * s) for s in range(10)])
    agg = _agg(tmp, rows)
    rep = SR.compute(agg)

    expected = sum(len(c["points"]) for c in COMPARISONS)
    check(f"exactly the pre-specified comparisons are computed ({expected} rows from 8 specs)",
          len(rep["comparisons"]) == expected)
    check("the count matches figures_study's specification, so figures and numbers agree",
          {k.rsplit("__", 1)[0] for k in rep["comparisons"]} == {c["id"] for c in COMPARISONS})
    check("the rule that none is added after the fact is stated in the output",
          "none is added after" in rep["rule"])
    check("p-values are labelled as companions, not evidence",
          "never" in rep["p_value_status"].lower() or "descriptive" in rep["p_value_status"])

    one = rep["comparisons"]["p1_generalization_step__final"]
    check("every per-seed difference is listed, not only summarized (plan 2)",
          len(one["per_seed_differences"]) == 10)
    check("both raw value lists are kept as well",
          len(one["values_transformer"]) == 10 and len(one["values_mlp"]) == 10)
    check("the interval, both tests, both effect sizes and the robust spread are all present",
          {"bootstrap_ci95", "sign_test_p", "wilcoxon_signed_rank_p", "cohens_dz",
           "cliffs_delta", "robust_spread_of_differences"} <= set(one))
    check("a constant negative shift is detected as unanimous",
          one["all_negative"] is True and one["ci95_spans_zero"] is False)
    check("the reading states in words whether the interval spans zero (plan 9)",
          "excludes zero" in one["reading"].lower()
          and "under the conditions examined" in one["reading"].lower())

    check("the Holm block is present and labelled a descriptive aid",
          "descriptive aid" in rep["holm_adjusted_wilcoxon_p"]["status"])
    check("the small-sample discipline is written into the artifact",
          any("fundamentally faster" in s for s in rep["small_sample_discipline"]))
    check("the timing block is included", rep["timing_interval_consistency"]["n_pairs"] == 10)

    # an unpaired seed must be visible, never silently dropped (plan 5)
    rows2 = ([_row("transformer", s, 6000) for s in range(10)]
             + [_row("mlp", s, 9000) for s in range(9)])
    rep2 = SR.compute(_agg(tmp / "b", rows2))
    c2 = rep2["comparisons"]["p1_generalization_step__final"]
    check("an unpaired seed reduces n_pairs and is named",
          c2["n_pairs"] == 9 and c2["seeds_only_transformer"] == [9])


def main():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        check_holm()
        check_timing_bounds()
        check_compute(tmp)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
