"""The pre-registered evidence gate and the decision tree it feeds (INTERFACES §13, PREREG §5–6).

WHY THIS MODULE EXISTS
----------------------
Master prompt §12 requires that what happens for **each possible outcome** is fixed *before* the
experiments, so that the analysis cannot silently assume H3 must hold. `docs/PREREGISTRATION.md` §5
fixes the four gate criteria and the ≥ 8-of-10-seeds rule; §6 fixes the branch. This module does
exactly that arithmetic and nothing else.

**It never changes a threshold.** Every constant below is copied from PREREGISTRATION §5, carries its
`[AI-PROPOSED]` provenance, and is echoed into the output so a reader sees which numbers produced the
verdict. If a threshold must change, that is a human decision in `docs/HUMAN_DECISIONS.md` (C3–C6),
and it forces a re-run and a labbook entry.

THE THREE-VALUED LOGIC, AND WHY IT MATTERS
-------------------------------------------
Every criterion returns `True`, `False`, or **`None` = not evaluable**. `None` is never coerced to
`False`. A criterion whose input is missing (an analysis that did not run) or whose control is
degenerate (see below) has *not* failed — it has not been tested, and a seed that cannot be evaluated
is excluded from the denominator with its reason recorded. Counting "not evaluable" as "failed" would
manufacture evidence against the hypothesis exactly as surely as the reverse would manufacture
evidence for it.

A DEGENERATE CONTROL FOUND IN THE REAL OUTPUTS (2026-09-03)
------------------------------------------------------------
G3's second condition asks that a formula's `r2_test` exceed the **random-frequency control's** 95th
percentile. In the study's own `logit_formula_fit` outputs the control has `set_size = 56` — every
frequency available at `p = 113` — because the odd-harmonic family of a 13-frequency key set aliases
onto the whole spectrum. A "random set of the same size" drawn from 56 available frequencies *is* the
same set, so the control's standard deviation is ~1e-16 and it cannot discriminate anything. This
module detects that (`control_set_size >= (p-1)//2`) and reports the condition as **not evaluable**
rather than as passed, which is what a zero-variance control honestly supports.

STATUS: applies pre-registered rules to already-measured numbers. It computes no metric of its own.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..utils import git_commit, runs_dir, utcnow

MODULE = "decision_tree"
MODULE_VERSION = "1.0"

# --- PREREGISTRATION §5, verbatim. [AI-PROPOSED]; HUMAN_DECISIONS C2–C6. Never tuned here. ------
G1_MIN_STRUCTURED_FRACTION = 0.25       # of live neurons
G1_MIN_MEDIAN_FAMILY_FRACTION = 0.50    # of u_a and of u_b
G2_MIN_R = 0.50                         # and R must exceed the permutation null's 95th percentile
G3_MIN_ARGMAX_ACC_TEST = 0.90
G3_DIFFERENCE_CONTROL_RATIO = 0.5       # the (a-b) control must stay below half the formula's r2
G3_FORMULAS = ("sparse_sinusoid", "odd_harmonics")
SEEDS_REQUIRED, SEEDS_TOTAL = 8, 10     # "≥ 8 of 10 seeds pass all four criteria"
#: Sensitivity re-runs of the gate under the other two structured-neuron definitions (§5, last line).
SENSITIVITY_DEFINITIONS = ("sensitivity_family_0.30", "sensitivity_family_0.70")
CRITERIA = ("G1", "G2", "G3", "G4")


def _num(x):
    """A usable number, or None. Booleans are not numbers here."""
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v and abs(v) != float("inf") else None


# --------------------------------------------------------------------------- #
# the four criteria                                                            #
# --------------------------------------------------------------------------- #
def g1(row: dict, definition: str | None = None) -> tuple[bool | None, dict]:
    """Periodic structure: structured fraction ≥ 0.25 of live **and** median family fraction ≥ 0.50."""
    key = ("structured_fraction_of_live" if definition is None
           else f"structured_fraction__{definition}")
    frac = _num(row.get(key))
    fam_a = _num(row.get("family_fraction_median_u_a"))
    fam_b = _num(row.get("family_fraction_median_u_b"))
    detail = {"structured_fraction_of_live": frac, "family_fraction_median_u_a": fam_a,
              "family_fraction_median_u_b": fam_b, "definition": definition or "primary",
              "thresholds": {"structured_fraction": G1_MIN_STRUCTURED_FRACTION,
                             "median_family_fraction": G1_MIN_MEDIAN_FAMILY_FRACTION}}
    if frac is None or fam_a is None or fam_b is None:
        detail["not_evaluable"] = "a mechanism number is missing"
        return None, detail
    return bool(frac >= G1_MIN_STRUCTURED_FRACTION
                and fam_a >= G1_MIN_MEDIAN_FAMILY_FRACTION
                and fam_b >= G1_MIN_MEDIAN_FAMILY_FRACTION), detail


def g2(row: dict) -> tuple[bool | None, dict]:
    """Phase addition: R above the permutation null's 95th percentile **and** R ≥ 0.5."""
    R = _num(row.get("phase_R"))
    q95 = _num(row.get("phase_null_q95"))
    detail = {"resultant_length": R, "null_q95": q95, "threshold_R": G2_MIN_R,
              "insufficient_neurons": row.get("phase_insufficient_neurons"),
              "n_neurons_in_test": row.get("phase_n")}
    if R is None:
        detail["not_evaluable"] = ("no resultant length — the structured set was empty at this "
                                   "checkpoint" if row.get("phase_insufficient_neurons")
                                   else "phase relation missing")
        return None, detail
    if q95 is None:
        detail["not_evaluable"] = "no permutation null to compare against"
        return None, detail
    return bool(R > q95 and R >= G2_MIN_R), detail


def g3(row: dict, p: int | None = None) -> tuple[bool | None, dict]:
    """End-to-end logit fit, with the degenerate-control guard described in the module docstring."""
    detail = {"formulas": {}, "thresholds": {"argmax_acc_test": G3_MIN_ARGMAX_ACC_TEST,
                                             "difference_control_ratio": G3_DIFFERENCE_CONTROL_RATIO}}
    control_q95 = _num(row.get("control_random_m__r2_test_q95"))
    control_std = _num(row.get("control_random_m__r2_test_std"))
    set_size = _num(row.get("control_set_size"))
    half = ((p - 1) // 2) if p else None
    degenerate = bool(set_size is not None and half is not None and set_size >= half)
    detail.update({"control_q95_r2_test": control_q95, "control_std_r2_test": control_std,
                   "control_set_size": set_size, "n_frequencies_available": half,
                   "control_degenerate": degenerate})
    diff_r2 = _num(row.get("control_difference__r2_test"))
    best = None
    for name in G3_FORMULAS:
        acc = _num(row.get(f"{name}__argmax_acc_test"))
        r2 = _num(row.get(f"{name}__r2_test"))
        detail["formulas"][name] = {"argmax_acc_test": acc, "r2_test": r2}
        if acc is None or r2 is None:
            continue
        ok_acc = acc >= G3_MIN_ARGMAX_ACC_TEST
        ok_diff = (diff_r2 is not None and diff_r2 < G3_DIFFERENCE_CONTROL_RATIO * r2)
        detail["formulas"][name].update({"passes_argmax": ok_acc, "passes_difference_control": ok_diff})
        if ok_acc and ok_diff:
            best = name
    detail["difference_control_r2_test"] = diff_r2
    detail["formula_satisfying_argmax_and_difference"] = best
    if all(detail["formulas"][n].get("argmax_acc_test") is None for n in G3_FORMULAS):
        detail["not_evaluable"] = "no logit_formula_fit numbers for this run and point"
        return None, detail
    if degenerate:
        detail["not_evaluable"] = (
            f"the random-frequency control draws {int(set_size)} of {half} available frequencies, so "
            "it is the same set every time (std ~ 0) and cannot discriminate; the r2-above-control "
            "condition is untestable as specified")
        return None, detail
    if control_q95 is None:
        detail["not_evaluable"] = "no random-frequency control to compare r2 against"
        return None, detail
    if best is None:
        return False, detail
    return bool(_num(row.get(f"{best}__r2_test")) > control_q95), detail


def g4(row: dict) -> tuple[bool | None, dict]:
    """Causal: read straight from `causal_ablation`'s own pre-registered G4 verdict."""
    detail = {"remove_structured_necessary": row.get("g4_remove_structured_necessary"),
              "remove_key_freqs_necessary": row.get("g4_remove_key_freqs_necessary"),
              "keep_structured_sufficient": row.get("g4_keep_structured_sufficient"),
              "ids_used": row.get("g4_ids_used")}
    verdict = row.get("g4_passed")
    if verdict is None:
        detail["not_evaluable"] = "causal_ablation did not report a G4 verdict for this run and point"
        return None, detail
    return bool(verdict), detail


# --------------------------------------------------------------------------- #
# the gate                                                                     #
# --------------------------------------------------------------------------- #
def _load(agg_dir: Path, module: str) -> dict:
    path = agg_dir / f"{module}.json"
    if not path.exists():
        return {"rows": [], "missing": [], "module": module, "absent": True}
    return json.loads(path.read_text(encoding="utf-8"))


def _is_point(row: dict, point: str) -> bool:
    """Does this row belong to the requested measurement point?

    The file tag encodes a *step*, not a point, so the point is resolved against the run's own
    metadata: `final` is the last completed step, `crossing` is the logged generalization crossing.
    Matching on the tag prefix alone would let BOTH checkpoints of a run through and count one seed
    twice in the gate arithmetic — which is exactly what it did before this was fixed.
    """
    tag = str(row.get("tag", ""))
    step = row.get("step")
    if tag == "final" or step in (None, "final"):
        return point == "final"
    try:
        step = int(step)
    except (TypeError, ValueError):
        return False
    if point == "final":
        last = row.get("steps_completed") or row.get("steps")
        return last is not None and step == int(last)
    if point == "crossing":
        gen = row.get("generalization_step")
        return gen is not None and step == int(gen)
    raise ValueError(f"unknown measurement point {point!r}; expected 'final' or 'crossing'")


def build_rows(agg_dir: Path, point: str = "final") -> dict:
    """Join the mechanism, logit-fit and ablation rows on (run_id, tag) for one measurement point."""
    merged: dict[tuple, dict] = {}
    for module in ("mlp_mechanism", "transformer_mechanism", "logit_formula_fit",
                   "causal_ablation", "h3_validity"):
        for row in _load(agg_dir, module).get("rows", []):
            if not _is_point(row, point):
                continue
            key = (row.get("run_id"), str(row.get("tag", "")))
            merged.setdefault(key, {}).update({k: v for k, v in row.items() if v is not None})
            merged[key].setdefault("_modules", []).append(module)
    return merged


def evaluate(agg_dir: Path | str | None = None, point: str = "final",
             definition: str | None = None) -> dict:
    """Apply the pre-registered gate per architecture and name the decision-tree branch."""
    agg_dir = Path(agg_dir) if agg_dir else (runs_dir().parent / "results" / "aggregate")
    merged = build_rows(agg_dir, point)
    per_arch: dict[str, dict] = {}
    for (run_id, tag), row in sorted(merged.items()):
        arch = row.get("arch") or "unknown"
        seed = row.get("seed")
        p = row.get("p")
        results = {"G1": g1(row, definition), "G2": g2(row), "G3": g3(row, p), "G4": g4(row)}
        verdicts = {c: results[c][0] for c in CRITERIA}
        evaluable = all(v is not None for v in verdicts.values())
        entry = {"run_id": run_id, "seed": seed, "tag": tag,
                 "verdicts": verdicts,
                 "all_four_passed": (all(verdicts.values()) if evaluable else None),
                 "evaluable": evaluable,
                 "not_evaluable_because": {c: results[c][1].get("not_evaluable")
                                           for c in CRITERIA
                                           if results[c][1].get("not_evaluable")},
                 "detail": {c: results[c][1] for c in CRITERIA},
                 "modules_present": sorted(set(row.get("_modules", [])))}
        per_arch.setdefault(arch, {"seeds": []})["seeds"].append(entry)

    for arch, block in per_arch.items():
        seeds = block["seeds"]
        passed = [s for s in seeds if s["all_four_passed"] is True]
        failed = [s for s in seeds if s["all_four_passed"] is False]
        unknown = [s for s in seeds if s["all_four_passed"] is None]
        block.update({
            "n_seeds_present": len(seeds), "n_evaluable": len(passed) + len(failed),
            "n_passed": len(passed), "n_failed": len(failed), "n_not_evaluable": len(unknown),
            "seeds_passed": sorted(s["seed"] for s in passed if s["seed"] is not None),
            "seeds_failed": sorted(s["seed"] for s in failed if s["seed"] is not None),
            "seeds_not_evaluable": sorted(s["seed"] for s in unknown if s["seed"] is not None),
            "rule": f">= {SEEDS_REQUIRED} of {SEEDS_TOTAL} seeds pass all four criteria",
            "per_criterion_passed": {c: sum(1 for s in seeds if s["verdicts"][c] is True)
                                     for c in CRITERIA},
            "per_criterion_not_evaluable": {c: sum(1 for s in seeds if s["verdicts"][c] is None)
                                            for c in CRITERIA},
        })
        # A verdict needs the whole seed set. With fewer seeds analysed than the design has, the
        # gate is UNDETERMINED — "not finished" is not "failed", and reporting False here would be
        # a claim about the architecture when the only fact is that the analysis is incomplete.
        n_seeds = len({s["seed"] for s in seeds if s["seed"] is not None})
        block["n_distinct_seeds"] = n_seeds
        if n_seeds < SEEDS_TOTAL:
            block["passes_gate"] = None
            block["verdict_reason"] = (
                f"only {n_seeds} of {SEEDS_TOTAL} seeds have been analysed at this point — the gate "
                "is undetermined, not failed")
        elif unknown and len(passed) < SEEDS_REQUIRED:
            # All seeds present, but some criteria untestable on some of them. Only if even the
            # untested seeds could not carry the rule is the verdict False.
            block["passes_gate"] = (False if len(passed) + len(unknown) < SEEDS_REQUIRED else None)
            block["verdict_reason"] = (
                f"{len(passed)} passed, {len(unknown)} not evaluable; "
                + ("even if every untested seed passed, the rule could not be met"
                   if block["passes_gate"] is False
                   else f"the untested seeds could still reach {SEEDS_REQUIRED}"))
        else:
            block["passes_gate"] = bool(len(passed) >= SEEDS_REQUIRED)
            block["verdict_reason"] = f"{len(passed)} of {SEEDS_TOTAL} seeds passed all four criteria"

    branch = decide_branch(per_arch)
    return {
        "module": MODULE, "module_version": MODULE_VERSION, "created_utc": utcnow(),
        "analysis_git_commit": git_commit(), "aggregate_dir": str(agg_dir),
        "measurement_point": point, "structured_definition": definition or "primary",
        "thresholds": {
            "G1_min_structured_fraction": G1_MIN_STRUCTURED_FRACTION,
            "G1_min_median_family_fraction": G1_MIN_MEDIAN_FAMILY_FRACTION,
            "G2_min_R": G2_MIN_R, "G3_min_argmax_acc_test": G3_MIN_ARGMAX_ACC_TEST,
            "G3_difference_control_ratio": G3_DIFFERENCE_CONTROL_RATIO,
            "seeds_required": SEEDS_REQUIRED, "seeds_total": SEEDS_TOTAL,
            "provenance": "docs/PREREGISTRATION.md §5 — [AI-PROPOSED], HUMAN_DECISIONS C2–C6; "
                          "this module never changes them"},
        "per_architecture": per_arch,
        "branch": branch,
    }


def decide_branch(per_arch: dict) -> dict:
    """PREREGISTRATION §6 / master prompt §12, applied literally to the gate verdicts."""
    passing = sorted(a for a, b in per_arch.items() if b.get("passes_gate") is True)
    failing = sorted(a for a, b in per_arch.items() if b.get("passes_gate") is False)
    unknown = sorted(a for a, b in per_arch.items() if b.get("passes_gate") is None)
    if unknown:
        return {"branch": "undetermined",
                "reason": f"the gate is not yet decided for {unknown} — analyses are incomplete",
                "architectures_passing": passing, "architectures_failing": failing,
                "architectures_undetermined": unknown,
                "action": "complete the missing analyses; no branch may be taken meanwhile"}
    if len(passing) == len(per_arch) and per_arch:
        return {"branch": "both_pass",
                "architectures_passing": passing, "architectures_failing": [],
                "action": "investigate H3 and compare how the Fourier principle is represented",
                "permitted_conclusion": (
                    "The results are consistent with the same Fourier principle being expressed "
                    "through different internal representations."),
                "banned": "the two architectures use the same circuit in different bases"}
    if len(passing) == 1:
        return {"branch": "one_passes",
                "architectures_passing": passing, "architectures_failing": failing,
                "action": (f"bounded alternative-mechanism analysis of {failing} "
                           "(analysis/bounded_alternative.py, PREREGISTRATION §6.3); H3 may be "
                           f"interpreted only for {passing}")}
    return {"branch": "neither_passes",
            "architectures_passing": [], "architectures_failing": failing,
            "action": ("report that the Fourier evidence tested does not identify the learned "
                       "mechanisms, and investigate the validity of the existing metrics; run the "
                       "bounded alternative-mechanism analysis for both architectures")}


def summary_lines(report: dict) -> list[str]:
    lines = [f"evidence gate at point={report['measurement_point']} "
             f"(definition: {report['structured_definition']})"]
    for arch, block in sorted(report["per_architecture"].items()):
        lines.append(f"  {arch:12s} passes_gate={block['passes_gate']}  "
                     f"({block['n_passed']} passed, {block['n_failed']} failed, "
                     f"{block['n_not_evaluable']} not evaluable, of {block['n_seeds_present']} rows)")
        lines.append(f"      per criterion passed: {block['per_criterion_passed']}")
        ne = {c: n for c, n in block["per_criterion_not_evaluable"].items() if n}
        if ne:
            lines.append(f"      NOT EVALUABLE:        {ne}")
    b = report["branch"]
    lines.append(f"  BRANCH: {b['branch']} — {b.get('action', '')}")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description="Pre-registered evidence gate + decision tree (§13)")
    ap.add_argument("--aggregate-dir", type=Path, default=None)
    ap.add_argument("--point", default="final")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--sensitivity", action="store_true",
                    help="also evaluate under the 0.30 and 0.70 structured-neuron definitions")
    args = ap.parse_args()
    report = evaluate(args.aggregate_dir, args.point)
    if args.sensitivity:
        report["sensitivity"] = {
            d: {a: {"passes_gate": b["passes_gate"], "n_passed": b["n_passed"],
                    "n_not_evaluable": b["n_not_evaluable"]}
                for a, b in evaluate(args.aggregate_dir, args.point, d)["per_architecture"].items()}
            for d in SENSITIVITY_DEFINITIONS}
    out = args.out or (runs_dir().parent / "results" / "decision_tree.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("\n".join(summary_lines(report)))
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
