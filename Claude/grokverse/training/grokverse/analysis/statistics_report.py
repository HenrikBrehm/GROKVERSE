"""The eight pre-specified paired comparisons, computed and stored (STATISTICAL_ANALYSIS_PLAN).

WHY THIS MODULE EXISTS
----------------------
`analysis/statistics.py` provides the primitives (paired bootstrap, exact sign and Wilcoxon tests,
effect sizes). This module applies them to **exactly** the eight comparisons `§3` of the plan fixes
in advance, over the aggregate tables, and writes one artifact. It adds no comparison of its own:
"No comparison is added to the primary list after seeing a result."

It reuses `figures_study.COMPARISONS` — the same specifications and the same extractors that draw the
figures — so a number in `results/statistics.json` and the number on the corresponding figure cannot
disagree.

WHAT THE PLAN REQUIRES THAT IS EASY TO GET WRONG
-------------------------------------------------
* **Every per-seed difference is listed**, never only summarized (§2).
* **P-values are companions, never evidence** (§4). Holm-adjusted values across the eight comparisons
  are reported *labelled as a descriptive aid*, because a reader will want them; a claim rests on the
  effect size, its interval and the per-seed pattern.
* **Timing is reported twice** (§6): the difference of the reported crossings, and the
  interval-consistent bounds that follow from test accuracy being evaluated only every 25 steps. A
  timing claim is made only if it survives the bounds, and the phrase "exact transition" is not used.
* **Runs are never silently dropped** (§5): `n started` and `n completed` come from the run manifest,
  and a broken pair is counted, not ignored.
* **Small-sample discipline** (§9): with `n = 10` an interval spanning zero is said to span zero, and
  a unanimous sign is reported as such because it is itself informative.

STATUS: applies pre-specified statistics to already-measured numbers. It measures nothing new.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..utils import git_commit, runs_dir, utcnow
from . import statistics as ST
from .figures_study import ARCH_A, ARCH_B, COMPARISONS, load_rows, values_by_arch

MODULE = "statistics_report"
MODULE_VERSION = "1.0"
N_BOOT = 10_000          # STATISTICAL_ANALYSIS_PLAN §2
SEED = 0
#: Test accuracy is evaluated every this many steps, so a crossing is known only to lie in
#: (first_crossing − EVAL_EVERY_TEST, first_crossing]. Taken from the run configs, not assumed.
EVAL_EVERY_TEST_DEFAULT = 25


def holm(pvalues: dict) -> dict:
    """Holm-adjusted p-values across the eight comparisons — a descriptive aid, never evidence."""
    items = [(k, v) for k, v in pvalues.items() if v is not None]
    if not items:
        return {}
    items.sort(key=lambda kv: kv[1])
    n = len(items)
    out, running = {}, 0.0
    for i, (key, p) in enumerate(items):
        running = max(running, min(1.0, (n - i) * p))
        out[key] = running
    for key, value in pvalues.items():
        out.setdefault(key, None)
    return out


def completeness(manifest_path: Path) -> dict:
    """`n started` and `n completed` per architecture, from the run manifest (§5)."""
    if not manifest_path.exists():
        return {"error": f"{manifest_path.name} not found"}
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = rows if isinstance(rows, list) else rows.get("runs", [])
    out: dict = {"n_started_total": 0, "per_architecture": {}, "not_completed": []}
    for r in rows:
        arch = r.get("arch") or (r.get("config") or {}).get("arch")
        study = r.get("study") or (r.get("config") or {}).get("study")
        frac = r.get("train_frac") or (r.get("config") or {}).get("train_frac")
        if study != "arch25k" or frac != 0.3 or (r.get("grokfast") or
                                                 (r.get("config") or {}).get("grokfast")):
            continue                                   # the primary block only
        block = out["per_architecture"].setdefault(arch, {"started": 0, "completed": 0})
        block["started"] += 1
        out["n_started_total"] += 1
        if r.get("status") == "completed":
            block["completed"] += 1
        else:
            out["not_completed"].append({"run_id": r.get("run_id"), "status": r.get("status"),
                                         "abort_reason": r.get("abort_reason")})
    return out


def timing_bounds(rows: list[dict]) -> dict:
    """The §6 interval-consistent bounds on the paired generalization-step difference.

    A crossing reported at step `c` is only known to lie in `(c − eval_every_test, c]`. The paired
    difference `a − b` is therefore compatible with anything in
    `[(a − e_a) − b, a − (b − e_b)]`. Reported next to the point estimate; a timing claim is made
    only if it survives these bounds, and no number here is called an exact transition.
    """
    per_arch: dict[str, dict[int, dict]] = {}
    for row in rows:
        seed, arch = row.get("seed"), row.get("arch")
        gen = row.get("generalization_step")
        if seed is None or arch is None or gen is None:
            continue
        per_arch.setdefault(arch, {})[int(seed)] = {
            "first_crossing": int(gen),
            "eval_every_test": int(row.get("eval_every_test") or EVAL_EVERY_TEST_DEFAULT)}
    a, b = per_arch.get(ARCH_A, {}), per_arch.get(ARCH_B, {})
    seeds = sorted(set(a) & set(b))
    if not seeds:
        return {"n_pairs": 0, "note": "no seed has both architectures"}
    per_seed = []
    for s in seeds:
        ca, cb = a[s]["first_crossing"], b[s]["first_crossing"]
        ea, eb = a[s]["eval_every_test"], b[s]["eval_every_test"]
        per_seed.append({"seed": s, f"{ARCH_A}_first_crossing": ca, f"{ARCH_B}_first_crossing": cb,
                         "point_difference": ca - cb,
                         "interval_consistent_low": (ca - ea) - cb,
                         "interval_consistent_high": ca - (cb - eb)})
    point = np.array([d["point_difference"] for d in per_seed], dtype=float)
    low = np.array([d["interval_consistent_low"] for d in per_seed], dtype=float)
    high = np.array([d["interval_consistent_high"] for d in per_seed], dtype=float)
    # Two DIFFERENT questions, kept apart. (1) Does the discrete evaluation grid ever flip a seed's
    # sign? — per-seed intervals. (2) Do all seeds agree on the direction at all? — the point
    # estimates. An earlier version tested only per-seed unanimity and reported it under the name
    # "survives interval bounds", which conflated grid uncertainty with seed spread.
    n_seed_intervals_excluding_zero = int(((low > 0) | (high < 0)).sum())
    same_sign = bool((point > 0).all() or (point < 0).all())
    mean_bounds = [float(low.mean()), float(high.mean())]
    mean_excludes_zero = bool(mean_bounds[1] < 0 or mean_bounds[0] > 0)
    reversing = [d["seed"] for d in per_seed
                 if (d["point_difference"] > 0) != (float(np.median(point)) > 0)]
    return {
        "n_pairs": len(seeds), "per_seed": per_seed,
        "mean_point_difference": float(point.mean()),
        "median_point_difference": float(np.median(point)),
        "mean_interval_consistent_bounds": mean_bounds,
        "mean_bounds_exclude_zero": mean_excludes_zero,
        "n_seed_intervals_excluding_zero": n_seed_intervals_excluding_zero,
        "direction_unanimous_on_point_estimates": same_sign,
        "seeds_reversing_the_median_direction": reversing,
        "reading": (
            f"the evaluation grid does not limit this comparison: {n_seed_intervals_excluding_zero} "
            f"of {len(seeds)} seeds have an interval excluding zero. What limits it is the seed "
            + ("spread: the direction is NOT unanimous — "
               f"seed(s) {reversing} reverse it — so the claim is about the typical difference "
               "under the conditions examined, not a universal one."
               if not same_sign else
               "spread, and here the direction is unanimous across all seeds.")),
        "definition": ("a crossing at step c is known only to lie in (c - eval_every_test, c]; the "
                       "paired difference bound is [(a - e_a) - b, a - (b - e_b)]"),
        "wording_rule": ("the phrase 'exact transition' is not used; every crossing is reported with "
                         "its evaluation interval"),
    }


def compute(agg_dir: Path | str | None = None) -> dict:
    """The eight pre-specified comparisons, with everything §2 requires for each."""
    results_dir = runs_dir().parent / "results"
    agg_dir = Path(agg_dir) if agg_dir else results_dir / "aggregate"
    report: dict = {
        "module": MODULE, "module_version": MODULE_VERSION, "created_utc": utcnow(),
        "analysis_git_commit": git_commit(), "aggregate_dir": str(agg_dir),
        "n_boot": N_BOOT, "seed": SEED,
        "difference_definition": f"d_s = x_{ARCH_A}(s) - x_{ARCH_B}(s)",
        "rule": ("exactly the eight comparisons of STATISTICAL_ANALYSIS_PLAN §3; none is added after "
                 "seeing a result. Anything else is exploratory and is not reported here."),
        "p_value_status": ("descriptive companions only (§4). No claim in this study rests on a "
                           "p-value; claims rest on the effect size, its interval and the per-seed "
                           "pattern."),
        "comparisons": {},
    }
    raw_p: dict[str, float | None] = {}
    for spec in COMPARISONS:
        rows = load_rows(agg_dir, spec["modules"])
        for point in spec["points"]:
            key = f"{spec['id']}__{point}"
            per_arch = values_by_arch(rows, point, spec["get"])
            a, b = per_arch.get(ARCH_A, {}), per_arch.get(ARCH_B, {})
            seeds = sorted(set(a) & set(b))
            entry: dict = {"label": spec["label"], "point": point,
                           f"n_{ARCH_A}": len(a), f"n_{ARCH_B}": len(b),
                           "n_pairs": len(seeds), "seeds": seeds,
                           "seeds_only_transformer": sorted(set(a) - set(b)),
                           "seeds_only_mlp": sorted(set(b) - set(a))}
            if not seeds:
                entry["not_computed"] = "no seed has both architectures"
                report["comparisons"][key] = entry
                raw_p[key] = None
                continue
            va = [a[s] for s in seeds]
            vb = [b[s] for s in seeds]
            st = ST.paired(va, vb, seeds, seeds, n_boot=N_BOOT, seed=SEED)
            diffs = np.asarray(st["differences"], dtype=float)
            entry.update({
                "values_transformer": va, "values_mlp": vb,
                "per_seed_differences": [float(d) for d in diffs],   # §2: all of them, always
                "mean_diff": st["mean_diff"], "median_diff": st["median_diff"],
                "bootstrap_ci95": st["bootstrap_ci95"],
                "ci95_spans_zero": bool(st["bootstrap_ci95"][0] <= 0 <= st["bootstrap_ci95"][1]),
                "sign_test_p": st["sign_test_p"],
                "wilcoxon_signed_rank_p": st["wilcoxon_signed_rank_p"],
                "cohens_dz": st["cohens_dz"], "cliffs_delta": st["cliffs_delta"],
                "direction": st["direction"], "all_positive": st["all_positive"],
                "all_negative": st["all_negative"],
                "robust_spread_of_differences": ST.robust(diffs),
                "n_dropped": st["n_dropped"], "dropped_seeds": st["dropped_seeds"],
            })
            entry["reading"] = (
                ("the 95 % interval spans zero" if entry["ci95_spans_zero"]
                 else "the 95 % interval excludes zero")
                + ("; the sign is unanimous across all seeds"
                   if (st["all_positive"] or st["all_negative"]) else "; the sign is mixed")
                + ". Under the conditions examined only.")
            report["comparisons"][key] = entry
            raw_p[key] = st["wilcoxon_signed_rank_p"]

    adjusted = holm(raw_p)
    report["holm_adjusted_wilcoxon_p"] = {
        "values": adjusted,
        "status": ("a descriptive aid across the eight comparisons (§4), not a decision rule; "
                   "reported because a reader will want it"),
    }
    timing_rows = load_rows(agg_dir, ("mlp_mechanism", "transformer_mechanism"))
    report["timing_interval_consistency"] = timing_bounds(timing_rows)
    report["completeness"] = completeness(results_dir / "run_manifest.json")
    report["small_sample_discipline"] = [
        "no claim of the form 'architecture X is fundamentally faster/better' — only 'under the "
        "conditions examined'",
        "an interval spanning zero is reported as spanning zero, in the same sentence as the point "
        "estimate",
        "a unanimous sign is reported as such: with n = 10 the exact two-sided sign-test p is "
        "1/512 ~ 0.002, informative even when the magnitude is uncertain",
        "n started and n completed are both given",
    ]
    return report


def summary_lines(report: dict) -> list[str]:
    lines = [f"{len(report['comparisons'])} pre-specified comparisons "
             f"(differences are {report['difference_definition']})"]
    for key, c in report["comparisons"].items():
        if "not_computed" in c:
            lines.append(f"  {key:44s} NOT COMPUTED — {c['not_computed']}")
            continue
        ci = c["bootstrap_ci95"]
        flag = "spans 0" if c["ci95_spans_zero"] else "excludes 0"
        unan = " unanimous" if (c["all_positive"] or c["all_negative"]) else ""
        lines.append(f"  {key:44s} n={c['n_pairs']:2d}  median_d={c['median_diff']:+.4g}  "
                     f"CI95=[{ci[0]:+.4g},{ci[1]:+.4g}] {flag}{unan}")
    t = report["timing_interval_consistency"]
    if t.get("n_pairs"):
        lines.append(f"  timing: mean point difference {t['mean_point_difference']:+.0f} steps; "
                     f"interval-consistent bounds "
                     f"[{t['mean_interval_consistent_bounds'][0]:+.0f}, "
                     f"{t['mean_interval_consistent_bounds'][1]:+.0f}] "
                     f"({'exclude' if t['mean_bounds_exclude_zero'] else 'span'} zero); "
                     f"{t['n_seed_intervals_excluding_zero']}/{t['n_pairs']} seed intervals exclude "
                     f"zero; direction unanimous: "
                     f"{t['direction_unanimous_on_point_estimates']}"
                     + (f" (seeds {t['seeds_reversing_the_median_direction']} reverse it)"
                        if t["seeds_reversing_the_median_direction"] else ""))
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description="The eight pre-specified paired comparisons (§3)")
    ap.add_argument("--aggregate-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    report = compute(args.aggregate_dir)
    out = args.out or (runs_dir().parent / "results" / "statistics.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("\n".join(summary_lines(report)))
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
