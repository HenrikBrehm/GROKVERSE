"""The control blocks: the Grokfast × train_frac factorial, parameter-matched, and two-hot.

WHY THIS MODULE EXISTS
----------------------
Master prompt §14 forbids attributing a change to Grokfast **or** to the training fraction while the
two have not been separated, and it requires a parameter-matched comparison and an input-parametrization
control. `docs/STATISTICAL_ANALYSIS_PLAN.md` §7–§8 fixes how each is analysed. This module does exactly
those three analyses and keeps them apart — §8: "Each is a **separate paired comparison**, never merged
with the primary one."

WHAT THE FACTORIAL CAN AND CANNOT DO
-------------------------------------
Three paired seeds per cell. The intervals are wide and are reported as such. The decomposition exists
so that the historical observation — the legacy speed ratio moving from ~2.9× to ~1.3× between two
settings that differed in **both** knobs — is not used as evidence about either knob. If the
decomposition does not separate them here either, that is the result.

WHAT EACH CONTROL CAN SUPPORT (§8, verbatim in the output)
-----------------------------------------------------------
* **parameter-matched** — that a difference survives, or does not survive, equalizing the parameter
  count. It does **not** equalize the *shape* of the budget, so it cannot rule out capacity-allocation
  effects.
* **two-hot**, 3 seeds — that a difference is, or is not, attributable to the input parametrization
  rather than the architecture. With 3 seeds it can only show a large effect; **a null here is weak
  evidence**, and the output says so next to the number.

STATUS: applies pre-specified comparisons to already-measured numbers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..utils import git_commit, runs_dir, utcnow
from . import statistics as ST
from .decision_tree import _is_point
from .figures_study import load_rows

MODULE = "controls_report"
MODULE_VERSION = "1.0"
N_BOOT = 10_000
SEED = 0
MODULES = ("mlp_mechanism", "transformer_mechanism", "wave_fitting", "causal_ablation")

#: Metrics compared in every control. Each is (label, extractor).
METRICS: dict[str, callable] = {
    "generalization_step": lambda r: r.get("generalization_step"),
    "grokking_gap": lambda r: (None if r.get("generalization_step") is None
                               or r.get("memorization_step") is None
                               else r["generalization_step"] - r["memorization_step"]),
    "structured_fraction_of_live": lambda r: r.get("structured_fraction_of_live"),
    "phase_R": lambda r: r.get("phase_R"),
    "square_or_harmonic_fraction": lambda r: (
        None if r.get("u_a__fraction_best_aic_square") is None else
        (r.get("u_a__fraction_best_aic_square") or 0.0)
        + (r.get("u_a__fraction_best_aic_odd_harmonics") or 0.0)),
}


def _num(x):
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v and abs(v) != float("inf") else None


def _cell(row) -> tuple:
    """(grokfast, train_frac) — the factorial cell a run belongs to."""
    return (bool(row.get("grokfast")), float(row.get("train_frac") or 0))


def _bootstrap_mean(values, n_boot=N_BOOT, seed=SEED) -> dict:
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if v.size == 0:
        return {"n": 0}
    rng = np.random.default_rng(seed)
    draws = np.array([v[rng.integers(0, v.size, v.size)].mean() for _ in range(n_boot)])
    return {"n": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)),
            "values": [float(x) for x in v],
            "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
            "ci95_contains_zero": bool(np.quantile(draws, 0.025) <= 0 <= np.quantile(draws, 0.975))}


def factorial(rows, metric_name: str, get, point: str = "final") -> dict:
    """The descriptive 2 × 2 decomposition of §7, per architecture, on paired seeds."""
    per_arch: dict[str, dict] = {}
    for row in rows:
        if not _is_point(row, point):
            continue
        arch, seed = row.get("arch"), row.get("seed")
        value = _num(get(row))
        if arch is None or seed is None or value is None:
            continue
        per_arch.setdefault(arch, {}).setdefault(_cell(row), {})[int(seed)] = value

    out: dict = {"metric": metric_name, "point": point, "per_architecture": {}}
    for arch, cells in sorted(per_arch.items()):
        need = [(False, 0.3), (True, 0.3), (False, 0.5), (True, 0.5)]
        present = {c: sorted(cells.get(c, {})) for c in need}
        seeds = sorted(set.intersection(*[set(cells.get(c, {})) for c in need])) \
            if all(cells.get(c) for c in need) else []
        block = {"cells_present": {f"gf={c[0]},frac={c[1]}": present[c] for c in need},
                 "paired_seeds": seeds, "n_pairs": len(seeds)}
        if not seeds:
            block["not_computed"] = ("no seed has all four cells; the decomposition needs the "
                                     "primary block's frac=0.3/no-Grokfast runs as one cell")
            out["per_architecture"][arch] = block
            continue
        gf_at_03 = [cells[(True, 0.3)][s] - cells[(False, 0.3)][s] for s in seeds]
        gf_at_05 = [cells[(True, 0.5)][s] - cells[(False, 0.5)][s] for s in seeds]
        fr_at_nogf = [cells[(False, 0.5)][s] - cells[(False, 0.3)][s] for s in seeds]
        fr_at_gf = [cells[(True, 0.5)][s] - cells[(True, 0.3)][s] for s in seeds]
        main_gf = [(a + b) / 2 for a, b in zip(gf_at_03, gf_at_05)]
        main_fr = [(a + b) / 2 for a, b in zip(fr_at_nogf, fr_at_gf)]
        inter = [a - b for a, b in zip(gf_at_05, gf_at_03)]
        block.update({
            "cell_means": {f"gf={c[0]},frac={c[1]}": float(np.mean([cells[c][s] for s in seeds]))
                           for c in need},
            "simple_effect_grokfast_at_frac0.3": _bootstrap_mean(gf_at_03),
            "simple_effect_grokfast_at_frac0.5": _bootstrap_mean(gf_at_05),
            "main_effect_grokfast": _bootstrap_mean(main_gf),
            "main_effect_train_frac": _bootstrap_mean(main_fr),
            "interaction": _bootstrap_mean(inter),
            "reading": (
                "with 3 paired seeds per cell the intervals are wide; no effect is attributed to "
                "Grokfast or to the training fraction alone unless this decomposition separates "
                "them"),
        })
        out["per_architecture"][arch] = block
    return out


def paired_control(rows, metric_name: str, get, arch_a: str, arch_b: str,
                   filter_a=None, filter_b=None, point: str = "final") -> dict:
    """One separate paired comparison — never merged with the primary one (§8)."""
    va: dict[int, float] = {}
    vb: dict[int, float] = {}
    for row in rows:
        if not _is_point(row, point):
            continue
        seed, value = row.get("seed"), _num(get(row))
        if seed is None or value is None:
            continue
        if row.get("arch") == arch_a and (filter_a is None or filter_a(row)):
            va[int(seed)] = value
        if row.get("arch") == arch_b and (filter_b is None or filter_b(row)):
            vb[int(seed)] = value
    seeds = sorted(set(va) & set(vb))
    out = {"metric": metric_name, "point": point, "a": arch_a, "b": arch_b,
           "n_pairs": len(seeds), "seeds": seeds,
           "seeds_only_a": sorted(set(va) - set(vb)), "seeds_only_b": sorted(set(vb) - set(va))}
    if not seeds:
        out["not_computed"] = "no seed has both sides"
        return out
    st = ST.paired([va[s] for s in seeds], [vb[s] for s in seeds], seeds, seeds,
                   n_boot=N_BOOT, seed=SEED)
    out.update({k: st[k] for k in ("values_a", "values_b", "differences", "mean_diff",
                                   "median_diff", "bootstrap_ci95", "direction", "all_positive",
                                   "all_negative", "sign_test_p", "cohens_dz", "cliffs_delta")})
    ci = st["bootstrap_ci95"]
    out["ci95_contains_zero"] = bool(ci[0] <= 0 <= ci[1])
    return out


def compute(agg_dir: Path | str | None = None, point: str = "final") -> dict:
    agg_dir = Path(agg_dir) if agg_dir else (runs_dir().parent / "results" / "aggregate")
    rows = load_rows(agg_dir, MODULES)
    # one row per (run, point) — the mechanism modules already split by architecture, and merging
    # keeps a run's metadata with its metrics
    merged: dict[tuple, dict] = {}
    for row in rows:
        key = (row.get("run_id"), str(row.get("tag")))
        merged.setdefault(key, {}).update({k: v for k, v in row.items() if v is not None})
    rows = list(merged.values())

    is_primary = lambda r: (not r.get("grokfast") and r.get("train_frac") == 0.3
                            and r.get("d_mlp") != 572 and r.get("arch") != "mlp_twohot")
    is_matched = lambda r: r.get("d_mlp") == 572

    report = {
        "module": MODULE, "module_version": MODULE_VERSION, "created_utc": utcnow(),
        "analysis_git_commit": git_commit(), "measurement_point": point,
        "n_rows": len(rows),
        "separation_rule": ("master prompt §14: no change is attributed to Grokfast or to the "
                            "training fraction alone unless the 2x2 decomposition separates them"),
        "controls_are_separate": ("STATISTICAL_ANALYSIS_PLAN §8: each control is its own paired "
                                  "comparison and is never merged with the primary one"),
        "factorial_grokfast_x_train_frac": {},
        "parameter_matched": {},
        "two_hot": {},
    }
    for name, get in METRICS.items():
        report["factorial_grokfast_x_train_frac"][name] = factorial(rows, name, get, point)
        report["parameter_matched"][name] = paired_control(
            rows, name, get, "transformer", "mlp",
            filter_a=is_primary, filter_b=is_matched, point=point)
        report["parameter_matched"][name]["what_it_can_support"] = (
            "that a difference survives, or does not survive, equalizing the parameter count; it "
            "does NOT equalize the shape of the budget, so capacity-allocation effects remain")
        report["two_hot"][name] = paired_control(
            rows, name, get, "mlp_twohot", "mlp",
            filter_a=None, filter_b=is_primary, point=point)
        report["two_hot"][name]["what_it_can_support"] = (
            "that a difference is, or is not, attributable to the input parametrization rather than "
            "the architecture; with 3 seeds it can only show a large effect, and A NULL HERE IS "
            "WEAK EVIDENCE")
    return report


def summary_lines(rep: dict) -> list[str]:
    lines = [f"controls at point={rep['measurement_point']} ({rep['n_rows']} rows)"]
    for metric, block in rep["factorial_grokfast_x_train_frac"].items():
        for arch, b in block["per_architecture"].items():
            if "not_computed" in b:
                lines.append(f"  factorial {metric:28s} {arch:12s} NOT COMPUTED — {b['not_computed'][:60]}")
                continue
            g, f_, i = b["main_effect_grokfast"], b["main_effect_train_frac"], b["interaction"]
            lines.append(
                f"  factorial {metric:28s} {arch:12s} n={b['n_pairs']}  "
                f"GF {g['mean']:+.4g} [{g['ci95'][0]:+.4g},{g['ci95'][1]:+.4g}]"
                f"{'*' if not g['ci95_contains_zero'] else ''}  "
                f"frac {f_['mean']:+.4g} [{f_['ci95'][0]:+.4g},{f_['ci95'][1]:+.4g}]"
                f"{'*' if not f_['ci95_contains_zero'] else ''}  "
                f"inter {i['mean']:+.4g}{'*' if not i['ci95_contains_zero'] else ''}")
    for label, key in (("param-matched", "parameter_matched"), ("two-hot", "two_hot")):
        for metric, b in rep[key].items():
            if "not_computed" in b:
                lines.append(f"  {label:14s} {metric:28s} NOT COMPUTED — {b['not_computed']}")
                continue
            ci = b["bootstrap_ci95"]
            lines.append(f"  {label:14s} {metric:28s} n={b['n_pairs']}  median_d {b['median_diff']:+.4g}  "
                         f"CI [{ci[0]:+.4g},{ci[1]:+.4g}] "
                         f"{'spans 0' if b['ci95_contains_zero'] else 'excludes 0'}")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description="Control blocks (STATISTICAL_ANALYSIS_PLAN §7–§8)")
    ap.add_argument("--aggregate-dir", type=Path, default=None)
    ap.add_argument("--point", default="final")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    rep = compute(args.aggregate_dir, args.point)
    out = args.out or (runs_dir().parent / "results" / "controls_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    print("\n".join(summary_lines(rep)))
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
