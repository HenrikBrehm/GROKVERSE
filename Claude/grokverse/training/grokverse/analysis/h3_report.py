"""H3 at study level: the artifact, the harmonic-aware structure, and the refutation criteria.

WHY THIS MODULE EXISTS
----------------------
`analysis/h3_validity.py` measures H3's quantities **per run** and deliberately makes no architecture
comparison, because H3₀ is a statement about the *gap between architectures*, paired across seeds.
This module makes that comparison, and applies `docs/PREREGISTRATION.md` §3's refutation criteria
**literally, in both directions**:

> **H3₀.** The architecture gap in structured-neuron fraction is the same under the top-1 and the
> family definitions.
>
> **Refuted if** the paired difference (gap under top-1 minus gap under family) has a 95 % CI
> containing 0, **or** if the MLP's family-based structured fraction is still lower than the
> transformer's with a CI excluding 0 — in which case the deficit is a real loss of structure, not a
> measurement artifact.

Both branches are evaluated and reported. A criterion that refutes the hypothesis is as much a result
as one that supports it, and §3.4 requires negative results to be preserved.

THE GATE COMES FIRST, AND IT IS NOT OPTIONAL
---------------------------------------------
Master prompt §12, verbatim: *"Harmonic-family concentration may be interpreted as the same Fourier
principle in a different form only for an architecture that passes these tests."* This module reads
`results/decision_tree_final.json` and states, at the top of its own output, which architectures
passed. If an architecture did not, its H3 numbers are measurements about a **metric** and may not be
read as evidence about a mechanism — and the output says so rather than leaving it to the reader.

THE CRITERION THAT CANNOT BE SATISFIED
---------------------------------------
"The harmonic family beats the cardinality-matched top-m concentration" is unsatisfiable by
construction. It is checked here across every run as an implementation audit — the maximum of
`family − matched_top_m` must be ≤ 0 — and never used as support.

STATUS: applies pre-registered criteria to already-measured numbers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..utils import git_commit, runs_dir, utcnow
from . import statistics as ST
from .figures_study import ARCH_A, ARCH_B, load_rows, values_by_arch

MODULE = "h3_report"
MODULE_VERSION = "1.0"
N_BOOT = 10_000
SEED = 0
FAMILY_KEY = "h3b__structured_fraction__primary_family_0.50"
TOP1_KEY = "h3b__structured_fraction__top1_0.50"
CURVES = ("u_a", "u_b", "out")
TOPK = (1, 4, 8)


def _paired(rows, point, get):
    per = values_by_arch(rows, point, get)
    a, b = per.get(ARCH_A, {}), per.get(ARCH_B, {})
    seeds = sorted(set(a) & set(b))
    if not seeds:
        return None, [], [], []
    va, vb = [a[s] for s in seeds], [b[s] for s in seeds]
    return ST.paired(va, vb, seeds, seeds, n_boot=N_BOOT, seed=SEED), seeds, va, vb


def _pick(st, extra=None):
    """The reporting subset of a `statistics.paired` result, with the per-seed values kept."""
    if st is None:
        return {"n_pairs": 0, "note": "no seed has both architectures"}
    out = {k: st[k] for k in ("n_pairs", "seeds", "values_a", "values_b", "differences",
                              "mean_diff", "median_diff", "bootstrap_ci95", "direction",
                              "all_positive", "all_negative", "sign_test_p",
                              "wilcoxon_signed_rank_p", "cohens_dz", "cliffs_delta") if k in st}
    ci = st["bootstrap_ci95"]
    out["ci95_contains_zero"] = bool(ci[0] <= 0 <= ci[1])
    out["a_is"] = ARCH_A
    out["b_is"] = ARCH_B
    if extra:
        out.update(extra)
    return out


def gate_status(path: Path) -> dict:
    """Which architectures passed the evidence gate — the precondition for reading H3 at all."""
    if not path.exists():
        # An absent gate is the MOST restrictive case, not a neutral one: without it nothing is
        # cleared. It carries the same `consequence` key as the normal path so a caller reading the
        # report cannot get a partially-populated gate block (that used to raise a KeyError, which
        # would have made a missing gate look like a crashed run rather than an unmet precondition).
        return {"available": False, "architectures_passing": [],
                "note": f"{path.name} not found; H3 numbers may not be interpreted without it",
                "interpretation_rule": (
                    "master prompt §12: harmonic-family concentration may be interpreted as the same "
                    "Fourier principle in a different form ONLY for an architecture that passes the "
                    "gate"),
                "consequence": ("the evidence gate has not been evaluated, so NO architecture is "
                                "cleared. Every H3 number below is a measurement about the METRIC, "
                                "not evidence about a mechanism.")}
    rep = json.loads(path.read_text(encoding="utf-8"))
    per = {a: b.get("passes_gate") for a, b in rep.get("per_architecture", {}).items()}
    passed = sorted(a for a, v in per.items() if v is True)
    return {
        "available": True, "source": path.name,
        "measurement_point": rep.get("measurement_point"),
        "passes_gate": per, "architectures_passing": passed,
        "branch": rep.get("branch", {}).get("branch"),
        "interpretation_rule": (
            "master prompt §12: harmonic-family concentration may be interpreted as the same Fourier "
            "principle in a different form ONLY for an architecture that passes the gate"),
        "consequence": (
            f"architectures cleared to carry an H3 mechanism reading: {passed or 'NONE'}. "
            + ("Every H3 number below is therefore a measurement about the METRIC, not evidence "
               "about a mechanism." if not passed else
               "For the others the numbers remain metric measurements only.")),
    }


def compute(agg_dir: Path | str | None = None, decision_path: Path | str | None = None,
            point: str = "final") -> dict:
    results = runs_dir().parent / "results"
    agg_dir = Path(agg_dir) if agg_dir else results / "aggregate"
    decision_path = Path(decision_path) if decision_path else results / "decision_tree_final.json"
    rows = load_rows(agg_dir, ("h3_validity",))

    report: dict = {
        "module": MODULE, "module_version": MODULE_VERSION, "created_utc": utcnow(),
        "analysis_git_commit": git_commit(), "measurement_point": point,
        "difference_definition": f"d_s = x_{ARCH_A}(s) - x_{ARCH_B}(s)",
        "gate": gate_status(decision_path),
    }

    # --- H3a: the metric's waveform sensitivity, per architecture --------------------------------
    h3a: dict = {"definition": (
        "two populations built from each checkpoint's own dominant (k, phase, amplitude), differing "
        "ONLY in waveform; sensitivity = median top-k(sinusoid) - median top-k(square). A positive "
        "value means the concentration metric scores the square population as less concentrated "
        "although both carry the same fundamentals — a property of the METRIC.")}
    for curve in CURVES:
        for k in TOPK:
            key = f"h3a__{curve}__waveform_sensitivity_top{k}"
            per = values_by_arch(rows, point, lambda r, kk=key: r.get(kk))
            h3a[f"{curve}__top{k}"] = {
                arch: ST.robust(list(vals.values())) for arch, vals in per.items()}
        per_shape = values_by_arch(rows, point, lambda r, c=curve: r.get(f"h3a__{c}__shape_separation"))
        h3a[f"{curve}__shape_separation"] = {
            arch: ST.robust(list(vals.values())) for arch, vals in per_shape.items()}
    report["h3a_metric_waveform_sensitivity"] = h3a

    # --- H3b: the two refutation criteria, applied literally -------------------------------------
    st_gap_t1, seeds, _, _ = _paired(rows, point, lambda r: r.get(TOP1_KEY))
    st_gap_fam, _, _, _ = _paired(rows, point, lambda r: r.get(FAMILY_KEY))
    criterion_1 = None
    if st_gap_t1 and st_gap_fam:
        d_top1 = np.asarray(st_gap_t1["differences"], dtype=float)
        d_fam = np.asarray(st_gap_fam["differences"], dtype=float)
        delta = d_top1 - d_fam                      # (gap under top-1) - (gap under family)
        rng = np.random.default_rng(SEED)
        draws = np.array([delta[rng.integers(0, delta.size, delta.size)].mean()
                          for _ in range(N_BOOT)])
        ci = [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]
        contains_zero = bool(ci[0] <= 0 <= ci[1])
        criterion_1 = {
            "quantity": "(architecture gap under top-1) minus (architecture gap under the family)",
            "per_seed": [float(x) for x in delta], "seeds": st_gap_t1["seeds"],
            "mean": float(delta.mean()), "median": float(np.median(delta)),
            "bootstrap_ci95": ci, "ci95_contains_zero": contains_zero, "n_boot": N_BOOT,
            "refutes_H3": contains_zero,
            "reading": ("the CI contains zero, so by the pre-registered criterion the gap is the "
                        "same under both definitions and H3_0 is NOT refuted by this branch"
                        if contains_zero else
                        "the CI excludes zero, so the gap differs between the two definitions"),
        }
    report["h3b_criterion_1_gap_difference"] = criterion_1

    st_family = _pick(st_gap_fam)
    if st_gap_fam:
        ci = st_gap_fam["bootstrap_ci95"]
        mlp_lower = bool(ci[0] > 0)                 # transformer - mlp > 0 means MLP is lower
        st_family.update({
            "quantity": "structured-neuron fraction under the FAMILY definition",
            "mlp_still_lower_with_ci_excluding_zero": mlp_lower,
            "refutes_H3": mlp_lower,
            "reading": ("the MLP's family-based structured fraction is still lower than the "
                        "transformer's with a CI excluding zero, so by the pre-registered criterion "
                        "the deficit is a real loss of structure, not a measurement artifact"
                        if mlp_lower else
                        "the MLP's family-based fraction is not lower with a CI excluding zero, so "
                        "this branch does not refute H3_0"),
        })
    report["h3b_criterion_2_family_gap"] = st_family
    report["h3b_gap_under_top1"] = _pick(st_gap_t1, {
        "quantity": "structured-neuron fraction under the TOP-1 definition"})

    refuted = [name for name, block in (("criterion_1_gap_difference", criterion_1),
                                        ("criterion_2_family_gap", st_family))
               if isinstance(block, dict) and block.get("refutes_H3")]
    # PREREGISTRATION §3 lists these as the conditions under which **H3** is refuted — not H3_0.
    # H3 predicts that the architecture gap CLOSES under the family-based definition; H3_0 is the
    # null that the gap is the same under both. Naming this block after H3_0 (as a first version did)
    # inverts the headline.
    report["h3_verdict"] = {
        "h3_refuted": bool(refuted), "refuting_criteria": refuted,
        "criteria_verbatim": (
            "H3 is refuted if the paired difference (gap under top-1 minus gap under family) has a "
            "95 % CI containing 0, OR if the MLP's family-based structured fraction is still lower "
            "than the transformer's with a CI excluding 0 — in which case the deficit is a real loss "
            "of structure, not a measurement artifact."),
        "h3_prediction": ("the architecture gap in the structured-neuron fraction closes under the "
                          "family-based definition but not under the top-1 definition"),
        "h3_0_is": "the null that the gap is the SAME under both definitions",
        "gate_precondition": report["gate"]["consequence"],
    }
    if criterion_1 and st_family.get("n_pairs"):
        gap_top1 = float(np.mean(criterion_1["per_seed"]) + st_family["median_diff"])
        report["h3_verdict"]["how_much_of_the_gap_the_family_closes"] = {
            "gap_under_top1_median": float(st_gap_t1["median_diff"]),
            "gap_under_family_median": float(st_family["median_diff"]),
            "closed_median": float(criterion_1["median"]),
            "closed_fraction_of_the_top1_gap": (
                float(criterion_1["median"] / st_gap_t1["median_diff"])
                if st_gap_t1["median_diff"] else None),
            "reading": ("this is the substantive quantity behind H3: how much of the architecture "
                        "gap the harmonic-aware definition removes"),
        }

    # --- H3c: the legacy embedding number, descriptive only --------------------------------------
    h3c: dict = {"object": next((r.get("h3c__object") for r in rows if r.get("h3c__object")), None),
                 "caveat": ("W_E alone decides nothing for the MLP, which reads it through two "
                            "halves of W_in; reported descriptively (PREREGISTRATION §3, H3c)")}
    per_top8 = values_by_arch(rows, point, lambda r: r.get("h3c__top8_concentration"))
    h3c["top8_concentration"] = {arch: ST.robust(list(v.values())) for arch, v in per_top8.items()}
    st_top8, _, _, _ = _paired(rows, point, lambda r: r.get("h3c__top8_concentration"))
    h3c["top8_paired"] = _pick(st_top8)
    for n_f in (1, 2, 4, 8):
        per_fam = values_by_arch(rows, point,
                                 lambda r, n=n_f: r.get(f"h3c__nf{n}__family_fraction"))
        per_null = values_by_arch(rows, point, lambda r, n=n_f: r.get(f"h3c__nf{n}__null_mean"))
        h3c[f"n_fundamentals_{n_f}"] = {
            "family_fraction": {a: ST.robust(list(v.values())) for a, v in per_fam.items()},
            "random_null": {a: ST.robust(list(v.values())) for a, v in per_null.items()}}
    report["h3c_legacy_embedding"] = h3c

    # --- the audit of the criterion that cannot be satisfied --------------------------------------
    worst = max((r.get(f"h3b__{c}__family_minus_matched_max") or -1.0)
                for r in rows for c in CURVES)
    report["forbidden_criterion_audit"] = {
        "max_family_minus_matched_top_m_over_all_runs": float(worst),
        "holds": bool(worst <= 1e-12),
        "statement": ("'the family beats the matched top-m' is unsatisfiable by construction; this "
                      "is an implementation audit, never support for H3"),
    }
    return report


def summary_lines(rep: dict) -> list[str]:
    g = rep["gate"]
    lines = [f"gate: passing = {g.get('architectures_passing')} (branch {g.get('branch')})",
             f"  {g.get('consequence')}"]
    c1 = rep.get("h3b_criterion_1_gap_difference")
    if c1:
        lines.append(f"  criterion 1 (top-1 gap minus family gap): median {c1['median']:+.4f} "
                     f"CI95 [{c1['bootstrap_ci95'][0]:+.4f}, {c1['bootstrap_ci95'][1]:+.4f}] "
                     f"-> refutes H3: {c1['refutes_H3']}")
    c2 = rep.get("h3b_criterion_2_family_gap") or {}
    if c2.get("n_pairs"):
        lines.append(f"  criterion 2 (family-definition gap): median {c2['median_diff']:+.4f} "
                     f"CI95 [{c2['bootstrap_ci95'][0]:+.4f}, {c2['bootstrap_ci95'][1]:+.4f}] "
                     f"-> refutes H3: {c2['refutes_H3']}")
    v = rep["h3_verdict"]
    lines.append(f"  H3 REFUTED: {v['h3_refuted']} by {v['refuting_criteria']}")
    closed = v.get("how_much_of_the_gap_the_family_closes")
    if closed:
        frac = closed["closed_fraction_of_the_top1_gap"]
        lines.append(f"    gap under top-1 {closed['gap_under_top1_median']:+.4f} -> under family "
                     f"{closed['gap_under_family_median']:+.4f}; the family definition closes "
                     f"{closed['closed_median']:+.4f}"
                     + (f" ({frac:.1%} of it)" if frac is not None else ""))
    a = rep["h3a_metric_waveform_sensitivity"]
    for arch, block in sorted(a.get("u_a__top1", {}).items()):
        lines.append(f"  H3a {arch:12s} u_a top-1 waveform sensitivity median "
                     f"{block.get('median', float('nan')):+.4f}")
    f = rep["forbidden_criterion_audit"]
    lines.append(f"  forbidden-criterion audit holds: {f['holds']} "
                 f"(max {f['max_family_minus_matched_top_m_over_all_runs']:+.2e})")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description="H3 at study level (PREREGISTRATION §3)")
    ap.add_argument("--aggregate-dir", type=Path, default=None)
    ap.add_argument("--decision", type=Path, default=None)
    ap.add_argument("--point", default="final")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    rep = compute(args.aggregate_dir, args.decision, args.point)
    out = args.out or (runs_dir().parent / "results" / "h3_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    print("\n".join(summary_lines(rep)))
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
