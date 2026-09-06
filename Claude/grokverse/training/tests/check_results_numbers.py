"""Cross-check the numbers written in RESULTS.md against the stored artifacts.

Run: ``python tests/check_results_numbers.py`` from ``training/``.

WHY THIS EXISTS
---------------
`RESULTS.md` is written by hand from the analysis artifacts, and a transcription error there is
invisible: the prose reads fine, the artifact is untouched, and no test covers the gap between them.
This study has already produced five defects whose common shape was "well-formed output that nobody
cross-checked" (labbook 58, 99, 101), so the document that quotes those outputs gets the same
treatment as the code that produces them.

Each entry below names a number as it appears in `RESULTS.md` and a callable that re-derives it from
`results/`. The check fails if the two disagree beyond the tolerance implied by the printed precision.
It is deliberately *not* wired into `run_all.py`: it needs the completed run matrix, while the suite
must stay runnable on a clean checkout.
"""
from __future__ import annotations

import json
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT.parent / "RESULTS.md"
RES = ROOT / "results"

FAILURES: list[str] = []


def load(rel: str) -> dict:
    return json.loads((RES / rel).read_text(encoding="utf-8"))


def rows(module: str, agg: str = "aggregate") -> list[dict]:
    return load(f"{agg}/{module}.json")["rows"]


def final(rs: list[dict], arch: str) -> list[dict]:
    return [r for r in rs if r.get("tag") == "step025000" and r.get("arch") == arch]


def med(rs: list[dict], key: str) -> float:
    v = [r[key] for r in rs if isinstance(r.get(key), (int, float)) and not isinstance(r.get(key), bool)]
    if not v:
        raise KeyError(f"no numeric values for {key!r}")
    return st.median(v)


def check(label: str, written: float, measured: float, tol: float) -> None:
    ok = abs(written - measured) <= tol
    print(f"[{'PASS' if ok else 'FAIL'}] {label}: RESULTS.md says {written:g}, artifacts give {measured:g}")
    if not ok:
        FAILURES.append(f"{label}: written {written:g} vs measured {measured:g} (tol {tol:g})")


def check_in_doc(label: str, needle: str) -> None:
    """The literal string must appear in RESULTS.md — for verdicts, not magnitudes."""
    ok = needle in DOC.read_text(encoding="utf-8")
    print(f"[{'PASS' if ok else 'FAIL'}] {label}: {needle!r} present in RESULTS.md")
    if not ok:
        FAILURES.append(f"{label}: {needle!r} missing from RESULTS.md")


# --------------------------------------------------------------------------- #
def check_transitions() -> None:
    print("\n-- section 2: transitions --")
    import glob
    for arch, pat, mem, gen in (("transformer", "txf_add_p113_wd1.0_frac0.3_seed?_arch25k", 140, 7588),
                                ("mlp", "mlp_add_p113_wd1.0_frac0.3_seed?_arch25k", 160, 9250)):
        t = [json.loads((Path(d) / "run.json").read_text(encoding="utf-8"))["transitions"]["primary"]
             for d in sorted(glob.glob(str(ROOT / "runs" / pat)))]
        check(f"{arch} memorization median", mem,
              st.median([x["memorization"]["first_crossing_step"] for x in t]), 0.5)
        check(f"{arch} generalization median", gen,
              st.median([x["generalization"]["first_crossing_step"] for x in t]), 0.5)


def check_gate() -> None:
    print("\n-- section 3: the gate --")
    dt = load("decision_tree_final.json")
    check_in_doc("branch is reported", "`neither_passes`")
    if dt["branch"]["branch"] != "neither_passes":
        FAILURES.append(f"branch is {dt['branch']['branch']!r}, RESULTS.md says neither_passes")
    for arch in ("transformer", "mlp"):
        S = dt["per_architecture"][arch]["seeds"]
        for g, want in (("G1", 10), ("G2", 10), ("G3", 10), ("G4", 0)):
            got = sum(1 for s in S if s["verdicts"][g] is True)
            check(f"{arch} {g} seeds passing", want, got, 0)


def check_structure() -> None:
    print("\n-- section 5: structure --")
    for arch, sf, fam, R, r2 in (("transformer", 0.982, 0.9923, 0.9921, 0.546),
                                 ("mlp", 0.885, 0.9973, 0.9996, 0.974)):
        mech = final(rows(f"{'transformer' if arch == 'transformer' else 'mlp'}_mechanism"), arch)
        check(f"{arch} structured fraction", sf, med(mech, "structured_fraction_of_live"), 0.001)
        check(f"{arch} family fraction u_a", fam, med(mech, "family_fraction_median_u_a"), 0.0001)
        check(f"{arch} phase R", R, med(mech, "phase_R"), 0.0001)
        lf = final(rows("logit_formula_fit"), arch)
        check(f"{arch} best-formula R2 (test)", r2, med(lf, "sparse_sinusoid__r2_test"), 0.001)
    check("transformer additivity_r2", 0.922,
          med(final(rows("transformer_mechanism"), "transformer"), "additivity_r2_median"), 0.001)


def check_ablations() -> None:
    print("\n-- section 6: ablations --")
    ca = rows("causal_ablation")
    for arch, ident, drop, ctrl in (
            ("mlp", "remove_structured", 0.991, 0.878),
            ("transformer", "remove_structured_neurons", 0.942, 0.910),
            ("mlp", "remove_key_freqs_from_curves", 0.989, 0.000),
            ("transformer", "remove_key_freqs_from_embedding", 0.984, 0.0005),
            ("transformer", "remove_key_subspace_from_residual", 0.235, 0.000),
            ("transformer", "fix_attention_to_mean", 0.336, None)):
        rs = final(ca, arch)
        check(f"{arch} {ident} drop", drop, med(rs, f"{ident}__drop"), 0.001)
        if ctrl is not None:
            check(f"{arch} {ident} control drop", ctrl, med(rs, f"{ident}__control_mean_drop"), 0.001)


def check_function_agreement() -> None:
    print("\n-- section 4: function agreement --")
    import glob
    fs = sorted(glob.glob(str(RES / "function_agreement" / "*_arch25k_step025000__vs__*.json")))
    ag = [json.loads(Path(f).read_text(encoding="utf-8"))["results"]["full_grid"]["agreement_rate"] for f in fs]
    lg = [json.loads(Path(f).read_text(encoding="utf-8"))["results"]["logits"]["logit_pearson"] for f in fs]
    check("agreement over the full grid", 0.99977, st.median(ag), 0.00001)
    check("logit Pearson correlation", 0.083, st.median(lg), 0.001)
    identical = sum(1 for a in ag if a == 1.0)
    check("seeds with identical functions", 4, identical, 0)


def check_h3() -> None:
    print("\n-- section 7: H3 --")
    h3 = load("h3_report.json")
    check("gap under top-1", 0.0898, h3["h3b_gap_under_top1"]["median_diff"], 0.0001)
    check("gap under the family", 0.0850, h3["h3b_criterion_2_family_gap"]["median_diff"], 0.0001)
    ws = h3["h3a_metric_waveform_sensitivity"]
    for arch in ("transformer", "mlp"):
        check(f"waveform sensitivity at top-1 ({arch})", 0.1893, ws["u_a__top1"][arch]["median"], 0.0001)
    check("waveform sensitivity at top-4", 0.0501, ws["u_a__top4"]["transformer"]["median"], 0.0001)
    check("waveform sensitivity at top-8", 0.0248, ws["u_a__top8"]["transformer"]["median"], 0.0001)
    # it is a property of the metric, so the two architectures must agree to numerical noise
    check("...identical for both architectures, as a metric property must be",
          ws["u_a__top1"]["transformer"]["median"], ws["u_a__top1"]["mlp"]["median"], 1e-12)
    check_in_doc("H3 refuted is stated", "**H3 as pre-registered is refuted.**")
    if not h3["h3_verdict"]["h3_refuted"]:
        FAILURES.append("h3_report says H3 is NOT refuted, RESULTS.md says it is")


def check_h4() -> None:
    """Section 8 / 8.1 — the trajectory numbers, including the two the index-vs-name defect hid."""
    print("\n-- section 8: H4 --")
    rs = rows("structure_over_time")
    # the six metrics that reach an onset before generalization in 10/10 seeds, both architectures
    for arch, onsets in (("transformer", {"logit_key_subspace_share": 500,
                                          "structured_fraction_of_live": 1000,
                                          "embedding_top8_concentration": 1000,
                                          "median_family_fraction": 2000,
                                          "fraction_best_aic_sinusoid": 2000,
                                          "fraction_best_aic_odd_harmonics": 3000}),
                         ("mlp", {"logit_key_subspace_share": 500,
                                  "structured_fraction_of_live": 1000,
                                  "embedding_top8_concentration": 1000,
                                  "median_family_fraction": 500,
                                  "fraction_best_aic_sinusoid": 750,
                                  "fraction_best_aic_odd_harmonics": 1000})):
        arch_rows = [r for r in rs if r["arch"] == arch]
        for metric, want in onsets.items():
            v = [r.get(f"h4__{metric}__onset_step") for r in arch_rows]
            defined = [x for x in v if x is not None]
            check(f"{arch} {metric} onset (median)", want, st.median(defined), 0.5)
            check(f"{arch} {metric} onset defined on all 10", 10, len(defined), 0)
    # section 8.1: the waveform composition table
    for arch, table in (("transformer", {"square": (0.787, 0.116, -0.6558),
                                         "sinusoid": (0.149, 0.646, +0.4946),
                                         "odd_harmonics": (0.021, 0.173, +0.1494)}),
                        ("mlp", {"square": (0.789, 0.421, -0.3662),
                                 "sinusoid": (0.148, 0.333, +0.1860),
                                 "odd_harmonics": (0.021, 0.218, +0.1938)})):
        arch_rows = [r for r in rs if r["arch"] == arch]
        for shape, (init, pre, change) in table.items():
            key = f"h4__fraction_best_aic_{shape}"
            check(f"{arch} {shape} init", init, med(arch_rows, f"{key}__init"), 0.001)
            check(f"{arch} {shape} pre-generalization", pre, med(arch_rows, f"{key}__pre_gen"), 0.001)
            check(f"{arch} {shape} change", change, med(arch_rows, f"{key}__change"), 0.0001)
        # the sign of the change must be unanimous for square (down) and sinusoid (up)
        for shape, want_sign in (("square", -1), ("sinusoid", +1)):
            v = [r[f"h4__fraction_best_aic_{shape}__change"] for r in arch_rows]
            unanimous = all((x < 0) if want_sign < 0 else (x > 0) for x in v)
            print(f"[{'PASS' if unanimous else 'FAIL'}] {arch} {shape} change is unanimous "
                  f"({'down' if want_sign < 0 else 'up'}) across 10 seeds")
            if not unanimous:
                FAILURES.append(f"{arch} {shape} change is not unanimous: {v}")
    # the square share must have NO onset -- the detector is rise-only and this metric falls
    for arch in ("transformer", "mlp"):
        arch_rows = [r for r in rs if r["arch"] == arch]
        v = [r.get("h4__fraction_best_aic_square__onset_step") for r in arch_rows]
        check(f"{arch} square-share onset is undefined on all 10 (it falls)",
              0, len([x for x in v if x is not None]), 0)
    check_in_doc("the rise-only limitation is stated", "detects only an *upward* crossing")


def check_bounded() -> None:
    print("\n-- section 9: bounded alternative --")
    ba = rows("bounded_alternative")
    for arch, er, r16 in (("transformer", 12.70, 0.9644), ("mlp", 66.64, 0.0000)):
        rs = [r for r in ba if r["arch"] == arch]
        check(f"{arch} hidden effective rank", er, med(rs, "spectrum_hidden__effective_rank"), 0.01)
        check(f"{arch} top-16 singular ablation drop", r16, med(rs, "svd_ablation__r16__drop"), 0.001)
    cka = load("cross_seed_cka.json")
    for arch, want in (("transformer", 0.0017), ("mlp", 0.0973)):
        M = cka[arch]["cka_matrix"]
        n = len(M)
        off = [M[i][j] for i in range(n) for j in range(n) if i < j]
        check(f"{arch} cross-seed CKA (off-diagonal median)", want, st.median(off), 0.0001)


def check_statistics() -> None:
    print("\n-- sections 2.1 / 11: the pre-specified comparisons --")
    S = load("statistics.json")["comparisons"]
    for cid, want, tol in (("p1_generalization_step__final", -1562.5, 0.5),
                           ("p3_structured_fraction__final", 0.0850, 0.0001),
                           ("p4_phase_R__final", -0.00744, 0.00001),
                           ("p5_square_or_harmonic_fraction__final", -0.3242, 0.0001),
                           ("p6_family_minus_top1__final", -0.00249, 0.00001),
                           ("p8_best_fourier_r2__final", -0.4257, 0.0001)):
        check(cid, want, S[cid]["median_diff"], tol)
    check_in_doc("the one interval spanning zero is named", "the one comparison of the eight pre-specified whose interval spans zero")


def check_controls() -> None:
    print("\n-- section 10: controls --")
    C = load("controls_report.json")
    pm = C["parameter_matched"]
    check("parameter-matched generalization step", -2212.5, pm["generalization_step"]["median_diff"], 0.5)
    check("parameter-matched structured fraction", 0.0969,
          pm["structured_fraction_of_live"]["median_diff"], 0.0001)
    th = C["two_hot"]
    check("two-hot square/harmonic fraction", 0.1836,
          th["square_or_harmonic_fraction"]["median_diff"], 0.0001)
    fac = C["factorial_grokfast_x_train_frac"]["generalization_step"]["per_architecture"]
    check("transformer train_frac main effect", -6367,
          fac["transformer"]["main_effect_train_frac"]["mean"], 1.0)
    check("mlp train_frac main effect", -8217,
          fac["mlp"]["main_effect_train_frac"]["mean"], 1.0)


def check_doc_has_no_stale_markers() -> None:
    print("\n-- the document itself --")
    text = DOC.read_text(encoding="utf-8")
    # A *marker* is a bare [AUDIT]; a *mention* is the backticked `[AUDIT]` in the sentence that
    # explains what the superseded version carried. Only the former is a leftover.
    markers = re.findall(r"(?<!`)\[AUDIT\](?!`)", text)
    ok = not markers
    print(f"[{'PASS' if ok else 'FAIL'}] RESULTS.md carries no un-backticked [AUDIT] marker "
          f"({len(re.findall(r'`\[AUDIT\]`', text))} historical mention(s) allowed)")
    if not ok:
        FAILURES.append(f"RESULTS.md still contains {len(markers)} bare [AUDIT] marker(s)")
    for bad in ("TODO", "TBD", "XXX", "FIXME"):
        ok = bad not in text
        print(f"[{'PASS' if ok else 'FAIL'}] RESULTS.md carries no {bad} marker")
        if not ok:
            FAILURES.append(f"RESULTS.md still contains {bad}")
    n = len(re.findall(r"HUMAN AUTHORS MUST COMPLETE", (DOC.parent / "AI_DISCLOSURE.md").read_text(encoding="utf-8")))
    print(f"[{'PASS' if n == 11 else 'FAIL'}] AI_DISCLOSURE.md keeps its 11 human placeholders (found {n})")
    if n != 11:
        FAILURES.append(f"AI_DISCLOSURE.md has {n} placeholders, expected 11")


def main() -> None:
    if not DOC.exists():
        raise SystemExit(f"cannot find {DOC}")
    check_transitions()
    check_gate()
    check_function_agreement()
    check_structure()
    check_ablations()
    check_h3()
    check_h4()
    check_bounded()
    check_statistics()
    check_controls()
    check_doc_has_no_stale_markers()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} DISAGREEMENT(S) between RESULTS.md and the artifacts:")
        for f in FAILURES:
            print("  -", f)
        sys.exit(1)
    print("RESULTS.md agrees with every artifact checked.")


if __name__ == "__main__":
    main()
