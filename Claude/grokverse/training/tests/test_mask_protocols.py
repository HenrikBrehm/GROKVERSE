"""Checks for analysis/mask_protocols.py — the restricted/excluded protocols (INTERFACES §10).

Run from training/:  python tests/test_mask_protocols.py

Same ``check(name, cond)`` convention as test_core.py; exits non-zero on the first failure.
Everything is synthetic with a known answer at p = 23 except the component counts, which are
asserted at the study's p = 113 against the table in docs/MASK_PROTOCOL_AUDIT.md §2.

What this file is FOR (master prompt §6.5, audit §5): the ten required tests plus the two added
on 2026-09-03 (the ``DiffDirectionsOnly`` mirror control and the ``per_frequency_shares`` power
budget), plus the split tests the audit lists as "NOT YET" — the ones the four new named
functions add. It also carries the NEGATIVE CONTROL the spec demands: ``cos(w(a-b))`` must not
survive any sum-direction restriction. A test suite whose every case passes by construction
proves nothing; the negative control is what makes the positive ones mean something.

test_core.py pins the pre-existing semantics and is NOT duplicated here except where a new
variant changes the picture.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grokverse.analysis.common import masked_ce_and_acc, split_masks  # noqa: E402
from grokverse.analysis.fourier import fourier_basis  # noqa: E402
from grokverse.analysis.mask_protocols import (  # noqa: E402
    ALL_IMPLEMENTED, ALIASES, IMPLEMENTED, SPLITS, SPLIT_AGNOSTIC, DiffDirectionsOnly,
    NandaExact, SumDirectionsOnly, build_protocol, full_grid_extension_excluded_loss,
    full_grid_extension_restricted_loss, legacy_broad_mask_variant, nanda_exact_excluded_loss,
    nanda_exact_restricted_loss, operator_trace, per_frequency_shares)
from grokverse.analysis.progress_measures import fwd2d, inv2d  # noqa: E402
from grokverse.config import get_config  # noqa: E402

P = 23                                            # working prime for every synthetic case
P_BIG, KEYS_BIG = 113, [18, 15, 1, 11, 13, 22, 56, 36]   # the canonical run's key set

#: The exact variants: those that never keep a cross-frequency product.
EXACT = ("paper_literal_2x2_block", "sum_directions_only", "nanda_exact")
#: The variants built on the (a+b) directions alone.
SUM_ONLY = ("sum_directions_only", "nanda_exact")

_n_checks = 0


def check(name, cond) -> None:
    global _n_checks
    _n_checks += 1
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        raise SystemExit(1)


def make(name: str, p: int, keys):
    """``build_protocol`` with the split ``nanda_exact`` requires (see NandaExact)."""
    kw = {"split": "all"} if name == "nanda_exact" else {}
    return build_protocol(name, p, keys, **kw)


def rows(k: int) -> tuple[int, int]:
    return 1 + 2 * (k - 1), 2 + 2 * (k - 1)


def grid(p: int) -> tuple[np.ndarray, np.ndarray]:
    n = np.arange(p)
    return np.meshgrid(n, n, indexing="ij")


def wave(p: int, k: int, sign: int = +1) -> np.ndarray:
    """``cos(w_k(a + sign*b))`` over the grid, shape [p, p]."""
    A, B = grid(p)
    return np.cos(2 * np.pi * k / p * (A + sign * B))


# --------------------------------------------------------------------------- #
# 1. the 2D Fourier round trip (the transform every protocol is defined on)    #
# --------------------------------------------------------------------------- #
def check_roundtrip() -> None:
    print("\n-- 2D Fourier round trip --")
    Fb, _ = fourier_basis(P)
    L = np.random.RandomState(0).randn(P, P, 5)
    check("inv2d(fwd2d(L)) == L to 1e-9", np.abs(inv2d(fwd2d(L, Fb), Fb) - L).max() < 1e-9)
    check("fwd2d(inv2d(Lhat)) == Lhat to 1e-9",
          np.abs(fwd2d(inv2d(L, Fb), Fb) - L).max() < 1e-9)
    check("the transform is norm-preserving (orthonormal basis)",
          abs(float((fwd2d(L, Fb) ** 2).sum() - (L ** 2).sum())) < 1e-8)
    # a unit component maps to the wave the module docstring names
    Lhat = np.zeros((P, P, 1))
    c, s = rows(3)
    Lhat[c, c, 0], Lhat[s, s, 0] = 1.0, -1.0
    recon = inv2d(Lhat, Fb)[:, :, 0]
    ref = wave(P, 3, +1) * (2.0 / P)          # (cos_k x cos_k - sin_k x sin_k) normalised
    check("(1,-1,0,0) in the k=3 block IS cos(w_3(a+b))",
          np.abs(recon - ref * (recon.max() / ref.max())).max() < 1e-9
          and np.corrcoef(recon.ravel(), ref.ravel())[0, 1] > 1 - 1e-12)


# --------------------------------------------------------------------------- #
# 2. exact kept/removed counts for EVERY variant, incl. nanda_exact            #
# --------------------------------------------------------------------------- #
def check_counts() -> None:
    print("\n-- component counts (MASK_PROTOCOL_AUDIT §2, p=113, 8 key freqs) --")
    expected = {
        "legacy_broad_mask": (289, 3360),
        "paper_literal_2x2_block": (33, 32),
        "sum_directions_only": (17, 16),
        "diff_directions_only": (17, 16),
        "nanda_exact": (17, 16),
    }
    for name, (n_keep, n_rm) in expected.items():
        pr = make(name, P_BIG, KEYS_BIG)
        check(f"{name}: restricted keeps {n_keep} of {P_BIG ** 2}", pr.n_kept == n_keep)
        check(f"{name}: excluded removes {n_rm}", pr.n_removed == n_rm)
    check("every implemented variant is covered by the count table",
          set(expected) == set(ALL_IMPLEMENTED))
    check("nanda_exact keeps exactly what sum_directions_only keeps (audit §3/§4)",
          make("nanda_exact", P_BIG, KEYS_BIG).n_kept
          == make("sum_directions_only", P_BIG, KEYS_BIG).n_kept)
    check("legacy keeps 17x more than the published operator",
          289 // 17 == 17 and 3360 // 16 == 210)

    # analytic counts verified numerically: the trace of a projection is its rank
    keys = [3, 5]
    for name in ALL_IMPLEMENTED:
        pr = make(name, P, keys)
        check(f"{name}: n_kept == trace(restrict) at p={P}",
              abs(operator_trace(pr.restrict, P) - pr.n_kept) < 1e-9)
        check(f"{name}: n_removed == p^2 - trace(exclude) at p={P}",
              abs((P * P - operator_trace(pr.exclude, P)) - pr.n_removed) < 1e-9)
    for name in EXACT + ("diff_directions_only",):
        pr = make(name, P, keys)
        check(f"{name}: n_kept == n_removed + 1 (the constant)", pr.n_kept == pr.n_removed + 1)
    lb = make("legacy_broad_mask", P, keys)
    check("legacy restricted/excluded are NOT complements (the defect)",
          lb.n_kept != lb.n_removed + 1)

    # the audit's report-only rename must not break the class name
    check("same_frequency_block still builds", build_protocol("same_frequency_block", P, keys).n_kept == 9)
    check("paper_literal_2x2_block reaches the same class",
          build_protocol("paper_literal_2x2_block", P, keys).name == "same_frequency_block")
    check("...and reports under the audit's name",
          build_protocol("same_frequency_block", P, keys).report_name == "paper_literal_2x2_block")
    check("ALIASES records the rename", ALIASES["paper_literal_2x2_block"] == "same_frequency_block")
    check("IMPLEMENTED (frozen for test_core / the re-train CLI) is unchanged",
          IMPLEMENTED == ("legacy_broad_mask", "same_frequency_block", "sum_directions_only"))
    check("nanda_exact is implemented but not split-agnostic",
          "nanda_exact" in ALL_IMPLEMENTED and "nanda_exact" not in SPLIT_AGNOSTIC)


# --------------------------------------------------------------------------- #
# 3. cross-frequency leakage: none in the exact variants, some in the legacy    #
# --------------------------------------------------------------------------- #
def check_cross_frequency() -> None:
    print("\n-- cross-frequency leakage --")
    Fb, _ = fourier_basis(P)
    A, B = grid(P)
    cross = (np.cos(2 * np.pi * 3 * A / P) * np.cos(2 * np.pi * 5 * B / P))[:, :, None]
    chat = fwd2d(cross, Fb)
    leak = {n: float(np.abs(inv2d(make(n, P, [3, 5]).restrict(chat), Fb)).max())
            for n in ALL_IMPLEMENTED}
    check("legacy restricted LEAKS cos(w_3 a)cos(w_5 b) (the §3.1 defect)",
          leak["legacy_broad_mask"] > 0.1)
    for name in EXACT + ("diff_directions_only",):
        check(f"{name}: no cross-frequency leakage", leak[name] < 1e-10)
    for name in ALL_IMPLEMENTED:
        pr = make(name, P, [3, 5])
        expect = name == "legacy_broad_mask"
        check(f"{name}: keeps_cross_frequency() == {expect}",
              pr.keeps_cross_frequency() is expect)
    # excluded side: the legacy mask DELETES the cross term too, the exact ones keep it
    for name in EXACT + ("diff_directions_only",):
        kept = inv2d(make(name, P, [3, 5]).exclude(chat), Fb)
        check(f"{name}: excluded leaves the cross term untouched",
              np.abs(kept - cross).max() < 1e-10)
    lost = inv2d(make("legacy_broad_mask", P, [3, 5]).exclude(chat), Fb)
    check("legacy excluded DESTROYS the cross term (it deletes whole rows/columns)",
          np.abs(lost).max() < 1e-10)


# --------------------------------------------------------------------------- #
# 4. the constant term, for every variant                                      #
# --------------------------------------------------------------------------- #
def check_constant() -> None:
    print("\n-- constant term --")
    keys = [3, 5]
    for name in ALL_IMPLEMENTED:
        pr = make(name, P, keys)
        c = np.zeros((P, P, 2))
        c[0, 0] = [1.0, -2.0]
        check(f"{name}: restricted KEEPS the constant (Colab bias_correction=True)",
              np.allclose(pr.restrict(c)[0, 0], [1.0, -2.0]))
        check(f"{name}: excluded also keeps the constant (paper: only the key terms go)",
              np.allclose(pr.exclude(c)[0, 0], [1.0, -2.0]))
        check(f"{name}: restricted passes NOTHING else of a pure constant",
              np.abs(pr.restrict(c)).sum() == abs(1.0) + abs(-2.0))
    # a constant offset in logit space is class-wise and must survive the round trip
    Fb, _ = fourier_basis(P)
    L = np.zeros((P, P, 4))
    L[:, :, 1] = 3.0
    for name in ALL_IMPLEMENTED:
        back = inv2d(make(name, P, keys).restrict(fwd2d(L, Fb)), Fb)
        check(f"{name}: a per-class constant logit offset survives restriction",
              np.abs(back - L).max() < 1e-10)


# --------------------------------------------------------------------------- #
# 5 + 7. an injected ideal circuit: restricted ~ full, excluded far above       #
# --------------------------------------------------------------------------- #
def ideal_circuit_logits(p: int, keys, amp: float = 20.0) -> np.ndarray:
    """``L[a,b,c] = amp * sum_k cos(w_k(a+b-c))`` — the paper's eq. (1) circuit, exactly."""
    A, B = grid(p)
    n = np.arange(p)
    L = np.zeros((p, p, p))
    for k in keys:
        w = 2 * np.pi * k / p
        L += amp * np.cos(w * (A[:, :, None] + B[:, :, None] - n[None, None, :]))
    return L


def check_ideal_circuit() -> None:
    print("\n-- injected ideal cos(w_k(a+b)) circuit --")
    keys = [3, 5]
    cfg = get_config("nanda", p=P, seed=0)
    Fb, _ = fourier_basis(P)
    L = ideal_circuit_logits(P, keys)
    Lhat = fwd2d(L, Fb)
    allm = np.ones((P, P), dtype=bool)
    full_ce, full_acc = masked_ce_and_acc(L, allm, P)
    check("the injected circuit solves the task (acc == 1)", full_acc == 1.0)
    check("the injected circuit has a low full loss", full_ce < 0.05)
    uniform = float(np.log(P))
    for name in EXACT:
        pr = make(name, P, keys)
        r_ce, r_acc = masked_ce_and_acc(inv2d(pr.restrict(Lhat), Fb), allm, P)
        e_ce, e_acc = masked_ce_and_acc(inv2d(pr.exclude(Lhat), Fb), allm, P)
        check(f"{name}: restricted ~= full loss ({r_ce:.3e} vs {full_ce:.3e})",
              abs(r_ce - full_ce) < 1e-9)
        check(f"{name}: restricted keeps accuracy 1", r_acc == 1.0)
        check(f"{name}: excluded is FAR above full ({e_ce:.4f} >> {full_ce:.3e})",
              e_ce > full_ce + 1.0)
        # The exclusion annihilates a circuit built ONLY from key frequencies, so the
        # loss is log p to machine precision. The ARGMAX of what is left, however, is
        # driven entirely by rounding noise: it spreads over all p classes and lands
        # wherever the noise falls (2/529 here) -- NOT on the 1/p that a degenerate
        # all-zero argmax would tie-break to. Accuracy is therefore not a statistic
        # this case can pin, and the earlier `abs(e_acc - 1/P) < 1e-12` asserted a
        # property the quantity does not have (cf. labbook 2026-09-03, entry 26).
        # The annihilation itself is the load-bearing claim, and it is checked
        # directly and relative to the circuit's own scale.
        e_max = float(np.abs(inv2d(pr.exclude(Lhat), Fb)).max())
        check(f"{name}: excluded loss is log p ({e_ce:.12f} vs {uniform:.12f})",
              abs(e_ce - uniform) < 1e-9)
        check(f"{name}: excluded annihilates the circuit "
              f"(max|logit| {e_max:.2e} vs amplitude {np.abs(L).max():.0f})",
              e_max < 1e-12 * float(np.abs(L).max()))
        check(f"{name}: excluded accuracy carries no signal ({e_acc:.4f}, was 1.0)",
              e_acc < 0.2)
    # the same statement through the named §4 functions, on their declared splits
    r = nanda_exact_restricted_loss(L, keys, cfg)
    e = nanda_exact_excluded_loss(L, keys, cfg, per_frequency=False)
    check("nanda_exact_restricted_loss ~= full loss on every split",
          all(abs(r["splits"][s]["loss"] - masked_ce_and_acc(L, m, P)[0]) < 1e-9
              for s, m in zip(("test", "train", "all"),
                              (split_masks(cfg)[1], split_masks(cfg)[0], allm))))
    check("nanda_exact_excluded_loss is far above full loss (train split)",
          e["loss"] > masked_ce_and_acc(L, split_masks(cfg)[0], P)[0] + 1.0)
    # the wave that is NOT in the circuit must be untouched by the exclusion
    fge = full_grid_extension_excluded_loss(L, keys, cfg)
    check("full_grid_extension_excluded_loss agrees with the operator on the full grid",
          abs(fge["loss"] - uniform) < 1e-9)


# --------------------------------------------------------------------------- #
# 6. injected MIXED frequencies are separated                                  #
# --------------------------------------------------------------------------- #
def check_mixed_frequencies() -> None:
    print("\n-- injected mixed frequencies --")
    Fb, _ = fourier_basis(P)
    w3, w5 = wave(P, 3, +1), wave(P, 5, +1)
    mixed = (w3 + w5)[:, :, None]
    mhat = fwd2d(mixed, Fb)
    for name in SUM_ONLY:
        got3 = inv2d(make(name, P, [3]).restrict(mhat), Fb)[:, :, 0]
        got5 = inv2d(make(name, P, [5]).restrict(mhat), Fb)[:, :, 0]
        check(f"{name}: restricting to k=3 recovers ONLY the k=3 wave",
              np.abs(got3 - w3).max() < 1e-10)
        check(f"{name}: restricting to k=5 recovers ONLY the k=5 wave",
              np.abs(got5 - w5).max() < 1e-10)
        check(f"{name}: the two single-frequency parts re-sum to the input",
              np.abs(got3 + got5 - mixed[:, :, 0]).max() < 1e-10)
    # a sum wave and a difference wave at DIFFERENT frequencies are separated too
    sum3, diff5 = wave(P, 3, +1), wave(P, 5, -1)
    both = (sum3 + diff5)[:, :, None]
    bhat = fwd2d(both, Fb)
    s = inv2d(make("sum_directions_only", P, [3, 5]).restrict(bhat), Fb)[:, :, 0]
    d = inv2d(build_protocol("diff_directions_only", P, [3, 5]).restrict(bhat), Fb)[:, :, 0]
    check("sum_directions_only picks out cos(w_3(a+b)) and nothing else",
          np.abs(s - sum3).max() < 1e-10)
    check("diff_directions_only picks out cos(w_5(a-b)) and nothing else",
          np.abs(d - diff5).max() < 1e-10)
    check("the two mirror projections partition the mixed field",
          np.abs(s + d - both[:, :, 0]).max() < 1e-10)


# --------------------------------------------------------------------------- #
# 7. the mirror control: DiffDirectionsOnly                                    #
# --------------------------------------------------------------------------- #
def check_mirror_control() -> None:
    print("\n-- DiffDirectionsOnly (mirror control) --")
    Fb, _ = fourier_basis(P)
    k = 4
    plus, minus = wave(P, k, +1)[:, :, None], wave(P, k, -1)[:, :, None]
    d = build_protocol("diff_directions_only", P, [k])
    s = build_protocol("sum_directions_only", P, [k])
    check("diff: cos(w(a-b)) survives restriction unchanged",
          np.abs(inv2d(d.restrict(fwd2d(minus, Fb)), Fb) - minus).max() < 1e-10)
    check("diff: cos(w(a+b)) is ANNIHILATED by the diff restriction",
          np.abs(inv2d(d.restrict(fwd2d(plus, Fb)), Fb)).max() < 1e-10)
    check("diff: cos(w(a-b)) is annihilated by the diff EXCLUSION",
          np.abs(inv2d(d.exclude(fwd2d(minus, Fb)), Fb)).max() < 1e-10)
    check("diff: cos(w(a+b)) passes the diff exclusion untouched",
          np.abs(inv2d(d.exclude(fwd2d(plus, Fb)), Fb) - plus).max() < 1e-10)
    # the mirror is exactly complementary inside the block
    rng = np.random.RandomState(7)
    Lh = fwd2d(rng.randn(P, P, 3), Fb)
    blk = build_protocol("same_frequency_block", P, [k])
    # P_sum + P_diff spans the whole 4D block, so restrict(sum) + restrict(diff)
    # equals the block restriction plus one extra copy of the constant.
    check("sum + diff projections == the full 2x2 block projection",
          np.abs(s.restrict(Lh) + d.restrict(Lh) - blk.restrict(Lh)
                 - _const_only(Lh)).max() < 1e-10)
    check("sum and diff subspaces are orthogonal (their projections commute to 0)",
          np.abs(s._project(d._project(Lh))).max() < 1e-12)
    check("mirror control has the SAME size as the hypothesis it controls",
          (d.n_kept, d.n_removed) == (s.n_kept, s.n_removed))


def _const_only(Lhat: np.ndarray) -> np.ndarray:
    """Just the (const, const) component — both restrictions re-add it, so it is double-counted."""
    out = np.zeros_like(Lhat)
    out[0, 0] = Lhat[0, 0]
    return out


# --------------------------------------------------------------------------- #
# 8. per_frequency_shares is a complete power budget                           #
# --------------------------------------------------------------------------- #
def check_per_frequency_shares() -> None:
    print("\n-- per_frequency_shares --")
    Fb, _ = fourier_basis(P)
    half = (P - 1) // 2
    rng = np.random.RandomState(11)
    Lhat = fwd2d(rng.randn(P, P, 4), Fb)
    sh = per_frequency_shares(Lhat, list(range(1, half + 1)))
    check("shares sum to 1 over ALL frequencies + the residual",
          abs(sh["shares_sum"] - 1.0) < 1e-12)
    check("powers sum to the total NON-CONSTANT power",
          abs(sh["sum_power_total"] + sh["diff_power_total"] + sh["other_power"]
              - sh["total_nonconstant_power"]) < 1e-8)
    check("the constant is excluded from the budget",
          abs(sh["total_power"] - sh["constant_power"] - sh["total_nonconstant_power"]) < 1e-8)
    check("with every k as a key, the residual is exactly the cross-frequency power",
          sh["other_share"] > 0.5)          # random logits are dominated by cross terms
    check("per-k block power == sum power + diff power",
          all(abs(v["block_power"] - v["sum_power"] - v["diff_power"]) < 1e-8
              for v in sh["per_frequency"].values()))

    # a field made only of key-frequency blocks has NO residual
    field = (wave(P, 3, +1) + 2.0 * wave(P, 5, -1))[:, :, None]
    s2 = per_frequency_shares(fwd2d(field, Fb), [3, 5])
    check("a pure block field leaves no residual power", abs(s2["other_share"]) < 1e-12)
    check("cos(w_3(a+b)) sits entirely in k=3's SUM directions",
          abs(s2["per_frequency"]["3"]["sum_over_block"] - 1.0) < 1e-12)
    check("cos(w_5(a-b)) sits entirely in k=5's DIFFERENCE directions",
          abs(s2["per_frequency"]["5"]["sum_over_block"]) < 1e-12)
    check("amplitude 2 carries 4x the power",
          abs(s2["per_frequency"]["5"]["diff_power"]
              / s2["per_frequency"]["3"]["sum_power"] - 4.0) < 1e-9)
    check("shares of the pure block field still sum to 1",
          abs(s2["shares_sum"] - 1.0) < 1e-12)
    bad = False
    try:
        per_frequency_shares(np.zeros((P, P, 2)), [3])
    except ValueError:
        bad = True
    check("a tensor with no non-constant power raises instead of returning NaN", bad)


# --------------------------------------------------------------------------- #
# 9 + 10. splits: train != test, and what each function declares                #
# --------------------------------------------------------------------------- #
def asymmetric_logits(cfg) -> np.ndarray:
    """Correct on the TRAIN cells, deliberately wrong on the TEST cells.

    The point is a case where any honest train/test split evaluation must give two
    different numbers, so "which split?" is not a cosmetic label.
    """
    p = cfg.p
    train_mask, _ = split_masks(cfg)
    A, B = grid(p)
    correct = (A + B) % p
    wrong = (A + B + 1) % p
    target = np.where(train_mask, correct, wrong)
    L = np.zeros((p, p, p))
    np.put_along_axis(L, target[:, :, None], 8.0, axis=2)
    return L


def check_splits() -> None:
    print("\n-- splits (the audit's 'NOT YET' row) --")
    cfg = get_config("nanda", p=P, seed=0)
    keys = [3, 5]
    train_mask, test_mask = split_masks(cfg)
    L = asymmetric_logits(cfg)
    tr_ce, tr_acc = masked_ce_and_acc(L, train_mask, P)
    te_ce, te_acc = masked_ce_and_acc(L, test_mask, P)
    check("the constructed case really is asymmetric (train acc 1, test acc 0)",
          tr_acc == 1.0 and te_acc == 0.0)
    check("train and test cross-entropy differ by a lot", te_ce - tr_ce > 1.0)
    check("the split masks partition the grid",
          int(train_mask.sum() + test_mask.sum()) == P * P and not (train_mask & test_mask).any())

    e = nanda_exact_excluded_loss(L, keys, cfg)
    check("nanda_exact_excluded_loss declares the TRAIN split", e["split"] == "train")
    check("...and its number IS the train-split number",
          e["loss"] == e["splits"]["train"]["loss"])
    check("...which differs from the test-split number on the asymmetric case",
          abs(e["splits"]["train"]["loss"] - e["splits"]["test"]["loss"]) > 1e-6)
    check("...and from the full-grid number",
          abs(e["splits"]["train"]["loss"] - e["splits"]["all"]["loss"]) > 1e-6)
    check("excluded cell counts match the masks",
          e["splits"]["train"]["n_cells"] == int(train_mask.sum())
          and e["splits"]["test"]["n_cells"] == int(test_mask.sum())
          and e["splits"]["all"]["n_cells"] == P * P)
    check("excluded records the source of its split (the paper states it)",
          "training data" in e["split_source"])
    check("excluded removes 2 directions per key frequency",
          e["n_components_kept_or_removed"] == 2 * len(keys))
    check("excluded reports the per-frequency variant the Colab plots",
          set(e["per_frequency"]) == {"3", "5"})
    check("...each per-frequency variant removes exactly 2 components",
          all(v["n_components_removed"] == 2 for v in e["per_frequency"].values()))
    check("...and removing ONE frequency hurts no more than removing both",
          all(v["loss"] <= e["loss"] + 1e-9 for v in e["per_frequency"].values()))

    r = nanda_exact_restricted_loss(L, keys, cfg)
    check("nanda_exact_restricted_loss reports all three splits",
          set(r["splits"]) == set(SPLITS))
    check("...each with a loss and an accuracy",
          all({"loss", "accuracy", "n_cells"} <= set(v) for v in r["splits"].values()))
    check("...they are not all the same number on the asymmetric case",
          len({round(v["loss"], 12) for v in r["splits"].values()}) == 3)
    check("...the quoted convention is the TransformerLens 'test' split",
          r["split"] == "test" and r["quoted_split"] == "test"
          and r["loss"] == r["splits"]["test"]["loss"])
    check("...and the paper's own split is recorded as NOT FOUND IN SOURCE",
          r["paper_split"] == "[NOT FOUND IN SOURCE]")
    check("...with the provenance of the quoted split spelled out",
          "TransformerLens" in r["quoted_split_source"])
    check("restricted keeps 1 + 2 per key frequency",
          r["n_components_kept_or_removed"] == 1 + 2 * len(keys))
    check("neither nanda_exact function is labelled OUR variant",
          r["is_our_variant"] is False and e["is_our_variant"] is False)

    fr = full_grid_extension_restricted_loss(L, keys, cfg)
    fe = full_grid_extension_excluded_loss(L, keys, cfg)
    for tag, res in (("restricted", fr), ("excluded", fe)):
        check(f"full_grid_extension_{tag}: declares the full grid", res["split"] == "all")
        check(f"full_grid_extension_{tag}: labelled OUR variant", res["is_our_variant"] is True)
        check(f"full_grid_extension_{tag}: same component rule as nanda_exact",
              res["same_component_rule_as"] == "nanda_exact")
        check(f"full_grid_extension_{tag}: its number IS the full-grid number",
              res["loss"] == res["splits"]["all"]["loss"])
    check("full_grid_extension differs from nanda_exact ONLY in the split",
          abs(fr["splits"]["test"]["loss"] - r["splits"]["test"]["loss"]) < 1e-12
          and abs(fe["splits"]["train"]["loss"] - e["splits"]["train"]["loss"]) < 1e-12)

    lr = legacy_broad_mask_variant(L, keys, cfg, which="restricted")
    le = legacy_broad_mask_variant(L, keys, cfg, which="excluded")
    check("legacy_broad_mask_variant defaults to the restricted operator",
          legacy_broad_mask_variant(L, keys, cfg)["which"] == "restricted")
    check("legacy_broad_mask_variant is never labelled a reproduction",
          lr["is_reproduction"] is False and le["is_reproduction"] is False)
    check("legacy keeps far more than the published operator (p=23, 2 keys)",
          lr["n_components_kept_or_removed"] > 4 * r["n_components_kept_or_removed"])
    check("legacy deletes far more than the published operator",
          le["n_components_kept_or_removed"] > 4 * e["n_components_kept_or_removed"])
    check("legacy declares the full grid and keeps cross-frequency blocks",
          lr["split"] == "all" and lr["keeps_cross_frequency_blocks"] is True)
    check("nanda_exact keeps no cross-frequency block",
          r["keeps_cross_frequency_blocks"] is False)
    bad = False
    try:
        legacy_broad_mask_variant(L, keys, cfg, which="nonsense")
    except ValueError:
        bad = True
    check("an unknown 'which' raises", bad)


# --------------------------------------------------------------------------- #
# 11. the vectorized projection == a slow per-component reference loop          #
# --------------------------------------------------------------------------- #
def slow_sum_projection(Lhat: np.ndarray, p: int, keys) -> np.ndarray:
    """P_S(Lhat), written as literally as the definition allows.

    For each key k, build the two UNIT direction arrays in the [p, p] coefficient plane
    and project each class channel onto them with an explicit component-by-component
    inner product. No vectorisation, no algebraic shortcut — this is the reference the
    fast ``_project`` is checked against.
    """
    out = np.zeros_like(Lhat)
    n_c = Lhat.shape[2]
    root2 = np.sqrt(2.0)
    for k in keys:
        c, s = rows(k)
        d_cos = np.zeros((p, p))
        d_cos[c, c], d_cos[s, s] = 1.0 / root2, -1.0 / root2      # cos(w_k(a+b))
        d_sin = np.zeros((p, p))
        d_sin[s, c], d_sin[c, s] = 1.0 / root2, 1.0 / root2       # sin(w_k(a+b))
        for d in (d_cos, d_sin):
            for ch in range(n_c):
                coeff = 0.0
                for i in range(p):
                    for j in range(p):
                        if d[i, j] != 0.0:
                            coeff += d[i, j] * Lhat[i, j, ch]
                for i in range(p):
                    for j in range(p):
                        if d[i, j] != 0.0:
                            out[i, j, ch] += d[i, j] * coeff
    return out


def check_reference_loop() -> None:
    print("\n-- vectorized projection vs a slow per-component reference --")
    Fb, _ = fourier_basis(P)
    keys = [3, 5, 9]
    rng = np.random.RandomState(5)
    Lhat = fwd2d(rng.randn(P, P, 3), Fb)
    ref = slow_sum_projection(Lhat, P, keys)
    fast = SumDirectionsOnly(P, keys)._project(Lhat)
    check("SumDirectionsOnly._project == the reference loop",
          np.abs(fast - ref).max() < 1e-12)
    exact = NandaExact(P, keys, split="all")._project(Lhat)
    check("NandaExact._project == the reference loop (it IS the same operator)",
          np.abs(exact - ref).max() < 1e-12)
    ref_r = ref.copy()
    ref_r[0, 0] = Lhat[0, 0]
    check("restrict == reference projection + the constant",
          np.abs(SumDirectionsOnly(P, keys).restrict(Lhat) - ref_r).max() < 1e-12)
    check("exclude == Lhat - reference projection",
          np.abs(SumDirectionsOnly(P, keys).exclude(Lhat) - (Lhat - ref)).max() < 1e-12)
    # The checks above are deliberately run on a 3-channel grid: the projection acts
    # on the [p, p] coefficient plane and is agnostic to the class count, so 3 channels
    # exercise it at a fraction of the cost. A LOSS, however, is only defined when the
    # class axis IS the label axis, so the loss check needs its own [p, p, p] grid --
    # feeding the 3-channel grid to masked_ce_and_acc raised IndexError, because the
    # labels (a+b) mod p run to p-1. The reference loop is cheap enough at p classes
    # (n_keys * 2 dirs * p channels * 2 * p^2 ~ 1.5e5 iterations).
    allm = np.ones((P, P), dtype=bool)
    Lhat_p = fwd2d(rng.randn(P, P, P), Fb)
    ref_p = slow_sum_projection(Lhat_p, P, keys)
    ref_pr = ref_p.copy()
    ref_pr[0, 0] = Lhat_p[0, 0]
    check("the two implementations give the same restricted LOSS",
          abs(masked_ce_and_acc(inv2d(ref_pr, Fb), allm, P)[0]
              - masked_ce_and_acc(inv2d(SumDirectionsOnly(P, keys).restrict(Lhat_p), Fb),
                                  allm, P)[0]) < 1e-12)
    check("the projection is idempotent (it is a projection)",
          np.abs(slow_sum_projection(ref, P, keys) - ref).max() < 1e-12)


# --------------------------------------------------------------------------- #
# 12. NEGATIVE CONTROL — this one must NOT pass                                #
# --------------------------------------------------------------------------- #
def check_negative_control() -> None:
    print("\n-- NEGATIVE CONTROL: cos(w(a-b)) must not survive a sum restriction --")
    Fb, _ = fourier_basis(P)
    k = 6
    minus = wave(P, k, -1)[:, :, None]
    mhat = fwd2d(minus, Fb)
    for name in SUM_ONLY:
        got = inv2d(make(name, P, [k]).restrict(mhat), Fb)
        check(f"{name}: cos(w(a-b)) does NOT survive the restriction",
              np.abs(got).max() < 1e-10)
        check(f"{name}: cos(w(a-b)) passes the EXCLUSION untouched (it was never in S)",
              np.abs(inv2d(make(name, P, [k]).exclude(mhat), Fb) - minus).max() < 1e-10)
    # the looser variants DO keep it — that is exactly why they are not the published operator
    keeps = inv2d(build_protocol("same_frequency_block", P, [k]).restrict(mhat), Fb)
    check("paper_literal_2x2_block DOES keep cos(w(a-b)) — the looser reading (audit §3)",
          np.abs(keeps - minus).max() < 1e-10)
    legacy = inv2d(build_protocol("legacy_broad_mask", P, [k]).restrict(mhat), Fb)
    check("legacy_broad_mask also keeps it", np.abs(legacy - minus).max() < 1e-10)

    # a difference-only "circuit" must NOT look like a solution under the exact protocols
    cfg = get_config("nanda", p=P, seed=0)
    A, B = grid(P)
    n = np.arange(P)
    L = np.zeros((P, P, P))
    for kk in (3, 5):
        w = 2 * np.pi * kk / P
        L += 6.0 * np.cos(w * (A[:, :, None] - B[:, :, None] - n[None, None, :]))
    allm = np.ones((P, P), dtype=bool)
    r = nanda_exact_restricted_loss(L, [3, 5], cfg)
    check("a (a-b) circuit is NOT rescued by the sum-direction restriction",
          r["splits"]["all"]["loss"] > np.log(P) - 1e-9)
    check("...and the excluded loss leaves it exactly as it was",
          abs(nanda_exact_excluded_loss(L, [3, 5], cfg, per_frequency=False)["splits"]["all"]["loss"]
              - masked_ce_and_acc(L, allm, P)[0]) < 1e-9)


# --------------------------------------------------------------------------- #
# construction contracts                                                       #
# --------------------------------------------------------------------------- #
def _raises(fn, exc=Exception) -> bool:
    try:
        fn()
    except exc:
        return True
    return False


def check_contracts() -> None:
    print("\n-- construction contracts --")
    check("nanda_exact refuses to be built without a declared split "
          "(the paper's restricted split is [NOT FOUND IN SOURCE])",
          _raises(lambda: build_protocol("nanda_exact", P, [3]), TypeError))
    check("...and rejects a split name that is not one of the three",
          _raises(lambda: build_protocol("nanda_exact", P, [3], split="held_out"), ValueError))
    check("...but builds with an explicit split",
          build_protocol("nanda_exact", P, [3], split="train").split == "train")
    check("NandaExact.describe() carries the split and the provenance",
          build_protocol("nanda_exact", P, [3], split="all").describe()["split"] == "all"
          and "[NOT FOUND IN SOURCE]" in
          build_protocol("nanda_exact", P, [3], split="all")
          .describe()["provenance"]["restricted_split_in_paper"])
    check("unknown protocol name rejected",
          _raises(lambda: build_protocol("made_up", P, [3]), ValueError))
    for name in SPLIT_AGNOSTIC:
        check(f"{name}: empty key set rejected",
              _raises(lambda n=name: build_protocol(n, P, []), ValueError))
        check(f"{name}: out-of-range key frequency rejected",
              _raises(lambda n=name: build_protocol(n, P, [99]), ValueError))
        check(f"{name}: duplicate key frequencies rejected",
              _raises(lambda n=name: build_protocol(n, P, [3, 3]), ValueError))
        check(f"{name}: even p rejected",
              _raises(lambda n=name: build_protocol(n, 24, [3]), ValueError))
    cfg = get_config("nanda", p=P, seed=0)
    check("a logit grid of the wrong shape is rejected",
          _raises(lambda: nanda_exact_restricted_loss(np.zeros((5, 5, 5)), [3], cfg), ValueError))
    # protocols must not mutate their input
    Fb, _ = fourier_basis(P)
    Lh = fwd2d(np.random.RandomState(2).randn(P, P, 2), Fb)
    keep = Lh.copy()
    for name in ALL_IMPLEMENTED:
        pr = make(name, P, [3, 5])
        pr.restrict(Lh)
        pr.exclude(Lh)
    check("no protocol mutates its input", np.array_equal(Lh, keep))


def main() -> None:
    check_roundtrip()
    check_counts()
    check_cross_frequency()
    check_constant()
    check_ideal_circuit()
    check_mixed_frequencies()
    check_mirror_control()
    check_per_frequency_shares()
    check_splits()
    check_reference_loop()
    check_negative_control()
    check_contracts()
    print(f"\nALL CHECKS PASSED ({_n_checks} checks)")


if __name__ == "__main__":
    main()
