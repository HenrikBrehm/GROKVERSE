"""Checks for analysis/function_agreement.py v1.1 (INTERFACES.md section 8).

Run from training/:  python tests/test_function_agreement.py
Same check() convention as test_core.py; exits non-zero on the first failure.

Synthetic inputs with known answers (p = 23): identical models, models differing on a
known set of exactly 10 cells, constructed error structures, and controls that must NOT
look like agreement (cell-shuffled, class-shifted, and two independently seeded models).
The end-to-end part writes tiny run directories -- legacy ``model_final.pt`` and
run-format-v2 checkpoints -- to the scratchpad and exercises ``load_side``,
``compare_sides``, ``compare``, ``compare_all_checkpoints``, ``baseline`` and the CLI,
including the JSON/npz outputs.

Conventions this file pins (module v1.1)
----------------------------------------
* Undefined ratios (Jaccard of two empty error sets, symmetric share with no error,
  entropy of an empty histogram) are reported as ``None`` -- "written as null with their
  counts next to them, never as a convention value" (module docstring). Every such case is
  asserted to be None AND asserted not to be a silent 0.
* ``pearson`` RAISES on a zero-variance input instead of returning a value.
* Split handling: the per-split blocks are always computed on each model's OWN split, so
  they coincide when the two runs share a split and differ when they do not. The split
  hashes and their equality are always recorded; ``require_same_split=True`` (the default,
  INTERFACES section 8 "assert and record") additionally raises on a mismatch, while
  ``require_same_split=False`` (CLI ``--allow-different-splits``, and what ``baseline``
  uses) records the mismatch instead of raising.
* ``baseline`` groups runs by ``config_key`` -- the ``run_id`` with its ``_seed{N}`` token
  removed -- not by architecture.
"""
from __future__ import annotations

import contextlib
import io
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grokverse.analysis.function_agreement import (  # noqa: E402
    BASELINE_IGNORED_FIELDS, BASELINE_METRICS, HISTOGRAM_NAMES, MODULE_VERSION,
    agreement_block, baseline, common_checkpoint_steps, compare, compare_all_checkpoints,
    compare_logits, compare_sides, config_key, correct_class_margin, disagreement_pairs,
    error_histograms, error_structure, headline, histogram_summary, load_side,
    logit_agreement, main, pearson, results_dir, symmetric_error_share,
    symmetric_share_control, target_grid, top2_classes)
from grokverse.checkpoints import checkpoint_path  # noqa: E402
from grokverse.config import get_config  # noqa: E402
from grokverse.models import build_model  # noqa: E402
from grokverse.seed import set_seed  # noqa: E402

# Short directory names on purpose: the checkpoint-sweep outputs are named
# "<run_id>_step{N:06d}__vs__<run_id>_step{N:06d}.npz" (~110 characters), and the
# scratchpad prefix alone is ~125, so a longer test directory would run into the
# Windows 260-character MAX_PATH limit. The real output directory
# (training/results/function_agreement) is short, so this only concerns the tests.
SCRATCH = Path("C:/Users/henri/AppData/Local/Temp/claude/C--Users-henri-Documents-Brain-bwki/"
               "19fe38fd-ad0c-4eae-bfeb-002bb125b3f5/scratchpad") / "fa_tests"
P = 23
N_CONTROL = 20


def check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        raise SystemExit(1)


def _raises(fn) -> bool:
    """True if ``fn()`` refuses its input. ``SystemExit`` is included on purpose: an
    argparse usage error is how the CLI refuses, and it is not an ``Exception``."""
    try:
        fn()
    except (Exception, SystemExit):
        return True
    return False


def _raises_with(fn, needle: str) -> bool:
    try:
        fn()
    except (Exception, SystemExit) as exc:
        return needle in str(exc)
    return False


def _cli(argv: list[str]) -> str:
    """Run the module CLI, returning what it printed (kept out of the PASS/FAIL log)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main(argv)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# synthetic models                                                             #
# --------------------------------------------------------------------------- #
def perfect_logits(p: int, task: str = "add", scale: float = 10.0, seed: int = 0) -> np.ndarray:
    """``scale * onehot(target) + uniform(0, 1)`` noise: the argmax is the target everywhere."""
    rng = np.random.default_rng(seed)
    y = target_grid(p, task)
    L = rng.uniform(0.0, 1.0, (p, p, p))
    ai, bi = np.indices((p, p))
    L[ai, bi, y] += scale
    return L


def force_wrong(L: np.ndarray, cells, p: int, task: str = "add") -> np.ndarray:
    """Copy of ``L`` where every ``(a, b)`` in ``cells`` predicts ``(target + 1) mod p``."""
    out = L.copy()
    y = target_grid(p, task)
    for a, b in cells:
        out[a, b, (y[a, b] + 1) % p] = out[a, b].max() + 5.0
    return out


def random_split(p: int, frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train = np.zeros(p * p, dtype=bool)
    train[rng.choice(p * p, size=int(round(frac * p * p)), replace=False)] = True
    train = train.reshape(p, p)
    return train, ~train


def preds_and_correct(L: np.ndarray, p: int):
    pred = L.argmax(-1)
    return pred, pred == target_grid(p, "add")


# --------------------------------------------------------------------------- #
def check_identical_models():
    print("-- identical models --")
    err_cells = [(0, 1), (5, 7), (12, 3), (22, 22), (9, 15)]
    L = force_wrong(perfect_logits(P), err_cells, P)
    splits = random_split(P, 0.3, seed=1)
    res, arrs = compare_logits(L, L.copy(), P, "add", splits, splits, N_CONTROL, 0)
    fg = res["full_grid"]
    acc = 1 - len(err_cells) / P ** 2
    check("identical: agreement_rate == 1", fg["agreement_rate"] == 1.0)
    check("identical: error_jaccard == 1 (5 shared errors)",
          fg["error_jaccard"] == 1.0 and fg["n_errors_union"] == 5 and fg["n_errors_intersection"] == 5)
    check("identical: both_correct == accuracy_a == accuracy_b",
          fg["both_correct"] == fg["accuracy_a"] == fg["accuracy_b"] and abs(fg["accuracy_a"] - acc) < 1e-12)
    check("identical: only_a == only_b == 0, both_wrong == 5/p^2",
          fg["only_a_correct"] == 0.0 and fg["only_b_correct"] == 0.0
          and abs(fg["both_wrong"] - 5 / P ** 2) < 1e-12)
    check("contingency fractions sum to 1",
          abs(fg["both_correct"] + fg["only_a_correct"] + fg["only_b_correct"] + fg["both_wrong"] - 1) < 1e-12)
    check("identical: no disagreement pairs",
          fg["n_disagreements"] == 0 and arrs["disagreement_pairs"].shape == (0, 2))
    lg = res["logits"]
    check("identical: logit_pearson == margin_pearson == 1",
          abs(lg["logit_pearson"] - 1) < 1e-12 and abs(lg["margin_pearson"] - 1) < 1e-12)
    check("identical: top-2 agreement (set and ordered) == 1",
          lg["top2_agreement"] == 1.0 and lg["top2_ordered_agreement"] == 1.0)
    tr, te = res["model_a_split"]["train"], res["model_a_split"]["test"]
    check("per-split blocks partition the grid", tr["n_cells"] + te["n_cells"] == P ** 2)
    check("per-split agreement == 1 on both splits", tr["agreement_rate"] == 1.0 and te["agreement_rate"] == 1.0)
    check("shared split: the per-split numbers are reported once (model_b_split == model_a_split)",
          res["model_b_split"] == res["model_a_split"])
    check("per-split errors add up to the full-grid count",
          tr["n_errors_a"] + te["n_errors_a"] == fg["n_errors_a"] == 5)
    es = res["error_structure"]
    check("identical: every error histogram of model a sums to its 5 errors",
          set(es["model_a"]["histograms"]) == set(HISTOGRAM_NAMES)
          and all(es["model_a"]["histograms"][k]["total"] == 5 for k in HISTOGRAM_NAMES))
    check("identical: the disagreement mask is empty and its histograms sum to 0",
          es["disagreement"]["n_errors"] == 0
          and all(es["disagreement"]["histograms"][k]["total"] == 0 for k in HISTOGRAM_NAMES))

    L0 = perfect_logits(P)
    res0, _ = compare_logits(L0, L0, P, "add", splits, splits, N_CONTROL, 0)
    fg0 = res0["full_grid"]
    check("identical, zero errors: jaccard of two empty error sets is None (0/0), not a silent 0",
          fg0["error_jaccard"] is None and fg0["n_errors_union"] == 0 and fg0["accuracy_a"] == 1.0)
    check("identical, zero errors: symmetric shares are None too, with n_errors 0 next to them",
          res0["error_structure"]["model_a"]["symmetric_error_share"] is None
          and res0["error_structure"]["model_a"]["symmetric_error_share_offdiag"] is None
          and res0["error_structure"]["model_a"]["n_errors"] == 0)


def check_known_disagreement():
    print("-- models differing on a known set of 10 inputs --")
    rng = np.random.default_rng(3)
    flat = rng.choice(P * P, size=10, replace=False)
    S = sorted((int(i // P), int(i % P)) for i in flat)
    L_a = perfect_logits(P)
    L_b = force_wrong(L_a, S, P)
    splits_a = random_split(P, 0.3, seed=1)
    splits_b = random_split(P, 0.3, seed=2)
    res, arrs = compare_logits(L_a, L_b, P, "add", splits_a, splits_b, N_CONTROL, 0)
    fg = res["full_grid"]
    check("agreement == 1 - 10/p^2", abs(fg["agreement_rate"] - (1 - 10 / P ** 2)) < 1e-12)
    got = [tuple(int(v) for v in row) for row in arrs["disagreement_pairs"]]
    check("disagreement list equals the known set exactly", got == S and len(S) == 10)
    check("disagreement_pairs() matches the npz array",
          np.array_equal(disagreement_pairs(L_a.argmax(-1), L_b.argmax(-1)), arrs["disagreement_pairs"]))
    check("only_a_correct == 10/p^2, only_b == both_wrong == 0",
          abs(fg["only_a_correct"] - 10 / P ** 2) < 1e-12
          and fg["only_b_correct"] == 0.0 and fg["both_wrong"] == 0.0)
    check("error_jaccard == 0 (a has no errors, b has 10) -- a DEFINED zero, union 10",
          fg["error_jaccard"] == 0.0 and fg["n_errors_a"] == 0 and fg["n_errors_b"] == 10
          and fg["n_errors_union"] == 10)
    es = res["error_structure"]
    check("every error histogram of model b sums to its 10 errors",
          all(es["model_b"]["histograms"][k]["total"] == 10 for k in HISTOGRAM_NAMES))
    check("every npz histogram sums to the error count too",
          all(int(arrs[f"model_b_errors_{k}"].sum()) == 10 for k in HISTOGRAM_NAMES))
    check("every disagreement histogram sums to the number of disagreements",
          es["disagreement"]["n_errors"] == 10
          and all(es["disagreement"]["histograms"][k]["total"] == 10 for k in HISTOGRAM_NAMES))
    # per-split numbers are reported on EACH model's own split when the splits differ
    in_train_a = sum(splits_a[0][a, b] for a, b in S)
    in_train_b = sum(splits_b[0][a, b] for a, b in S)
    tr_a, tr_b = res["model_a_split"]["train"], res["model_b_split"]["train"]
    check("model_a_split train: only_a_correct == |S & train_a| / n_train",
          abs(tr_a["only_a_correct"] - in_train_a / tr_a["n_cells"]) < 1e-12)
    check("model_b_split train: only_a_correct == |S & train_b| / n_train",
          abs(tr_b["only_a_correct"] - in_train_b / tr_b["n_cells"]) < 1e-12)
    check("different splits give different per-split blocks", in_train_a != in_train_b
          or res["model_a_split"] != res["model_b_split"])
    check("logit_pearson stays high when only 10 cells changed", res["logits"]["logit_pearson"] > 0.95)


def check_scrambled_controls():
    print("-- scrambled controls (must NOT look like agreement) --")
    L_a = perfect_logits(P)
    rng = np.random.default_rng(7)
    perm = rng.permutation(P * P)
    L_perm = L_a.reshape(P * P, P)[perm].reshape(P, P, P)      # cells shuffled
    splits = random_split(P, 0.3, seed=1)
    res, _ = compare_logits(L_a, L_perm, P, "add", splits, splits, N_CONTROL, 0)
    fg, lg = res["full_grid"], res["logits"]
    check("cell-shuffled control: agreement far below 1", fg["agreement_rate"] < 0.2)
    check("cell-shuffled control: accuracy_b near chance", fg["accuracy_b"] < 0.2)
    check("cell-shuffled control: logit_pearson near 0", abs(lg["logit_pearson"]) < 0.2)
    check("cell-shuffled control: top-2 agreement far below 1", lg["top2_agreement"] < 0.2)
    check("cell-shuffled control: error_jaccard == 0 (a has no errors)", fg["error_jaccard"] == 0.0)

    L_shift = np.roll(L_a, 1, axis=-1)                            # every class shifted by one
    res2, _ = compare_logits(L_a, L_shift, P, "add", splits, splits, N_CONTROL, 0)
    check("class-shifted control: agreement == 0 and accuracy_b == 0",
          res2["full_grid"]["agreement_rate"] == 0.0 and res2["full_grid"]["accuracy_b"] == 0.0)
    check("class-shifted control: only_a_correct == 1", res2["full_grid"]["only_a_correct"] == 1.0)


def check_negative_control():
    print("-- NEGATIVE CONTROL: two independently seeded models must NOT agree --")
    # Two models drawn from independent seeds share no structure: their argmax agreement
    # must sit at chance (1/p), NOT near 1. This check fails the moment the comparison
    # starts reporting agreement for unrelated models.
    L_a = np.random.default_rng(101).normal(size=(P, P, P))
    L_b = np.random.default_rng(202).normal(size=(P, P, P))
    splits = random_split(P, 0.3, seed=1)
    res, _ = compare_logits(L_a, L_b, P, "add", splits, splits, N_CONTROL, 0)
    fg, lg = res["full_grid"], res["logits"]
    check("independent seeds: agreement_rate NOT near 1 (< 0.2)", fg["agreement_rate"] < 0.2)
    check("independent seeds: agreement_rate sits at chance 1/p",
          abs(fg["agreement_rate"] - 1 / P) < 0.05)
    check("independent seeds: logit_pearson near 0", abs(lg["logit_pearson"]) < 0.1)
    check("independent seeds: top-2 set agreement far below 1", lg["top2_agreement"] < 0.2)
    check("independent seeds: per-split agreement is at chance on both splits too",
          res["model_a_split"]["train"]["agreement_rate"] < 0.2
          and res["model_a_split"]["test"]["agreement_rate"] < 0.2)
    # forced inequality: the identical-model case DOES reach 1.0, so the control discriminates
    same, _ = compare_logits(L_a, L_a.copy(), P, "add", splits, splits, N_CONTROL, 0)
    check("the negative control discriminates (identical logits reach 1.0, noise does not)",
          same["full_grid"]["agreement_rate"] == 1.0
          and same["full_grid"]["agreement_rate"] > fg["agreement_rate"] + 0.5)


def check_error_structure():
    print("-- error structure --")
    rng = np.random.default_rng(11)
    E = np.zeros(P * P, dtype=bool)
    E[rng.choice(P * P, size=37, replace=False)] = True
    E = E.reshape(P, P)
    hists = error_histograms(E, P)
    check("every histogram the module produces has p bins and sums to 37",
          set(hists) == set(HISTOGRAM_NAMES)
          and all(hists[k].shape == (P,) and int(hists[k].sum()) == 37 for k in HISTOGRAM_NAMES))
    a, b = np.nonzero(E)
    r0 = int(((a + b) % P == 4).sum())
    check("by_residue_sum bin 4 matches a hand count", int(hists["by_residue_sum"][4]) == r0)
    check("by_target_class == by_residue_sum for task 'add'",
          np.array_equal(hists["by_target_class"], hists["by_residue_sum"]))
    check("by_abs_diff bin 0 counts the diagonal errors",
          int(hists["by_abs_diff"][0]) == int(np.diag(E).sum()))
    hs = histogram_summary(hists["by_a"])
    check("histogram summary: total 37, entropy in (0, 1]",
          hs["total"] == 37 and 0.0 < hs["normalized_entropy"] <= 1.0)
    check("histogram summary of an empty histogram: entropy None (not 0), total 0",
          histogram_summary(np.zeros(P, int))["normalized_entropy"] is None
          and histogram_summary(np.zeros(P, int))["total"] == 0)

    # error_structure(): the JSON summary and the npz arrays agree with each other
    summary, arrays = error_structure(E, P, "add", N_CONTROL, 0)
    check("error_structure: summary n_errors and every histogram total == 37",
          summary["n_errors"] == 37
          and all(summary["histograms"][k]["total"] == 37 for k in HISTOGRAM_NAMES))
    check("error_structure: npz arrays hold the same histograms + the control draws",
          all(np.array_equal(arrays[f"errors_{k}"], hists[k]) for k in HISTOGRAM_NAMES)
          and arrays["symmetric_control_values"].shape == (N_CONTROL,))

    # symmetric errors by construction: (a, b) AND (b, a) wrong for 6 off-diagonal pairs
    pairs = [(1, 2), (3, 9), (4, 20), (7, 8), (10, 15), (0, 22)]
    S = np.zeros((P, P), dtype=bool)
    for x, y in pairs:
        S[x, y] = S[y, x] = True
    ss = symmetric_error_share(S)
    check("constructed symmetric errors: share == 1 (incl. and excl. diagonal), n_errors 12",
          ss["symmetric_error_share"] == 1.0 and ss["symmetric_error_share_offdiag"] == 1.0
          and ss["n_errors"] == 12 and ss["n_symmetric"] == 12)
    # hand-constructed mixed case: 4 wrong cells, exactly 2 of them mirrored -> share 1/2
    M = np.zeros((P, P), dtype=bool)
    M[1, 2] = M[2, 1] = True      # a mirrored pair
    M[5, 8] = M[13, 4] = True     # two cells whose mirrors are correct
    sm = symmetric_error_share(M)
    check("hand-constructed mixed case: 4 errors, 2 symmetric -> share exactly 0.5",
          sm["n_errors"] == 4 and sm["n_symmetric"] == 2 and sm["symmetric_error_share"] == 0.5
          and sm["symmetric_error_share_offdiag"] == 0.5)
    # a diagonal cell is its own mirror: symmetric inside, invisible off-diagonal
    M2 = M.copy()
    M2[6, 6] = True
    sm2 = symmetric_error_share(M2)
    check("adding one diagonal error: 5 errors, 3 symmetric, off-diagonal share unchanged",
          sm2["n_errors"] == 5 and sm2["n_symmetric"] == 3
          and abs(sm2["symmetric_error_share"] - 3 / 5) < 1e-12
          and sm2["symmetric_error_share_offdiag"] == 0.5 and sm2["n_errors_offdiag"] == 4)

    ctrl = symmetric_share_control(S, N_CONTROL, seed=0)
    check("random-placement control is far below the constructed share",
          ctrl["control_mean"] < 0.3 and ctrl["z"] is not None and ctrl["z"] > 3.0)
    check("control echoes n_control, seed and the analytic expectation (n-1)/(p^2-1)",
          ctrl["n_control"] == N_CONTROL and ctrl["seed"] == 0
          and abs(ctrl["expected_share_offdiag_random_placement"] - 11 / (P * P - 1)) < 1e-12)
    again = symmetric_share_control(S, N_CONTROL, seed=0)
    check("control is reproducible: the same seed gives the same numbers, draw for draw",
          np.array_equal(again["control_values"], ctrl["control_values"])
          and again["control_mean"] == ctrl["control_mean"])
    other = symmetric_share_control(S, N_CONTROL, seed=1)
    check("a different seed gives a different draw (the seed is really used)",
          not np.array_equal(other["control_values"], ctrl["control_values"]))
    check("control values are stored per draw", ctrl["control_values"].shape == (N_CONTROL,))
    # size-matching probe: with EVERY cell wrong, a size-matched draw can only be the full
    # grid, so every control value is exactly 1.0 and the spread is 0
    full = symmetric_share_control(np.ones((P, P), dtype=bool), 5, seed=0)
    check("control is size-matched: n_errors = p^2 forces every draw to share exactly 1.0",
          full["control_mean"] == 1.0 and full["control_std"] == 0.0
          and full["n_control_defined"] == 5)

    # one-sided errors: mirrors are all correct -> share 0
    O = np.zeros((P, P), dtype=bool)
    for x, y in pairs:
        O[x, y] = True
    so = symmetric_error_share(O)
    check("one-sided errors: share == 0", so["symmetric_error_share"] == 0.0
          and so["symmetric_error_share_offdiag"] == 0.0)
    # diagonal-only errors: symmetric by construction, off-diagonal share undefined
    D = np.zeros((P, P), dtype=bool)
    for x in (2, 5, 9, 17):
        D[x, x] = True
    sd = symmetric_error_share(D)
    check("diagonal-only errors: share 1 incl. diagonal, off-diagonal share None (0/0)",
          sd["symmetric_error_share"] == 1.0 and sd["symmetric_error_share_offdiag"] is None
          and sd["n_errors_offdiag"] == 0)
    # random errors sit inside the control distribution (z small)
    cr = symmetric_share_control(E, N_CONTROL, seed=0)
    check("random errors: |z| < 3 against the random-placement control",
          cr["z"] is None or abs(cr["z"]) < 3.0)
    empty = symmetric_error_share(np.zeros((P, P), dtype=bool))
    check("no errors: both shares None (never a silent 0), n_errors 0",
          empty["symmetric_error_share"] is None and empty["symmetric_error_share_offdiag"] is None
          and empty["n_errors"] == 0)
    ce = symmetric_share_control(np.zeros((P, P), dtype=bool), N_CONTROL, seed=0)
    check("no errors: control undefined (0 defined draws), z None, expectation None",
          ce["n_control_defined"] == 0 and ce["z"] is None
          and ce["expected_share_offdiag_random_placement"] is None)
    check("error_histograms rejects a wrong-shape mask",
          _raises(lambda: error_histograms(np.zeros((P, P + 1), bool), P)))


def check_logit_level():
    print("-- logit-level quantities --")
    L = perfect_logits(P)
    y = target_grid(P, "add")
    a, b = 4, 9
    others = np.delete(L[a, b], y[a, b])
    check("margin matches a hand-computed cell",
          abs(correct_class_margin(L, P, "add")[a, b] - (L[a, b, y[a, b]] - others.max())) < 1e-12)
    L_wrong = force_wrong(L, [(a, b)], P)
    check("margin is negative on a wrong cell", correct_class_margin(L_wrong, P, "add")[a, b] < 0)
    t2 = top2_classes(L)
    check("top-2 best class is the target everywhere", np.array_equal(t2[..., 0], y))
    # swap the first two classes in ONE cell: same set, different order
    L_swap = L.copy()
    c1, c2 = t2[0, 0]
    L_swap[0, 0, c1], L_swap[0, 0, c2] = L[0, 0, c2], L[0, 0, c1]
    lg = logit_agreement(L, L_swap, P, "add")
    check("swapped top-2 order: set agreement 1, ordered agreement 1 - 1/p^2",
          lg["top2_agreement"] == 1.0 and abs(lg["top2_ordered_agreement"] - (1 - 1 / P ** 2)) < 1e-12)
    lg_id = logit_agreement(L, L.copy(), P, "add")
    check("identical logits: logit_agreement reports pearson 1.0 on both correlations",
          abs(lg_id["logit_pearson"] - 1) < 1e-12 and abs(lg_id["margin_pearson"] - 1) < 1e-12)
    noise = np.random.default_rng(77).normal(size=(P, P, P))
    lg_noise = logit_agreement(L, noise, P, "add")
    check("independent noise: logit_pearson and margin_pearson near 0",
          abs(lg_noise["logit_pearson"]) < 0.1 and abs(lg_noise["margin_pearson"]) < 0.1)
    x = np.arange(10.0)
    check("pearson(x, x) == 1 and pearson(x, -x) == -1",
          abs(pearson(x, x) - 1) < 1e-12 and abs(pearson(x, -x) + 1) < 1e-12)
    check("pearson RAISES on zero variance (documented: r undefined, no value invented)",
          _raises_with(lambda: pearson(x, np.ones(10)), "undefined"))
    check("pearson names the metric and the zero-variance side",
          _raises_with(lambda: pearson(np.ones(10), x, "logit_pearson"), "logit_pearson")
          and _raises_with(lambda: pearson(np.ones(10), x), "model a")
          and _raises_with(lambda: pearson(x, np.ones(10)), "model b"))
    check("pearson raises on shape mismatch", _raises(lambda: pearson(x, x[:5])))
    check("target grid: mul", int(target_grid(P, "mul")[5, 7]) == (5 * 7) % P)
    check("unknown task rejected", _raises(lambda: target_grid(P, "sub")))
    bad = L.copy()
    bad[0, 0, 0] = np.nan
    splits = random_split(P, 0.3, seed=1)
    check("non-finite logits rejected", _raises(lambda: compare_logits(L, bad, P, "add", splits, splits, 5, 0)))
    check("wrong-shape logits rejected",
          _raises(lambda: compare_logits(L, L[:, :, :5], P, "add", splits, splits, 5, 0)))
    check("non-partitioning split rejected",
          _raises(lambda: compare_logits(L, L, P, "add", (splits[0], splits[0]), splits, 5, 0)))
    pa, ca = preds_and_correct(L, P)
    check("agreement_block rejects an empty mask",
          _raises(lambda: agreement_block(pa, pa, ca, ca, np.zeros((P, P), bool))))


# --------------------------------------------------------------------------- #
# end to end on tiny run directories                                           #
# --------------------------------------------------------------------------- #
def make_run_dir(name: str, cfg, state: dict | None = None, init_seed: int = 0,
                 checkpoint_steps: tuple[int, ...] = ()) -> Path:
    """A minimal run directory: run.json + model_final.pt (+ v2 checkpoints if asked)."""
    d = SCRATCH / "runs" / name
    d.mkdir(parents=True, exist_ok=True)
    set_seed(init_seed)
    model = build_model(cfg)
    if state is not None:
        model.load_state_dict(state)
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(sd, d / "model_final.pt")
    (d / "run.json").write_text(json.dumps({"config": cfg.to_dict(), "git_commit": "test"}))
    if checkpoint_steps:
        entries = []
        for s in checkpoint_steps:
            path = checkpoint_path(d, s)
            torch.save(sd, path)
            entries.append({"step": int(s), "path": path.name, "kind": "grid"})
        (d / "checkpoints.json").write_text(json.dumps(entries, indent=2))
    return d


def check_end_to_end():
    print("-- end to end: load_side/compare/baseline/CLI on tiny run directories --")
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    out = SCRATCH / "out"
    cfg_a = get_config("nanda", p=P, arch="mlp", d_mlp=16, seed=0)
    cfg_b = get_config("nanda", p=P, arch="mlp", d_mlp=16, seed=1)             # other split
    cfg_f = get_config("nanda", p=P, arch="mlp", d_mlp=16, seed=2)             # other split + weights
    cfg_c = get_config("nanda", p=P, arch="mlp", d_mlp=16, seed=0, study="c")  # a's split, other weights
    cfg_d = get_config("nanda", p=29, arch="mlp", d_mlp=16, seed=0)            # other p
    cfg_e = get_config("nanda", p=P, d_model=16, n_heads=2, d_head=8, d_mlp=16, seed=0)  # transformer
    cfg_m = get_config("nanda", p=P, arch="mlp", d_mlp=16, seed=0, task="mul")  # other task
    # same config_key as cfg_a (steps is not part of run_id) but a differing config field
    cfg_x = get_config("nanda", p=P, arch="mlp", d_mlp=16, seed=3, steps=7)
    run_a = make_run_dir("a", cfg_a, init_seed=0, checkpoint_steps=(0, 5))
    state_a = torch.load(run_a / "model_final.pt")
    run_b = make_run_dir("b", cfg_b, state=state_a)
    run_f = make_run_dir("f", cfg_f, init_seed=9)
    run_c = make_run_dir("c", cfg_c, init_seed=5, checkpoint_steps=(0, 5))
    run_d = make_run_dir("d", cfg_d, init_seed=0)
    run_e = make_run_dir("e", cfg_e, init_seed=0)
    run_m = make_run_dir("m", cfg_m, init_seed=0)
    run_x = make_run_dir("x", cfg_x, init_seed=0)

    # --- loading one side ------------------------------------------------- #
    side_a = load_side(run_a)
    check("load_side: the legacy final model is identified by the bare run_id",
          side_a.id == cfg_a.run_id and side_a.meta["step"] == "final")
    check("load_side: logits [p, p, p] and a split that partitions the grid",
          side_a.logits.shape == (P, P, P)
          and bool((side_a.train_mask ^ side_a.test_mask).all())
          and len(side_a.split_hash) == 64)
    side_a5 = load_side(run_a, 5)
    check("load_side: a v2 checkpoint is identified as <run_id>_step{N:06d}",
          side_a5.id == f"{cfg_a.run_id}_step000005" and side_a5.meta["step"] == 5)

    # --- split handling --------------------------------------------------- #
    check("different splits raise by default (assert and record, INTERFACES section 8)",
          _raises_with(lambda: compare(run_a, run_b, out_dir=out), "do not share a train/test split"))
    ab = compare(run_a, run_b, n_control=N_CONTROL, seed=0, out_dir=out, require_same_split=False)
    r = ab["results"]
    check("same weights, other seed: agreement 1 and identical error sets",
          r["full_grid"]["agreement_rate"] == 1.0 and r["full_grid"]["error_jaccard"] == 1.0
          and r["full_grid"]["n_errors_union"] > 0)
    check("same weights: both_correct == accuracy_a == accuracy_b",
          r["full_grid"]["both_correct"] == r["full_grid"]["accuracy_a"]
          == r["full_grid"]["accuracy_b"])
    check("split mismatch is RECORDED, not raised, under require_same_split=False",
          r["split"]["split_hash_equal"] is False
          and r["split"]["split_hash_a"] != r["split"]["split_hash_b"]
          and len(r["split"]["split_hash_a"]) == 64)
    check("per-split numbers are reported for EACH model's own split",
          r["model_a_split"]["test"]["n_cells"] == r["split"]["n_test_a"]
          and r["model_b_split"]["test"]["n_cells"] == r["split"]["n_test_b"]
          and r["model_a_split"]["train"]["n_cells"] == r["split"]["n_train_a"]
          and r["model_b_split"]["train"]["n_cells"] == r["split"]["n_train_b"])
    check("envelope: module, version, run ids, steps and both checkpoint hashes",
          ab["module"] == "function_agreement" and ab["module_version"] == MODULE_VERSION
          and ab["run_id"] == cfg_a.run_id and ab["run_id_b"] == cfg_b.run_id
          and ab["step"] == "final" and ab["step_b"] == "final"
          and len(ab["checkpoint_sha256"]) == 64 and len(ab["checkpoint_sha256_b"]) == 64
          and ab["checkpoint_sha256"] == ab["checkpoint_sha256_b"])
    check("params echo n_control, seed, both steps and require_same_split",
          ab["params"]["n_control"] == N_CONTROL and ab["params"]["seed"] == 0
          and ab["params"]["require_same_split"] is False
          and ab["params"]["step_a"] == "final" and ab["params"]["step_b"] == "final")
    json_path = out / f"{cfg_a.run_id}__vs__{cfg_b.run_id}.json"
    npz_path = out / f"{cfg_a.run_id}__vs__{cfg_b.run_id}.npz"
    check("JSON + npz written under <id_a>__vs__<id_b>", json_path.exists() and npz_path.exists()
          and ab["output_json"] == str(json_path))
    on_disk = json.loads(json_path.read_text())
    check("JSON round-trips the headline number",
          on_disk["results"]["full_grid"]["agreement_rate"] == 1.0 and on_disk["arrays_file"] == npz_path.name)
    with np.load(npz_path) as z:
        keys = set(z.files)
        n_dis = z["disagreement_pairs"].shape[0]
        masks_ok = z["correct_a"].shape == (P, P) and z["train_mask_a"].sum() == r["split"]["n_train_a"]
        split_masks_differ = not np.array_equal(z["train_mask_a"], z["train_mask_b"])
    check("npz holds disagreement pairs, correctness masks, splits, histograms and control draws",
          {"disagreement_pairs", "correct_a", "correct_b", "train_mask_a", "train_mask_b",
           "model_a_errors_by_residue_sum", "model_b_errors_by_abs_diff",
           "disagreement_errors_by_a", "model_a_symmetric_control_values"} <= keys
          and n_dis == 0 and masks_ok and split_masks_differ)
    check("npz holds every histogram the module produces, for all three masks",
          all(f"{label}_errors_{k}" in keys
              for label in ("model_a", "model_b", "disagreement") for k in HISTOGRAM_NAMES))

    # --- shared split + run-level negative control ------------------------ #
    ac = compare(run_a, run_c, n_control=N_CONTROL, seed=0, out_dir=out)
    rc = ac["results"]
    check("same seed, other weights: split hashes equal (no raise under the default)",
          rc["split"]["split_hash_equal"] is True
          and rc["split"]["split_hash_a"] == rc["split"]["split_hash_b"])
    check("shared split: the per-split numbers are reported once (both blocks identical)",
          rc["model_a_split"] == rc["model_b_split"])
    check("NEGATIVE CONTROL (runs): two independently seeded models do NOT agree near 1",
          rc["full_grid"]["agreement_rate"] < 0.9)
    with np.load(out / f"{cfg_a.run_id}__vs__{cfg_c.run_id}.npz") as z:
        n_dis = z["disagreement_pairs"].shape[0]
    check("disagreement count == p^2 * (1 - agreement)",
          n_dis == rc["full_grid"]["n_disagreements"]
          and abs(n_dis - P * P * (1 - rc["full_grid"]["agreement_rate"])) < 1e-9)
    same = compare_sides(load_side(run_a), load_side(run_c), N_CONTROL, 0, out)
    check("compare_sides(load_side, load_side) reproduces compare() exactly",
          same["results"]["full_grid"] == rc["full_grid"]
          and same["results"]["logits"] == rc["logits"])
    check("p mismatch raises", _raises_with(lambda: compare(run_a, run_d, out_dir=out), "p=29"))
    check("task mismatch raises", _raises_with(lambda: compare(run_a, run_m, out_dir=out), "task="))
    check("missing run directory raises", _raises(lambda: compare(run_a, SCRATCH / "nope", out_dir=out)))

    # --- checkpoint sweep -------------------------------------------------- #
    check("common_checkpoint_steps returns the steps listed in BOTH runs",
          common_checkpoint_steps(run_a, run_c) == [0, 5])
    check("common_checkpoint_steps raises for a legacy run (no checkpoints.json)",
          _raises_with(lambda: common_checkpoint_steps(run_a, run_e), "checkpoints.json"))
    sweep = compare_all_checkpoints(run_a, run_c, N_CONTROL, 0, out)
    check("compare_all_checkpoints: one payload per common step, in step order",
          [pl["step"] for pl in sweep] == [0, 5] and [pl["step_b"] for pl in sweep] == [0, 5])
    check("each checkpoint pair writes <id_a>__vs__<id_b> with the step-tagged ids",
          all((out / f"{cfg_a.run_id}_step{s:06d}__vs__{cfg_c.run_id}_step{s:06d}.json").exists()
              and (out / f"{cfg_a.run_id}_step{s:06d}__vs__{cfg_c.run_id}_step{s:06d}.npz").exists()
              for s in (0, 5)))

    # --- baseline over run groups ------------------------------------------ #
    hl = headline(ab)
    check("headline carries every baseline metric", all(m in hl for m in BASELINE_METRICS))
    check("config_key strips the seed token and groups seed-only variants together",
          config_key(cfg_a) == config_key(cfg_b) == config_key(cfg_f)
          and "_seed" not in config_key(cfg_a)
          and config_key(cfg_c) != config_key(cfg_a) and config_key(cfg_e) != config_key(cfg_a))
    bl = baseline([run_a, run_b, run_f, run_c, run_e], n_control=N_CONTROL, seed=0, out_dir=out)
    grp = bl["by_config"][config_key(cfg_a)]
    lone_c = bl["by_config"][config_key(cfg_c)]
    lone_e = bl["by_config"][config_key(cfg_e)]
    check("baseline groups by config_key: 3 seeds -> 3 pairs; single-run groups reported with 0 pairs",
          grp["n_runs"] == 3 and grp["n_pairs"] == 3 and sorted(grp["seeds"]) == [0, 1, 2]
          and lone_c["n_runs"] == 1 and lone_c["n_pairs"] == 0
          and lone_e["n_runs"] == 1 and lone_e["n_pairs"] == 0)
    check("baseline: every pair of a group shares the architecture",
          all(pr["arch_a"] == pr["arch_b"] == "mlp" for pr in grp["pairs"])
          and grp["arch"] == "mlp" and lone_e["arch"] == "transformer")
    check("baseline compares seeds across their differing splits and records the mismatch",
          bl["params"]["require_same_split"] is False
          and all(pr["split_hash_equal"] is False for pr in grp["pairs"])
          and bl["params"]["ignored_config_fields"] == list(BASELINE_IGNORED_FIELDS))
    s = grp["summary"]["agreement_rate"]
    check("baseline summary: n 3, max 1.0 (the identical-weights pair), min < 1",
          s["n"] == 3 and s["max"] == 1.0 and s["min"] < 1.0)
    check("baseline summary counts undefined entries instead of hiding them",
          all("n_undefined" in grp["summary"][m] for m in BASELINE_METRICS))
    check("baseline summary of an empty group is n=0, no invented numbers",
          lone_e["summary"]["agreement_rate"] == {"n": 0, "n_undefined": 0})
    check("baseline table written", Path(bl["output_json"]).exists()
          and Path(bl["output_json"]).name.startswith("baseline__"))
    check("baseline needs >= 2 runs", _raises(lambda: baseline([run_a], out_dir=out)))
    check("baseline rejects a run paired with itself",
          _raises_with(lambda: baseline([run_a, run_a], out_dir=out), "duplicate run ids"))
    check("baseline rejects a same-key group whose configs differ outside seed/label/device",
          _raises_with(lambda: baseline([run_a, run_x], n_control=5, out_dir=out), "differ in config"))

    # --- CLI ---------------------------------------------------------------- #
    rd = results_dir()
    check("the CLI's default output directory is training/results/function_agreement",
          rd.name == "function_agreement" and rd.parent.name == "results"
          and rd.parent.parent.name == "training")
    cli_out = SCRATCH / "cli"          # an explicit out-dir, so the test leaves the repo clean
    printed = json.loads(_cli([str(run_a), str(run_c), "--n-control", "5",
                               "--out-dir", str(cli_out)]))
    check("CLI writes <id_a>__vs__<id_b>.json plus its .npz, and prints the payload",
          (cli_out / f"{cfg_a.run_id}__vs__{cfg_c.run_id}.json").exists()
          and (cli_out / f"{cfg_a.run_id}__vs__{cfg_c.run_id}.npz").exists()
          and printed["results"]["full_grid"]["agreement_rate"]
          == rc["full_grid"]["agreement_rate"])
    _cli([str(run_a), str(run_c), "--step", "5", "--n-control", "5", "--out-dir", str(cli_out)])
    check("CLI --step writes the step-tagged pair",
          (cli_out / f"{cfg_a.run_id}_step000005__vs__{cfg_c.run_id}_step000005.json").exists()
          and (cli_out / f"{cfg_a.run_id}_step000005__vs__{cfg_c.run_id}_step000005.npz").exists())
    printed_all = json.loads(_cli([str(run_a), str(run_c), "--all-checkpoints",
                                   "--n-control", "5", "--out-dir", str(cli_out)]))
    check("CLI --all-checkpoints reports every common step and the file it wrote",
          printed_all["steps"] == [0, 5] and len(printed_all["output_json"]) == 2
          and all(Path(pth).exists() for pth in printed_all["output_json"]))
    check("CLI refuses two runs with different splits unless --allow-different-splits",
          _raises_with(lambda: _cli([str(run_a), str(run_b), "--n-control", "5",
                                     "--out-dir", str(cli_out)]),
                       "do not share a train/test split"))
    _cli([str(run_a), str(run_b), "--n-control", "5", "--out-dir", str(cli_out),
          "--allow-different-splits"])
    check("CLI --allow-different-splits writes the cross-split pair",
          (cli_out / f"{cfg_a.run_id}__vs__{cfg_b.run_id}.json").exists())
    _cli(["--baseline", str(run_a), str(run_b), str(run_f), "--n-control", "5",
          "--out-dir", str(cli_out)])
    check("CLI --baseline writes a baseline table", any(cli_out.glob("baseline__*.json")))
    check("CLI rejects a single run directory (argparse usage error)",
          _raises(lambda: _cli([str(run_a), "--out-dir", str(cli_out)])))
    check("CLI rejects --baseline combined with positional run dirs",
          _raises(lambda: _cli([str(run_a), str(run_c), "--baseline", str(run_a), str(run_b),
                                "--out-dir", str(cli_out)])))
    shutil.rmtree(SCRATCH)


def main_tests():
    check_identical_models()
    check_known_disagreement()
    check_scrambled_controls()
    check_negative_control()
    check_error_structure()
    check_logit_level()
    check_end_to_end()
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main_tests()
