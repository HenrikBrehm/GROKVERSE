"""Checks for analysis/function_agreement.py (INTERFACES.md section 8).

Run from training/:  python tests/test_function_agreement.py
Same check() convention as test_core.py; exits non-zero on the first failure.

Synthetic inputs with known answers (p = 23): identical models, models differing on a
known set of 10 cells, constructed error structures, plus scrambled controls that must
NOT look like agreement. The end-to-end part writes tiny run directories to the
scratchpad and exercises compare()/baseline() including the JSON/npz outputs.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grokverse.analysis.function_agreement import (  # noqa: E402
    BASELINE_METRICS, EMPTY_UNION_JACCARD, agreement_block, baseline, compare,
    compare_logits, correct_class_margin, disagreement_pairs, error_histograms,
    headline, histogram_summary, logit_agreement, main, pearson,
    symmetric_error_share, symmetric_share_control, target_grid, top2_classes)
from grokverse.config import get_config  # noqa: E402
from grokverse.models import build_model  # noqa: E402
from grokverse.seed import set_seed  # noqa: E402

SCRATCH = Path("C:/Users/henri/AppData/Local/Temp/claude/C--Users-henri-Documents-Brain-bwki/"
               "19fe38fd-ad0c-4eae-bfeb-002bb125b3f5/scratchpad") / "function_agreement_tests"
P = 23
N_CONTROL = 20


def check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        raise SystemExit(1)


def _raises(fn) -> bool:
    try:
        fn()
    except Exception:
        return True
    return False


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
    check("shared split: model_b_split == model_a_split", res["model_b_split"] == res["model_a_split"])
    check("per-split errors add up to the full-grid count",
          tr["n_errors_a"] + te["n_errors_a"] == fg["n_errors_a"] == 5)

    L0 = perfect_logits(P)
    res0, _ = compare_logits(L0, L0, P, "add", splits, splits, N_CONTROL, 0)
    fg0 = res0["full_grid"]
    check("identical, zero errors: jaccard reports the declared empty-union value + union 0",
          fg0["error_jaccard"] == EMPTY_UNION_JACCARD and fg0["n_errors_union"] == 0
          and fg0["accuracy_a"] == 1.0)


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
    check("disagreement list equals the known set", got == S)
    check("disagreement_pairs() matches the npz array",
          np.array_equal(disagreement_pairs(L_a.argmax(-1), L_b.argmax(-1)), arrs["disagreement_pairs"]))
    check("only_a_correct == 10/p^2, only_b == both_wrong == 0",
          abs(fg["only_a_correct"] - 10 / P ** 2) < 1e-12
          and fg["only_b_correct"] == 0.0 and fg["both_wrong"] == 0.0)
    check("error_jaccard == 0 (a has no errors, b has 10)",
          fg["error_jaccard"] == 0.0 and fg["n_errors_a"] == 0 and fg["n_errors_b"] == 10)
    es = res["error_structure"]
    check("error-structure histograms of model b sum to its 10 errors",
          all(es["model_b"]["histograms"][k]["total"] == 10
              for k in ("by_residue_sum", "by_a", "by_b", "by_abs_diff")))
    check("npz histograms sum to the error count too",
          all(int(arrs[f"model_b_errors_{k}"].sum()) == 10
              for k in ("by_residue_sum", "by_a", "by_b", "by_abs_diff")))
    check("disagreement histograms sum to the number of disagreements",
          es["disagreement"]["n_errors"] == 10
          and es["disagreement"]["histograms"]["by_a"]["total"] == 10)
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


def check_error_structure():
    print("-- error structure --")
    rng = np.random.default_rng(11)
    E = np.zeros(P * P, dtype=bool)
    E[rng.choice(P * P, size=37, replace=False)] = True
    E = E.reshape(P, P)
    hists = error_histograms(E, P)
    check("all four histograms have p bins and sum to 37",
          all(v.shape == (P,) and int(v.sum()) == 37 for v in hists.values()))
    a, b = np.nonzero(E)
    r0 = int(((a + b) % P == 4).sum())
    check("by_residue_sum bin 4 matches a hand count", int(hists["by_residue_sum"][4]) == r0)
    check("by_abs_diff bin 0 counts the diagonal errors",
          int(hists["by_abs_diff"][0]) == int(np.diag(E).sum()))
    hs = histogram_summary(hists["by_a"])
    check("histogram summary: total 37, entropy in (0, 1]",
          hs["total"] == 37 and 0.0 < hs["normalized_entropy"] <= 1.0)
    check("histogram summary of an empty histogram: entropy None, total 0",
          histogram_summary(np.zeros(P, int))["normalized_entropy"] is None)

    # symmetric errors by construction: (a, b) AND (b, a) wrong for 6 off-diagonal pairs
    pairs = [(1, 2), (3, 9), (4, 20), (7, 8), (10, 15), (0, 22)]
    S = np.zeros((P, P), dtype=bool)
    for x, y in pairs:
        S[x, y] = S[y, x] = True
    ss = symmetric_error_share(S)
    check("constructed symmetric errors: share == 1 (incl. and excl. diagonal), n_errors 12",
          ss["symmetric_error_share"] == 1.0 and ss["symmetric_error_share_offdiag"] == 1.0
          and ss["n_errors"] == 12)
    ctrl = symmetric_share_control(S, N_CONTROL, seed=0)
    check("random-placement control is far below the constructed share",
          ctrl["control_mean"] < 0.3 and ctrl["z"] is not None and ctrl["z"] > 3.0)
    check("control echoes n_control, seed and the analytic expectation (n-1)/(p^2-1)",
          ctrl["n_control"] == N_CONTROL and ctrl["seed"] == 0
          and abs(ctrl["expected_share_offdiag_random_placement"] - 11 / (P * P - 1)) < 1e-12)
    check("control is reproducible under its seed",
          symmetric_share_control(S, N_CONTROL, seed=0)["control_mean"] == ctrl["control_mean"])
    check("control values are stored per draw", ctrl["control_values"].shape == (N_CONTROL,))

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
    check("diagonal-only errors: share 1 incl. diagonal, off-diagonal share None",
          sd["symmetric_error_share"] == 1.0 and sd["symmetric_error_share_offdiag"] is None
          and sd["n_errors_offdiag"] == 0)
    # random errors sit inside the control distribution (z small)
    cr = symmetric_share_control(E, N_CONTROL, seed=0)
    check("random errors: |z| < 3 against the random-placement control",
          cr["z"] is None or abs(cr["z"]) < 3.0)
    empty = symmetric_error_share(np.zeros((P, P), dtype=bool))
    check("no errors: shares None, n_errors 0", empty["symmetric_error_share"] is None and empty["n_errors"] == 0)
    ce = symmetric_share_control(np.zeros((P, P), dtype=bool), N_CONTROL, seed=0)
    check("no errors: control undefined (0 defined draws), z None",
          ce["n_control_defined"] == 0 and ce["z"] is None)


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
    x = np.arange(10.0)
    check("pearson(x, x) == 1 and pearson(x, -x) == -1",
          abs(pearson(x, x) - 1) < 1e-12 and abs(pearson(x, -x) + 1) < 1e-12)
    check("pearson raises on zero variance", _raises(lambda: pearson(x, np.ones(10))))
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
def make_run_dir(name: str, cfg, state: dict | None = None, init_seed: int = 0) -> Path:
    d = SCRATCH / "runs" / name
    d.mkdir(parents=True, exist_ok=True)
    set_seed(init_seed)
    model = build_model(cfg)
    if state is not None:
        model.load_state_dict(state)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, d / "model_final.pt")
    (d / "run.json").write_text(json.dumps({"config": cfg.to_dict(), "git_commit": "test"}))
    return d


def check_end_to_end():
    print("-- end to end: compare() and baseline() on tiny run directories --")
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    out = SCRATCH / "out"
    cfg_a = get_config("nanda", p=P, arch="mlp", d_mlp=16, seed=0)
    cfg_b = get_config("nanda", p=P, arch="mlp", d_mlp=16, seed=1)          # other split
    cfg_c = get_config("nanda", p=P, arch="mlp", d_mlp=16, seed=0, study="c")  # other weights
    cfg_d = get_config("nanda", p=29, arch="mlp", d_mlp=16, seed=0)
    cfg_e = get_config("nanda", p=P, d_model=16, n_heads=2, d_head=8, d_mlp=16, seed=0)
    run_a = make_run_dir("a", cfg_a, init_seed=0)
    state_a = torch.load(run_a / "model_final.pt")
    run_b = make_run_dir("b", cfg_b, state=state_a)
    run_c = make_run_dir("c", cfg_c, init_seed=5)
    run_d = make_run_dir("d", cfg_d, init_seed=0)
    run_e = make_run_dir("e", cfg_e, init_seed=0)

    ab = compare(run_a, run_b, n_control=N_CONTROL, seed=0, out_dir=out)
    r = ab["results"]
    check("same weights, other seed: agreement 1 and identical error sets",
          r["full_grid"]["agreement_rate"] == 1.0 and r["full_grid"]["error_jaccard"] == 1.0)
    check("split hashes recorded and unequal (no raise)",
          r["split"]["split_hash_equal"] is False and r["split"]["split_hash_a"] != r["split"]["split_hash_b"]
          and len(r["split"]["split_hash_a"]) == 64)
    check("per-split numbers are reported for each model's own split",
          r["model_a_split"]["test"]["n_cells"] == r["split"]["n_test_a"]
          and r["model_b_split"]["test"]["n_cells"] == r["split"]["n_test_b"])
    check("envelope: module, run ids, steps and both checkpoint hashes",
          ab["module"] == "function_agreement" and ab["run_id"] == cfg_a.run_id
          and ab["run_id_b"] == cfg_b.run_id and ab["step"] == "final" and ab["step_b"] == "final"
          and len(ab["checkpoint_sha256"]) == 64 and len(ab["checkpoint_sha256_b"]) == 64
          and ab["checkpoint_sha256"] == ab["checkpoint_sha256_b"])
    check("params echo n_control, seed and the empty-union jaccard value",
          ab["params"]["n_control"] == N_CONTROL and ab["params"]["seed"] == 0
          and ab["params"]["error_jaccard_empty_union_value"] == EMPTY_UNION_JACCARD)
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
    check("npz holds disagreement pairs, correctness masks, splits, histograms and control draws",
          {"disagreement_pairs", "correct_a", "correct_b", "train_mask_a", "train_mask_b",
           "model_a_errors_by_residue_sum", "model_b_errors_by_abs_diff",
           "disagreement_errors_by_a", "model_a_symmetric_control_values"} <= keys
          and n_dis == 0 and masks_ok)

    ac = compare(run_a, run_c, n_control=N_CONTROL, seed=0, out_dir=out)
    rc = ac["results"]
    check("same seed, other weights: split hashes equal", rc["split"]["split_hash_equal"] is True)
    check("different random models disagree", rc["full_grid"]["agreement_rate"] < 0.9)
    with np.load(out / f"{cfg_a.run_id}__vs__{cfg_c.run_id}.npz") as z:
        n_dis = z["disagreement_pairs"].shape[0]
    check("disagreement count == p^2 * (1 - agreement)",
          n_dis == rc["full_grid"]["n_disagreements"]
          and abs(n_dis - P * P * (1 - rc["full_grid"]["agreement_rate"])) < 1e-9)
    check("p mismatch raises", _raises(lambda: compare(run_a, run_d, out_dir=out)))
    check("missing run directory raises", _raises(lambda: compare(run_a, SCRATCH / "nope", out_dir=out)))

    hl = headline(ab)
    check("headline carries every baseline metric", all(m in hl for m in BASELINE_METRICS))
    bl = baseline([run_a, run_b, run_c, run_e], n_control=N_CONTROL, seed=0, out_dir=out)
    mlp, txf = bl["by_arch"]["mlp"], bl["by_arch"]["transformer"]
    check("baseline: 3 mlp pairs, 0 transformer pairs (single run, reported)",
          mlp["n_pairs"] == 3 and mlp["n_runs"] == 3 and txf["n_pairs"] == 0 and txf["n_runs"] == 1)
    check("baseline: no cross-architecture pair",
          all(pr["arch_a"] == pr["arch_b"] == "mlp" for pr in mlp["pairs"]))
    s = mlp["summary"]["agreement_rate"]
    check("baseline summary: n 3, max 1.0 (the identical-weights pair), min < 1",
          s["n"] == 3 and s["max"] == 1.0 and s["min"] < 1.0)
    check("baseline summary of an empty group is n=0, no invented numbers",
          txf["summary"]["agreement_rate"] == {"n": 0})
    check("baseline table written", Path(bl["output_json"]).exists()
          and Path(bl["output_json"]).name.startswith("baseline__"))
    check("baseline needs >= 2 runs", _raises(lambda: baseline([run_a], out_dir=out)))
    main([str(run_a), str(run_b), "--n-control", "5", "--out-dir", str(out)])
    check("CLI runs on two run directories", True)
    shutil.rmtree(SCRATCH)


def main_tests():
    check_identical_models()
    check_known_disagreement()
    check_scrambled_controls()
    check_error_structure()
    check_logit_level()
    check_end_to_end()
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main_tests()
