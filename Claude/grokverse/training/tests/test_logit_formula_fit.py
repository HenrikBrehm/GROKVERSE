"""Checks for analysis/logit_formula_fit.py (INTERFACES §7).

Run from training/:  python tests/test_logit_formula_fit.py
Same check() convention as test_core.py; exits non-zero on the first failure.  Synthetic tensors
with known answers at p = 23; the random-frequency control is exercised at the study's p = 113,
where a random set of the family's size rarely contains the whole key set (at p = 23 it does in
~27 % of draws, which would make the control meaningless rather than the code wrong).
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grokverse.analysis.common import center_logits, split_masks  # noqa: E402
from grokverse.analysis.logit_formula_fit import (  # noqa: E402
    FORMULA_IDS, FeatureSet, analyse, design_matrix, diff_features, fit_all_formulas,
    fit_formula, normal_equations, phase_grid, resolve_key_set, search_square_phases,
    square_features, square_wave, sum_direction_power, sum_features)
from grokverse.config import get_config  # noqa: E402
from grokverse.models import build_model  # noqa: E402
from grokverse.seed import set_seed  # noqa: E402

SCRATCH = Path("C:/Users/henri/AppData/Local/Temp/claude/C--Users-henri-Documents-Brain-bwki/"
               "19fe38fd-ad0c-4eae-bfeb-002bb125b3f5/scratchpad")
P = 23


def check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        raise SystemExit(1)


def _raises(fn, exc=Exception) -> bool:
    try:
        fn()
    except exc:
        return True
    return False


# --------------------------------------------------------------------------- #
# synthetic tensors with known answers                                         #
# --------------------------------------------------------------------------- #
def residues(p: int) -> tuple[np.ndarray, np.ndarray]:
    a = np.arange(p)[:, None, None]
    b = np.arange(p)[None, :, None]
    c = np.arange(p)[None, None, :]
    return (a + b - c) % p, (a - b - c) % p


def sinusoid_tensor(p: int, K, amps, phases, index: str = "sum") -> np.ndarray:
    """L[a,b,c] = Σ_k A_k cos(2πk n/p − φ_k), n = (a+b−c) or (a−b−c) mod p."""
    n_sum, n_diff = residues(p)
    n = n_sum if index == "sum" else n_diff
    return sum(A * np.cos(2 * np.pi * k * n / p - phi) for k, A, phi in zip(K, amps, phases))


def square_tensor(p: int, K, phases) -> np.ndarray:
    """L[a,b,c] = Σ_k sq(2πk n/p + φ_k), n = (a+b−c) mod p."""
    n_sum, _ = residues(p)
    return sum(square_wave(k, p, phi)[n_sum] for k, phi in zip(K, phases))


def masks(p: int, seed: int = 0):
    return split_masks(get_config("nanda", p=p, seed=seed))


# --------------------------------------------------------------------------- #
# 1. the residue-aggregated normal equations equal the explicit design matrix  #
# --------------------------------------------------------------------------- #
def check_normal_equations():
    print("\n-- normal equations vs explicit p^3-row design matrix --")
    rng = np.random.default_rng(0)
    L = rng.standard_normal((P, P, P))
    for fs in (sum_features([3, 5], P), diff_features([3, 5], P),
               square_features([2, 7], P, [0.3, 1.9]), sum_features(range(1, (P - 1) // 2 + 1), P)):
        X = design_matrix(fs)
        check(f"{fs.index_variable} {fs.names[-1]}: design matrix has p^3 rows",
              X.shape == (P ** 3, fs.n_features) and X.dtype == np.float32)
        # float64 reference built the slow way from the same column table
        Xref = design_matrix_float64(fs)
        XtX_ref, Xty_ref = Xref.T @ Xref, Xref.T @ L.reshape(-1)
        XtX, Xty = normal_equations(L, fs)
        check(f"{fs.index_variable} {fs.names[-1]}: X^T X by residue aggregation is exact",
              np.abs(XtX - XtX_ref).max() < 1e-8 * np.abs(XtX_ref).max())
        check(f"{fs.index_variable} {fs.names[-1]}: X^T y by residue aggregation is exact",
              np.abs(Xty - Xty_ref).max() < 1e-9 * max(1.0, np.abs(Xty_ref).max()))
    check("full_sum_basis has p parameters",
          sum_features(range(1, (P - 1) // 2 + 1), P).n_params == P)
    check("sparse_sinusoid has 2|K|+1 parameters", sum_features([3, 5], P).n_params == 5)
    check("ideal_square charges 2|K|+1 parameters (phases are free)",
          square_features([3, 5], P, [0.1, 0.2]).n_params == 5)


def design_matrix_float64(fs: FeatureSet) -> np.ndarray:
    n_sum, n_diff = residues(fs.p)
    n = n_sum if fs.index_variable == "a+b-c" else n_diff
    return fs.columns[n.reshape(-1)]


# --------------------------------------------------------------------------- #
# 2. exact sinusoid circuit                                                    #
# --------------------------------------------------------------------------- #
def check_sinusoid_circuit():
    print("\n-- synthetic circuit: L = sum_k cos(w_k(a+b-c)), K = {3, 5} --")
    K = (3, 5)
    tr, te = masks(P)
    L = sinusoid_tensor(P, K, (1.0, 1.0), (0.0, 0.0))
    res, arrays = fit_all_formulas(L, K, tr, te, seed=0, n_control=10)
    f = res["formulas"]
    check("sparse_sinusoid r2_all > 0.999", f["sparse_sinusoid"]["r2_all"] > 0.999)
    check("sparse_sinusoid r2_train and r2_test > 0.999",
          f["sparse_sinusoid"]["r2_train"] > 0.999 and f["sparse_sinusoid"]["r2_test"] > 0.999)
    check("sparse_sinusoid argmax accuracy 1.0 on train and test",
          f["sparse_sinusoid"]["argmax_accuracy_train"] == 1.0
          and f["sparse_sinusoid"]["argmax_accuracy_test"] == 1.0
          and f["sparse_sinusoid"]["n_cells_argmax_tied"] == 0)
    check("control_difference r2_all < 0.05 (wrong symmetry)",
          f["control_difference"]["r2_all"] < 0.05)
    check("control_difference r2_test < 0.05", f["control_difference"]["r2_test"] < 0.05)
    check("odd_harmonics contains K and reaches r2 > 0.999",
          set(K) <= set(f["odd_harmonics"]["frequencies"]) and f["odd_harmonics"]["r2_all"] > 0.999)
    check("full_sum_basis r2 > 0.999 (the ceiling)", f["full_sum_basis"]["r2_all"] > 0.999)
    check("control_top_m picks K from the logit sum-direction power",
          set(K) <= set(f["control_top_m"]["frequencies"]) and f["control_top_m"]["r2_all"] > 0.999)
    share = np.asarray(res["logit_sum_direction_power"]["share"])
    check("sum-direction power share sits on k=3 and k=5 only",
          share[2] + share[4] > 0.999 and share.sum() < 1.0 + 1e-9)
    check("ideal_square does NOT reach the sinusoid's r2 on a sinusoid",
          f["ideal_square"]["r2_all"] < 0.95)
    check("per-class r2 quantiles reported with n = p classes",
          f["sparse_sinusoid"]["per_class_r2"]["n"] == P
          and f["sparse_sinusoid"]["per_class_r2"]["q05"] > 0.999)
    check("residual RMS ~ 0 for the exact circuit", f["sparse_sinusoid"]["residual_rms_all"] < 1e-9)
    check("residual RMS map is [p, p] float32",
          arrays["residual_rms_map_sparse_sinusoid"].shape == (P, P)
          and arrays["residual_rms_map_sparse_sinusoid"].dtype == np.float32)
    check("random control values stored (n_control draws, sets of size |F|)",
          arrays["control_random_r2_all"].shape == (10,)
          and arrays["control_random_frequency_sets"].shape == (10, res["family_size"]))
    check("every formula id of the §7 table is reported", tuple(f) == FORMULA_IDS)

    # wrong key set: same flexibility, no overlap with the circuit -> nothing explained
    wrong = fit_formula(L, sum_features([1, 2], P), tr, te)["report"]
    check("wrong key set {1, 2}: r2 < 0.05", wrong["r2_all"] < 0.05)

    # amplitude + phase recovery in the §0 convention A cos(w n - phi)
    L2 = sinusoid_tensor(P, K, (1.0, 0.7), (0.4, -1.1))
    rep = fit_formula(L2, sum_features(K, P), tr, te)["report"]
    tab = {r["k"]: r for r in rep["coefficient_table"]["per_frequency"]}
    check("amplitudes recovered", abs(tab[3]["amplitude"] - 1.0) < 1e-9
          and abs(tab[5]["amplitude"] - 0.7) < 1e-9)
    check("phases recovered (phi = atan2(beta_sin, beta_cos))",
          abs(tab[3]["phase_rad"] - 0.4) < 1e-9 and abs(tab[5]["phase_rad"] + 1.1) < 1e-9)
    check("constant coefficient ~ 0 on centred logits", abs(rep["coefficient_table"]["const"]) < 1e-9)


# --------------------------------------------------------------------------- #
# 3. scrambled control: must NOT pass                                          #
# --------------------------------------------------------------------------- #
def check_scrambled_control():
    print("\n-- scrambled control: class axis permuted per (a, b) --")
    K = (3, 5)
    tr, te = masks(P)
    L = sinusoid_tensor(P, K, (1.0, 1.0), (0.0, 0.0))
    rng = np.random.default_rng(1)
    Ls = np.stack([np.stack([L[a, b, rng.permutation(P)] for b in range(P)]) for a in range(P)])
    rep = fit_formula(Ls, sum_features(K, P), tr, te)["report"]
    check("scrambled: sparse_sinusoid r2 < 0.05", rep["r2_all"] < 0.05)
    check("scrambled: argmax accuracy near chance (< 0.15)", rep["argmax_accuracy_test"] < 0.15)
    full = fit_formula(Ls, sum_features(range(1, (P - 1) // 2 + 1), P), tr, te)["report"]
    check("scrambled: even the full sum basis explains < 0.05", full["r2_all"] < 0.05)
    diff_L = sinusoid_tensor(P, K, (1.0, 1.0), (0.0, 0.0), index="diff")
    sdp = sum_direction_power(diff_L)
    check("a (a-b-c) tensor has ~no power in the sum directions", sdp["share"].max() < 1e-9)


# --------------------------------------------------------------------------- #
# 4. discrete square-wave tensor                                               #
# --------------------------------------------------------------------------- #
def check_square_tensor():
    print("\n-- synthetic square-wave tensor: L = sum_k sq(w_k(a+b-c) + phi_k) + noise --")
    K, phis = (3, 5), (0.3, 2.0)
    tr, te = masks(P)
    rng = np.random.default_rng(2)
    L = square_tensor(P, K, phis) + 0.05 * rng.standard_normal((P, P, P))
    res, _ = fit_all_formulas(L, K, tr, te, seed=0, n_control=10)
    f = res["formulas"]
    check("ideal_square r2 > 0.99", f["ideal_square"]["r2_all"] > 0.99)
    check("ideal_square beats sparse_sinusoid by AIC",
          f["ideal_square"]["aic"] < f["sparse_sinusoid"]["aic"])
    check("odd_harmonics beats sparse_sinusoid by AIC",
          f["odd_harmonics"]["aic"] < f["sparse_sinusoid"]["aic"])
    check("odd_harmonics r2 exceeds sparse_sinusoid r2 on a square wave",
          f["odd_harmonics"]["r2_all"] > f["sparse_sinusoid"]["r2_all"] + 0.05)
    check("AIC ranking puts ideal_square first",
          res["aic_comparison"]["ranking"][0] == "ideal_square"
          and res["aic_comparison"]["delta_aic"]["ideal_square"] == 0.0)
    got = f["ideal_square"]["phases_rad"]
    same = all((square_wave(k, P, g) == square_wave(k, P, t)).all()
               for k, g, t in zip(K, got, phis))
    check("searched phases reproduce the true sign patterns exactly", same)
    corr = res["ideal_square_phase_search"]["correlation_max"]
    check("phase-search correlations are high and reported per k",
          len(corr) == 2 and min(corr) > 0.5)
    check("phase grid has 4p points", res["ideal_square_phase_search"]["n_grid"] == 4 * P)
    grid = phase_grid(P)
    n = np.arange(P)
    min_cos = min(np.abs(np.cos(2 * np.pi * k * n[None, :] / P + grid[:, None])).min()
                  for k in range(1, (P - 1) // 2 + 1))
    check("no grid phase lands on a zero crossing of the sampled square wave", min_cos > 1e-6)
    check("phase search alone recovers the phases (grid indices reported)",
          len(search_square_phases(L, K, tr, te)["grid_index"]) == 2)

    # AIC on the sinusoid side too: the sparse model must not lose to the ceiling on noise
    Ls = sinusoid_tensor(P, K, (1.0, 1.0), (0.0, 0.0)) + 0.1 * rng.standard_normal((P, P, P))
    rs, _ = fit_all_formulas(Ls, K, tr, te, seed=0, n_control=5)
    check("noisy sinusoid circuit: sparse_sinusoid beats full_sum_basis by AIC",
          rs["formulas"]["sparse_sinusoid"]["aic"] < rs["formulas"]["full_sum_basis"]["aic"])
    check("noisy sinusoid circuit: sparse_sinusoid beats ideal_square by AIC",
          rs["formulas"]["sparse_sinusoid"]["aic"] < rs["formulas"]["ideal_square"]["aic"])


# --------------------------------------------------------------------------- #
# 5. iid Gaussian logits: every r2 ~ n_params / p^3                            #
# --------------------------------------------------------------------------- #
def check_gaussian_logits():
    print("\n-- iid Gaussian logits --")
    tr, te = masks(P)
    L = np.random.default_rng(3).standard_normal((P, P, P))
    res, _ = fit_all_formulas(L, (3, 5), tr, te, seed=0, n_control=10)
    N = P ** 3
    for fid, rep in res["formulas"].items():
        check(f"{fid}: r2_all < 0.02 and within 0.005 of n_params/p^3 ({rep['n_params']}/{N})",
              0.0 <= rep["r2_all"] < 0.02 and abs(rep["r2_all"] - rep["n_params"] / N) < 0.005)
    c = res["control_random_m"]["r2_all"]
    check("random control r2 is also ~ n_params/p^3", c["control_mean"] < 0.02)
    check("z-scores are reported for every formula and metric",
          all(m in res["z_vs_control_random_m"][fid] for fid in FORMULA_IDS
              for m in ("r2_all", "r2_test", "argmax_accuracy_test")))


# --------------------------------------------------------------------------- #
# 6. random control at the study's p = 113                                    #
# --------------------------------------------------------------------------- #
def check_random_control_p113():
    print("\n-- random-frequency control at p = 113 (K = {3, 5, 7}) --")
    p, K = 113, (3, 5, 7)
    tr, te = masks(p)
    L = sinusoid_tensor(p, K, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    res, arrays = fit_all_formulas(L, K, tr, te, seed=0, n_control=50)
    f, c = res["formulas"], res["control_random_m"]
    check("p=113: sparse_sinusoid r2 > 0.999 and test accuracy 1.0",
          f["sparse_sinusoid"]["r2_all"] > 0.999 and f["sparse_sinusoid"]["argmax_accuracy_test"] == 1.0)
    check("p=113: control uses 50 seeded draws of size |F|",
          c["n_control"] == 50 and c["seed"] == 0 and c["set_size"] == res["family_size"]
          and arrays["control_random_frequency_sets"].shape == (50, res["family_size"]))
    check("p=113: control r2 distribution lies below the true-K fit (q95 < r2)",
          c["r2_all"]["control_q95"] < f["sparse_sinusoid"]["r2_all"]
          and c["r2_test"]["control_q95"] < f["sparse_sinusoid"]["r2_test"])
    check("p=113: control mean r2 well below the true-K fit",
          c["r2_all"]["control_mean"] < f["sparse_sinusoid"]["r2_all"] - 0.5)
    check("p=113: z-score of sparse_sinusoid against the control >= 3",
          res["z_vs_control_random_m"]["sparse_sinusoid"]["r2_test"] >= 3.0)
    check("p=113: control_difference sits inside the control's bulk (z < 1)",
          res["z_vs_control_random_m"]["control_difference"]["r2_test"] < 1.0)
    check("p=113: control draws are reproducible under the seed",
          np.array_equal(arrays["control_random_frequency_sets"],
                         fit_all_formulas(L, K, tr, te, seed=0, n_control=50)[1]
                         ["control_random_frequency_sets"]))


# --------------------------------------------------------------------------- #
# 7. centering                                                                 #
# --------------------------------------------------------------------------- #
def check_centering():
    print("\n-- centering: a per-(a, b) constant changes nothing --")
    K = (3, 5)
    tr, te = masks(P)
    rng = np.random.default_rng(4)
    L = sinusoid_tensor(P, K, (1.0, 0.5), (0.2, 0.9)) + 0.3 * rng.standard_normal((P, P, P))
    L_shift = L + 5.0 * rng.standard_normal((P, P, 1))
    r1, _ = fit_all_formulas(L, K, tr, te, seed=0, n_control=5)
    r2, _ = fit_all_formulas(L_shift, K, tr, te, seed=0, n_control=5)
    keys = ("r2_all", "r2_train", "r2_test", "aic", "argmax_accuracy_train",
            "argmax_accuracy_test", "residual_rms_all")
    same = all(abs(r1["formulas"][fid][k] - r2["formulas"][fid][k]) < 1e-7 * max(1.0, abs(r1["formulas"][fid][k]))
               for fid in FORMULA_IDS for k in keys)
    check("all r2 / aic / accuracy identical after adding a per-(a,b) constant", same)
    check("centering is idempotent", np.abs(center_logits(center_logits(L)) - center_logits(L)).max() < 1e-12)


# --------------------------------------------------------------------------- #
# 8. key-set resolution and error handling                                     #
# --------------------------------------------------------------------------- #
def check_key_set_and_errors():
    print("\n-- key set resolution + explicit errors --")
    n = np.arange(P)
    W = np.stack([np.cos(2 * np.pi * 4 * n / P), np.sin(2 * np.pi * 4 * n / P)], axis=1)
    K, info = resolve_key_set(None, None, None, "embedding_top8", {"W_E": W}, P)
    check("embedding_top8 recovers the injected embedding frequency",
          K == [4] and info["cap_binding"] is False and info["threshold"] == 0.9 and info["max_k"] == 8)
    K2, info2 = resolve_key_set(None, None, [5, 3], None, {}, P)
    check("explicit key set recorded as rule 'explicit'", K2 == [5, 3] and info2["key_rule"] == "explicit")
    check("both key_freqs and key_rule -> ValueError",
          _raises(lambda: resolve_key_set(None, None, [3], "embedding_top8", {"W_E": W}, P), ValueError))
    check("neither key_freqs nor key_rule -> ValueError",
          _raises(lambda: resolve_key_set(None, None, None, None, {}, P), ValueError))
    check("empty explicit key set -> ValueError",
          _raises(lambda: resolve_key_set(None, None, [], None, {}, P), ValueError))
    check("embedding_top8 without W_E -> ValueError",
          _raises(lambda: resolve_key_set(None, None, None, "embedding_top8", {}, P), ValueError))
    try:
        resolve_key_set(None, None, None, "neuron_clusters", {"W_E": W}, P)
        named = False
    except NotImplementedError as exc:
        named = "key_frequencies" in str(exc) and "neuron_clusters" in str(exc)
    check("other rules raise NotImplementedError naming analysis.key_frequencies", named)

    tr, te = masks(P)
    L = np.random.default_rng(5).standard_normal((P, P, P))
    check("even p rejected", _raises(lambda: sum_features([1], 22), ValueError))
    check("duplicate frequencies rejected", _raises(lambda: sum_features([3, 3], P), ValueError))
    check("out-of-range frequency rejected", _raises(lambda: sum_features([12], P), ValueError))
    check("phase count mismatch rejected",
          _raises(lambda: square_features([3, 5], P, [0.1]), ValueError))
    check("wrong logit shape rejected",
          _raises(lambda: fit_formula(L[:, :, :5], sum_features([3], P), tr, te), ValueError))
    check("non-bool mask rejected",
          _raises(lambda: fit_formula(L, sum_features([3], P), tr.astype(int), te), ValueError))
    check("overlapping masks rejected",
          _raises(lambda: fit_formula(L, sum_features([3], P), tr, tr), ValueError))
    check("all-zero logits rejected",
          _raises(lambda: fit_formula(np.zeros((P, P, P)), sum_features([3], P), tr, te), ValueError))
    check("non-finite logits rejected",
          _raises(lambda: fit_formula(L * np.nan, sum_features([3], P), tr, te), ValueError))
    check("empty key set for the full table rejected",
          _raises(lambda: fit_all_formulas(L, (), tr, te), ValueError))
    check("n_control < 2 rejected",
          _raises(lambda: fit_all_formulas(L, (3,), tr, te, n_control=1), ValueError))


# --------------------------------------------------------------------------- #
# 9. analyse() end to end on a tiny synthetic run directory                    #
# --------------------------------------------------------------------------- #
def check_analyse_end_to_end():
    print("\n-- analyse(): tiny legacy run directory (p = 23 MLP) --")
    cfg = get_config("nanda", p=P, arch="mlp", d_mlp=16, seed=0)
    run_dir = SCRATCH / "lff_test_run"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    set_seed(0)
    model = build_model(cfg)
    torch.save(model.state_dict(), run_dir / "model_final.pt")
    (run_dir / "run.json").write_text(json.dumps({"config": cfg.to_dict(), "git_commit": "test"}))

    out = analyse(run_dir, step=None, key_freqs=[3, 5], key_rule=None, seed=7, n_control=5)
    path = run_dir / "analysis" / "logit_formula_fit" / "final.json"
    check("writes analysis/logit_formula_fit/final.json", path.exists())
    check("writes the npz next to it", path.with_suffix(".npz").exists())
    saved = json.loads(path.read_text())
    check("envelope carries module / run_id / step / checkpoint hash",
          saved["module"] == "logit_formula_fit" and saved["run_id"] == cfg.run_id
          and saved["step"] == "final" and len(saved["checkpoint_sha256"]) == 64)
    check("params echo key set, rule, seed, n_control and the phase-grid size",
          saved["params"]["key_frequencies"] == [3, 5] and saved["params"]["key_rule"] == "explicit"
          and saved["params"]["seed"] == 7 and saved["params"]["n_control"] == 5
          and saved["params"]["phase_grid_points"] == 4 * P)
    check("results hold every formula and the random control",
          tuple(saved["results"]["formulas"]) == FORMULA_IDS
          and saved["results"]["control_random_m"]["n_control"] == 5)
    npz = np.load(path.with_suffix(".npz"))
    check("npz holds coefficients + residual maps per formula and the control draws",
          all(f"coefficients_{fid}" in npz and f"residual_rms_map_{fid}" in npz for fid in FORMULA_IDS)
          and "control_random_r2_test" in npz and "control_random_frequency_sets" in npz)
    check("returned payload names the output path", out["output_path"] == str(path))

    with_rule = analyse(run_dir, key_rule="embedding_top8", seed=0, n_control=3)
    check("embedding_top8 rule recorded with its cap provenance",
          with_rule["params"]["key_rule"] == "embedding_top8"
          and "cap_binding" in with_rule["params"]["key_selection"])
    mul_cfg = get_config("nanda", p=P, arch="mlp", d_mlp=16, seed=0, task="mul")
    mul_dir = SCRATCH / "lff_test_run_mul"
    if mul_dir.exists():
        shutil.rmtree(mul_dir)
    mul_dir.mkdir(parents=True)
    torch.save(model.state_dict(), mul_dir / "model_final.pt")
    (mul_dir / "run.json").write_text(json.dumps({"config": mul_cfg.to_dict()}))
    check("a non-addition task is refused (formulas are defined for a+b-c only)",
          _raises(lambda: analyse(mul_dir, key_freqs=[3]), ValueError))
    shutil.rmtree(run_dir)
    shutil.rmtree(mul_dir)


def main():
    check_normal_equations()
    check_sinusoid_circuit()
    check_scrambled_control()
    check_square_tensor()
    check_gaussian_logits()
    check_random_control_p113()
    check_centering()
    check_key_set_and_errors()
    check_analyse_end_to_end()
    print("\nALL LOGIT_FORMULA_FIT CHECKS PASSED")


if __name__ == "__main__":
    main()
