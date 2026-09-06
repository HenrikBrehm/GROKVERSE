"""Checks for analysis/causal_ablation.py (INTERFACES §9, CAUSAL_ABLATION_PLAN.md).

Run: ``python tests/test_causal_ablation.py`` from ``training/``.

Every input is SYNTHETIC or a freshly initialized model — no trained run is read, so this file runs
before, during and after the study. Each positive check is paired with a control that must NOT pass.

THE FOUR CHECKS INTERFACES §9 NAMES, ON A HAND-BUILT IDEAL CIRCUIT
------------------------------------------------------------------
The circuit is the analytic one of `docs/MLP_MECHANISM_DERIVATION.md` §4.1: rectifying
``cos(w_k a − φ) + cos(w_k b − φ)`` produces a cross term in ``cos(w_k(a+b) − 2φ)``, so a bank of
neurons at spread phases φ with output weights ``cos(w_k c − 2φ)`` sums to ``cos(w_k(a+b−c))``. Two
frequencies of that, plus harmless distractor neurons and a group of neurons whose output weights are
zero, give a model that solves the task exactly and in which the roles are known by construction:

1. `remove_structured` destroys accuracy while the size-matched random control keeps > 0.9;
2. `keep_structured` keeps accuracy;
3. `remove_key_freqs` destroys accuracy;
4. **the no-damage case is representable** — removing the zero-output neurons changes the logits by
   exactly 0.0, and the module labels that `no_damage`, not `necessary`.

Point 4 is the one an implementation is most likely to get wrong by making every ablation look
damaging. `CAUSAL_ABLATION_PLAN.md` §6 fixes it as an ordinary outcome: "removing `C` does no damage
at all → `C` is present but unused — a finding about the structure metric, not about the model."

WHY THE CONTROL IS ASSERTED ON ITS MEAN
---------------------------------------
Gate criterion G4 and plan §6 read the control's *mean* drop (`control_mean_drop < 0.1`), so that is
what is asserted here. Individual draws can dip lower — a random draw occasionally hits several
circuit neurons — and pretending otherwise would be a test tuned to a lucky seed.
"""
from __future__ import annotations

import dataclasses
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grokverse.analysis import causal_ablation as CA  # noqa: E402
from grokverse.analysis.fourier import fourier_basis  # noqa: E402
from grokverse.config import get_config  # noqa: E402
from grokverse.models import build_model  # noqa: E402
from grokverse.seed import set_seed  # noqa: E402

P = 23
CIRCUIT_FREQS = (3, 5)


def check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        raise SystemExit(1)


def _raises(fn) -> bool:
    try:
        fn()
    except (ValueError, TypeError):
        return True
    return False


# --------------------------------------------------------------------------- #
# the hand-built ideal circuit                                                 #
# --------------------------------------------------------------------------- #
def ideal_circuit(freqs=CIRCUIT_FREQS, n_phase: int = 8, copies: int = 4, n_distractor: int = 800,
                  n_unused: int = 300, distractor_amp: float = 0.02, seed: int = 0):
    """A synthetic MLP that solves `(a+b) mod p` through a known Fourier circuit.

    Three neuron groups, in this column order: the **circuit** (``len(freqs)*n_phase*copies``
    neurons), harmless **distractors** (small random weights, small output weights), and **unused**
    neurons (real activations, output weights exactly zero). ``copies`` phase-duplicates keep the
    circuit redundant, so a random draw that happens to hit one or two of its neurons does not break
    it — which is what makes the size-matched control a fair comparison rather than a rigged one.

    The embedding is the identity, so ``effective_curves`` returns exactly the ``u_a``/``u_b`` built
    here (``W_E[:p] @ W_in[:d] = I @ u_a``).
    """
    rng = np.random.default_rng(seed)
    n_circuit = len(freqs) * n_phase * copies
    n = n_circuit + n_distractor + n_unused
    u_a = np.zeros((P, n))
    u_b = np.zeros((P, n))
    W_out = np.zeros((n, P))
    nn = np.arange(P)
    col = 0
    for k in freqs:
        w = 2 * np.pi * k / P
        for j in range(n_phase):
            phi = 2 * np.pi * j / n_phase
            for _ in range(copies):
                u_a[:, col] = np.cos(w * nn - phi)
                u_b[:, col] = np.cos(w * nn - phi)
                W_out[col] = np.cos(w * nn - 2 * phi) / copies
                col += 1
    u_a[:, col:col + n_distractor] = distractor_amp * rng.standard_normal((P, n_distractor))
    u_b[:, col:col + n_distractor] = distractor_amp * rng.standard_normal((P, n_distractor))
    W_out[col:col + n_distractor] = distractor_amp * rng.standard_normal((n_distractor, P))
    col += n_distractor
    u_a[:, col:] = rng.standard_normal((P, n_unused))      # real activations ...
    u_b[:, col:] = rng.standard_normal((P, n_unused))      # ... but W_out stays exactly 0
    cfg = get_config("nanda", p=P, arch="mlp", d_model=P, d_mlp=n, seed=0)
    state = {"W_E": np.vstack([np.eye(P), np.zeros(P)]),
             "W_in": np.vstack([u_a, u_b]), "W_out": W_out,
             "b_in": np.zeros(n), "b_out": np.zeros(P)}
    groups = {"circuit": np.zeros(n, bool), "distractor": np.zeros(n, bool),
              "unused": np.zeros(n, bool)}
    groups["circuit"][:n_circuit] = True
    groups["distractor"][n_circuit:n_circuit + n_distractor] = True
    groups["unused"][n_circuit + n_distractor:] = True
    return cfg, state, groups


def _make_run(tmp: Path, arch: str, seed: int = 0):
    """A freshly initialized model written out as a minimal legacy run directory."""
    kw = dict(d_model=16, d_head=4, n_heads=4) if arch == "transformer" else {}
    cfg = get_config("nanda", p=P, arch=arch, d_mlp=12, seed=seed, **kw)
    set_seed(seed)
    model = build_model(cfg)
    model.eval()
    run_dir = tmp / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps(
        {"config": dataclasses.asdict(cfg), "git_commit": "synthetic-test"}))
    torch.save(model.state_dict(), run_dir / "model_final.pt")
    return cfg, model, run_dir


# --------------------------------------------------------------------------- #
# 1. evaluate / margin / report                                                #
# --------------------------------------------------------------------------- #
def check_evaluate_report():
    print("\n-- evaluate, margin, report (INTERFACES 9 common block) --")
    cfg, state, _ = ideal_circuit(n_distractor=20, n_unused=10, copies=1)
    pieces = CA._mlp_pieces(state, cfg)
    base = CA.evaluate(pieces["base_logits"], cfg)
    check("the injected circuit solves the task on both splits",
          base["train_acc"] == 1.0 and base["test_acc"] == 1.0)
    check("evaluate accepts a callable as well as an array (INTERFACES 9 spelling)",
          CA.evaluate(lambda: pieces["base_logits"], cfg)["test_acc"] == base["test_acc"])
    check("a wrong-shaped logit grid is rejected",
          _raises(lambda: CA.evaluate(np.zeros((P, P, 3)), cfg)))

    check("margin of the correct circuit is positive", CA.margin(base["logits"], P) > 0)
    flat = np.zeros((P, P, P))
    check("margin of an all-equal grid is 0", abs(CA.margin(flat, P)) < 1e-12)

    same = CA.report(base, base, cfg, 0)
    check("report of a model against itself has zero deltas",
          all(abs(same[f"delta_{k}"]) < 1e-12
              for k in ("train_loss", "test_loss", "train_acc", "test_acc")))
    check("...zero logit change and zero margin change",
          same["mean_abs_logit_change"] == 0.0 and abs(same["margin_change"]) < 1e-12)
    check("...and zero test_accuracy_drop", abs(same["test_accuracy_drop"]) < 1e-12)
    check("per-class accuracy change is per class and summarized",
          same["_per_class_accuracy_change"].shape == (P,)
          and same["per_class_accuracy_change"]["n"] == P)

    zeroed = CA.evaluate(np.zeros((P, P, P)), cfg)
    rep = CA.report(base, zeroed, cfg, 5)
    check("destroying the model gives a large positive test_accuracy_drop",
          rep["test_accuracy_drop"] > 0.9)
    check("absolute AND relative changes are both reported (plan 2)",
          "delta_test_acc" in rep and "relative_test_acc" in rep)
    check("n_components_removed is carried through", rep["n_components_removed"] == 5)
    check("a relative change against a zero baseline is None, never nan or inf",
          CA.report({"train_loss": 0.0, "test_loss": 1.0, "train_acc": 1.0, "test_acc": 1.0,
                     "logits": base["logits"]}, zeroed, cfg)["relative_train_loss"] is None)


# --------------------------------------------------------------------------- #
# 2. the four INTERFACES §9 checks on the ideal circuit                        #
# --------------------------------------------------------------------------- #
def check_ideal_circuit_ablations():
    print("\n-- ideal circuit: necessity, sufficiency, and the no-damage case --")
    cfg, state, groups = ideal_circuit()
    pieces = CA._mlp_pieces(state, cfg)
    base = CA.evaluate(pieces["base_logits"], cfg)
    n = pieces["n_neurons"]
    S = groups["circuit"]
    check(f"the circuit solves the task exactly (n={n}, |S|={int(S.sum())})",
          base["train_acc"] == 1.0 and base["test_acc"] == 1.0)

    removed = CA.evaluate(CA._logits_from_neuron_mask(pieces, ~S, P), cfg)
    check(f"remove_structured destroys accuracy ({removed['test_acc']:.4f} <= 0.1)",
          removed["test_acc"] <= 0.1)
    kept = CA.evaluate(CA._logits_from_neuron_mask(pieces, S, P), cfg)
    check(f"keep_structured keeps accuracy ({kept['test_acc']:.4f} >= 0.9)",
          kept["test_acc"] >= 0.9)

    rng = np.random.default_rng(1)
    ctrl = []
    for _ in range(30):
        m = np.ones(n, bool)
        m[rng.choice(n, size=int(S.sum()), replace=False)] = False
        ctrl.append(CA.evaluate(CA._logits_from_neuron_mask(pieces, m, P), cfg)["test_acc"])
    check(f"the size-matched random control keeps > 0.9 on average ({np.mean(ctrl):.4f})",
          float(np.mean(ctrl)) > 0.9)
    check("...so the plan's rule can separate the two at all",
          (base["test_acc"] - removed["test_acc"]) >= CA.NECESSARY_TEST_ACC_DROP
          and (base["test_acc"] - float(np.mean(ctrl))) < CA.NECESSARY_CONTROL_MAX_DROP)

    for mode, expect in (("remove", True), ("keep", False)):
        lg = CA._logits_from_curves(
            CA.filter_curve_frequencies(pieces["curves"]["u_a"], P, CIRCUIT_FREQS, mode),
            CA.filter_curve_frequencies(pieces["curves"]["u_b"], P, CIRCUIT_FREQS, mode), pieces, P)
        acc = CA.evaluate(lg, cfg)["test_acc"]
        if expect:
            check(f"remove_key_freqs destroys accuracy ({acc:.4f} <= 0.1)", acc <= 0.1)
        else:
            check(f"keep_key_freqs retains accuracy ({acc:.4f} >= 0.9)", acc >= 0.9)

    # --- THE NO-DAMAGE CASE, which must be representable --------------------------------------
    U = groups["unused"]
    unused_logits = CA._logits_from_neuron_mask(pieces, ~U, P)
    check(f"removing the {int(U.sum())} zero-output neurons changes the logits by exactly 0",
          float(np.abs(unused_logits - pieces["base_logits"]).max()) == 0.0)
    obs = CA.report(base, CA.evaluate(unused_logits, cfg), cfg, int(U.sum()))
    entry = CA.with_control(obs, [obs] * 5)
    check("...and the module labels that no_damage, not necessary",
          entry["reading"]["no_damage"] and not entry["reading"]["necessary_by_plan_rule"])
    check("CONTROL: the circuit removal is NOT labelled no_damage",
          not CA.with_control(CA.report(base, removed, cfg, int(S.sum())),
                              [obs] * 5)["reading"]["no_damage"])


# --------------------------------------------------------------------------- #
# 3. frequency filtering, phase scrambling, control draws                      #
# --------------------------------------------------------------------------- #
def check_frequency_filtering():
    print("\n-- filter_curve_frequencies --")
    rng = np.random.default_rng(0)
    Y = rng.standard_normal((P, 7))
    F, _ = fourier_basis(P)
    dc = F.T @ np.vstack([(F @ Y)[0], np.zeros((P - 1, Y.shape[1]))])
    keys = [3, 5]
    kept = CA.filter_curve_frequencies(Y, P, keys, "keep")
    removed = CA.filter_curve_frequencies(Y, P, keys, "remove")
    check("keep + remove reconstructs the curve plus its constant (keep carries the DC)",
          float(np.abs(kept + removed - (Y + dc)).max()) < 1e-10)
    allk = list(range(1, (P - 1) // 2 + 1))
    check("keeping every frequency is the identity",
          float(np.abs(CA.filter_curve_frequencies(Y, P, allk, "keep") - Y).max()) < 1e-10)
    check("removing every frequency leaves only the constant",
          float(np.abs(CA.filter_curve_frequencies(Y, P, allk, "remove") - dc).max()) < 1e-10)
    pure = np.cos(2 * np.pi * 3 * np.arange(P) / P)[:, None]
    check("removing a curve's own frequency annihilates it",
          float(np.abs(CA.filter_curve_frequencies(pure, P, [3], "remove")).max()) < 1e-10)
    check("CONTROL: removing a different frequency leaves it untouched",
          float(np.abs(CA.filter_curve_frequencies(pure, P, [4], "remove") - pure).max()) < 1e-10)
    check("an unknown mode raises",
          _raises(lambda: CA.filter_curve_frequencies(Y, P, keys, "delete")))


def check_phase_scramble():
    print("\n-- phase_scramble (the waveform-substitution control) --")
    rng = np.random.default_rng(0)
    Y = np.stack([np.cos(2 * np.pi * 3 * np.arange(P) / P - phi) for phi in (0.0, 1.0, 2.0)], axis=1)
    S = CA.phase_scramble(Y, np.random.default_rng(0))
    check("scrambling preserves every column's multiset of values (it is a rotation)",
          all(np.allclose(np.sort(Y[:, j]), np.sort(S[:, j])) for j in range(Y.shape[1])))
    check("...and therefore the amplitude exactly",
          float(np.abs(np.abs(S).max(axis=0) - np.abs(Y).max(axis=0)).max()) < 1e-12)
    check("...but changes the phase of at least one column",
          float(np.abs(S - Y).max()) > 1e-6)
    check("it is deterministic under its seed",
          np.array_equal(S, CA.phase_scramble(Y, np.random.default_rng(0))))
    check("CONTROL: a different seed gives a different scramble",
          not np.array_equal(S, CA.phase_scramble(Y, np.random.default_rng(7))))


def check_control_draws():
    print("\n-- size-matched control draws --")
    rng = np.random.default_rng(0)
    sets = CA.random_frequency_sets(P, [3, 5], 10, rng)
    check("random frequency sets match the key set's cardinality",
          len(sets) == 10 and all(len(s) == 2 for s in sets))
    check("...and are drawn from the complement of the key set",
          all(3 not in s and 5 not in s for s in sets))
    check("an empty key set yields no control sets",
          CA.random_frequency_sets(P, [], 5, rng) == [])
    masks = CA.random_neuron_masks(50, 7, 4, np.random.default_rng(0))
    check("random neuron masks select exactly the requested count",
          len(masks) == 4 and all(int(m.sum()) == 7 for m in masks))
    check("a size larger than the population yields no masks",
          CA.random_neuron_masks(5, 9, 3, rng) == [])
    check("draws are reproducible under the seed",
          np.array_equal(CA.random_neuron_masks(50, 7, 1, np.random.default_rng(0))[0], masks[0]))


# --------------------------------------------------------------------------- #
# 4. with_control: the z statistic and its degenerate case                     #
# --------------------------------------------------------------------------- #
def check_with_control():
    print("\n-- with_control: z, its degenerate case, and the plan's 6 labels --")
    def rep(drop, acc):
        return {"test_accuracy_drop": drop, "test_acc": acc, "mean_abs_logit_change": 0.0,
                "_per_class_accuracy_change": np.zeros(P)}
    rng = np.random.default_rng(0)
    ctrl = [rep(float(d), 1.0 - float(d)) for d in rng.normal(0.02, 0.01, size=40)]
    e = CA.with_control(rep(0.95, 0.05), ctrl)
    check("a large observed damage against a tight control gives a large z", e["z"] > 10)
    check("...and is labelled necessary", e["reading"]["necessary_by_plan_rule"])
    check("the control distribution is summarized, not dumped",
          set(e["control"]["test_accuracy_drop"]) >= {"mean", "std", "q05", "q95"})

    flat = [rep(0.0, 1.0) for _ in range(20)]
    z0 = CA.with_control(rep(0.95, 0.05), flat)
    check("a zero-spread control gives z = None, not inf (common._jsonable forbids non-finite)",
          z0["z"] is None and "zero spread" in z0["z_undefined_reason"])
    check("...and the z-condition falls back to exceeding every control",
          z0["exceeds_all_controls"] and z0["reading"]["necessary_by_plan_rule"])
    check("CONTROL: a zero-spread control that is NOT exceeded is not necessary",
          not CA.with_control(rep(0.0, 1.0), flat)["reading"]["necessary_by_plan_rule"])

    heavy = [rep(float(d), 1.0 - float(d)) for d in rng.normal(0.90, 0.02, size=40)]
    e2 = CA.with_control(rep(0.95, 0.05), heavy)
    check("a control that does comparable damage is NOT necessary (plan 6, 'removing more removes "
          "more')", not e2["reading"]["necessary_by_plan_rule"])
    check("an ablation with no control at all reports z = None with its reason",
          CA.with_control(rep(0.1, 0.9), [])["z"] is None)
    check("sufficiency reads the retained accuracy, not the drop",
          CA.with_control(rep(0.0, 0.95), ctrl)["reading"]["sufficient_by_plan_rule"])
    check("the [AI-PROPOSED] threshold provenance travels with every label",
          "HUMAN_DECISIONS" in e["reading"]["threshold_status"])


# --------------------------------------------------------------------------- #
# 5. the pruning sweep                                                         #
# --------------------------------------------------------------------------- #
def check_pruning_curve():
    print("\n-- cumulative pruning sweep (Doshi protocol + our random-order control) --")
    cfg, state, groups = ideal_circuit(n_distractor=40, n_unused=10, copies=2)
    pieces = CA._mlp_pieces(state, cfg)
    base = CA.evaluate(pieces["base_logits"], cfg)
    n = pieces["n_neurons"]
    order = np.arange(n)
    curve = CA._pruning_curve_generic(pieces, order, cfg)
    check("the sweep is reported ascending in n_pruned",
          curve["n_pruned"] == sorted(curve["n_pruned"]))
    check("pruning nothing reproduces the unablated model",
          curve["n_pruned"][0] == 0 and abs(curve["test_acc"][0] - base["test_acc"]) < 1e-12)
    check("pruning everything leaves only the constant term",
          curve["n_pruned"][-1] == n
          and abs(curve["test_acc"][-1]
                  - CA.evaluate(np.broadcast_to(state["b_out"], (P, P, P)), cfg)["test_acc"]) < 1e-12)
    check("every requested fraction is evaluated",
          len(curve["n_pruned"]) == len({int(round(f * n)) for f in CA.PRUNING_FRACTIONS}))
    mid = CA._pruning_curve_generic(pieces, order, cfg, fractions=(0.5,))["test_acc"][0]
    keep = np.zeros(n, bool)
    keep[order[int(round(0.5 * n)):]] = True
    check("a sweep point equals the direct evaluation of that same neuron set",
          abs(mid - CA.evaluate(CA._logits_from_neuron_mask(pieces, keep, P), cfg)["test_acc"]) < 1e-12)


# --------------------------------------------------------------------------- #
# 6. transformer ablations                                                     #
# --------------------------------------------------------------------------- #
def check_transformer_ablations(tmp: Path):
    print("\n-- transformer ablations (plan 5) --")
    from grokverse.analysis import transformer_mechanism as TM

    cfg, model, _ = _make_run(tmp, "transformer", seed=3)
    st = {k: v.detach().numpy().astype(np.float64) for k, v in model.state_dict().items()}
    st["W_V"] = st["W_V"].copy()
    st["W_O"] = st["W_O"].copy()
    st["W_V"][0] = 0.0
    st["W_O"][0] = 0.0                                    # head 0 now contributes nothing
    decomp = TM.forward_decomposition(st, cfg)
    n = cfg.d_mlp
    masks = {"sets": {"primary_family_0.50": np.zeros(n, bool)}, "alive": np.ones(n, bool),
             "primary": "primary_family_0.50", "source": "test"}
    masks["sets"]["primary_family_0.50"][: n // 2] = True
    res, arrays = CA.transformer_ablations(st, cfg, masks, [3, 5], decomp, seed=0, n_control=3)

    check("every plan-5 ablation id is present",
          {"remove_structured_neurons", "keep_structured_neurons",
           "remove_key_freqs_from_embedding", "keep_key_freqs_in_embedding",
           "fix_attention_to_mean", "remove_key_subspace_from_residual",
           "restricted_circuit_only", "ipr_ranked_pruning"} <= set(res))
    check("every head is ablated in both modes",
          all(f"ablate_head_{h}__{m}" in res
              for h in range(cfg.n_heads) for m in ("zero", "mean")))
    h0 = res["ablate_head_0__zero"]["observed"]
    check("ablating the all-zero head changes NOTHING (INTERFACES 9 test)",
          abs(h0["test_accuracy_drop"]) < 1e-12 and h0["mean_abs_logit_change"] < 1e-12)
    check("...and is labelled no_damage",
          res["ablate_head_0__zero"]["reading"]["no_damage"])
    live = [res[f"ablate_head_{h}__zero"]["observed"]["mean_abs_logit_change"]
            for h in range(1, cfg.n_heads)]
    check("CONTROL: at least one live head DOES change the logits when zeroed",
          max(live) > 1e-9)
    check("fix_attention_to_mean declares that it has no size-matched control (plan 5)",
          res["fix_attention_to_mean"]["z"] is None
          and "structural test" in res["fix_attention_to_mean"]["note"])
    check("the embedding ablation records that attention is recomputed under it",
          "attention is recomputed" in res["remove_key_freqs_from_embedding"]["note"])
    check("per-class accuracy changes reach the npz arrays",
          any(k.endswith("__per_class_accuracy_change") for k in arrays))
    prot = res["ipr_ranked_pruning"]["protocol"].upper()
    check("the IPR pruning block carries our random-order control, which Doshi's protocol has not",
          "random_order_control" in res["ipr_ranked_pruning"]
          and ("OURS" in prot or "NEW" in prot))
    check("...and both ranking directions are swept",
          {"lowest_ipr_first", "highest_ipr_first"} <= set(res["ipr_ranked_pruning"]))


# --------------------------------------------------------------------------- #
# 7. gate criterion G4                                                         #
# --------------------------------------------------------------------------- #
def check_gate_g4():
    print("\n-- gate criterion G4 (PREREGISTRATION 5) --")
    def entry(nec, suf):
        return {"reading": {"necessary_by_plan_rule": nec, "sufficient_by_plan_rule": suf}}
    passing = {"remove_structured": entry(True, False), "keep_structured": entry(False, True),
               "remove_key_freqs_from_curves": entry(True, False)}
    check("all three conditions met -> G4 passes", CA._gate_g4(passing, "d")["g4_passed"] is True)
    failing = dict(passing, remove_structured=entry(False, False))
    check("one condition unmet -> G4 fails", CA._gate_g4(failing, "d")["g4_passed"] is False)
    partial = {"remove_structured": entry(True, False)}
    g = CA._gate_g4(partial, "d")
    check("a missing ablation makes G4 null, never a silent False", g["g4_passed"] is None)
    check("...and says which ids it did find", g["ids_used"]["remove_structured"] == "remove_structured")
    txf = {"remove_structured_neurons": entry(True, False),
           "keep_structured_neurons": entry(False, True),
           "remove_key_subspace_from_residual": entry(True, False)}
    gt = CA._gate_g4(txf, "d")
    check("the transformer's own ids are recognised", gt["g4_passed"] is True)
    check("...and the id actually used is named",
          gt["ids_used"]["remove_key_freqs"] == "remove_key_subspace_from_residual")
    check("the aggregation caveat travels with the number", "8 of 10 seeds" in gt["note"])


# --------------------------------------------------------------------------- #
# 8. analyse(): the run-level output contract                                  #
# --------------------------------------------------------------------------- #
def check_analyse(tmp: Path):
    print("\n-- analyse(): run-level entry point + output contract --")
    for arch in ("mlp", "transformer"):
        _cfg, _model, run_dir = _make_run(tmp, arch, seed=5)
        payload, path = CA.analyse(run_dir, None, CA.PRIMARY_KEY_RULE, 0, n_control=3)
        check(f"{arch}: the result lands at analysis/causal_ablation/final.json",
              path == run_dir / "analysis" / "causal_ablation" / "final.json" and path.exists())
        check(f"{arch}: the npz is written next to it", path.with_suffix(".npz").exists())
        on_disk = json.loads(path.read_text())
        check(f"{arch}: the provenance envelope is complete",
              all(on_disk.get(k) for k in ("module", "module_version", "run_id", "arch", "p",
                                           "created_utc")) and on_disk["arch"] == arch)
        res = on_disk["results"]
        check(f"{arch}: baseline, ablations and the G4 verdict are all reported",
              {"baseline", "ablations", "gate_criterion_g4", "n_structured"} <= set(res))
        check(f"{arch}: the no-retraining rule is recorded in params",
              "no retraining" in on_disk["params"]["rules"])
        check(f"{arch}: the interpretation thresholds carry their [AI-PROPOSED] status",
              "AI-PROPOSED" in on_disk["params"]["interpretation_thresholds"]["status"])
        check(f"{arch}: the structured-mask source is recorded",
              "structured_mask_source" in on_disk["params"])
        check(f"{arch}: the result is labelled measurement-only",
              on_disk["params"]["status"].startswith("MEASUREMENT ONLY"))
        check(f"{arch}: the sensitivity definitions are ablated too (plan 9)",
              any(k.endswith("sensitivity_family_0.30") for k in res["ablations"]))
    check("an unknown structured definition is refused, naming what is available",
          _raises(lambda: CA.analyse(run_dir, None, CA.PRIMARY_KEY_RULE, 0,
                                     structured_definition="no_such_definition", n_control=2)))


# --------------------------------------------------------------------------- #
# 11. graded structured-neuron ablation (PREREGISTRATION 14)                   #
# --------------------------------------------------------------------------- #
def _graded_setup():
    """The ideal circuit as a `logits_fn(keep_mask)` plus its known neuron groups."""
    cfg, state, groups = ideal_circuit()
    pieces = CA._mlp_pieces(state, cfg)
    base = CA.evaluate(pieces["base_logits"], cfg)
    n = pieces["n_neurons"]
    alive = np.ones(n, dtype=bool)
    return cfg, groups, base, alive, (lambda keep: CA._logits_from_neuron_mask(pieces, keep, cfg.p)), n


def _discriminable(entry) -> bool:
    """PREREGISTRATION 14.5: exceeds every control AND z >= 3."""
    z = entry.get("z")
    return bool(entry.get("exceeds_all_controls")) and z is not None and z >= CA.NECESSARY_Z


def check_graded_group_size():
    check("fraction->cardinality rounds half UP, not to even (0.05 of 50 == 3)",
          CA.graded_group_size(0.05, 50) == 3)
    check("a fraction that would round to zero still selects one neuron",
          CA.graded_group_size(0.001, 100) == 1)
    check("1 % of 1164 live neurons is 12", CA.graded_group_size(0.01, 1164) == 12)
    check("50 % of 1164 live neurons is 582", CA.graded_group_size(0.5, 1164) == 582)


def check_graded_ranking_is_nested():
    """The top-n sets must be nested, and must sit inside the B1 set while n <= |B1|."""
    n = 20
    score = np.linspace(0.0, 1.0, n)
    alive = np.ones(n, dtype=bool)
    o5 = CA.graded_top_indices(score, alive, 5)
    o10 = CA.graded_top_indices(score, alive, 10)
    check("the top-5 is a subset of the top-10 (nested family)", set(o5) <= set(o10))
    check("the top-5 really are the five highest scores", sorted(o5) == [15, 16, 17, 18, 19])
    dead = alive.copy()
    dead[19] = False
    check("dead neurons are never ranked", 19 not in set(CA.graded_top_indices(score, dead, 5)))
    tied = np.zeros(n)
    check("ties break deterministically by index", list(CA.graded_top_indices(tied, alive, 3)) == [0, 1, 2])


def check_graded_ablation_separates_a_known_circuit():
    cfg, groups, base, alive, logits_fn, n = _graded_setup()
    circuit_first = np.where(groups["circuit"], 1.0,
                             np.where(groups["distractor"], 0.5, 0.0))
    block, arrays = CA.graded_ablation(base, logits_fn, alive, circuit_first, cfg,
                                       seed=0, n_control=20, fractions=(0.05,))
    entry = block["by_fraction"]["0.05"]
    check(f"n_selected is the size-matched cardinality ({entry['n_selected']})",
          entry["n_selected"] == CA.graded_group_size(0.05, n))
    check("every random control group has EXACTLY the selected cardinality",
          entry["control_group_sizes_unique"] == [entry["n_selected"]])
    check("the control distribution has the requested number of draws",
          entry["remove_top"]["control"]["n"] == 20)
    rt = entry["remove_top"]
    check(f"removing the top {entry['n_selected']} circuit neurons does real damage "
          f"(drop {rt['observed']['test_accuracy_drop']:.4f})",
          rt["observed"]["test_accuracy_drop"] >= 0.5)
    check(f"...more than EVERY size-matched random group "
          f"(control mean {rt['control']['test_accuracy_drop']['mean']:.4f})",
          _discriminable(rt))
    check("keep_only_top is reported in the sufficiency direction too",
          "keep_only_top" in entry and "observed" in entry["keep_only_top"])


def check_graded_ablation_control_does_not_separate():
    """CONTROL: a structure-blind ranking must NOT come out discriminable."""
    cfg, groups, base, alive, logits_fn, n = _graded_setup()
    blind = np.where(groups["distractor"], 1.0, 0.0)          # ranks the harmless neurons first
    block, _ = CA.graded_ablation(base, logits_fn, alive, blind, cfg,
                                  seed=0, n_control=20, fractions=(0.05,))
    rt = block["by_fraction"]["0.05"]["remove_top"]
    check(f"a distractor-first ranking does NOT separate from its control "
          f"(drop {rt['observed']['test_accuracy_drop']:.4f})",
          not _discriminable(rt))


def check_graded_ablation_is_deterministic():
    cfg, groups, base, alive, logits_fn, n = _graded_setup()
    score = np.where(groups["circuit"], 1.0, 0.0)
    kw = dict(cfg=cfg, n_control=8, fractions=(0.05,))
    a, aa = CA.graded_ablation(base, logits_fn, alive, score, seed=0, **kw)
    b, bb = CA.graded_ablation(base, logits_fn, alive, score, seed=0, **kw)
    c, cc = CA.graded_ablation(base, logits_fn, alive, score, seed=1, **kw)
    key = "graded__0.05__remove_top__control_drops"
    check("the same seed reproduces the control draws exactly", np.array_equal(aa[key], bb[key]))
    check("a different seed draws different controls", not np.array_equal(aa[key], cc[key]))
    check("the observed value does not depend on the control seed",
          a["by_fraction"]["0.05"]["remove_top"]["observed"]["test_acc"]
          == c["by_fraction"]["0.05"]["remove_top"]["observed"]["test_acc"])


def check_structured_mask_order_is_deterministic(tmp: Path):
    """The key order of `masks["sets"]` must not depend on per-process string hashing.

    Both ablation functions iterate this dict while drawing from ONE shared rng, so an unordered
    source made which 50 size-matched control draws went to which definition change from process to
    process. Observed values were unaffected; control means moved by up to 0.0034 between runs of
    identical code on identical inputs (HUMAN_DECISIONS D8, LABBOOK 109). `wanted` is ordered, so
    the returned order must equal it.
    """
    run_dir = tmp / "order_probe"
    (run_dir / "analysis" / "mlp_mechanism").mkdir(parents=True, exist_ok=True)
    n = 8
    names = ("primary_family_0.50",) + CA.SENSITIVITY_DEFINITIONS
    np.savez(run_dir / "analysis" / "mlp_mechanism" / "stepX.npz",
             **{f"mask__{d}": np.ones(n, bool) for d in names},
             mask__alive=np.ones(n, bool))
    got = CA.resolve_structured_masks(run_dir, "stepX", "mlp", {}, None, [],
                                      "primary_family_0.50", 0)
    check(f"the npz path returns the definitions in `wanted` order, not set order ({list(got['sets'])})",
          list(got["sets"]) == list(names))
    check("...and it really came from the npz, not the recompute fallback",
          got["source"].endswith("stepX.npz"))


# --------------------------------------------------------------------------- #
def main():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        check_evaluate_report()
        check_ideal_circuit_ablations()
        check_frequency_filtering()
        check_phase_scramble()
        check_control_draws()
        check_with_control()
        check_pruning_curve()
        check_transformer_ablations(tmp)
        check_gate_g4()
        check_analyse(tmp)
        check_graded_group_size()
        check_graded_ranking_is_nested()
        check_graded_ablation_separates_a_known_circuit()
        check_graded_ablation_control_does_not_separate()
        check_graded_ablation_is_deterministic()
        check_structured_mask_order_is_deterministic(tmp)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
