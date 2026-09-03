"""Checks for analysis/transformer_mechanism.py (INTERFACES §6).

Run: ``python tests/test_transformer_mechanism.py`` from ``training/``.

Every input is SYNTHETIC or a freshly initialized model — no trained run is read, so this
file runs before, during and after the study. Each positive check is paired with a control
that must NOT pass.

THE FOUR CHECKS INTERFACES §6 NAMES, AND WHY EACH ONE BITES
------------------------------------------------------------
1. ``W_Q = W_K = 0`` makes attention exactly uniform, so the grid-mean attention IS the
   attention on every input and the effective curves must reproduce the true
   pre-activation *exactly* — ``additivity_r2 == 1``. This is the only configuration in
   which the module's one approximation is not an approximation, so it is the sharp test
   of the effective-curve formula. Paired with a trained-shape control (random ``W_Q``,
   ``W_K``) whose ``additivity_r2`` must NOT be 1, which is what proves the check is
   measuring the attention dependence rather than always returning 1.
2. ``forward_decomposition`` must reproduce the model's own ``logits_last``. Against a
   float32 model the agreement is ~1e-4 (float32 accumulation); against the same model in
   float64 it is ~1e-13, and BOTH are asserted so a real regression cannot hide inside the
   loose tolerance.
3. Ablating an all-zero head must change nothing, and ablating a head that carries the
   whole circuit must change everything — the null alone would pass for a no-op ablation.
4. The key-subspace share of a synthetic ``cos(w_k(a+b))`` field must be 1 at ``k`` and 0
   at every other frequency.
"""
from __future__ import annotations

import dataclasses
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grokverse.analysis import transformer_mechanism as TM  # noqa: E402
from grokverse.config import get_config  # noqa: E402
from grokverse.models import build_model  # noqa: E402
from grokverse.seed import set_seed  # noqa: E402

P = 23                       # prime, so the odd harmonics of any k stay distinct
D_MLP = 16


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
# builders                                                                     #
# --------------------------------------------------------------------------- #
def _make_run(tmp: Path, seed: int = 0, **over):
    """A freshly initialized transformer written out as a minimal legacy run directory."""
    cfg = get_config("nanda", p=P, arch="transformer", d_model=16, d_head=4, n_heads=4,
                     d_mlp=D_MLP, seed=seed, **over)
    set_seed(seed)
    model = build_model(cfg)
    model.eval()
    run_dir = tmp / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps(
        {"config": dataclasses.asdict(cfg), "git_commit": "synthetic-test"}))
    torch.save(model.state_dict(), run_dir / "model_final.pt")
    state = {k: v.detach().numpy().astype(np.float64) for k, v in model.state_dict().items()}
    return cfg, model, state, run_dir


def _zero_qk(state: dict) -> dict:
    """Zero the query and key maps: every score is 0, so the softmax is exactly uniform."""
    out = {k: v.copy() for k, v in state.items()}
    out["W_Q"] = np.zeros_like(out["W_Q"])
    out["W_K"] = np.zeros_like(out["W_K"])
    return out


def sum_field(p: int, k: int, n_ch: int = 3, phase: float = 0.0) -> np.ndarray:
    """``[p, p, n_ch]`` field ``cos(w_k(a+b) - phase)`` — pure sum-direction power at k."""
    a = np.arange(p)[:, None]
    b = np.arange(p)[None, :]
    return np.repeat((np.cos(2 * np.pi * k * (a + b) / p - phase))[:, :, None], n_ch, axis=2)


# --------------------------------------------------------------------------- #
# 1. forward decomposition                                                     #
# --------------------------------------------------------------------------- #
def check_forward(tmp: Path):
    print("\n-- forward_decomposition (INTERFACES 6) --")
    cfg, model, state, _ = _make_run(tmp, seed=0)
    d = TM.forward_decomposition(state, cfg)

    toks = torch.tensor(TM._token_grid(cfg))
    with torch.no_grad():
        ref32 = model.logits_last(toks).numpy()[:, :P].reshape(P, P, P)
        ref64 = model.double().logits_last(toks).numpy()[:, :P].reshape(P, P, P)
    check("logits match the float32 model to 1e-4 (INTERFACES 6 tolerance)",
          float(np.abs(d["logits"] - ref32).max()) < 1e-4)
    check("...and match the SAME model in float64 to 1e-10 (so 1e-4 is float32, not error)",
          float(np.abs(d["logits"] - ref64).max()) < 1e-10)
    check("direct path + mlp path == logits exactly",
          float(np.abs(d["direct_path_logits"] + d["mlp_path_logits"] - d["logits"]).max()) == 0.0)
    check("shapes follow INTERFACES 6",
          d["attn"].shape == (P, P, cfg.n_heads, 3)
          and d["hidden"].shape == (P, P, cfg.d_mlp)
          and d["r_pre"].shape == (P, P, cfg.d_model)
          and d["logits"].shape == (P, P, P))
    check("read-out attention is a distribution over the 3 positions",
          np.allclose(d["attn"].sum(axis=-1), 1.0) and (d["attn"] >= 0).all())
    check("hidden is a ReLU output (no negative entries)", (d["hidden"] >= 0).all())

    # accepting an nn.Module is the INTERFACES 6 spelling; both must agree
    dm = TM.forward_decomposition(model.float(), cfg)
    check("forward_decomposition accepts an nn.Module as well as a state dict",
          float(np.abs(dm["logits"] - d["logits"]).max()) < 1e-4)
    check("a wrong-shaped attention override is rejected",
          _raises(lambda: TM.forward_decomposition(state, cfg, attn=np.zeros((P, P, 2, 3)))))


# --------------------------------------------------------------------------- #
# 2. effective curves and additivity                                           #
# --------------------------------------------------------------------------- #
def check_effective_curves(tmp: Path):
    print("\n-- effective_curves + additivity (the module's one approximation) --")
    cfg, _model, state, _ = _make_run(tmp, seed=1)

    # (a) uniform attention -> the mean-attention curves are EXACT
    st0 = _zero_qk(state)
    d0 = TM.forward_decomposition(st0, cfg)
    check("W_Q = W_K = 0 gives exactly uniform attention (1/3 on each position)",
          float(np.abs(d0["attn"] - 1.0 / 3.0).max()) < 1e-12)
    abar0 = d0["attn"].reshape(P * P, cfg.n_heads, 3).mean(axis=0)
    cur0 = TM.effective_curves(st0, cfg, abar0)
    add0 = TM.additivity(d0, cur0, cfg, st0)
    check("uniform attention: additivity_r2 == 1 for every neuron (INTERFACES 6 test)",
          float(np.nanmin(add0["additivity_r2"])) > 1 - 1e-10)
    check("uniform attention: no neuron has a constant pre-activation (the check is live)",
          add0["n_constant_preactivation"] == 0)

    # the reconstruction is not merely correlated -- it is the pre-activation itself
    a_idx, b_idx = np.repeat(np.arange(P), P), np.tile(np.arange(P), P)
    recon = cur0["u_a"][a_idx] + cur0["u_b"][b_idx] + cur0["b_in"]
    true_pre = d0["r_pre"].reshape(P * P, cfg.d_model) @ st0["W_in"]
    check("uniform attention: the reconstruction equals the true pre-activation to 1e-10",
          float(np.abs(recon - true_pre).max()) < 1e-10)
    check("uniform attention: ReLU of the reconstruction IS the true hidden",
          float(np.abs(np.maximum(recon, 0.0)
                       - d0["hidden"].reshape(P * P, cfg.d_mlp)).max()) < 1e-10)

    # (b) CONTROL: with real Q/K the attention is input dependent, so r2 must NOT be 1
    d1 = TM.forward_decomposition(state, cfg)
    abar1 = d1["attn"].reshape(P * P, cfg.n_heads, 3).mean(axis=0)
    add1 = TM.additivity(d1, TM.effective_curves(state, cfg, abar1), cfg, state)
    check("CONTROL: input-dependent attention does NOT give additivity_r2 == 1",
          float(np.nanmax(add1["additivity_r2"])) < 1 - 1e-6)
    check("CONTROL: the attention actually varies across inputs",
          float(d1["attn"].std(axis=(0, 1)).max()) > 1e-6)

    # (c) shapes and the MLP-comparable contract
    check("the curves carry exactly the MLP's keys and shapes",
          set(cur0) == {"u_a", "u_b", "out", "b_in"}
          and cur0["u_a"].shape == (P, cfg.d_mlp) and cur0["out"].shape == (P, cfg.d_mlp)
          and cur0["b_in"].shape == (cfg.d_mlp,))
    check("out[c, f] is what neuron f adds to the logit of class c",
          float(np.abs(cur0["out"].T - (state["W_out"] @ state["W_U"][:, :P])).max()) < 1e-12)
    check("a wrong-shaped attn_mean is rejected",
          _raises(lambda: TM.effective_curves(state, cfg, np.zeros((cfg.n_heads, 2)))))


# --------------------------------------------------------------------------- #
# 3. key-frequency subspace variance                                           #
# --------------------------------------------------------------------------- #
def check_key_subspace():
    print("\n-- key_subspace_variance (sum-direction projection + size-matched null) --")
    k = 5
    T = sum_field(P, k)
    res = TM.key_subspace_variance(T, P, [k], n_control=20, seed=0)
    check(f"a pure cos(w_{k}(a+b)) field has share 1 at k={k}",
          abs(res["share_total"] - 1.0) < 1e-10)
    others = [j for j in range(1, (P - 1) // 2 + 1) if j != k]
    shares = [TM.key_subspace_variance(T, P, [j], n_control=0)["share_total"] for j in others]
    check("...and share 0 at every other frequency",
          max(abs(s) for s in shares) < 1e-10)
    check("the size-matched random control is ~0 for that field",
          res["control_mean"] < 1e-9 and res["n_control"] == 20)

    # a phase shift must not change the share: the subspace is 2-D (cos AND sin)
    check("the share is phase invariant (the cos/sin pair is projected, not just cos)",
          abs(TM.key_subspace_variance(sum_field(P, k, phase=1.1), P, [k],
                                       n_control=0)["share_total"] - 1.0) < 1e-10)

    # CONTROL: a DIFFERENCE field must not be picked up by a SUM projection
    a, b = np.arange(P)[:, None], np.arange(P)[None, :]
    diff = np.repeat(np.cos(2 * np.pi * k * (a - b) / P)[:, :, None], 3, axis=2)
    check("CONTROL: a cos(w_k(a-b)) field has ~0 sum-direction share",
          abs(TM.key_subspace_variance(diff, P, [k], n_control=0)["share_total"]) < 1e-10)

    # a constant offset must not inflate the denominator
    check("a large constant offset does not change the share (DC is excluded)",
          abs(TM.key_subspace_variance(T + 50.0, P, [k], n_control=0)["share_total"] - 1.0) < 1e-10)

    # random noise: the share should sit near the control, not near 1
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((P, P, 4))
    rn = TM.key_subspace_variance(noise, P, [k, 7], n_control=30, seed=1)
    check("CONTROL: white noise scores within the random-set null (z below 3)",
          abs(rn["z"]) < 3.0)
    check("the control definition is recorded in the output",
          "complement of the key set" in rn["control_definition"])


# --------------------------------------------------------------------------- #
# 4. attention report                                                          #
# --------------------------------------------------------------------------- #
def check_attention_report(tmp: Path):
    print("\n-- attention_report (differentiated, descriptive) --")
    cfg, _model, state, _ = _make_run(tmp, seed=3)
    d = TM.forward_decomposition(_zero_qk(state), cfg)
    rep = TM.attention_report(d["attn"], P)
    check("uniform attention: every head reports mean 1/3 with zero spread",
          all(abs(h["a"]["mean"] - 1 / 3) < 1e-12 and h["a"]["std"] < 1e-12
              for h in rep["per_head"]))
    check("uniform attention: nothing explains its (zero) variance",
          all(h["a"]["variance_explained_by"]["sum"] == 0.0 for h in rep["per_head"]))
    check("every head and every attended position is reported",
          len(rep["per_head"]) == cfg.n_heads
          and all({"a", "b", "eq"} <= set(h) for h in rep["per_head"]))
    check("the 50/50 caveat travels with the number (master prompt 5)",
          "not causal evidence" in rep["note"])

    # a hand-made attention pattern that depends ONLY on (a+b) must be detected as such
    a, b = np.arange(P)[:, None], np.arange(P)[None, :]
    w = 0.5 + 0.4 * np.cos(2 * np.pi * 3 * (a + b) / P)
    fake = np.zeros((P, P, cfg.n_heads, 3))
    fake[:, :, :, 0] = w[:, :, None]
    fake[:, :, :, 1] = 1.0 - w[:, :, None]
    rep2 = TM.attention_report(fake, P)
    ve = rep2["per_head"][0]["a"]["variance_explained_by"]
    check("an attention pattern that is a function of (a+b) is attributed to 'sum'",
          ve["sum"] > 0.999)
    check("...and NOT to a alone or b alone",
          ve["a"] < 0.1 and ve["b"] < 0.1)


# --------------------------------------------------------------------------- #
# 5. causal head ablation                                                      #
# --------------------------------------------------------------------------- #
def check_head_ablation(tmp: Path):
    print("\n-- head_ablation (causal, on the unmodified checkpoint) --")
    cfg, _model, state, _ = _make_run(tmp, seed=4)

    # a head whose W_V and W_O are zero contributes nothing: ablating it must be a no-op
    st = {k: v.copy() for k, v in state.items()}
    st["W_V"][0] = 0.0
    st["W_O"][0] = 0.0
    res = TM.head_ablation(st, cfg)
    h0 = res["per_head"][0]
    check("ablating an all-zero head changes NOTHING (INTERFACES 6 test)",
          all(abs(h0[mode][f"delta_{m}"]) < 1e-12
              for mode in ("zero", "mean")
              for m in ("train_loss", "test_loss", "train_acc", "test_acc")))

    # CONTROL: a live head must matter, otherwise the check above is vacuous
    live = [h for h in res["per_head"][1:]
            if abs(h["zero"]["delta_train_loss"]) > 1e-9]
    check("CONTROL: at least one live head DOES change the loss when zeroed",
          len(live) > 0)

    check("both absolute and relative changes are reported (master prompt 11)",
          {"train_loss", "delta_train_loss", "relative_train_loss"} <= set(h0["zero"]))
    check("the baseline is reported next to the ablations",
          {"train_loss", "test_loss", "train_acc", "test_acc"} <= set(res["baseline"]))
    check("no retraining is claimed anywhere in the record",
          "no retraining" in res["rules"])

    # fixing attention to its mean is a no-op exactly when attention is already constant
    fixed0 = TM.head_ablation(_zero_qk(state), cfg)["fix_attention_to_mean"]
    check("uniform attention: fixing attention to its mean changes nothing",
          abs(fixed0["delta_train_loss"]) < 1e-10 and abs(fixed0["delta_test_loss"]) < 1e-10)
    check("CONTROL: with input-dependent attention it is NOT a no-op",
          abs(res["fix_attention_to_mean"]["delta_train_loss"]) > 1e-9)


# --------------------------------------------------------------------------- #
# 6. grouped variance helper                                                   #
# --------------------------------------------------------------------------- #
def check_variance_helper():
    print("\n-- variance_explained_by --")
    a, b = np.arange(P)[:, None], np.arange(P)[None, :]
    a_idx, b_idx = np.repeat(np.arange(P), P), np.tile(np.arange(P), P)
    s = ((a + b) % P).ravel()
    f_sum = np.cos(2 * np.pi * 4 * s / P)[:, None]
    check("a function of (a+b) is fully explained by (a+b)",
          abs(TM.variance_explained_by(f_sum, s, P)[0] - 1.0) < 1e-12)
    check("...and barely at all by a alone",
          TM.variance_explained_by(f_sum, a_idx, P)[0] < 0.1)
    f_a = np.cos(2 * np.pi * 4 * a_idx / P)[:, None]
    check("a function of a alone is fully explained by a",
          abs(TM.variance_explained_by(f_a, a_idx, P)[0] - 1.0) < 1e-12)
    check("a constant column reports 0, not nan",
          TM.variance_explained_by(np.ones((P * P, 1)), s, P)[0] == 0.0)
    check("a 1-D input is accepted and returns one value",
          TM.variance_explained_by(f_sum.ravel(), s, P).shape == (1,))
    # unequal group sizes must be weighted by count, not averaged flat
    g = np.zeros(P * P, dtype=int)
    g[:10] = 1
    v = np.concatenate([np.ones(10), np.zeros(P * P - 10)])[:, None]
    check("unequal group sizes are weighted by count",
          abs(TM.variance_explained_by(v, g, 2)[0] - 1.0) < 1e-12)


# --------------------------------------------------------------------------- #
# 7. direction spectra                                                         #
# --------------------------------------------------------------------------- #
def check_direction_spectra(tmp: Path):
    print("\n-- direction_spectra --")
    cfg, _model, state, _ = _make_run(tmp, seed=5)
    sp = TM.direction_spectra(state, cfg)
    check("the direct, per-head, logit-map and unembedding spectra are all present",
          {"embedding_to_neuron_direct", "embedding_to_neuron_head0", "neuron_logit_map",
           "unembedding", "W_pos_norms"} <= set(sp))
    check("one entry per head",
          all(f"embedding_to_neuron_head{h}" in sp for h in range(cfg.n_heads)))
    check("W_pos norms are reported per position",
          len(sp["W_pos_norms"]) == cfg.n_ctx)

    # inject a pure sinusoid into W_E and it must show up as that frequency
    k = 6
    st = {kk: v.copy() for kk, v in state.items()}
    st["W_E"] = np.zeros_like(st["W_E"])
    st["W_E"][:P, 0] = np.cos(2 * np.pi * k * np.arange(P) / P)
    st["W_in"] = np.zeros_like(st["W_in"])
    st["W_in"][0, :] = 1.0
    sp2 = TM.direction_spectra(st, cfg)
    hist = sp2["embedding_to_neuron_direct"]["dominant_frequency_histogram"]
    check(f"an injected frequency {k} is recovered by the direct spectrum",
          set(hist) == {k})
    check("...and it is reported as fully concentrated",
          sp2["embedding_to_neuron_direct"]["dominant_fraction"]["median"] > 0.999)


# --------------------------------------------------------------------------- #
# 8. analyse(): the run-level output contract                                  #
# --------------------------------------------------------------------------- #
def check_analyse(tmp: Path):
    print("\n-- analyse(): run-level entry point + output contract --")
    cfg, _model, _state, run_dir = _make_run(tmp, seed=6)
    payload, path = TM.analyse(run_dir, None, TM.PRIMARY_KEY_RULE, 0,
                               n_boot=50, n_perm=50, n_exemplars=4, n_control=5)
    check("the result lands at analysis/transformer_mechanism/final.json",
          path == run_dir / "analysis" / "transformer_mechanism" / "final.json" and path.exists())
    check("the npz is written next to it", path.with_suffix(".npz").exists())
    on_disk = json.loads(path.read_text())
    check("the provenance envelope is complete",
          all(on_disk.get(k) for k in ("module", "module_version", "run_id", "arch", "p",
                                       "created_utc"))
          and on_disk["arch"] == "transformer")
    res = on_disk["results"]
    check("every INTERFACES 6 result block is present",
          {"forward_check", "additivity", "neuron_tables", "wave_fits", "activation",
           "logit_contributions", "structured_neurons", "key_subspace_variance",
           "attention", "head_ablation", "direction_spectra",
           "direct_path_share_of_logit_variance"} <= set(res))
    check("the forward decomposition is re-checked against the model on every run",
          res["forward_check"]["relative_logit_error_vs_model"] < 1e-5
          and res["forward_check"]["direct_plus_mlp_equals_logits"] == 0.0)
    check("the logit-variance shares still sum to 1 with the [p,p,p] direct path as bias",
          abs(res["logit_contributions"]["total_variance_share"] - 1.0) < 1e-9)
    check("the key-frequency rule (and any fallback) is recorded",
          "key_rule" in on_disk["params"]["key_frequency_selection"])
    check("the mean-attention approximation is declared in params, not hidden",
          "additivity_r2" in on_disk["params"]["effective_curve_definition"])
    check("the activation source is declared as the TRUE hidden layer",
          "true hidden" in on_disk["params"]["activation_source"])
    check("the storage deviation from INTERFACES 6 is recorded in params",
          "not written to the npz" in on_disk["params"]["storage_deviation"].lower())
    check("the result is labelled measurement-only",
          on_disk["params"]["status"].startswith("MEASUREMENT ONLY"))

    with np.load(path.with_suffix(".npz")) as z:
        keys = set(z.files)
        n = int(res["n_neurons"])
        check("per-neuron additivity is stored", z["additivity_r2"].shape == (n,))
        check("the effective curves are stored",
              z["curve__u_a"].shape == (P, n) and z["curve__b_in"].shape == (n,))
        check("the structured masks are stored", "mask__alive" in keys)
        check("the read-out attention tensor is stored",
              z["attention_read_out"].shape == (P, P, cfg.n_heads, 3))
        check("no [p, p, d_mlp] hidden tensor is stored (master prompt 20)",
              "hidden" not in keys and all(z[k].size <= P * P * cfg.n_heads * 3 for k in keys))

    check("an MLP run is refused with a pointer to the right module",
          _raises(lambda: TM.analyse(_make_mlp_run(tmp))))


def _make_mlp_run(tmp: Path):
    cfg = get_config("nanda", p=P, arch="mlp", d_mlp=D_MLP, seed=9)
    set_seed(9)
    model = build_model(cfg)
    run_dir = tmp / ("mlp_" + cfg.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps(
        {"config": dataclasses.asdict(cfg), "git_commit": "synthetic-test"}))
    torch.save(model.state_dict(), run_dir / "model_final.pt")
    return run_dir


# --------------------------------------------------------------------------- #
def main():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        check_forward(tmp)
        check_effective_curves(tmp)
        check_key_subspace()
        check_attention_report(tmp)
        check_head_ablation(tmp)
        check_variance_helper()
        check_direction_spectra(tmp)
        check_analyse(tmp)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
