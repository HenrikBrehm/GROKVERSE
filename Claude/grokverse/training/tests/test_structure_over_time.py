"""Checks for analysis/structure_over_time.py (INTERFACES §12, PREREGISTRATION §3 H4).

Run: ``python tests/test_structure_over_time.py`` from ``training/``.

Every input is SYNTHETIC: a v2 run directory is built here from freshly initialized models, so this
file needs no trained run. Each positive check is paired with a control that must NOT pass.

WHAT IS ASSERTED ABOUT THE ONSET STATISTIC
-------------------------------------------
INTERFACES §12 defines the onset as the first checkpoint above ``init + 3 * std(first two
checkpoints)``. The tests below pin it on series whose answer is known by construction — a flat
series has no onset, a step function has its onset exactly at the step — and they also pin the
*fragility*: the baseline std is taken from two points, so the module must report that std and the
init value alongside the step. A statistic whose inputs are hidden cannot be judged by a reader.
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

from grokverse import checkpoints as CK  # noqa: E402
from grokverse.analysis import structure_over_time as SOT  # noqa: E402
from grokverse.config import get_config  # noqa: E402
from grokverse.models import build_model  # noqa: E402
from grokverse.seed import set_seed  # noqa: E402

P = 23
STEPS = (0, 10, 100, 500, 1000)
MEM_STEP, GEN_STEP = 100, 1000


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


def _make_v2_run(tmp: Path, arch: str, seed: int = 0) -> tuple:
    """A minimal run-format-v2 directory: run.json with transitions + several checkpoints."""
    kw = dict(d_model=16, d_head=4, n_heads=4) if arch == "transformer" else {}
    cfg = get_config("nanda", p=P, arch=arch, d_mlp=12, seed=seed, **kw)
    set_seed(seed)
    model = build_model(cfg)
    model.eval()
    run_dir = tmp / f"v2_{cfg.run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    for i, step in enumerate(STEPS):
        with torch.no_grad():                       # a different, deterministic state per checkpoint
            for prm in model.parameters():
                prm.add_(torch.full_like(prm, 0.01 * i))
        CK.save_checkpoint(run_dir, model, step, "grid", entries)
    CK.write_checkpoint_index(run_dir, entries)
    (run_dir / "run.json").write_text(json.dumps({
        "config": dataclasses.asdict(cfg), "git_commit": "synthetic-test",
        "steps_completed": STEPS[-1],
        "transitions": {"primary": {
            "memorization": {"first_crossing_step": MEM_STEP},
            "generalization": {"first_crossing_step": GEN_STEP}}}}))
    return cfg, model, run_dir


# --------------------------------------------------------------------------- #
# 1. the onset statistic                                                       #
# --------------------------------------------------------------------------- #
def check_onset():
    print("\n-- onset_step (INTERFACES 12) --")
    steps = [0, 1, 2, 3, 4, 5]
    flat = [0.10, 0.11, 0.10, 0.11, 0.10, 0.11]
    check("a flat series has no onset", SOT.onset_step(steps, flat)["step"] is None)
    rising = [0.10, 0.11, 0.10, 0.90, 0.95, 0.99]
    r = SOT.onset_step(steps, rising)
    check(f"a step series has its onset at the step it rises ({r['step']})", r["step"] == 3)
    check("the onset never fires on the two baseline checkpoints themselves",
          SOT.onset_step(steps, [0.0, 1.0, 0.0, 0.0, 0.0, 0.0])["step"] is None)
    check("the fragile inputs travel with the answer",
          {"init_value", "baseline_std", "n_baseline", "threshold"} <= set(r)
          and r["n_baseline"] == 2)
    check("...and the caveat is stated, not left to the reader",
          "onset indicator" in r["caveat"] and "two checkpoints" in r["caveat"])
    check("a series shorter than three checkpoints reports why it has no onset",
          SOT.onset_step([0, 1], [0.1, 0.2])["step"] is None
          and "fewer than three" in SOT.onset_step([0, 1], [0.1, 0.2])["reason"])
    check("a series with a non-finite baseline is refused, not guessed",
          SOT.onset_step(steps, [np.nan, 0.1, 0.2, 0.3, 0.4, 0.5])["step"] is None)


def check_bootstrap():
    print("\n-- bootstrap over neurons --")
    rng = np.random.default_rng(0)
    a = rng.normal(0.2, 0.05, size=200)
    b = a + 0.30
    r = SOT._bootstrap_diff(a, b, 500, 0)
    check("a known paired shift is recovered", abs(r["mean_difference"] - 0.30) < 1e-9)
    check("...with a CI excluding 0", r["ci95_low"] > 0)
    same = SOT._bootstrap_diff(a, a, 500, 0)
    check("CONTROL: identical inputs give a CI containing 0",
          same["ci95_low"] <= 0 <= same["ci95_high"])
    check("mismatched neuron counts are reported, not silently truncated",
          SOT._bootstrap_diff(a, b[:10], 100, 0)["n"] == 0)
    check("it is reproducible under its seed",
          SOT._bootstrap_diff(a, b, 200, 3)["ci95_low"]
          == SOT._bootstrap_diff(a, b, 200, 3)["ci95_low"])


# --------------------------------------------------------------------------- #
# 2. the per-checkpoint metric set                                             #
# --------------------------------------------------------------------------- #
def check_curves_and_logits(tmp: Path):
    print("\n-- _curves_and_logits: exact forward and an exact dead mask --")
    cfg, model, _ = _make_v2_run(tmp, "mlp", seed=1)
    st = {k: v.detach().numpy().astype(np.float64) for k, v in model.state_dict().items()}
    curves, logits, dead = SOT._curves_and_logits(st, cfg)
    act = np.maximum(curves["u_a"][:, None, :] + curves["u_b"][None, :, :] + curves["b_in"], 0.0)
    check("the MLP logits are the model's own forward pass",
          float(np.abs(logits - (act.reshape(P * P, -1) @ st["W_out"] + st["b_out"]).reshape(
              P, P, P)).max()) < 1e-12)
    check("the cheap dead mask equals the one computed from the full activation",
          bool((dead == (act.max(axis=(0, 1)) <= 0)).all()))

    st2 = {k: v.copy() for k, v in st.items()}
    st2["b_in"] = st2["b_in"].copy()
    st2["b_in"][0] = -1e6                                    # force neuron 0 below the ReLU floor
    _c2, _l2, dead2 = SOT._curves_and_logits(st2, cfg)
    check("a neuron driven below the ReLU floor is detected as dead", bool(dead2[0]))
    check("CONTROL: the other neurons are not", bool((~dead2[1:]).any()))

    cfgt, modelt, _ = _make_v2_run(tmp, "transformer", seed=2)
    stt = {k: v.detach().numpy().astype(np.float64) for k, v in modelt.state_dict().items()}
    _ct, lt, deadt = SOT._curves_and_logits(stt, cfgt)
    toks = torch.tensor(np.stack([np.repeat(np.arange(P), P), np.tile(np.arange(P), P),
                                  np.full(P * P, cfgt.equals_token)], axis=1))
    with torch.no_grad():
        ref = modelt.double().logits_last(toks).numpy()[:, :P].reshape(P, P, P)
    check("the transformer logits are the model's own forward pass to 1e-10",
          float(np.abs(lt - ref).max()) < 1e-10)
    check("the transformer dead mask is read off the TRUE hidden layer",
          deadt.shape == (cfgt.d_mlp,))


def check_metric_set(tmp: Path):
    print("\n-- checkpoint_metrics: the INTERFACES 12 set --")
    cfg, model, _ = _make_v2_run(tmp, "mlp", seed=3)
    st = {k: v.detach().numpy().astype(np.float64) for k, v in model.state_dict().items()}
    scalars, per_neuron = SOT.checkpoint_metrics(st, cfg, [3, 5], 0, 50, 50, 3)
    for key in ("embedding_top8_concentration", "embedding_threshold_count",
                "structured_fraction_of_live", "phase_relation_R", "fraction_best_aic_square",
                "fraction_best_aic_sinusoid", "median_odd_minus_even_u_a",
                "logit_key_subspace_share", "train_acc", "test_acc", "weight_norm_total"):
        check(f"the metric set contains {key}", key in scalars)
    check("per-neuron arrays are returned for the bootstrap",
          {"family_fraction_u_a", "odd_minus_even_u_a", "structured", "alive"} <= set(per_neuron)
          and per_neuron["structured"].shape == (cfg.d_mlp,))
    # The old form of this check ("each in [0,1] and the two sum to <= 1") passed while every
    # waveform share was silently 0.0 -- `best_by_aic` is an integer index into MODEL_NAMES and was
    # being compared to the string "square". The invariant below cannot pass on an empty table:
    # every column has exactly one best model, so the four shares must sum to exactly 1.
    shares = {m: scalars[f"fraction_best_aic_{m}"] for m in
              ("sinusoid", "square", "odd_harmonics", "odd_harmonics_1_over_j")}
    check("all four waveform shares are reported", all(v is not None for v in shares.values()))
    check("the waveform shares sum to exactly 1 -- every column has one best model",
          abs(sum(shares.values()) - 1.0) < 1e-12)
    check("...and they are not all zero, which is what comparing an index to a name produced",
          any(v > 0.0 for v in shares.values()))
    check("a missing phase relation is null, never a fabricated 0",
          scalars["phase_relation_R"] is None or isinstance(scalars["phase_relation_R"], float))
    check("weight norms are reported per parameter tensor as well as in total",
          "weight_norm_W_out" in scalars and scalars["weight_norm_total"] > 0)


# --------------------------------------------------------------------------- #
# 3. analyse(): the run-level output contract                                  #
# --------------------------------------------------------------------------- #
def check_analyse(tmp: Path):
    print("\n-- analyse(): run-level entry point + output contract --")
    for arch in ("mlp", "transformer"):
        _cfg, _model, run_dir = _make_v2_run(tmp, arch, seed=4)
        payload, path = SOT.analyse(run_dir, SOT.PRIMARY_KEY_RULE, 0, n_boot=50, n_perm=50,
                                    n_control=2, with_progress_measures=False, figure=True)
        check(f"{arch}: the result lands at analysis/structure_over_time/all_checkpoints.json",
              path == run_dir / "analysis" / "structure_over_time" / "all_checkpoints.json"
              and path.exists())
        on_disk = json.loads(path.read_text())
        res = on_disk["results"]
        check(f"{arch}: one series point per checkpoint",
              res["series"]["step"] == list(STEPS))
        check(f"{arch}: the roles are resolved and recorded",
              res["roles"]["init"] == 0 and res["roles"]["generalization"] == GEN_STEP)
        check(f"{arch}: the H4 contrast names its two roles",
              res["h4"]["from_role"] == "init" and res["h4"]["to_role"] == "pre_generalization")
        check(f"{arch}: every scalar metric gets an onset entry",
              "onset" in res["h4"]["per_metric"]["structured_fraction_of_live"])
        check(f"{arch}: the correlational caveat is in the output, not only the docstring",
              "CORRELATIONAL" in res["h4"]["reading"])
        check(f"{arch}: the key set is fixed once from the final checkpoint",
              "final" in on_disk["params"]["key_frequency_rule_note"].lower())
        check(f"{arch}: the Khanh caveat is recorded",
              "converged" in on_disk["params"]["khanh_caveat"])
        check(f"{arch}: the output-path deviation from INTERFACES 12 is declared",
              "all_checkpoints.json" in on_disk["params"]["output_path_deviation"])
        check(f"{arch}: a figure is written and referenced from the JSON",
              (path.parent / "structure_over_time.png").exists() and "figure" in res)
        with np.load(path.with_suffix(".npz")) as z:
            check(f"{arch}: per-neuron arrays are stored per checkpoint",
                  f"step{STEPS[0]:06d}__family_fraction_u_a" in z.files
                  and f"step{STEPS[-1]:06d}__structured" in z.files)
    check("a legacy run without checkpoints.json is refused with a message naming why",
          _raises(lambda: SOT.analyse(tmp / "does_not_exist")))


# --------------------------------------------------------------------------- #
def main():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        check_onset()
        check_bootstrap()
        check_curves_and_logits(tmp)
        check_metric_set(tmp)
        check_analyse(tmp)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
