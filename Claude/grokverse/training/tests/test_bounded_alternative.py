"""Checks for analysis/bounded_alternative.py (INTERFACES §13, PREREGISTRATION §6.3).

Run: ``python tests/test_bounded_alternative.py`` from ``training/``.

Every input is synthetic or a freshly initialized model.

THE CONTROL THAT CARRIES THIS MODULE
-------------------------------------
A probe's accuracy on its own is worthless: a rich representation plus enough parameters fits
anything. The shuffled-label control — same architecture, same budget, same learning rate, same seed,
only the labels permuted — is what makes the number mean something. The checks below assert both
directions: on a representation that really encodes `(a+b) mod p` the real probe beats its control by
a wide margin, and on **pure noise** the real probe does NOT beat its control. A module that passes
only the first test would happily report structure in random data.
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

from grokverse.analysis import bounded_alternative as BA  # noqa: E402
from grokverse.config import get_config  # noqa: E402
from grokverse.models import build_model  # noqa: E402
from grokverse.seed import set_seed  # noqa: E402

P = 23
PROBE_STEPS = 120


def check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        raise SystemExit(1)


def _cfg(arch="mlp", seed=0, **kw):
    extra = dict(d_model=16, d_head=4, n_heads=4) if arch == "transformer" else {}
    return get_config("nanda", p=P, arch=arch, d_mlp=12, seed=seed, **extra, **kw)


# --------------------------------------------------------------------------- #
# 1. the probes, and the control that must not pass                            #
# --------------------------------------------------------------------------- #
def check_probes():
    print("\n-- probes for (a+b) mod p, each against its shuffled-label control --")
    cfg = _cfg()
    a = np.repeat(np.arange(P), P)
    b = np.tile(np.arange(P), P)
    y = (a + b) % P
    # a representation that encodes the answer perfectly: one-hot of (a+b) mod p
    onehot = np.eye(P)[y]
    res = BA.probe_battery(onehot, cfg, seed=0, steps=PROBE_STEPS)
    lin = res["linear"]
    check(f"the linear probe decodes a one-hot sum representation "
          f"({lin['real']['test_acc']:.3f})", lin["real"]["test_acc"] > 0.9)
    check(f"...while its shuffled-label control stays at chance "
          f"({lin['shuffled_label_control']['test_acc']:.3f} vs {res['chance_level']:.3f})",
          lin["shuffled_label_control"]["test_acc"] < 5 * res["chance_level"])
    check("the gap above the control is reported, not just the raw accuracy",
          lin["test_acc_above_control"] > 0.8)
    check("both probe depths are run", "one_hidden_layer" in res and "linear" in res)
    check("the protocol records that it trains on the model's own split",
          "OWN train split" in res["protocol"])

    # CONTROL: pure noise must NOT beat its shuffled control
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((P * P, 40))
    nres = BA.probe_battery(noise, cfg, seed=0, steps=PROBE_STEPS)
    check(f"CONTROL: on noise the linear probe does not beat its shuffled control "
          f"({nres['linear']['real']['test_acc']:.3f} vs "
          f"{nres['linear']['shuffled_label_control']['test_acc']:.3f})",
          nres["linear"]["test_acc_above_control"] < 0.1)
    check("CONTROL: and it stays near chance on the test split",
          nres["linear"]["real"]["test_acc"] < 5 * nres["chance_level"])
    check("the real and control probes are given identical budgets",
          lin["real"]["steps"] == lin["shuffled_label_control"]["steps"]
          and lin["real"]["n_params"] == lin["shuffled_label_control"]["n_params"])


# --------------------------------------------------------------------------- #
# 2. spectra and CKA                                                           #
# --------------------------------------------------------------------------- #
def check_spectrum():
    print("\n-- singular spectra --")
    rng = np.random.default_rng(0)
    u = rng.standard_normal((200, 1))
    v = rng.standard_normal((1, 30))
    rank1 = u @ v
    s1 = BA.spectrum(rank1, "rank1")
    check(f"a rank-1 matrix has effective rank ~1 ({s1['effective_rank_entropy']:.3f})",
          abs(s1["effective_rank_entropy"] - 1.0) < 0.05)
    check("...and one component explains ~all of it", s1["n_components_for_90pct"] == 1)
    iso = rng.standard_normal((400, 30))
    s2 = BA.spectrum(iso, "isotropic")
    check(f"CONTROL: isotropic noise has a much higher effective rank "
          f"({s2['effective_rank_entropy']:.1f})", s2["effective_rank_entropy"] > 20)
    check("both effective-rank definitions are named in the output",
          set(s1["definitions"]) == {"effective_rank_entropy", "participation_ratio"})
    check("a degenerate (all-zero) input is flagged, not divided by zero",
          BA.spectrum(np.zeros((10, 4)), "zero").get("degenerate") is True)


def check_cka():
    print("\n-- linear CKA --")
    rng = np.random.default_rng(1)
    X = rng.standard_normal((300, 20))
    check("CKA of a representation with itself is 1", abs(BA.linear_cka(X, X) - 1.0) < 1e-10)
    Q, _ = np.linalg.qr(rng.standard_normal((20, 20)))
    check("CKA is invariant to an orthogonal rotation", abs(BA.linear_cka(X, X @ Q) - 1.0) < 1e-8)
    check("...and to isotropic scaling", abs(BA.linear_cka(X, 7.5 * X) - 1.0) < 1e-8)
    Y = rng.standard_normal((300, 20))
    check(f"CONTROL: two independent representations have low CKA ({BA.linear_cka(X, Y):.3f})",
          BA.linear_cka(X, Y) < 0.3)
    check("CKA is symmetric", abs(BA.linear_cka(X, Y) - BA.linear_cka(Y, X)) < 1e-10)


# --------------------------------------------------------------------------- #
# 3. the causal test on singular directions                                    #
# --------------------------------------------------------------------------- #
def check_singular_ablation():
    print("\n-- removing the top-r singular directions vs r random directions --")
    cfg = _cfg()
    a = np.repeat(np.arange(P), P)
    b = np.tile(np.arange(P), P)
    y = (a + b) % P
    rng = np.random.default_rng(0)
    # hidden = a strong 2-D signal carrying the answer, plus many weak noise dimensions
    signal = np.stack([np.cos(2 * np.pi * y / P), np.sin(2 * np.pi * y / P)], axis=1) * 20.0
    noise = 0.05 * rng.standard_normal((P * P, 18))
    hidden = np.concatenate([signal, noise], axis=1)
    readout = np.zeros((20, P))
    n = np.arange(P)
    readout[0] = np.cos(2 * np.pi * n / P)
    readout[1] = np.sin(2 * np.pi * n / P)
    readout *= 20.0
    res = BA.singular_direction_ablation(hidden, readout, np.zeros(P), cfg,
                                         ranks=(1, 2, 4), n_control=8, seed=0)
    check(f"the synthetic circuit solves the task first "
          f"({res['baseline']['test_acc']:.3f})", res["baseline"]["test_acc"] > 0.95)
    r2 = res["per_rank"]["2"]
    check(f"removing the 2 directions it uses destroys it "
          f"({r2['observed']['test_acc']:.3f})", r2["observed"]["test_acc"] < 0.2)
    check(f"CONTROL: removing 2 RANDOM directions barely hurts "
          f"({r2['control']['mean_drop']:.3f} mean drop)", r2["control"]["mean_drop"] < 0.2)
    check("the observed damage exceeds every random control",
          r2["exceeds_all_controls"] is True)
    check("the control definition is recorded with the number",
          "orthonormal subspaces" in res["control_definition"])
    check("every requested rank is reported", set(res["per_rank"]) == {"1", "2", "4"})


# --------------------------------------------------------------------------- #
# 4. run-level contract                                                        #
# --------------------------------------------------------------------------- #
def _make_run(tmp: Path, arch: str, seed: int = 0):
    cfg = _cfg(arch, seed=seed)
    set_seed(seed)
    model = build_model(cfg)
    model.eval()
    run_dir = tmp / f"{arch}_{cfg.run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps(
        {"config": dataclasses.asdict(cfg), "git_commit": "synthetic-test"}), encoding="utf-8")
    torch.save(model.state_dict(), run_dir / "model_final.pt")
    return run_dir


def check_analyse(tmp: Path):
    print("\n-- analyse(): the run-level output contract --")
    for arch in ("mlp", "transformer"):
        run_dir = _make_run(tmp, arch, seed=1)
        payload, path = BA.analyse(run_dir, None, seed=0, n_control=3, probe_steps=40)
        check(f"{arch}: the result lands at analysis/bounded_alternative/final.json",
              path == run_dir / "analysis" / "bounded_alternative" / "final.json" and path.exists())
        on_disk = json.loads(path.read_text(encoding="utf-8"))
        res = on_disk["results"]
        check(f"{arch}: all four pre-committed questions are answered",
              {"probes", "spectrum_hidden", "spectrum_logits",
               "singular_direction_ablation"} <= set(res))
        check(f"{arch}: the Khanh caveat travels with the numbers",
              "converged values" in on_disk["params"]["khanh_caveat"])
        check(f"{arch}: the bounded scope is stated, including that a negative result is valid",
              "no fifth" in on_disk["params"]["scope"]
              and "negative result" in on_disk["params"]["scope"])
        check(f"{arch}: it records that it only runs for an architecture that FAILED the gate",
              "FAILED the evidence gate" in on_disk["params"]["when_this_runs"])
        check(f"{arch}: the shuffled-label control is present for both probe depths",
              all("shuffled_label_control" in res["probes"][k]
                  for k in ("linear", "one_hidden_layer")))


def check_cka_across_runs(tmp: Path):
    print("\n-- compare_seeds(): cross-seed CKA --")
    runs = [_make_run(tmp / "cka", "mlp", seed=s) for s in (2, 3)]
    res = BA.compare_seeds(runs)
    check("the matrix is square with 1 on the diagonal",
          len(res["cka_matrix"]) == 2 and res["cka_matrix"][0][0] == 1.0)
    check("it is symmetric",
          abs(res["cka_matrix"][0][1] - res["cka_matrix"][1][0]) < 1e-10)
    check("each run is identified with its checkpoint",
          all({"run_id", "step"} <= set(r) for r in res["runs"]))
    check("the checkpoint caveat is attached", "not interpretable" in res["note"])


def main():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        check_probes()
        check_spectrum()
        check_cka()
        check_singular_ablation()
        check_analyse(tmp)
        check_cka_across_runs(tmp)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
