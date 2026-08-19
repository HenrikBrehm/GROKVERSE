"""Lightweight correctness checks for the GROKVERSE core (run: python test_core.py).

Not a full test suite — fast asserts on the science-bearing functions
(determinism, dataset labels, Fourier-basis orthonormality + frequency recovery,
PCA shapes, MLP wiring) so regressions are caught. Exits non-zero on failure.
"""
from __future__ import annotations

import numpy as np

from grokverse.analysis.compare import agg
from grokverse.analysis.fourier import embedding_power_spectrum, fourier_basis
from grokverse.analysis.pca import fit_pca, project
from grokverse.analysis.progress_measures import _mode_indices, fwd2d, inv2d
from grokverse.config import get_config
from grokverse.data import make_dataset
from grokverse.models import build_model
from grokverse.seed import set_seed
from grokverse.train import detect_transition
from grokverse.utils import log_step_schedule


def check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        raise SystemExit(1)


def main():
    cfg = get_config("nanda", p=23, seed=0)
    d = make_dataset(cfg)
    x, y = d["all_x"], d["all_y"]
    a, b = x[:, 0], x[:, 1]
    check("labels == (a+b) mod p", bool(((a + b) % cfg.p == y).all()))
    check("equals token in last column", bool((x[:, 2] == cfg.equals_token).all()))
    check("train+test partition complete", d["train_x"].shape[0] + d["test_x"].shape[0] == cfg.p * cfg.p)
    check("split is deterministic", bool((d["train_x"] == make_dataset(cfg)["train_x"]).all()))

    set_seed(0); o1 = build_model(cfg).logits_last(x).sum().item()
    set_seed(0); o2 = build_model(cfg).logits_last(x).sum().item()
    check("model init deterministic", o1 == o2)

    cfgm = get_config("nanda", p=23, task="mul")
    dm = make_dataset(cfgm)
    check("mul labels == (a*b) mod p", bool(((dm["all_x"][:, 0] * dm["all_x"][:, 1]) % cfgm.p == dm["all_y"]).all()))

    cfgmlp = get_config("nanda", p=23, arch="mlp")
    out = build_model(cfgmlp).logits_last(make_dataset(cfgmlp)["all_x"])
    check("mlp outputs p classes", out.shape[-1] == cfgmlp.p)

    F, _ = fourier_basis(cfg.p)
    check("fourier basis orthonormal", np.allclose(F @ F.T, np.eye(cfg.p), atol=1e-8))
    check("fourier basis has p rows", F.shape[0] == cfg.p)

    p, k = cfg.p, 3
    n = np.arange(p)
    W = np.stack([np.cos(2 * np.pi * k * n / p), np.sin(2 * np.pi * k * n / p)], axis=1)
    spec = embedding_power_spectrum(W, p)
    check("spectrum recovers injected frequency k", spec["freqs"][int(np.argmax(spec["fraction"]))] == k)

    Xr = np.random.RandomState(0).randn(p, 8)
    coords = project(Xr, fit_pca(Xr, 3))
    check("pca projects to 3D", coords.shape == (p, 3))

    # --- the functions the headline numbers come from ---
    curves = {"step": [0, 100, 200, 300], "train_acc": [0.1, 0.995, 1.0, 1.0],
              "test_acc": [0.01, 0.02, 0.05, 0.97]}
    t = detect_transition(curves)
    check("transition: train sat at 0.99 crossing", t["train_saturated_step"] == 100)
    check("transition: test gen at 0.95 crossing", t["test_generalized_step"] == 300)
    check("transition: grok gap", t["grok_gap"] == 200)
    t2 = detect_transition({"step": [0, 1], "train_acc": [1.0, 1.0], "test_acc": [0.1, 0.2]})
    check("transition: no generalization -> None gap", t2["grok_gap"] is None)

    sched = log_step_schedule(1000, 20)
    check("log schedule includes 0 and final step", sched[0] == 0 and sched[-1] == 1000)
    check("log schedule strictly increasing", all(b > a for a, b in zip(sched, sched[1:])))
    check("log schedule for smoke run", log_step_schedule(1, 2) == [0, 1])

    Fb, _ = fourier_basis(p)
    L = np.random.RandomState(1).randn(p, p, 4)
    check("2D fourier roundtrip inverts", np.abs(inv2d(fwd2d(L, Fb), Fb) - L).max() < 1e-9)
    keep, is_key = _mode_indices(p, [3, 5])
    check("mode masks: const kept, not a key freq", bool(keep[0]) and not bool(is_key[0]))
    check("mode masks: 2 rows per key freq", int(is_key.sum()) == 4 and int(keep.sum()) == 5)

    a3 = agg([675, 859, 716])
    check("agg uses sample std (ddof=1)", abs(a3["std"] - np.std([675, 859, 716], ddof=1)) < 0.01)
    check("agg drops None but counts totals", agg([100, None])["n"] == 1 and agg([100, None])["n_total"] == 2)

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
