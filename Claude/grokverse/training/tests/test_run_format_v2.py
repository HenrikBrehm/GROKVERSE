"""Checks for run format v2 (docs/dev/RUN_FORMAT_V2.md §6).

Run from training/:  python tests/test_run_format_v2.py
Same check() convention as test_core.py; exits non-zero on the first failure.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grokverse.checkpoints import (assign_roles, assign_roles_from,  # noqa: E402
                                   list_checkpoints, load_checkpoint, load_run)
from grokverse.config import (CHECKPOINT_GRID, THRESHOLD_SETS, Config,  # noqa: E402
                              get_config)
from grokverse.data import make_dataset, split_hash  # noqa: E402
from grokverse.manifest import (REQUIRED, aggregate, flatten,  # noqa: E402
                                validate_manifest)
from grokverse.matrix import BLOCKS, block_configs, command, parse_seeds  # noqa: E402
from grokverse.models import build_model, embedding_object  # noqa: E402
from grokverse.seed import set_seed  # noqa: E402
from grokverse.train import (all_transitions, detect_transition_dense,  # noqa: E402
                             train)

SCRATCH = Path("C:/Users/henri/AppData/Local/Temp/claude/C--Users-henri-Documents-Brain-bwki/"
               "19fe38fd-ad0c-4eae-bfeb-002bb125b3f5/scratchpad")


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
def check_dense_transitions():
    print("-- dense transition detection --")
    dense = {"train_steps": [0, 10, 20, 30, 40], "train_acc": [0.1, 0.5, 0.985, 0.995, 1.0],
             "train_loss": [4.7, 3.0, 0.5, 0.1, 0.01],
             "test_steps": [0, 25, 50, 75], "test_acc": [0.01, 0.2, 0.93, 0.97],
             "test_loss": [4.7, 4.0, 1.0, 0.2],
             "eval_every_train": 10, "eval_every_test": 25}
    t = detect_transition_dense(dense, 0.99, 0.95)
    check("memorization first crossing at 30, previous 20",
          t["memorization"]["first_crossing_step"] == 30
          and t["memorization"]["previous_evaluated_step"] == 20)
    check("generalization first crossing at 75, previous 50",
          t["generalization"]["first_crossing_step"] == 75
          and t["generalization"]["previous_evaluated_step"] == 50)
    check("grok gap = 45 with interval [20, 55]",
          t["grok_gap"] == 45 and t["grok_gap_interval"] == [50 - 30, 75 - 20])
    check("eval_every is carried per crossing",
          t["memorization"]["eval_every"] == 10 and t["generalization"]["eval_every"] == 25)
    t2 = detect_transition_dense(dense, 0.99, 0.999)
    check("no generalization crossing -> null with last evaluated step",
          t2["generalization"]["first_crossing_step"] is None
          and t2["generalization"]["last_evaluated_step"] == 75 and t2["grok_gap"] is None)
    allt = all_transitions(dense)
    check("every threshold set is reported", set(allt) == set(THRESHOLD_SETS))
    loose, strict = allt["sens_loose"], allt["sens_strict"]
    check("stricter threshold never memorizes EARLIER than the looser one",
          strict["memorization"]["first_crossing_step"] >= loose["memorization"]["first_crossing_step"])
    check("no key or docstring uses the word 'exact'",
          "exact" not in json.dumps(allt).lower()
          and "exact" not in (detect_transition_dense.__doc__ or "").lower())


def check_checkpoint_rules():
    print("-- checkpoint grid + roles --")
    cfg = get_config("arch25k", steps=300)
    check("grid filtered to <= steps", {s for s in cfg.checkpoint_grid if s <= cfg.steps} == {0})
    check("default grid is the pre-specified constant", get_config("arch25k").checkpoint_grid == CHECKPOINT_GRID)
    check("grid is normalized to a sorted tuple",
          get_config("nanda", checkpoint_grid=[30, 0, 20, 20]).checkpoint_grid == (0, 20, 30))
    entries = [{"step": 0, "path": "a", "kind": "grid"}, {"step": 100, "path": "b", "kind": "grid"},
               {"step": 150, "path": "c", "kind": "memorization"},
               {"step": 200, "path": "d", "kind": "grid"}, {"step": 300, "path": "e", "kind": "grid"},
               {"step": 400, "path": "f", "kind": "grid"},
               {"step": 450, "path": "g", "kind": "generalization"},
               {"step": 500, "path": "h", "kind": "grid,final"}]
    r = assign_roles_from(entries, 150, 450, 500)
    check("init = step 0", r["init"]["step"] == 0)
    check("pre_memorization = last grid before 150 -> 100", r["pre_memorization"]["step"] == 100)
    check("memorization = event 150", r["memorization"]["step"] == 150)
    check("mid_plateau = grid nearest (150+450)/2=300", r["mid_plateau"]["step"] == 300)
    check("pre_generalization = last grid before 450 -> 400", r["pre_generalization"]["step"] == 400)
    check("generalization = event 450", r["generalization"]["step"] == 450)
    check("final = step 500", r["final"]["step"] == 500)
    r2 = assign_roles_from(entries, 150, None, 500)
    check("never generalized -> generalization/mid/pre_gen are None",
          r2["generalization"] is None and r2["mid_plateau"] is None and r2["pre_generalization"] is None)
    r3 = assign_roles_from(entries, 0, 450, 500)
    check("memorization at step 0 -> no pre_memorization", r3["pre_memorization"] is None)


def check_manifest_validation():
    print("-- manifest validation --")
    cfg = get_config("nanda", p=23, d_mlp=16, arch="mlp")
    set_seed(0)
    from grokverse.manifest import build_manifest
    m = build_manifest(cfg, "abc", build_model(cfg), "2026-09-02T00:00:00+00:00")
    validate_manifest(m)
    check("complete manifest validates", True)
    for key in REQUIRED:
        broken = dict(m)
        del broken[key]
        check(f"removing {key!r} raises", _raises(lambda: validate_manifest(broken)))
    bad = dict(m); bad["status"] = "done"
    check("unknown status raises", _raises(lambda: validate_manifest(bad)))
    bad2 = dict(m); bad2["seed"] = "0"
    check("mistyped field raises", _raises(lambda: validate_manifest(bad2)))
    flat = flatten({"a": {"b": 1, "c": [1, 2]}, "d": "x"})
    check("flatten uses dotted keys and JSON lists", flat == {"a.b": 1, "a.c": "[1, 2]", "d": "x"})
    check("per_module keys are named_parameters names",
          set(m["n_params"]["per_module"]) == {n for n, _ in build_model(cfg).named_parameters()})
    check("effective weight decay per step = lr*wd", m["effective_weight_decay_per_step"] == cfg.lr * cfg.weight_decay)


def check_split_hash():
    print("-- split hash --")
    a = make_dataset(get_config("nanda", p=23, seed=0))
    b = make_dataset(get_config("nanda", p=23, seed=0, arch="mlp"))
    c = make_dataset(get_config("nanda", p=23, seed=1))
    check("split hash deterministic", split_hash(a["train_idx"]) == split_hash(a["train_idx"]))
    check("split hash independent of arch", split_hash(a["train_idx"]) == split_hash(b["train_idx"]))
    check("split hash differs between seeds", split_hash(a["train_idx"]) != split_hash(c["train_idx"]))
    check("split hash is order-invariant", split_hash(a["train_idx"].flip(0)) == split_hash(a["train_idx"]))
    check("train_idx + test_idx partition the grid",
          sorted(a["train_idx"].tolist() + a["test_idx"].tolist()) == list(range(23 * 23)))


def check_twohot():
    print("-- two-hot MLP --")
    cfg = get_config("arch25k_twohot", p=23, d_mlp=16)
    set_seed(0)
    m = build_model(cfg)
    x = make_dataset(cfg)["all_x"]
    out = m.logits_last(x)
    check("two-hot output is [B, p]", tuple(out.shape) == (23 * 23, 23))
    check("n_params equals the built module", sum(p.numel() for p in m.parameters()) == cfg.n_params)
    check("n_params formula for two-hot", cfg.n_params == 2 * 23 * 16 + 16 + 16 * 23 + 23)
    a, b = 5, 9
    onehot = torch.zeros(2 * 23); onehot[a] = 1; onehot[23 + b] = 1
    with torch.no_grad():
        want = torch.relu(onehot @ m.W_in + m.b_in) @ m.W_out + m.b_out
        got = m.logits_last(torch.tensor([[a, b, 23]]))[0]
    check("forward equals the two-hot matrix product", torch.allclose(want, got, atol=1e-6))
    check("embedding object is the a-half of W_in", embedding_object(m) == "W_in_a_half")
    check("embedding object of the MLP / transformer is W_E",
          embedding_object(build_model(get_config("nanda", p=23, arch="mlp"))) == "W_E"
          and embedding_object(build_model(get_config("nanda", p=23))) == "W_E")
    check("wrong arch rejected", _raises(lambda: Config(arch="kan")))


def check_run_ids_and_pairing():
    print("-- run ids + paired seeds --")
    check("legacy ids unchanged when study is empty",
          get_config("nanda", arch="mlp", train_frac=0.3, seed=0).run_id == "mlp_add_p113_wd1.0_frac0.3_seed0"
          and get_config("nanda", train_frac=0.3, seed=0).run_id == "txf_add_p113_wd1.0_frac0.3_seed0"
          and get_config("grokfast", train_frac=0.5, seed=1).run_id == "txf_add_p113_wd1.0_frac0.5_gf2.0_seed1")
    check("study suffix applied", get_config("arch25k", seed=3).run_id == "txf_add_p113_wd1.0_frac0.3_seed3_arch25k")
    check("m2h prefix for the two-hot MLP", get_config("arch25k_twohot", seed=0).run_id.startswith("m2h_"))
    check("param-matched keeps the dm tag", "dm572" in get_config("arch25k_param_matched", seed=0).run_id)
    check("arch25k preset: 25k steps, frac 0.3, wd 1.0, no grokfast",
          get_config("arch25k").steps == 25000 and get_config("arch25k").train_frac == 0.3
          and get_config("arch25k").weight_decay == 1.0 and not get_config("arch25k").grokfast)
    for block in BLOCKS:
        cfgs = block_configs(block, parse_seeds(None, block))
        by_arch: dict = {}
        for _, c in cfgs:
            by_arch.setdefault(c.arch, set()).add(c.seed)
        seed_sets = list(by_arch.values())
        check(f"block {block}: identical seeds for every architecture",
              all(s == seed_sets[0] for s in seed_sets))
        check(f"block {block}: no duplicate run ids", len({c.run_id for _, c in cfgs}) == len(cfgs))
    check("primary block has 20 runs", len(block_configs("primary", list(range(10)))) == 20)
    check("confound block has 18 runs (3 cells x 2 archs x 3 seeds)", len(block_configs("confound", [0, 1, 2])) == 18)
    check("confound block excludes the primary cell",
          all(not (not c.grokfast and c.train_frac == 0.3) for _, c in block_configs("confound", [0, 1, 2])))
    check("longest configs first (transformer before mlp)",
          [c.arch for _, c in block_configs("primary", [0, 1])] == ["transformer"] * 2 + ["mlp"] * 2)
    check("seed spec parsing", parse_seeds("0-2,5", "primary") == [0, 1, 2, 5])
    gf_cfgs = [(p, c) for p, c in block_configs("confound", [0]) if c.grokfast]
    no_gf = [(p, c) for p, c in block_configs("confound", [0]) if not c.grokfast]
    cmd_gf, cmd_no = command(*gf_cfgs[0]), command(*no_gf[0])
    check("commands pin threads=1", cmd_gf[cmd_gf.index("--threads") + 1] == "1"
          and cmd_no[cmd_no.index("--threads") + 1] == "1")
    check("grokfast flag only on grokfast cells", "--grokfast" in cmd_gf and "--grokfast" not in cmd_no)
    check("config round-trip through to_dict compares equal",
          get_config("arch25k").to_dict() == Config(**{k: v for k, v in get_config("arch25k").to_dict().items()
                                                       if k in {f.name for f in __import__('dataclasses').fields(Config)}}).to_dict())


def check_dense_does_not_change_trajectory():
    print("-- dense evaluation leaves the trajectory bit-identical --")
    base = dict(p=23, d_mlp=16, steps=60, n_logged_steps=5, checkpoint_grid=(0, 30, 60))
    for arch in ("transformer", "mlp", "mlp_twohot"):
        on = get_config("nanda", arch=arch, eval_every_train=10, eval_every_test=10, **base)
        off = get_config("nanda", arch=arch, eval_every_train=10 ** 9, eval_every_test=10 ** 9, **base)
        _, m_on, _ = train(on, out_dir=None, verbose=False)
        _, m_off, _ = train(off, out_dir=None, verbose=False)
        same = all(torch.equal(a, b) for (_, a), (_, b) in
                   zip(m_on.state_dict().items(), m_off.state_dict().items()))
        check(f"{arch}: final state identical with dense eval on vs off", same)


def check_end_to_end_smoke():
    print("-- end-to-end smoke run into a temp dir --")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="v2smoke_", dir=SCRATCH))
    try:
        cfg = get_config("nanda", p=23, d_mlp=16, arch="mlp", steps=30, n_logged_steps=4,
                         eval_every_train=10, eval_every_test=15, checkpoint_grid=(0, 20, 30),
                         study="smoke")
        out = tmp / cfg.run_id
        res, model, emb = train(cfg, out_dir=out, verbose=False)
        run = json.loads((out / "run.json").read_text())
        check("run.json has dense block with both grids",
              run["dense"]["train_steps"] == [0, 10, 20, 30] and run["dense"]["test_steps"] == [0, 15, 30])
        check("run.json has transitions for all threshold sets", set(run["transitions"]) == set(THRESHOLD_SETS))
        check("legacy transition block kept and labelled", run["transition"]["legacy_log_grid"] is True)
        check("protocol_version 2 + embedding_object recorded",
              run["protocol_version"] == 2 and run["embedding_object"] == "W_E")
        ck = list_checkpoints(out)
        check("checkpoints.json lists the 3 grid entries", [e["step"] for e in ck] == [0, 20, 30])
        check("final step is tagged grid,final", ck[-1]["kind"] == "grid,final")
        check("checkpoint files exist with sha256", all((out / e["path"]).exists() and len(e["sha256"]) == 64 for e in ck))
        sd_final = torch.load(out / "model_final.pt")
        sd_ck = load_checkpoint(out, 30)
        check("model_final.pt equals ckpt_step000030.pt",
              all(torch.equal(sd_final[k], sd_ck[k]) for k in sd_final))
        man = json.loads((out / "manifest.json").read_text())
        validate_manifest(man)
        check("manifest validates and is completed",
              man["status"] == "completed" and man["steps_completed"] == 30 and man["end_utc"] is not None)
        check("manifest has init and final norms and 3 checkpoints",
              man["weight_norms_final"] is not None and len(man["checkpoints"]) == 3)
        check("manifest split hash matches the data", man["split_hash"] == split_hash(make_dataset(cfg)["train_idx"]))
        cfg2, run2, man2 = load_run(out)
        check("load_run rebuilds the config", cfg2 == cfg and man2["run_id"] == cfg.run_id)
        roles = assign_roles(out)
        check("assign_roles works on the smoke run (init + final present)",
              roles["init"]["step"] == 0 and roles["final"]["step"] == 30)
        check("embeddings.npy shape [T, p, d]", emb.shape[0] == len(run["logged_steps"]) and emb.shape[1] == 23)
        rows = aggregate(tmp, tmp / "results")
        check("aggregate writes csv + json with the run", len(rows) == 1 and (tmp / "results" / "run_manifest.csv").exists())
        # a legacy-style dir (run.json only) must appear as legacy_no_manifest
        leg = tmp / "legacy_run"; leg.mkdir()
        (leg / "run.json").write_text(json.dumps({"config": {"run_id": "legacy_run", "arch": "mlp", "seed": 0},
                                                 "transition": {}, "logged_steps": [0, 5]}))
        rows = aggregate(tmp, tmp / "results")
        check("legacy dir listed as legacy_no_manifest", any(r["status"] == "legacy_no_manifest" for r in rows))
        # a corrupt manifest must block the aggregate
        (leg / "manifest.json").write_text(json.dumps({"run_id": "x"}))
        check("invalid manifest blocks the aggregate", _raises(lambda: aggregate(tmp, tmp / "results")))
        # config-collision guard still works
        other = get_config("nanda", p=23, d_mlp=16, arch="mlp", steps=31, n_logged_steps=4, study="smoke")
        check("collision guard rejects a different config in the same dir",
              _raises(lambda: train(other, out_dir=out, verbose=False)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    check_dense_transitions()
    check_checkpoint_rules()
    check_manifest_validation()
    check_split_hash()
    check_twohot()
    check_run_ids_and_pairing()
    check_dense_does_not_change_trajectory()
    check_end_to_end_smoke()
    print("\nALL V2 CHECKS PASSED")


if __name__ == "__main__":
    main()
