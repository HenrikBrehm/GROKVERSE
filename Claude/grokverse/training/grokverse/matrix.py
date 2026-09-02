"""Launch a pre-specified block of study runs as parallel single-thread processes.

RUN_FORMAT_V2.md §5. Every run is its own ``python -m grokverse.train``
subprocess with ``threads=1`` (RESEARCH_SPEC §3.8: parallelize across
processes, never across threads), logging to ``runs/<run_id>/train.log``.
Both architectures always receive IDENTICAL seeds, so every comparison is
paired by seed (master prompt §14).

Usage (from training/):
    python -m grokverse.matrix --block primary --workers 8 [--dry-run] [--seeds 0-9]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from .config import Config, get_config
from .manifest import read_manifest
from .utils import runs_dir

DEFAULT_SEEDS: dict[str, list[int]] = {
    "pilot": [0],
    "primary": list(range(10)),
    "confound": [0, 1, 2],
    "param_matched": list(range(10)),
    "twohot": [0, 1, 2],
}
BLOCKS = tuple(DEFAULT_SEEDS)


def block_configs(block: str, seeds: list[int]) -> list[tuple[str, Config]]:
    """``[(preset, Config), ...]`` for one block. Seeds are shared by every
    architecture in the block, which is what makes the design paired."""
    if block not in BLOCKS:
        raise ValueError(f"unknown block {block!r}; choices: {list(BLOCKS)}")
    out: list[tuple[str, Config]] = []
    if block in ("pilot", "primary"):
        for arch in ("transformer", "mlp"):
            for s in seeds:
                out.append(("arch25k", get_config("arch25k", arch=arch, seed=s)))
    elif block == "confound":
        for gf in (False, True):
            for frac in (0.3, 0.5):
                if not gf and frac == 0.3:
                    continue      # this cell IS the primary block
                for arch in ("transformer", "mlp"):
                    for s in seeds:
                        out.append(("arch25k", get_config("arch25k", arch=arch, seed=s,
                                                          grokfast=gf, train_frac=frac)))
    elif block == "param_matched":
        for s in seeds:
            out.append(("arch25k_param_matched", get_config("arch25k_param_matched", seed=s)))
    elif block == "twohot":
        for s in seeds:
            out.append(("arch25k_twohot", get_config("arch25k_twohot", seed=s)))
    # longest first: transformer before MLPs, frac 0.5 before 0.3, then seed
    out.sort(key=lambda pc: (pc[1].arch != "transformer", -pc[1].train_frac,
                             pc[1].arch, pc[1].seed))
    return out


def command(preset: str, cfg: Config) -> list[str]:
    cmd = [sys.executable, "-m", "grokverse.train", "--config", preset,
           "--arch", cfg.arch, "--seed", str(cfg.seed), "--steps", str(cfg.steps),
           "--train-frac", str(cfg.train_frac), "--weight-decay", str(cfg.weight_decay),
           "--threads", "1", "--study", cfg.study, "--d-mlp", str(cfg.d_mlp),
           "--eval-every-train", str(cfg.eval_every_train),
           "--eval-every-test", str(cfg.eval_every_test)]
    if cfg.grokfast:
        cmd += ["--grokfast", "--grokfast-lambda", str(cfg.grokfast_lambda),
                "--grokfast-alpha", str(cfg.grokfast_alpha)]
    return cmd


def parse_seeds(spec: str | None, block: str) -> list[int]:
    if spec is None:
        return list(DEFAULT_SEEDS[block])
    seeds: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            seeds.extend(range(int(lo), int(hi) + 1))
        elif part:
            seeds.append(int(part))
    if not seeds:
        raise ValueError(f"no seeds parsed from {spec!r}")
    return sorted(set(seeds))


def is_completed(cfg: Config) -> bool:
    m = read_manifest(runs_dir() / cfg.run_id)
    return bool(m and m.get("status") == "completed")


def run_one(preset: str, cfg: Config) -> dict:
    """Run one config as a subprocess; stdout+stderr go to train.log."""
    run_dir = runs_dir() / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    cwd = Path(__file__).resolve().parents[1]           # training/
    t0 = time.perf_counter()
    with open(run_dir / "train.log", "a", encoding="utf-8") as log:
        log.write(f"\n### matrix launch {datetime.now(timezone.utc).isoformat()}\n")
        log.flush()
        proc = subprocess.run(command(preset, cfg), cwd=cwd, env=env,
                              stdout=log, stderr=subprocess.STDOUT)
    return {"run_id": cfg.run_id, "arch": cfg.arch, "seed": cfg.seed,
            "exit_code": int(proc.returncode),
            "elapsed_seconds": round(time.perf_counter() - t0, 1)}


def launch(block: str, workers: int, seeds: list[int], dry_run: bool = False) -> dict:
    plan = block_configs(block, seeds)
    todo = [(p, c) for p, c in plan if not is_completed(c)]
    skipped = [c.run_id for p, c in plan if is_completed(c)]
    pairing = {}
    for _, c in plan:
        pairing.setdefault(c.arch, []).append(c.seed)
    start = datetime.now(timezone.utc)
    summary = {"block": block, "workers": workers, "start_utc": start.isoformat(),
               "planned": len(plan), "skipped_completed": skipped,
               "seed_pairing": pairing, "commands": [" ".join(command(p, c)) for p, c in todo]}
    if dry_run:
        summary["dry_run"] = True
        return summary
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        results = list(ex.map(lambda pc: run_one(*pc), todo))
    summary["runs"] = results
    summary["end_utc"] = datetime.now(timezone.utc).isoformat()
    summary["n_failed"] = sum(1 for r in results if r["exit_code"] != 0)
    out_dir = runs_dir().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"matrix_{block}_{start.strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(summary, indent=2))
    summary["summary_file"] = str(path)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="launch a block of study runs")
    ap.add_argument("--block", required=True, choices=list(BLOCKS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seeds", default=None, help="e.g. 0-9 or 0,3,5 (default per block)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    seeds = parse_seeds(args.seeds, args.block)
    summary = launch(args.block, args.workers, seeds, args.dry_run)
    print(json.dumps(summary, indent=2))
    if summary.get("n_failed"):
        sys.exit(1)


if __name__ == "__main__":
    main()
