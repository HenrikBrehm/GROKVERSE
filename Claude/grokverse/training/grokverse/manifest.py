"""Machine-readable run manifest + aggregate (RUN_FORMAT_V2.md §4, master prompt §20).

Every v2 run writes ``runs/<run_id>/manifest.json`` at start (``status:
"running"``) and rewrites it at the end (``completed`` / ``failed`` /
``aborted``). ``python -m grokverse.manifest`` aggregates all manifests into
``results/run_manifest.csv`` and ``results/run_manifest.json``. Legacy run
directories (no manifest) appear in the aggregate as ``legacy_no_manifest``
rows built from their run.json, so nothing is silently omitted.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
from pathlib import Path

import torch

from .config import Config
from .utils import git_commit, lib_versions, runs_dir

MANIFEST_NAME = "manifest.json"
STATUSES = ("running", "completed", "failed", "aborted")

#: required key -> accepted python types (None allowed where noted)
REQUIRED: dict[str, tuple] = {
    "run_id": (str,), "study": (str,), "arch": (str,), "seed": (int,),
    "task": (str,), "p": (int,), "train_frac": (float, int), "weight_decay": (float, int),
    "grokfast": (bool,), "steps": (int,), "split_hash": (str,),
    "git_commit": (str,), "python_version": (str,), "torch_version": (str,),
    "numpy_version": (str,), "platform": (str,), "processor": (str,),
    "cpu_count": (int,), "torch_num_threads": (int,), "config": (dict,),
    "start_utc": (str,), "end_utc": (str, type(None)),
    "elapsed_seconds": (float, int, type(None)), "steps_completed": (int,),
    "status": (str,), "abort_reason": (str, type(None)),
    "n_params": (dict,), "weight_norms_init": (dict,),
    "weight_norms_final": (dict, type(None)),
    "effective_weight_decay_per_step": (float, int),
    "checkpoints": (list,), "result_files": (list,),
}


# --------------------------------------------------------------------------- #
# model bookkeeping                                                            #
# --------------------------------------------------------------------------- #
def count_params(model: torch.nn.Module) -> dict:
    per = {n: int(p.numel()) for n, p in model.named_parameters()}
    return {"total": int(sum(per.values())),
            "trainable": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "per_module": per}


@torch.no_grad()
def weight_norms(model: torch.nn.Module) -> dict:
    """L2 norm per parameter tensor and of the whole parameter vector."""
    per = {n: float(p.detach().norm().item()) for n, p in model.named_parameters()}
    total = float(sum(v * v for v in per.values()) ** 0.5)
    return {"total": total, "per_module": per}


# --------------------------------------------------------------------------- #
# build / write / validate                                                     #
# --------------------------------------------------------------------------- #
def build_manifest(cfg: Config, split: str, model: torch.nn.Module,
                   start_utc: str) -> dict:
    """The start-of-run manifest (``status: running``); train() fills the rest."""
    vers = lib_versions()
    return {
        "run_id": cfg.run_id, "study": cfg.study, "arch": cfg.arch, "seed": cfg.seed,
        "task": cfg.task, "p": cfg.p, "train_frac": cfg.train_frac,
        "weight_decay": cfg.weight_decay, "grokfast": cfg.grokfast, "steps": cfg.steps,
        "split_hash": split,
        "git_commit": git_commit(),
        "python_version": vers["python"], "torch_version": vers["torch"],
        "numpy_version": vers["numpy"],
        "platform": platform.platform(), "processor": platform.processor() or "unknown",
        "cpu_count": int(os.cpu_count() or 0), "torch_num_threads": torch.get_num_threads(),
        "config": cfg.to_dict(),
        "start_utc": start_utc, "end_utc": None, "elapsed_seconds": None,
        "steps_completed": 0, "status": "running", "abort_reason": None,
        "n_params": count_params(model),
        "weight_norms_init": weight_norms(model),
        "weight_norms_final": None,
        # AdamW's decoupled decay shrinks every weight by lr*wd per step
        "effective_weight_decay_per_step": float(cfg.lr * cfg.weight_decay),
        "checkpoints": [], "result_files": [],
    }


def validate_manifest(d: dict) -> None:
    """Raise ``ValueError`` naming the first missing or mistyped field."""
    if not isinstance(d, dict):
        raise ValueError("manifest must be a dict")
    for key, types in REQUIRED.items():
        if key not in d:
            raise ValueError(f"manifest missing required key {key!r}")
        if not isinstance(d[key], types):
            raise ValueError(f"manifest key {key!r} has type {type(d[key]).__name__}, "
                             f"expected one of {[t.__name__ for t in types]}")
    if d["status"] not in STATUSES:
        raise ValueError(f"manifest status {d['status']!r} not in {STATUSES}")
    for sub in ("total", "per_module"):
        if sub not in d["n_params"] or sub not in d["weight_norms_init"]:
            raise ValueError(f"n_params / weight_norms_init must contain {sub!r}")


def write_manifest(run_dir: Path, d: dict) -> Path:
    validate_manifest(d)
    path = Path(run_dir) / MANIFEST_NAME
    path.write_text(json.dumps(d, indent=2))
    return path


def read_manifest(run_dir: Path) -> dict | None:
    path = Path(run_dir) / MANIFEST_NAME
    return json.loads(path.read_text()) if path.exists() else None


# --------------------------------------------------------------------------- #
# aggregate                                                                    #
# --------------------------------------------------------------------------- #
def flatten(d: dict, prefix: str = "") -> dict:
    """Nested dict -> dotted keys; lists become JSON strings (CSV-friendly)."""
    out: dict = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        elif isinstance(v, list):
            out[key] = json.dumps(v)
        else:
            out[key] = v
    return out


def _legacy_row(run_dir: Path) -> dict:
    run = json.loads((run_dir / "run.json").read_text())
    cfg = run.get("config", {})
    return {
        "run_id": cfg.get("run_id", run_dir.name), "study": cfg.get("study", ""),
        "arch": cfg.get("arch"), "seed": cfg.get("seed"), "task": cfg.get("task"),
        "p": cfg.get("p"), "train_frac": cfg.get("train_frac"),
        "weight_decay": cfg.get("weight_decay"), "grokfast": cfg.get("grokfast"),
        "steps": cfg.get("steps"), "git_commit": run.get("git_commit"),
        "python_version": run.get("lib_versions", {}).get("python"),
        "torch_version": run.get("lib_versions", {}).get("torch"),
        "numpy_version": run.get("lib_versions", {}).get("numpy"),
        "torch_num_threads": run.get("torch_num_threads"),
        "start_utc": run.get("created_utc"), "status": "legacy_no_manifest",
        "steps_completed": (run.get("logged_steps") or [None])[-1],
        "legacy_transition": run.get("transition"),
        "result_files": sorted(p.name for p in run_dir.iterdir() if p.is_file()),
    }


def aggregate(runs: Path | None = None, results: Path | None = None) -> list[dict]:
    """Collect every run directory into results/run_manifest.{csv,json}.

    Validates every manifest.json first and refuses to write anything if one is
    invalid (a partial aggregate would look complete).
    """
    runs = Path(runs) if runs else runs_dir()
    results = Path(results) if results else runs.parent / "results"
    rows, problems = [], []
    for d in sorted(p for p in runs.iterdir() if p.is_dir()):
        man = d / MANIFEST_NAME
        if man.exists():
            m = json.loads(man.read_text())
            try:
                validate_manifest(m)
            except ValueError as e:
                problems.append(f"{man}: {e}")
                continue
            rows.append(m)
        elif (d / "run.json").exists():
            rows.append(_legacy_row(d))
    if problems:
        raise ValueError("invalid manifests — aggregate NOT written:\n  " + "\n  ".join(problems))
    results.mkdir(parents=True, exist_ok=True)
    (results / "run_manifest.json").write_text(json.dumps(rows, indent=2))
    flat = [flatten(r) for r in rows]
    cols = sorted(set().union(*(f.keys() for f in flat))) if flat else []
    with open(results / "run_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in flat:
            w.writerow(r)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="aggregate run manifests")
    ap.add_argument("--runs", type=Path, default=None, help="runs directory (default training/runs)")
    ap.add_argument("--out", type=Path, default=None, help="results directory (default training/results)")
    args = ap.parse_args()
    rows = aggregate(args.runs, args.out)
    out = (args.out or (args.runs or runs_dir()).parent / "results")
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(json.dumps({"n_runs": len(rows), "by_status": counts,
                      "csv": str(out / "run_manifest.csv"),
                      "json": str(out / "run_manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
