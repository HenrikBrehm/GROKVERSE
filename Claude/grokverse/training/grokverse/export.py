"""Export a run to the web data contract (PLAN Phase 5).

Writes web/public/data/<id>.meta.json + <id>.coords.bin (Float32 [T, p, 3],
the offline PCA-projected token coordinates per logged step) and rebuilds
index.json. If a run directory also holds progress_measures.json (the Nanda
restricted/excluded-loss curves from analysis/progress_measures.py), those
curves are embedded verbatim; runs without it simply lack the field — the
frontend must never invent the missing data. The frontend treats this contract
as the single source of truth and never invents values not present here.

Usage (from training/):
    python -m grokverse.export runs/txf_add_p113_wd1.0_frac0.3_seed0
    python -m grokverse.export --all
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .analysis.fourier import dominant_frequencies, frequency_concentration_over_time
from .analysis.pca import project_embeddings_over_time
from .utils import runs_dir


def web_data_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "web" / "public" / "data"


def export_run(run_dir: Path) -> dict:
    run = json.loads((run_dir / "run.json").read_text())
    embeds = np.load(run_dir / "embeddings.npy")
    cfg = run["config"]
    p, rid = cfg["p"], cfg["run_id"]

    dom = dominant_frequencies(embeds[-1], p)
    proj = project_embeddings_over_time(embeds, p, 3)
    coords = np.ascontiguousarray(proj["coords"], dtype=np.float32)   # [T, p, 3]
    conc = frequency_concentration_over_time(embeds, p, dom["dominant"])

    out = web_data_dir()
    out.mkdir(parents=True, exist_ok=True)
    coords.tofile(out / f"{rid}.coords.bin")

    meta = {
        "id": rid,
        "arch": cfg["arch"],
        "task": cfg["task"],
        "p": p,
        "hyperparams": {k: cfg[k] for k in (
            "d_model", "n_heads", "d_head", "d_mlp",
            "weight_decay", "train_frac", "lr", "steps", "init_scale",
            "grokfast", "grokfast_lambda")},
        "seed": cfg["seed"],
        "lib_versions": run["lib_versions"],
        "git_commit": run["git_commit"],
        "logged_steps": run["logged_steps"],
        "curves": run["curves"],
        "progress_measure": conc,
        "dominant_frequencies": dom["dominant"],
        "fourier_spectrum": dom["spectrum"],
        "transition": run["transition"],
        "coords_shape": list(coords.shape),
        "explained_variance_ratio": proj["explained_variance_ratio"],
    }

    pm_path = run_dir / "progress_measures.json"
    if pm_path.exists():
        pm = json.loads(pm_path.read_text())
        # only ship measures whose re-train verifiably matched the recorded run
        # (absent key = no run.json existed to compare against at compute time)
        if pm.get("sanity", {}).get("matches_recorded_run", True) is False:
            print(f"[WARN] {rid}: progress_measures transition mismatch vs "
                  f"run.json — NOT exporting the measure", flush=True)
        else:
            meta["progress_measures"] = {
                "measured_steps": pm["measured_steps"],
                "full_loss": pm["full_loss"],
                "restricted_loss": pm["restricted_loss"],
                "excluded_loss": pm["excluded_loss"],
                "key_frequencies": pm["key_frequencies"],
            }

    # allow_nan=False: a diverged run's NaN/Infinity must fail the export loudly,
    # not ship as JSON literals the browser's parser rejects at runtime
    (out / f"{rid}.meta.json").write_text(json.dumps(meta, indent=2, allow_nan=False))
    return {"id": rid, "coords_shape": list(coords.shape), "coords_bytes": int(coords.nbytes)}


def rebuild_index() -> list:
    out = web_data_dir()
    index = []
    for m in sorted(out.glob("*.meta.json")):
        d = json.loads(m.read_text())
        index.append({
            "id": d["id"], "arch": d["arch"], "task": d["task"], "p": d["p"],
            "weight_decay": d["hyperparams"]["weight_decay"],
            "train_frac": d["hyperparams"]["train_frac"], "seed": d["seed"],
            "grokfast": d["hyperparams"].get("grokfast", False),
            "dominant_frequencies": d["dominant_frequencies"],
            "transition": d["transition"],
        })
    # Ordering drives the explorer's default view: transformer first (the
    # headline), un-accelerated before Grokfast, grokked before non-grokked —
    # so the canonical un-accelerated reproduction is always the default run.
    index.sort(key=lambda r: (
        r["arch"] != "transformer", r["arch"], r["grokfast"],
        r["transition"]["test_generalized_step"] is None, r["seed"], r["id"]))
    (out / "index.json").write_text(json.dumps(index, indent=2, allow_nan=False))
    return index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?")
    ap.add_argument("--all", action="store_true", help="export every run in runs/")
    args = ap.parse_args()

    if args.all:
        results = []
        for d in sorted(runs_dir().iterdir()):
            if not (d / "run.json").exists():
                continue
            if not (d / "embeddings.npy").exists():
                # run.json is written per-snapshot during training, embeddings.npy
                # only on completion — an in-progress/interrupted run has no
                # embeddings yet and must not be exported half-baked.
                print(f"[skip] {d.name}: no embeddings.npy (run incomplete)", flush=True)
                continue
            results.append(export_run(d))
    elif args.run_dir:
        results = [export_run(Path(args.run_dir))]
    else:
        ap.error("provide a run_dir or --all")
    idx = rebuild_index()
    print(json.dumps({"exported": results, "index_runs": [i["id"] for i in idx]}, indent=2))


if __name__ == "__main__":
    main()
