"""Run the bounded alternative-mechanism analysis over the primary block.

PREREGISTRATION §6.3 / master prompt §12: the `neither_passes` branch pre-commits this follow-up for
BOTH architectures. It is not in analysis/driver.py's module list because it is conditional — the
decision tree decides whether it runs, and it did.

    python run_bounded_alternative.py [--n-control 5] [--probe-steps 300]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from grokverse.analysis import bounded_alternative as BA
from grokverse.analysis.driver import resolve_runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="*_frac0.3_seed*_arch25k")
    ap.add_argument("--n-control", type=int, default=5)
    ap.add_argument("--probe-steps", type=int, default=300)
    ap.add_argument("--step", type=int, default=25000)
    args = ap.parse_args()

    run_dirs = resolve_runs([args.runs])
    print(f"{len(run_dirs)} runs", flush=True)
    ok = failed = 0
    for run_dir in run_dirs:
        t = time.time()
        try:
            payload, _path = BA.analyse(run_dir, args.step, seed=0, n_control=args.n_control,
                                        probe_steps=args.probe_steps)
            r = payload["results"]
            lin = r["probes"]["linear"]
            print(f"  {run_dir.name:46s} linear {lin['real']['test_acc']:.3f} "
                  f"(control {lin['shuffled_label_control']['test_acc']:.3f}, "
                  f"chance {r['probes']['chance_level']:.3f})  "
                  f"eff.rank {r['spectrum_hidden']['effective_rank_entropy']:.1f}  "
                  f"[{time.time() - t:.0f}s]", flush=True)
            ok += 1
        except Exception as exc:                       # recorded, never swallowed
            print(f"  {run_dir.name:46s} FAILED {type(exc).__name__}: {exc}", flush=True)
            failed += 1
    print(f"\n{ok} ok, {failed} failed", flush=True)


if __name__ == "__main__":
    main()
