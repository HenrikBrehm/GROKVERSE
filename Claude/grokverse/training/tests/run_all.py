"""Run every test file of the study plus the legacy test_core.py (run from training/).

    python tests/run_all.py            # everything
    python tests/run_all.py wave       # only test files whose name contains 'wave'

Each test file is a standalone script using the check()/SystemExit convention; this runner executes
each in a subprocess with the same interpreter, reports PASS/FAIL per file, and exits non-zero if any
file failed. Nothing is skipped silently.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    here = Path(__file__).resolve().parent
    training = here.parent
    pattern = sys.argv[1] if len(sys.argv) > 1 else ""
    files = sorted(p for p in here.glob("test_*.py") if pattern in p.name)
    core = training / "test_core.py"
    if core.exists() and pattern in core.name:
        files.append(core)
    if not files:
        raise SystemExit(f"no test files match {pattern!r} under {here}")

    failed: list[str] = []
    for f in files:
        print(f"===== {f.relative_to(training)} =====", flush=True)
        proc = subprocess.run([sys.executable, str(f)], cwd=training)
        status = "PASS" if proc.returncode == 0 else f"FAIL (exit {proc.returncode})"
        print(f"----- {f.name}: {status}\n", flush=True)
        if proc.returncode != 0:
            failed.append(f.name)

    print(f"{len(files) - len(failed)}/{len(files)} test files passed")
    if failed:
        print("FAILED: " + ", ".join(failed))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
