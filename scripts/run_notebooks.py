#!/usr/bin/env python3
"""
run_notebooks.py — execute every notebook and report pass/fail.

Used by `make notebooks` and by CI. A notebook that fails here is a notebook
that will fail in front of the class, so this runs before every session.

    python3 scripts/run_notebooks.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"
PY = ROOT / ".venv" / "bin" / "python"
KERNEL = "cc3po-ai-q3"


def check_kernelspec() -> bool:
    """Verify every notebook points at a kernel that exists."""
    spec_dirs = [
        Path.home() / "Library/Jupyter/kernels",
        Path.home() / ".local/share/jupyter/kernels",
        ROOT / ".venv/share/jupyter/kernels",
    ]
    installed = set()
    for d in spec_dirs:
        if d.exists():
            for p in d.iterdir():
                if p.is_dir():
                    installed.add(p.name)
    if KERNEL not in installed:
        print(f"  WARNING: kernel '{KERNEL}' is not installed.")
        print(f"           Run: {PY} -m ipykernel install --user --name {KERNEL}")
        return False
    return True


def run_one(path: Path) -> tuple[bool, str, float]:
    started = time.perf_counter()
    proc = subprocess.run(
        [str(PY), "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
         "--stdout", str(path)],
        capture_output=True, text=True, cwd=str(ROOT), timeout=600,
    )
    elapsed = time.perf_counter() - started
    if proc.returncode == 0:
        return True, "", elapsed
    tail = "\n".join(proc.stderr.strip().splitlines()[-12:])
    return False, tail, elapsed


def main() -> int:
    if not PY.exists():
        print(f"  No venv at {PY}. Run `make setup` first.")
        return 1

    check_kernelspec()

    notebooks = sorted(NB_DIR.glob("*.ipynb"))
    if not notebooks:
        print("  No notebooks found.")
        return 1

    print(f"  Executing {len(notebooks)} notebooks...\n")
    failures = []
    for nb in notebooks:
        ok, err, elapsed = run_one(nb)
        status = "OK  " if ok else "FAIL"
        print(f"  [{status}] {nb.name:34s} {elapsed:5.1f}s")
        if not ok:
            failures.append((nb.name, err))
            print("        " + err.replace("\n", "\n        ")[:900])

    print()
    if failures:
        print(f"  {len(failures)} notebook(s) FAILED:")
        for name, _ in failures:
            print(f"    - {name}")
        return 1
    print(f"  All {len(notebooks)} notebooks executed cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
