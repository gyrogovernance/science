#!/usr/bin/env python3
"""Run Yang-Mills mass-gap certificate pipeline; write Yang_Mills_Mass_Gap_results.txt.

Path: JW/OS/SC/Λ² translation (_1–_4) + Formalism Clay delivery + H7 (_5).
"""

from __future__ import annotations

import argparse
import sys
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, cast

DIR = Path(__file__).resolve().parent
if str(DIR) not in sys.path:
    sys.path.insert(0, str(DIR))

import Yang_Mills_Mass_Gap_common  # noqa: F401 — repo path setup

from Yang_Mills_Mass_Gap_common import RUN_SECTIONS, progress
from Yang_Mills_Mass_Gap_1 import run_jw_wilson
from Yang_Mills_Mass_Gap_2 import run_curvature_3d
from Yang_Mills_Mass_Gap_3 import run_sc_h6
from Yang_Mills_Mass_Gap_4 import run_refine
from Yang_Mills_Mass_Gap_5 import run_formalism

RESULTS_FILE = DIR / "Yang_Mills_Mass_Gap_results.txt"


class _TeeStdout:
    """Mirror stdout to the results file and the console in real time."""

    def __init__(self, console, file_handle):
        self._console = console
        self._file = file_handle

    def write(self, s: str) -> int:
        self._file.write(s)
        self._file.flush()
        self._console.write(s)
        self._console.flush()
        return len(s)

    def flush(self) -> None:
        self._file.flush()
        self._console.flush()

    def isatty(self) -> bool:
        return self._console.isatty()


def _print_outline() -> None:
    print("DERIVATION OUTLINE (publication order):")
    for line in RUN_SECTIONS:
        print(" ", line)
    print("-" * 5)


def main(fast: bool = False, out_path: Path | None = None) -> int:
    out_path = (out_path or RESULTS_FILE).resolve()
    form: dict = {}
    progress(f"run start (fast={fast})")
    progress(f"results -> {out_path}")
    with open(out_path, "w", encoding="utf-8") as fh:
        tee = _TeeStdout(sys.__stdout__, fh)
        with redirect_stdout(cast(IO[str], tee)), redirect_stderr(cast(IO[str], tee)):
            print("=" * 5)
            print("YANG-MILLS MASS GAP — COMPUTATIONAL CERTIFICATES")
            print("started:", datetime.now(timezone.utc).isoformat())
            print("fast:", fast)
            _print_outline()
            progress("JW + Wilson")
            run_jw_wilson(fast=fast)
            print()
            progress("OS positivity")
            run_curvature_3d()
            print()
            progress("SC + H6 + D + infinite-vol OS")
            run_sc_h6()
            print()
            progress("Λ² lock + intertwiner")
            run_refine()
            print()
            progress("Formalism Clay checklist + H7")
            form = run_formalism(fast=fast)
            print()
            print("DELIVERY SUMMARY")
            print("-" * 5)
            print("  formalism_checklist_closed :", form.get("formalism_checklist_closed"))
            print("  H7_closed                  :", form.get("H7_closed"))
            print("=" * 5)
            print("END")
    progress("run complete")
    return 0 if form.get("H7_closed") else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run YM mass gap deliverable")
    ap.add_argument("--fast", action="store_true", help="skip section 4 dense build; omit d=6 in section 5")
    ap.add_argument(
        "-o", "--output", type=Path, default=None,
        help=f"output path (default: {RESULTS_FILE.name})",
    )
    args = ap.parse_args()
    raise SystemExit(main(fast=args.fast, out_path=args.output))
