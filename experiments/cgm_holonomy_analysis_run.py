#!/usr/bin/env python3
"""
cgm_holonomy_analysis_run.py

CLI report for CGM holonomy analysis (parts 1-2).

Writes experiments/cgm_holonomy_analysis_results.txt.
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parent
_REPO = _EXP.parent
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from cgm_holonomy_analysis_common import RESULTS_PATH, ReportState, Tee
from cgm_holonomy_analysis_1 import run_holonomy_1
from cgm_holonomy_analysis_2 import run_holonomy_2


def main() -> None:
    parser = argparse.ArgumentParser(description="CGM holonomy analysis report")
    parser.add_argument(
        "--part",
        type=int,
        choices=(1, 2),
        default=None,
        help="Run only report part 1 or 2 (debug)",
    )
    args = parser.parse_args()

    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = Tee(old, buf)
    try:
        print("CGM HOLONOMY ANALYSIS")
        print("=" * 5)
        print()
        state = ReportState()
        if args.part == 2:
            # part 2 needs carry fields; run part 1 into a discarded buffer first
            sys.stdout = old
            silent = io.StringIO()
            sys.stdout = Tee(silent)
            try:
                run_holonomy_1(state)
            finally:
                sys.stdout = Tee(old, buf)
            run_holonomy_2(state)
        elif args.part == 1:
            run_holonomy_1(state)
        else:
            run_holonomy_1(state)
            run_holonomy_2(state)
    finally:
        sys.stdout = old

    RESULTS_PATH.write_text(buf.getvalue(), encoding="utf-8")
    print(f"wrote {RESULTS_PATH}")
    if any(not ok for _, ok in state.gates):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
