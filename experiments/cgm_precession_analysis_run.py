#!/usr/bin/env python3
"""
CLI for CGM precession analysis.

Writes experiments/cgm_precession_analysis_results.txt.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parent
_REPO = _EXP.parent
for _p in (_EXP, _REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cgm_holonomy_analysis_common import Tee
from cgm_precession_analysis_1 import RESULTS_PATH, run
from cgm_precession_analysis_2 import run_mechanics, run_ontology


def main() -> None:
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = Tee(old, buf)
    try:
        gates, state = run()
        gates.extend(run_mechanics(state))
        run_ontology(state)
    finally:
        sys.stdout = old

    RESULTS_PATH.write_text(buf.getvalue(), encoding="utf-8")
    print(f"wrote {RESULTS_PATH}")
    if any(not ok for _, ok in gates):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
