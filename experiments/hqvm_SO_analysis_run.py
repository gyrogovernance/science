#!/usr/bin/env python3
"""hqvm_SO_analysis_run.py — Orchestrator for the SO(3) study.

Runs parts 1 and 2, captures output to results file.

Usage:
  python experiments/hqvm_SO_analysis_run.py
  python experiments/hqvm_SO_analysis_run.py --part 1
  python experiments/hqvm_SO_analysis_run.py --quick   (reduced sampling)
"""
from __future__ import annotations
import io, sys, argparse, time
from pathlib import Path
_EXP = Path(__file__).resolve().parent
if str(_EXP) not in sys.path: sys.path.insert(0, str(_EXP))

def _preflight() -> bool:
    """Fail fast with a clear message if dependencies are missing."""
    ok = True
    try:
        import numpy  # noqa: F401
    except ImportError:
        print('ERROR: numpy is required. Install with:  pip install numpy')
        ok = False
    try:
        import scipy  # noqa: F401
    except ImportError:
        print('ERROR: scipy is required. Install with:  pip install scipy')
        ok = False
    try:
        import mpmath  # noqa: F401
    except ImportError:
        print('ERROR: mpmath is required. Install with:  pip install mpmath')
        ok = False
    if not ok:
        print('Run:  python3 -m venv /tmp/so_venv && '
              '/tmp/so_venv/bin/pip install numpy scipy mpmath && '
              '/tmp/so_venv/bin/python3 experiments/hqvm_SO_analysis_run.py')
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description='SO(3) study runner')
    parser.add_argument('--part', type=int, choices=(1, 2), default=None,
                        help='Run only part 1 or 2')
    args = parser.parse_args()

    if not _preflight():
        sys.exit(2)

    from hqvm_SO_analysis_common import RESULTS_PATH, ReportState, Tee
    from hqvm_SO_analysis_1 import run_part1
    from hqvm_SO_analysis_2 import run_part2

    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = Tee(old, buf)

    print('=' * 60)
    print('hQVM SO(3) COMPLETE STUDY')
    print('Date: 2026-08-19')
    print('=' * 60)

    state = ReportState()
    t_all = time.perf_counter()
    try:
        if args.part in (1, None):
            t0 = time.perf_counter()
            run_part1(state)
            print(f'\n[part 1 finished in {time.perf_counter()-t0:.1f}s]', flush=True)
        if args.part in (2, None):
            t0 = time.perf_counter()
            run_part2(state)
            print(f'\n[part 2 finished in {time.perf_counter()-t0:.1f}s]', flush=True)
    finally:
        sys.stdout = old

    output = buf.getvalue()
    print(output, end='')

    RESULTS_PATH.write_text(output, encoding='utf-8')
    print(f'\nWrote {RESULTS_PATH}')

    passed = sum(1 for _, ok in state.gates if ok)
    failed = sum(1 for _, ok in state.gates if not ok)
    print(f'\n{"=" * 60}')
    print(f'SUMMARY: {passed} passed, {failed} failed out of {len(state.gates)} checks'
          f'  (total {time.perf_counter()-t_all:.1f}s)')
    print(f'{"=" * 60}')
    if failed:
        for label, ok in state.gates:
            if not ok:
                print(f'  FAIL: {label}')
        sys.exit(1)

if __name__ == '__main__':
    main()