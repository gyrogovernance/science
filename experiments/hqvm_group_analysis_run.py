#!/usr/bin/env python3
"""hqvm_group_analysis_run.py — Orchestrator for the hQVM group analysis study.

Usage:
  python experiments/hqvm_group_analysis_run.py
  python experiments/hqvm_group_analysis_run.py --only 1
  python experiments/hqvm_group_analysis_run.py --only 2
  python experiments/hqvm_group_analysis_run.py --only 3
  python experiments/hqvm_group_analysis_run.py --only 4
  python experiments/hqvm_group_analysis_run.py --only 5
  python experiments/hqvm_group_analysis_run.py --only all
  python experiments/hqvm_group_analysis_run.py --verbose
"""
from __future__ import annotations
import io, sys, argparse, time, importlib
from pathlib import Path

_EXP = Path(__file__).resolve().parent
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

SCRIPTS = {
    '1': 'hqvm_group_analysis_1.py',
    '2': 'hqvm_group_analysis_2.py',
    '3': 'hqvm_group_analysis_3.py',
    '4': 'hqvm_group_analysis_4.py',
    '5': 'hqvm_group_analysis_5.py',
}


def _preflight() -> bool:
    ok = True
    for pkg, hint in (('numpy', 'numpy'), ('scipy', 'scipy'), ('mpmath', 'mpmath')):
        try:
            __import__(pkg)
        except ImportError:
            print(f'ERROR: {pkg} is required. Install with:  pip install {hint}')
            ok = False
    return ok


def _load_runner(num: str):
    mod = importlib.import_module(f'hqvm_group_analysis_{num}')
    return mod.run


def main():
    choices = tuple(SCRIPTS) + ('all',)
    parser = argparse.ArgumentParser(description='hQVM group analysis runner')
    parser.add_argument('--only', choices=choices,
                        default='all', help='Run one script or all (default: all)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print sample tables and flat dumps (default: quiet)')
    args = parser.parse_args()

    if not _preflight():
        sys.exit(2)

    from hqvm_group_analysis_common import (
        RESULTS_PATH, ReportState, Tee, set_verbose,
    )
    set_verbose(args.verbose)

    selected = list(SCRIPTS) if args.only == 'all' else [args.only]

    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = Tee(old, buf)

    print('=' * 5)
    print('hQVM GROUP ANALYSIS')
    print('=' * 5)

    state = ReportState()
    t_all = time.perf_counter()
    try:
        for num in selected:
            t0 = time.perf_counter()
            _load_runner(num)(state)
            print(f'\n[{SCRIPTS[num]} finished in {time.perf_counter()-t0:.1f}s]',
                  flush=True)
    finally:
        sys.stdout = old

    passed = sum(1 for _, ok in state.gates if ok)
    failed = sum(1 for _, ok in state.gates if not ok)
    summary = (
        f'\n{"=" * 5}\n'
        f'SUMMARY: {passed} passed, {failed} failed out of {len(state.gates)} checks'
        f'  (total {time.perf_counter()-t_all:.1f}s)\n'
        f'{"=" * 5}\n'
    )
    if failed:
        summary += ''.join(f'  FAIL: {label}\n' for label, ok in state.gates if not ok)

    output = buf.getvalue() + summary
    print(output, end='')
    RESULTS_PATH.write_text(output, encoding='utf-8')
    print(f'\nWrote {RESULTS_PATH}')
    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
