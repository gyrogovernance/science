#!/usr/bin/env python3
"""
hQVM Cohomology Analysis -- runner and shared library
=====================================================

Runs hqvm_Cohomology_analysis_{1,2,3,4}.py, saving combined stdout/stderr to
hqvm_Cohomology_analysis_results.txt. This module also hosts the shared
kernel/geometry helpers imported by the analysis scripts (grading,
permutations, cycle signatures, Walsh observables, Boolean CHSH extreme).
Importing this module does not execute the runner; main() is guarded.

Usage:
  python experiments/hqvm_Cohomology_analysis_run.py
  python experiments/hqvm_Cohomology_analysis_run.py -o path/to/out.txt
  python experiments/hqvm_Cohomology_analysis_run.py --quick

Interpretation: docs/Findings/Analysis_hQVM_Cohomology.md (written separately).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from typing import Any
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_EXPERIMENTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EXPERIMENTS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gyroscopic.hQVM.api import (
    OMEGA_STATES_4096,
    state24_to_omega12,
    component12_to_spin6,
)
from gyroscopic.hQVM.constants import LAYER_MASK_12, step_state_by_byte

DEFAULT_OUTPUT = _EXPERIMENTS_DIR / "hqvm_Cohomology_analysis_results.txt"

SCRIPT = "hqvm_Cohomology_analysis_1.py"
SCRIPT_2 = "hqvm_Cohomology_analysis_2.py"
SCRIPT_3 = "hqvm_Cohomology_analysis_3.py"
SCRIPT_4 = "hqvm_Cohomology_analysis_4.py"

OMEGA_SIZE = len(OMEGA_STATES_4096)

# Loaded once; provides apply_word_to_state, W2_word, W2p_word, F_cycle_word
# without triggering the experiments package __init__ (emoji clean_pycache crash).
_COMMON: Any | None = None


def load_common() -> Any:
    """Load hqvm_gravity_common without triggering the experiments package.

    The experiments package __init__ runs an emoji-printing clean_pycache()
    at import time and crashes on non-utf8 consoles. Loading the module file
    directly avoids that side effect. Cached after first call.
    """
    global _COMMON
    if _COMMON is not None:
        return _COMMON
    import importlib.util as _ilu

    p = _REPO_ROOT / "experiments" / "hqvm_gravity_common.py"
    spec = _ilu.spec_from_file_location("hqvm_gravity_common_loaded", p)
    assert spec is not None and spec.loader is not None
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _COMMON = mod
    return mod


# Grading and permutations


def shell_grading() -> tuple[np.ndarray, dict[int, int]]:
    """Per-state shell index and a state->index lookup over OMEGA_STATES_4096."""
    shell_of = np.zeros(OMEGA_SIZE, dtype=np.int64)
    idx_of: dict[int, int] = {}
    for i, s in enumerate(OMEGA_STATES_4096):
        idx_of[s] = i
        shell_of[i] = state24_to_omega12(s).shell
    return shell_of, idx_of


def word_permutation(word: list[int], idx_of: dict[int, int]) -> np.ndarray:
    """Permutation of OMEGA_STATES_4096 indices induced by applying `word`."""
    common = load_common()
    apply_word_to_state = common.apply_word_to_state

    n = OMEGA_SIZE
    perm = np.empty(n, dtype=np.int64)
    for i, s in enumerate(OMEGA_STATES_4096):
        nxt = apply_word_to_state(word, s)
        perm[i] = idx_of[nxt]
    return perm


def byte_perm_of(b: int, idx_of: dict[int, int]) -> np.ndarray:
    """Permutation of OMEGA_STATES_4096 indices induced by byte b."""
    n = OMEGA_SIZE
    perm = np.empty(n, dtype=np.int64)
    for i, s in enumerate(OMEGA_STATES_4096):
        perm[i] = idx_of[step_state_by_byte(s, b)]
    return perm


def cycle_signature(perm: np.ndarray) -> dict[int, int]:
    """Cycle-length counts of a permutation (1 = fixed point)."""
    n = perm.shape[0]
    seen = np.full(n, False)
    sig: dict[int, int] = {}
    for i in range(n):
        if seen[i]:
            continue
        L = 0
        j = i
        while not seen[j]:
            seen[j] = True
            j = int(perm[j])
            L += 1
        sig[L] = sig.get(L, 0) + 1
    return sig


def all_byte_perms(idx_of: dict[int, int]) -> list[np.ndarray]:
    """All 256 byte permutations as a precomputed list (cache once per run)."""
    return [byte_perm_of(b, idx_of) for b in range(256)]


# CHSH Boolean extreme (Walsh observables)


def spin_pair_of_face(face12: int) -> tuple[int, ...]:
    """Six dipole spins in {-1,+1} from a 12-bit face (pair 10 -> +1, 01 -> -1)."""
    return component12_to_spin6(int(face12) & LAYER_MASK_12)


def walsh_matrix(spins: np.ndarray) -> np.ndarray:
    """Walsh/parity observation matrix over non-constant masks (1..63)."""
    N = spins.shape[0]
    out = np.ones((N, 64), dtype=np.float64)
    for m in range(1, 64):
        bits = [i for i in range(6) if (m >> i) & 1]
        out[:, m] = spins[:, bits].prod(axis=1)
    return out


def max_chsh_on_index_set(idx: list[int]) -> dict[str, object]:
    """Max Boolean CHSH over a subset of Omega states (Walsh/parity observables).

    For states in `idx`, extract the 6 dipole spins of each face, build the
    (N,64) Walsh observation matrices, form the (63,63) correlation matrix over
    non-constant masks, and maximize S = E[a0,b0] + E[a0,b1] + E[a1,b0] - E[a1,b1]
    exactly over all 63^2 (a0,a1) choices using NumPy broadcasting:
        S(a0,a1) = max_b0 (C[a0,b0] + C[a1,b0]) + max_b1 (C[a0,b1] - C[a1,b1]).
    """
    a_spins = []
    b_spins = []
    for i in idx:
        s = OMEGA_STATES_4096[i]
        a12, b12 = (s >> 12) & LAYER_MASK_12, s & LAYER_MASK_12
        try:
            a_spins.append(spin_pair_of_face(a12))
            b_spins.append(spin_pair_of_face(b12))
        except ValueError:
            continue
    if not a_spins:
        return {
            "n_states": 0,
            "max_CHSH_Boolean": float("nan"),
            "max_CHSH_masks": None,
            "max_abs_single_dipole_corr": float("nan"),
            "max_abs_corr": float("nan"),
        }
    A_obs = walsh_matrix(np.array(a_spins, dtype=np.float64))
    B_obs = walsh_matrix(np.array(b_spins, dtype=np.float64))
    N = A_obs.shape[0]

    masks = list(range(1, 64))
    C = (A_obs[:, masks].T @ B_obs[:, masks]) / N

    M = C.shape[0]
    C_add = C[:, None, :] + C[None, :, :]  # (a0, a1, b0)
    C_sub = C[:, None, :] - C[None, :, :]  # (a0, a1, b1)
    S = np.max(C_add, axis=2) + np.max(C_sub, axis=2)
    best = float(np.max(S))
    a0, a1 = divmod(int(np.argmax(S)), M)
    row = C[a0] + C[a1]
    b0 = int(np.argmax(row))
    row2 = C[a0] - C[a1]
    b1 = int(np.argmax(row2))
    best_tuple = (masks[a0], masks[a1], masks[b0], masks[b1])

    single = (A_obs[:, 1:7].T @ B_obs[:, 1:7]) / N
    max_abs_single = float(np.max(np.abs(single)))
    max_abs_corr = float(np.max(np.abs(C)))
    return {
        "n_states": N,
        "max_CHSH_Boolean": best,
        "max_CHSH_masks": best_tuple,
        "max_abs_single_dipole_corr": max_abs_single,
        "max_abs_corr": max_abs_corr,
    }


def _configure_stdout_utf8() -> None:
    import codecs

    if hasattr(sys.stdout, "buffer"):
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")


def run_script(
    script_name: str,
    timeout_s: float | None,
    extra_args: list[str] | None = None,
) -> tuple[int, str, str, float]:
    path = _EXPERIMENTS_DIR / script_name
    if not path.is_file():
        return 127, "", f"missing file: {path}\n", 0.0

    cmd = [sys.executable, str(path)]
    if extra_args:
        cmd.extend(extra_args)
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(_EXPERIMENTS_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        dt = time.perf_counter() - t0
        out = exc.stdout or ""
        err = str(exc.stderr or "") + f"\nTIMEOUT after {timeout_s}s\n"
        return 124, str(out), str(err), dt

    return (
        proc.returncode,
        proc.stdout or "",
        proc.stderr or "",
        time.perf_counter() - t0,
    )


def format_block(
    script_name: str, code: int, stdout: str, stderr: str, dt: float
) -> str:
    lines = [f"######## {script_name} ########", ""]
    if stdout:
        lines.append(stdout.rstrip())
        lines.append("")
    if stderr:
        lines.append("--- stderr ---")
        lines.append(stderr.rstrip())
        lines.append("")
    lines.append(f"exit={code}  duration={dt:.2f}s")
    lines.append("")
    return "\n".join(lines)


def run_all(output_path: Path, timeout_s: float | None) -> int:
    started = datetime.now(timezone.utc).astimezone()
    blocks: list[str] = [
        "hQVM Cohomology analysis",
        f"started: {started.isoformat(timespec='seconds')}",
        f"python: {sys.executable}",
        "",
    ]

    print(f"Running {SCRIPT} ...", flush=True)
    code, out, err, dt = run_script(SCRIPT, timeout_s)
    blocks.append(format_block(SCRIPT, code, out, err, dt))
    status = "ok" if code == 0 else f"exit {code}"
    print(f"  {status} ({dt:.1f}s)", flush=True)

    print(f"Running {SCRIPT_2} ...", flush=True)
    code2, out2, err2, dt2 = run_script(SCRIPT_2, timeout_s)
    blocks.append(format_block(SCRIPT_2, code2, out2, err2, dt2))
    status2 = "ok" if code2 == 0 else f"exit {code2}"
    print(f"  {status2} ({dt2:.1f}s)", flush=True)
    code = max(code, code2)
    dt = dt + dt2

    print(f"Running {SCRIPT_3} ...", flush=True)
    code3, out3, err3, dt3 = run_script(SCRIPT_3, timeout_s)
    blocks.append(format_block(SCRIPT_3, code3, out3, err3, dt3))
    status3 = "ok" if code3 == 0 else f"exit {code3}"
    print(f"  {status3} ({dt3:.1f}s)", flush=True)
    code = max(code, code3)
    dt = dt + dt3

    print(f"Running {SCRIPT_4} ...", flush=True)
    code4, out4, err4, dt4 = run_script(SCRIPT_4, timeout_s)
    blocks.append(format_block(SCRIPT_4, code4, out4, err4, dt4))
    status4 = "ok" if code4 == 0 else f"exit {code4}"
    print(f"  {status4} ({dt4:.1f}s)", flush=True)
    code = max(code, code4)
    dt = dt + dt4

    blocks.append(
        f"finished: {datetime.now().astimezone().isoformat(timespec='seconds')}"
    )
    blocks.append(f"total_duration={dt:.2f}s")
    blocks.append(f"worst_exit={code}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(blocks), encoding="utf-8")
    print(f"Wrote {output_path}")
    return code


def main() -> None:
    _configure_stdout_utf8()
    parser = argparse.ArgumentParser(
        description="Run hQVM Cohomology analysis; save output to a text file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output file (default: {DEFAULT_OUTPUT.name})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        metavar="SEC",
        help="Per-script timeout in seconds (default: none)",
    )
    args = parser.parse_args()

    print("hQVM Cohomology analysis -- runner")
    print("=" * 5)
    print(f"  Output: {args.output.resolve()}")
    raise SystemExit(run_all(args.output.resolve(), args.timeout))


if __name__ == "__main__":
    main()
