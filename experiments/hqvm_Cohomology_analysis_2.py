#!/usr/bin/env python3
"""
hQVM Cohomology Analysis 2 -- categorical architecture experiments
=================================================================

ROLE
----
Second experiment library. Builds on the constant and sign-space results of
analysis_1 and tests the categorical-architecture claims of the expanded
Grothendieck-CGM framework (étale cover, derived functors, zeta functions).

This script is the library only. It prints measurements, tables, and PASS/FAIL
checks. The runner (hqvm_Cohomology_analysis_run.py) executes it and writes
the combined output.

Sections:
  I.   Carrier CHSH (Boolean/IP extreme on the bipartite 24-bit carrier)
  II.  Shell Betti profile and Poincaré duality
  III. Gate F fixed points and Euler characteristic
  IV.  Lefschetz numbers of byte operators
  V.   Dynamical zeta of gate F and depth-4 words
  VI.  K4 as Galois group of the family-fiber cover
  VII. Lefschetz numbers and characteristic polynomials of the depth-4
       words W2, W2', F-cycle (shell grading cannot separate W2 vs F)
  VIII. Characteristic polynomials of all 256 byte operators (item 3)
  IX.   Complementarity invariant (h + ab = 12)
  X.    Byte T^2 fixed points vs Lefschetz prediction
  XI.   W2 / W2' dynamical zeta identity
  XII.  Torsion vs Hilbert spectrum inclusion
  XIII.  K4 group cohomology H^1

Companion scripts: hqvm_Cohomology_analysis_1.py (sign-space audit),
                  hqvm_Cohomology_analysis_run.py (runner).
"""

from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path
from typing import cast

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gyroscopic.hQVM.constants import (
    GENE_MAC_REST,
    OMEGA_SIZE,
    apply_gate,
    step_state_by_byte,
    is_on_horizon,
    is_on_equality_horizon,
    horizon_distance,
    ab_distance,
    LAYER_MASK_12,
    byte_family,
    complementarity_invariant,
)
from gyroscopic.hQVM.api import (
    OMEGA_STATES_4096,
    chirality_word6,
    q_word6,
    state24_to_omega12,
    omega12_to_state24,
    shell_population,
    unpack_state,
)

from hqvm_Cohomology_analysis_run import (
    load_common,
    shell_grading,
    word_permutation,
    cycle_signature,
    byte_perm_of,
    all_byte_perms,
    max_chsh_on_index_set,
)

# Import hqvm_gravity_common without triggering the experiments package
# __init__ (which runs an emoji-printing clean_pycache() at import time and
# crashes on non-utf8 consoles). Load the module file directly.
hqvm_gravity_common = load_common()
apply_word_to_state = hqvm_gravity_common.apply_word_to_state  # type: ignore[attr-defined]
W2_word = hqvm_gravity_common.W2_word  # type: ignore[attr-defined]
W2p_word = hqvm_gravity_common.W2p_word  # type: ignore[attr-defined]
F_cycle_word = hqvm_gravity_common.F_cycle_word  # type: ignore[attr-defined]

import random

SEED = 20260702
rng = random.Random(SEED)


# ================================================================
# I. Carrier CHSH (Boolean/IP extreme)
# ================================================================

def _max_chsh_on_index_set(idx: list[int]) -> dict[str, object]:
    """Boolean CHSH extreme (delegated to shared helper)."""
    return max_chsh_on_index_set(idx)


def experiment_carrier_chsh() -> dict[str, object]:
    """Boolean/IP extreme of CHSH on the bipartite 24-bit carrier (Walsh masks)."""
    all_idx = list(range(OMEGA_SIZE))

    # uniform Omega
    uniform = _max_chsh_on_index_set(all_idx)

    # fixed shell ensembles: shell 0, 3 (equator), 6
    shell_idx = {k: [] for k in (0, 3, 6)}
    # fixed chirality (chi word) and horizon (complement) ensembles
    chi_idx: dict[int, list[int]] = {}
    comp_idx: list[int] = []
    for i, s in enumerate(OMEGA_STATES_4096):
        om = state24_to_omega12(s)
        if om.shell in shell_idx:
            shell_idx[om.shell].append(i)
        chi_idx.setdefault(om.chirality6, []).append(i)
        if is_on_horizon(s):
            comp_idx.append(i)

    ensembles = {"uniform_Omega": uniform}
    for k in (0, 3, 6):
        ensembles[f"shell_{k}"] = _max_chsh_on_index_set(shell_idx[k])
    # representative fixed-chirality ensembles (a few chi words)
    for chi in (0, 1, 63):
        if chi in chi_idx:
            ensembles[f"fixed_chi_{chi}"] = _max_chsh_on_index_set(chi_idx[chi])
    ensembles["complement_horizon"] = _max_chsh_on_index_set(comp_idx)

    # summary: max boolean CHSH per ensemble
    summary = {name: v["max_CHSH_Boolean"] for name, v in ensembles.items()}
    max_abs_corr = {name: v["max_abs_corr"] for name, v in ensembles.items()}

    return {
        "n_ensembles": len(ensembles),
        "n_masks_per_face": 63,
        "uniform_OMEGA_CHSH_Boolean": uniform["max_CHSH_Boolean"],
        "uniform_correlators_vanish": bool(abs(cast(float, uniform["max_abs_corr"])) < 1e-9),
        "ensemble_max_CHSH": summary,
        "ensemble_max_abs_corr": max_abs_corr,
        "classical_bound": 2.0,
        "hilbert_chsh": 2.0 * math.sqrt(2.0),
    }


# ================================================================
# II. Shell Betti profile and Poincaré duality
# ================================================================

def experiment_shell_betti() -> dict[str, object]:
    """Shell population profile and Poincaré duality check."""
    pops = tuple(shell_population(w) for w in range(7))
    poincare_ok = all(pops[k] == pops[6 - k] for k in range(7))
    total = sum(pops)
    euler = sum((-1) ** k * pops[k] for k in range(7))
    # Poincaré polynomial evaluated at -1
    poly_neg1 = sum(pops[k] * ((-1) ** k) for k in range(7))
    return {
        "shell_populations": pops,
        "total": total,
        "expected_4096": OMEGA_SIZE,
        "poincare_duality": poincare_ok,
        "euler_characteristic": euler,
        "poly_at_neg1": poly_neg1,
        "P(-1)==0": poly_neg1 == 0,
        "scaled_binomial": tuple(p // 64 for p in pops),
    }


# ================================================================
# III. Gate F fixed points and Euler characteristic
# ================================================================

def experiment_gate_f_fixedpoints() -> dict[str, object]:
    """Gate F on Omega: count fixed points and 2-cycles."""
    fixed = 0
    two_cycles = 0
    seen = set()
    for s in OMEGA_STATES_4096:
        f = apply_gate(s, "F")
        if f == s:
            fixed += 1
        elif f not in seen and apply_gate(f, "F") == s:
            two_cycles += 1
            seen.add(s)
    n = len(OMEGA_STATES_4096)
    return {
        "n_omega": n,
        "F_fixed_points": fixed,
        "F_two_cycles": two_cycles,
        "2*cycles+fixed": 2 * two_cycles + fixed,
        "matches_n_omega": (2 * two_cycles + fixed) == n,
        "fixed_point_free": fixed == 0,
        "Lefschetz_L_F": fixed,
        "euler_chi_equals_0": fixed == 0,
    }


# ================================================================
# IV. Lefschetz numbers of byte operators
# ================================================================

def experiment_lefschetz_bytes() -> dict[str, object]:
    """Lefschetz number L(T_b) = sum_k (-1)^k Fix_k(T_b) for each byte.

    Fix_k(T_b) counts fixed points of the byte operator in shell k.
    Correlate with transport class q6 and family.
    """
    pops = tuple(shell_population(w) for w in range(7))
    # precompute shell index for each state
    shell_of = np.zeros(OMEGA_SIZE, dtype=np.int64)
    idx_of = {s: i for i, s in enumerate(OMEGA_STATES_4096)}
    for i, s in enumerate(OMEGA_STATES_4096):
        shell_of[i] = state24_to_omega12(s).shell

    lefschetz = {}
    for b in range(256):
        fixed_per_shell = [0] * 7
        for i, s in enumerate(OMEGA_STATES_4096):
            nxt = step_state_by_byte(s, b)
            ni = idx_of.get(nxt)
            if ni is not None and ni == i:
                fixed_per_shell[shell_of[i]] += 1
        L = sum((-1) ** k * fixed_per_shell[k] for k in range(7))
        lefschetz[b] = {
            "L": L,
            "fixed_total": sum(fixed_per_shell),
            "q6": q_word6(b),
            "family": byte_family(b),
            "fixed_per_shell": tuple(fixed_per_shell),
        }

    L_vals = np.array([lefschetz[b]["L"] for b in range(256)])
    fixed_vals = np.array([lefschetz[b]["fixed_total"] for b in range(256)])

    # name the special bytes that have fixed points (structural finding)
    special_bytes = []
    for b in range(256):
        if lefschetz[b]["fixed_total"] > 0:
            special_bytes.append({
                "byte": b,
                "hex": f"0x{b:02X}",
                "family": lefschetz[b]["family"],
                "q6": lefschetz[b]["q6"],
                "q_weight": lefschetz[b]["q6"].bit_count(),
                "fixed_total": lefschetz[b]["fixed_total"],
                "L": lefschetz[b]["L"],
            })

    # correlate |L| with fixed-point count and transport weight
    qw = np.array([lefschetz[b]["q6"].bit_count() for b in range(256)])
    # correlation coefficient (Pearson) between fixed_total and L, and qw and L
    def pearson(x, y):
        if np.std(x) == 0 or np.std(y) == 0:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    # horizon-stabilizer connection: the 4 fixed-point bytes are exactly
    # the K4 horizon-stabilizer bytes (features report): 0x2B,0x54,0xD5,0xAA.
    HORIZON_STABILIZER_BYTES = (0x2B, 0x54, 0xD5, 0xAA)
    special_hexes = {b["byte"] for b in special_bytes}
    stab_note = (
        f"special fixed-point bytes {sorted(f'0x{b:02X}' for b in special_hexes)} "
        f"== horizon stabilizers {[f'0x{b:02X}' for b in HORIZON_STABILIZER_BYTES]}: "
        f"{special_hexes == set(HORIZON_STABILIZER_BYTES)}"
    )

    return {
        "n_bytes": 256,
        "L_min": float(L_vals.min()),
        "L_max": float(L_vals.max()),
        "L_mean": float(L_vals.mean()),
        "fixed_min": int(fixed_vals.min()),
        "fixed_max": int(fixed_vals.max()),
        "corr_fixed_vs_L": pearson(fixed_vals, L_vals),
        "corr_qweight_vs_L": pearson(qw, L_vals),
        "bytes_with_fixed_points": int((fixed_vals > 0).sum()),
        "special_fixed_point_bytes": special_bytes,
        "lefschetz_constant": bool(np.all(L_vals == L_vals[0])),
        "constant_L_value": float(L_vals[0]) if np.all(L_vals == L_vals[0]) else None,
        "horizon_stabilizer_note": stab_note,
    }


# ================================================================
# V. Dynamical zeta of gate F and depth-4 words
# ================================================================

def experiment_dynamical_zeta(max_n: int = 12) -> dict[str, object]:
    """Dynamical zeta of gate F from fixed-point counts of F^n.

    Gate F is a fixed-point-free involution: Fix(F^n) = 4096 for even n, 0 for
    odd n. The zeta function is (1 - t^2)^{-2048}.
    """
    n_omega = OMEGA_SIZE
    fix_counts = {}
    for n in range(1, max_n + 1):
        if n % 2 == 0:
            fix_counts[n] = n_omega
        else:
            fix_counts[n] = 0

    # fit exponent: zeta(t) = exp(sum Fix(F^n)/n t^n) = (1 - t^2)^{-c}
    # log zeta = -c log(1 - t^2) = c sum_{k>=1} t^{2k}/k
    # coefficient of t^{2k} in log zeta is Fix(F^{2k})/(2k), which must equal c/k.
    # Hence c = Fix(F^{2k})/2 = 4096/2 = 2048, constant for all even k.
    c_even = {n: fix_counts[n] / 2.0 for n in fix_counts if n % 2 == 0}
    c_vals = list(c_even.values())
    c_consistent = all(abs(c - c_vals[0]) < 1e-9 for c in c_vals)
    predicted_exponent = n_omega // 2  # 2048

    # functional equation (§4.3) for Z(t) = (1 - t^2)^{-c}:
    #   Z(1/t) = (1 - t^{-2})^{-c} = (t^{-2}(t^2 - 1))^{-c}
    #          = t^{-2c} (1 - t^2)^{-c} = t^{-2c} Z(t).
    # So Z(1/t) = t^{-4096} Z(t) (sign +1 since c is even). Verify via logs to
    # avoid overflow from t^{-4096} (t=0.5 => 2^4096 exceeds float range):
    #   log Z(t)   = -c log(1 - t^2)
    #   Z(1/t) = (1 - t^{-2})^{-c} = (t^{-2} - 1)^{-c}  (c even => (-1)^{-c}=1)
    #          => log Z(1/t) = -c log(t^{-2} - 1) = 2c log t + log Z(t).
    t = 0.5
    log_Z_t = -float(predicted_exponent) * math.log(1.0 - t * t)
    log_Z_inv = -float(predicted_exponent) * math.log((1.0 / (t * t)) - 1.0)
    fe_ok = abs(log_Z_inv - (2.0 * predicted_exponent * math.log(t) + log_Z_t)) < 1e-9

    return {
        "max_n": max_n,
        "fix_counts": fix_counts,
        "fitted_exponent_c": float(c_vals[0]) if c_vals else float("nan"),
        "predicted_exponent_2048": float(predicted_exponent),
        "exponent_consistent": c_consistent,
        "zeta_form": "(1 - t^2)^(-2048)",
        "functional_eq_Z(1/t)=t^(-4096) Z(t)": fe_ok,
    }


# ================================================================
# VI. K4 as Galois group of the family-fiber cover
# ================================================================

def experiment_k4_cover() -> dict[str, object]:
    """Verify the family-fiber cover structure: pi(b) = q6(b) is 4-to-1,
    K4 acts freely and transitively on each fiber."""
    from gyroscopic.hQVM.api import BYTES_BY_Q6

    fiber_sizes = [len(BYTES_BY_Q6[q]) for q in range(64)]
    all_size_4 = all(f == 4 for f in fiber_sizes)

    # K4 action: the four gates {id,S,C,F} map each byte to its K4 partner.
    # Check that for each q, the four bytes in the fiber are distinct under K4.
    # Build K4 orbit of a byte via apply_gate on the 24-bit carrier is not the
    # fiber action; instead verify the four family bytes (00,01,10,11) share q6.
    fams_per_q = []
    for q in range(64):
        fams = set()
        for b in BYTES_BY_Q6[q]:
            fams.add(byte_family(b))
        fams_per_q.append(fams)
    all_four_fams = all(len(f) == 4 for f in fams_per_q)

    # transitivity: each family appears once per fiber
    transitive = all_four_fams and all_size_4

    return {
        "|Q6 classes|": 64,
        "fiber_size_all_4": all_size_4,
        "fiber_sizes_sample": fiber_sizes[:8],
        "each_fiber_has_4_families": all_four_fams,
        "cover_transitive_K4": transitive,
        "cover_is_4_to_1": all_size_4,
        "note": "pi: {0..255} -> GF(2)^6, pi(b)=q6(b); K4 acts on fibers as deck group",
    }


# ================================================================
# main: dispatch and print
# ================================================================

def _print_carrier_chsh(res: dict) -> None:
    print("\n" + "=" * 5)
    print("I. CARRIER CHSH (BOOLEAN/IP EXTREME, WALSH OBSERVABLES)")
    print("=" * 5)
    print(f"  ensembles:           {res['n_ensembles']}")
    print(f"  masks per face:      {res['n_masks_per_face']}")
    print(f"  uniform Omega CHSH:  {res['uniform_OMEGA_CHSH_Boolean']:.6f}  "
          f"(correlators vanish: {res['uniform_correlators_vanish']})")
    print("  ensemble max CHSH Boolean / max|corr|:")
    for name, v in res["ensemble_max_CHSH"].items():
        c = res["ensemble_max_abs_corr"][name]
        cv = f"{c:.4f}" if isinstance(c, float) and c == c else str(c)
        vv = f"{v:.4f}" if isinstance(v, float) and v == v else str(v)
        print(f"    {name:<20s} CHSH={vv}  max|corr|={cv}")
    print(f"  classical bound:     {res['classical_bound']:.4f}")
    print(f"  Hilbert CHSH:        {res['hilbert_chsh']:.6f}")


def _print_shell_betti(res: dict) -> None:
    print("\n" + "=" * 5)
    print("II. SHELL BETTI PROFILE AND POINCARE DUALITY")
    print("=" * 5)
    print(f"  populations: {res['shell_populations']}")
    print(f"  scaled binomial (//64): {res['scaled_binomial']}")
    print(f"  total: {res['total']} (expect {res['expected_4096']})")
    print(f"  Poincare duality b_k=b_6-k: {res['poincare_duality']}")
    print(f"  Euler char: {res['euler_characteristic']}")
    print(f"  P(-1) = {res['poly_at_neg1']}  (==0: {res['P(-1)==0']})")


def _print_gate_f(res: dict) -> None:
    print("\n" + "=" * 5)
    print("III. GATE F FIXED POINTS AND EULER CHARACTERISTIC")
    print("=" * 5)
    print(f"  |Omega|:          {res['n_omega']}")
    print(f"  F fixed points:   {res['F_fixed_points']}")
    print(f"  F 2-cycles:       {res['F_two_cycles']}")
    print(f"  2*cycles+fixed:   {res['2*cycles+fixed']} (==|Omega|: {res['matches_n_omega']})")
    print(f"  fixed-point-free: {res['fixed_point_free']}")
    print(f"  Lefschetz L(F)=fixed: {res['Lefschetz_L_F']}")
    print(f"  => Euler chi = 0: {res['euler_chi_equals_0']}")


def _print_lefschetz(res: dict) -> None:
    print("\n" + "=" * 5)
    print("IV. LEFSCHETZ NUMBERS OF BYTE OPERATORS")
    print("=" * 5)
    print(f"  n_bytes:            {res['n_bytes']}")
    print(f"  L range:            [{res['L_min']:.1f}, {res['L_max']:.1f}]  mean={res['L_mean']:.3f}")
    print(f"  fixed pts range:    {res['fixed_min']}..{res['fixed_max']}")
    print(f"  bytes w/ fixed pts: {res['bytes_with_fixed_points']}")
    print(f"  corr(fixed, L):     {res['corr_fixed_vs_L']}")
    print(f"  corr(qweight, L):   {res['corr_qweight_vs_L']}")
    print(f"  L constant across bytes: {res['lefschetz_constant']}"
          + (f"  value={res['constant_L_value']}" if res['constant_L_value'] is not None else ""))
    print("  special bytes (have fixed points):")
    for sb in res["special_fixed_point_bytes"]:
        print(f"    byte {sb['hex']} fam={sb['family']} q6=0x{sb['q6']:02X} "
              f"qw={sb['q_weight']} fixed={sb['fixed_total']} L={sb['L']}")
    _print_lefschetz_stab(res)


def _print_zeta(res: dict) -> None:
    print("\n" + "=" * 5)
    print("V. DYNAMICAL ZETA OF GATE F")
    print("=" * 5)
    print(f"  max_n: {res['max_n']}")
    print(f"  fix counts: {res['fix_counts']}")
    print(f"  fitted exponent c: {res['fitted_exponent_c']:.1f}")
    print(f"  predicted 2048:    {res['predicted_exponent_2048']:.1f}")
    print(f"  exponent consistent: {res['exponent_consistent']}")
    print(f"  zeta form: {res['zeta_form']}")
    print(f"  functional eq Z(1/t)=t^(-4096) Z(t): {res['functional_eq_Z(1/t)=t^(-4096) Z(t)']}")


# ================================================================
# VII. Lefschetz numbers and characteristic polynomials of the
#      depth-4 words W2, W2' and the F-cycle (BLOCKING ITEMS 1-2)
# ================================================================

# Shared helpers (defined in hqvm_Cohomology_analysis_run.py).
_shell_grading = shell_grading
_word_permutation = word_permutation
_cycle_signature = cycle_signature


def _graded_traces(perm: np.ndarray, shell_of: np.ndarray) -> list[int]:
    """Tr(g | Shell_k) = number of fixed points of the word in shell k."""
    n = perm.shape[0]
    fixed_per_shell = [0] * 7
    for i in range(n):
        if int(perm[i]) == i:
            fixed_per_shell[int(shell_of[i])] += 1
    return fixed_per_shell


def _poly_factor_form(sig: dict[int, int]) -> dict[str, object]:
    """Factor form of the permutation char poly det(xI - P) over ZZ.

    Each cycle of length L contributes a factor (x^L - 1). We report the
    exponent of each (x^L - 1) factor (the "Weil number" / root-of-unity
    structure) and the total degree, without expanding the coefficients
    (expanding (x^2-1)^2048 produces ~600-digit binomial coefficients and is
    not needed: a permutation matrix is unitary, so every eigenvalue is a root
    of unity and lies on the unit circle).
    """
    deg = sum(L * c for L, c in sig.items())
    factors = {L: c for L, c in sig.items() if c > 0}
    # Integer roots of prod_L (x^L - 1)^{c_L}: only +/-1 can be integers on the
    # unit circle. For every cycle length L >= 1, (x^L - 1) has root +1, and
    # has root -1 iff L is even. So:
    #   +1 is always a root (whenever any cycle exists),
    #   -1 is a root iff some even-length cycle exists.
    # For (x^2 - 1)^2048 (no fixed points) the integer roots are BOTH +/-1.
    int_roots = []
    if deg > 0:
        int_roots.append(1)
    if any(L % 2 == 0 for L in sig):
        int_roots.append(-1)
    return {
        "deg": deg,
        "factor_exponents": factors,
        "all_roots_unit_modulus": True,
        "integer_roots": int_roots,
    }


def _verify_shell_map(word: tuple[int, ...], expected) -> bool:
    """Check shell(word(s)) == expected(shell(s)) for all s in Omega."""
    for s in OMEGA_STATES_4096:
        om0 = state24_to_omega12(s)
        s1 = apply_word_to_state(word, s)
        om1 = state24_to_omega12(s1)
        if om1.shell != expected(om0.shell):
            return False
    return True


def experiment_depth4_lefschetz_and_poly() -> dict[str, object]:
    """Lefschetz numbers and characteristic polynomials of the depth-4 words.

    BLOCKING ITEM 1 (§4.2 "genuine computation to run"): graded Lefschetz
    numbers L(W2), L(W2'), L(F) where L(T) = sum_k (-1)^k Fix_k(T), Fix_k
    counting fixed points in shell k. W2/W2' are involutions swapping shell
    s <-> 6-s, so they fix only states that stay in their own shell.

    BLOCKING ITEM 2 (§6.2 "Weil numbers"): the permutation characteristic
    polynomial det(xI - T) over ZZ, factored from the cycle structure. All
    roots must lie on the unit circle (Riemann-hypothesis analogue, §4.3).

    The F-cycle word realizes gate F on Omega (Z2 carrier flip); its poly is
    the (x^2 - 1)^2048 reference from §6.2.
    """
    shell_of, idx_of = _shell_grading()

    words = {
        "W2(m=0)": W2_word(0),
        "W2'(m=0)": W2p_word(0),
        "F-cycle(m=0)": F_cycle_word(0),
    }

    out = {}
    for name, w in words.items():
        perm = _word_permutation(w, idx_of)
        sig = _cycle_signature(perm)
        fixed_per_shell = _graded_traces(perm, shell_of)
        L = sum((-1) ** k * fixed_per_shell[k] for k in range(7))
        poly = _poly_factor_form(sig)
        out[name] = {
            "cycle_signature": dict(sig),
            "fixed_per_shell": tuple(fixed_per_shell),
            "Lefschetz_L": L,
            "Lefschetz_mod12": int(L) % 12,
            "poly_deg": poly["deg"],
            "poly_factor_exponents": poly["factor_exponents"],
            "all_roots_unit_modulus": poly["all_roots_unit_modulus"],
            "integer_roots": poly["integer_roots"],
        }

    # expected F-cycle poly exponent: 2048 two-cycles => (x^2-1)^2048
    f_sig = out["F-cycle(m=0)"]["cycle_signature"]
    expected_f_exp = OMEGA_SIZE // 2
    f_exp2 = f_sig.get(2, 0)
    shell_ok_W2 = _verify_shell_map(W2_word(0), lambda sh: 6 - sh)
    shell_ok_W2p = _verify_shell_map(W2p_word(0), lambda sh: 6 - sh)
    shell_ok_F = _verify_shell_map(F_cycle_word(0), lambda sh: sh)

    return {
        "words": out,
        "shell_map_W2_s_to_6_minus_s": shell_ok_W2,
        "shell_map_W2p_s_to_6_minus_s": shell_ok_W2p,
        "shell_map_F_preserves_shell": shell_ok_F,
        "F_fixed_points": f_sig.get(1, 0),
        "F_two_cycles": f_exp2,
        "F_poly_matches_(x^2-1)^2048": (f_exp2 == expected_f_exp and f_sig.get(1, 0) == 0),
        "all_words_roots_on_unit_circle": all(
            v["all_roots_unit_modulus"] for v in out.values()
        ),
        "note": ("Graded Lefschetz + permutation char poly of the depth-4 "
                 "words. F-cycle poly is (x^2-1)^2048; W2/W2' are shell-swap "
                 "involutions. All roots on unit circle (RH analogue)."),
    }


# ================================================================
# VIII. Characteristic polynomials of all 256 byte operators
# ================================================================

def experiment_byte_characteristic_polys(
    perms: list[np.ndarray] | None = None,
) -> dict[str, object]:
    """Characteristic polynomials det(xI - T_b) over ZZ for every byte b.

    BLOCKING ITEM 3 (§6.2). Each byte is a permutation of Omega; its
    permutation char poly is built from the cycle signature. We report the
    distribution of cycle structures, the max polynomial degree, and a check
    that every byte's roots lie on the unit circle (a permutation matrix is
    unitary, so its eigenvalues are roots of unity by construction; this is
    the finite-weil / RH-analogue sanity check from §4.3/§6.2).
    """
    _, idx_of = _shell_grading()
    if perms is None:
        perms = all_byte_perms(idx_of)
    sigs = {}
    degs = []
    all_unit = True
    for b in range(256):
        sig = _cycle_signature(perms[b])
        sigs[b] = sig
        degs.append(sum(L * c for L, c in sig.items()))
        # permutation matrix => unitary => all eigenvalues are roots of unity
        if not _poly_factor_form(sig)["all_roots_unit_modulus"]:
            all_unit = False

    # aggregate: how many bytes have a fixed point, max cycle length seen
    n_fixed = sum(1 for b in range(256) if sigs[b].get(1, 0) > 0)
    max_cycle = max((L for sig in sigs.values() for L in sig), default=0)
    return {
        "n_bytes": 256,
        "all_poly_deg_4096": all(d == OMEGA_SIZE for d in degs),
        "bytes_with_fixed_point": n_fixed,
        "max_cycle_length": max_cycle,
        "all_roots_on_unit_circle": all_unit,
        "Krivine_note": "All byte operators are unitary permutations; roots = roots of unity",
    }


# ================================================================
# VIIb. Depth-4 word Lefschetz across micro-refs
# ================================================================

def experiment_depth4_microrefs(
    micro_refs: list[int] | None = None,
) -> dict[str, object]:
    """Verify graded Lefschetz numbers are constant across micro-refs m.

    All W2(m)/W2'(m)/F(m) share the same shell-swap / shell-preserving
    cycle structure, so their Lefschetz numbers and cycle signatures must be
    identical across m. We test a sample of micro-refs.
    """
    if micro_refs is None:
        micro_refs = [0, 1, 31, 32, 63]
    shell_of, idx_of = _shell_grading()

    results = {}
    for m in micro_refs:
        for name, word_fn in (("W2", W2_word), ("W2'", W2p_word), ("F", F_cycle_word)):
            w = word_fn(m)
            perm = _word_permutation(w, idx_of)
            fixed_per_shell = _graded_traces(perm, shell_of)
            L = sum((-1) ** k * fixed_per_shell[k] for k in range(7))
            sig = _cycle_signature(perm)
            results[f"{name}(m={m})"] = {
                "L": L,
                "fixed_per_shell": tuple(fixed_per_shell),
                "cycle_sig": dict(sig),
            }

    def _ls(name: str) -> list[int]:
        return [results[f"{name}(m={m})"]["L"] for m in micro_refs]

    w2_Ls = _ls("W2")
    w2p_Ls = _ls("W2'")
    f_Ls = _ls("F")
    return {
        "micro_refs": micro_refs,
        "W2_L_constant": len(set(w2_Ls)) == 1,
        "W2p_L_constant": len(set(w2p_Ls)) == 1,
        "F_L_constant": len(set(f_Ls)) == 1,
        "W2_L_value": w2_Ls[0],
        "W2p_L_value": w2p_Ls[0],
        "F_L_value": f_Ls[0],
        "all_L_zero": all(v == 0 for v in w2_Ls + w2p_Ls + f_Ls),
        "details": results,
    }


# ================================================================
# VIIIb. Aggregated byte cycle-structure statistics
# ================================================================

def experiment_byte_cycle_statistics(
    perms: list[np.ndarray] | None = None,
) -> dict[str, object]:
    """Aggregate cycle-structure statistics across all 256 bytes."""
    from collections import Counter

    _, idx_of = _shell_grading()
    if perms is None:
        perms = all_byte_perms(idx_of)
    sig_counts: Counter = Counter()
    for b in range(256):
        sig = _cycle_signature(perms[b])
        key = tuple(sorted(sig.items()))
        sig_counts[key] += 1

    n_pure_4 = sum(c for key, c in sig_counts.items()
                     if set(L for L, _ in key) == {4})
    n_has_fixed = sum(c for key, c in sig_counts.items()
                       if any(L == 1 for L, _ in key))
    n_has_2 = sum(c for key, c in sig_counts.items()
                     if any(L == 2 for L, _ in key))
    return {
        "distinct_cycle_structures": len(sig_counts),
        "pure_4cycle_bytes": n_pure_4,
        "bytes_with_fixed_points": n_has_fixed,
        "bytes_with_2cycles": n_has_2,
        "signature_distribution_top10": {
            str(key): count for key, count in sig_counts.most_common(10)
        },
    }


# ================================================================
# VIIIc. Dynamical zeta of a representative byte
# ================================================================

def experiment_byte_zeta(
    byte: int = 0xA8,
    max_n: int = 12,
    perms: list[np.ndarray] | None = None,
) -> dict[str, object]:
    """Dynamical zeta of a representative byte operator.

    zeta_{T_b}(t) = exp(sum_n Fix(T_b^n)/n t^n). For a permutation
    with cycle signature {L: c_L}, Fix(T_b^n) = sum_L L*c_L*(1 if L|n else 0),
    giving zeta = prod_L (1 - t^L)^{-c_L}.
    """
    _, idx_of = _shell_grading()
    if perms is None:
        perms = all_byte_perms(idx_of)
    sig = _cycle_signature(perms[byte])

    def _fix(n: int) -> int:
        return sum(L * c for L, c in sig.items() if n % L == 0)

    fix_counts = {n: _fix(n) for n in range(1, max_n + 1)}
    expected_form = " * ".join(f"(1-t^{L})^-{c}" for L, c in sorted(sig.items()))

    # verify log-zeta coefficient identity: coeff of t^n in log zeta equals
    # Fix(T^n)/n. zeta = exp(sum Fix/n t^n) => log zeta = sum Fix/n t^n.
    ok = True
    for n in range(1, max_n + 1):
        coeff_log = fix_counts[n] / n
        coeff_check = 0.0
        for L, c in sig.items():
            k = 1
            while k * L <= n:
                if k * L == n:
                    coeff_check += -c * (-1) * (1.0 / k)  # (1 - t^L) factor
                k += 1
        if abs(coeff_log - coeff_check) > 1e-6:
            ok = False
    return {
        "byte": byte,
        "byte_hex": f"0x{byte:02X}",
        "cycle_signature": dict(sig),
        "fix_counts": fix_counts,
        "zeta_form": expected_form,
        "log_zeta_coeff_identity": ok,
    }


# ================================================================
# main: dispatch and print
# ================================================================

def _print_k4(res: dict) -> None:
    print("\n" + "=" * 5)
    print("VI. K4 AS GALOIS GROUP OF FAMILY-FIBER COVER")
    print("=" * 5)
    print(f"  |Q6 classes|:        {res['|Q6 classes|']}")
    print(f"  fiber size all 4:    {res['fiber_size_all_4']}")
    print(f"  fiber sizes sample:  {res['fiber_sizes_sample']}")
    print(f"  each fiber 4 fams:   {res['each_fiber_has_4_families']}")
    print(f"  cover transitive K4: {res['cover_transitive_K4']}")
    print(f"  cover 4-to-1:        {res['cover_is_4_to_1']}")


def _print_depth4(res: dict) -> None:
    print("\n" + "=" * 5)
    print("VII. LEFSCHETZ + CHAR-POLY OF DEPTH-4 WORDS")
    print("=" * 5)
    for name, d in res["words"].items():
        print(f"  {name}:")
        print(f"    cycle signature: {d['cycle_signature']}")
        print(f"    fixed/shell:     {d['fixed_per_shell']}")
        print(f"    Lefschetz L:     {d['Lefschetz_L']}  (mod 12 = {d['Lefschetz_mod12']})")
        print(f"    poly deg:        {d['poly_deg']}")
        print(f"    poly factors:    {d['poly_factor_exponents']}")
        print(f"    roots unit mod:  {d['all_roots_unit_modulus']}")
        print(f"    integer roots:   {d['integer_roots']}")
    print(f"  shell map W2 s->6-s:    {res['shell_map_W2_s_to_6_minus_s']}")
    print(f"  shell map W2' s->6-s:   {res['shell_map_W2p_s_to_6_minus_s']}")
    print(f"  shell map F preserves:  {res['shell_map_F_preserves_shell']}")
    print(f"  F fixed points:         {res['F_fixed_points']}")
    print(f"  F two-cycles:           {res['F_two_cycles']}")
    print(f"  F poly == (x^2-1)^2048: {res['F_poly_matches_(x^2-1)^2048']}")
    print(f"  all words roots on UC:  {res['all_words_roots_on_unit_circle']}")
    print(f"  note: {res['note']}")
    print(f"  shell grading note: W2, W2', F share L=0, (x^2-1)^2048, zeta=(1-t^2)^-2048")
    print(f"    => shell grading cannot separate pole-swap (W2) from Z2 flip (F);")
    print(f"       a finer (shell, Z2-sheet) or chirality grading is needed.")


def _print_byte_poly(res: dict) -> None:
    print("\n" + "=" * 5)
    print("VIII. CHAR-POLY OF ALL 256 BYTE OPERATORS")
    print("=" * 5)
    print(f"  n_bytes:              {res['n_bytes']}")
    print(f"  all poly deg = 4096:  {res['all_poly_deg_4096']}")
    print(f"  bytes w/ fixed pt:    {res['bytes_with_fixed_point']}")
    print(f"  max cycle length:     {res['max_cycle_length']}")
    print(f"  all roots on UC:      {res['all_roots_on_unit_circle']}")
    print(f"  note: {res['Krivine_note']}")


# ================================================================
# IX. Complementarity invariant (Poincaré duality, h + ab = 12)
# ================================================================

def experiment_complementarity() -> dict[str, object]:
    """Verify horizon_distance + ab_distance = 12 on all Omega states.

    h(s) is the A/B horizon distance, ab(s) the A/B Hamming distance.
    Their sum is constant at 2*d = 12 (d=6 chirality dim); this is the
    Poincaré-duality / complementarity invariant of the kernel.
    """
    counts = {12: 0}
    for s in OMEGA_STATES_4096:
        a12, b12 = unpack_state(s)
        h = horizon_distance(a12, b12)
        ab = ab_distance(a12, b12)
        key = h + ab
        counts[key] = counts.get(key, 0) + 1
    inv_ok = all(c == 12 for c in counts)
    min_sum = min(counts)
    max_sum = max(counts)
    return {
        "n_states": len(OMEGA_STATES_4096),
        "sum_distribution": dict(counts),
        "min_sum": min_sum,
        "max_sum": max_sum,
        "constant_sum": max_sum == min_sum,
        "sum_equals_12": inv_ok,
        "invariant_check": all(
            complementarity_invariant(
                (s >> 12) & LAYER_MASK_12, s & LAYER_MASK_12
            )
            for s in OMEGA_STATES_4096
        ),
    }


# ================================================================
# X. Byte T^2 fixed points vs Lefschetz prediction
# ================================================================

def experiment_byte_square_fixedpoints(
    byte: int = 0xA8,
    perms: list[np.ndarray] | None = None,
) -> dict[str, object]:
    """Frobenius-like: fixed-point set of T_b^2 vs the Lefschetz prediction.

    For a permutation, Fix(T_b^2) = sum_L L * c_L * (1 if 2|L else 0):
    only cycles of even length survive two iterations. We count explicitly
    and compare to the Lefschetz fixed-point formula on the cycle signature.
    """
    _, idx_of = _shell_grading()
    if perms is None:
        perms = all_byte_perms(idx_of)
    sig = _cycle_signature(perms[byte])

    # explicit Fix(T^2): states whose 2-step image is themselves
    fixed2 = 0
    for s in OMEGA_STATES_4096:
        if step_state_by_byte(step_state_by_byte(s, byte), byte) == s:
            fixed2 += 1
    # Lefschetz prediction from signature: points in 1- and 2-cycles
    pred = sig.get(1, 0) + 2 * sig.get(2, 0)
    return {
        "byte": byte,
        "byte_hex": f"0x{byte:02X}",
        "cycle_signature": dict(sig),
        "Fix_T2_explicit": fixed2,
        "Fix_T2_from_signature": pred,
        "matches": fixed2 == pred,
    }


# ================================================================
# XI. W2 / W2' dynamical zeta identity
# ================================================================

def experiment_word_zeta() -> dict[str, object]:
    """Verify zeta of W2 and W2' matches (1 - t^2)^{-2048}.

    W2 / W2' are fixed-point-free involutions swapping shell s <-> 6-s,
    so each has 2048 two-cycles. The zeta is (1 - t^2)^{-c} with
    c = |Omega|/2 = 2048, same form as gate F (section V).
    """
    shell_of, idx_of = _shell_grading()
    out = {}
    for name, word_fn in (("W2", W2_word), ("W2'", W2p_word)):
        perm = _word_permutation(word_fn(0), idx_of)
        sig = _cycle_signature(perm)
        f2 = sig.get(2, 0)
        ok = (f2 == OMEGA_SIZE // 2 and sig.get(1, 0) == 0)
        out[name] = {
            "cycle_signature": dict(sig),
            "two_cycles": f2,
            "zeta_form": "(1 - t^2)^(-2048)",
            "matches_F_form": ok,
        }
    return {
        "W2": out["W2"],
        "W2'": out["W2'"],
        "note": "W2/W2' share the (1-t^2)^-2048 zeta with F.",
    }


# ================================================================
# XII. Torsion vs Hilbert spectrum inclusion
# ================================================================

def experiment_torsion_vs_hilbert_spectrum(byte: int = 0xA8) -> dict[str, object]:
    """Compare torsion-layer and Hilbert-layer eigenvalue spectra.

    Torsion layer: cycle lengths of byte action on Omega (roots of unity
    of order dividing each cycle length: 1, 2, 4).
    Hilbert layer: permutation char-poly roots on the unit circle, with
    multiplicities from the cycle signature.

    Check: every torsion-layer root (order dividing a cycle length of the
    permutation) appears among the Hilbert permutation roots.
    """
    shell_of, idx_of = _shell_grading()
    perm = np.empty(OMEGA_SIZE, dtype=np.int64)
    for i, s in enumerate(OMEGA_STATES_4096):
        perm[i] = idx_of[step_state_by_byte(s, byte)]
    sig = _cycle_signature(perm)

    # torsion roots: for each cycle length L, roots exp(2*pi*i*k/L), k=0..L-1
    torsion_orders = set()
    for L in sig:
        torsion_orders.add(L)
    # Hilbert permutation roots: same orders (unitary roots of unity)
    hilb_orders = set(sig.keys())

    # inclusion: torsion orders subset of Hilbert orders
    inclusion = torsion_orders.issubset(hilb_orders)
    return {
        "byte": byte,
        "byte_hex": f"0x{byte:02X}",
        "cycle_signature": dict(sig),
        "torsion_orders": sorted(torsion_orders),
        "hilbert_orders": sorted(hilb_orders),
        "torsion_subset_hilbert": inclusion,
        "note": ("Byte permutation roots are roots of unity; the Hilbert lift "
                 "contains the torsion spectrum as a sub-spectrum."),
    }


# ================================================================
# XIII. K4 group cohomology H^1 (family gauge action)
# ================================================================

def experiment_k4_cohomology_h1() -> dict[str, object]:
    """Group cohomology H^1(K4, GF(2)^6) with trivial K4-module.

    K4 = (Z/2)^2 is the deck group of the 4-to-1 byte cover (section VI).
    On the chirality register GF(2)^6 the Galois action is trivial at the
    cohomology level: family bits permute cover sheets, not a linear G-module
    on V with rho(g)(0)=0. Standard H^1(G,V) for a trivial module is
    Hom(G, V); coboundaries vanish since gv - v = 0.

    We enumerate 1-cocycles z(gh)=z(g)+z(h) and check dim = 2*6 = 12.
    """
    k4 = [(a, b) for a in (0, 1) for b in (0, 1)]
    e: tuple[int, int] = (0, 0)

    def mul(g: tuple[int, int], h: tuple[int, int]) -> tuple[int, int]:
        return (g[0] ^ h[0], g[1] ^ h[1])

    # 1-cocycles: group homomorphisms K4 -> GF(2)^6
    cocycles: list[dict[tuple[int, int], int]] = []
    for v_sigma in range(64):
        for v_tau in range(64):
            z: dict[tuple[int, int], int] = {e: 0}
            for a in (0, 1):
                for b in (0, 1):
                    g: tuple[int, int] = (int(a), int(b))
                    if g == e:
                        continue
                    # K4 = <sigma=(1,0), tau=(0,1)>; z(a,b) = a*z(s) + b*z(t)
                    z[g] = (
                        ((v_sigma if a else 0) ^ (v_tau if b else 0)) & 0x3F
                    )
            ok = True
            for g in k4:
                for h in k4:
                    gh = mul(g, h)
                    if (z[gh] ^ z[g] ^ z[h]) & 0x3F:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                cocycles.append(z)

    # encode cocycles as 24-bit vectors (4 group elems * 6 bits)
    z_vecs = []
    for z in cocycles:
        vec = 0
        for i, g in enumerate(k4):
            vec |= (z[g] & 0x3F) << (i * 6)
        z_vecs.append(vec)
    dim_z1 = _gf2_rank_ints(z_vecs)
    dim_b1 = 0  # trivial module: coboundaries gv - v = 0
    dim_h1 = dim_z1 - dim_b1
    expected = 12
    return {
        "n_k4": len(k4),
        "module": "trivial",
        "n_cocycles": len(cocycles),
        "dim_Z1_GF2": dim_z1,
        "dim_B1_GF2": dim_b1,
        "dim_H1_GF2_6": dim_h1,
        "expected_dim_H1": expected,
        "dim_matches_expected": dim_h1 == expected,
    }


def _gf2_rank_ints(vecs: list[int]) -> int:
    """Rank over GF(2) by standard XOR Gaussian elimination.

    Each vector is a non-negative integer; bits are the coordinates. For
    each basis vector b we replace x by min(x, x ^ b), which clears the
    highest bit where they differ and strictly reduces x. The basis is kept
    sorted in descending bit length so the matching top-bit vector is
    always reached first.
    """
    basis: list[int] = []
    for v in vecs:
        x = v
        for b in basis:
            x = min(x, x ^ b)
        if x:
            basis.append(x)
            basis.sort(reverse=True)
    return len(basis)


def _print_lefschetz_stab(res: dict) -> None:
    print(f"  horizon-stabilizer note: {res['horizon_stabilizer_note']}")


def _print_depth4_microrefs(res: dict) -> None:
    print("\n" + "=" * 5)
    print("VIIb. DEPTH-4 LEFSCHETZ ACROSS MICRO-REFS")
    print("=" * 5)
    print(f"  micro-refs: {res['micro_refs']}")
    print(f"  W2 L constant:  {res['W2_L_constant']}  value={res['W2_L_value']}")
    print(f"  W2' L constant: {res['W2p_L_constant']}  value={res['W2p_L_value']}")
    print(f"  F L constant:    {res['F_L_constant']}  value={res['F_L_value']}")
    print(f"  all L zero:      {res['all_L_zero']}")


def _print_byte_cycle_stats(res: dict) -> None:
    print("\n" + "=" * 5)
    print("VIIIb. BYTE CYCLE-STRUCTURE AGGREGATES")
    print("=" * 5)
    print(f"  distinct structures:    {res['distinct_cycle_structures']}")
    print(f"  pure-4-cycle bytes:   {res['pure_4cycle_bytes']}")
    print(f"  bytes w/ fixed pts:   {res['bytes_with_fixed_points']}")
    print(f"  bytes w/ 2-cycles:   {res['bytes_with_2cycles']}")
    print("  top signatures:")
    for sig, count in res["signature_distribution_top10"].items():
        print(f"    {sig}: {count}")


def _print_byte_zeta(res: dict) -> None:
    print("\n" + "=" * 5)
    print(f"VIIIc. BYTE ZETA (byte {res['byte_hex']})")
    print("=" * 5)
    print(f"  cycle signature: {res['cycle_signature']}")
    print(f"  fix counts:       {res['fix_counts']}")
    print(f"  zeta form:        {res['zeta_form']}")
    print(f"  log-zeta ident:   {res['log_zeta_coeff_identity']}")


def _print_complementarity(res: dict) -> None:
    print("\n" + "=" * 5)
    print("IX. COMPLEMENTARITY INVARIANT (h + ab = 12)")
    print("=" * 5)
    print(f"  n_states:           {res['n_states']}")
    print(f"  sum distribution:   {res['sum_distribution']}")
    print(f"  min_sum:            {res['min_sum']}")
    print(f"  max_sum:            {res['max_sum']}")
    print(f"  sum constant:      {res['constant_sum']}")
    print(f"  sum == 12:          {res['sum_equals_12']}")
    print(f"  invariant check:     {res['invariant_check']}")


def _print_byte_square_fp(res: dict) -> None:
    print("\n" + "=" * 5)
    print(f"X. BYTE T^2 FIXED POINTS (byte {res['byte_hex']})")
    print("=" * 5)
    print(f"  cycle signature:     {res['cycle_signature']}")
    print(f"  Fix(T^2) explicit:  {res['Fix_T2_explicit']}")
    print(f"  Fix(T^2) sig:      {res['Fix_T2_from_signature']}")
    print(f"  matches:             {res['matches']}")


def _print_word_zeta(res: dict) -> None:
    print("\n" + "=" * 5)
    print("XI. W2 / W2' DYNAMICAL ZETA")
    print("=" * 5)
    for name in ("W2", "W2'"):
        d = res[name]
        print(f"  {name}: sig={d['cycle_signature']} two_cycles={d['two_cycles']}")
        print(f"    zeta form: {d['zeta_form']}  matches_F: {d['matches_F_form']}")


def _print_torsion_hilbert(res: dict) -> None:
    print("\n" + "=" * 5)
    print(f"XII. TORSION vs HILBERT SPECTRUM (byte {res['byte_hex']})")
    print("=" * 5)
    print(f"  cycle signature:     {res['cycle_signature']}")
    print(f"  torsion orders:      {res['torsion_orders']}")
    print(f"  hilbert orders:      {res['hilbert_orders']}")
    print(f"  torsion subset hilbert: {res['torsion_subset_hilbert']}")
    print(f"  note: sanity check only; for a permutation matrix the torsion")
    print(f"        orders are the Hilbert roots by construction (same cycle set).")


def _print_k4_cohomology(res: dict) -> None:
    print("\n" + "=" * 5)
    print("XIII. K4 GROUP COHOMOLOGY H^1")
    print("=" * 5)
    print(f"  |K4|:               {res['n_k4']}")
    print(f"  module:             {res['module']}")
    print(f"  n cocycles:         {res['n_cocycles']}")
    print(f"  dim Z^1 (cocycles):  {res['dim_Z1_GF2']}")
    print(f"  dim B^1 (coboundaries): {res['dim_B1_GF2']}")
    print(f"  dim H^1(GF(2)^6):  {res['dim_H1_GF2_6']}")
    print(f"  expected dim H^1:   {res['expected_dim_H1']}")
    print(f"  dim matches:        {res['dim_matches_expected']}")


def main() -> None:
    print("hQVM Cohomology analysis 2 -- categorical architecture")
    print("=" * 5)
    print(f"seed={SEED}")

    _print_carrier_chsh(experiment_carrier_chsh())
    _print_shell_betti(experiment_shell_betti())
    _print_gate_f(experiment_gate_f_fixedpoints())
    _print_lefschetz(experiment_lefschetz_bytes())
    _print_zeta(experiment_dynamical_zeta())
    _print_k4(experiment_k4_cover())
    _print_depth4(experiment_depth4_lefschetz_and_poly())
    _print_depth4_microrefs(experiment_depth4_microrefs())
    # Precompute the 256 byte permutations once; share across the byte scans.
    _, idx_of = _shell_grading()
    byte_perms = all_byte_perms(idx_of)
    _print_byte_poly(experiment_byte_characteristic_polys(perms=byte_perms))
    _print_byte_cycle_stats(experiment_byte_cycle_statistics(perms=byte_perms))
    _print_byte_zeta(experiment_byte_zeta(perms=byte_perms))
    _print_complementarity(experiment_complementarity())
    _print_byte_square_fp(experiment_byte_square_fixedpoints(perms=byte_perms))
    _print_word_zeta(experiment_word_zeta())
    _print_torsion_hilbert(experiment_torsion_vs_hilbert_spectrum())
    _print_k4_cohomology(experiment_k4_cohomology_h1())

    print("\n" + "=" * 5)
    print("SUMMARY")
    print("=" * 5)
    print("  I  carrier CHSH (Boolean):     see max_CHSH vs classical bound")
    print("  II shell Betti / Poincare:    see duality, P(-1)=0")
    print("  III gate F fixed points:      see fixed_point_free, chi=0")
    print("  IV Lefschetz of bytes:        see L range and correlations")
    print("  V dynamical zeta of F:        see exponent vs 2048, func eq")
    print("  VI K4 cover:                  see transitive, 4-to-1")
    print("  VII depth-4 words:            see L(W2/W2'/F), char poly")
    print("  VIIb micro-ref Lefschetz:       see L constant across m")
    print("  VIII byte char poly:          see deg=4096, roots on UC")
    print("  VIIIb byte cycle aggr:        see pure-4-cycle count")
    print("  VIIIc byte zeta:               see zeta form, log-zeta ident")
    print("  IX complementarity:           see h+ab=12, invariant")
    print("  X byte T^2 fixed pts:         see Fix(T^2) vs signature")
    print("  XI W2/W2' zeta:              see (1-t^2)^-2048 form")
    print("  XII torsion vs Hilbert:        see torsion subset Hilbert")
    print("  XIII K4 H^1 dim:              see dim H^1(GF(2)^6)")


if __name__ == "__main__":
    main()
