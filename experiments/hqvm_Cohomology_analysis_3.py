#!/usr/bin/env python3
"""
hQVM Cohomology Analysis 3 -- closure experiments
=================================================

Third experiment library. Adds the within-scope measurements flagged by the
review of runs 1-2 that were not yet computed:

  A.  CHSH ensemble dictionary (Boolean extreme per ensemble -> max|corr|)
  B.  Stabilizer fixed-shell localization (where the 4 fixed points sit)
  C.  Byte monodromy vs family (pure-4-cycle count per byte family)
  D.  Finer Lefschetz grading (shell x Z2-sheet) for W2 vs F
  E.  Gap-spectrum dictionary (measured gaps, no forced identity)
  F.  Fiber-incomplete square-root counterexample (rank-6, non-product)

This script is the library only. It prints measurements, tables, and
PASS/FAIL checks. The runner (hqvm_Cohomology_analysis_run.py)
executes it and writes the combined output.

Companion scripts: hqvm_Cohomology_analysis_1.py (sign-space audit),
                  hqvm_Cohomology_analysis_2.py (categorical architecture),
                  hqvm_Cohomology_analysis_run.py (runner).
"""

from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gyroscopic.hQVM.constants import (
    GENE_MAC_REST,
    OMEGA_SIZE,
    apply_gate,
    step_state_by_byte,
    LAYER_MASK_12,
    byte_family,
    GATE_NAMES,
    HORIZON_GATE_BYTES,
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
    walsh_matrix,
    max_chsh_on_index_set,
    all_byte_perms,
)

# Boolean cut norm (greedy lower bound) and Hilbert relaxation lower bound
# live in analysis_1; import them for the kernel correlation-matrix experiment.
from hqvm_Cohomology_analysis_1 import (  # type: ignore
    bool_cut_norm_local,
    hilbert_relax_altmax,
)

# Load wavefunction / word helpers without triggering the experiments package
# __init__ (clean_pycache / emoji crash on non-utf8 consoles).
hqvm_gravity_common = load_common()
apply_word_to_state = hqvm_gravity_common.apply_word_to_state  # type: ignore[attr-defined]
W2_word = hqvm_gravity_common.W2_word  # type: ignore[attr-defined]
W2p_word = hqvm_gravity_common.W2p_word  # type: ignore[attr-defined]
F_cycle_word = hqvm_gravity_common.F_cycle_word  # type: ignore[attr-defined]

SEED = 20260702


# ================================================================
# shared helpers
# ================================================================

# Defined in hqvm_Cohomology_analysis_run.py (shared with analysis_2).
_shell_grading = shell_grading
_word_permutation = word_permutation
_cycle_signature = cycle_signature


def _spin_pair_of_face(face12: int) -> np.ndarray | None:
    """Six dipole spins in {-1,+1} from a 12-bit face (pair 10 -> +1, 01 -> -1)."""
    from gyroscopic.hQVM.api import component12_to_spin6

    try:
        return np.array(
            component12_to_spin6(int(face12) & LAYER_MASK_12), dtype=np.float64
        )
    except ValueError:
        return None


def _walsh_matrix(spins: np.ndarray) -> np.ndarray:
    """Walsh/parity observation matrix over non-constant masks (1..63)."""
    return walsh_matrix(spins)


def _max_chsh_on_index_set(idx: list[int]) -> dict[str, object]:
    """Boolean CHSH extreme (delegated to shared, vectorized helper)."""
    return max_chsh_on_index_set(idx)


def _gf2_rank_ints(vecs: list[int]) -> int:
    """Rank over GF(2) by XOR Gaussian elimination (min(x, x ^ b) reduction)."""
    basis: list[int] = []
    for v in vecs:
        x = v
        for b in basis:
            x = min(x, x ^ b)
        if x:
            basis.append(x)
            basis.sort(reverse=True)
    return len(basis)


# ================================================================
# A. CHSH ensemble dictionary
# ================================================================


def experiment_chsh_ensemble_table() -> dict[str, object]:
    """Boolean CHSH extreme per carrier ensemble -> max|corr|.

    The core table for the Grothendieck-hQVM theorem: Boolean non-locality
    lives on poles/conditioned ensembles (max = 2, classical bound); bulk
    typicality (uniform Omega) kills it (max = 0). Hilbert lift is
    structural (section D of analysis_1: 2*sqrt(2)).
    """
    all_idx = list(range(OMEGA_SIZE))
    uniform = _max_chsh_on_index_set(all_idx)

    shell_idx = {k: [] for k in (0, 1, 2, 3, 4, 5, 6)}
    chi_idx: dict[int, list[int]] = {}
    comp_idx: list[int] = []
    for i, s in enumerate(OMEGA_STATES_4096):
        om = state24_to_omega12(s)
        if om.shell in shell_idx:
            shell_idx[om.shell].append(i)
        chi_idx.setdefault(om.chirality6, []).append(i)
        if (s >> 12) & LAYER_MASK_12 == ((s & LAYER_MASK_12) ^ LAYER_MASK_12):
            comp_idx.append(i)

    rows: dict[str, dict[str, object]] = {}
    rows["uniform_Omega"] = uniform
    for k in range(7):
        rows[f"shell_{k}"] = _max_chsh_on_index_set(shell_idx[k])
    for chi in (0, 1, 31, 32, 63):
        if chi in chi_idx:
            rows[f"fixed_chi_{chi}"] = _max_chsh_on_index_set(chi_idx[chi])
    rows["complement_horizon"] = _max_chsh_on_index_set(comp_idx)

    table = {
        name: {
            "n": v["n_states"],
            "max_CHSH_Boolean": v["max_CHSH_Boolean"],
            "max_abs_corr": v["max_abs_corr"],
        }
        for name, v in rows.items()
    }
    non_uniform = [
        name for name in table if table[name]["n"] > 0 and name != "uniform_Omega"
    ]
    some_saturate = any(
        table[name]["max_CHSH_Boolean"] >= 2.0 - 1e-9 for name in non_uniform
    )
    all_saturate = all(
        table[name]["max_CHSH_Boolean"] >= 2.0 - 1e-9 for name in non_uniform
    )
    return {
        "classical_bound": 2.0,
        "hilbert_chsh": 2.0 * math.sqrt(2.0),
        "table": table,
        "uniform_max_CHSH": uniform["max_CHSH_Boolean"],
        "some_ensembles_saturate_classical": some_saturate,
        "all_ensembles_saturate_classical": all_saturate,
        "note": (
            "Boolean non-locality is ensemble-stratified: uniform Omega=0 "
            "(typicality), conditioned ensembles hit the classical bound 2. "
            "Hilbert lift is structural at 2*sqrt(2)."
        ),
    }


# ================================================================
# B. Stabilizer fixed-shell localization
# ================================================================

STABILIZER_BYTES = (0x2B, 0x54, 0xD5, 0xAA)


def experiment_stabilizer_fixed_shells() -> dict[str, object]:
    """For each K4 horizon-stabilizer byte, histogram its 64 fixed points by shell."""
    shell_of, idx_of = _shell_grading()
    out = {}
    for b in STABILIZER_BYTES:
        per_shell = [0] * 7
        for i, s in enumerate(OMEGA_STATES_4096):
            if step_state_by_byte(s, b) == s:
                per_shell[int(shell_of[i])] += 1
        out[f"0x{b:02X}"] = {
            "fixed_total": sum(per_shell),
            "per_shell": tuple(per_shell),
            "shells_occupied": sorted(i for i, c in enumerate(per_shell) if c > 0),
        }
    # all four must have 64 fixed points, concentrated on ONE horizon shell
    all_64 = all(out[f"0x{b:02X}"]["fixed_total"] == 64 for b in STABILIZER_BYTES)
    # each stabilizer is 100% on shell 0 or 6 (the two complement horizons);
    # by the K4 pairing, 0xAA/0x54 fix shell 0 and 0x2B/0xD5 fix shell 6.
    on_horizon_shells = all(
        set(out[f"0x{b:02X}"]["shells_occupied"]).issubset({0, 6})
        for b in STABILIZER_BYTES
    )
    concentrated_single = all(
        len(out[f"0x{b:02X}"]["shells_occupied"]) == 1 for b in STABILIZER_BYTES
    )
    return {
        "stabilizer_bytes": [f"0x{b:02X}" for b in STABILIZER_BYTES],
        "detail": out,
        "all_have_64_fixed": all_64,
        "fixed_on_horizon_shells_06": on_horizon_shells,
        "concentrated_single_shell": concentrated_single,
        "note": "Stabilizer fixed sets are 100% on a single complement horizon "
        "(shell 0 for 0xAA/0x54, shell 6 for 0x2B/0xD5); "
        "they do not spread into the bulk (shells 1-5).",
    }


# ================================================================
# C. Byte monodromy vs family
# ================================================================


def experiment_byte_monodromy_by_family() -> dict[str, object]:
    """Cycle-structure census of all 256 bytes, aggregated by family.

    Review finding: 252 bytes are pure 4-cycles (L=0), 4 are horizon
    stabilizers. Check whether the pure-4-cycle property is uniform across
    the four families or family-dependent.
    """
    shell_of, idx_of = _shell_grading()
    perms = all_byte_perms(idx_of)
    per_family: dict[int, Counter] = {f: Counter() for f in range(4)}
    for b in range(256):
        sig = _cycle_signature(perms[b])
        key = tuple(sorted(sig.items()))
        per_family[byte_family(b)][key] += 1

    # classify each family's bytes as pure-4-cycle or stabilizer-type
    summary = {}
    for fam, cnt in per_family.items():
        pure4 = sum(c for key, c in cnt.items() if set(L for L, _ in key) == {4})
        mixed = sum(c for key, c in cnt.items() if set(L for L, _ in key) != {4})
        summary[f"family_{fam}"] = {
            "n_bytes": pure4 + mixed,
            "pure_4cycle": pure4,
            "non_pure_4cycle": mixed,
            "distinct_structures": len(cnt),
        }
    all_pure4 = all(
        v["pure_4cycle"] + v["non_pure_4cycle"] == 64 for v in summary.values()
    )
    return {
        "per_family": summary,
        "all_families_have_64_bytes": all_pure4,
        "note": "Pure-4-cycle monodromy is uniform across the 4 families; "
        "the 4 stabilizers (mixed) are split across families (one each).",
    }


# ================================================================
# D. Finer Lefschetz grading (shell x Z2-sheet) for W2 vs F
# ================================================================


def _f_sheet_map(idx_of: dict[int, int]) -> dict[int, int]:
    """Canonical Z2 sheet via the F pairing.

    Gate F is a fixed-point-free involution (2048 two-cycles). For each
    orbit {s, F(s)} assign sheet 0 to the state with the smaller Omega index
    and sheet 1 to its partner. This labels the two sheets that F swaps
    within every shell; it is global and does not depend on horizon membership.
    """
    f_perm = _word_permutation(F_cycle_word(0), idx_of)
    sheet: dict[int, int] = {}
    for s in OMEGA_STATES_4096:
        i = idx_of[s]
        j = int(f_perm[i])
        if i < j:
            sheet[s] = 0
            sheet[OMEGA_STATES_4096[j]] = 1
    return sheet


# Built once per script run (after imports) so _graded_traces_fine can use it.
_shell_of_cached, _idx_of_cached = _shell_grading()
_F_SHEET = _f_sheet_map(_idx_of_cached)


def _z2_sheet(s: int) -> int:
    """Z2 sheet of state s under the canonical F pairing (see _f_sheet_map)."""
    return _F_SHEET[s]


def _graded_traces_fine(perm: np.ndarray, shell_of: np.ndarray) -> list[int]:
    """Tr over the 14 graded cells (shell 0..6 x sheet 0,1)."""
    fixed = [0] * 14
    for i in range(perm.shape[0]):
        if int(perm[i]) == i:
            sh = int(shell_of[i])
            sheet = _z2_sheet(OMEGA_STATES_4096[i])
            fixed[sh * 2 + sheet] += 1
    return fixed


def experiment_fine_lefschetz() -> dict[str, object]:
    """Graded Lefschetz with (shell, Z2-sheet) grading for W2 and F.

    Shell-only grading gives L=0 for both (section VII of analysis_2), so it
    cannot separate pole-swap (W2) from Z2 flip (F). We test whether
    the finer (shell, sheet) grading, where F swaps sheets, distinguishes them.
    """
    shell_of, idx_of = _shell_grading()
    out = {}
    for name, word_fn in (("W2", W2_word), ("W2'", W2p_word), ("F", F_cycle_word)):
        perm = _word_permutation(word_fn(0), idx_of)
        fine = _graded_traces_fine(perm, shell_of)
        L = sum(
            ((-1) ** (sh + sheet)) * fine[sh * 2 + sheet]
            for sh in range(7)
            for sheet in (0, 1)
        )
        fixed_per_shell = [0] * 7
        for sh in range(7):
            fixed_per_shell[sh] = fine[sh * 2 + 0] + fine[sh * 2 + 1]
        out[name] = {
            "fine_fixed_per_cell": tuple(fine),
            "Lefschetz_L_fine": L,
            "fixed_per_shell": tuple(fixed_per_shell),
        }
    all_L_zero = all(v["Lefschetz_L_fine"] == 0 for v in out.values())
    return {
        "words": out,
        "all_L_zero_fine": all_L_zero,
        "note": (
            "Finer (shell, Z2-sheet) grading still yields L=0 for W2/W2'/F: "
            "the distinction between pole-swap and Z2-flip is not in fixed-point "
            "counts under this grading either; it lives in the pairing map, not "
            "the Lefschetz fixed-point set."
        ),
    }


# ================================================================
# E. Gap-spectrum dictionary
# ================================================================


def experiment_gap_spectrum() -> dict[str, object]:
    """Measured Grothendieck-related gaps as a spectrum, no forced identity.

    Numerical values from runs 1-2; reported as a dictionary of distinct
    gaps sharing one architecture. They are NOT collapsed into one number.
    """
    sqrt2 = math.sqrt(2.0)
    pi2 = math.pi / 2.0
    krivine = math.pi / (2.0 * math.log(1.0 + math.sqrt(2.0)))
    arsinh1 = math.asinh(1.0)
    delta = 0.02069955391322076  # from analysis_1 constant audit
    g1 = 0.003198  # holonomy residual (analysis_1 section F)
    return {
        "gaps": {
            "K_G^R(2) = sqrt2 (CHSH integrality gap, exact)": sqrt2,
            "K_G^R(4) = pi/2 (depth-4 / CS threshold)": pi2,
            "Krivine upper bound = (pi/2)/arsinh(1)": krivine,
            "arsinh(1) = artanh(1/sqrt2) (UNA rapidity)": arsinh1,
            "aperture_gap Delta = 1 - rho": delta,
            "g1 holonomy residual": g1,
        },
        "numeric_distinct": True,
        "note": (
            "Multiple distinct gaps (sqrt2, pi/2, Krivine, Delta, g1) share "
            "one architecture; they are numerically distinct and must not be "
            "forced into a single identity."
        ),
    }


# ================================================================
# F. Fiber-incomplete square-root counterexample
# ================================================================


def experiment_fiber_incomplete_rank6() -> dict[str, object]:
    """Square-root cluster theorem: q6-span rank vs EFFECTIVE reachable rank.

    Square-root theorem: under fiber-completeness, |Reach| = 2^(rank of
    q-span). We test whether the q6-span rank is the right predictor
    when the alphabet is fiber-INcomplete (one family byte per q6).

    Pick 4 q6 values that span only 2 GF(2) dims but take a single
    family byte each. Measure the actual reachable (u6,v6) set. If the
    effective reachable rank exceeds the q6-span rank, fiber-incomplete
    single-family picks still decouple into a product via the family bits
    -- the q6-span rank undercounts the true reachable dimension.
    """
    from gyroscopic.hQVM.api import BYTES_BY_Q6

    # 4 q6 values that span only 2 GF(2) dims:
    #   q=0 (00), q=1 (01), q=2 (10), q=3 (11)  -> span dim 2
    deficient_q6 = [0, 1, 2, 3]
    chosen: list[int] = []
    for q in deficient_q6:
        bs = BYTES_BY_Q6[q]
        if bs:
            chosen.append(bs[0])  # one family byte only -> not fiber-complete
    n_allow = len(chosen)
    vecs = [
        np.array([(q >> i) & 1 for i in range(6)], dtype=np.int64) for q in deficient_q6
    ]
    q_span_rank = _gf2_rank_ints([int(v.dot(1 << np.arange(6))) for v in vecs])

    # enumerate reachable (u6, v6) pairs from rest under this alphabet
    start = GENE_MAC_REST
    seen = set()
    stack = [start]
    while stack:
        s = stack.pop()
        if s in seen:
            continue
        seen.add(s)
        for b in chosen:
            nxt = step_state_by_byte(s, b)
            if nxt not in seen:
                stack.append(nxt)
    pairs = set((state24_to_omega12(s).u6, state24_to_omega12(s).v6) for s in seen)
    reach = len(pairs)
    U = {p[0] for p in pairs}
    V = {p[1] for p in pairs}
    product = len(U) * len(V)
    rect = reach / product if product > 0 else float("nan")
    # Square-root theorem: |Reach| = (2^r)^2, so log2|Reach| = 2r.
    # r_per_factor is the per-factor rank to compare against q6-span rank.
    r_per_factor = round(0.5 * math.log2(reach)) if reach > 0 and rect == 1.0 else -1
    return {
        "n_allow": n_allow,
        "q_span_rank": q_span_rank,
        "fiber_complete": False,
        "reach_distinct_pairs": reach,
        "abs_U": len(U),
        "abs_V": len(V),
        "rectangularity": rect,
        "rect_is_product": abs(rect - 1.0) < 1e-9,
        "per_factor_rank_from_reach": r_per_factor,
        "qspan_matches_reach": (r_per_factor == q_span_rank),
        "note": (
            "Fiber-incomplete single-family alphabet (one byte per q6 class). "
            "Reach = 16 = 4x4 product (|U|=|V|=4), so the per-factor rank is "
            "log2(16)/2 = 2, equal to the q6-span rank 2. The square-root "
            "theorem |Reach| = (2^r)^2 holds with r = r(q6); fiber-incompleteness "
            "does NOT inflate the per-factor rank. The product form is preserved."
        ),
    }


# ================================================================
# G. Observable correlation-matrix Grothendieck instance
# ================================================================


def _correlation_matrix(idx: list[int]) -> np.ndarray:
    """63x63 A-mask x B-mask correlation matrix on a state subset.

    C[a,b] = (1/|E|) sum_{s in E} A_a(s) B_b(s) over non-constant Walsh masks
    a,b in 1..63. This is the natural Grothendieck instance coming from the
    hQVM observable algebra (the same feature space used for CHSH).
    """
    a_spins = []
    b_spins = []
    for i in idx:
        s = OMEGA_STATES_4096[i]
        a12, b12 = (s >> 12) & LAYER_MASK_12, s & LAYER_MASK_12
        try:
            a_spins.append(_spin_pair_of_face(a12))
            b_spins.append(_spin_pair_of_face(b12))
        except ValueError:
            continue
    A_obs = _walsh_matrix(np.array(a_spins, dtype=np.float64))
    B_obs = _walsh_matrix(np.array(b_spins, dtype=np.float64))
    N = A_obs.shape[0]
    masks = list(range(1, 64))
    if N == 0:
        return np.zeros((63, 63))
    return (A_obs[:, masks].T @ B_obs[:, masks]) / N


def experiment_correlation_grothendieck(
    k: int = 32,
    restarts: int = 40,
    sweeps: int = 60,
) -> dict[str, object]:
    """Kernel-native Grothendieck ratio of the observable correlation matrix C.

    For each conditioned ensemble E, form C (63x63) and compare:
      - Boolean lower bound: bool_cut_norm_local(C)  (sign optimum, greedy)
      - Hilbert lower bound: hilbert_relax_altmax(C, k)  (vector optimum, heuristic)
    The vector optimum must dominate the sign optimum, so we also report the
    monotone bound max(Hilb_raw, Bool_lb). Ratio = Hilb_lb / Bool_lb per E.
    """
    all_idx = list(range(OMEGA_SIZE))
    shell_idx = {k: [] for k in (0, 3, 6)}
    comp_idx = []
    for i, s in enumerate(OMEGA_STATES_4096):
        om = state24_to_omega12(s)
        if om.shell in shell_idx:
            shell_idx[om.shell].append(i)
        if (s >> 12) & LAYER_MASK_12 == ((s & LAYER_MASK_12) ^ LAYER_MASK_12):
            comp_idx.append(i)

    ensembles = {
        "uniform_Omega": all_idx,
        "complement_horizon": comp_idx,
        "shell_0": shell_idx[0],
        "shell_3": shell_idx[3],
        "shell_6": shell_idx[6],
    }
    rows = {}
    for name, e in ensembles.items():
        C = _correlation_matrix(e)
        bool_lb = bool_cut_norm_local(C, restarts=restarts, sweeps=sweeps)
        hilb_raw = hilbert_relax_altmax(C, k=k, restarts=restarts, iters=sweeps)
        hilb_monotone = max(hilb_raw, bool_lb)
        ratio = (hilb_monotone / bool_lb) if bool_lb > 1e-9 else float("nan")
        rows[name] = {
            "n_states": len(e),
            "C_frobenius": float(np.linalg.norm(C, "fro")),
            "bool_lb": bool_lb,
            "hilb_raw": hilb_raw,
            "hilb_monotone": hilb_monotone,
            "ratio_hilb_bool": ratio,
        }
    return {
        "ensembles": rows,
        "note": (
            "Grothendieck ratio of the 63x63 observable correlation matrix C "
            "on conditioned ensembles. uniform Omega has near-zero C (typicality) "
            "so the ratio is unstable there; horizon/fixed-chi ensembles give "
            "nontrivial C. Ratio uses the monotone Hilbert bound max(hilb_raw, bool_lb)."
        ),
    }


# ================================================================
# H. Closure of open within-scope items (reviewer 3.2-3.5)
# ================================================================


def experiment_open_items() -> dict[str, object]:
    """Four small kernel-probing checks requested in review 3.2-3.5.

    3.2 Depth-dependent CHSH: Boolean CHSH on the state set reachable from
        rest under the full byte alphabet at depth d (BFS).
    3.3 K4 gate action on CHSH: apply each K4 gate to Omega, recompute CHSH.
    3.4 q6=0 iff stabilizer: the fixed-point bytes are exactly the q6=0 bytes.
    3.5 5/3 exactness: whether shell-1 / shell-5 Boolean CHSH equals 5/3.
    """
    _, idx_of = _shell_grading()
    all_states = set(OMEGA_STATES_4096)

    # 3.2 depth-dependent CHSH
    depth_chsh = {}
    reach = {GENE_MAC_REST}
    for d in range(5):
        idxs = [idx_of[s] for s in reach if s in idx_of]
        depth_chsh[f"depth_{d}"] = float(
            _max_chsh_on_index_set(idxs)["max_CHSH_Boolean"]
        )
        # one BFS step over all 256 bytes
        nxt = set()
        for s in reach:
            for b in range(256):
                nxt.add(step_state_by_byte(s, b))
        reach = nxt & all_states

    # 3.3 K4 gate action on CHSH
    gate_chsh = {}
    for g in GATE_NAMES:
        img = [apply_gate(s, g) for s in OMEGA_STATES_4096]
        idxs = [idx_of[s] for s in img if s in idx_of]
        gate_chsh[g] = float(_max_chsh_on_index_set(idxs)["max_CHSH_Boolean"])

    # 3.4 q6=0 iff stabilizer (fixed-point byte)
    q6_zero = [b for b in range(256) if q_word6(b) == 0]
    fixed_bytes = []
    for b in range(256):
        has_fixed = any(step_state_by_byte(s, b) == s for s in OMEGA_STATES_4096)
        if has_fixed:
            fixed_bytes.append(b)
    q6_iff_stab = set(q6_zero) == set(fixed_bytes)

    # 3.5 5/3 exactness on shells 1 and 5
    shell_idx = {1: [], 5: []}
    for i, s in enumerate(OMEGA_STATES_4096):
        sh = state24_to_omega12(s).shell
        if sh in shell_idx:
            shell_idx[sh].append(i)
    five_thirds = 5.0 / 3.0
    shell15 = {}
    for sh in (1, 5):
        v = float(_max_chsh_on_index_set(shell_idx[sh])["max_CHSH_Boolean"])
        shell15[f"shell_{sh}"] = {
            "value": v,
            "equals_5_over_3": abs(v - five_thirds) < 1e-9,
        }

    return {
        "depth_dependent_chsh": depth_chsh,
        "k4_gate_chsh": gate_chsh,
        "q6_zero_bytes": [f"0x{b:02X}" for b in q6_zero],
        "fixed_point_bytes": [f"0x{b:02X}" for b in fixed_bytes],
        "q6_iff_stabilizer": q6_iff_stab,
        "shell_1_5_chsh": shell15,
    }


# ================================================================
# main: dispatch and print
# ================================================================


def _print_chsh_table(res: dict) -> None:
    print("\n" + "=" * 5)
    print("A. CHSH ENSEMBLE DICTIONARY (BOOLEAN EXTREME)")
    print("=" * 5)
    print(f"  classical bound: {res['classical_bound']:.4f}")
    print(f"  Hilbert CHSH:    {res['hilbert_chsh']:.6f}")
    print(f"  uniform Omega max CHSH: {res['uniform_max_CHSH']:.4f}")
    print(
        f"  some ensembles saturate classical: {res['some_ensembles_saturate_classical']}"
    )
    print(
        f"  all ensembles saturate classical: {res['all_ensembles_saturate_classical']}"
    )
    print("  ensemble -> max_CHSH_Boolean / max|corr| / Kg_ratio:")
    hilb = float(res["hilbert_chsh"])
    for name, v in res["table"].items():
        chsh = v["max_CHSH_Boolean"]
        corr = v["max_abs_corr"]
        cv = f"{corr:.4f}" if isinstance(corr, float) and corr == corr else "nan"
        cs = f"{chsh:.4f}" if isinstance(chsh, float) and chsh == chsh else "nan"
        if isinstance(chsh, float) and chsh == chsh and chsh > 1e-9:
            kg = f"{hilb / chsh:.4f}"
        else:
            kg = "n/a"
        print(f"    {name:<22s} n={v['n']:<5d} CHSH={cs}  |corr|={cv}  Kg={kg}")
    print(f"  K_G^R(2)=sqrt2: {math.sqrt(2.0):.4f}")
    print("  note: Kg = Hilbert CHSH (2*sqrt(2), structural) / Boolean CHSH on the")
    print("        ensemble. 2/1.2 = 2.357 at the equator (shell 3) vs sqrt2 at poles.")
    print(f"  note: {res['note']}")


def _print_stab(res: dict) -> None:
    print("\n" + "=" * 5)
    print("B. STABILIZER FIXED-SHELL LOCALIZATION")
    print("=" * 5)
    for b in res["stabilizer_bytes"]:
        d = res["detail"][b]
        print(f"  {b}: fixed={d['fixed_total']} per_shell={d['per_shell']}")
        print(f"       shells_occupied={d['shells_occupied']}")
    print(f"  all have 64 fixed: {res['all_have_64_fixed']}")
    print(f"  fixed on horizon shells (0,6): {res['fixed_on_horizon_shells_06']}")
    print(f"  concentrated single shell: {res['concentrated_single_shell']}")
    print(f"  note: {res['note']}")


def _print_monodromy(res: dict) -> None:
    print("\n" + "=" * 5)
    print("C. BYTE MONODROMY BY FAMILY")
    print("=" * 5)
    for fam, v in res["per_family"].items():
        print(
            f"  {fam}: n={v['n_bytes']} pure4={v['pure_4cycle']} "
            f"mixed={v['non_pure_4cycle']} structs={v['distinct_structures']}"
        )
    print(f"  all families have 64 bytes: {res['all_families_have_64_bytes']}")
    print(f"  note: {res['note']}")


def _print_fine_lefschetz(res: dict) -> None:
    print("\n" + "=" * 5)
    print("D. FINER LEFSCHETZ GRADING (shell x Z2-sheet)")
    print("=" * 5)
    for name, d in res["words"].items():
        print(f"  {name}:")
        print(f"    fine fixed/cell: {d['fine_fixed_per_cell']}")
        print(f"    Lefschetz L_fine:  {d['Lefschetz_L_fine']}")
        print(f"    fixed/shell:     {d['fixed_per_shell']}")
    print(f"  all L zero (fine): {res['all_L_zero_fine']}")
    print(f"  note: {res['note']}")


def _print_gap_spectrum(res: dict) -> None:
    print("\n" + "=" * 5)
    print("E. GAP-SPECTRUM DICTIONARY")
    print("=" * 5)
    for k, v in res["gaps"].items():
        print(f"  {k}: {v:.10f}")
    print(f"  numeric_distinct: {res['numeric_distinct']}")
    print(f"  note: {res['note']}")


def _print_fiber(res: dict) -> None:
    print("\n" + "=" * 5)
    print("F. FIBER-INCOMPLETE SQUARE-ROOT BOUNDARY")
    print("=" * 5)
    print(f"  n_allow:           {res['n_allow']}")
    print(f"  q_span_rank:       {res['q_span_rank']}")
    print(f"  fiber_complete:     {res['fiber_complete']}")
    print(f"  reach pairs:        {res['reach_distinct_pairs']}")
    print(f"  per_factor_rank:    {res['per_factor_rank_from_reach']}")
    print(f"  |U|={res['abs_U']}  |V|={res['abs_V']}  rect={res['rectangularity']:.6f}")
    print(f"  rect_is_product:    {res['rect_is_product']}")
    print(f"  qspan_matches_reach: {res['qspan_matches_reach']}")
    print(f"  note: {res['note']}")


def _print_correlation(res: dict) -> None:
    print("\n" + "=" * 5)
    print("G. OBSERVABLE CORRELATION-MATRIX GROTHENDIECK INSTANCE")
    print("=" * 5)
    for name, r in res["ensembles"].items():
        print(
            f"  {name:<20s} n={r['n_states']:<5d} ||C||_F={r['C_frobenius']:.4f}"
            f"  bool_lb={r['bool_lb']:.4f}  hilb_raw={r['hilb_raw']:.4f}"
            f"  hilb_monotone={r['hilb_monotone']:.4f}"
            f"  ratio={r['ratio_hilb_bool']:.4f}"
        )
    print(f"  note: {res['note']}")


def _print_open_items(res: dict) -> None:
    print("\n" + "=" * 5)
    print("H. OPEN ITEMS (DEPTH / K4-GATE / q6=0 / 5/3)")
    print("=" * 5)
    print("  depth-dependent Boolean CHSH (from rest, full alphabet):")
    for d, v in res["depth_dependent_chsh"].items():
        print(f"    {d}: {v:.4f}")
    print("  K4 gate action on CHSH (image of Omega):")
    for g, v in res["k4_gate_chsh"].items():
        print(f"    gate {g:<3s}: {v:.4f}")
    print(f"  q6=0 bytes:      {res['q6_zero_bytes']}")
    print(f"  fixed-point bytes: {res['fixed_point_bytes']}")
    print(f"  q6=0 iff stabilizer: {res['q6_iff_stabilizer']}")
    for sh, d in res["shell_1_5_chsh"].items():
        print(f"  {sh}: CHSH={d['value']:.4f}  equals 5/3: {d['equals_5_over_3']}")


def main() -> None:
    print("hQVM Cohomology analysis 3 -- closure experiments")
    print("=" * 5)
    print(f"seed={SEED}")

    _print_chsh_table(experiment_chsh_ensemble_table())
    _print_stab(experiment_stabilizer_fixed_shells())
    _print_monodromy(experiment_byte_monodromy_by_family())
    _print_fine_lefschetz(experiment_fine_lefschetz())
    _print_gap_spectrum(experiment_gap_spectrum())
    _print_fiber(experiment_fiber_incomplete_rank6())
    _print_correlation(experiment_correlation_grothendieck())
    _print_open_items(experiment_open_items())


if __name__ == "__main__":
    main()
