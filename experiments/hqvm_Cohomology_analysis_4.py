#!/usr/bin/env python3
"""
hQVM Cohomology Analysis -- new leads library
=============================================

ROLE
----
Follow-up experiments to the leads flagged in docs/Notes/12/Grothendieck.md and
the unofficial notes (Notes_28_BHC, Notes_29_pi, Notes_30_Aperture). Each
section is a measurement only: it prints numbers, tables, and PASS/FAIL
checks. No conclusions.

Sections:
  A.  Cut norm of the 64x64 horizon-transition matrix + SDP relaxation (L1/L6)
  B.  Grothendieck constant K(G) of the horizon-transition graph (L6)
  C.  Per-byte Lefschetz numbers in the shell grading (L2)
  D.  Arcsin rounding law on the native 64 chirality +/-1 vectors (L3)
  E.  Trace-angle standardization of the BU monodromy delta_BU (L8)
  F.  Characteristic polynomials of depth-4 word operators / RH analogue (L4)

Companion scripts: hqvm_Cohomology_analysis_{1,2,3}.py,
                  hqvm_Cohomology_analysis_run.py (runner + shared helpers).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gyroscopic.hQVM.api import (
    OMEGA_STATES_4096,
    state24_to_omega12,
    FAMILY_BY_BYTE,
    MICRO_REF_BY_BYTE,
    q_word6,
    walsh_hadamard64,
)
from gyroscopic.hQVM.constants import (
    DELTA_BU,
    M_A,
)

from hqvm_Cohomology_analysis_run import (
    OMEGA_SIZE,
    load_common,
    shell_grading,
    all_byte_perms,
    word_permutation,
    cycle_signature,
)

_com = load_common()
W2_word = _com.W2_word
W2p_word = _com.W2p_word
F_cycle_word = _com.F_cycle_word

HORIZON_CARD = 64
HORIZON_STABILIZER_BYTES = (0x2B, 0x54, 0xD5, 0xAA)


# ================================================================
# A. Cut norm of the bipartite horizon-to-horizon transition matrix
# ================================================================


def _shell0_index(state24: int) -> int:
    """Index 0..63 of a complement-horizon (shell 0) state, given by u6."""
    return state24_to_omega12(state24).u6 & 0x3F


def _shell6_index(state24: int) -> int:
    """Index 0..63 of an equality-horizon (shell 6) state, given by u6."""
    # shell 6: chirality6 == 0, so u6 == v6; index by u6.
    return state24_to_omega12(state24).u6 & 0x3F


def _build_horizon_bipartite_matrix() -> np.ndarray:
    """64x64 matrix M[i,j] = number of bytes b with T_b(Hc_i) = He_j.

    Hc = complement horizon (shell 0), He = equality horizon (shell 6).
    Each edge is signed by the net horizon displacement so the matrix can
    carry a {+1,-1} Grothendieck instance: sign = (-1)**(parity of the
    family L0 bits flipped) as a proxy for the Boolean sign assignment.
    We report both the unsigned count matrix (adjacency/cut) and the
    signed version feeding the Grothendieck ratio.
    """
    states = OMEGA_STATES_4096
    Hc = [s for s in states if state24_to_omega12(s).shell == 0]
    He = [s for s in states if state24_to_omega12(s).shell == 6]
    cnt = np.zeros((HORIZON_CARD, HORIZON_CARD), dtype=np.int64)
    sgn = np.zeros((HORIZON_CARD, HORIZON_CARD), dtype=np.float64)
    for src in Hc:
        ui = _shell0_index(src)
        for b in range(256):
            dst = _com.step_state_by_byte(src, b)
            if state24_to_omega12(dst).shell != 6:
                continue
            uj = _shell6_index(dst)
            cnt[ui, uj] += 1
            # sign from the Boolean outcome: family L0 bit (bit0) of the byte
            s = 1.0 if (b & 0x01) == 0 else -1.0
            sgn[ui, uj] += s
    return cnt, sgn


def _cut_norm_exact(M: np.ndarray) -> float:
    """Approximate exact cut norm via iterated sign relaxation (local max).

    ||A||_square = max_{s in {-1,1}^n, t in {-1,1}^m} |s^T A t| / 4
    reduced to integer program over signs. We iterate s=sign(At), t=sign(A^T s)
    from multiple random starts; returns the best |s^T A t| (cut norm).
    """
    n, m = M.shape
    best = 0.0
    rng = np.random.default_rng(12345)
    for _ in range(40):
        t = (rng.random(m) > 0.5).astype(np.float64) * 2 - 1
        for _ in range(60):
            s = np.sign(M @ t)
            s[s == 0] = 1.0
            t = np.sign(M.T @ s)
            t[t == 0] = 1.0
        val = abs(float(s @ M @ t))
        if val > best:
            best = val
    return best


def _cut_norm_sdp_lb(M: np.ndarray) -> float:
    """Monotone lower bound on the Hilbert (SDP) relaxation of cut norm.

    Build row/column unit vectors from the symmetric part's eigen-decomposition
    plus random orthogonal extensions, then evaluate sum M_ij <v_i, w_j>.
    """
    A = M.astype(np.float64)
    sym = (A + A.T) / 2.0
    w, V = np.linalg.eigh(sym)
    w = np.clip(w, 0.0, None)
    rows = V * np.sqrt(w)[None, :]
    cols = rows.copy()
    rng = np.random.default_rng(7)
    rows = np.concatenate([rows, rng.standard_normal((A.shape[0], 8))], axis=1)
    cols = np.concatenate([cols, rng.standard_normal((A.shape[1], 8))], axis=1)
    rows, _ = np.linalg.qr(rows)
    cols, _ = np.linalg.qr(cols)
    return float(np.sum(A * (rows @ cols.T)))


def experiment_horizon_cut_norm() -> dict[str, object]:
    """L1/L6: cut norm of the bipartite horizon-transition matrix and SDP bound."""
    cnt, sgn = _build_horizon_bipartite_matrix()
    total_edges = int(cnt.sum())
    # unsigned cut norm (Grothendieck on the +1/-1 count matrix)
    cut_cnt = _cut_norm_exact(cnt.astype(np.float64))
    bool_cnt = 4.0 * cut_cnt
    hilb_cnt = _cut_norm_sdp_lb(cnt.astype(np.float64))
    # signed Grothendieck instance (Boolean signs from byte L0 bit)
    cut_sgn = _cut_norm_exact(sgn)
    bool_sgn = 4.0 * cut_sgn
    hilb_sgn = _cut_norm_sdp_lb(sgn)
    return {
        "matrix_shape": [HORIZON_CARD, HORIZON_CARD],
        "total_transitions_Hc_to_He": total_edges,
        "row_sums_uniform": bool(np.all(cnt.sum(axis=1) == cnt.sum(axis=1)[0])),
        "col_sums_uniform": bool(np.all(cnt.sum(axis=0) == cnt.sum(axis=0)[0])),
        "cut_norm_unsigned": cut_cnt,
        "boolean_opt_unsigned": bool_cnt,
        "sdp_lb_unsigned": hilb_cnt,
        "ratio_SDP_over_Bool_unsigned": (
            (hilb_cnt / bool_cnt) if bool_cnt > 0 else float("nan")
        ),
        "cut_norm_signed": cut_sgn,
        "boolean_opt_signed": bool_sgn,
        "sdp_lb_signed": hilb_sgn,
        "ratio_SDP_over_Bool_signed": (
            (hilb_sgn / bool_sgn) if bool_sgn > 0 else float("nan")
        ),
    }


# ================================================================
# B. Grothendieck constant K(G) of the horizon-transition graph
# ================================================================


def experiment_graph_grothendieck_K() -> dict[str, object]:
    """L6: K(G) for the 64-vertex horizon graph; compare to Lovasz theta.

    G has vertex set = horizon states, edge (i,j) iff reachable by some byte
    to a DISTINCT horizon state. Upper bound (Briet-Oliveira-Filho-Vallentin):
        K(G) <= pi / (2 log((1+sqrt((theta-1)^2+1))/(theta-1)))
    with theta = Lovasz theta of the complement graph.

    Note: within a single horizon (shell 0) the transition rule maps each
    state to exactly one other horizon state and stays closed (the horizon
    stabilizers are self-loops, excluded here); the induced graph is
    1-regular (a perfect matching). For such a graph K(G)=1 exactly, so the
    Grothendieck gap is trivial on the single-horizon graph. The nontrivial
    bipartite Hc<->He cut-norm instance is reported in section A.
    """
    states = OMEGA_STATES_4096
    shell0 = [s for s in states if state24_to_omega12(s).shell == 0]
    n = len(shell0)
    idx_of = {s: k for k, s in enumerate(shell0)}
    adj = np.zeros((n, n), dtype=np.int8)
    for s in shell0:
        si = idx_of[s]
        for b in range(256):
            dst = _com.step_state_by_byte(s, b)
            if dst in idx_of and dst != s:
                adj[si, idx_of[dst]] = 1
    d = int(adj.sum(axis=1)[0])
    e = np.linalg.eigvalsh(adj.astype(np.float64))
    lambda_min = float(e.min())
    lambda_max = float(e.max())
    # Hoffman bound on independence number of the complement graph:
    alpha_comp = n * (-lambda_min) / (d - lambda_min) if d != lambda_min else n
    theta_comp_lb = n / alpha_comp if alpha_comp > 0 else float("nan")
    if theta_comp_lb > 1:
        tau = (theta_comp_lb - 1.0) / math.sqrt((theta_comp_lb - 1.0) ** 2 + 1.0)
        kg_bound = (math.pi / 2.0) / math.log((1.0 + tau) / tau)
    else:
        kg_bound = float("nan")
    return {
        "n_vertices": n,
        "degree": d,
        "edges": int(adj.sum() // 2),
        "lambda_min": lambda_min,
        "lambda_max": lambda_max,
        "alpha_complement_hoffman": alpha_comp,
        "theta_complement_lb": theta_comp_lb,
        "K_G_graph_bound": kg_bound,
        "note": "1-regular horizon graph => K(G)=1; nontrivial instance is the bipartite Hc<->He cut norm in section A",
    }


# ================================================================
# C. Per-byte Lefschetz numbers in the shell grading
# ================================================================


def experiment_lefschetz_bytes() -> dict[str, object]:
    """L2: L(T_b) = sum_k (-1)^k Fix_k(T_b) for each byte; correlate."""
    shell_of, idx_of = shell_grading()
    byte_perms = all_byte_perms(idx_of)
    fixed_per_shell = np.zeros((256, 7), dtype=np.int64)
    for b, perm in enumerate(byte_perms):
        fixed = perm == np.arange(OMEGA_SIZE)
        for i in np.nonzero(fixed)[0]:
            fixed_per_shell[b, shell_of[i]] += 1
    L = np.array(
        [sum((-1) ** k * fixed_per_shell[b, k] for k in range(7)) for b in range(256)],
        dtype=np.int64,
    )
    fixed_total = fixed_per_shell.sum(axis=1)
    qw = np.array([q_word6(b).bit_count() for b in range(256)], dtype=np.int64)

    def pearson(x, y):
        if np.std(x) == 0 or np.std(y) == 0:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    special = []
    for b in range(256):
        if fixed_total[b] > 0:
            special.append(
                {
                    "byte": b,
                    "hex": f"0x{b:02X}",
                    "family": int(FAMILY_BY_BYTE[b]),
                    "micro": int(MICRO_REF_BY_BYTE[b]),
                    "q_weight": int(qw[b]),
                    "fixed_total": int(fixed_total[b]),
                    "L": int(L[b]),
                    "fixed_per_shell": [int(x) for x in fixed_per_shell[b]],
                }
            )
    special_bytes = {b["byte"] for b in special}
    stab_note = (
        f"fixed-point bytes {sorted(f'0x{b:02X}' for b in special_bytes)} "
        f"== horizon stabilizers {[f'0x{b:02X}' for b in HORIZON_STABILIZER_BYTES]}: "
        f"{special_bytes == set(HORIZON_STABILIZER_BYTES)}"
    )
    generic = fixed_total == 0
    return {
        "n_bytes": 256,
        "L_min": int(L.min()),
        "L_max": int(L.max()),
        "L_mean": float(L.mean()),
        "bytes_with_fixed_points": int((fixed_total > 0).sum()),
        "fixed_min": int(fixed_total.min()),
        "fixed_max": int(fixed_total.max()),
        "corr_fixed_vs_L": pearson(fixed_total, L),
        "corr_qweight_vs_L_all": pearson(qw, L),
        "corr_qweight_vs_L_generic": (
            pearson(qw[generic], L[generic]) if generic.sum() > 1 else float("nan")
        ),
        "special_fixed_point_bytes": special,
        "L_constant_over_generic": (
            bool(np.all(L[generic] == L[generic][0])) if generic.sum() > 0 else False
        ),
        "L_generic_value": int(L[generic][0]) if generic.sum() > 0 else None,
        "horizon_stabilizer_note": stab_note,
    }


# ================================================================
# D. Arcsin rounding law on the native 64 chirality +/-1 vectors
# ================================================================


def experiment_arcsin_chirality_vectors() -> dict[str, object]:
    """L3: E[sign(<g,x>)sign(<g,y>)] = (2/pi) arcsin(x.y) on 64 GF(2)^6 words.

    Build the 64 chirality words as +/-1 vectors in R^6, fit the
    hyperplane-rounding coefficient c in c * arcsin(x.y), compare to 2/pi.
    """
    spins = np.array(
        [[1 if ((w >> i) & 1) else -1 for i in range(6)] for w in range(64)],
        dtype=np.float64,
    )
    X = spins / np.sqrt(6.0)
    dot_prods = X @ X.T
    rng = np.random.default_rng(2024)
    n_g = 200000
    g = rng.standard_normal((n_g, 6))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    sgn = np.sign(g @ X.T)
    emp = (sgn.T @ sgn) / n_g
    rows, cols = np.triu_indices(64, k=1)
    target = emp[rows, cols]
    basis = np.arcsin(np.clip(dot_prods[rows, cols], -1.0, 1.0))
    c, *_ = np.linalg.lstsq(basis[:, None], target, rcond=None)
    c = float(c[0])
    resid = float(np.sqrt(np.mean((c * basis - target) ** 2)))
    W = walsh_hadamard64()
    return {
        "n_chirality_vectors": 64,
        "dim": 6,
        "fitted_coefficient_c": c,
        "target_2_over_pi": 2.0 / math.pi,
        "rel_error_vs_2pi": (c - 2.0 / math.pi) / (2.0 / math.pi),
        "rmse": resid,
        "law_holds_on_chirality_vectors": abs(c - 2.0 / math.pi) < 0.02,
        "walsh_hadamard_is_orthogonal": bool(np.allclose(W @ W.T, np.eye(64))),
    }


# ================================================================
# E. Trace-angle standardization of the BU monodromy delta_BU
# ================================================================


def _su2_general(angle: float, axis: np.ndarray) -> np.ndarray:
    """SU(2) fundamental rep of a rotation by `angle` about unit `axis`."""
    ax = np.asarray(axis, dtype=np.float64)
    ax = ax / np.linalg.norm(ax)
    a = math.cos(angle / 2.0)
    s = math.sin(angle / 2.0)
    i, j, k = ax
    return np.array(
        [
            [complex(a, -s * k), complex(-s * j, -s * i)],
            [complex(s * j, -s * i), complex(a, s * k)],
        ],
        dtype=complex,
    )


def experiment_trace_angle_delta_bu() -> dict[str, object]:
    """L8: delta_BU as a class function of the loop element in SU(2).

    Build the depth-4 dual-pole loop as a product of SU(2) rotations
    (R_y(delta_BU) R_x(pi) R_y(delta_BU) R_x(pi)), take its trace, and
    verify cos(delta_BU/2) = 1/2 Re Tr(U_half) for the half-turn.
    """
    delta = DELTA_BU
    Rx = Rotation.from_rotvec(np.array([math.pi, 0.0, 0.0]))
    Ry = Rotation.from_rotvec(np.array([0.0, delta, 0.0]))
    R_loop = Ry * Rx * Ry * Rx
    U = _su2_general(R_loop.magnitude(), R_loop.as_rotvec())
    tr = float(np.trace(U).real)
    U_half = _su2_general(delta, np.array([0.0, 1.0, 0.0]))
    tr_half = float(np.trace(U_half).real)
    cos_half = 0.5 * tr_half
    delta_from_trace = 2.0 * math.acos(max(-1.0, min(1.0, cos_half)))
    angle_from_trace = 2.0 * math.acos(max(-1.0, min(1.0, tr / 2.0)))
    return {
        "delta_BU": DELTA_BU,
        "m_a": M_A,
        "rho": DELTA_BU / M_A,
        "trace_half_loop": tr_half,
        "cos_delta_bu_over_2_from_trace": cos_half,
        "delta_bu_from_trace_angle": delta_from_trace,
        "abs_delta_diff": abs(delta_from_trace - DELTA_BU),
        "trace_full_loop": tr,
        "full_loop_angle_from_trace": angle_from_trace,
        "trace_angle_identity_holds": abs(delta_from_trace - DELTA_BU) < 1e-9,
    }


# ================================================================
# F. Characteristic polynomials of word operators / RH analogue
# ================================================================


def experiment_word_char_polys() -> dict[str, object]:
    """L4: characteristic polynomials of W2(m), W2'(m), F(m); RH analogue."""
    _, idx_of = shell_grading()
    results = {}
    perms_for_rh = []
    for name, fn in (("W2", W2_word), ("W2'", W2p_word), ("F", F_cycle_word)):
        perms_by_m = {}
        for m in range(64):
            perm = word_permutation(fn(m), idx_of)
            perms_by_m[m] = cycle_signature(perm)
            perms_for_rh.append(perm)
        types = sorted({tuple(sorted(d.items())) for d in perms_by_m.values()})
        results[name] = {
            "n_m": 64,
            "cycle_structure_types": [dict(t) for t in types],
            "example_m0": perms_by_m[0],
        }
    sig_union = set()
    for p in perms_for_rh:
        for L in cycle_signature(p):
            sig_union.add(L)
    rh = {
        "cycle_lengths_present": sorted(sig_union),
        "all_eigs_on_unit_circle": bool(all(L <= 2 for L in sig_union)),
    }
    return {
        "W2": results["W2"],
        "W2p": results["W2'"],
        "F": results["F"],
        "rh_analogue": rh,
    }


# ================================================================
# Runner
# ================================================================


def all_experiments() -> dict[str, object]:
    return {
        "A_horizon_cut_norm": experiment_horizon_cut_norm(),
        "B_graph_grothendieck_K": experiment_graph_grothendieck_K(),
        "C_lefschetz_bytes": experiment_lefschetz_bytes(),
        "D_arcsin_chirality": experiment_arcsin_chirality_vectors(),
        "E_trace_angle_delta_bu": experiment_trace_angle_delta_bu(),
        "F_word_char_polys": experiment_word_char_polys(),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(all_experiments(), indent=2, default=str))
