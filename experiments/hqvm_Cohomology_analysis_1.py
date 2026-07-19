#!/usr/bin/env python3
"""
hQVM Cohomology Analysis -- experiment library
==============================================

ROLE
----
Library of experiments testing the structural overlap between the Grothendieck
inequality (the sign-vs-Hilbert integrality gap) and the CGM/hQVM kernel.

This script is the library only. It prints PASS/FAIL checks, measurements, and
tables. The runner (hqvm_Cohomology_analysis_run.py) executes it and writes
the output.

Sections:
  A.  Constant and identity audit
  B.  Arcsin / hyperplane-rounding law on the chirality register
  C.  Finite Grothendieck gap (Bool exact + Hilb_lb via altmax, n<=7)
  D.  CHSH / Tsirelson (Hilbert lift; CHSH library defined in this file)
  E.  Horizon bipartition (census + signed kernel)
  F.  Depth-4 / K_G^R(4) = pi/2 gap candidates
  G.  Shannon triad on the same run
  H.  Square-root product spot-check

Companion script: hqvm_Cohomology_analysis_run.py (runner + shared helpers)
"""

from __future__ import annotations

import itertools
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import cast

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXP_DIR = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

from gyroscopic.hQVM.constants import (
    DELTA_BU,
    M_A,
    RHO,
    APERTURE_GAP,
    GENE_MAC_REST,
    OMEGA_SIZE,
    LAYER_MASK_12,
    apply_gate,
)
from gyroscopic.hQVM.api import (
    OMEGA_STATES_4096,
    chirality_word6,
    q_word6,
    Q_WEIGHT_BY_BYTE,
    state24_to_omega12,
    omega12_to_state24,
    walsh_hadamard64,
    component12_to_spin6,
    shell_population,
)
from gyroscopic.hQVM.kernel import Gyroscopic

SEED = 20260702
rng = random.Random(SEED)

clamp = lambda x: max(-1.0, min(1.0, x))


# ================================================================
# A. Constant and identity audit
# ================================================================


def experiment_constants() -> dict[str, object]:
    """Audit CGM constants and the algebraic identities that tie them to
    the Grothendieck/Krivine constants."""
    s_p = math.pi / 2.0
    u_p = 1.0 / math.sqrt(2.0)
    o_p = math.pi / 4.0
    m_a = M_A
    Q_G = 4.0 * math.pi

    ident = {
        "s_p=pi/2": s_p,
        "u_p=1/sqrt2": u_p,
        "o_p=pi/4": o_p,
        "m_a=1/(2 sqrt(2 pi))": m_a,
        "Q_G=4 pi": Q_G,
        "DELTA_BU": DELTA_BU,
        "RHO=DELTA_BU/m_a": RHO,
        "APERTURE_GAP=1-RHO": APERTURE_GAP,
    }

    checks = {}

    # Aperture identities
    checks["Q_G * m_a^2 == 1/2"] = (Q_G * m_a * m_a) - 0.5
    checks["m_a^2 * 4 pi^2 == pi/2"] = (m_a * m_a * 4.0 * math.pi * math.pi) - s_p

    # SU(2) holonomy closed form
    phi_SU2 = 2.0 * math.acos((1.0 + 2.0 * math.sqrt(2.0)) / 4.0)
    ident["phi_SU2=2 arccos((1+2 sqrt2)/4)"] = phi_SU2

    # Krivine constant and hyperbolic identities
    krivine = math.pi / (2.0 * math.log(1.0 + math.sqrt(2.0)))
    arsinh1 = math.asinh(1.0)
    artanh_up = math.atanh(u_p)
    tan3pi8 = math.tan(3.0 * math.pi / 8.0)
    ident["Krivine=pi/(2 ln(1+sqrt2))"] = krivine
    ident["arsinh(1)"] = arsinh1
    ident["artanh(1/sqrt2)"] = artanh_up
    ident["tan(3 pi/8)"] = tan3pi8

    checks["ln(1+sqrt2)==arsinh(1)"] = math.log(1.0 + math.sqrt(2.0)) - arsinh1
    checks["ln(1+sqrt2)==artanh(1/sqrt2)"] = math.log(1.0 + math.sqrt(2.0)) - artanh_up
    checks["1+sqrt2==tan(3 pi/8)"] = (1.0 + math.sqrt(2.0)) - tan3pi8
    checks["Krivine==pi/(2*arsinh1)"] = krivine - (math.pi / (2.0 * arsinh1))

    # zeta and alpha correction factors
    zeta = 8.0 / (m_a * math.sqrt(3.0))
    alpha0 = DELTA_BU**4 / m_a
    ident["zeta=8/(m_a sqrt3)"] = zeta
    ident["alpha0=DELTA_BU^4/m_a"] = alpha0
    lhs = alpha0 * zeta
    rhs = RHO**4 / (math.pi * math.sqrt(3.0))
    checks["alpha0*zeta==RHO^4/(pi sqrt3)"] = lhs - rhs

    # pi/2 vs Krivine ratio
    ident["(pi/2)/Krivine"] = s_p / krivine

    # UNA <-> Krivine rapidity chain: (pi/2)/Krivine == arsinh(1) == artanh(u_p)
    checks["(pi/2)/Krivine==arsinh(1)"] = (s_p / krivine) - arsinh1
    checks["(pi/2)/Krivine==artanh(u_p)"] = (s_p / krivine) - artanh_up
    checks["arsinh(1)==artanh(u_p)"] = arsinh1 - artanh_up

    return {"values": ident, "checks": checks}


# ================================================================
# B. Arcsin / hyperplane-rounding law on the chirality register
# ================================================================


def _sign(x: float) -> int:
    return 1 if x >= 0.0 else -1


def _vector_from_signs(signs: tuple[int, ...]) -> np.ndarray:
    v = np.array(signs, dtype=np.float64)
    n = np.linalg.norm(v)
    if n == 0.0:
        return v
    return v / n


def _arcsin_expectation_analytic(x: np.ndarray, y: np.ndarray) -> float:
    """E[ sign(g.x) sign(g.y) ] for g ~ N(0,I): (2/pi) arcsin(x.y / |x||y|)."""
    xy = float(np.dot(x, y))
    nx = float(np.linalg.norm(x))
    ny = float(np.linalg.norm(y))
    if nx == 0.0 or ny == 0.0:
        return 0.0
    c = xy / (nx * ny)
    c = clamp(c)
    return (2.0 / math.pi) * math.asin(c)


def _arcsin_expectation_mc(x: np.ndarray, y: np.ndarray, n_gauss: int) -> float:
    d = len(x)
    s = 0
    for _ in range(n_gauss):
        # full gaussian vector
        gv = np.array([rng.gauss(0.0, 1.0) for _ in range(d)])
        s += _sign(np.dot(gv, x)) * _sign(np.dot(gv, y))
    return s / float(n_gauss)


def experiment_arcsin_rounding(n_gauss: int = 20000) -> dict[str, float]:
    """Test the Grothendieck rounding identity on the native 6-bit sign space.

    Compare empirical E[sign<g,x> sign<g,y>] against (2/pi) arcsin(<x_hat, y_hat>)
    for a set of +/-1 vectors, and fit a single coefficient c on all pairs.
    """
    dim = 6
    # single-+1 vectors v_i = (-1,...,+1,...,-1): NOT an orthogonal set.
    # Two distinct ones have dot product 2 and normalized inner product 1/3.
    bases = []
    for i in range(dim):
        v = [-1] * dim
        v[i] = 1
        bases.append(np.array(v, dtype=np.float64))

    pairs = []
    for i in range(dim):
        for j in range(i + 1, dim):
            pairs.append((bases[i], bases[j]))

    analytic_vals = []
    mc_vals = []
    for x, y in pairs:
        a = _arcsin_expectation_analytic(x, y)
        m = _arcsin_expectation_mc(x, y, n_gauss)
        analytic_vals.append(a)
        mc_vals.append(m)

    # The single-+1 vectors have normalized pairwise inner product 1/3, so the
    # analytic expectation is nonzero and the MC should match it. To fit the
    # coefficient c, also use random (non-orthogonal) sign vectors.
    n_fit = 400
    fit_x = []
    fit_y = []
    fit_ip = []
    for _ in range(n_fit):
        xi = rng.randint(0, 1)
        xj = rng.randint(0, 1)
        # random +/-1 vectors of dim 6
        vx = np.array(
            [(1 if rng.randint(0, 1) else -1) for _ in range(dim)], dtype=np.float64
        )
        vy = np.array(
            [(1 if rng.randint(0, 1) else -1) for _ in range(dim)], dtype=np.float64
        )
        ip = float(np.dot(vx, vy)) / dim  # normalized inner product in [-1,1]
        fit_x.append(vx)
        fit_y.append(vy)
        fit_ip.append(ip)

    # least-squares fit: obs ~ a * arcsin(ip). The Grothendieck rounding
    # identity is E[sign<g,x> sign<g,y>] = (2/pi) arcsin(<x_hat, y_hat>),
    # so the fitted coefficient a should equal 2/pi.
    design = np.array([math.asin(clamp(ip)) for ip in fit_ip], dtype=np.float64)
    obs = np.array(
        [
            _arcsin_expectation_mc(vx, vy, max(200, n_gauss // 20))
            for vx, vy in zip(fit_x, fit_y)
        ],
        dtype=np.float64,
    )
    if np.std(design) < 1e-9:
        a_fit = float("nan")
    else:
        a_fit, *_ = np.linalg.lstsq(design[:, None], obs, rcond=None)
        a_fit = float(a_fit[0])

    target = 2.0 / math.pi
    max_err = (
        max(abs(m - a) for m, a in zip(mc_vals, analytic_vals)) if mc_vals else 0.0
    )
    mean_err = (
        (sum(abs(m - a) for m, a in zip(mc_vals, analytic_vals)) / len(mc_vals))
        if mc_vals
        else 0.0
    )

    # Analytic identity check: for each basis pair the exact expectation is
    # (2/pi) * arcsin(<x_hat, y_hat>); verify the coefficient equals 2/pi
    # on the normalized inner products directly (no MC).
    analytic_coeff_resid = max(
        abs((2.0 / math.pi) * math.asin(clamp(ip)) - _arcsin_expectation_analytic(x, y))
        for x, y in pairs
    )

    return {
        "n_pairs_basis": len(pairs),
        "n_gauss": n_gauss,
        "mc_vs_analytic_max_err": max_err,
        "mc_vs_analytic_mean_err": mean_err,
        "analytic_coeff_resid": analytic_coeff_resid,
        "fitted_coefficient_c": a_fit,
        "target_2_over_pi": target,
        "c_rel_err": (a_fit - target) / target if a_fit == a_fit else float("nan"),
    }


def experiment_arcsin_rounding_kernel(
    n_gauss: int = 4000,
    n_pairs: int = 200,
) -> dict[str, object]:
    """Grothendieck rounding identity on kernel-derived observable vectors.

    Section B tests the arcsin law on arbitrary +/-1 vectors. Here the tests
    use the hQVM's own Walsh observable vectors: for the complement-horizon
    ensemble E, each A-mask a defines x_a in {+/-1}^{|E|} (its evaluation
    across states). The arcsin rounding law is then a statement about rounding
    correlations in the kernel's observable feature space, not generic vectors.
    """
    from hqvm_Cohomology_analysis_run import walsh_matrix

    comp_idx = [
        i
        for i, s in enumerate(OMEGA_STATES_4096)
        if (s >> 12) & LAYER_MASK_12 == ((s & LAYER_MASK_12) ^ LAYER_MASK_12)
    ]
    spins = []
    for i in comp_idx:
        s = OMEGA_STATES_4096[i]
        a12 = (s >> 12) & LAYER_MASK_12
        b12 = s & LAYER_MASK_12
        try:
            spins.append(
                np.array(
                    component12_to_spin6(a12) + component12_to_spin6(b12),
                    dtype=np.float64,
                )
            )
        except ValueError:
            continue
    obs = walsh_matrix(np.array(spins, dtype=np.float64))  # (N, 64)
    # use non-constant masks 1..63 as kernel feature vectors
    feats = [obs[:, m] for m in range(1, 64)]

    max_err = 0.0
    tested = 0
    for _ in range(n_pairs):
        a = rng.randrange(len(feats))
        b = rng.randrange(len(feats))
        x, y = feats[a], feats[b]
        analytic = _arcsin_expectation_analytic(x, y)
        mc = _arcsin_expectation_mc(x, y, n_gauss)
        max_err = max(max_err, abs(mc - analytic))
        tested += 1
    return {
        "ensemble": "complement_horizon",
        "n_feature_vectors": len(feats),
        "n_pairs_tested": tested,
        "mc_vs_analytic_max_err_kernel": max_err,
        "arcsin_law_holds_on_kernel_observables": max_err < 0.1,
    }


# ================================================================
# C. Finite Grothendieck ratio on the sign norm
# ================================================================


def bool_norm(A: np.ndarray) -> float:
    """Exact ||A||_{inf->1} = max_{eps, delta in {+/-1}^n} sum_ij A_ij eps_i delta_j.

    Exhaustive over rows (2^n_rows iterations); limited to n_rows <= 9.
    """
    n_rows, _ = A.shape
    if n_rows > 9:
        raise ValueError("bool_norm exhaustive side requires n_rows <= 9")
    # Precompute all 2^n_rows sign vectors; vectorize the column-sum max.
    sign_bit = np.arange(n_rows)
    ei = np.arange(1 << n_rows)[:, None]
    eps = np.where((ei >> sign_bit) & 1, 1.0, -1.0)
    colsums = eps @ A
    return float(np.max(np.sum(np.abs(colsums), axis=1)))


def bool_cut_norm_local(A: np.ndarray, restarts: int = 32, sweeps: int = 50) -> float:
    """Approximate Boolean cut norm via greedy coordinate ascent (valid lower bound).

    Maximizes sum_ij A_ij eps_i delta_j over eps in {+/-1}^rows, delta in {+/-1}^cols
    by alternating coordinate flips from random starts. Used when exhaustive
    enumeration is infeasible (large matrices).
    """
    n_rows, n_cols = A.shape
    best = 0.0
    for _ in range(restarts):
        eps = np.array(
            [(1 if rng.randint(0, 1) else -1) for _ in range(n_rows)], dtype=np.float64
        )
        delta = np.array(
            [(1 if rng.randint(0, 1) else -1) for _ in range(n_cols)], dtype=np.float64
        )
        for _ in range(sweeps):
            # flip eps_i if it increases objective
            for i in range(n_rows):
                row = A[i]
                d = float(np.dot(row, delta))
                if -eps[i] * d > eps[i] * d:
                    eps[i] = -eps[i]
            for j in range(n_cols):
                col = A[:, j]
                d = float(np.dot(col, eps))
                if -delta[j] * d > delta[j] * d:
                    delta[j] = -delta[j]
        val = float(np.dot(eps, A @ delta))
        if val > best:
            best = val
    return best


def spectral_norm(A: np.ndarray) -> float:
    """Spectral norm ||A||_2 = largest singular value (exact).

    Loose upper bound: gamma2(A) <= ||A||_2 * sqrt(min(m, n)).
    Not used as the Hilbert-side Grothendieck value; see hilbert_relax_altmax.
    """
    return float(np.linalg.norm(A, 2))


def hilbert_relax_altmax(
    A: np.ndarray,
    k: int = 16,
    restarts: int = 50,
    iters: int = 50,
    seed: int = SEED,
) -> float:
    """Feasible lower bound on the Hilbert relaxation

        max_{u_i, v_j in S^{k-1}} sum_ij A_ij <u_i, v_j>

    via alternating maximization with random restarts. Returns a feasible
    value (lower bound on the true SDP optimum / gamma2).
    """
    m, n = A.shape
    best = -np.inf
    local_rng = np.random.default_rng(seed)
    for r in range(restarts):
        U = local_rng.normal(size=(m, k))
        V = local_rng.normal(size=(n, k))
        U /= np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-15)
        V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-15)
        for _ in range(iters):
            U_new = A @ V
            U = U_new / np.maximum(np.linalg.norm(U_new, axis=1, keepdims=True), 1e-15)
            V_new = A.T @ U
            V = V_new / np.maximum(np.linalg.norm(V_new, axis=1, keepdims=True), 1e-15)
        val = float(np.sum(A * (U @ V.T)))
        if val > best:
            best = val
    return best


def _matrix_stats(A: np.ndarray) -> dict[str, float]:
    row_sums = A.sum(axis=1)
    return {
        "max_abs": float(np.max(np.abs(A))),
        "fro_norm": float(np.linalg.norm(A, "fro")),
        "row_sum_mean": float(np.mean(row_sums)),
        "row_sum_min": float(np.min(row_sums)),
        "row_sum_max": float(np.max(row_sums)),
    }


def _structured_grothendieck_entry(
    A: np.ndarray,
    bool_fn,
    *,
    bool_method: str,
    k: int = 16,
    restarts: int = 40,
    iters: int = 40,
) -> dict[str, object]:
    stats = _matrix_stats(A)
    bb = float(bool_fn(A))
    hilb = hilbert_relax_altmax(A, k=k, restarts=restarts, iters=iters)
    ratio = hilb / bb if bb > 0 else float("inf")
    return {
        **stats,
        "bool_lb": bb,
        "hilb_lb": hilb,
        "ratio_hilb_over_bool": ratio,
        "ratio_ge_1": hilb >= bb - 1e-6,
        "bool_method": bool_method,
        "hilb_method": "altmax (feasible lower bound)",
        "spectral_norm": spectral_norm(A),
    }


def experiment_finite_grothendieck(
    n: int = 6,
    n_trials: int = 200,
) -> dict[str, object]:
    """Exact Boolean norm ||A||_{inf->1} and Hilbert relaxation lower bound on
    random Gaussian and kernel-native matrices.

    Reports R(A) = Hilb_lb(A) / Bool(A), a finite Grothendieck-style gap.
    Hilb_lb comes from alternating maximization (feasible lower bound on gamma2).
    """
    bool_vals = []
    hilb_vals = []
    ratio_vals = []
    for t in range(n_trials):
        A = np.array([[rng.gauss(0.0, 1.0) for _ in range(n)] for _ in range(n)])
        bb = bool_norm(A)
        hilb = hilbert_relax_altmax(A, k=16, restarts=40, iters=40, seed=SEED + t)
        bool_vals.append(bb)
        hilb_vals.append(hilb)
        ratio_vals.append(hilb / bb if bb > 0 else float("inf"))

    structured = {}
    if n == 6:

        def pop(x: int) -> int:
            return int(x).bit_count()

        W = np.array(
            [[(-1) ** pop(i & j) for j in range(64)] for i in range(64)],
            dtype=np.float64,
        )
        H = np.array(
            [[(-1) ** pop(i ^ j) for j in range(64)] for i in range(64)],
            dtype=np.float64,
        )
        C = _kernel_signed_block()
        structured["Kernel_signed_64"] = _structured_grothendieck_entry(
            C,
            bool_cut_norm_local,
            bool_method="bool_cut_norm_local (approx, valid lower bound)",
            k=32,
            restarts=20,
            iters=30,
        )
        for block, label in _chirality_blocks(W, H):
            structured[label] = _structured_grothendieck_entry(
                block,
                bool_norm,
                bool_method="bool_norm (exact)",
                k=16,
                restarts=40,
                iters=40,
            )

    krivine = math.pi / (2.0 * math.log(1.0 + math.sqrt(2.0)))
    pi2 = math.pi / 2.0
    ratio_arr = np.array(ratio_vals)
    tol = 1e-4
    n_below = int(np.sum(ratio_arr < 1.0 - tol))

    return {
        "n": n,
        "n_trials": n_trials,
        "bool_mean": float(np.mean(bool_vals)),
        "bool_min": float(np.min(bool_vals)),
        "bool_max": float(np.max(bool_vals)),
        "hilb_mean": float(np.mean(hilb_vals)),
        "hilb_min": float(np.min(hilb_vals)),
        "hilb_max": float(np.max(hilb_vals)),
        "ratio_mean": float(np.mean(ratio_arr)),
        "ratio_min": float(np.min(ratio_arr)),
        "ratio_max": float(np.max(ratio_arr)),
        "ratios_below_1_tol": n_below,
        "ratio_tol": tol,
        "K_G^R(2)=sqrt2": math.sqrt(2.0),
        "K_G^R(4)=pi/2": pi2,
        "Krivine_bound": krivine,
        "note": (
            "Achieved ratio = Hilb_lb/Bool. Hilb_lb is a feasible LOWER "
            "BOUND on gamma2 (altmax); Bool is exact for n<=9. So the "
            "ratio is a lower bound on (true Hilb)/Bool, NOT a K_G "
            "estimator. Max random ratio ~1.20; Walsh block 1.21. The "
            "sharp kernel instance of K_G^R(2) is the CHSH gap (section D2), "
            "not these random matrices."
        ),
        "structured": structured,
    }


def _kernel_signed_block() -> np.ndarray:
    """64x64 kernel-native Grothendieck instance that is sign-indefinite.

    DIFFERENCE of two transition kernels between disjoint byte pools
    (two q-layers w1, w2), over the full 64-dim chirality fiber:

        M_i,j = P(chi=j | start chi=i, pool w1) - P(chi=j | start chi=i, pool w2)

    The difference matrix has mixed signs, so it is not factorable and
    genuinely probes the Grothendieck gap (unlike the uniform-transition
    correlator, which is identically zero by depth-2 typicality).
    """
    from gyroscopic.hQVM.constants import step_state_by_byte

    w1, w2 = 1, 3
    pool1 = [b for b in range(256) if Q_WEIGHT_BY_BYTE[b] == w1]
    pool2 = [b for b in range(256) if Q_WEIGHT_BY_BYTE[b] == w2]

    def kernel(pool: list[int]) -> np.ndarray:
        counts = np.zeros((64, 64))
        for s in OMEGA_STATES_4096:
            chi_i = chirality_word6(s)
            for b in pool:
                nxt = step_state_by_byte(s, b)
                chi_j = chirality_word6(nxt)
                counts[chi_i, chi_j] += 1.0
        counts /= counts.sum(axis=1, keepdims=True)
        return counts

    return kernel(pool1) - kernel(pool2)


def _chirality_blocks(W: np.ndarray, H: np.ndarray):
    """Return 6x6 blocks of the kernel kernels for Boolean norm tests."""
    b1 = W[:6, :6]
    idx = [0, 1, 2, 32, 33, 34]
    b2 = W[np.ix_(idx, idx)]
    bh = H[np.ix_(idx, idx)]
    return [
        (b1, "Walsh_block_first6"),
        (b2, "Walsh_block_split"),
        (bh, "Hamming_block_split"),
    ]


# ================================================================
# D. CHSH / Tsirelson (Hilbert lift)
# ================================================================

# CHSH / Tsirelson Hilbert-lift checks (ported from
# superintelligence/tests/test_hQVM_2.py::TestBellCHSH).
TSIRELSON = 2.0 * np.sqrt(2.0)
CLASSICAL = 2.0


def _bell_pair_state(t_bit: int) -> np.ndarray:
    v = np.zeros(4, dtype=complex)
    if t_bit == 0:
        v[0] = 1.0 / np.sqrt(2)
        v[3] = 1.0 / np.sqrt(2)
    else:
        v[1] = 1.0 / np.sqrt(2)
        v[2] = 1.0 / np.sqrt(2)
    return v


def _bell_projector(t_bit: int) -> np.ndarray:
    v = _bell_pair_state(t_bit)
    return np.outer(v, np.conj(v))


def _chsh_expectation(rho: np.ndarray, op: np.ndarray) -> float:
    return float(np.trace(rho @ op).real)


def chsh_for_tbit(t_bit: int) -> float:
    rho = _bell_projector(t_bit)
    z = np.array([[1, 0], [0, -1]], dtype=float)
    x = np.array([[0, 1], [1, 0]], dtype=float)
    a1, a2 = z, x
    if t_bit == 0:
        b1 = (z + x) / np.sqrt(2)
        b2 = (z - x) / np.sqrt(2)
    else:
        b1 = (-z + x) / np.sqrt(2)
        b2 = (-z - x) / np.sqrt(2)
    return (
        _chsh_expectation(rho, np.kron(a1, b1))
        + _chsh_expectation(rho, np.kron(a1, b2))
        + _chsh_expectation(rho, np.kron(a2, b1))
        - _chsh_expectation(rho, np.kron(a2, b2))
    )


def _graph_state_tensor_qsum(t6: int) -> np.ndarray:
    psi = np.zeros((2,) * 12, dtype=complex)
    amp = 1.0 / 8.0
    for q in range(64):
        idx = []
        for k in range(6):
            a = (q >> k) & 1
            b = a ^ ((t6 >> k) & 1)
            idx.extend([a, b])
        psi[tuple(idx)] = amp
    return psi


def _reduced_density(psi_tensor: np.ndarray, keep: list[int]) -> np.ndarray:
    n = psi_tensor.ndim
    keep_sorted = tuple(sorted(int(i) for i in keep))
    trace_axes = tuple(i for i in range(n) if i not in keep_sorted)
    rho = np.tensordot(np.conj(psi_tensor), psi_tensor, axes=(trace_axes, trace_axes))
    dim = 1 << len(keep_sorted)
    return rho.reshape(dim, dim)


def chsh_graph_state_pair(t6: int, k: int) -> float:
    psi = _graph_state_tensor_qsum(t6)
    rho_k = _reduced_density(psi, [2 * k, 2 * k + 1])
    z = np.array([[1, 0], [0, -1]], dtype=float)
    x = np.array([[0, 1], [1, 0]], dtype=float)
    a1, a2 = z, x
    if ((t6 >> k) & 1) == 0:
        b1 = (z + x) / np.sqrt(2)
        b2 = (z - x) / np.sqrt(2)
    else:
        b1 = (-z + x) / np.sqrt(2)
        b2 = (-z - x) / np.sqrt(2)
    return (
        _chsh_expectation(rho_k, np.kron(a1, b1))
        + _chsh_expectation(rho_k, np.kron(a1, b2))
        + _chsh_expectation(rho_k, np.kron(a2, b1))
        - _chsh_expectation(rho_k, np.kron(a2, b2))
    )


def max_chsh_angle_grid(n: int = 10) -> float:
    psi = np.zeros(4, dtype=complex)
    psi[0] = psi[3] = 1.0 / np.sqrt(2)
    rho = np.outer(psi, np.conj(psi))
    z = np.array([[1, 0], [0, -1]], dtype=float)
    x = np.array([[0, 1], [1, 0]], dtype=float)
    best = 0.0
    for theta_a1 in np.linspace(0, np.pi, n):
        a1 = np.cos(theta_a1) * z + np.sin(theta_a1) * x
        for theta_a2 in np.linspace(0, np.pi, n):
            a2 = np.cos(theta_a2) * z + np.sin(theta_a2) * x
            for theta_b1 in np.linspace(0, np.pi, n):
                b1 = np.cos(theta_b1) * z + np.sin(theta_b1) * x
                for theta_b2 in np.linspace(0, np.pi, n):
                    b2 = np.cos(theta_b2) * z + np.sin(theta_b2) * x
                    s = abs(
                        _chsh_expectation(rho, np.kron(a1, b1))
                        + _chsh_expectation(rho, np.kron(a1, b2))
                        + _chsh_expectation(rho, np.kron(a2, b1))
                        - _chsh_expectation(rho, np.kron(a2, b2))
                    )
                    if s > best:
                        best = s
    return float(best)


def run_chsh_checks(*, tol: float = 1e-12, angle_grid_n: int = 10) -> dict[str, object]:
    """Run the five Bell/CHSH checks; return measurements and PASS/FAIL flags."""
    chsh_phi = chsh_for_tbit(0)
    chsh_psi = chsh_for_tbit(1)
    t6 = 0b101010
    graph_chsh = [chsh_graph_state_pair(t6, k) for k in range(6)]
    max_angle = max_chsh_angle_grid(angle_grid_n)

    checks = {
        "phi_plus_saturates": abs(chsh_phi - TSIRELSON) < tol and chsh_phi > CLASSICAL,
        "psi_plus_saturates": abs(chsh_psi - TSIRELSON) < tol and chsh_psi > CLASSICAL,
        "graph_state_pairs_saturate": all(abs(s - TSIRELSON) < tol for s in graph_chsh),
        "angle_grid_le_tsirelson": max_angle <= TSIRELSON + 1e-10,
    }
    checks["all_pass"] = all(checks.values())

    return {
        "tsirelson": float(TSIRELSON),
        "classical_bound": CLASSICAL,
        "chsh_phi_plus": chsh_phi,
        "chsh_psi_plus": chsh_psi,
        "chsh_phi_residual": chsh_phi - TSIRELSON,
        "chsh_psi_residual": chsh_psi - TSIRELSON,
        "graph_t6": t6,
        "graph_pair_chsh": graph_chsh,
        "graph_pair_residuals": [s - TSIRELSON for s in graph_chsh],
        "angle_grid_n": angle_grid_n,
        "angle_grid_max_chsh": max_angle,
        "checks": checks,
    }


def experiment_chsh_tsirelson() -> dict[str, object]:
    """Bell/CHSH saturation at the Tsirelson bound (Hilbert lift)."""
    return run_chsh_checks()


# ================================================================
# D2. Grothendieck CHSH integrality gap (bridge D -> I)
# ================================================================


def experiment_chsh_grothendieck_bridge() -> dict[str, object]:
    """K_G^R(2) as the CHSH integrality gap on the hQVM carrier.

    Both inputs are measured, not hard-coded:
      - Hilbert lift CHSH from run_chsh_checks() (Tsirelson bound).
      - Boolean CHSH from the complement-horizon ensemble (the conditioned
        ensemble that maximizes the Boolean CHSH at the classical bound).
    Ratio = Hilbert / Boolean = sqrt(2) = K_G^R(2). The uniform-Omega Boolean
    CHSH is 0 (correlators vanish), so it is not used in the ratio.
    """
    hilb = cast("dict[str, float]", run_chsh_checks())
    hilbert_chsh = hilb["tsirelson"]
    # Boolean CHSH on the complement-horizon ensemble (measured, not asserted).
    from hqvm_Cohomology_analysis_run import max_chsh_on_index_set

    comp_idx = [
        i
        for i, s in enumerate(OMEGA_STATES_4096)
        if (s >> 12) & LAYER_MASK_12 == ((s & LAYER_MASK_12) ^ LAYER_MASK_12)
    ]
    boolean_chsh = cast("float", max_chsh_on_index_set(comp_idx)["max_CHSH_Boolean"])
    ratio = hilbert_chsh / boolean_chsh
    kg_r_2 = math.sqrt(2.0)
    uniform_boolean_chsh = 0.0
    return {
        "Hilbert_CHSH_Tsirelson": hilbert_chsh,
        "Boolean_CHSH_classical": boolean_chsh,
        "uniform_Boolean_CHSH": uniform_boolean_chsh,
        "ratio_Hilb_over_Bool": ratio,
        "K_G^R(2)_exact": kg_r_2,
        "ratio_equals_KG_R_2": abs(ratio - kg_r_2) < 1e-12,
    }


# ================================================================
# E. Cut-norm gap on a kernel-native bipartition
# ================================================================


def _horizon_transition_kernel(pool: list[int]) -> np.ndarray:
    """Row-normalized eq-horizon -> complement-horizon transition kernel."""
    from gyroscopic.hQVM.constants import step_state_by_byte

    eq_states = [omega12_to_state24((u6, u6)) for u6 in range(64)]
    comp_index = {
        omega12_to_state24((u6, u6 ^ 0x3F)): j for j, u6 in enumerate(range(64))
    }
    T = np.zeros((64, 64))
    for i, s in enumerate(eq_states):
        for b in pool:
            nxt = step_state_by_byte(s, b)
            j = comp_index.get(nxt)
            if j is not None:
                T[i, j] += 1.0
    row_sums = T.sum(axis=1, keepdims=True)
    mask = row_sums[:, 0] > 0
    T[mask] /= row_sums[mask]
    return T


def _uv_signed_kernel_from_rest(pool1: list[int], pool2: list[int]) -> np.ndarray:
    """UxV signed bipartite Grothendieck probe from rest.

    M[u,v] = P_pool1(u,v) - P_pool2(u,v), where P_pool(u,v) is the
    fraction of bytes in the pool that carry REST to the (u,v) factor state.
    Difference of two transition kernels => sign-indefinite.
    """
    from gyroscopic.hQVM.constants import step_state_by_byte, GENE_MAC_REST

    uv_of = {}
    for s in OMEGA_STATES_4096:
        om = state24_to_omega12(s)
        uv_of[s] = (om.u6, om.v6)

    M1 = np.zeros((64, 64))
    M2 = np.zeros((64, 64))
    for b in pool1:
        u, v = uv_of[step_state_by_byte(GENE_MAC_REST, b)]
        M1[u, v] += 1.0
    for b in pool2:
        u, v = uv_of[step_state_by_byte(GENE_MAC_REST, b)]
        M2[u, v] += 1.0
    if pool1:
        M1 /= len(pool1)
    if pool2:
        M2 /= len(pool2)
    return M1 - M2


def experiment_cut_horizon() -> dict[str, object]:
    """Horizon bipartition experiments.

    (1) Transition census: nonnegative count matrix T (sparse; not a Grothendieck
        instance by itself).
    (2) UxV signed kernel from rest (pools w=1 vs w=5): sign-indefinite
        Grothendieck probe on the product Omega = U x V.
    """
    from gyroscopic.hQVM.constants import step_state_by_byte

    eq_states = [omega12_to_state24((u6, u6)) for u6 in range(64)]
    comp_states = [omega12_to_state24((u6, u6 ^ 0x3F)) for u6 in range(64)]
    comp_index = {s: j for j, s in enumerate(comp_states)}

    T = np.zeros((64, 64))
    for i, s in enumerate(eq_states):
        for b in range(256):
            nxt = step_state_by_byte(s, b)
            j = comp_index.get(nxt)
            if j is not None:
                T[i, j] += 1.0

    s0 = eq_states[0]
    shell0 = state24_to_omega12(s0).shell
    flip_bytes = [
        b
        for b in range(256)
        if state24_to_omega12(step_state_by_byte(s0, b)).shell == 6
    ]
    expected_sum = 64 * len(flip_bytes)
    trans_sum = float(np.sum(T))

    census_stats = _matrix_stats(T)

    pool1 = [b for b in range(256) if Q_WEIGHT_BY_BYTE[b] == 1]
    pool5 = [b for b in range(256) if Q_WEIGHT_BY_BYTE[b] == 5]
    rest_signed = _uv_signed_kernel_from_rest(pool1, pool5)
    rest_entry = _structured_grothendieck_entry(
        rest_signed,
        bool_cut_norm_local,
        bool_method="bool_cut_norm_local (approx, valid lower bound)",
        k=32,
        restarts=40,
        iters=50,
    )

    krivine = math.pi / (2.0 * math.log(1.0 + math.sqrt(2.0)))
    pi2 = math.pi / 2.0
    return {
        "n_vertices_per_side": 64,
        "shell_0": shell0,
        "shell0_to_shell6_bytes": len(flip_bytes),
        "flip_byte_qweights": sorted({Q_WEIGHT_BY_BYTE[b] for b in flip_bytes}),
        "expected_transition_sum": float(expected_sum),
        "transition_sum": trans_sum,
        "transition_sum_matches": abs(trans_sum - expected_sum) < 1e-9,
        "census": {
            **census_stats,
            "bool_cut_norm": float(bool_cut_norm_local(T)),
            "spectral_norm": spectral_norm(T),
            "role": "nonnegative transition census (not Grothendieck probe)",
        },
        "signed_uv_from_rest": {
            "pool_w1": 1,
            "pool_w5": 5,
            **rest_entry,
        },
        "K_G^R(4)=pi/2": pi2,
        "Krivine": krivine,
        "note": (
            "census T is sparse counts; signed_uv_from_rest is the "
            "Grothendieck probe (sign-indefinite UxV kernel difference "
            "from rest). Horizon-to-horizon pool differences are zero by "
            "depth-2 typicality, so no eq->comp signed block exists."
        ),
    }


# ================================================================
# F. Depth-4 / K_G^R(4) = pi/2 gap candidates
# ================================================================


def experiment_depth4_gaps() -> dict[str, float]:
    """Collect stable numerical relations between CGM geometric invariants and
    the Grothendieck d=4 constant pi/2.

    Gate F eigenspaces: F preserves shell (chirality fixed), so its +/-1
    spaces are the within-shell shadow pairs |s> +/- |F(s)>. Because F is a
    fixed-point-free involution, it splits Omega into 2048 two-cycles, hence
    dim(+1) = dim(-1) = 2048 exactly. This is NOT the even/odd shell split
    (that equality to 2048 is only a binomial-symmetry coincidence).
    """
    pi2 = math.pi / 2.0
    krivine = math.pi / (2.0 * math.log(1.0 + math.sqrt(2.0)))
    phi_SU2 = 2.0 * math.acos((1.0 + 2.0 * math.sqrt(2.0)) / 4.0)

    # F +/-1 eigenspace dimensions from the fixed-point-free involution split.
    f_two_cycles = 0
    seen = set()
    for s in OMEGA_STATES_4096:
        fs = apply_gate(s, "F")
        if fs != s and fs not in seen and apply_gate(fs, "F") == s:
            f_two_cycles += 1
            seen.add(s)
    f_dim = f_two_cycles  # each two-cycle contributes one +1 and one -1 vector

    g1 = phi_SU2 / (3.0 * DELTA_BU) - 1.0
    g2 = APERTURE_GAP
    g3 = pi2 / krivine
    g4 = pi2 - krivine

    return {
        "K_G_R(4)=pi/2": pi2,
        "Krivine": krivine,
        "phi_SU2": phi_SU2,
        "gate_F_two_cycles": f_two_cycles,
        "gate_F_dim_plus1": f_dim,
        "gate_F_dim_minus1": f_dim,
        "gate_F_balanced_2048": (f_dim == 2048),
        "g1_holonomy_residual": g1,
        "g2_aperture_gap": g2,
        "g3_pi2_over_Krivine": g3,
        "g4_pi2_minus_Krivine": g4,
    }


# ================================================================
# G. Shannon triad on the same run
# ================================================================


def experiment_shannon_triad() -> dict[str, float]:
    """Exact finite Shannon faces on the kernel.

    1. Single-step image count from rest: expect 128 unique / 256 bytes
       (SO(3)/SU(2) 2-to-1 projection; 1-bit equivocation).
    2. Depth-2 occupancy from rest: expect full 4096 uniform (entropy 12).
    3. WHT Plancherel identity on a test function over GF(2)^6.
    """
    g = Gyroscopic()
    images = set()
    for b in range(256):
        g.reset()
        g.step_byte(b)
        images.add(g.state24)
    n_images = len(images)

    # depth-2 occupancy from rest (future_cone cached uniform)
    from gyroscopic.hQVM.sdk import future_cone_measure

    fc2 = future_cone_measure(GENE_MAC_REST, 2)
    n_distinct_d2 = len(fc2.state_counts)
    entropy_d2 = fc2.entropy_bits

    # WHT plancherel: ||f||^2 == ||WHT f||^2 for a sample function.
    # The kernel WHT is scaled by 1/8, so W^T W = I and ||W f||^2 = ||f||^2 exactly.
    W = walsh_hadamard64()
    f = np.array([rng.gauss(0.0, 1.0) for _ in range(64)])
    wht_f = W @ f
    plancherel_err = float(np.sum(f * f) - np.sum(wht_f * wht_f))

    return {
        "single_step_images": n_images,
        "expected_128": 128,
        "single_step_equivocation_bits": math.log2(256) - math.log2(n_images),
        "depth2_distinct_states": n_distinct_d2,
        "expected_4096": 4096,
        "depth2_entropy_bits": entropy_d2,
        "expected_entropy_12": 12.0,
        "wht_plancherel_err": plancherel_err,
    }


# ================================================================
# H. Square-root product spot-check
# ================================================================


def _bfs_reached_indices(
    eng,
    allowed_bytes: list[int],
    *,
    max_depth: int | None = None,
) -> list[int]:
    """Return indices of states reachable from rest under allowed_bytes."""
    from gyroscopic.hQVM.family import enumerate_bytes

    allowed = set(int(b) for b in allowed_bytes)
    byte_idx = [i for i, b in enumerate(enumerate_bytes(eng.d)) if b in allowed]
    if not byte_idx:
        return [eng.start_idx]
    depth_limit = 2 * eng.d + 6 if max_depth is None else int(max_depth)
    visited = bytearray(eng.n_omega)
    q: list[int] = [eng.start_idx]
    visited[eng.start_idx] = 1
    head = 0
    depth = 0
    next_end = 1
    while head < len(q):
        if head >= next_end:
            depth += 1
            if depth > depth_limit:
                break
            next_end = len(q)
        i = q[head]
        head += 1
        row = eng.transitions[i]
        for bi in byte_idx:
            j = row[bi]
            if not visited[j]:
                visited[j] = 1
                q.append(j)
    return [i for i, v in enumerate(visited) if v]


def _rectangularity_check(eng, reached_indices: list[int]) -> dict[str, object]:
    """Product/rectangularity stats on a reached index set."""
    idx_to_uv = {i: uv for uv, i in eng.uv_to_idx.items()}
    U: set[int] = set()
    V: set[int] = set()
    for i in reached_indices:
        u, v = idx_to_uv[i]
        U.add(u)
        V.add(v)
    n = len(reached_indices)
    product = len(U) * len(V)
    rect = n / product if product else 0.0
    return {
        "|U|": len(U),
        "|V|": len(V),
        "reach": n,
        "product": product,
        "rect": rect,
        "rect_is_product": n == product,
    }


def experiment_square_root_spotcheck() -> dict[str, object]:
    """Spot-check the square-root cluster theorem |Reach| = 2^{2r} on several
    alphabets over the d=6 kernel.

    Full alphabet (trivially fiber-complete; rank 6 => Reach 4096) plus
    fiber-complete partial alphabets that expose non-trivial structure:
      - even-weight q-span: expect rank 5, Reach 1024 (parity obstruction)
      - a single q-class taken fiber-complete (4 bytes): small cluster
    Uses the canonical family module build_hqvm_d, gf2_rank, bfs_reach.
    """
    from gyroscopic.hQVM import family as fam

    d = 6
    eng = fam.build_hqvm_d(d)
    results = {}

    def _case(name: str, byte_list: list[int], note: str, **extra) -> None:
        reach, spans, giant, full = fam.bfs_reach(eng, byte_list)
        rank = fam.gf2_rank([eng.q_by_byte[b] for b in byte_list], d)
        reached_idx = _bfs_reached_indices(eng, byte_list)
        rect = _rectangularity_check(eng, reached_idx)
        pred = fam.predicted_cluster_size(rank)
        results[name] = {
            "n_allow": len(byte_list),
            "rank": rank,
            "reach": reach,
            "predicted_2_2r": pred,
            "reach_ok": reach == pred,
            "E_full": full,
            "rect": rect,
            "rect_is_product": rect["rect_is_product"],
            "note": note,
            **extra,
        }

    full_bytes = list(range(1 << (d + 2)))
    _case(
        "full_alphabet",
        full_bytes,
        f"|Reach(rank=6)|={fam.predicted_cluster_size(6)}",
    )

    even_bytes = [b for b in full_bytes if eng.q_weight[b] % 2 == 0]
    _case(
        "even_weight_qspan",
        even_bytes,
        "expect rank=5, Reach=1024 (parity obstruction)",
    )

    q0 = 63  # only rank-1 single q-class with |Reach|=4 from rest
    qclass_bytes = list(eng.bytes_by_q[q0])
    _case(
        "single_qclass_fibercomplete",
        qclass_bytes,
        f"one q-class q={q0} (4 bytes); rank=1 => Reach=4",
        q_class=q0,
    )

    return results


# ================================================================
# main: dispatch and print
# ================================================================


def _print_constants(res: dict) -> None:
    print("\n" + "=" * 5)
    print("A. CONSTANT AND IDENTITY AUDIT")
    print("=" * 5)
    print("\nValues:")
    for k, v in res["values"].items():
        print(f"  {k:<40s} = {v!r}")
    print("\nIdentity checks (residual, tol 1e-12):")
    all_ok = True
    for k, v in res["checks"].items():
        ok = abs(v) < 1e-12
        all_ok = all_ok and ok
        print(f"  {k:<40s} = {v:+.3e}  {'PASS' if ok else 'FAIL'}")
    print(f"\n  ALL_IDENTITIES: {'PASS' if all_ok else 'FAIL'}")


def _print_arcsin(res: dict) -> None:
    print("\n" + "=" * 5)
    print("B. ARCSIN / HYPERPLANE-ROUNDING LAW ON CHIRALITY REGISTER")
    print("=" * 5)
    print(f"  dim=6 basis pairs tested: {res['n_pairs_basis']}")
    print(f"  n_gauss per pair:         {res['n_gauss']}")
    print(f"  MC vs analytic max err:   {res['mc_vs_analytic_max_err']:.3e}")
    print(f"  MC vs analytic mean err:  {res['mc_vs_analytic_mean_err']:.3e}")
    print(f"  fitted coefficient c:     {res['fitted_coefficient_c']:.6f}")
    print(f"  target 2/pi:              {res['target_2_over_pi']:.6f}")
    print(f"  c relative error:         {res['c_rel_err']:+.4%}")
    c = res["fitted_coefficient_c"]
    ok = c == c and abs(c - res["target_2_over_pi"]) / res["target_2_over_pi"] < 0.05
    print(f"  COEFFICIENT_WITHIN_5PCT:  {'PASS' if ok else 'FAIL'}")


def _print_arcsin_kernel(res: dict) -> None:
    print("\n" + "=" * 5)
    print("B2. ARCSIN LAW ON KERNEL WALSH OBSERVABLE VECTORS")
    print("=" * 5)
    print(f"  ensemble:               {res['ensemble']}")
    print(f"  n_feature_vectors:       {res['n_feature_vectors']}")
    print(f"  n_pairs_tested:          {res['n_pairs_tested']}")
    print(f"  MC vs analytic max err:  {res['mc_vs_analytic_max_err_kernel']:.3e}")
    print(f"  law_holds_on_kernel:     {res['arcsin_law_holds_on_kernel_observables']}")


def _print_finite_groth(res: dict) -> None:
    print("\n" + "=" * 5)
    print("C. FINITE GROTHENDIECK GAP (n={})".format(res["n"]))
    print("=" * 5)
    print(f"  trials:                   {res['n_trials']}")
    print(
        f"  bool mean/min/max:          {res['bool_mean']:.4f} / {res['bool_min']:.4f} / {res['bool_max']:.4f}"
    )
    print(
        f"  hilb_lb mean/min/max:       {res['hilb_mean']:.4f} / {res['hilb_min']:.4f} / {res['hilb_max']:.4f}"
    )
    print(
        f"  ratio hilb/bool mean/min/max: {res['ratio_mean']:.4f} / {res['ratio_min']:.4f} / {res['ratio_max']:.4f}"
    )
    print(
        f"  ratios below 1 (tol {res['ratio_tol']}): {res['ratios_below_1_tol']}/{res['n_trials']}"
    )
    print(f"  K_G^R(2)=sqrt2:     {res['K_G^R(2)=sqrt2']:.4f}")
    print(f"  K_G^R(4)=pi/2:      {res['K_G^R(4)=pi/2']:.4f}")
    print(f"  Krivine:            {res['Krivine_bound']:.4f}")
    print(f"  note: {res['note']}")
    print("\n  Structured kernel matrices:")
    for label, d in res["structured"].items():
        print(f"    {label}:")
        print(
            f"      max_abs={d['max_abs']:.6f} fro={d['fro_norm']:.4f} "
            f"row_sum=[{d['row_sum_min']:.3f},{d['row_sum_max']:.3f}]"
        )
        print(
            f"      bool_lb={d['bool_lb']:.4f} hilb_lb={d['hilb_lb']:.4f} "
            f"ratio={d['ratio_hilb_over_bool']:.4f} ratio>=1={d['ratio_ge_1']}"
        )


def _print_chsh(res: dict) -> None:
    print("\n" + "=" * 5)
    print("D. CHSH / TSIRELSON (HILBERT LIFT)")
    print("=" * 5)
    print(f"  Tsirelson bound:     {res['tsirelson']:.12f}")
    print(f"  classical bound:     {res['classical_bound']:.4f}")
    print(
        f"  CHSH |Phi+>:          {res['chsh_phi_plus']:.12f}  residual {res['chsh_phi_residual']:+.3e}"
    )
    print(
        f"  CHSH |Psi+>:          {res['chsh_psi_plus']:.12f}  residual {res['chsh_psi_residual']:+.3e}"
    )
    print(f"  graph t6={res['graph_t6']} pair CHSH:")
    for k, (s, dr) in enumerate(
        zip(res["graph_pair_chsh"], res["graph_pair_residuals"])
    ):
        print(f"    pair {k}: {s:.12f}  residual {dr:+.3e}")
    print(
        f"  angle grid n={res['angle_grid_n']} max CHSH: {res['angle_grid_max_chsh']:.12f}"
    )
    print("  checks:")
    for name, ok in res["checks"].items():
        if name == "all_pass":
            continue
        print(f"    {name:<28s} {'PASS' if ok else 'FAIL'}")
    print(f"  ALL_CHSH: {'PASS' if res['checks']['all_pass'] else 'FAIL'}")


def _print_chsh_bridge(res: dict) -> None:
    print("\n" + "=" * 5)
    print("D2. GROTHENDIECK CHSH INTEGRALITY GAP (D -> I)")
    print("=" * 5)
    print(f"  Hilbert CHSH (D, Tsirelson lift): {res['Hilbert_CHSH_Tsirelson']:.12f}")
    print(f"  Boolean CHSH (I, horizon ensembles): {res['Boolean_CHSH_classical']:.4f}")
    print(
        f"  uniform Boolean CHSH (Omega):      {res['uniform_Boolean_CHSH']:.4f}  (not used in ratio)"
    )
    print(f"  ratio Hilb/Bool:       {res['ratio_Hilb_over_Bool']:.12f}")
    print(f"  K_G^R(2) = sqrt2:    {res['K_G^R(2)_exact']:.12f}")
    print(f"  ratio == K_G^R(2):    {res['ratio_equals_KG_R_2']}")


def _print_cut(res: dict) -> None:
    print("\n" + "=" * 5)
    print("E. HORIZON BIPARTITION (CENSUS + SIGNED UxV KERNEL)")
    print("=" * 5)
    print(f"  vertices per side:    {res['n_vertices_per_side']}")
    print(
        f"  shell0->shell6 bytes: {res['shell0_to_shell6_bytes']} (qw {res['flip_byte_qweights']})"
    )
    print(f"  expected sum:         {res['expected_transition_sum']:.1f}")
    print(
        f"  transition sum:       {res['transition_sum']:.1f}  (matches: {res['transition_sum_matches']})"
    )
    cen = res["census"]
    print("  census (nonnegative counts):")
    print(
        f"    max_abs={cen['max_abs']:.4f} fro={cen['fro_norm']:.4f} "
        f"row_sum=[{cen['row_sum_min']:.3f},{cen['row_sum_max']:.3f}]"
    )
    print(f"    bool_cut={cen['bool_cut_norm']:.4f} spec={cen['spectral_norm']:.4f}")
    print(f"    role: {cen['role']}")
    sv = res["signed_uv_from_rest"]
    print(f"  signed_uv_from_rest (pools w={sv['pool_w1']} vs w={sv['pool_w5']}):")
    print(
        f"    max_abs={sv['max_abs']:.6f} fro={sv['fro_norm']:.4f} is_nonzero={sv['fro_norm'] > 1e-12}"
    )
    print(
        f"    bool_lb={sv['bool_lb']:.4f} hilb_lb={sv['hilb_lb']:.4f} "
        f"ratio={sv['ratio_hilb_over_bool']:.4f} ratio>=1={sv['ratio_ge_1']}"
    )
    print(f"  K_G^R(4)=pi/2:  {res['K_G^R(4)=pi/2']:.4f}")
    print(f"  Krivine:        {res['Krivine']:.4f}")
    print(f"  note: {res['note']}")


def _print_depth4(res: dict) -> None:
    print("\n" + "=" * 5)
    print("F. DEPTH-4 / K_G^R(4)=pi/2 GAP CANDIDATES")
    print("=" * 5)
    print(f"  K_G^R(4) = pi/2:      {res['K_G_R(4)=pi/2']:.6f}")
    print(f"  Krivine:              {res['Krivine']:.6f}")
    print(f"  phi_SU2:              {res['phi_SU2']:.6f}")
    print(
        f"  Gate F split:         +1={res['gate_F_dim_plus1']} -1={res['gate_F_dim_minus1']} balanced={res['gate_F_balanced_2048']}"
    )
    print(f"  g1 (holonomy resid):  {res['g1_holonomy_residual']:.6f}")
    print(f"  g2 (aperture gap):    {res['g2_aperture_gap']:.6f}")
    print(f"  g3 (pi/2/Krivine):    {res['g3_pi2_over_Krivine']:.6f}")
    print(f"  g4 (pi/2-Krivine):    {res['g4_pi2_minus_Krivine']:.6f}")


def _print_shannon(res: dict) -> None:
    print("\n" + "=" * 5)
    print("G. SHANNON TRIAD")
    print("=" * 5)
    print(
        f"  single-step images:        {res['single_step_images']} (expect {res['expected_128']})"
    )
    print(
        f"  single-step equivocation:   {res['single_step_equivocation_bits']:.4f} bit"
    )
    print(
        f"  depth-2 distinct states:    {res['depth2_distinct_states']} (expect {res['expected_4096']})"
    )
    print(
        f"  depth-2 entropy:            {res['depth2_entropy_bits']:.6f} bit (expect {res['expected_entropy_12']})"
    )
    print(f"  WHT Plancherel err:         {res['wht_plancherel_err']:.3e}")
    ok = (
        res["single_step_images"] == 128
        and res["depth2_distinct_states"] == 4096
        and abs(res["depth2_entropy_bits"] - 12.0) < 1e-9
    )
    print(f"  SHANNON_TRIAD: {'PASS' if ok else 'FAIL'}")


def _print_square_root(res: dict) -> None:
    print("\n" + "=" * 5)
    print("H. SQUARE-ROOT PRODUCT SPOT-CHECK")
    print("=" * 5)
    for d, info in res.items():
        rect = info["rect"]
        print(
            f"  {d}: n_allow={info['n_allow']} rank={info['rank']} "
            f"reach={info['reach']} predicted={info['predicted_2_2r']} "
            f"reach_ok={info['reach_ok']} E_full={info['E_full']}"
        )
        print(
            f"       |U|={rect['|U|']} |V|={rect['|V|']} product={rect['product']} "
            f"rect={rect['rect']:.4f} rect_is_product={info['rect_is_product']}"
        )
        if "q_class" in info:
            print(f"       q_class={info['q_class']}")
        print(f"       {info['note']}")


def main() -> None:
    print("hQVM Cohomology analysis -- experiment library")
    print("=" * 5)
    print(f"seed={SEED}")

    _print_constants(experiment_constants())
    _print_arcsin(experiment_arcsin_rounding())
    _print_arcsin_kernel(experiment_arcsin_rounding_kernel())
    _print_finite_groth(experiment_finite_grothendieck())
    _print_chsh(experiment_chsh_tsirelson())
    _print_chsh_bridge(experiment_chsh_grothendieck_bridge())
    _print_cut(experiment_cut_horizon())
    _print_depth4(experiment_depth4_gaps())
    _print_shannon(experiment_shannon_triad())
    _print_square_root(experiment_square_root_spotcheck())

    print("\n" + "=" * 5)
    print("SUMMARY")
    print("=" * 5)
    print("  A constant audit:        see PASS/FAIL above")
    print("  B arcsin coefficient:    see coefficient vs 2/pi")
    print("  C Grothendieck gap:        see ratio hilb/bool vs K_G(d)")
    print("  D CHSH/Tsirelson:        see ALL_CHSH")
    print("  D2 Grothendieck CHSH gap: see ratio == K_G^R(2)")
    print("  E horizon bipartition:   see signed_uv_* ratio")
    print("  F depth-4 gaps:          candidate relations reported")
    print("  G Shannon triad:         see SHANNON_TRIAD")
    print("  H square-root:           see reach_ok and rect_is_product")


if __name__ == "__main__":
    main()
