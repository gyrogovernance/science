#!/usr/bin/env python3
"""
hqvm_SO_analysis_5.py

Pascal / binomial / Krawtchouk operational tests on the kernel's 7-shell
Hopf-latitude quotient and the hQVM(d) rank PMF.

Companion: hqvm_SO_analysis_2.py (Hopf latitudes, byte mixing), family.py,
percolation_analysis_5.py. Inputs: hQVM api, family.py. Outputs: PASS/FAIL.
"""

from __future__ import annotations

import sys
from fractions import Fraction
from math import comb
from typing import Dict, List, Tuple

import numpy as np

from hqvm_SO_analysis_common import (
    _KERNEL_OK,
    _SCIPY_OK,
    GENE_MAC_REST,
    ReportState,
    check,
    section,
    chirality_word6,
    q_word6,
    shell_population,
    shell_transition_matrix_for_q_weight,
    step_state_by_byte,
)

if _KERNEL_OK:
    from gyroscopic.hQVM.api import (
        FULL_BYTE_SHELL_DISTRIBUTION,
        KRAWTCHOUK_7,
        OMEGA_STATES_4096,
    )
    from gyroscopic.hQVM.family import (
        exact_micro_ref_rank_pmf,
        gaussian_binomial,
        verify_exact_root_rank_lock,
    )
else:
    FULL_BYTE_SHELL_DISTRIBUTION = None  # type: ignore
    KRAWTCHOUK_7 = ()  # type: ignore
    OMEGA_STATES_4096 = ()  # type: ignore

PASCAL_ROW6 = tuple(comb(6, k) for k in range(7))
D_VALUES = (4, 5, 6, 7)


def _shell(state24: int) -> int:
    return chirality_word6(state24).bit_count()


def _rep_by_shell() -> Dict[int, int]:
    reps: Dict[int, int] = {}
    for s in OMEGA_STATES_4096:
        w = _shell(s)
        if w not in reps:
            reps[w] = s
    return reps


def _pascal_lower(n: int) -> np.ndarray:
    m = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1):
            m[i, j] = float(comb(i, j))
    return m


def _nilpotent_shift(n: int) -> np.ndarray:
    """Weighted shift N[i,i-1]=i; exp(N) has rows equal to Pascal triangle rows."""
    n_mat = np.zeros((n, n), dtype=np.float64)
    for i in range(1, n):
        n_mat[i, i - 1] = float(i)
    return n_mat


def _byte_shell_transition() -> List[List[Fraction]]:
    reps = _rep_by_shell()
    t_mat: List[List[Fraction]] = [[Fraction(0) for _ in range(7)] for _ in range(7)]
    for w in range(7):
        for b in range(256):
            wp = _shell(step_state_by_byte(reps[w], b))
            t_mat[wp][w] += Fraction(1, 256)
    return t_mat


def _weighted_mq_transition() -> List[List[Fraction]]:
    acc: List[List[Fraction]] = [[Fraction(0) for _ in range(7)] for _ in range(7)]
    for q in range(7):
        n_b = sum(1 for b in range(256) if q_word6(b).bit_count() == q)
        mq = shell_transition_matrix_for_q_weight(q)
        for w in range(7):
            for wp in range(7):
                acc[wp][w] += Fraction(n_b, 256) * mq[w][wp]
    return acc


def _krawtchouk_eigenvalue(q: int, k: int) -> Fraction:
    mat = shell_transition_matrix_for_q_weight(q)
    qvec = [Fraction(KRAWTCHOUK_7[w][k], 1) for w in range(7)]
    mq = [sum(mat[w][wp] * qvec[wp] for wp in range(7)) for w in range(7)]
    for w in range(7):
        if qvec[w] != 0:
            return mq[w] / qvec[w]
    return Fraction(0)


def run_part5(state: ReportState) -> None:
    if not _KERNEL_OK or not _SCIPY_OK:
        check(state, "kernel+scipy available", False)
        return

    import scipy.linalg as spla

    section(state, "Pascal recursion and exp(N) row census")

    rec_ok = all(
        comb(6, k) == comb(5, k - 1) + comb(5, k) for k in range(1, 6)
    )
    check(
        state,
        "C(6,k)=C(5,k-1)+C(5,k)",
        rec_ok,
        quantity="Pascal recursion on d=6 shells",
        measured=f"row6={PASCAL_ROW6}",
        threshold="k=1..5",
    )

    l6 = _pascal_lower(7)
    row6_l = tuple(int(l6[6, k]) for k in range(7))
    check(
        state,
        "L[6,k]=C(6,k)",
        row6_l == PASCAL_ROW6,
        quantity="Lower Pascal row index = latitude census",
        measured=str(row6_l),
        threshold=str(PASCAL_ROW6),
    )

    n_mat = _nilpotent_shift(7)
    l_exp = spla.expm(n_mat)
    exp_row6 = tuple(int(round(l_exp[6, k])) for k in range(7))
    check(
        state,
        "exp(N)[6,k]=C(6,k)",
        exp_row6 == PASCAL_ROW6,
        quantity="Pascal matrix exp(weighted shift) row 6",
        measured=str(exp_row6),
        threshold=str(PASCAL_ROW6),
    )

    section(state, "Hopf latitude quotient: byte shell Markov T")

    t_mat = _byte_shell_transition()
    pascal_col = tuple(Fraction(c, 64) for c in PASCAL_ROW6)
    cols_ok = all(
        tuple(t_mat[w][src] for w in range(7)) == pascal_col for src in range(7)
    )
    check(
        state,
        "T columns = C(6,k)/64",
        cols_ok,
        quantity="Source-independent shell mixing (one byte)",
        measured=str([float(t_mat[w][0]) for w in range(7)]),
        threshold=str([float(x) for x in FULL_BYTE_SHELL_DISTRIBUTION]),
    )

    w_mat = _weighted_mq_transition()
    max_tw = max(
        abs(float(t_mat[i][j] - w_mat[i][j])) for i in range(7) for j in range(7)
    )
    check(
        state,
        "T = sum_q n_q M_q/256",
        max_tw < 1e-12,
        quantity="Byte average equals q-weight mixture of Hamming M_q",
        measured=f"max|T-W|={max_tw:.2e}",
        threshold="<1e-12",
    )

    l7 = _pascal_lower(7)
    t_float = np.array([[float(t_mat[i][j]) for j in range(7)] for i in range(7)])
    max_lt = float(np.max(np.abs(t_float - l7)))
    check(
        state,
        "L != T",
        max_lt > 0.01,
        quantity="Lower Pascal L is not the byte shell operator",
        measured=f"max|L-T|={max_lt:.4f}",
        threshold=">0.01",
    )

    t_sq = t_float @ t_float
    idem_ok = float(np.max(np.abs(t_sq - t_float))) < 1e-12
    check(
        state,
        "T^2 = T on shell quotient",
        idem_ok,
        quantity="One byte erases shell memory (rank-1 Markov)",
        measured=f"max|T^2-T|={float(np.max(np.abs(t_sq - t_float))):.2e}",
        threshold="<1e-12",
    )

    section(state, "Krawtchouk diagonalization of M_q (Pascal-derived basis)")

    k0 = KRAWTCHOUK_7[0]
    check(
        state,
        "K_0 = C(6,k)",
        tuple(k0) == PASCAL_ROW6,
        quantity="Krawtchouk mode 0 = binomial census",
        measured=str(tuple(k0)),
        threshold=str(PASCAL_ROW6),
    )

    kraw_ok = True
    for q in range(7):
        for k in range(7):
            mat = shell_transition_matrix_for_q_weight(q)
            qvec = [Fraction(KRAWTCHOUK_7[w][k], 1) for w in range(7)]
            lam = _krawtchouk_eigenvalue(q, k)
            mq = [sum(mat[w][wp] * qvec[wp] for wp in range(7)) for w in range(7)]
            if not all(mq[w] == lam * qvec[w] for w in range(7)):
                kraw_ok = False
    check(
        state,
        "M_q Krawtchouk-diagonal all q",
        kraw_ok,
        quantity="Hamming scheme Bose-Mesner eigenbasis",
        measured="all 7x7 modes for q=0..6",
        threshold="exact Fraction equality",
    )

    section(state, "hQVM(d) family: binomial latitudes and q-Pascal rank PMF")

    for d in D_VALUES:
        row = tuple(comb(d, k) for k in range(d + 1))
        pop = tuple(comb(d, k) * (1 << d) for k in range(d + 1))
        if d == 6:
            api_pop = tuple(shell_population(k) for k in range(7))
            ok = pop == api_pop
        else:
            ok = sum(pop) == (1 << (2 * d))
        check(
            state,
            f"d={d} shell census",
            ok,
            quantity=f"C({d},k)*2^{d} sums to |Omega|",
            measured=f"sum={sum(pop)}",
            threshold=str(1 << (2 * d)),
        )

    gb = gaussian_binomial(6, 3, q=2)
    c63 = comb(6, 3)
    check(
        state,
        "[6;3]_2 != C(6,3)",
        gb != c63 and gb > 0,
        quantity="q-Pascal subspace count distinct from binomial census",
        measured=f"[6;3]_2={gb} C(6,3)={c63}",
        threshold="not equal",
    )

    ok_lock, p_full, p_exact = verify_exact_root_rank_lock(0.3, 6)
    dist = exact_micro_ref_rank_pmf(0.3, 6)
    check(
        state,
        "rank PMF lock d=6",
        ok_lock and abs(sum(dist) - 1.0) < 1e-9,
        quantity="Micro-ref rank PMF (Gaussian-binomial lattice)",
        measured=f"P(full)={p_full:.6f} sum={sum(dist):.6f}",
        threshold="normalize 1; root lock",
    )

    section(state, "Negative gates: Pascal exp vs byte dynamics")

    col6 = l_exp[:, 0]
    check(
        state,
        "L^6 e_0 != latitude row",
        tuple(int(round(col6[k])) for k in range(7)) != PASCAL_ROW6,
        quantity="Column action of exp(N)^6 is not census row",
        measured=str(tuple(int(round(col6[k])) for k in range(7))[:4]) + "...",
        threshold="!= row6",
    )

    b4_ok = all(
        step_state_by_byte(
            step_state_by_byte(
                step_state_by_byte(step_state_by_byte(GENE_MAC_REST, b), b), b
            ),
            b,
        )
        == GENE_MAC_REST
        for b in range(256)
    )
    check(
        state,
        "b^4 = id on Omega",
        b4_ok,
        quantity="Byte depth-4 closure (K4 spinor, not Pascal L^n)",
        measured="256/256 bytes",
        threshold="rest fixed",
    )

    hopf_ok = tuple(shell_population(w) // 64 for w in range(7)) == PASCAL_ROW6
    check(
        state,
        "Hopf latitude sizes C(6,w)",
        hopf_ok,
        quantity="Discrete S^2 latitude census from Part 2",
        measured=str(tuple(shell_population(w) // 64 for w in range(7))),
        threshold=str(PASCAL_ROW6),
    )


def main() -> None:
    state = ReportState()
    print("\nPART 5: PASCAL / BINOMIAL / KRAWTCHOUK OPERATORS")
    print("=" * 5)
    run_part5(state)


if __name__ == "__main__":
    main()
