#!/usr/bin/env python3
"""
hqvm_cgm_allometry_3.py

Kernel-side composition invariants under fiber-complete alphabet union.
Uses gyroscopic.hQVM.family (real engine). Does not re-prove SRCT
(see hqvm_percolation_analysis_4.py / allometry_1 square_root_live_gate).
Does not map kernel states to biological units.

Companion: hqvm_cgm_allometry_1.py, hqvm_cgm_allometry_run.py.
Theory: hqvm_cgm_allometry_notes.md; QuBEC quotient/product: hQVM_QuBEC_Theory.md.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_REPO = Path(__file__).resolve().parents[1]
_EXP = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from gyroscopic.hQVM.family import (
    HqvmD,
    bfs_reach,
    build_hqvm_d,
    fiber_complete,
    gf2_rank,
    predicted_cluster_size,
)

from hqvm_cgm_allometry_1 import A_BULK, A_SR, A_SURFACE, EXACT_TOL, N_DOF

_ENGINES: Dict[int, HqvmD] = {}


def _engine(d: int) -> HqvmD:
    if d not in _ENGINES:
        _ENGINES[d] = build_hqvm_d(d)
    return _ENGINES[d]


def _popcount(x: int) -> int:
    return int(x).bit_count()


def fiber_alphabet_from_qs(eng: HqvmD, qs: Sequence[int]) -> List[int]:
    out: List[int] = []
    for q in qs:
        out.extend(eng.bytes_by_q[int(q)])
    return out


def alphabet_q_weight_exact(eng: HqvmD, w: int) -> List[int]:
    return [b for b in range(eng.n_bytes) if eng.q_weight[b] == w]


def alphabet_metrics(eng: HqvmD, alphabet: Sequence[int]) -> Tuple[int, int, int, float, bool, bool]:
    """Return (rank, reach, root, a_SR, fiber_ok, srct_ok)."""
    if not alphabet:
        return 0, 1, 1, float("nan"), True, True
    qs = [eng.q_by_byte[b] for b in alphabet]
    r = gf2_rank(qs, eng.d)
    reach, _, _, _ = bfs_reach(eng, alphabet)
    pred = predicted_cluster_size(r)
    root = 1 << r if r >= 1 else 1
    if reach <= 1 or root <= 0:
        a = float("nan")
    else:
        a = math.log(root) / math.log(reach)
    fc = fiber_complete(alphabet, eng)
    srct = reach == pred
    return r, reach, root, a, fc, srct


@dataclass(frozen=True)
class UnionRow:
    d: int
    label: str
    n_A: int
    n_B: int
    n_U: int
    r_A: int
    r_B: int
    r_U: int
    reach_A: int
    reach_B: int
    reach_U: int
    root_A: int
    root_B: int
    root_U: int
    a_U: float
    fiber_A: bool
    fiber_B: bool
    fiber_U: bool
    srct_A: bool
    srct_B: bool
    srct_U: bool
    rank_mono: bool
    reach_mono: bool
    root_mono: bool
    a_is_half: bool


def _union_row(
    eng: HqvmD,
    label: str,
    A: Sequence[int],
    B: Sequence[int],
) -> UnionRow:
    U = sorted(set(A) | set(B))
    rA, reachA, rootA, _, fcA, srctA = alphabet_metrics(eng, A)
    rB, reachB, rootB, _, fcB, srctB = alphabet_metrics(eng, B)
    rU, reachU, rootU, aU, fcU, srctU = alphabet_metrics(eng, U)
    a_half = (
        (not math.isnan(aU))
        and reachU > 1
        and rU >= 1
        and abs(aU - A_SR) < EXACT_TOL
    )
    return UnionRow(
        d=eng.d,
        label=label,
        n_A=len(A),
        n_B=len(B),
        n_U=len(U),
        r_A=rA,
        r_B=rB,
        r_U=rU,
        reach_A=reachA,
        reach_B=reachB,
        reach_U=reachU,
        root_A=rootA,
        root_B=rootB,
        root_U=rootU,
        a_U=aU,
        fiber_A=fcA,
        fiber_B=fcB,
        fiber_U=fcU,
        srct_A=srctA,
        srct_B=srctB,
        srct_U=srctU,
        rank_mono=rU >= max(rA, rB),
        reach_mono=reachU >= max(reachA, reachB),
        root_mono=rootU >= max(rootA, rootB),
        a_is_half=a_half,
    )


def weight_shell_unions(d: int = N_DOF) -> List[UnionRow]:
    """Pairwise union of distinct exact-q-weight fiber alphabets."""
    eng = _engine(d)
    shells = {w: alphabet_q_weight_exact(eng, w) for w in range(d + 1)}
    rows: List[UnionRow] = []
    for w1 in range(d + 1):
        for w2 in range(w1 + 1, d + 1):
            A = shells[w1]
            B = shells[w2]
            if not A or not B:
                continue
            rows.append(_union_row(eng, f"w{w1}+w{w2}", A, B))
    return rows


def cumulative_weight_ladder(d: int = N_DOF) -> List[UnionRow]:
    """Successive union of exact-weight shells 0..k (composition ladder)."""
    eng = _engine(d)
    rows: List[UnionRow] = []
    acc: List[int] = []
    for w in range(d + 1):
        shell = alphabet_q_weight_exact(eng, w)
        if not shell:
            continue
        if not acc:
            # self-union baseline: A=shell, B=empty → treat as shell+empty skip
            acc = list(shell)
            continue
        rows.append(_union_row(eng, f"cum<=w{w}", acc, shell))
        acc = sorted(set(acc) | set(shell))
    return rows


def alphabet_even_q(eng: HqvmD) -> List[int]:
    return [b for b in range(eng.n_bytes) if _popcount(eng.q_by_byte[b]) % 2 == 0]


def alphabet_odd_q(eng: HqvmD) -> List[int]:
    return [b for b in range(eng.n_bytes) if _popcount(eng.q_by_byte[b]) % 2 == 1]


def parity_shell_unions(d: int = N_DOF) -> List[UnionRow]:
    """Union of even-q and odd-q fiber alphabets (percolation structured cases)."""
    eng = _engine(d)
    even = alphabet_even_q(eng)
    odd = alphabet_odd_q(eng)
    rows = [_union_row(eng, "even+odd", even, odd)]
    # even with each odd weight shell (structured FC pieces)
    for w in range(1, d + 1, 2):
        shell = alphabet_q_weight_exact(eng, w)
        if shell:
            rows.append(_union_row(eng, f"even+w{w}", even, shell))
    return rows


@dataclass(frozen=True)
class ScopeBoundaryRow:
    d: int
    label: str
    n: int
    rank: int
    reach: int
    pred: int
    fiber_ok: bool
    srct_ok: bool


def scope_boundary_single_q(d: int = N_DOF) -> List[ScopeBoundaryRow]:
    """Single-q fibers: fiber-complete but outside the structured SRCT testbed.

    Weight / fold / even-odd shells are the verified families (percolation_4).
    Arbitrary fiber-complete q-subsets are not claimed here.
    """
    eng = _engine(d)
    rows: List[ScopeBoundaryRow] = []
    for q in (1, 2, 4, (1 << d) - 1):
        A = fiber_alphabet_from_qs(eng, [q])
        r, reach, _, _, fc, srct = alphabet_metrics(eng, A)
        rows.append(
            ScopeBoundaryRow(
                d=d,
                label=f"q={q}",
                n=len(A),
                rank=r,
                reach=reach,
                pred=predicted_cluster_size(r),
                fiber_ok=fc,
                srct_ok=srct,
            )
        )
    return rows


@dataclass(frozen=True)
class ProductGeomRow:
    r1: int
    r2: int
    reach1: int
    reach2: int
    reach_prod: int
    reach_sum_rank: int
    equal: bool


def product_geometry_identity(
    ranks: Sequence[Tuple[int, int]] = (
        (1, 1),
        (2, 2),
        (3, 3),
        (1, 5),
        (2, 4),
        (3, 3),
        (6, 6),
    ),
) -> List[ProductGeomRow]:
    """Algebraic check: |Ω1|·|Ω2| = (2^{r1+r2})^2 when each factor is a product cluster.

    No second engine is built. This is the integer identity implied by product
    geometry under independent composition of two SRCT clusters.
    """
    rows: List[ProductGeomRow] = []
    for r1, r2 in ranks:
        reach1 = predicted_cluster_size(r1)
        reach2 = predicted_cluster_size(r2)
        reach_prod = reach1 * reach2
        reach_sum = predicted_cluster_size(r1 + r2)
        rows.append(
            ProductGeomRow(
                r1=r1,
                r2=r2,
                reach1=reach1,
                reach2=reach2,
                reach_prod=reach_prod,
                reach_sum_rank=reach_sum,
                equal=reach_prod == reach_sum,
            )
        )
    return rows


@dataclass(frozen=True)
class QuotientCensus:
    d: int
    n_bytes: int
    n_even_q_bytes: int
    n_horizon: int
    n_shells: int
    chain_ok: bool


def quotient_census(d: int = N_DOF) -> QuotientCensus:
    """Measured sizes for the documented QuBEC quotient chain at this d.

    At d=6 the chain is 256 → 128 → 64 → 7 (alphabet → even-q → horizon → shells).
    """
    eng = _engine(d)
    n_bytes = eng.n_bytes
    n_even = sum(1 for b in range(n_bytes) if _popcount(eng.q_by_byte[b]) % 2 == 0)
    n_horizon = 1 << d
    n_shells = d + 1
    if d == 6:
        chain_ok = (
            n_bytes == 256 and n_even == 128 and n_horizon == 64 and n_shells == 7
        )
    else:
        expect_bytes = 1 << (d + 2)
        expect_even = expect_bytes // 2
        chain_ok = (
            n_bytes == expect_bytes
            and n_even == expect_even
            and n_horizon == (1 << d)
            and n_shells == d + 1
        )
    return QuotientCensus(
        d=d,
        n_bytes=n_bytes,
        n_even_q_bytes=n_even,
        n_horizon=n_horizon,
        n_shells=n_shells,
        chain_ok=chain_ok,
    )


@dataclass(frozen=True)
class InterfaceRow:
    D: int
    a: float
    target: float
    ok: bool


def interface_rule_eval(
    cases: Sequence[Tuple[int, float]] = ((1, A_SR), (2, A_SURFACE), (3, A_BULK)),
) -> List[InterfaceRow]:
    """a(D)=D/(D+1) at the three organizational scales (arithmetic only)."""
    rows: List[InterfaceRow] = []
    for D, target in cases:
        a = D / float(D + 1)
        rows.append(InterfaceRow(D=D, a=a, target=target, ok=abs(a - target) < EXACT_TOL))
    return rows


def composition_pass(
    weight_rows: Optional[Sequence[UnionRow]] = None,
    cum_rows: Optional[Sequence[UnionRow]] = None,
    parity_rows: Optional[Sequence[UnionRow]] = None,
    prod_rows: Optional[Sequence[ProductGeomRow]] = None,
    census: Optional[QuotientCensus] = None,
    iface: Optional[Sequence[InterfaceRow]] = None,
    scope_rows: Optional[Sequence[ScopeBoundaryRow]] = None,
) -> Dict[str, bool]:
    if weight_rows is None:
        weight_rows = weight_shell_unions()
    if cum_rows is None:
        cum_rows = cumulative_weight_ladder()
    if parity_rows is None:
        parity_rows = parity_shell_unions()
    if prod_rows is None:
        prod_rows = product_geometry_identity()
    if census is None:
        census = quotient_census()
    if iface is None:
        iface = interface_rule_eval()
    if scope_rows is None:
        scope_rows = scope_boundary_single_q()

    all_unions = list(weight_rows) + list(cum_rows) + list(parity_rows)

    def _ok_union(r: UnionRow) -> bool:
        return (
            r.fiber_A
            and r.fiber_B
            and r.fiber_U
            and r.srct_A
            and r.srct_B
            and r.srct_U
            and r.rank_mono
            and r.reach_mono
            and r.root_mono
            and (r.r_U < 1 or r.a_is_half)
        )

    # q=all-ones often matches SRCT; low-weight single-q fibers do not.
    # Gate: at least one low-q fiber is fiber-complete yet not SRCT-sized.
    scope_detected = any(
        r.fiber_ok and (not r.srct_ok) and r.label.startswith("q=") and r.label != f"q={(1 << census.d) - 1}"
        for r in scope_rows
    )
    all_ones_ok = any(
        r.label == f"q={(1 << census.d) - 1}" and r.fiber_ok and r.srct_ok
        for r in scope_rows
    )

    return {
        "weight_shell_unions_ok": bool(weight_rows) and all(_ok_union(r) for r in weight_rows),
        "cumulative_ladder_ok": bool(cum_rows) and all(_ok_union(r) for r in cum_rows),
        "parity_shell_unions_ok": bool(parity_rows) and all(_ok_union(r) for r in parity_rows),
        "all_unions_fiber_complete": all(r.fiber_U for r in all_unions),
        "all_unions_srct": all(r.srct_U for r in all_unions),
        "product_geometry_identity": all(r.equal for r in prod_rows),
        "quotient_census_chain": bool(census.chain_ok),
        "interface_rule_a_D": all(r.ok for r in iface),
        "scope_single_q_outside_srct": scope_detected,
        "scope_all_ones_fiber_srct": all_ones_ok,
    }
