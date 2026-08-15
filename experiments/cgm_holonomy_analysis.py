#!/usr/bin/env python3
"""
CGM Holonomy Analysis

Foundational measurements of holonomy in the Common Governance Model.

Layers:
  1. Exact continuous algebra — thresholds, SU(2) commutator, TW calibration
  2. Declared continuous CGM realization — payload embedding, analytic BU holonomy,
     matrix realization, conjugacy, invariances, aperture structure
  3. Exact finite hQVM realization — fold statistics, K4/W2, continuous-finite bridge

Writes experiments/cgm_holonomy_analysis_results.txt.

Companion finding doc: docs/Findings/Analysis_Holonomy.md
"""

from __future__ import annotations

import io
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mpmath as mp
import numpy as np

_EXP = Path(__file__).resolve().parent
_REPO = _EXP.parent
RESULTS_PATH = _EXP / "cgm_holonomy_analysis_results.txt"

if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from functions.gyrovector_ops import GyroVectorSpace

try:
    from gyroscopic.hQVM.constants import (
        BU_APERTURE_GAP,
        BU_CLOSURE_RATIO,
        BU_HOLONOMY_ANGLE,
    )
except Exception:  # pragma: no cover
    BU_HOLONOMY_ANGLE = None  # type: ignore[assignment]
    BU_CLOSURE_RATIO = None  # type: ignore[assignment]
    BU_APERTURE_GAP = None  # type: ignore[assignment]

try:
    from hqvm_wavefunction_kernel import (
        build_kernel,
        decompose_byte,
        fold_disagreement,
        verify_k4_w2,
    )

    _FINITE_OK = True
    _FINITE_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover
    build_kernel = None  # type: ignore[assignment]
    decompose_byte = None  # type: ignore[assignment]
    fold_disagreement = None  # type: ignore[assignment]
    verify_k4_w2 = None  # type: ignore[assignment]
    _FINITE_OK = False
    _FINITE_IMPORT_ERROR = str(exc)

mp.mp.dps = 80

TOL_ANGLE = 1e-12
TOL_TW_SLOPE = 1e-3
TOL_TW_RESID = 1e-6
TOL_SO3 = 1e-10
TOL_MAP = 1e-9
TOL_MATRIX = 5e-8
TOL_ANGLE_INV = 1e-9


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------


class Tee:
    def __init__(self, *streams: Any) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def _section(title: str) -> None:
    print(title)
    print("=" * 5)


def _check(gates: list[tuple[str, bool]], label: str, ok: bool) -> None:
    print(f"  check  {label:56s} {'PASS' if ok else 'FAIL'}")
    gates.append((label, ok))


def mp_to_str(x: Any, digits: int = 16) -> str:
    s = mp.nstr(x, n=digits)
    return s if s is not None else str(x)


# ---------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class CGMThresholds:
    theta_cs: Any
    u_p: Any
    theta_una: Any
    theta_ona: Any
    m_a: Any
    q_g: Any

    @classmethod
    def make(cls) -> CGMThresholds:
        u_p = mp.mpf(1) / mp.sqrt(2)
        return cls(
            theta_cs=mp.pi / 2,
            u_p=u_p,
            theta_una=mp.acos(u_p),
            theta_ona=mp.pi / 4,
            m_a=mp.mpf(1) / (2 * mp.sqrt(2 * mp.pi)),
            q_g=4 * mp.pi,
        )

    @property
    def qg_ma2(self) -> Any:
        return self.q_g * self.m_a**2

    @property
    def angle_sum(self) -> Any:
        return self.theta_cs + self.theta_una + self.theta_ona


# ---------------------------------------------------------------------
# Rotation / SO(3) helpers
# ---------------------------------------------------------------------


def normalize_vector(v: np.ndarray, tol: float = 1e-15) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < tol:
        return np.zeros(3, dtype=float)
    return v / n


def so3_residuals(R: np.ndarray) -> tuple[float, float]:
    R = np.asarray(R, dtype=float)
    orth = float(np.linalg.norm(R.T @ R - np.eye(3)))
    det = abs(float(np.linalg.det(R)) - 1.0)
    return orth, det


def quaternion_from_rotation_matrix(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=float)
    tr = float(np.trace(R))
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(R)))
        if idx == 0:
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif idx == 1:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    q = np.array([w, x, y, z], dtype=float)
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return q


@dataclass(frozen=True)
class RotationReport:
    angle: float
    angle_deg: float
    axis: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]


def rotation_report_from_matrix(R: np.ndarray) -> RotationReport:
    q = quaternion_from_rotation_matrix(R)
    w = float(np.clip(q[0], -1.0, 1.0))
    angle = 2.0 * math.acos(w)
    if angle > math.pi:
        angle = 2.0 * math.pi - angle
        q = -q
        w = float(q[0])
    s = math.sqrt(max(0.0, 1.0 - w * w))
    if s < 1e-14:
        axis = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        axis = normalize_vector(np.array([q[1], q[2], q[3]], dtype=float) / s)
    return RotationReport(
        angle=angle,
        angle_deg=math.degrees(angle),
        axis=(float(axis[0]), float(axis[1]), float(axis[2])),
        quaternion=(float(q[0]), float(q[1]), float(q[2]), float(q[3])),
    )


def rotation_matrix_z(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


# ---------------------------------------------------------------------
# Exact SU(2) commutator holonomy (mpmath matrices)
# ---------------------------------------------------------------------


def _mp_su2(axis: tuple[Any, Any, Any], angle: Any) -> Any:
    nx, ny, nz = axis
    c = mp.cos(angle / 2)
    s = mp.sin(angle / 2)
    # U = c I - i s (n·σ)
    return mp.matrix(
        [
            [c - 1j * s * nz, -1j * s * (nx - 1j * ny)],
            [-1j * s * (nx + 1j * ny), c + 1j * s * nz],
        ]
    )


def compute_exact_su2_holonomy() -> tuple[Any, str, Any]:
    """
    C = U_x(pi/4) U_y(pi/4) U_x^dagger U_y^dagger
    Closed form: 2 * arccos((1 + 2*sqrt(2)) / 4)
    """
    theta = mp.pi / 4
    Ux = _mp_su2((mp.mpf(1), mp.mpf(0), mp.mpf(0)), theta)
    Uy = _mp_su2((mp.mpf(0), mp.mpf(1), mp.mpf(0)), theta)
    Ux_d = Ux.H
    Uy_d = Uy.H
    C = Ux * Uy * Ux_d * Uy_d
    tr = C[0, 0] + C[1, 1]
    half = mp.re(tr) / 2
    half = min(mp.mpf(1), max(mp.mpf(-1), half))
    phi_num = 2 * mp.acos(half)
    phi_closed = 2 * mp.acos((1 + 2 * mp.sqrt(2)) / 4)
    closed_form = "2 * arccos((1 + 2*sqrt(2)) / 4)"
    return phi_closed, closed_form, abs(phi_num - phi_closed)


# ---------------------------------------------------------------------
# Thomas-Wigner formulas
# ---------------------------------------------------------------------


def half_rapidity_tanh(beta: Any) -> Any:
    """tanh(atanh(beta)/2) = beta / (1 + sqrt(1 - beta^2))."""
    b = mp.mpf(beta)
    if not (0 <= b < 1):
        raise ValueError("beta must satisfy 0 <= beta < 1")
    return b / (1 + mp.sqrt(1 - b * b))


def tw_angle_unequal(beta_1: Any, beta_2: Any, theta: Any) -> Any:
    """Wigner angle for boosts of unequal magnitudes separated by theta."""
    k1 = half_rapidity_tanh(beta_1)
    k2 = half_rapidity_tanh(beta_2)
    z = k1 * k2
    th = mp.mpf(theta)
    return 2 * mp.atan((mp.sin(th) * z) / (1 + mp.cos(th) * z))


def tw_angle_exact(beta: Any, theta: Any) -> Any:
    """Equal-speed special case."""
    return tw_angle_unequal(beta, beta, theta)


def analytic_bu_holonomy(t: CGMThresholds) -> tuple[Any, Any]:
    """
    Declared embedding: orthogonal boosts of magnitudes theta_ona and m_a.
    omega = 2 atan(k(theta_ona) k(m_a))
    delta_BU = 2 omega = 4 atan(k(theta_ona) k(m_a))
    """
    omega = tw_angle_unequal(t.theta_ona, t.m_a, mp.pi / 2)
    return omega, 2 * omega


def _bisect_root(
    f: Any,
    lo: Any,
    hi: Any,
    *,
    tol: Any = mp.mpf("1e-40"),
    max_iter: int = 200,
) -> Any:
    flo = f(lo)
    fhi = f(hi)
    if flo * fhi > 0:
        raise RuntimeError("root not bracketed")
    a, b = lo, hi
    fa = flo
    for _ in range(max_iter):
        mid = (a + b) / 2
        fm = f(mid)
        if abs(fm) < tol or abs(b - a) < tol:
            return mid
        if fa * fm <= 0:
            b = mid
        else:
            a, fa = mid, fm
    return (a + b) / 2


def solve_beta_star(theta: Any, target: Any) -> Any:
    def f(b: Any) -> Any:
        return tw_angle_exact(b, theta) - target

    return _bisect_root(f, mp.mpf("1e-12"), mp.mpf(1) - mp.mpf("1e-12"))


def solve_theta_star(beta: Any, target: Any) -> Any:
    def f(th: Any) -> Any:
        return tw_angle_exact(beta, th) - target

    return _bisect_root(f, mp.mpf("1e-12"), mp.pi / 2)


# ---------------------------------------------------------------------
# TW small-angle calibration
# ---------------------------------------------------------------------


def tw_small_angle_theory(u: np.ndarray, v: np.ndarray, c: float) -> float:
    return float(np.linalg.norm(np.cross(u, v)) / (2.0 * c**2))


def tw_angle_from_gyration(gs: GyroVectorSpace, u: np.ndarray, v: np.ndarray) -> float:
    return rotation_report_from_matrix(np.asarray(gs.gyration(u, v), dtype=float)).angle


def _unit_grid() -> list[np.ndarray]:
    raw = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, -1.0, 0.0],
        [1.0, 0.0, -1.0],
    ]
    return [normalize_vector(np.asarray(v, dtype=float)) for v in raw]


def run_tw_benchmark(gs: GyroVectorSpace) -> list[dict[str, float]]:
    units = _unit_grid()
    scales = (0.25, 0.5, 0.75, 1.0)
    beta_max_values = [0.02, 0.03, 0.05, 0.08, 0.10]
    rows: list[dict[str, float]] = []
    for beta_max in beta_max_values:
        xs: list[float] = []
        ys: list[float] = []
        for i, ui in enumerate(units):
            for uj in units[i + 1 :]:
                for su in scales:
                    for sv in scales:
                        u = ui * (beta_max * su)
                        v = uj * (beta_max * sv)
                        if float(np.linalg.norm(u)) < 1e-15 or float(np.linalg.norm(v)) < 1e-15:
                            continue
                        x = tw_small_angle_theory(u, v, 1.0)
                        if x < 1e-18:
                            continue
                        ys.append(tw_angle_from_gyration(gs, u, v))
                        xs.append(x)
        X = np.asarray(xs)
        Y = np.asarray(ys)
        slope = float(np.dot(X, Y) / np.dot(X, X))
        resid = Y - slope * X
        rows.append(
            {
                "beta_max": float(beta_max),
                "n": float(len(xs)),
                "slope": slope,
                "mean_abs_residual": float(np.mean(np.abs(resid))),
                "max_abs_residual": float(np.max(np.abs(resid))),
            }
        )
    return rows


# ---------------------------------------------------------------------
# Declared embedding and loops
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class EmbeddingSpec:
    name: str
    c: float
    una_vector: tuple[float, float, float]
    ona_vector: tuple[float, float, float]
    bu_plus_vector: tuple[float, float, float]
    bu_minus_vector: tuple[float, float, float]


def declared_embedding(
    t: CGMThresholds,
    *,
    u_p: float | None = None,
    theta_ona: float | None = None,
    m_a: float | None = None,
) -> EmbeddingSpec:
    """
    Payload-only embedding into the Einstein gyrovector model with c = 1.

    Modeling choice: theta_ona enters as a coordinate magnitude (not only as an
    angle between boosts). CS is the gauge frame and is not traversed.
    """
    return EmbeddingSpec(
        name="payload_scalar_embedding",
        c=1.0,
        una_vector=(float(t.u_p if u_p is None else u_p), 0.0, 0.0),
        ona_vector=(0.0, float(t.theta_ona if theta_ona is None else theta_ona), 0.0),
        bu_plus_vector=(0.0, 0.0, float(t.m_a if m_a is None else m_a)),
        bu_minus_vector=(0.0, 0.0, -float(t.m_a if m_a is None else m_a)),
    )


def assert_in_ball(v: tuple[float, float, float], c: float) -> None:
    norm = float(np.linalg.norm(np.asarray(v, dtype=float)))
    if not (norm < c):
        raise ValueError(
            f"Vector {v} has norm {norm:.12f}, outside open Einstein ball of radius {c}."
        )


@dataclass(frozen=True)
class LoopHolonomyResult:
    name: str
    path: tuple[str, ...]
    leg_angles: tuple[float, ...]
    leg_matrices: tuple[np.ndarray, ...]
    total: RotationReport
    product: np.ndarray


def compute_loop_holonomy(
    gs: GyroVectorSpace,
    name: str,
    path: tuple[str, ...],
    points: dict[str, tuple[float, float, float]],
) -> LoopHolonomyResult:
    for stage in set(path):
        assert_in_ball(points[stage], float(gs.c))

    leg_angles: list[float] = []
    leg_matrices: list[np.ndarray] = []
    total = np.eye(3)
    for i in range(len(path) - 1):
        a = np.asarray(points[path[i]], dtype=float)
        b = np.asarray(points[path[i + 1]], dtype=float)
        G = np.asarray(gs.gyration(a, b), dtype=float)
        leg_angles.append(rotation_report_from_matrix(G).angle)
        leg_matrices.append(G)
        total = total @ G

    return LoopHolonomyResult(
        name=name,
        path=path,
        leg_angles=tuple(float(x) for x in leg_angles),
        leg_matrices=tuple(leg_matrices),
        total=rotation_report_from_matrix(total),
        product=total,
    )


def points_from_embedding(emb: EmbeddingSpec) -> dict[str, tuple[float, float, float]]:
    return {
        "UNA": emb.una_vector,
        "ONA": emb.ona_vector,
        "BU+": emb.bu_plus_vector,
        "BU-": emb.bu_minus_vector,
    }


# ---------------------------------------------------------------------
# Finite layer
# ---------------------------------------------------------------------


def compute_finite_holonomy() -> dict[str, Any] | None:
    if not _FINITE_OK or build_kernel is None or verify_k4_w2 is None:
        return None
    if decompose_byte is None or fold_disagreement is None:
        return None

    kernel = build_kernel()
    distribution: dict[int, int] = {}
    bu_center = 0
    for byte in range(256):
        d = fold_disagreement(byte)
        distribution[d] = distribution.get(d, 0) + 1
        fiber = decompose_byte(byte)
        if fiber.phase_net[3] == 1:
            bu_center += 1

    k4 = verify_k4_w2()

    return {
        "flat_bytes": len(kernel.bytes_flat),
        "curved_bytes": len(kernel.bytes_curved),
        "fold_disagreement_distribution": distribution,
        "bu_boundary_disagreement_count": bu_center,
        "k4_all_pass": bool(k4.all_pass),
        "w2_rest_ok": bool(k4.w2_rest_ok),
        "w2p_rest_ok": bool(k4.w2p_rest_ok),
        "w2_involution_ok": bool(k4.w2_involution_ok),
        "t2_shell_ok": bool(k4.t2_shell_ok),
        "t2_chi_ok": bool(k4.t2_chi_ok),
    }


def binomial_c4(k: int) -> int:
    # C(4,k)
    return math.comb(4, k)


# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------


def run() -> list[tuple[str, bool]]:
    gates: list[tuple[str, bool]] = []
    t = CGMThresholds.make()
    gs = GyroVectorSpace(c=1.0)

    # A. Thresholds
    _section("A. THRESHOLDS")
    print(f"  theta_cs                         {mp_to_str(t.theta_cs, 20)}")
    print(f"  u_p                              {mp_to_str(t.u_p, 20)}")
    print(f"  theta_una = arccos(u_p)          {mp_to_str(t.theta_una, 20)}")
    print(f"  theta_ona                        {mp_to_str(t.theta_ona, 20)}")
    print(f"  m_a                              {mp_to_str(t.m_a, 20)}")
    print(f"  q_g                              {mp_to_str(t.q_g, 20)}")
    print(f"  theta_cs+theta_una+theta_ona     {mp_to_str(t.angle_sum, 20)}")
    print(f"  q_g * m_a^2                      {mp_to_str(t.qg_ma2, 20)}")
    _check(
        gates,
        f"|angle_sum - pi| < 1e-15 (got {abs(float(t.angle_sum) - math.pi):.3e})",
        abs(float(t.angle_sum) - math.pi) < 1e-15,
    )
    _check(
        gates,
        f"|q_g*m_a^2 - 1/2| < 1e-15 (got {abs(float(t.qg_ma2) - 0.5):.3e})",
        abs(float(t.qg_ma2) - 0.5) < 1e-15,
    )
    print()

    # B. SU(2)
    _section("B. EXACT SU(2) COMMUTATOR HOLONOMY")
    phi_su2, closed_form, su2_err = compute_exact_su2_holonomy()
    print(f"  phi_SU2 (rad)                    {mp_to_str(phi_su2, 20)}")
    print(f"  phi_SU2 (deg)                    {float(phi_su2) * 180.0 / math.pi:.12f}")
    print(f"  closed_form                      {closed_form}")
    print(f"  |numeric - closed_form|           {mp_to_str(su2_err, 12)}")
    _check(gates, f"SU(2) mpmath product matches closed form (err {float(su2_err):.3e})", float(su2_err) < 1e-40)
    print()

    # C. TW calibration
    _section("C. THOMAS-WIGNER SMALL-ANGLE CALIBRATION")
    print("  gyration angle vs ||u x v||/(2c^2) on a fixed velocity grid")
    tw_rows = run_tw_benchmark(gs)
    print(f"  {'beta_max':>10s} {'n':>6s} {'slope':>12s} {'mean|res|':>14s} {'max|res|':>14s}")
    row_005: dict[str, float] | None = None
    for row in tw_rows:
        print(
            f"  {row['beta_max']:10.3f} {int(row['n']):6d} {row['slope']:12.6f} "
            f"{row['mean_abs_residual']:14.3e} {row['max_abs_residual']:14.3e}"
        )
        if abs(row["beta_max"] - 0.05) < 1e-12:
            row_005 = row
    assert row_005 is not None
    slope_err = abs(row_005["slope"] - 1.0)
    tw_ok = (slope_err < TOL_TW_SLOPE) and (row_005["max_abs_residual"] < TOL_TW_RESID)
    print(f"  slope_error (beta<=0.05)         {slope_err:.3e}  (tol {TOL_TW_SLOPE})")
    print(f"  max_residual (beta<=0.05)        {row_005['max_abs_residual']:.3e}  (tol {TOL_TW_RESID})")
    _check(
        gates,
        f"TW slope~1 and residual (beta<=0.05): |slope-1|={slope_err:.3e}, max|res|={row_005['max_abs_residual']:.3e}",
        tw_ok,
    )
    betas = np.asarray([r["beta_max"] for r in tw_rows], dtype=float)
    slope_errors = np.asarray([abs(r["slope"] - 1.0) for r in tw_rows], dtype=float)
    max_resids = np.asarray([r["max_abs_residual"] for r in tw_rows], dtype=float)
    slope_order = float(np.polyfit(np.log(betas), np.log(slope_errors), 1)[0])
    resid_order = float(np.polyfit(np.log(betas), np.log(max_resids), 1)[0])
    print(f"  slope-error convergence order   {slope_order:.6f}")
    print(f"  residual convergence order      {resid_order:.6f}")
    _check(gates, f"slope error scales ~ beta^2 (order {slope_order:.3f})", 1.8 < slope_order < 2.2)
    _check(gates, f"abs residual scales ~ beta^4 (order {resid_order:.3f})", 3.8 < resid_order < 4.2)
    print()

    # D. Embedding
    _section("D. DECLARED PAYLOAD EMBEDDING")
    emb = declared_embedding(t)
    points = points_from_embedding(emb)
    print(f"  name                             {emb.name}")
    print(f"  c                                {emb.c}")
    print(f"  UNA                              {emb.una_vector}")
    print(f"  ONA                              {emb.ona_vector}")
    print(f"  BU+                              {emb.bu_plus_vector}")
    print(f"  BU-                              {emb.bu_minus_vector}")
    print("  CS                               gauge frame (not a traversed vertex)")
    print("  modeling_choice                  theta_ona used as coordinate magnitude")
    domain_ok = True
    for name, vec in points.items():
        nrm = float(np.linalg.norm(np.asarray(vec, dtype=float)))
        inside = nrm < emb.c
        print(f"  ||{name}||                          {nrm:.12f}  inside_ball={inside}")
        domain_ok = domain_ok and inside
    _check(gates, "all payload stage vectors satisfy ||v|| < c", domain_ok)
    print()

    # E. Analytic BU holonomy
    _section("E. ANALYTIC BU HOLONOMY")
    print("  formula  delta_BU = 4*atan(k(theta_ona)*k(m_a))")
    print("           k(beta) = beta/(1+sqrt(1-beta^2)), theta = pi/2")
    omega_exact, delta_bu_exact = analytic_bu_holonomy(t)
    k_ona = half_rapidity_tanh(t.theta_ona)
    k_ma = half_rapidity_tanh(t.m_a)
    rho_exact = delta_bu_exact / t.m_a
    rho_zero = 2 * k_ona
    rho_corr = rho_exact - rho_zero
    delta_gap_exact = 1 - rho_exact
    print(f"  k(theta_ona)                     {mp_to_str(k_ona, 20)}")
    print(f"  k(m_a)                           {mp_to_str(k_ma, 20)}")
    print(f"  omega_exact (ONA-BU corner)      {mp_to_str(omega_exact, 20)}")
    print(f"  delta_BU_exact                   {mp_to_str(delta_bu_exact, 20)}")
    print(f"  rho_exact = delta_BU/m_a         {mp_to_str(rho_exact, 20)}")
    print(f"  Delta_exact = 1 - rho            {mp_to_str(delta_gap_exact, 20)}")
    print(f"  rho(m_a -> 0) = 2*k(theta_ona)   {mp_to_str(rho_zero, 20)}")
    print(f"  finite-BU correction rho-rho0    {mp_to_str(rho_corr, 20)}")
    print(f"  baseline gap 1-rho0              {mp_to_str(1 - rho_zero, 20)}")
    print(f"  final gap Delta                  {mp_to_str(delta_gap_exact, 20)}")
    delta_bu = float(delta_bu_exact)
    omega_analytic = float(omega_exact)
    rho = float(rho_exact)
    delta_gap = float(delta_gap_exact)
    print()

    # F. Matrix realization
    _section("F. MATRIX REALIZATION OF BU LOOP")
    bu_loop = compute_loop_holonomy(gs, "bu_dual_pole_loop", ("ONA", "BU+", "BU-", "ONA"), points)
    omega_egress = bu_loop.leg_angles[0]
    omega_middle = bu_loop.leg_angles[1]
    omega_ingress = bu_loop.leg_angles[2]
    omega_mean = 0.5 * (omega_egress + omega_ingress)
    print(f"  path                             {' -> '.join(bu_loop.path)}")
    print(f"  omega_egress  = angle(ONA->BU+)  {omega_egress:.16f}")
    print(f"  omega_middle  = angle(BU+->BU-)  {omega_middle:.16f}")
    print(f"  omega_ingress = angle(BU-->ONA)  {omega_ingress:.16f}")
    print(f"  omega_mean                       {omega_mean:.16f}")
    print(f"  |omega_egress - omega_ingress|   {abs(omega_egress - omega_ingress):.3e}")
    print(f"  delta_BU_matrix = total angle    {bu_loop.total.angle:.16f}")
    print(f"  |delta_BU_matrix - (e+i)|        {abs(bu_loop.total.angle - (omega_egress + omega_ingress)):.3e}")
    print(f"  axis                             {bu_loop.total.axis}")
    print(f"  quaternion                       {bu_loop.total.quaternion}")
    matrix_err = abs(bu_loop.total.angle - delta_bu)
    print(f"  |matrix - analytic| delta_BU     {matrix_err:.3e}")
    print(f"  |omega_mean - analytic omega|    {abs(omega_mean - omega_analytic):.3e}")
    orth, det = so3_residuals(bu_loop.product)
    print(f"  BU product ||R^T R - I||         {orth:.3e}")
    print(f"  BU product |det(R)-1|            {det:.3e}")
    evals = np.linalg.eigvals(bu_loop.product)
    evals = evals[np.argsort(np.angle(evals))]
    trace_expected = 1.0 + 2.0 * math.cos(delta_bu)
    trace_resid = abs(float(np.trace(bu_loop.product)) - trace_expected)
    print(f"  BU eigenvalues                   {evals}")
    print(f"  |tr(R) - (1+2cos(delta_BU))|     {trace_resid:.3e}")
    _check(gates, f"BU+->BU- collinear angle ~ 0 (got {omega_middle:.3e})", abs(omega_middle) < TOL_ANGLE)
    _check(
        gates,
        f"|omega_egress - omega_ingress| < 1e-8 (got {abs(omega_egress - omega_ingress):.3e})",
        abs(omega_egress - omega_ingress) < 1e-8,
    )
    _check(gates, f"matrix BU angle matches analytic (err {matrix_err:.3e})", matrix_err < TOL_MATRIX)
    _check(gates, f"BU product in SO(3) (orth {orth:.3e}, det {det:.3e})", orth < TOL_SO3 and det < TOL_SO3)
    _check(gates, f"BU trace = 1+2cos(delta_BU) (resid {trace_resid:.3e})", trace_resid < 1e-8)
    print()

    # Canonical gyration SO(3) audit
    _section("G. CANONICAL GYRATION SO(3) AUDIT")
    pair_names = [
        ("UNA", "ONA"),
        ("ONA", "UNA"),
        ("ONA", "BU+"),
        ("BU+", "BU-"),
        ("BU-", "ONA"),
        ("UNA", "BU+"),
        ("UNA", "BU-"),
        ("ONA", "BU-"),
        ("BU+", "ONA"),
        ("BU-", "BU+"),
    ]
    max_orth = 0.0
    max_det = 0.0
    for a, b in pair_names:
        G = np.asarray(
            gs.gyration(np.asarray(points[a], dtype=float), np.asarray(points[b], dtype=float)),
            dtype=float,
        )
        o, d = so3_residuals(G)
        max_orth = max(max_orth, o)
        max_det = max(max_det, d)
        print(f"  {a}->{b:3s}  ||G^T G - I||={o:.3e}  |det-1|={d:.3e}")
    _check(
        gates,
        f"all canonical gyrations in SO(3) (max orth {max_orth:.3e}, max |det-1| {max_det:.3e})",
        max_orth < TOL_SO3 and max_det < TOL_SO3,
    )
    print()

    # Ungar inverse pairs
    _section("H. GYRATION INVERSE PAIRS ON EMBEDDING")
    print("  Validate implementation of Ungar theorem: gyr(u,v)^-1 = gyr(v,u)")
    inv_pairs = [("UNA", "ONA"), ("UNA", "BU+"), ("UNA", "BU-"), ("ONA", "BU+"), ("ONA", "BU-"), ("BU+", "BU-")]
    max_inv = 0.0
    for a, b in inv_pairs:
        Ga = np.asarray(
            gs.gyration(np.asarray(points[a], dtype=float), np.asarray(points[b], dtype=float)),
            dtype=float,
        )
        Gb = np.asarray(
            gs.gyration(np.asarray(points[b], dtype=float), np.asarray(points[a], dtype=float)),
            dtype=float,
        )
        resid = float(np.linalg.norm(Ga @ Gb - np.eye(3)))
        max_inv = max(max_inv, resid)
        print(f"  ||gyr({a},{b})*gyr({b},{a}) - I||  {resid:.3e}")
    _check(gates, f"gyr(u,v)*gyr(v,u)~I on payload pairs (max resid {max_inv:.3e})", max_inv < TOL_MATRIX)
    print()

    # Path reversal and cyclic re-rooting
    _section("I. PATH REVERSAL AND CYCLIC RE-ROOTING")
    print("  Empirical matrix-layer noise floor at canonical magnitudes ~ 1e-8")
    bu_rev = compute_loop_holonomy(
        gs, "bu_dual_pole_reverse", ("ONA", "BU-", "BU+", "ONA"), points
    )
    rev_resid = float(np.linalg.norm(bu_rev.product - bu_loop.product.T))
    rev_angle_diff = abs(bu_rev.total.angle - bu_loop.total.angle)
    print(f"  reverse path                     {' -> '.join(bu_rev.path)}")
    print(f"  reverse angle                    {bu_rev.total.angle:.16f}")
    print(f"  |reverse_angle - forward_angle|  {rev_angle_diff:.3e}")
    print(f"  ||H(rev) - H(fwd)^T||            {rev_resid:.3e}")
    _check(gates, f"path reversal gives inverse holonomy (resid {rev_resid:.3e})", rev_resid < TOL_MATRIX)
    _check(
        gates,
        f"reverse angle equals forward angle (diff {rev_angle_diff:.3e})",
        rev_angle_diff < TOL_MATRIX,
    )

    bu_base = compute_loop_holonomy(
        gs, "bu_loop_from_BU+", ("BU+", "BU-", "ONA", "BU+"), points
    )
    base_angle_diff = abs(bu_base.total.angle - bu_loop.total.angle)
    # Conjugacy invariants under cyclic re-rooting: angle and trace
    base_trace_diff = abs(float(np.trace(bu_base.product)) - float(np.trace(bu_loop.product)))
    print(f"  cyclic re-root path              {' -> '.join(bu_base.path)}")
    print(f"  cyclic re-root angle             {bu_base.total.angle:.16f}")
    print(f"  |angle - ONA-started angle|      {base_angle_diff:.3e}")
    print(f"  |trace difference|               {base_trace_diff:.3e}")
    _check(
        gates,
        f"holonomy angle invariant under cyclic re-rooting (diff {base_angle_diff:.3e})",
        base_angle_diff < 1e-10,
    )
    _check(
        gates,
        f"trace invariant under cyclic re-rooting (diff {base_trace_diff:.3e})",
        base_trace_diff < TOL_MATRIX,
    )
    print()

    # Rotational covariance
    _section("J. GLOBAL ROTATIONAL COVARIANCE")
    Q = rotation_matrix_z(math.pi / 7)
    rotated = {name: tuple((Q @ np.asarray(vec, dtype=float)).tolist()) for name, vec in points.items()}
    bu_rot = compute_loop_holonomy(gs, "rotated_bu_loop", ("ONA", "BU+", "BU-", "ONA"), rotated)
    cov_resid = float(np.linalg.norm(bu_rot.product - Q @ bu_loop.product @ Q.T))
    ang_resid = abs(bu_rot.total.angle - bu_loop.total.angle)
    print(f"  rotation                         Rz(pi/7)")
    print(f"  rotated angle                    {bu_rot.total.angle:.16f}")
    print(f"  |angle_rotated - angle|          {ang_resid:.3e}")
    print(f"  ||H(Q gamma) - Q H Q^T||         {cov_resid:.3e}")
    _check(gates, f"holonomy angle basis-invariant (diff {ang_resid:.3e})", ang_resid < TOL_ANGLE_INV)
    _check(gates, f"global rotational covariance (resid {cov_resid:.3e})", cov_resid < TOL_MATRIX)
    print()

    # Palindromic conjugacy
    _section("K. PALINDROMIC CONJUGACY")
    print("  path  UNA -> ONA -> BU+ -> BU- -> ONA -> UNA")
    print("  6 payload positions on a 5-edge closed path; CS is gauge frame")
    print("  Gyrogroup theorem: gyr(v,u) = gyr(u,v)^-1  (Ungar)")
    print("  Consequence: H_pal = A H_BU A^-1 with A = gyr(UNA,ONA)")
    pal = compute_loop_holonomy(
        gs,
        "payload_palindrome_loop",
        ("UNA", "ONA", "BU+", "BU-", "ONA", "UNA"),
        points,
    )
    A = np.asarray(
        gs.gyration(np.asarray(points["UNA"], dtype=float), np.asarray(points["ONA"], dtype=float)),
        dtype=float,
    )
    A_rev = np.asarray(
        gs.gyration(np.asarray(points["ONA"], dtype=float), np.asarray(points["UNA"], dtype=float)),
        dtype=float,
    )
    # Independent inverse for SO(3): A^-1 = A^T (not the path-built A_rev)
    reverse_transport_resid = float(np.linalg.norm(A_rev - A.T))
    a_orth = float(np.linalg.norm(A @ A.T - np.eye(3)))
    conj_exact = A @ bu_loop.product @ A.T
    conj_resid = float(np.linalg.norm(pal.product - conj_exact))
    axis_bu = np.asarray(bu_loop.total.axis, dtype=float)
    axis_pal = np.asarray(pal.total.axis, dtype=float)
    axis_expected = normalize_vector(A @ axis_bu)
    axis_align = abs(float(np.dot(axis_expected, axis_pal)))
    print(f"  palindrome total angle           {pal.total.angle:.16f}")
    print(f"  BU loop total angle              {bu_loop.total.angle:.16f}")
    print(f"  |palindrome - delta_BU_matrix|   {abs(pal.total.angle - bu_loop.total.angle):.3e}")
    print(f"  BU axis                          {bu_loop.total.axis}")
    print(f"  palindrome axis                  {pal.total.axis}")
    print(f"  transported BU axis A*n_BU       {tuple(float(x) for x in axis_expected)}")
    print(f"  |dot(transported, pal axis)|     {axis_align:.16f}")
    print(f"  ||gyr(ONA,UNA) - A^T||           {reverse_transport_resid:.3e}")
    print(f"  ||A A^T - I||                    {a_orth:.3e}")
    print(f"  ||H_pal - A H_BU A^T||           {conj_resid:.3e}")
    print(f"  quaternion w (BU)                {bu_loop.total.quaternion[0]:.16f}")
    print(f"  quaternion w (palindrome)        {pal.total.quaternion[0]:.16f}")
    orth_p, det_p = so3_residuals(pal.product)
    print(f"  palindrome ||R^T R - I||         {orth_p:.3e}")
    print(f"  palindrome |det(R)-1|            {det_p:.3e}")
    _check(
        gates,
        f"reverse transport equals A^-1=A^T (resid {reverse_transport_resid:.3e})",
        reverse_transport_resid < TOL_MATRIX,
    )
    _check(gates, f"A A^T = I (resid {a_orth:.3e})", a_orth < TOL_SO3)
    _check(gates, f"palindrome = A * H_BU * A^-1 (resid {conj_resid:.3e})", conj_resid < TOL_MATRIX)
    _check(
        gates,
        f"palindrome axis = transported BU axis (1-|dot|={1.0 - axis_align:.3e})",
        (1.0 - axis_align) < TOL_MATRIX,
    )
    _check(
        gates,
        f"conjugacy preserves quaternion scalar w (diff {abs(bu_loop.total.quaternion[0] - pal.total.quaternion[0]):.3e})",
        abs(bu_loop.total.quaternion[0] - pal.total.quaternion[0]) < 1e-12,
    )
    _check(gates, "palindrome product in SO(3)", orth_p < TOL_SO3 and det_p < TOL_SO3)
    print()

    # Aperture from analytic
    _section("L. APERTURE AND PRECISION GOVERNANCE")
    print(f"  BU_HOLONOMY_ANGLE (analytic)     {delta_bu:.16f}")
    print(f"  m_a                              {float(t.m_a):.16f}")
    print(f"  rho = BU_HOLONOMY_ANGLE / m_a    {rho:.16f}")
    print(f"  Delta = 1 - rho                  {delta_gap:.16f}")
    print(f"  closure_percent                  {100.0 * rho:.10f}")
    print(f"  aperture_percent                 {100.0 * delta_gap:.10f}")
    if BU_HOLONOMY_ANGLE is not None:
        shared = float(BU_HOLONOMY_ANGLE)
        rel = abs(delta_bu - shared) / delta_bu
        print(f"  shared BU_HOLONOMY_ANGLE         {shared:.16f}")
        print(f"  |script - shared|                {abs(delta_bu - shared):.3e}")
        print(f"  relative |script - shared|       {rel:.3e}")
        # alpha inherits ~4x relative sensitivity via alpha ~ delta_BU^4 / m_a
        print(f"  4 * relative (alpha sensitivity) {4.0 * rel:.3e}")
        _check(
            gates,
            f"script analytic matches shared BU_HOLONOMY_ANGLE (rel {rel:.3e})",
            rel < 1e-14,
        )
    _check(gates, f"0 < rho < 1 (rho={rho:.12f})", 0.0 < rho < 1.0)
    print()

    # Embedding sensitivity / rank-2 dependency
    _section("M. RANK-2 DEPENDENCY AND EMBEDDING SENSITIVITY")
    print("  Analytic: delta_BU = 4*atan(k(theta_ona)*k(m_a)) depends only on (theta_ona, m_a).")
    print("  UNA enters only via conjugation (axis transport), not the angle.")
    print("  CS is gauge frame only.")
    print("  Magnitude channel: ONA x BU; orientation channel: UNA conjugation.")
    print(f"  d(delta_BU)/du_p                 0 exactly under declared embedding")
    eps = 1e-5
    th0 = float(t.theta_ona)
    m0 = float(t.m_a)

    def _delta_at(th: float, m: float) -> float:
        e = declared_embedding(t, theta_ona=th, m_a=m)
        r = compute_loop_holonomy(gs, "pert", ("ONA", "BU+", "BU-", "ONA"), points_from_embedding(e))
        return r.total.angle

    d_dth_mat = (_delta_at(th0 + eps, m0) - _delta_at(th0 - eps, m0)) / (2 * eps)
    d_dm_mat = (_delta_at(th0, m0 + eps) - _delta_at(th0, m0 - eps)) / (2 * eps)
    h_th = mp.mpf(str(eps))
    h_m = mp.mpf(str(eps))
    d_dth_an = float(
        (
            2 * tw_angle_unequal(t.theta_ona + h_th, t.m_a, mp.pi / 2)
            - 2 * tw_angle_unequal(t.theta_ona - h_th, t.m_a, mp.pi / 2)
        )
        / (2 * h_th)
    )
    d_dm_an = float(
        (
            2 * tw_angle_unequal(t.theta_ona, t.m_a + h_m, mp.pi / 2)
            - 2 * tw_angle_unequal(t.theta_ona, t.m_a - h_m, mp.pi / 2)
        )
        / (2 * h_m)
    )
    print(f"  d(delta_BU)/dtheta_ona (matrix)  {d_dth_mat:.8f}")
    print(f"  d(delta_BU)/dtheta_ona (analytic){d_dth_an:.8f}")
    print(f"  d(delta_BU)/dm_a (matrix)        {d_dm_mat:.8f}")
    print(f"  d(delta_BU)/dm_a (analytic)      {d_dm_an:.8f}")
    print(f"  (theta_ona/delta_BU)*d/dtheta    {th0 * d_dth_mat / delta_bu:.8f}")
    print(f"  (m_a/delta_BU)*d/dm_a            {m0 * d_dm_mat / delta_bu:.8f}")
    print(f"  |matrix - analytic| d/dtheta     {abs(d_dth_mat - d_dth_an):.3e}")
    print(f"  |matrix - analytic| d/dm_a       {abs(d_dm_mat - d_dm_an):.3e}")
    _check(
        gates,
        f"matrix d/dtheta_ona matches analytic (diff {abs(d_dth_mat - d_dth_an):.3e})",
        abs(d_dth_mat - d_dth_an) < 5e-4,
    )
    _check(
        gates,
        f"matrix d/dm_a matches analytic (diff {abs(d_dm_mat - d_dm_an):.3e})",
        abs(d_dm_mat - d_dm_an) < 5e-4,
    )
    print()

    # Open comparisons (quarantined)
    _section("N. OPEN COMPARISONS")
    print("  Quarantined: no identity asserted; excluded from foundational claims.")
    three_delta = 3.0 * delta_bu
    w_resid = float(phi_su2) - three_delta
    print(f"  phi_SU2                          {float(phi_su2):.16f}")
    print(f"  3 * delta_BU                     {three_delta:.16f}")
    print(f"  W = phi_SU2 - 3*delta_BU         {w_resid:.16f}")
    print(f"  W / phi_SU2                      {w_resid / float(phi_su2):.16f}")
    dyadic_bu = 50.0 / 256.0
    dyadic_gap = 5.0 / 256.0
    print(f"  50/256                           {dyadic_bu:.16f}")
    print(f"  |delta_BU - 50/256|              {abs(delta_bu - dyadic_bu):.16f}")
    print(f"  rel |delta_BU - 50/256|          {abs(delta_bu - dyadic_bu) / delta_bu:.6e}")
    print(f"  5/256                            {dyadic_gap:.16f}")
    print(f"  |Delta - 5/256|                  {abs(delta_gap - dyadic_gap):.16f}")
    print(f"  rel |Delta - 5/256|              {abs(delta_gap - dyadic_gap) / delta_gap:.6e}")
    print()

    # Wigner map + Jacobian
    _section("O. WIGNER MAP AT (u_p, theta_ona)")
    print("  Equal-speed Wigner angle at thresholds vs aperture scale m_a.")
    print("  These are independent quantities; equality is not assumed.")
    w_canon = float(tw_angle_exact(t.u_p, t.theta_ona))
    m_a_f = float(t.m_a)
    offset = w_canon - m_a_f
    print(f"  omega(u_p, theta_ona)            {w_canon:.16f}")
    print(f"  m_a                              {m_a_f:.16f}")
    print(f"  omega - m_a                      {offset:.16f}")
    print(f"  (omega - m_a)/m_a                {offset / m_a_f:.10f}")
    print(f"  equal within tol {TOL_MAP}:      {abs(offset) < TOL_MAP}")

    h = mp.mpf("1e-20")
    d_om_db = (tw_angle_exact(t.u_p + h, t.theta_ona) - tw_angle_exact(t.u_p - h, t.theta_ona)) / (2 * h)
    d_om_dth = (tw_angle_exact(t.u_p, t.theta_ona + h) - tw_angle_exact(t.u_p, t.theta_ona - h)) / (2 * h)
    d_om_db_exact = (12 * mp.sqrt(2) - 4) / 17
    d_om_dth_exact = (21 - 12 * mp.sqrt(2)) / 17
    jac_sum = d_om_db_exact + d_om_dth_exact
    print(f"  d_omega/dbeta  (numeric)         {float(d_om_db):.16f}")
    print(f"  d_omega/dtheta (numeric)         {float(d_om_dth):.16f}")
    print(f"  d_omega/dbeta  exact form        {mp_to_str(d_om_db_exact, 20)}")
    print(f"  d_omega/dtheta exact form        {mp_to_str(d_om_dth_exact, 20)}")
    print(f"  exact forms                      (12*sqrt(2)-4)/17 , (21-12*sqrt(2))/17")
    print(f"  d_omega/dbeta + d_omega/dtheta   {mp_to_str(jac_sum, 20)}")
    print(f"  boost-magnitude response share   {float(d_om_db_exact):.6%}")
    print(f"  angular response share           {float(d_om_dth_exact):.6%}")
    _check(
        gates,
        f"d_omega/dbeta matches exact form (err {float(abs(d_om_db - d_om_db_exact)):.3e})",
        abs(d_om_db - d_om_db_exact) < mp.mpf("1e-20"),
    )
    _check(
        gates,
        f"d_omega/dtheta matches exact form (err {float(abs(d_om_dth - d_om_dth_exact)):.3e})",
        abs(d_om_dth - d_om_dth_exact) < mp.mpf("1e-20"),
    )
    _check(
        gates,
        f"canonical Wigner response derivatives sum to 1 (got {float(jac_sum):.16f})",
        abs(jac_sum - 1) < mp.mpf("1e-40"),
    )

    beta_star = solve_beta_star(t.theta_ona, t.m_a)
    theta_star = solve_theta_star(t.u_p, t.m_a)
    beta_star_f = float(beta_star)
    theta_star_f = float(theta_star)
    w_check_b = float(tw_angle_exact(beta_star, t.theta_ona))
    w_check_t = float(tw_angle_exact(t.u_p, theta_star))
    beta_pred = float(t.u_p) - offset / float(d_om_db_exact)
    print(f"  beta_star                        {beta_star_f:.16f}")
    print(f"  beta_star / u_p                  {beta_star_f / float(t.u_p):.16f}")
    print(f"  linear beta* prediction          {beta_pred:.16f}")
    print(f"  |linear - exact| beta*           {abs(beta_pred - beta_star_f):.3e}")
    print(f"  omega(beta_star, theta_ona)      {w_check_b:.16f}")
    print(f"  theta_star                       {theta_star_f:.16f}")
    print(f"  theta_star / theta_ona           {theta_star_f / float(t.theta_ona):.16f}")
    print(f"  omega(u_p, theta_star)           {w_check_t:.16f}")
    _check(
        gates,
        f"|omega(beta_star,theta_ona)-m_a| < 1e-12 (got {abs(w_check_b - m_a_f):.3e})",
        abs(w_check_b - m_a_f) < 1e-12,
    )
    _check(
        gates,
        f"|omega(u_p,theta_star)-m_a| < 1e-12 (got {abs(w_check_t - m_a_f):.3e})",
        abs(w_check_t - m_a_f) < 1e-12,
    )
    print()

    # Finite layer
    _section("P. FINITE hQVM HOLONOMY")
    finite = compute_finite_holonomy()
    if finite is None:
        print(f"  unavailable                      {_FINITE_IMPORT_ERROR}")
        _check(gates, "finite hQVM layer importable", False)
        bu_frac = None
    else:
        dist = finite["fold_disagreement_distribution"]
        print(f"  flat_bytes                       {finite['flat_bytes']}")
        print(f"  curved_bytes                     {finite['curved_bytes']}")
        print(f"  fold_disagreement_distribution   {dist}")
        expected_bin = {k: 16 * binomial_c4(k) for k in range(5)}
        print(f"  16*C(4,k) expected               {expected_bin}")
        bin_ok = all(dist.get(k, 0) == expected_bin[k] for k in range(5)) and sum(dist.values()) == 256
        print(f"  distribution total               {sum(dist.values())}")
        _check(gates, "fold disagreement = 16*C(4,k) and totals 256", bin_ok)
        print(f"  bu_boundary_disagreement_count   {finite['bu_boundary_disagreement_count']}")
        bu_frac = finite["bu_boundary_disagreement_count"] / 256.0
        print(f"  bu_boundary_fraction             {bu_frac:.16f}")
        print(f"  canonical W2 rest_ok             {finite['w2_rest_ok']}")
        print(f"  canonical W2' rest_ok            {finite['w2p_rest_ok']}")
        print(f"  W2 involution_ok                 {finite['w2_involution_ok']}")
        print(f"  T2 shell_ok                      {finite['t2_shell_ok']}")
        print(f"  T2 chirality_ok                  {finite['t2_chi_ok']}")
        print(f"  canonical certificate all_pass   {finite['k4_all_pass']}")
        _check(gates, "canonical W2/W2' finite holonomy certificate", bool(finite["k4_all_pass"]))
    print()

    # Cross-layer scale comparison
    _section("Q. CROSS-LAYER SCALE COMPARISON")
    print("  Cross-layer scale comparison; not an independent finite derivation of Delta.")
    print("  Finite kernel imports continuous aperture; endpoint is not re-derived here.")
    print(f"  BU_HOLONOMY_ANGLE                {delta_bu:.16f}")
    print(f"  Delta                            {delta_gap:.16f}")
    if bu_frac is not None:
        print(f"  byte BU-boundary fraction        {bu_frac:.16f}")
        print(f"  compression bu_frac / Delta      {bu_frac / delta_gap:.10f}")
    print()

    # Dictionary
    _section("R. CONTINUOUS-FINITE STRUCTURAL CORRESPONDENCE")
    print("  Rows identify analogous architectural roles.")
    print("  They do not assert equality of mathematical objects.")
    rows = [
        ("closed path in continuous model", "operator word on Omega"),
        ("holonomy angle / conjugacy class", "nontrivial finite involution or cycle"),
        ("BU dual-pole loop", "W2 pole exchange"),
        ("closure under return", "W2^2 = id"),
        ("palindromic payload path", "byte fold across BU boundary"),
        ("6 payload positions", "6 payload bits / 6 se(3) modes"),
        ("CS gauge frame", "byte bits 0 and 7 (family selector)"),
        ("continuous conjugacy spectrum", "finite involution spectrum (see wavefunction analysis)"),
    ]
    for left, right in rows:
        print(f"  {left:36s} -> {right}")
    print()
    print("  local curvature note:")
    print("    continuous: holonomy accumulates at ONA-BU corners; BU+->BU- is flat")
    print("    finite: fold disagreement is counted at the BU|BU boundary")
    print()

    # Status table
    _section("S. RESULT STATUS")
    status_rows = [
        ("theta_cs+theta_una+theta_ona = pi", "exact_algebraic", "threshold definitions"),
        ("q_g * m_a^2 = 1/2", "exact_algebraic", "threshold definitions"),
        ("phi_SU2 closed form", "exact_algebraic", "SU(2) threshold angles"),
        ("TW small-angle + convergence order", "standard_analytic+numerical", "GyroVectorSpace.gyration"),
        ("delta_BU analytic formula", "exact_under_declared_embedding", "theta_ona, m_a, orthogonal boosts"),
        ("matrix BU realization", "numerical_crosscheck", "declared embedding"),
        ("palindromic conjugacy", "gyrogroup_theorem+numerical", "Ungar inverse + SO(3) conjugation"),
        ("path reversal / cyclic re-rooting", "numerical_invariance", "SO(3) realization"),
        ("rho, Delta", "derived_definitions", "BU_HOLONOMY_ANGLE, m_a"),
        ("rho(m->0) expansion", "exact_under_declared_embedding", "analytic BU formula"),
        ("rank-2 dependency", "exact_under_declared_embedding", "delta_BU(theta_ona,m_a) only"),
        ("Wigner Jacobian sum=1 at threshold", "exact_algebraic", "beta=u_p, theta=pi/4"),
        ("canonical W2/W2' certificate", "exact_finite", "hQVM transition law"),
        ("fold distribution 16*C(4,k)", "exact_finite", "byte fold algebra"),
        ("continuous-finite dictionary", "structural_correspondence", "architecture analogy"),
        ("W = phi_SU2 - 3*delta_BU", "open_comparison", "no derivation claimed"),
        ("delta_BU vs 50/256", "open_comparison", "no derivation claimed"),
    ]
    print(f"  {'result':40s} {'status':32s} dependency")
    for name, status, dep in status_rows:
        print(f"  {name:40s} {status:32s} {dep}")
    print()

    # Summary
    _section("T. INTEGRITY CHECK SUMMARY")
    n_pass = sum(1 for _, ok in gates if ok)
    n_fail = sum(1 for _, ok in gates if not ok)
    for label, ok in gates:
        print(f"  check  {label:56s} {'PASS' if ok else 'FAIL'}")
    print(f"  passed={n_pass}  failed={n_fail}  total={len(gates)}")
    return gates


def main() -> None:
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = Tee(old, buf)
    try:
        print("CGM HOLONOMY ANALYSIS")
        print("=" * 5)
        print()
        gates = run()
    finally:
        sys.stdout = old

    RESULTS_PATH.write_text(buf.getvalue(), encoding="utf-8")
    print(f"wrote {RESULTS_PATH}")
    if any(not ok for _, ok in gates):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
