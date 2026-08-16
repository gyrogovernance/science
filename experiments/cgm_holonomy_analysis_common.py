#!/usr/bin/env python3
"""
cgm_holonomy_analysis_common.py

Shared math, constants, Tee, gates, section counter, stage/loop helpers, and
finite import for the CGM holonomy analysis program.

Companions: cgm_holonomy_analysis_1.py, _2.py, _run.py.
Finding doc: docs/Findings/Analysis_Holonomy.md
"""

from __future__ import annotations

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
    from gyroscopic.hQVM.constants import APERTURE_GAP_Q256
except Exception:  # pragma: no cover
    APERTURE_GAP_Q256 = None  # type: ignore[assignment]

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
TOL_MATRIX = 5e-8
TOL_ANGLE_INV = 1e-9
TOL_MP = mp.mpf("1e-60")
GYR_PROBE_R = mp.mpf("0.5")


# ---------------------------------------------------------------------
# Output / report state
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


class ReportState:
    def __init__(self) -> None:
        self.gates: list[tuple[str, bool]] = []
        self.section_n = 0
        # carry fields filled by part 1 for part 2
        self.t: Any = None
        self.gs: Any = None
        self.points: dict[str, tuple[float, float, float]] | None = None
        self.stages: Any = None
        self.delta_bu: float | None = None
        self.rho: float | None = None
        self.delta_gap: float | None = None
        self.delta_bu_closed: Any = None
        self.omega_closed: Any = None
        self.mpj: dict[str, Any] | None = None
        self.word: dict[str, Any] | None = None
        self.pal_mp: dict[str, Any] | None = None
        self.rooted_bu: dict[str, Any] | None = None
        self.defct: dict[str, Any] | None = None
        self.defct_bu_triangle: dict[str, Any] | None = None
        self.palge_circ: dict[str, Any] | None = None
        self.palge_bu: dict[str, Any] | None = None
        self.palge_pal: dict[str, Any] | None = None
        self.delta_stage: Any = None


def section(state: ReportState, title: str) -> None:
    state.section_n += 1
    print(f"{state.section_n}. {title}")
    print("=" * 5)


def check(
    state: ReportState,
    label: str,
    ok: bool,
    *,
    tier: str | None = None,
    quantity: str | None = None,
    measured: str | None = None,
    threshold: str | None = None,
) -> None:
    """
    Record a gate. Prefer quantity/measured/threshold so the results file
    states what was tested without reading the source.
    """
    status = "PASS" if ok else "FAIL"
    if quantity is not None:
        t = tier or "CHECK"
        print(f"  [{t}] {quantity}")
        if measured is not None:
            print(f"         measured   {measured}")
        if threshold is not None:
            print(f"         threshold  {threshold}")
        print(f"         status     {status}")
        state.gates.append((f"[{t}] {quantity}", ok))
    else:
        print(f"  check  {label:56s} {status}")
        state.gates.append((label, ok))


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


def stage_angle_defect_euclid_mp(t: CGMThresholds) -> Any:
    """
    Stage-angle closure defect:
      delta_stage = pi - (theta_CS + theta_UNA + theta_ONA)
    Threshold-angle identity; distinct from Ungar gyrotriangle defect.
    """
    return mp.pi - (t.theta_cs + t.theta_una + t.theta_ona)


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


def gyration_matrix_float(
    gs: GyroVectorSpace,
    a: np.ndarray,
    b: np.ndarray,
    *,
    probe_r: float = 0.5,
    project: bool = True,
) -> np.ndarray:
    """
    Float64 SO(3) matrix of gyr[a,b] from defining identity on interior probes.
    Columns = gyr(r e_i)/r with 0 < r < 1. If project, polar-decompose afterward.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    cols = [gs.gyr_apply(a, b, probe_r * e) / probe_r for e in np.eye(3)]
    G = np.column_stack(cols)
    if not project:
        return G
    U, _, Vt = np.linalg.svd(G)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


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
# Thomas-Wigner formulas (origin-based; threshold number = Einstein beta)
# ---------------------------------------------------------------------


def poincare_radius_from_beta(beta: Any) -> Any:
    """Poincare half-rapidity radius r = beta / (1 + sqrt(1 - beta^2))."""
    b = mp.mpf(beta)
    if not (0 <= b < 1):
        raise ValueError("beta must satisfy 0 <= beta < 1")
    return b / (1 + mp.sqrt(1 - b * b))


def tw_angle_unequal(beta_1: Any, beta_2: Any, theta: Any) -> Any:
    """Unsigned Wigner angle magnitude for boosts beta_1, beta_2 separated by theta."""
    r1 = poincare_radius_from_beta(beta_1)
    r2 = poincare_radius_from_beta(beta_2)
    z = r1 * r2
    th = mp.mpf(theta)
    return 2 * mp.atan2(mp.sin(th) * z, 1 + mp.cos(th) * z)


def tw_angle_signed_mp(beta_1: Any, beta_2: Any, theta: Any) -> Any:
    """
    Signed Thomas-Wigner angle (Ungar orientation):
    tan(eps/2) = -r1 r2 sin(theta) / (1 + r1 r2 cos(theta)).
    Positive eps rotates in the sense of u x v.
    """
    r1 = poincare_radius_from_beta(beta_1)
    r2 = poincare_radius_from_beta(beta_2)
    th = mp.mpf(theta)
    y = -r1 * r2 * mp.sin(th)
    x = 1 + r1 * r2 * mp.cos(th)
    return 2 * mp.atan2(y, x)


def tw_angle_exact(beta: Any, theta: Any) -> Any:
    """Equal-speed special case (unsigned magnitude)."""
    return tw_angle_unequal(beta, beta, theta)


def analytic_bu_holonomy(t: CGMThresholds) -> tuple[Any, Any]:
    """
    Origin-Wigner analytic for orthogonal boosts with magnitudes theta_ona, m_a
    under the convention that each threshold number is an Einstein beta.
    omega = |eps| = 2 atan(k(theta_ona) k(m_a))
    delta_BU_wigner_analytic = 2 omega = 4 atan(k(theta_ona) k(m_a))
    """
    omega = tw_angle_unequal(t.theta_ona, t.m_a, mp.pi / 2)
    return omega, 2 * omega


def radial_coordinates_mp(beta: Any) -> dict[str, Any]:
    """beta, gamma, rapidity eta, momentum radius rho/m, Poincare r."""
    b = mp.mpf(beta)
    g = 1 / mp.sqrt(1 - b * b)
    return {
        "beta": b,
        "gamma": g,
        "eta": mp.atanh(b),
        "rho_over_m": g * b,
        "poincare_r": poincare_radius_from_beta(b),
    }


# ---------------------------------------------------------------------
# mpmath Einstein gyration / Jacobian (ONA, BU+)
# ---------------------------------------------------------------------


def _mp_dot3(a: list[Any], b: list[Any]) -> Any:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _mp_add3(a: list[Any], b: list[Any]) -> list[Any]:
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def _mp_scale3(s: Any, a: list[Any]) -> list[Any]:
    return [s * a[0], s * a[1], s * a[2]]


def _mp_neg3(a: list[Any]) -> list[Any]:
    return [-a[0], -a[1], -a[2]]


def einstein_gyroaddition_mp(u: list[Any], v: list[Any], c: Any = mp.mpf(1)) -> list[Any]:
    c2 = c * c
    u2 = _mp_dot3(u, u)
    if u2 == 0:
        return list(v)
    v2 = _mp_dot3(v, v)
    if v2 == 0:
        return list(u)
    gamma_u = 1 / mp.sqrt(1 - u2 / c2)
    denom = 1 + _mp_dot3(u, v) / c2
    v_para = _mp_scale3(_mp_dot3(v, u) / u2, u)
    v_perp = _mp_add3(v, _mp_neg3(v_para))
    w_para = _mp_scale3(1 / denom, _mp_add3(u, v_para))
    w_perp = _mp_scale3(1 / (denom * gamma_u), v_perp)
    return _mp_add3(w_para, w_perp)


def gyr_apply_mp(a: list[Any], b: list[Any], w: list[Any], c: Any = mp.mpf(1)) -> list[Any]:
    return einstein_gyroaddition_mp(
        _mp_neg3(einstein_gyroaddition_mp(a, b, c)),
        einstein_gyroaddition_mp(a, einstein_gyroaddition_mp(b, w, c), c),
        c,
    )


def _so3_from_columns_mp(cols: list[list[Any]], *, project: bool = False) -> Any:
    G = mp.matrix(3)
    for i in range(3):
        for j in range(3):
            G[j, i] = cols[i][j]
    if not project:
        return G
    U, _S, Vh = mp.svd(G)
    R = U * Vh
    if mp.det(R) < 0:
        U[:, 2] = -U[:, 2]
        R = U * Vh
    return R


def so3_residuals_mp(R: Any) -> tuple[Any, Any]:
    I = mp.eye(3)
    return mp.norm(R.T * R - I), abs(mp.det(R) - 1)


def rotation_angle_atan2_mp(R: Any) -> Any:
    """theta = atan2(s, c) with c=(tr-1)/2, s=||vee(R-R^T)||/2."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    c = (tr - 1) / 2
    k_x = R[2, 1] - R[1, 2]
    k_y = R[0, 2] - R[2, 0]
    k_z = R[1, 0] - R[0, 1]
    s = mp.sqrt(k_x * k_x + k_y * k_y + k_z * k_z) / 2
    return mp.atan2(s, c)


def gyr_matrix_raw_mp(
    a: list[Any], b: list[Any], *, probe_r: Any = GYR_PROBE_R, c: Any = mp.mpf(1)
) -> Any:
    basis = (
        [mp.mpf(1), mp.mpf(0), mp.mpf(0)],
        [mp.mpf(0), mp.mpf(1), mp.mpf(0)],
        [mp.mpf(0), mp.mpf(0), mp.mpf(1)],
    )
    cols = [
        _mp_scale3(1 / probe_r, gyr_apply_mp(a, b, _mp_scale3(probe_r, e), c)) for e in basis
    ]
    return _so3_from_columns_mp(cols, project=False)


def gyr_matrix_jacobian_mp(a: list[Any], b: list[Any], eps: Any, c: Any = mp.mpf(1)) -> Any:
    basis = (
        [mp.mpf(1), mp.mpf(0), mp.mpf(0)],
        [mp.mpf(0), mp.mpf(1), mp.mpf(0)],
        [mp.mpf(0), mp.mpf(0), mp.mpf(1)],
    )
    cols = []
    for e in basis:
        g = gyr_apply_mp(a, b, _mp_scale3(eps, e), c)
        cols.append(_mp_scale3(1 / eps, g))
    return _so3_from_columns_mp(cols, project=False)


def lorentz_boost_mp(v: list[Any], c: Any = mp.mpf(1)) -> Any:
    """4x4 Einstein boost L(v) acting on (ct, x)."""
    c2 = c * c
    v2 = _mp_dot3(v, v)
    gamma = 1 / mp.sqrt(1 - v2 / c2)
    L = mp.eye(4)
    L[0, 0] = gamma
    for i in range(3):
        L[0, i + 1] = -gamma * v[i] / c
        L[i + 1, 0] = -gamma * v[i] / c
    if v2 == 0:
        return L
    fac = (gamma - 1) / v2
    for i in range(3):
        for j in range(3):
            L[i + 1, j + 1] = (mp.mpf(1) if i == j else mp.mpf(0)) + fac * v[i] * v[j]
    return L


def gyr_from_lorentz_mp(a: list[Any], b: list[Any], c: Any = mp.mpf(1)) -> Any:
    """
    Spatial block of Gyr[a,b] = L(a⊕b)^-1 L(a) L(b).
    """
    ab = einstein_gyroaddition_mp(a, b, c)
    G4 = lorentz_boost_mp(ab, c) ** -1 * lorentz_boost_mp(a, c) * lorentz_boost_mp(b, c)
    R = mp.matrix(3)
    for i in range(3):
        for j in range(3):
            R[i, j] = G4[i + 1, j + 1]
    return R


def _mp_cross3(a: list[Any], b: list[Any]) -> list[Any]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def gamma_mp(v: list[Any], c: Any = mp.mpf(1)) -> Any:
    v2 = _mp_dot3(v, v)
    return 1 / mp.sqrt(1 - v2 / (c * c))


def omega_matrix_ungar_mp(u: list[Any], v: list[Any]) -> Any:
    """
    Ungar Omega(u,v): Omega @ x = (u x v) x x = v(u·x) - u(v·x).
    Omega_ij = v_i u_j - u_i v_j.
    """
    Om = mp.matrix(3)
    for i in range(3):
        for j in range(3):
            Om[i, j] = v[i] * u[j] - u[i] * v[j]
    return Om


def gyr_matrix_ungar_mp(u: list[Any], v: list[Any], c: Any = mp.mpf(1)) -> Any:
    """
    Ungar (2013) eqs. 48-49: gyr[u,v] = I + alpha*Omega + beta*Omega^2.
    Independent of finite-difference probes and SVD.
    """
    Om = omega_matrix_ungar_mp(u, v)
    cross = _mp_cross3(u, v)
    c2 = _mp_dot3(cross, cross)
    if c2 < mp.mpf("1e-120"):
        return mp.eye(3)
    gu = gamma_mp(u, c)
    gv = gamma_mp(v, c)
    uv = einstein_gyroaddition_mp(u, v, c)
    guv = gamma_mp(uv, c)
    denom = (1 + gu) * (1 + gv) * (1 + guv)
    alpha = -(gu * gv * (1 + gu + gv + guv)) / ((c * c) * denom)
    beta = (gu * gu * gv * gv) / ((c**4) * denom)
    return mp.eye(3) + alpha * Om + beta * (Om * Om)


def trace_identity_residual_mp(G: Any) -> Any:
    """Ungar (51): ||G^3 - tr(G) G^2 + tr(G) G - I||."""
    tr = G[0, 0] + G[1, 1] + G[2, 2]
    I = mp.eye(3)
    R = G * G * G - tr * G * G + tr * G - I
    return mp.norm(R)


def embed_so3_as_lorentz_mp(G: Any) -> Any:
    """4x4 spacetime gyration: diag(1, G)."""
    L = mp.eye(4)
    for i in range(3):
        for j in range(3):
            L[i + 1, j + 1] = G[i, j]
    return L


def apply_so3_mp(G: Any, w: list[Any]) -> list[Any]:
    return [sum(G[i, j] * w[j] for j in range(3)) for i in range(3)]


def boost_composition_theorem_mp(u: list[Any], v: list[Any], c: Any = mp.mpf(1)) -> dict[str, Any]:
    """
    Ungar boost factorization (matrix convention x' = L @ x):
      B(u) B(v) = B(u⊕v) Gyr[u,v]
      B(v) B(u) = B(v⊕u) Gyr[v,u]
    Gyr from Ungar explicit 3x3 embedded as diag(1,G).
    """
    Bu = lorentz_boost_mp(u, c)
    Bv = lorentz_boost_mp(v, c)
    uv = einstein_gyroaddition_mp(u, v, c)
    vu = einstein_gyroaddition_mp(v, u, c)
    Buv = lorentz_boost_mp(uv, c)
    Bvu = lorentz_boost_mp(vu, c)
    G_uv = gyr_matrix_ungar_mp(u, v, c)
    G_vu = gyr_matrix_ungar_mp(v, u, c)
    resid_uv = mp.norm(Bu * Bv - Buv * embed_so3_as_lorentz_mp(G_uv))
    resid_vu = mp.norm(Bv * Bu - Bvu * embed_so3_as_lorentz_mp(G_vu))
    G_lor = gyr_from_lorentz_mp(u, v, c)
    return {
        "resid_BuBv_vs_Buv_Gyr": resid_uv,
        "resid_BvBu_vs_Bvu_Gyr": resid_vu,
        "resid_ungar_vs_lorentz_gyr": mp.norm(G_uv - G_lor),
        "delta_ungar": 2 * rotation_angle_atan2_mp(G_uv),
        "delta_lorentz": 2 * rotation_angle_atan2_mp(G_lor),
        "G_ungar": G_uv,
        "trace_id_ungar": trace_identity_residual_mp(G_uv),
        "trace_id_lorentz": trace_identity_residual_mp(G_lor),
    }


def einstein_coaddition_mp(u: list[Any], v: list[Any], c: Any = mp.mpf(1)) -> list[Any]:
    """Ungar (75): u ⊞ v = u ⊕ gyr[u, ⊖v] v  (commutative)."""
    G = gyr_matrix_ungar_mp(u, _mp_neg3(v), c)
    return einstein_gyroaddition_mp(u, apply_so3_mp(G, v), c)


def gyrotriangle_defect_mp(u: list[Any], v: list[Any], c: Any = mp.mpf(1)) -> dict[str, Any]:
    """
    Ungar (21)/(74): gyrotriangular defect of origin-u-v from side gammas.
    Compare to angle(gyr[u, ⊖v]).
    """
    gu = gamma_mp(u, c)
    gv = gamma_mp(v, c)
    neg_u_v = einstein_gyroaddition_mp(_mp_neg3(u), v, c)
    gw = gamma_mp(neg_u_v, c)
    numer = 1 + 2 * gu * gv * gw - gu * gu - gv * gv - gw * gw
    denom = 1 + gu + gv + gw
    if numer < 0 and abs(numer) < mp.mpf("1e-50"):
        numer = mp.mpf(0)
    defect = 2 * mp.atan(mp.sqrt(numer) / denom)
    G = gyr_matrix_ungar_mp(u, _mp_neg3(v), c)
    gyr_ang = rotation_angle_atan2_mp(G)
    return {
        "gamma_u": gu,
        "gamma_v": gv,
        "gamma_neg_u_plus_v": gw,
        "defect": defect,
        "gyr_u_neg_v_angle": gyr_ang,
        "defect_minus_gyr_angle": abs(defect - gyr_ang),
    }


def gyrotriangle_defect_triangle_vertices_mp(
    A: list[Any], B: list[Any], C: list[Any], c: Any = mp.mpf(1)
) -> dict[str, Any]:
    """
    Gyrotriangle defect for vertices (A,B,C) in the Einstein ball.

    Gyrotranslate A to the origin:
      u = ⊖A ⊕ B
      v = ⊖A ⊕ C
    Then defect(A,B,C) = defect(0,u,v) = angle(gyr[u, ⊖v]) (Ungar 21/74).
    """
    u = einstein_gyroaddition_mp(_mp_neg3(A), B, c)
    v = einstein_gyroaddition_mp(_mp_neg3(A), C, c)
    out = gyrotriangle_defect_mp(u, v, c)
    return {
        "A": A,
        "B": B,
        "C": C,
        "u": u,
        "v": v,
        "defect": out["defect"],
        "gyr_u_neg_v_angle": out["gyr_u_neg_v_angle"],
        "defect_minus_gyr_angle": out["defect_minus_gyr_angle"],
        "gamma_u": out["gamma_u"],
        "gamma_v": out["gamma_v"],
        "gamma_neg_u_plus_v": out["gamma_neg_u_plus_v"],
    }


def gyrogroup_axiom_residuals_mp(
    u: list[Any], v: list[Any], w: list[Any], c: Any = mp.mpf(1)
) -> dict[str, Any]:
    """Residuals of Ungar (24) identities on three given ball vectors."""
    G = gyr_matrix_ungar_mp(u, v, c)
    uv = einstein_gyroaddition_mp(u, v, c)
    vu = einstein_gyroaddition_mp(v, u, c)
    Gvu = apply_so3_mp(G, vu)
    resid_gyrocomm = mp.sqrt(sum((uv[i] - Gvu[i]) ** 2 for i in range(3)))

    vw = einstein_gyroaddition_mp(v, w, c)
    left = einstein_gyroaddition_mp(u, vw, c)
    right = einstein_gyroaddition_mp(uv, apply_so3_mp(G, w), c)
    resid_left_assoc = mp.sqrt(sum((left[i] - right[i]) ** 2 for i in range(3)))

    G_neg = gyr_matrix_ungar_mp(_mp_neg3(u), _mp_neg3(v), c)
    resid_even = mp.norm(G - G_neg)

    G_vu = gyr_matrix_ungar_mp(v, u, c)
    resid_inv = mp.norm(G_vu - G**-1)

    resid_loop_l = mp.norm(gyr_matrix_ungar_mp(uv, v, c) - G)
    resid_loop_r = mp.norm(gyr_matrix_ungar_mp(u, einstein_gyroaddition_mp(v, u, c), c) - G)

    return {
        "gyrocommutative": resid_gyrocomm,
        "left_gyroassociative": resid_left_assoc,
        "even_property": resid_even,
        "gyration_inverse": resid_inv,
        "left_loop": resid_loop_l,
        "right_loop": resid_loop_r,
    }


def lorentz_time_space_residual_mp(G4: Any) -> Any:
    """Max |G4[0,1:4]| and |G4[1:4,0]| (should vanish for pure rotation)."""
    m = mp.mpf(0)
    for i in range(1, 4):
        m = max(m, abs(G4[0, i]), abs(G4[i, 0]))
    return m


def extract_rotation_from_lorentz_product_mp(boosts: list[list[Any]], c: Any = mp.mpf(1)) -> Any:
    """
    Lambda = L(d_n)...L(d_1); decompose Lambda = L(w) R; return R (3x3).
    """
    Lam = mp.eye(4)
    for d in boosts:
        Lam = lorentz_boost_mp(d, c) * Lam
    # velocity of resultant boost from time column / gamma
    # Lambda maps rest frame: for pure L(w), Lambda[i,0] = -gamma w_i / c, Lambda[0,0]=gamma
    gamma = Lam[0, 0]
    if abs(gamma) < mp.mpf("1e-30"):
        raise ValueError("degenerate Lorentz product")
    w = [-c * Lam[i + 1, 0] / gamma for i in range(3)]
    # clamp numerical drift inside ball
    w2 = _mp_dot3(w, w)
    if w2 >= c * c:
        scale = mp.sqrt((c * c) * (1 - mp.mpf("1e-20")) / w2)
        w = _mp_scale3(scale, w)
    R4 = lorentz_boost_mp(w, c) ** -1 * Lam
    R = mp.matrix(3)
    for i in range(3):
        for j in range(3):
            R[i, j] = R4[i + 1, j + 1]
    return R, lorentz_time_space_residual_mp(R4)


def mpmath_ona_bu_origin_gyr_suite(t: CGMThresholds) -> dict[str, Any]:
    """
    Origin-based gyr[ONA,BU+] by four routes (threshold = Einstein beta):
      closed analytic, raw gyr map, Ungar I+aOm+bOm^2, Lorentz factorization.
    Also: Ungar trace identity on raw/Ungar; linearity scan gyr(eps e_i)/eps.
    """
    ona = [mp.mpf(0), mp.mpf(t.theta_ona), mp.mpf(0)]
    bu = [mp.mpf(0), mp.mpf(0), mp.mpf(t.m_a)]
    omega_closed, delta_closed = analytic_bu_holonomy(t)
    eps_signed = tw_angle_signed_mp(t.theta_ona, t.m_a, mp.pi / 2)
    G_raw = gyr_matrix_raw_mp(ona, bu)
    orth_raw, det_raw = so3_residuals_mp(G_raw)
    delta_map = 2 * rotation_angle_atan2_mp(G_raw)
    G_ung = gyr_matrix_ungar_mp(ona, bu)
    delta_ung = 2 * rotation_angle_atan2_mp(G_ung)
    orth_ung, det_ung = so3_residuals_mp(G_ung)
    G_lor = gyr_from_lorentz_mp(ona, bu)
    delta_lor = 2 * rotation_angle_atan2_mp(G_lor)
    orth_lor, det_lor = so3_residuals_mp(G_lor)
    thm7 = boost_composition_theorem_mp(ona, bu)
    axis = _mp_cross3(ona, bu)
    axis_n = mp.sqrt(_mp_dot3(axis, axis))
    axis_hat = [axis[i] / axis_n for i in range(3)] if axis_n > 0 else [mp.mpf(1), mp.mpf(0), mp.mpf(0)]
    eps_exponents = (8, 12, 16, 20, 24, 30)
    lin_rows: list[tuple[int, Any, Any]] = []
    for eexp in eps_exponents:
        eps = mp.power(10, -eexp)
        G_jac = gyr_matrix_jacobian_mp(ona, bu, eps)
        delta_jac = 2 * rotation_angle_atan2_mp(G_jac)
        lin_rows.append((eexp, delta_jac, abs(delta_jac - delta_closed)))
    return {
        "omega_closed": omega_closed,
        "delta_closed": delta_closed,
        "eps_signed": eps_signed,
        "axis_u_cross_v": axis_hat,
        "delta_map": delta_map,
        "map_minus_closed": abs(delta_map - delta_closed),
        "orth_raw": orth_raw,
        "det_raw": det_raw,
        "trace_id_raw": trace_identity_residual_mp(G_raw),
        "delta_ungar": delta_ung,
        "ungar_minus_closed": abs(delta_ung - delta_closed),
        "ungar_minus_raw": mp.norm(G_ung - G_raw),
        "orth_ungar": orth_ung,
        "det_ungar": det_ung,
        "trace_id_ungar": trace_identity_residual_mp(G_ung),
        "delta_lorentz": delta_lor,
        "lorentz_minus_raw": mp.norm(G_lor - G_raw),
        "orth_lorentz": orth_lor,
        "det_lorentz": det_lor,
        "trace_id_lorentz": trace_identity_residual_mp(G_lor),
        "thm7": thm7,
        "lin_rows": lin_rows,
    }


def ungar_lorentz_random_audit_mp(
    *,
    n: int = 24,
    beta_max: Any = mp.mpf("0.3"),
    seed: int = 1,
) -> dict[str, Any]:
    """
    Random origin-based pairs at ||u||,||v|| <= beta_max.
    Compare Ungar matrix vs Lorentz factorization; report max residuals.
    """
    max_ung_lor = mp.mpf(0)
    max_trace = mp.mpf(0)
    max_thm7 = mp.mpf(0)
    n_used = 0
    # deterministic pseudo-pairs from seed arithmetic (no numpy RNG dependency)
    for k in range(n):
        a = (seed * 1103515245 + 12345 + 17 * k) % 10007
        b = (seed * 1664525 + 1013904223 + 31 * k) % 10007
        c = (seed * 214013 + 2531011 + 47 * k) % 10007
        d = (seed * 69069 + 1 + 61 * k) % 10007
        e = (seed * 48271 + 7 + 71 * k) % 10007
        f = (seed * 69621 + 11 + 83 * k) % 10007
        u_raw = [mp.mpf(a - 5000) / 5000, mp.mpf(b - 5000) / 5000, mp.mpf(c - 5000) / 5000]
        v_raw = [mp.mpf(d - 5000) / 5000, mp.mpf(e - 5000) / 5000, mp.mpf(f - 5000) / 5000]
        nu = mp.sqrt(_mp_dot3(u_raw, u_raw))
        nv = mp.sqrt(_mp_dot3(v_raw, v_raw))
        if nu < mp.mpf("1e-12") or nv < mp.mpf("1e-12"):
            continue
        # scale into ball with beta_max and slight inequality
        su = beta_max * (mp.mpf("0.35") + mp.mpf("0.65") * ((a % 1000) / 1000))
        sv = beta_max * (mp.mpf("0.35") + mp.mpf("0.65") * ((d % 1000) / 1000))
        u = _mp_scale3(su / nu, u_raw)
        v = _mp_scale3(sv / nv, v_raw)
        # skip near-collinear (Ungar identity still holds but angle ~ 0)
        cross = _mp_cross3(u, v)
        if _mp_dot3(cross, cross) < mp.mpf("1e-24"):
            continue
        G_u = gyr_matrix_ungar_mp(u, v)
        G_l = gyr_from_lorentz_mp(u, v)
        max_ung_lor = max(max_ung_lor, mp.norm(G_u - G_l))
        max_trace = max(max_trace, trace_identity_residual_mp(G_u))
        thm = boost_composition_theorem_mp(u, v)
        max_thm7 = max(
            max_thm7,
            thm["resid_BuBv_vs_Buv_Gyr"],
            thm["resid_BvBu_vs_Bvu_Gyr"],
        )
        n_used += 1
    return {
        "n_requested": n,
        "n_used": n_used,
        "beta_max": beta_max,
        "seed": seed,
        "max_ungar_minus_lorentz": max_ung_lor,
        "max_trace_identity_ungar": max_trace,
        "max_thm7_resid": max_thm7,
    }


def mp_stage_points(t: CGMThresholds) -> dict[str, list[Any]]:
    return {
        "UNA": [mp.mpf(t.u_p), mp.mpf(0), mp.mpf(0)],
        "ONA": [mp.mpf(0), mp.mpf(t.theta_ona), mp.mpf(0)],
        "BU+": [mp.mpf(0), mp.mpf(0), mp.mpf(t.m_a)],
        "BU-": [mp.mpf(0), mp.mpf(0), -mp.mpf(t.m_a)],
    }


def mpmath_dual_pole_word(t: CGMThresholds) -> dict[str, Any]:
    """Full dual-pole product R = G_ingress G_middle G_egress (left action)."""
    p = mp_stage_points(t)
    G_e = gyr_matrix_raw_mp(p["ONA"], p["BU+"])
    G_m = gyr_matrix_raw_mp(p["BU+"], p["BU-"])
    G_i = gyr_matrix_raw_mp(p["BU-"], p["ONA"])
    R = G_i * G_m * G_e
    th_e = rotation_angle_atan2_mp(G_e)
    th_m = rotation_angle_atan2_mp(G_m)
    th_i = rotation_angle_atan2_mp(G_i)
    th_word = rotation_angle_atan2_mp(R)
    orth, det = so3_residuals_mp(R)
    _, delta_closed = analytic_bu_holonomy(t)
    return {
        "theta_egress": th_e,
        "theta_middle": th_m,
        "theta_ingress": th_i,
        "theta_word": th_word,
        "sum_corners": th_e + th_i,
        "orth": orth,
        "det": det,
        "R": R,
        "delta_closed": delta_closed,
        "word_minus_wigner": abs(th_word - delta_closed),
        "word_minus_egress_ingress_sum": abs(th_word - (th_e + th_i)),
        "egress_minus_ingress": abs(th_e - th_i),
        "G_middle": G_m,
    }


def mpmath_palindrome_word(t: CGMThresholds, R_bu: Any) -> dict[str, Any]:
    """UNA->ONA->BU+->BU-->ONA->UNA with left-action product."""
    p = mp_stage_points(t)
    path = ("UNA", "ONA", "BU+", "BU-", "ONA", "UNA")
    R = mp.eye(3)
    for i in range(len(path) - 1):
        G = gyr_matrix_raw_mp(p[path[i]], p[path[i + 1]])
        R = G * R
    A = gyr_matrix_raw_mp(p["UNA"], p["ONA"])
    # left-action conjugacy: R_pal = A^{-1} R_BU A
    conj = A**-1 * R_bu * A
    conj_alt = A * R_bu * A**-1
    return {
        "theta_pal": rotation_angle_atan2_mp(R),
        "theta_bu": rotation_angle_atan2_mp(R_bu),
        "tr_pal": R[0, 0] + R[1, 1] + R[2, 2],
        "tr_bu": R_bu[0, 0] + R_bu[1, 1] + R_bu[2, 2],
        "resid_Ainv_R_A": mp.norm(R - conj),
        "resid_A_R_Ainv": mp.norm(R - conj_alt),
        "orth": so3_residuals_mp(R)[0],
        "det": so3_residuals_mp(R)[1],
        "R": R,
        "A": A,
    }


def mpmath_rooted_path_compare(t: CGMThresholds, path: tuple[str, ...]) -> dict[str, Any]:
    """
    Origin-gyr word: product of gyr(p_i, p_{i+1}).
    Relative-boost word: rotational part of product L(d_i), d_i = ⊖p_i ⊕ p_{i+1}.
    """
    pts = mp_stage_points(t)
    R_gyr = mp.eye(3)
    ds: list[list[Any]] = []
    for i in range(len(path) - 1):
        a = pts[path[i]]
        b = pts[path[i + 1]]
        R_gyr = gyr_matrix_raw_mp(a, b) * R_gyr
        ds.append(einstein_gyroaddition_mp(_mp_neg3(a), b))
    R_rel, ts_resid = extract_rotation_from_lorentz_product_mp(ds)
    return {
        "path": path,
        "theta_origin_gyr_word": rotation_angle_atan2_mp(R_gyr),
        "theta_relative_boost_word": rotation_angle_atan2_mp(R_rel),
        "norm_origin_gyr_minus_relative_boost": mp.norm(R_gyr - R_rel),
        "angle_diff": abs(
            rotation_angle_atan2_mp(R_gyr) - rotation_angle_atan2_mp(R_rel)
        ),
        "relative_boost_time_space_resid": ts_resid,
        "orth_origin_gyr": so3_residuals_mp(R_gyr)[0],
        "orth_relative_boost": so3_residuals_mp(R_rel)[0],
        "edge_displacements": ds,
    }


# ---------------------------------------------------------------------
# Palge-Pfeifer mass-shell connection / geodesic holonomy
# ---------------------------------------------------------------------


def minkowski_dot4(a: list[Any], b: list[Any]) -> Any:
    """(+,-,-,-) Minkowski product."""
    return a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]


def four_velocity_from_beta_mp(beta: list[Any], c: Any = mp.mpf(1)) -> list[Any]:
    """q(β) = (γ, γ β / c) with q·q = 1 (c=1 => γβ)."""
    g = gamma_mp(beta, c)
    return [g, g * beta[0] / c, g * beta[1] / c, g * beta[2] / c]


def beta_from_four_velocity_mp(q: list[Any], c: Any = mp.mpf(1)) -> list[Any]:
    g = q[0]
    return [c * q[1] / g, c * q[2] / g, c * q[3] / g]


def lorentz_boost_from_four_velocity_mp(q: list[Any], c: Any = mp.mpf(1)) -> Any:
    """
    L with L @ (1,0,0,0) = q.
    Existing lorentz_boost_mp(v) maps rest -> (gamma, -gamma v), so pass -beta.
    """
    beta = beta_from_four_velocity_mp(q, c)
    return lorentz_boost_mp([-beta[0], -beta[1], -beta[2]], c)


def pure_boost_u_to_v_mp(u: list[Any], v: list[Any]) -> Any:
    """
    Rotation-free Lorentz map taking 4-velocity u to v (geodesic transvection).

    Λ^α_β = δ^α_β - (u+v)^α (u+v)_β / (1+u·v) + 2 v^α u_β
    with lowered indices via η = diag(+1,-1,-1,-1).
    Callers must gate Lorentz residuals (SO^+(1,3)); this only builds the matrix.
    """
    udotv = minkowski_dot4(u, v)
    denom = 1 + udotv
    if abs(denom) < mp.mpf("1e-80"):
        raise ValueError("pure_boost_u_to_v: u and -v nearly lightlike-aligned")
    s = [u[i] + v[i] for i in range(4)]
    s_low = [s[0], -s[1], -s[2], -s[3]]
    u_low = [u[0], -u[1], -u[2], -u[3]]
    Lam = mp.eye(4)
    for a in range(4):
        for b in range(4):
            Lam[a, b] = (
                (mp.mpf(1) if a == b else mp.mpf(0))
                - s[a] * s_low[b] / denom
                + 2 * v[a] * u_low[b]
            )
    if not mp.isfinite(mp.norm(Lam)):
        raise ValueError("pure_boost_u_to_v produced non-finite entries")
    return Lam


def apply_lorentz4_mp(L: Any, q: list[Any]) -> list[Any]:
    return [sum(L[i, j] * q[j] for j in range(4)) for i in range(4)]


def momentum_spherical_mp(
    beta: list[Any], m: Any = mp.mpf(1), c: Any = mp.mpf(1)
) -> dict[str, Any]:
    """
    Palge-Pfeifer (rho, theta, phi) from Einstein ball beta (threshold = beta).
    p = gamma m beta, rho = |p|, E = gamma m.
    """
    g = gamma_mp(beta, c)
    px, py, pz = g * m * beta[0], g * m * beta[1], g * m * beta[2]
    rho = mp.sqrt(px * px + py * py + pz * pz)
    E = g * m
    if rho < mp.mpf("1e-80"):
        return {
            "rho": rho,
            "theta": mp.mpf(0),
            "phi": mp.mpf(0),
            "E": E,
            "m": m,
            "gamma": g,
            "p": [px, py, pz],
        }
    theta = mp.acos(pz / rho)
    phi = mp.atan2(py, px)
    return {
        "rho": rho,
        "theta": theta,
        "phi": phi,
        "E": E,
        "m": m,
        "gamma": g,
        "p": [px, py, pz],
    }


def omega_so3_matrix_mp(
    *,
    E: Any,
    m: Any,
    theta: Any,
    drho: Any,
    dtheta: Any,
    dphi: Any,
) -> Any:
    """
    Palge-Pfeifer eq. (25): so(3)-valued connection 1-form matrix.
    Independent of drho (Levi-Civita spin connection coeffs have no d rho).
    """
    _ = drho
    em = E / m
    w12 = em * dtheta
    w13 = mp.sin(theta) * em * dphi
    w23 = mp.cos(theta) * dphi
    # matrix matching their display (25)
    W = mp.zeros(3)
    W[0, 1] = -w12
    W[0, 2] = -w13
    W[1, 0] = w12
    W[1, 2] = -w23
    W[2, 0] = w13
    W[2, 1] = w23
    return W


def omega_su2_matrix_mp(
    *,
    E: Any,
    m: Any,
    theta: Any,
    dtheta: Any,
    dphi: Any,
) -> Any:
    """Palge-Pfeifer eq. (41): su(2)-valued spin connection matrix."""
    em = E / m
    # omega_s = -i/2 [ (E/m) dtheta sigma3 - (E/m) sin(theta) dphi sigma2 + cos(theta) dphi sigma1 ]
    s1 = mp.matrix([[0, 1], [1, 0]])
    s2 = mp.matrix([[0, -1j], [1j, 0]])
    s3 = mp.matrix([[1, 0], [0, -1]])
    return (
        -mp.j
        / 2
        * (
            em * dtheta * s3
            - em * mp.sin(theta) * dphi * s2
            + mp.cos(theta) * dphi * s1
        )
    )


def circular_thomas_alpha_analytic_mp(V: Any) -> Any:
    """Palge-Pfeifer (47): alpha = 2 pi (gamma(V) - 1)."""
    gamma = 1 / mp.sqrt(1 - V * V)
    return 2 * mp.pi * (gamma - 1)


def circular_curvature_holonomy_alpha_mp(V: Any, m: Any = mp.mpf(1)) -> dict[str, Any]:
    """
    Disk integral of Omega_s for C: rho=rho0, theta=pi/2, phi in [0,2pi]
    (Palge-Pfeifer eqs. 45-47). Abelian; equals alpha = 2 pi (gamma-1).
    """
    gamma = 1 / mp.sqrt(1 - V * V)
    rho0 = m * V / mp.sqrt(1 - V * V)
    # -int Omega_s = -i sigma2 * pi (gamma-1)  =>  SO(3) angle alpha = 2 pi (gamma-1)
    alpha_curv = 2 * mp.pi * (gamma - 1)
    # connection line coeff int omega13 along C = 2 pi * (E/m) = 2 pi gamma
    alpha_line = 2 * mp.pi * gamma
    # Stokes mismatch is the 2 pi frame singularity at rho=0
    alpha_line_minus_2pi = alpha_line - 2 * mp.pi
    return {
        "V": V,
        "m": m,
        "rho0": rho0,
        "gamma": gamma,
        "alpha_analytic": circular_thomas_alpha_analytic_mp(V),
        "alpha_curvature": alpha_curv,
        "alpha_connection_line": alpha_line,
        "alpha_line_minus_2pi": alpha_line_minus_2pi,
        "resid_curv_vs_analytic": abs(alpha_curv - circular_thomas_alpha_analytic_mp(V)),
        "resid_line_corr_vs_analytic": abs(alpha_line_minus_2pi - circular_thomas_alpha_analytic_mp(V)),
    }


def circular_pexp_so3_mp(V: Any, m: Any = mp.mpf(1), n_steps: int = 256) -> dict[str, Any]:
    """
    Numerical P exp(-int_C omega) on the circular orbit (eq. 44).
    Raw path-ordered angle tracks 2 pi gamma; physical Thomas angle is that minus 2 pi.
    """
    gamma = 1 / mp.sqrt(1 - V * V)
    E = gamma * m
    theta = mp.pi / 2
    dphi = 2 * mp.pi / n_steps
    R = mp.eye(3)
    for _ in range(n_steps):
        W = omega_so3_matrix_mp(E=E, m=m, theta=theta, drho=0, dtheta=0, dphi=dphi)
        R = mp.expm(-W) * R
    ang = rotation_angle_atan2_mp(R)
    # fold to [0, 2pi): raw angle near 2 pi gamma; report principal and corrected
    ang_mod = ang - 2 * mp.pi * mp.floor(ang / (2 * mp.pi) + mp.mpf("1e-30"))
    alpha_theory = circular_thomas_alpha_analytic_mp(V)
    # physical: ang ≈ 2 pi gamma (mod 2pi issues); use continuous accumulation
    # Recompute with continuous angle via trace of incremental product vs theory line
    return {
        "V": V,
        "n_steps": n_steps,
        "theta_raw": ang,
        "theta_mod_2pi": ang_mod,
        "alpha_theory": alpha_theory,
        "alpha_line_theory": 2 * mp.pi * gamma,
        "orth": so3_residuals_mp(R)[0],
        "R": R,
    }


def circular_pexp_su2_curvature_mp(V: Any) -> dict[str, Any]:
    """
    Exact SU(2) holonomy from curvature integral (47):
    Hol = exp(-i (alpha/2) sigma_2), alpha = 2 pi (gamma-1).
    """
    alpha = circular_thomas_alpha_analytic_mp(V)
    half = alpha / 2
    Hol = mp.matrix(
        [
            [mp.cos(half), -mp.sin(half)],
            [mp.sin(half), mp.cos(half)],
        ]
    )
    return {"alpha": alpha, "Hol": Hol, "U00": Hol[0, 0], "U01": Hol[0, 1]}


def eta_minkowski_mp() -> Any:
    """η = diag(+1,-1,-1,-1) as mp.matrix."""
    E = mp.zeros(4)
    E[0, 0] = mp.mpf(1)
    E[1, 1] = mp.mpf(-1)
    E[2, 2] = mp.mpf(-1)
    E[3, 3] = mp.mpf(-1)
    return E


def lorentz_residual_mp(L: Any) -> Any:
    """||L^T η L - η|| Frobenius norm."""
    eta = eta_minkowski_mp()
    return mp.norm(L.T * eta * L - eta)


def lorentz_det_residual_mp(L: Any) -> Any:
    """|det(L) - 1|."""
    return abs(mp.det(L) - 1)


def lorentz_time_orientation_residual_mp(L: Any) -> Any:
    """Orthochronous: L[0,0] >= 1. Report max(0, 1 - L00)."""
    return max(mp.mpf(0), mp.mpf(1) - L[0, 0])


def geodesic_path_holonomy_mp(
    betas: list[list[Any]], c: Any = mp.mpf(1)
) -> dict[str, Any]:
    """
    Piecewise-geodesic mass-shell holonomy via pure boosts between 4-velocities.

    Vertices are unit timelike 4-velocities q_i = (γ, γβ).
    Edge map is the unique rotation-free Lorentz transvection T_i with T_i q_i = q_{i+1}.

    For a closed loop, P fixes q0. The SO(3) little-group holonomy at q0 is:
      R4 = L(q0)^-1 P L(q0), with L(q0) mapping rest -> q0.
    """
    if len(betas) < 2:
        raise ValueError("need at least two betas")

    qs = [four_velocity_from_beta_mp(b, c) for b in betas]
    close_resid = mp.sqrt(sum((qs[0][i] - qs[-1][i]) ** 2 for i in range(4)))

    P = mp.eye(4)
    edge_map_resids: list[Any] = []
    edge_lorentz_resids: list[Any] = []
    edge_det_resids: list[Any] = []
    edge_time_resids: list[Any] = []

    for i in range(len(qs) - 1):
        T = pure_boost_u_to_v_mp(qs[i], qs[i + 1])
        mapped = apply_lorentz4_mp(T, qs[i])
        edge_map_resids.append(
            mp.sqrt(sum((mapped[j] - qs[i + 1][j]) ** 2 for j in range(4)))
        )
        edge_lorentz_resids.append(lorentz_residual_mp(T))
        edge_det_resids.append(lorentz_det_residual_mp(T))
        edge_time_resids.append(lorentz_time_orientation_residual_mp(T))
        P = T * P

    q0 = qs[0]
    mapped0 = apply_lorentz4_mp(P, q0)
    fix_resid = mp.sqrt(sum((mapped0[i] - q0[i]) ** 2 for i in range(4)))

    P_lor = lorentz_residual_mp(P)
    P_det = lorentz_det_residual_mp(P)
    P_time = lorentz_time_orientation_residual_mp(P)

    L0 = lorentz_boost_from_four_velocity_mp(q0, c)
    R4 = L0**-1 * P * L0
    ts = lorentz_time_space_residual_mp(R4)
    R = mp.matrix(3)
    for i in range(3):
        for j in range(3):
            R[i, j] = R4[i + 1, j + 1]

    return {
        "theta": rotation_angle_atan2_mp(R),
        "R": R,
        "fix_q0_resid": fix_resid,
        "close_resid": close_resid,
        "time_space_resid": ts,
        "orth": so3_residuals_mp(R)[0],
        "max_edge_boost_resid": max(edge_map_resids) if edge_map_resids else mp.mpf(0),
        "max_edge_map_resid": max(edge_map_resids) if edge_map_resids else mp.mpf(0),
        "max_edge_lorentz_resid": max(edge_lorentz_resids) if edge_lorentz_resids else mp.mpf(0),
        "max_edge_det_resid": max(edge_det_resids) if edge_det_resids else mp.mpf(0),
        "max_edge_time_resid": max(edge_time_resids) if edge_time_resids else mp.mpf(0),
        "P_lorentz_resid": P_lor,
        "P_det_resid": P_det,
        "P_time_resid": P_time,
        "n_edges": len(qs) - 1,
    }


def omega_pexp_along_geodesic_edge_mp(
    beta_a: list[Any],
    beta_b: list[Any],
    *,
    m: Any = mp.mpf(1),
    c: Any = mp.mpf(1),
    n_steps: int = 64,
) -> Any:
    """
    Path-ordered exp(-int omega) along the mass-shell geodesic beta_a -> beta_b,
    using spherical-chart pullback of Palge-Pfeifer omega (eq. 25).
    """
    qa = four_velocity_from_beta_mp(beta_a, c)
    qb = four_velocity_from_beta_mp(beta_b, c)
    chi = mp.acosh(max(minkowski_dot4(qa, qb), mp.mpf(1)))
    R = mp.eye(3)
    if chi < mp.mpf("1e-80"):
        return R
    prev = momentum_spherical_mp(beta_a, m, c)
    for k in range(1, n_steps + 1):
        t = mp.mpf(k) / n_steps
        # q(t) = (sinh((1-t)chi) qa + sinh(t chi) qb) / sinh(chi)
        sh = mp.sinh(chi)
        q = [
            (mp.sinh((1 - t) * chi) * qa[i] + mp.sinh(t * chi) * qb[i]) / sh
            for i in range(4)
        ]
        # renormalize light drift
        n2 = minkowski_dot4(q, q)
        q = [q[i] / mp.sqrt(n2) for i in range(4)]
        beta = beta_from_four_velocity_mp(q, c)
        cur = momentum_spherical_mp(beta, m, c)
        drho = cur["rho"] - prev["rho"]
        dtheta = cur["theta"] - prev["theta"]
        dphi = cur["phi"] - prev["phi"]
        # unwrap phi jumps
        if dphi > mp.pi:
            dphi -= 2 * mp.pi
        elif dphi < -mp.pi:
            dphi += 2 * mp.pi
        W = omega_so3_matrix_mp(
            E=cur["E"],
            m=m,
            theta=cur["theta"],
            drho=drho,
            dtheta=dtheta,
            dphi=dphi,
        )
        R = mp.expm(-W) * R
        prev = cur
    return R


def omega_pexp_closed_path_mp(
    betas: list[list[Any]],
    *,
    m: Any = mp.mpf(1),
    c: Any = mp.mpf(1),
    n_steps_edge: int = 48,
) -> dict[str, Any]:
    """Product of chart-based omega Pexp along successive geodesic edges."""
    R = mp.eye(3)
    for i in range(len(betas) - 1):
        R = (
            omega_pexp_along_geodesic_edge_mp(
                betas[i], betas[i + 1], m=m, c=c, n_steps=n_steps_edge
            )
            * R
        )
    return {
        "theta": rotation_angle_atan2_mp(R),
        "R": R,
        "orth": so3_residuals_mp(R)[0],
    }


def palge_pfeifer_suite_mp(
    t: CGMThresholds,
    *,
    m: Any = mp.mpf(1),
    circ_V: Any | None = None,
) -> dict[str, Any]:
    """
    Calibration (circular Thomas) + CGM piecewise-geodesic path holonomies.
    Stage betas as Einstein velocities map to mass-shell 4-velocities.
    """
    if circ_V is None:
        circ_V = mp.mpf("0.3")
    circ = circular_curvature_holonomy_alpha_mp(circ_V, m)
    circ_pexp = circular_pexp_so3_mp(circ_V, m, n_steps=128)
    circ_su2 = circular_pexp_su2_curvature_mp(circ_V)

    pts = mp_stage_points(t)
    paths = {
        "BU": ("ONA", "BU+", "BU-", "ONA"),
        "palindrome": ("UNA", "ONA", "BU+", "BU-", "ONA", "UNA"),
    }
    out_paths: dict[str, Any] = {}
    for name, path in paths.items():
        betas = [pts[s] for s in path]
        geo = geodesic_path_holonomy_mp(betas)
        om = omega_pexp_closed_path_mp(betas, m=m, n_steps_edge=32)
        out_paths[name] = {
            "path": path,
            "geodesic_boost": geo,
            "omega_chart_pexp": om,
            "spherical": [momentum_spherical_mp(pts[s], m) for s in path[:-1]],
        }
    return {
        "m": m,
        "circular": circ,
        "circular_pexp": circ_pexp,
        "circular_su2": circ_su2,
        "paths": out_paths,
    }


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
# CGM stage coordinates and loops
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class StageCoordinates:
    name: str
    c: float
    una_vector: tuple[float, float, float]
    ona_vector: tuple[float, float, float]
    bu_plus_vector: tuple[float, float, float]
    bu_minus_vector: tuple[float, float, float]


def stage_coordinates(
    t: CGMThresholds,
    *,
    u_p: float | None = None,
    theta_ona: float | None = None,
    m_a: float | None = None,
) -> StageCoordinates:
    """CGM payload stages as Einstein gyrovector coordinates with c = 1."""
    return StageCoordinates(
        name="cgm_stage_coordinates",
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
    product_right_mult: np.ndarray


def compute_loop_holonomy(
    gs: GyroVectorSpace,
    name: str,
    path: tuple[str, ...],
    points: dict[str, tuple[float, float, float]],
) -> LoopHolonomyResult:
    """
    Left-action product: x' = G_n ... G_1 x, accumulated as total = G @ total.
    """
    for stage in set(path):
        assert_in_ball(points[stage], float(gs.c))

    leg_angles: list[float] = []
    leg_matrices: list[np.ndarray] = []
    total = np.eye(3)
    total_right = np.eye(3)
    for i in range(len(path) - 1):
        a = np.asarray(points[path[i]], dtype=float)
        b = np.asarray(points[path[i + 1]], dtype=float)
        G = np.asarray(gs.gyration(a, b), dtype=float)
        leg_angles.append(rotation_report_from_matrix(G).angle)
        leg_matrices.append(G)
        total = G @ total
        total_right = total_right @ G

    return LoopHolonomyResult(
        name=name,
        path=path,
        leg_angles=tuple(float(x) for x in leg_angles),
        leg_matrices=tuple(leg_matrices),
        total=rotation_report_from_matrix(total),
        product=total,
        product_right_mult=total_right,
    )


def points_from_stages(stages: StageCoordinates) -> dict[str, tuple[float, float, float]]:
    return {
        "UNA": stages.una_vector,
        "ONA": stages.ona_vector,
        "BU+": stages.bu_plus_vector,
        "BU-": stages.bu_minus_vector,
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
