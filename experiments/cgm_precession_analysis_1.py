#!/usr/bin/env python3
"""
CGM precession analysis, part 1: measure holonomy of stage loops
under Fermi-Walker, inertial-frame, and chart transport.

Companions: cgm_precession_analysis_{2,run}.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import mpmath as mp

_EXP = Path(__file__).resolve().parent
_ROOT = _EXP.parent
for _p in (_EXP, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cgm_holonomy_analysis_common import (
    CGMThresholds,
    TOL_MP,
    analytic_bu_holonomy,
    beta_from_four_velocity_mp,
    compute_exact_su2_holonomy,
    einstein_gyroaddition_mp,
    extract_rotation_from_lorentz_product_mp,
    four_velocity_from_beta_mp,
    gyr_matrix_raw_mp,
    gyrotriangle_defect_mp,
    gyrotriangle_defect_triangle_vertices_mp,
    lorentz_boost_from_four_velocity_mp,
    minkowski_dot4,
    momentum_spherical_mp,
    mp_stage_points,
    mpmath_dual_pole_word,
    mpmath_palindrome_word,
    omega_so3_matrix_mp,
    poincare_radius_from_beta,
    pure_boost_u_to_v_mp,
    rotation_angle_atan2_mp,
    stage_angle_defect_euclid_mp,
    tw_angle_exact,
    tw_angle_unequal,
)
from gyroscopic.hQVM.constants import APERTURE_GAP, BU_HOLONOMY_ANGLE, M_A, RHO

RESULTS_PATH = _EXP / "cgm_precession_analysis_results.txt"

STAGES = ("UNA", "ONA", "BU+", "BU-")
REACH_L_MAX = 5
PEXP_STEPS = 32
THOMAS_STEPS = 32
LAB_CLOSE_TOL = 1e-10
ANGLE_BIN = 1e-10

BU_PATH = ("ONA", "BU+", "BU-", "ONA")
PAL_PATH = ("UNA", "ONA", "BU+", "BU-", "ONA", "UNA")

NAMED_LOOPS: list[tuple[str, tuple[str, ...]]] = [
    ("outback_ONA_BUp", ("ONA", "BU+", "ONA")),
    ("outback_UNA_ONA", ("UNA", "ONA", "UNA")),
    ("outback_UNA_BUp", ("UNA", "BU+", "UNA")),
    ("BU", BU_PATH),
    ("tri_UNA_BUp_BUm", ("UNA", "BU+", "BU-", "UNA")),
    ("pal", PAL_PATH),
]


def _deg(r: float) -> float:
    return r * 180.0 / math.pi


def _mpf(x: Any) -> float:
    return float(x)


def _neg3(a: list[Any]) -> list[Any]:
    return [-a[0], -a[1], -a[2]]


def _norm3(v: list[Any]) -> float:
    return float(mp.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]))


def _dot3(a: list[Any], b: list[Any]) -> float:
    return float(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])


def _angle3(a: list[Any], b: list[Any]) -> float:
    na = _norm3(a)
    nb = _norm3(b)
    if na < 1e-30 or nb < 1e-30:
        return 0.0
    c = max(-1.0, min(1.0, _dot3(a, b) / (na * nb)))
    return float(mp.acos(c))


def _gate(label: str, ok: bool) -> tuple[str, bool]:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    return label, ok


def _axis_of(R: Any) -> tuple[float, float, float]:
    kx = _mpf(R[2, 1] - R[1, 2])
    ky = _mpf(R[0, 2] - R[2, 0])
    kz = _mpf(R[1, 0] - R[0, 1])
    n = math.sqrt(kx * kx + ky * ky + kz * kz)
    if n < 1e-15:
        return (0.0, 0.0, 0.0)
    return (kx / n, ky / n, kz / n)


def _axis_dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _so3_from_matrix(R4: Any) -> Any:
    R = mp.matrix(3)
    for i in range(3):
        for j in range(3):
            R[i, j] = R4[i + 1, j + 1]
    return R


def _perm_xyz(beta: list[Any], kind: str) -> list[Any]:
    x, y, z = beta[0], beta[1], beta[2]
    if kind == "z":
        return [x, y, z]
    if kind == "x":
        return [y, z, x]
    if kind == "y":
        return [z, x, y]
    return [x, y, z]


def _rotation_axis_to_z(axis: tuple[float, float, float]) -> Any:
    a = mp.matrix([mp.mpf(axis[0]), mp.mpf(axis[1]), mp.mpf(axis[2])])
    an = mp.norm(a)
    if an < mp.mpf("1e-30"):
        return mp.eye(3)
    a = a / an
    z = mp.matrix([mp.mpf(0), mp.mpf(0), mp.mpf(1)])
    c = a[0] * z[0] + a[1] * z[1] + a[2] * z[2]
    v = mp.matrix(
        [
            a[1] * z[2] - a[2] * z[1],
            a[2] * z[0] - a[0] * z[2],
            a[0] * z[1] - a[1] * z[0],
        ]
    )
    vn2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
    if vn2 < mp.mpf("1e-30"):
        if c > 0:
            return mp.eye(3)
        R = mp.eye(3)
        R[1, 1] = mp.mpf(-1)
        R[2, 2] = mp.mpf(-1)
        return R
    vx = mp.zeros(3)
    vx[0, 1] = -v[2]
    vx[0, 2] = v[1]
    vx[1, 0] = v[2]
    vx[1, 2] = -v[0]
    vx[2, 0] = -v[1]
    vx[2, 1] = v[0]
    return mp.eye(3) + vx + vx * vx * (1 - c) / vn2


def _apply_R3(R: Any, beta: list[Any]) -> list[Any]:
    r = R * mp.matrix(beta)
    return [r[0], r[1], r[2]]


def omega_pexp_edge_mp(
    beta_a: list[Any],
    beta_b: list[Any],
    chart: str,
    *,
    n_steps: int = PEXP_STEPS,
    frame_R: Any | None = None,
) -> Any:
    qa = four_velocity_from_beta_mp(beta_a)
    qb = four_velocity_from_beta_mp(beta_b)
    chi = mp.acosh(max(minkowski_dot4(qa, qb), mp.mpf(1)))
    R = mp.eye(3)
    if chi < mp.mpf("1e-80"):
        return R

    def sph(beta: list[Any]) -> dict[str, Any]:
        b = _apply_R3(frame_R, beta) if frame_R is not None else _perm_xyz(beta, chart)
        return momentum_spherical_mp(b)

    prev = sph(beta_a)
    sh = mp.sinh(chi)
    for k in range(1, n_steps + 1):
        t = mp.mpf(k) / n_steps
        q = [
            (mp.sinh((1 - t) * chi) * qa[i] + mp.sinh(t * chi) * qb[i]) / sh
            for i in range(4)
        ]
        n2 = minkowski_dot4(q, q)
        q = [q[i] / mp.sqrt(n2) for i in range(4)]
        cur = sph(beta_from_four_velocity_mp(q))
        dphi = cur["phi"] - prev["phi"]
        if dphi > mp.pi:
            dphi -= 2 * mp.pi
        elif dphi < -mp.pi:
            dphi += 2 * mp.pi
        W = omega_so3_matrix_mp(
            E=cur["E"],
            m=mp.mpf(1),
            theta=cur["theta"],
            drho=cur["rho"] - prev["rho"],
            dtheta=cur["theta"] - prev["theta"],
            dphi=dphi,
        )
        R = mp.expm(-W) * R
        prev = cur
    return R


def omega_pexp_path_mp(
    betas: list[list[Any]],
    chart: str,
    *,
    frame_R: Any | None = None,
) -> float:
    R = mp.eye(3)
    for i in range(len(betas) - 1):
        R = omega_pexp_edge_mp(betas[i], betas[i + 1], chart, frame_R=frame_R) * R
    return _mpf(rotation_angle_atan2_mp(R))


def _so3_from_omega_vec(wx: Any, wy: Any, wz: Any) -> Any:
    W = mp.zeros(3)
    W[0, 1] = -wz
    W[1, 0] = wz
    W[0, 2] = wy
    W[2, 0] = -wy
    W[1, 2] = -wx
    W[2, 1] = wx
    return W


def thomas_pexp_edge_mp(
    beta_a: list[Any],
    beta_b: list[Any],
    *,
    n_steps: int = THOMAS_STEPS,
) -> Any:
    """
    Palge-Pfeifer / Thomas spin connection in Cartesian velocity coordinates:
    omega = (gamma^2 / (gamma+1)) beta x d beta, regular at rest and at the poles.
    """
    qa = four_velocity_from_beta_mp(beta_a)
    qb = four_velocity_from_beta_mp(beta_b)
    chi = mp.acosh(max(minkowski_dot4(qa, qb), mp.mpf(1)))
    R = mp.eye(3)
    if chi < mp.mpf("1e-80"):
        return R
    sh = mp.sinh(chi)
    prev: list[Any] | None = None
    for k in range(n_steps + 1):
        t = mp.mpf(k) / n_steps
        q = [
            (mp.sinh((1 - t) * chi) * qa[i] + mp.sinh(t * chi) * qb[i]) / sh
            for i in range(4)
        ]
        n2 = minkowski_dot4(q, q)
        q = [q[i] / mp.sqrt(n2) for i in range(4)]
        cur = beta_from_four_velocity_mp(q)
        if prev is not None:
            dbeta = [cur[i] - prev[i] for i in range(3)]
            mid = [(prev[i] + cur[i]) / 2 for i in range(3)]
            b2 = mid[0] * mid[0] + mid[1] * mid[1] + mid[2] * mid[2]
            g = 1 / mp.sqrt(max(mp.mpf("1e-30"), 1 - b2))
            kth = g * g / (g + 1)
            wx = kth * (mid[1] * dbeta[2] - mid[2] * dbeta[1])
            wy = kth * (mid[2] * dbeta[0] - mid[0] * dbeta[2])
            wz = kth * (mid[0] * dbeta[1] - mid[1] * dbeta[0])
            R = mp.expm(-_so3_from_omega_vec(wx, wy, wz)) * R
        prev = cur
    return R


def thomas_pexp_path_mp(betas: list[list[Any]], *, n_steps: int = THOMAS_STEPS) -> float:
    R = mp.eye(3)
    for i in range(len(betas) - 1):
        R = thomas_pexp_edge_mp(betas[i], betas[i + 1], n_steps=n_steps) * R
    return _mpf(rotation_angle_atan2_mp(R))


def thomas_pexp_richardson_mp(betas: list[list[Any]], *, n_steps: int = THOMAS_STEPS) -> dict[str, float]:
    th_n = thomas_pexp_path_mp(betas, n_steps=n_steps)
    th_2n = thomas_pexp_path_mp(betas, n_steps=2 * n_steps)
    return {
        "theta_n": th_n,
        "theta_2n": th_2n,
        "theta": (4.0 * th_2n - th_n) / 3.0,
    }


def vertex_chart_coords(
    pts: dict[str, list[Any]], chart: str, frame_R: Any | None = None
) -> dict[str, tuple[float, float, float]]:
    out: dict[str, tuple[float, float, float]] = {}
    for name, beta in pts.items():
        b = _apply_R3(frame_R, beta) if frame_R is not None else _perm_xyz(beta, chart)
        s = momentum_spherical_mp(b)
        out[name] = (_mpf(s["theta"]), _mpf(s["phi"]), _mpf(mp.sin(s["theta"])))
    return out


def factor_boost_word_lab_mp(ds: list[list[Any]]) -> dict[str, Any]:
    zero = [mp.mpf(0), mp.mpf(0), mp.mpf(0)]
    if not ds:
        return {"theta": 0.0, "R": mp.eye(3), "u_final": zero, "corners": []}
    u = [ds[0][0], ds[0][1], ds[0][2]]
    R = mp.eye(3)
    corners: list[dict[str, float]] = []
    for v_raw in ds[1:]:
        v_eff = _apply_R3(R, v_raw)
        G = gyr_matrix_raw_mp(u, v_eff)
        corners.append(
            {
                "theta": _mpf(rotation_angle_atan2_mp(G)),
                "beta_u": _norm3(u),
                "beta_v": _norm3(v_eff),
                "sep": _angle3(u, v_eff),
            }
        )
        u = einstein_gyroaddition_mp(u, v_eff)
        R = G * R
    return {
        "theta": _mpf(rotation_angle_atan2_mp(R)),
        "R": R,
        "u_final": u,
        "corners": corners,
    }


class EdgeCache:
    def __init__(self, t: CGMThresholds) -> None:
        self.pts = mp_stage_points(t)
        self.gyr: dict[tuple[str, str], Any] = {}
        self.disp: dict[tuple[str, str], list[Any]] = {}
        self.T: dict[tuple[str, str], Any] = {}
        self.q: dict[str, list[Any]] = {}
        self.L0: dict[str, Any] = {}
        self.L0inv: dict[str, Any] = {}
        for s in STAGES:
            self.q[s] = four_velocity_from_beta_mp(self.pts[s])
            self.L0[s] = lorentz_boost_from_four_velocity_mp(self.q[s])
            self.L0inv[s] = self.L0[s] ** -1
        for a in STAGES:
            for b in STAGES:
                if a == b:
                    continue
                pa, pb = self.pts[a], self.pts[b]
                self.gyr[(a, b)] = gyr_matrix_raw_mp(pa, pb)
                self.disp[(a, b)] = einstein_gyroaddition_mp(_neg3(pa), pb)
                self.T[(a, b)] = pure_boost_u_to_v_mp(self.q[a], self.q[b])

    def origin_gyr(self, path: tuple[str, ...]) -> tuple[float, Any]:
        R = mp.eye(3)
        for i in range(len(path) - 1):
            R = self.gyr[(path[i], path[i + 1])] * R
        return _mpf(rotation_angle_atan2_mp(R)), R

    def lab(self, path: tuple[str, ...]) -> tuple[float, Any]:
        ds = [self.disp[(path[i], path[i + 1])] for i in range(len(path) - 1)]
        R, _ = extract_rotation_from_lorentz_product_mp(ds)
        return _mpf(rotation_angle_atan2_mp(R)), R

    def geodesic(self, path: tuple[str, ...]) -> tuple[float, Any]:
        P = mp.eye(4)
        for i in range(len(path) - 1):
            P = self.T[(path[i], path[i + 1])] * P
        start = path[0]
        R4 = self.L0inv[start] * P * self.L0[start]
        R = _so3_from_matrix(R4)
        return _mpf(rotation_angle_atan2_mp(R)), R


def priors(t: CGMThresholds) -> dict[str, float]:
    return {
        "u_p": _mpf(t.u_p),
        "o_p": _mpf(t.theta_ona),
        "m_a": _mpf(t.m_a),
        "theta_cs": _mpf(t.theta_cs),
    }


def forced_basis(t: CGMThresholds) -> dict[str, float]:
    omega_corner, delta_bu = analytic_bu_holonomy(t)
    rho = _mpf(delta_bu) / float(M_A)
    rho0 = 2.0 * _mpf(poincare_radius_from_beta(t.theta_ona))
    return {
        "omega_corner": _mpf(omega_corner),
        "delta_BU": _mpf(delta_bu),
        "rho": rho,
        "Delta": 1.0 - rho,
        "phi_SU2": _mpf(compute_exact_su2_holonomy()[0]),
        "omega0": _mpf(tw_angle_exact(t.u_p, t.theta_ona)),
        "rho0": rho0,
        "two_1_rho0": 2.0 * (1.0 - rho0),
    }


def connection_measurements(t: CGMThresholds, cache: EdgeCache) -> dict[str, Any]:
    word = mpmath_dual_pole_word(t)
    pal = mpmath_palindrome_word(t, word["R"])
    pts = cache.pts
    geo_bu, _ = cache.geodesic(BU_PATH)
    geo_pal, _ = cache.geodesic(PAL_PATH)
    th_gyr_bu, R_gyr_bu = cache.origin_gyr(BU_PATH)
    th_gyr_pal, R_gyr_pal = cache.origin_gyr(PAL_PATH)
    th_lab_bu, _ = cache.lab(BU_PATH)
    th_lab_pal, _ = cache.lab(PAL_PATH)
    bu_betas = [pts[s] for s in BU_PATH]
    pal_betas = [pts[s] for s in PAL_PATH]
    sph_bu = omega_pexp_path_mp(bu_betas, "z")
    sph_pal = omega_pexp_path_mp(pal_betas, "z")
    thom_bu = thomas_pexp_richardson_mp(bu_betas)
    thom_pal = thomas_pexp_richardson_mp(pal_betas)
    defct_corner = gyrotriangle_defect_mp(pts["ONA"], pts["BU+"])
    defct_bu = gyrotriangle_defect_triangle_vertices_mp(pts["ONA"], pts["BU+"], pts["BU-"])
    return {
        "can": {
            "word": _mpf(word["theta_word"]),
            "origin_gyr_BU": th_gyr_bu,
            "origin_gyr_pal": th_gyr_pal,
            "geodesic_BU": geo_bu,
            "geodesic_pal": geo_pal,
            "defect_corner": _mpf(defct_corner["defect"]),
            "defect_BU_tri": _mpf(defct_bu["defect"]),
            "R_BU": R_gyr_bu,
            "R_pal": R_gyr_pal,
            "A": pal["A"],
            "resid_Ainv_R_A": _mpf(pal["resid_Ainv_R_A"]),
            "resid_A_R_Ainv": _mpf(pal["resid_A_R_Ainv"]),
        },
        "lab": {
            "relative_boost_BU": th_lab_bu,
            "relative_boost_pal": th_lab_pal,
        },
        "chart": {
            "omega_chart_complete_BU": thom_bu["theta"],
            "omega_chart_complete_pal": thom_pal["theta"],
            "thomas_n_BU": thom_bu["theta_n"],
            "thomas_2n_BU": thom_bu["theta_2n"],
            "omega_chart_sph_BU": sph_bu,
            "omega_chart_sph_pal": sph_pal,
            "offset_BU": abs(geo_bu - sph_bu),
            "offset_pal": abs(geo_pal - sph_pal),
        },
    }


def closed_walks(max_edges: int) -> list[tuple[str, ...]]:
    out: list[tuple[str, ...]] = []
    for n_e in range(2, max_edges + 1):

        def rec(path: list[str], n_edges: int = n_e) -> None:
            if len(path) == n_edges + 1:
                if path[-1] == path[0]:
                    out.append(tuple(path))
                return
            for s in STAGES:
                if s != path[-1]:
                    rec(path + [s], n_edges)

        for s0 in STAGES:
            rec([s0])
    return out


def _rank_3rows(rows: list[list[float]]) -> int:
    if not rows:
        return 0

    def collinear(a: list[float], b: list[float]) -> bool:
        cr = [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
        return sum(x * x for x in cr) < 1e-20

    if all(collinear(rows[0], rows[i]) for i in range(1, len(rows))):
        return 1

    def coplanar(a: list[float], b: list[float], c: list[float]) -> bool:
        return abs(
            a[0] * (b[1] * c[2] - b[2] * c[1])
            - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0])
        ) < 1e-20

    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            for k in range(j + 1, len(rows)):
                if not coplanar(rows[i], rows[j], rows[k]):
                    return 3
    return 2


def path_topology(path: tuple[str, ...], cache: EdgeCache) -> dict[str, Any]:
    uniq: list[str] = []
    for s in path:
        if s not in uniq:
            uniq.append(s)
    rank = _rank_3rows([[float(cache.pts[s][k]) for k in range(3)] for s in uniq])
    return {"dim_spanned": rank, "is_planar": rank < 3, "n_stages": len(uniq)}


def print_chart(basis: dict[str, float], conn: dict[str, Any], cache: EdgeCache) -> list[tuple[str, bool]]:
    gates: list[tuple[str, bool]] = []
    d = basis["delta_BU"]
    pts = cache.pts
    bu_betas = [pts[s] for s in BU_PATH]
    pal_betas = [pts[s] for s in PAL_PATH]
    diag_R = _rotation_axis_to_z((1.0, 1.0, 1.0))
    ch = conn["chart"]

    print("  complete Palge-Pfeifer (Cartesian Thomas, Richardson n and 2n)")
    print(
        f"  BU   n={ch['thomas_n_BU']:.10f}  2n={ch['thomas_2n_BU']:.10f}  "
        f"Rich={ch['omega_chart_complete_BU']:.10f}  geo={conn['can']['geodesic_BU']:.10f}"
    )
    print(f"  pal  Rich={ch['omega_chart_complete_pal']:.10f}  geo={conn['can']['geodesic_pal']:.10f}")
    gates.append(
        _gate(
            "chart complete BU = delta_BU",
            abs(ch["omega_chart_complete_BU"] - d) < 1e-8,
        )
    )
    gates.append(
        _gate(
            "chart complete pal = delta_BU",
            abs(ch["omega_chart_complete_pal"] - d) < 1e-8,
        )
    )
    print()

    print("  spherical-chart Pexp (singular at poles and at rest)")
    print("  vertex (theta, sin_theta); POLE if |sin_theta|<1e-8")
    for chart, frame in (("z", None), ("x", None), ("y", None), ("diag", diag_R)):
        coords = vertex_chart_coords(pts, chart if frame is None else "z", frame)
        bits = []
        for name in STAGES:
            th, _ph, s = coords[name]
            pole = " POLE" if abs(s) < 1e-8 else ""
            bits.append(f"{name}({th:.4f},{s:.1e}{pole})")
        print(f"  {chart:<4}  " + "  ".join(bits))
    print()

    charts = (("z", "z", None), ("x", "x", None), ("y", "y", None), ("diag", "z", diag_R))
    print(f"  {'chart':<6} {'th_BU':>12} {'th_pal':>12} {'G_BU':>12} {'G_pal':>12}")
    g_by: dict[str, float] = {}
    for name, kind, frame in charts:
        if name == "z":
            th_bu = ch["omega_chart_sph_BU"]
            th_pal = ch["omega_chart_sph_pal"]
        else:
            th_bu = omega_pexp_path_mp(bu_betas, kind, frame_R=frame)
            th_pal = omega_pexp_path_mp(pal_betas, kind, frame_R=frame)
        g_bu, g_pal = th_bu - d, th_pal - d
        g_by[name] = g_bu
        print(f"  {name:<6} {th_bu:12.8f} {th_pal:12.8f} {g_bu:12.8f} {g_pal:12.8f}")
    print()

    g_z, g_diag = g_by["z"], g_by["diag"]
    ch["G_diag_BU"] = g_diag
    print(f"  G_sph_diag - G_sph_z = {g_diag - g_z:.10f}")
    gates.append(_gate("z-chart poles at BU", sum(1 for s in STAGES if abs(vertex_chart_coords(pts, "z")[s][2]) < 1e-8) >= 2))
    gates.append(
        _gate(
            "diag-chart has no stage poles",
            sum(1 for s in STAGES if abs(vertex_chart_coords(pts, "z", diag_R)[s][2]) < 1e-8) == 0,
        )
    )

    topo_bu = path_topology(BU_PATH, cache)
    topo_pal = path_topology(PAL_PATH, cache)
    print(f"  BU dim={topo_bu['dim_spanned']} planar={topo_bu['is_planar']}   pal dim={topo_pal['dim_spanned']} planar={topo_pal['is_planar']}")
    gates.append(_gate("BU loop planar (yz)", topo_bu["is_planar"] and topo_bu["dim_spanned"] == 2))
    gates.append(_gate("palindrome spans 3 axes", topo_pal["dim_spanned"] == 3))
    print()
    return gates


def print_lab(basis: dict[str, float], conn: dict[str, Any], cache: EdgeCache) -> list[tuple[str, bool]]:
    gates: list[tuple[str, bool]] = []
    rows: list[dict[str, Any]] = []
    print(f"  {'loop':<18} {'L':>2} {'th_can':>12} {'th_inert':>12} {'F_sc':>12} {'th_RF':>12}")
    for name, path in NAMED_LOOPS:
        th_c, R_c = cache.origin_gyr(path)
        th_l, R_l = cache.lab(path)
        R_F = R_c ** -1 * R_l
        th_rf = _mpf(rotation_angle_atan2_mp(R_F))
        rec = {
            "name": name,
            "path": path,
            "th_can": th_c,
            "th_lab": th_l,
            "F": th_l - th_c,
            "theta_RF": th_rf,
            "axis_RF": _axis_of(R_F),
        }
        rows.append(rec)
        print(
            f"  {name:<18} {len(path)-1:2d} {th_c:12.8f} {th_l:12.8f} "
            f"{rec['F']:12.8f} {th_rf:12.8f}"
        )
    print()

    by_name = {r["name"]: r for r in rows}
    f_bu = by_name["BU"]["F"]
    f_pal = by_name["pal"]["F"]
    f_ob = by_name["outback_ONA_BUp"]["F"]
    f_una = by_name["tri_UNA_BUp_BUm"]["F"]
    f_ob_una = by_name["outback_UNA_BUp"]["F"]
    print(f"  F = th_inert - th_can.  theta_RF = angle of R_can^-1 R_inert.")
    print(f"  F_BU                         {f_bu:.12f}")
    print(f"  theta_RF BU                  {by_name['BU']['theta_RF']:.12f}")
    print(f"  th_can + th_inert BU         {by_name['BU']['th_can'] + by_name['BU']['th_lab']:.12f}")
    print(f"  F_pal                        {f_pal:.12f}")
    print(f"  theta_RF pal                 {by_name['pal']['theta_RF']:.12f}")
    gates.append(_gate("F_BU = lab(ONA-BU+ out-back)", abs(f_bu - f_ob) < 1e-12))
    gates.append(_gate("F(UNA dual-pole) = lab(UNA-BU+ out-back)", abs(f_una - f_ob_una) < 1e-12))
    gates.append(
        _gate(
            "BU: theta_RF = th_can + th_inert",
            abs(by_name["BU"]["theta_RF"] - (by_name["BU"]["th_can"] + by_name["BU"]["th_lab"])) < 1e-10,
        )
    )
    gates.append(
        _gate(
            "out-back: theta(R_can^{-1} R_lab) = theta_lab",
            all(abs(by_name[n]["theta_RF"] - by_name[n]["th_lab"]) < 1e-10 for n in ("outback_ONA_BUp", "outback_UNA_ONA", "outback_UNA_BUp")),
        )
    )
    print()

    print(f"  {'loop':<6} {'dim':>3} {'planar':<6} {'theta_inert':>12} {'sum_corners':>12} {'resid':>10}")
    for name, path in (("BU", BU_PATH), ("pal", PAL_PATH)):
        ds = [cache.disp[(path[i], path[i + 1])] for i in range(len(path) - 1)]
        fac = factor_boost_word_lab_mp(ds)
        topo = path_topology(path, cache)
        csum = sum(c["theta"] for c in fac["corners"])
        resid = abs(fac["theta"] - csum)
        print(
            f"  {name:<6} {topo['dim_spanned']:3d} {str(topo['is_planar']):<6} "
            f"{fac['theta']:12.8f} {csum:12.8f} {resid:10.3e}"
        )
        for j, c0 in enumerate(fac["corners"]):
            print(f"    corner{j+1}  theta={c0['theta']:.10f}  |u|={c0['beta_u']:.6f}  |v|={c0['beta_v']:.6f}  sep={c0['sep']:.6f}")
        if topo["is_planar"]:
            gates.append(_gate(f"{name} planar: corner-sum = theta_lab", resid < 1e-8))
        else:
            gates.append(_gate(f"{name} non-planar: corner-sum != theta_lab", resid > 1e-8))
        gates.append(_gate(f"{name} factorization = theta_lab", abs(fac["theta"] - by_name[name]["th_lab"]) < 1e-8))
    conn["lab"]["named_rows"] = rows
    print()
    return gates


def print_reachability(basis: dict[str, float], conn: dict[str, Any], cache: EdgeCache) -> list[tuple[str, bool]]:
    gates: list[tuple[str, bool]] = []
    d = basis["delta_BU"]
    walks = closed_walks(REACH_L_MAX)
    recs: list[dict[str, Any]] = []
    lab_closed: list[tuple[str, ...]] = []
    for path in walks:
        th_c, R_c = cache.origin_gyr(path)
        th_l, _ = cache.lab(path)
        th_g, _ = cache.geodesic(path)
        ds = [cache.disp[(path[i], path[i + 1])] for i in range(len(path) - 1)]
        net_boost = _norm3(factor_boost_word_lab_mp(ds)["u_final"])
        if net_boost < LAB_CLOSE_TOL:
            lab_closed.append(path)
        recs.append(
            {
                "L": len(path) - 1,
                "path": path,
                "th_can": th_c,
                "th_lab": th_l,
                "th_geo": th_g,
                "net_boost": net_boost,
                "can_eq_geo": abs(th_c - th_g) < 1e-8,
                "R": R_c,
            }
        )

    def spectrum(key: str) -> list[tuple[float, int, int]]:
        buckets: dict[int, list[dict[str, Any]]] = {}
        for r in recs:
            buckets.setdefault(int(round(r[key] / ANGLE_BIN)), []).append(r)
        out = []
        for k, grp in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
            out.append((k * ANGLE_BIN, len(grp), min(g["L"] for g in grp)))
        return out

    spec_can = spectrum("th_can")
    spec_lab = spectrum("th_lab")
    n_agree = sum(1 for r in recs if r["can_eq_geo"])
    print(f"  closed walks of length 2 through {REACH_L_MAX}: n={len(walks)}")
    print(f"  origin_gyr equals geodesic on {n_agree}/{len(recs)}")
    print(f"  {'theta':>14}  {'mult':>5}  {'min_L':>5}  connection")
    for ang, mult, min_L in spec_can:
        tag = "  delta_BU" if abs(ang - d) < 1e-8 else ""
        print(f"  {ang:14.10f}  {mult:5d}  {min_L:5d}  Fermi-Walker{tag}")
    for ang, mult, min_L in spec_lab[:8]:
        print(f"  {ang:14.10f}  {mult:5d}  {min_L:5d}  inertial-frame")
    print()

    hits = [r for r in recs if abs(r["th_can"] - d) < 1e-8]
    L_span = min((r["L"] for r in hits), default=-1)
    gates.append(_gate("delta_BU at L=3 on nabla^can", L_span == 3))

    th_pal, R_pal = cache.origin_gyr(PAL_PATH)
    th_bu, R_bu = cache.origin_gyr(BU_PATH)
    print(f"  conjugacy |theta_pal-theta_BU|={abs(th_pal-th_bu):.3e}  |axis dot|={abs(_axis_dot(_axis_of(R_pal), _axis_of(R_bu))):.6f}")
    print(f"  ||R_pal - A^-1 R_BU A|| = {conn['can']['resid_Ainv_R_A']:.3e}")
    gates.append(_gate("palindrome preserves nabla^can angle", abs(th_pal - th_bu) < 1e-10))
    gates.append(_gate("palindrome transports axis", abs(_axis_dot(_axis_of(R_pal), _axis_of(R_bu))) < 0.999))

    bu_family = [
        ("ONA", "BU+", "BU-", "ONA"),
        ("BU+", "BU-", "ONA", "BU+"),
        ("BU-", "ONA", "BU+", "BU-"),
        ("ONA", "BU-", "BU+", "ONA"),
    ]
    print(f"  {'path':<28} {'Fermi-Walker':>12} {'geodesic':>12} {'inertial':>12}")
    can_vals, geo_vals, lab_vals = [], [], []
    for path in bu_family:
        tc, _ = cache.origin_gyr(path)
        tg, _ = cache.geodesic(path)
        tl, _ = cache.lab(path)
        can_vals.append(tc)
        geo_vals.append(tg)
        lab_vals.append(tl)
        print(f"  {'-'.join(path):<28} {tc:12.8f} {tg:12.8f} {tl:12.8f}")
    can_span = max(can_vals) - min(can_vals)
    geo_span = max(geo_vals) - min(geo_vals)
    lab_span = max(lab_vals) - min(lab_vals)
    print(f"  span Fermi-Walker={can_span:.3e}  geodesic={geo_span:.3e}  inertial={lab_span:.3e}")
    gates.append(_gate("nabla^can invariant on BU refactorizations", can_span < 1e-10 and geo_span < 1e-10))
    gates.append(_gate("nabla^can refactorizations equal delta_BU", all(abs(v - d) < 1e-10 for v in can_vals + geo_vals)))
    gates.append(_gate("nabla^lab not invariant on BU refactorizations", lab_span > 1e-10))

    outback = [r for r in recs if r["L"] == 2]
    max_out_can = max(r["th_can"] for r in outback)
    print(f"  length-2 out-and-back n={len(outback)}  max theta_can={max_out_can:.3e}  max theta_inert={max(r['th_lab'] for r in outback):.6f}")
    gates.append(_gate("nabla^can out-back ~ 0", max_out_can < 1e-10))
    gates.append(_gate("origin_gyr == geodesic on all walks", n_agree == len(recs)))
    gates.append(_gate("lab spectrum finer than canonical", len(spec_lab) > len(spec_can)))
    print(f"  distinct angles: Fermi-Walker={len(spec_can)}  inertial-frame={len(spec_lab)}")
    print()
    print(f"  inertial-frame products with vanishing net boost: {len(lab_closed)}/{len(walks)}")
    for path in lab_closed:
        print(f"    CLOSED_INERTIAL  L={len(path)-1}  {'-'.join(path)}")
    only_collinear = all(set(path) <= {"BU+", "BU-"} for path in lab_closed)
    gates.append(_gate("lab-closed walks are collinear BU+-BU- out-backs", only_collinear and len(lab_closed) == 4))
    gates.append(_gate("lab-closed walks have theta_lab = 0", all(abs(r["th_lab"]) < 1e-10 for r in recs if r["net_boost"] < LAB_CLOSE_TOL)))
    print()

    pts = cache.pts
    defects = [
        ("origin-UNA-ONA", gyrotriangle_defect_mp(pts["UNA"], pts["ONA"])["defect"]),
        ("origin-ONA-BU+", gyrotriangle_defect_mp(pts["ONA"], pts["BU+"])["defect"]),
        ("origin-BU+-UNA", gyrotriangle_defect_mp(pts["BU+"], pts["UNA"])["defect"]),
        ("UNA-ONA-BU+", gyrotriangle_defect_triangle_vertices_mp(pts["UNA"], pts["ONA"], pts["BU+"])["defect"]),
        ("UNA-BU+-BU-", gyrotriangle_defect_triangle_vertices_mp(pts["UNA"], pts["BU+"], pts["BU-"])["defect"]),
        ("ONA-BU+-BU-", gyrotriangle_defect_triangle_vertices_mp(pts["ONA"], pts["BU+"], pts["BU-"])["defect"]),
    ]
    can_angles = [ang for ang, _, _ in spec_can]
    print(f"  {'triangle':<18} {'defect':>14}  nearest Fermi-Walker")
    for name, defct in defects:
        df = _mpf(defct)
        nearest = min(can_angles, key=lambda a: abs(a - df))
        print(f"  {name:<18} {df:14.10f}  {nearest:14.10f}")
        if name == "origin-ONA-BU+":
            gates.append(_gate("defect(origin,ONA,BU+) = omega_corner", abs(df - basis["omega_corner"]) < 1e-10))
        if name == "ONA-BU+-BU-":
            gates.append(_gate("defect(ONA,BU+,BU-) = delta_BU", abs(df - d) < 1e-10))
        if name == "UNA-ONA-BU+":
            gates.append(_gate("defect(UNA,ONA,BU+) in can spectrum", any(abs(df - a) < 1e-8 for a in can_angles)))
    conn["reach"] = {
        "spec_can": spec_can,
        "n_spec_can": len(spec_can),
        "n_spec_lab": len(spec_lab),
        "walk_count": len(walks),
        "lab_span": lab_span,
        "n_lab_closed": len(lab_closed),
        "lab_closed_paths": ["-".join(p) for p in lab_closed],
    }
    return gates


def print_wigner_thomas(
    basis: dict[str, float], cache: EdgeCache, conn: dict[str, Any]
) -> list[tuple[str, bool]]:
    uo = float(gyrotriangle_defect_mp(cache.pts["UNA"], cache.pts["ONA"])["defect"])
    ob = float(gyrotriangle_defect_mp(cache.pts["ONA"], cache.pts["BU+"])["defect"])
    ub = float(gyrotriangle_defect_mp(cache.pts["BU+"], cache.pts["UNA"])["defect"])
    pexp = conn["chart"]["omega_chart_complete_BU"]
    print("  Wigner rotation of two origin boosts (finite SO(3); no duration)")
    print(f"    UNA then ONA              {uo:.12f}")
    print(f"    ONA then BU+              {ob:.12f}")
    print(f"    BU+ then UNA              {ub:.12f}")
    print("  Thomas 1-form  omega = (gamma^2/(gamma+1)) beta x d beta")
    print(f"    Pexp of omega on BU loop  {pexp:.12f}")
    print(f"    2*(ONA then BU+)          {2.0 * ob:.12f}")
    print("  nabla^can = Fermi-Walker / geodesic transvection")
    print("  nabla^inert = product of successive Lorentz boosts in one inertial frame")
    return [
        _gate("Wigner ONA-BU+ = omega_corner", abs(ob - basis["omega_corner"]) < 1e-12),
        _gate("2 * Wigner ONA-BU+ = delta_BU", abs(2.0 * ob - basis["delta_BU"]) < 1e-12),
        _gate("Thomas Pexp BU = 2 * Wigner ONA-BU+", abs(pexp - 2.0 * ob) < 1e-8),
    ]


def run() -> tuple[list[tuple[str, bool]], dict[str, Any]]:
    mp.mp.dps = 50
    t = CGMThresholds.make()
    p = priors(t)
    b = forced_basis(t)
    cache = EdgeCache(t)
    conn = connection_measurements(t, cache)
    gates: list[tuple[str, bool]] = []

    print("CGM PRECESSION ANALYSIS")
    print("=" * 5)
    print("Machine-generated numerical verification report.")
    print("The public interpretation and definitions appear in")
    print("docs/Findings/Analysis_Precession.md.")
    print()

    print("1. PRIORS")
    print("-" * 5)
    print("  Constitutional stage thresholds. u_p, o_p, m_a are Einstein speeds")
    print("  of UNA, ONA, and BU. theta_cs is the Common Source angle pi/2.")
    print("  theta_una = arccos(u_p). The Euclidean defect of CS-UNA-ONA is")
    print("  theta_cs + theta_una + theta_ona - pi.")
    print(f"  u_p (UNA)     = {p['u_p']:.16f}")
    print(f"  o_p (ONA)     = {p['o_p']:.16f}")
    print(f"  m_a (BU)      = {p['m_a']:.16f}")
    print(f"  theta_cs (CS) = {p['theta_cs']:.16f}")
    print(f"  theta_una     = {_mpf(t.theta_una):.16f}")
    print(f"  theta_cs+una+ona = {_mpf(t.angle_sum):.16f}")
    euclid = _mpf(stage_angle_defect_euclid_mp(t))
    print(f"  euclid_defect = {euclid:.16e}")
    gates.append(_gate("stage-angle Euclidean defect = 0", abs(euclid) < 1e-20))
    print()

    print("2. CANONICAL BASIS")
    print("-" * 5)
    print("  omega_corner is the ONA-BU pair precession. delta_BU is twice that")
    print("  corner, the dual-pole holonomy. rho = delta_BU / m_a is the closure")
    print("  ratio. Delta = 1 - rho is the aperture gap. phi_SU2 is the compact")
    print("  commutator angle. omega0 is the equal-speed Wigner calibration.")
    print("  rho0 is the closure slope at vanishing amplitude.")
    for k in ("omega_corner", "delta_BU", "rho", "Delta", "phi_SU2", "omega0", "rho0", "two_1_rho0"):
        v = b[k]
        tag = f"  ({_deg(v):.4f} deg)" if k in ("omega_corner", "delta_BU", "phi_SU2", "omega0") else ""
        extra = "  equal-speed TW(u_p, o_p)" if k == "omega0" else ""
        print(f"  {k:<14} {v:.16f}{tag}{extra}")
    gates.append(_gate("delta_BU = 2*omega_corner", abs(b["delta_BU"] - 2 * b["omega_corner"]) < float(TOL_MP)))
    print()

    print("3. THREE CONNECTIONS")
    print("-" * 5)
    print("  Each row is a transport rule evaluated on the BU dual-pole loop and")
    print("  on the palindrome. geodesic and origin_gyr are Fermi-Walker.")
    print("  relative_boost is the product of successive boosts in one inertial")
    print("  frame. chart complete is the Cartesian Thomas path-ordered")
    print("  integral. chart spherical z is the same integral in a singular")
    print("  spherical chart. G_sph is the spherical offset from delta_BU.")
    c, l, ch = conn["can"], conn["lab"], conn["chart"]
    print(f"  {'transport':<28} {'BU':>14}  {'palindrome':>14}")
    print(f"  {'geodesic':<28} {c['geodesic_BU']:14.10f}  {c['geodesic_pal']:14.10f}")
    print(f"  {'origin_gyr':<28} {c['origin_gyr_BU']:14.10f}  {c['origin_gyr_pal']:14.10f}")
    print(f"  {'dual-pole word':<28} {c['word']:14.10f}  {'n/a':>14}")
    print(f"  {'defect BU tri':<28} {c['defect_BU_tri']:14.10f}  {'n/a':>14}")
    print(f"  {'inertial-frame boost product':<28} {l['relative_boost_BU']:14.10f}  {l['relative_boost_pal']:14.10f}")
    print(f"  {'chart complete (Thomas)':<28} {ch['omega_chart_complete_BU']:14.10f}  {ch['omega_chart_complete_pal']:14.10f}")
    print(f"  {'chart spherical z':<28} {ch['omega_chart_sph_BU']:14.10f}  {ch['omega_chart_sph_pal']:14.10f}")
    print(f"  {'G_sph = sph-can':<28} {ch['offset_BU']:14.10f}  {ch['offset_pal']:14.10f}")
    print()

    print("4. CANONICAL UNIQUENESS")
    print("-" * 5)
    print("  The dual-pole word, origin gyration, geodesic transvection, and")
    print("  Ungar defect on ONA-BU+-BU- are compared with delta_BU. Closed")
    print("  forms of omega_corner, omega0, and phi_SU2 are checked against")
    print("  their algebraic expressions.")
    d = b["delta_BU"]
    tol = float(TOL_MP)
    for label, val in (
        ("dual-pole word", c["word"]),
        ("origin-gyr BU", c["origin_gyr_BU"]),
        ("origin-gyr pal", c["origin_gyr_pal"]),
        ("geodesic BU", c["geodesic_BU"]),
        ("geodesic pal", c["geodesic_pal"]),
        ("Ungar defect ONA-BU+-BU-", c["defect_BU_tri"]),
    ):
        gates.append(_gate(f"{label} == delta_BU", abs(val - d) < tol))
    gates.append(_gate("corner defect == omega_corner", abs(c["defect_corner"] - b["omega_corner"]) < tol))
    gates.append(_gate("delta_BU == BU_HOLONOMY_ANGLE", abs(d - float(BU_HOLONOMY_ANGLE)) < 1e-12))
    gates.append(_gate("Delta == APERTURE_GAP", abs(b["Delta"] - float(APERTURE_GAP)) < 1e-12))
    gates.append(_gate("rho == RHO", abs(b["rho"] - float(RHO)) < 1e-12))
    gates.append(
        _gate(
            "omega_corner == TW(ona,ma,pi/2)",
            abs(b["omega_corner"] - _mpf(tw_angle_unequal(t.theta_ona, t.m_a, mp.pi / 2))) < tol,
        )
    )
    gates.append(_gate("omega0 closed form", abs(b["omega0"] - _mpf(2 * mp.atan((5 - 3 * mp.sqrt(2)) / 7))) < tol))
    gates.append(_gate("phi_SU2 closed form", abs(b["phi_SU2"] - _mpf(2 * mp.acos((1 + 2 * mp.sqrt(2)) / 4))) < tol))
    print()

    print("5. CHART CONNECTION")
    print("-" * 5)
    print("  Cartesian Palge-Pfeifer path-ordered Thomas integral, Richardson")
    print("  extrapolated from step n and 2n, compared with the geodesic angle.")
    print("  Spherical charts are singular at poles and at rest. G = th - delta_BU")
    print("  is the chart offset. dim is the number of axes spanned by the path.")
    gates.extend(print_chart(b, conn, cache))

    print("6. INERTIAL-FRAME BOOST COMPOSITION")
    print("-" * 5)
    print("  th_can is the Fermi-Walker angle. th_inert is the net rotation of")
    print("  the boost product in one inertial frame. F_sc = th_inert - th_can.")
    print("  th_RF is the angle of R_can^{-1} R_inert. A word closes to a pure")
    print("  rotation only when the leftover boost vanishes.")
    gates.extend(print_lab(b, conn, cache))

    print("7. REACHABILITY")
    print("-" * 5)
    print("  Every closed walk of length two through five on the four payload")
    print("  points is enumerated. theta is the rotation angle, mult is how many")
    print("  walks share that angle, min_L is the shortest such walk. Fermi-Walker")
    print("  and inertial-frame products are listed separately.")
    gates.extend(print_reachability(b, conn, cache))

    print("8. WIGNER PAIR / THOMAS 1-FORM")
    print("-" * 5)
    print("  Each pair angle is the Wigner rotation of two origin boosts. The")
    print("  Thomas 1-form is the infinitesimal version of the same rotation.")
    print("  Its path-ordered exponential on the BU loop equals twice the")
    print("  ONA-BU pair angle.")
    gates.extend(print_wigner_thomas(b, cache, conn))
    print()

    state = {
        "t": t,
        "cache": cache,
        "priors": p,
        "basis": b,
        "conn": conn,
        "lab_rows": conn["lab"].get("named_rows", []),
        "spec_can": conn.get("reach", {}).get("spec_can", []),
        "n_spec_can": conn.get("reach", {}).get("n_spec_can", 0),
        "n_spec_lab": conn.get("reach", {}).get("n_spec_lab", 0),
    }
    return gates, state


def main() -> None:
    gates, _state = run()
    if any(not ok for _, ok in gates):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
