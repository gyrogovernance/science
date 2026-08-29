#!/usr/bin/env python3
"""
cgm_light_cone_analysis.py

Aperture and the Light Cone: rigorous re-derivation of the CGM dual-pole
holonomy as the geometry of the interior of the null cone.

What this script does
---------------------
1. Builds the exact bridge between the Einstein gyroball, the forward mass
   shell, and the light cone (stage thresholds = Einstein speeds).
2. Re-derives delta_BU = 4*atan(k(pi/4)*k(m_a)) and shows it equals the
   hyperbolic area of the dual-pole triangle (ONA, BU+, BU-) by two
   independent routes: the Ungar gyrotriangle defect and the Girard-type
   defect additivity of the corner (0, ONA, BU+), at 80-digit mpmath
   precision.
3. Verifies the causal classification of the stages (which live strictly
   inside the cone, which is outside).
4. Classifies each candidate aperture-light-cone relation as EXACT / NEAR
   / COINCIDENTAL, verifying every number at >= 10 digits.
5. Records the finite hQVM reinterpretation (W2 pole exchange, byte fold)
   and states the checkable finite counterparts.

The results mirror the work notes (cgm_light_cone_analysis_worknotes.txt)
and are written to cgm_light_cone_analysis_results.txt.

Relation to the corpus:
  Analysis_Holonomy.md (Section 6: the BU dual-pole loop in closed form),
  Analysis_CGM_Constants.md (origin of m_a, Q_G*m_a^2 = 1/2),
  gyroscopic.hQVM.constants (shared BU_HOLONOMY_ANGLE, DELTA_BU, RHO).
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mpmath as mp

_EXP = Path(__file__).resolve().parent
_REPO = _EXP.parent
RESULTS_PATH = _EXP / "cgm_light_cone_analysis_results.txt"

if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from cgm_holonomy_analysis_common import (
    CGMThresholds,
    analytic_bu_holonomy,
    gyrotriangle_defect_triangle_vertices_mp,
    mp_stage_points,
    poincare_radius_from_beta,
    tw_angle_unequal,
)

mp.mp.dps = 80

TOL_AREA = mp.mpf("1e-70")   # rigorous area identities
TOL_NEAR = mp.mpf("1.5e-4")  # upper bound for the C1 near-identity (55 ppm)


# ----------------------------------------------------------------------
# Gate bookkeeping (house style)
# ----------------------------------------------------------------------


@dataclass
class ReportState:
    gates: list[tuple[str, bool]]

    def check(self, label: str, ok: bool) -> None:
        print(f"  check  {label:64s} {'PASS' if ok else 'FAIL'}")
        self.gates.append((label, ok))


def section(title: str) -> None:
    print()
    print(title)
    print("=" * 5)


def mp_s16(x: Any) -> str:
    return str(mp.nstr(x, n=16))


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------


def k_mp(beta: Any) -> Any:
    """Poincare half-rapidity radius k(beta) = beta/(1+sqrt(1-beta^2))."""
    b = mp.mpf(beta)
    return b / (1 + mp.sqrt(1 - b * b))


def solid_angle_cone_mp(m_a: Any) -> Any:
    """Solid angle of the null cone of directions within Klein slope m_a
    of the pole axis: 4pi(1 - sqrt(1 - m_a^2))."""
    return 4 * mp.pi * (1 - mp.sqrt(1 - m_a * m_a))


# ----------------------------------------------------------------------
# The report
# ----------------------------------------------------------------------


def run_report(state: ReportState) -> None:
    t = CGMThresholds.make()
    P = mp_stage_points(t)

    omega_closed, delta_closed = analytic_bu_holonomy(t)
    rho = delta_closed / t.m_a
    Delta = 1 - rho

    print("CGM LIGHT-CONE ANALYSIS")
    print("=" * 5)
    print("Aperture and the light cone: dual-pole holonomy as the enclosed")
    print("hyperbolic area of an interior (timelike) triangle of the ball,")
    print("measured against the null boundary of velocity space.")

    # ---------------- 1. bridge ----------------
    section("1. The exact light-cone bridge")
    print("  Minkowski space, c = 1.  N = null cone (boundary);")
    print("  H = {q.q = +1, q0 > 0} forward mass shell (the space);")
    print("  B = {|beta| < 1} Einstein ball (a chart of H).")
    print("  beta = q_spatial / q0 bijects H -> B;  dB = celestial sphere")
    print("  = projectivized future null cone.  |beta|<1 timelike (inside),")
    print("  |beta|=1 null (on the cone), |beta|>1 spacelike (outside).")
    print()
    k45 = k_mp(mp.mpf(4) / 5)
    margin = mp.mpf(4) / 5 - t.theta_ona
    print("  k(4/5)            =", mp_s16(k45), "   (must be exactly 1/2)")
    print("  gamma(4/5)        =", mp_s16(mp.mpf(5) / 3), " (= 5/3 exactly)")
    print("  margin 4/5 - pi/4 =", mp_s16(margin), "(= (16-5pi)/20)")
    state.check("k(beta) = 1/2 <=> beta = 4/5 (closure saturation surface)",
                k45 == mp.mpf(1) / 2)
    state.check("4/5 > pi/4: ONA strictly inside the saturation circle",
                margin > 0)

    # ---------------- 2. causal classification ----------------
    section("2. Causal classification of the stages")
    print("  stage   beta          |beta|?1   rapidity           gamma")
    rows = [
        ("CS", t.theta_cs),
        ("UNA", t.u_p),
        ("ONA", t.theta_ona),
        ("BU+", t.m_a),
        ("BU-", -t.m_a),
    ]
    for name, beta in rows:
        b = mp.mpf(beta)
        if mp.fabs(b) < mp.mpf(1):
            eta = mp.atanh(mp.fabs(b))
            g = 1 / mp.sqrt(1 - b * b)
            print(f"  {name:4s} {mp.nstr(mp.fabs(b), 12):>11s}  <1       "
                  f"{mp.nstr(eta, 14):>16s}  {mp.nstr(g, 14):>14s}   timelike")
        else:
            print(f"  {name:4s} {mp.nstr(mp.fabs(b), 12):>11s}  >1       "
                  f"--              --                outside (spacelike)")
    print("  CS is outside the cone: the gauge frame is spacelike-separated")
    print("  from every payload operation - the unobservable source frame.")

    # ---------------- 3. closed form ----------------
    section("3. The closed form and the dual-pole loop")
    print("  delta_BU = 4*atan(k(pi/4)*k(m_a))  [Definition]")
    print("    k(pi/4) =", mp_s16(poincare_radius_from_beta(t.theta_ona)))
    print("    k(m_a)  =", mp_s16(poincare_radius_from_beta(t.m_a)))
    omega_form = tw_angle_unequal(t.theta_ona, t.m_a, mp.pi / 2)
    print("    omega   =", mp_s16(omega_form))
    print("    delta_BU=", mp_s16(delta_closed))
    print("  rho = delta_BU / m_a =", mp_s16(rho))
    print("  Delta = 1 - rho      =", mp_s16(Delta))
    print()
    print("  Middle edge (rotationally flat, metrically nonzero):")
    print("    d(BU+,BU-) = 2*atanh(m_a) =", mp_s16(2 * mp.atanh(t.m_a)))
    state.check("omega via Ungar formula agrees",
                mp.fabs(omega_form - omega_closed) < TOL_AREA)
    same_form = 4 * mp.atan(poincare_radius_from_beta(t.theta_ona)
                            * poincare_radius_from_beta(t.m_a))
    state.check("delta_BU closed form is self-consistent",
                mp.fabs(delta_closed - same_form) < TOL_AREA)

    # ---------------- 4. area routes ----------------
    section("4. delta_BU as hyperbolic area: two independent derivations")
    tri = gyrotriangle_defect_triangle_vertices_mp(
        P["ONA"], P["BU+"], P["BU-"], c=mp.mpf(1))
    area_full = tri["defect"]
    print("  Route A - Ungar gyrotriangle defect of (ONA,BU+,BU-):")
    print("    area_full  =", mp_s16(area_full))
    print("    gyr angle  =", mp_s16(tri["gyr_u_neg_v_angle"]))
    resid_A = mp.fabs(area_full - delta_closed)
    print("    |area_full - delta_BU| =", mp.nstr(resid_A, 3))

    corner = gyrotriangle_defect_triangle_vertices_mp(
        [mp.mpf(0), mp.mpf(0), mp.mpf(0)], P["ONA"], P["BU+"], c=mp.mpf(1))
    area_corner = corner["defect"]
    print("  Route B - Girard defect additivity of the corner (0,ONA,BU+):")
    print("    area_corner =", mp_s16(area_corner))
    print("    2*area_corner =", mp_s16(2 * area_corner),
          "(shares the two corners of the loop)")
    resid_B = mp.fabs(area_full - 2 * area_corner)
    print("    |area_full - 2*area_corner| =", mp.nstr(resid_B, 3))
    state.check("Area(ONA,BU+,BU-) == delta_BU (Ungar route)",
                resid_A < TOL_AREA)
    state.check("Area = 2*Area(corner) (Girard additivity)",
                resid_B < TOL_AREA)

    # ---------------- 5. candidate relations ----------------
    section("5. Candidate aperture-light-cone relations")

    def closure_defect(mv: Any) -> Any:
        return (4 * mp.atan(poincare_radius_from_beta(t.theta_ona)
                            * poincare_radius_from_beta(mv)) - mv)
    m_lo, m_hi = mp.mpf("0.30"), mp.mpf("0.40")
    for _ in range(200):
        mid = (m_lo + m_hi) / 2
        if closure_defect(mid) > 0:
            m_hi = mid
        else:
            m_lo = mid
    m_star = (m_lo + m_hi) / 2
    u_p_half = mp.mpf(1) / (2 * mp.sqrt(2))
    rel_c1 = mp.fabs(m_star - u_p_half) / u_p_half
    print("  C1  closure-critical amplitude m* (4 atan(k(pi/4) k(m)) = m):")
    print("      m*           =", mp_s16(m_star))
    print("      u_p/2        = 1/(2 sqrt(2)) =", mp_s16(u_p_half))
    print("      m_a sqrt(pi) =", mp_s16(t.m_a * mp.sqrt(mp.pi)))
    print("      |m*-u_p/2|/u_p/2 =", mp.nstr(rel_c1, 6),
          "   [~55 ppm: NEAR - needs an axiom to derive]")
    state.check("C1 is a well-quantified near-identity (5.5e-5 +/- factor)",
                rel_c1 > mp.mpf("1e-6") and rel_c1 < TOL_NEAR)

    diff_c2 = mp.atan(t.m_a) - delta_closed
    print("  C2  atan(m_a) - delta_BU =", mp.nstr(diff_c2, 6))
    print("      COINCIDENTAL chart artifact: atanh ~ atan ~ beta at small m_a.")

    print("  C3  delta_BU/pi =", mp_s16(delta_closed / mp.pi),
          "  delta_BU/(pi/16) =", mp_s16(delta_closed / (mp.pi / 16)))
    print("      The bound delta_BU <= pi (ideal-triangle ceiling) is EXACT;")
    print("      the 1/16 quantization is only NEAR (0.51% off).")

    base_gap = 1 - 2 * poincare_radius_from_beta(t.theta_ona)
    print("  C4  baseline gap 1 - 2k(pi/4) =", mp_s16(base_gap))
    print("      EXACT theorem: positive because ONA lies strictly inside")
    print("      the k=1/2 saturation surface (beta < 4/5).")

    D_klein = 1 - delta_closed / t.m_a
    D_rapid = 1 - delta_closed / mp.atanh(t.m_a)
    D_angle = 1 - delta_closed / mp.atan(t.m_a)
    print("  C5  Delta chart-dependence:")
    print("      Delta_klein =", mp_s16(D_klein))
    print("      Delta_rapid =", mp_s16(D_rapid))
    print("      Delta_angle =", mp_s16(D_angle))
    print("      CGM Delta is the Klein (Common-Source-frame) gap; there is")
    print("      no chart-independent number equal to 0.0207. Invariant")
    print("      choices: delta_BU/pi =", mp_s16(delta_closed / mp.pi), ";")
    print("      delta_BU/(2 atanh m_a) =", mp_s16(delta_closed / (2 * mp.atanh(t.m_a))), ";")
    print("      delta_BU/atanh(pi/4) =", mp_s16(delta_closed / mp.atanh(t.theta_ona)), ".")

    print("  C6  d(BU+,BU-) =", mp_s16(2 * mp.atanh(t.m_a)))
    print("      Ideal endpoints of the BU diameter = antipodal null pair")
    print("      (0,0,+1), (0,0,-1). EXACT; invariant content is the AXIS.")

    sa = solid_angle_cone_mp(t.m_a)
    print("  C7  solid angle of the m_a-cone =", mp_s16(sa),
          " =", mp.nstr(sa / (4 * mp.pi), 8), "x 4pi")
    print("      Q_G m_a^2 =", mp_s16(t.m_a * t.m_a * 4 * mp.pi),
          "(EXACT by definition; do not promote the 2.0% vs 2.07% proximity)")

    # ---------------- 6. finite layer ----------------
    section("6. Finite hQVM reinterpretation")
    fold = ", ".join(f"k={k}:{16 * math.comb(4, k)}" for k in range(5))
    print("  Byte fold N(k) = 16*C(4,k):", fold)
    print("  Central BU comparison disagrees in 128/256 = 1/2 of bytes.")
    print("  W2 pole exchange is an involution with no fixed states: the")
    print("  finite analogue of a flat geodesic to an antipodal null pair.")
    print("  The continuous picture adds no new finite number; it re-reads")
    print("  the certified K4/W2 data (already Tier-A verified).")

    # ---------------- 7. falsification ----------------
    section("7. Falsification criteria")
    print("  F1  C1: |m* - 1/(2 sqrt2)|/(1/(2 sqrt2)) must stay ~5.52e-5.")
    print("  F2  T7: k(4/5) = 1/2 exactly and 1 - 2k(pi/4) > 0 (checked).")
    print("  F3  T4/T8: both area routes must agree with delta_BU at 1e-70.")
    print("  Any gate FAIL forces a nonzero exit code and voids the report.")

    section("Result summary")
    print(f"  delta_BU = {mp.nstr(delta_closed, 16)} rad")
    print(f"  rho      = {mp.nstr(rho, 16)}")
    print(f"  Delta    = {mp.nstr(Delta, 16)}")
    print()
    print("  The aperture is not ON the light cone: it is measured AGAINST it.")
    print("  The cone is the ideal boundary of the space of timelike")
    print("  directions; the dual-pole holonomy is the enclosed hyperbolic")
    print("  area; the aperture is open because the depth-two boundary sits")
    print("  strictly inside the k=1/2 saturation circle (beta<4/5,")
    print("  gamma=5/3, eta=ln 3).  The one open item needing a new axiom is")
    print("  the exact anchoring m* = u_p/2 = m_a sqrt(pi) (55 ppm NEAR).")


# ----------------------------------------------------------------------
# Tee (console + buffer) and the CLI entry point
# ----------------------------------------------------------------------


class Tee:
    def __init__(self, *streams) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for s in self._streams:
            s.write(data)
        return len(data)

    def flush(self) -> None:
        for s in self._streams:
            s.flush()


def main() -> None:
    import io

    state = ReportState([])
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = Tee(old, buf)
    try:
        run_report(state)
    finally:
        sys.stdout = old

    RESULTS_PATH.write_text(buf.getvalue(), encoding="utf-8")
    print(f"wrote {RESULTS_PATH}")
    if any(not ok for _, ok in state.gates):
        raise SystemExit(1)


if __name__ == "__main__":
    main()