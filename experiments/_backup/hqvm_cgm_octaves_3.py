#!/usr/bin/env python3
"""
hqvm_cgm_octaves_3.py

Delta-ruler octave phase, aperture-as-comma, frozen physical landmarks,
allometry/gravity octave readouts, and cross-chart octave covariance with nulls.

No printing. Invoked by hqvm_cgm_octaves_run.py.
"""
from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

_REPO = Path(__file__).resolve().parents[1]
_EXP = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from hqvm_cgm_octaves_common import (
    C1,
    C2,
    C3,
    CHIRALITY_SPACE,
    DELTA,
    DELTA_CONT,
    DELTA_DEPTH4,
    DELTA_DYADIC_8,
    DELTA_BU,
    E_CS_GEV,
    EVEN_H,
    G_KERNEL,
    H_CARD,
    M_A,
    NULL_SEED,
    OMEGA,
    PHI_SU2,
    PREDECESSOR_48,
    Q_G,
    RHO,
    STATUS_CONTINUUM,
    STATUS_DERIVED,
    STATUS_EMPIRICAL,
    STATUS_EXACT,
    STATUS_HYP,
    STATUS_KERNEL_EXACT,
    STATUS_NULL,
    TICKS_PER_OCTAVE,
    V_EV,
    V_GEV,
    aperture_comma_table,
    best_dyadic_denom256,
    circular_tick_distance,
    curvature_manifestations_table,
    energy_octave_coordinate,
    foundation_lock_scalars,
    interval_by_name,
    nearest_octave_boundary_ticks,
    octave_coordinate,
    stf_quadrupole_layer,
    ticks_of_energy,
)

try:
    from hqvm_cgm_trestleboard_common import RHO as _RHO_TB  # noqa: F401
    from hqvm_cgm_trestleboard_1 import Trestleboard
except Exception:  # pragma: no cover
    Trestleboard = None  # type: ignore


@dataclass
class Landmark:
    name: str
    sector: str
    E_eV: float
    source: str


@dataclass
class Octaves3Census:
    aperture_cents: Dict[str, object]
    foundation_locks: Dict[str, object]
    ym_shadow_locks: Dict[str, object]
    curvature_audit: Dict[str, object]
    stf_layer: Dict[str, object]
    landmarks: List[Dict[str, object]]
    clustering: Dict[str, object]
    ew_masses: List[Dict[str, object]]
    nuclear: List[Dict[str, object]]
    gravity: Dict[str, object]
    allometry: List[Dict[str, object]]
    cmb_harmonics: List[Dict[str, object]]
    forty_eight: List[Dict[str, object]]
    cross_chart: Dict[str, object]
    audits: List[Tuple[str, str, float]]  # name, status, p_hat
    gates: List[Tuple[str, bool]]
    gate_kinds: Dict[str, str]


def _foundation_locks_block() -> Dict[str, object]:
    """Certify §4 closure identities; import gravity tau_G when available."""
    L = foundation_lock_scalars()
    checks = {
        "wave_normalization_QG_ma2_half": abs(L["Q_G_m_a2"] - 0.5) < 1e-12,
        "QG_ma2_is_1_2": abs(L["Q_G_m_a2"] - 0.5) < 1e-12,
        "sp_over_ma2_is_4pi2": abs(L["s_p_over_m_a2"] - L["four_pi2"]) < 1e-12,
        "gyro_sum_is_pi": abs(L["gyro_sum"] - math.pi) < 1e-12,
        "chirality_space_2_3": abs(L["chirality_space"] - 2.0 / 3.0) < 1e-15,
        "48_in_EVEN_H": PREDECESSOR_48 in EVEN_H,
        "M_shell_is_6x32": abs(L["M_shell_over_32"] - 6.0) < 1e-12,
        "alpha0_zeta_matches_rho4": abs(L["alpha0_zeta"] - L["rho4_over_pi_sqrt3"]) < 1e-12,
    }
    claim_kinds = {
        "wave_normalization_QG_ma2_half": STATUS_KERNEL_EXACT,
        "QG_ma2_is_1_2": STATUS_KERNEL_EXACT,
        "sp_over_ma2_is_4pi2": STATUS_KERNEL_EXACT,
        "gyro_sum_is_pi": STATUS_KERNEL_EXACT,
        "chirality_space_2_3": STATUS_KERNEL_EXACT,
        "48_in_EVEN_H": STATUS_KERNEL_EXACT,
        "M_shell_is_6x32": STATUS_KERNEL_EXACT,
        "alpha0_zeta_matches_rho4": STATUS_KERNEL_EXACT,
    }
    tau_G = None
    try:
        from hqvm_gravity_common import tau_G_formula as _tg

        tau_G = float(_tg)
    except Exception:
        tau_G = float(OMEGA) * DELTA * float(RHO) ** 5 * (
            1.0 - 4.0 * float(RHO) * DELTA**2
        )
    L = dict(L)
    L["tau_G_leading"] = tau_G
    L["phi_SU2"] = PHI_SU2
    L["checks"] = checks
    L["claim_kinds"] = claim_kinds
    L["all_checks_pass"] = all(checks.values())
    L["status"] = STATUS_EXACT if L["all_checks_pass"] else STATUS_EMPIRICAL
    return L


def _aperture_cents_block() -> Dict[str, object]:
    pc = interval_by_name("pythagorean_comma")
    sc = interval_by_name("syntonic_comma")
    cgm_cents = DELTA_CONT * 1200.0
    pc_cents = pc.cents
    sc_cents = sc.cents
    depth4_cents = DELTA_DEPTH4 * 1200.0  # 25 exactly
    dyadic8_cents = DELTA_DYADIC_8 * 1200.0
    cycle_48 = 48.0 * DELTA_CONT
    cycle_residue_oct = 1.0 - cycle_48
    k, err = best_dyadic_denom256(DELTA_CONT)
    return {
        "Delta_cont_cents": cgm_cents,
        "Delta_cents": cgm_cents,
        "PC_cents": pc_cents,
        "SC_cents": sc_cents,
        "Delta_depth4_cents": depth4_cents,
        "Delta_kernel_cents": depth4_cents,
        "Delta_dyadic_8_cents": dyadic8_cents,
        "Delta_byte_cents": dyadic8_cents,
        "cgm_over_PC": cgm_cents / pc_cents,
        "semitone_ratio_2_1_12": 2.0 ** (1.0 / 12.0),
        "48_Delta": cycle_48,
        "48_Delta_residue_octaves": cycle_residue_oct,
        "48_Delta_residue_cents": cycle_residue_oct * 1200.0,
        "ticks_per_octave": TICKS_PER_OCTAVE,
        "best_dyadic_k_for_Delta_cont": k,
        "best_dyadic_err": err,
        "Delta_dyadic_8_is_best": k == 5 and abs(err - abs(DELTA_CONT - DELTA_DYADIC_8)) < 1e-15,
        "comma_table_n": len(aperture_comma_table()),
        "status": STATUS_HYP,
    }


def _ym_shadow_locks_block() -> Dict[str, object]:
    """YM mass-gap note certificates: E_grade2, m_gap, Delta_W(256), lim=1/2."""
    L = foundation_lock_scalars()
    n = 256
    delta_w = n / (2.0 * (n - 1.0))
    delta_w_lim = 0.5
    e_g2 = float(L["E_grade2_GeV"])
    m_gap = float(L["m_gap_RouteA_GeV"])
    # Glueball 0++ window annotation only (not a gate)
    glueball_lo = 1.5
    glueball_hi = 1.8
    checks = {
        "E_grade2_eq_v_Delta2": abs(e_g2 - V_GEV * (DELTA_CONT**2)) < 1e-12,
        "m_gap_eq_C2_E_grade2": abs(m_gap - float(C2) * e_g2) < 1e-12,
        "Delta_W_256_eq_128_255": abs(delta_w - 128.0 / 255.0) < 1e-15,
        "lim_Delta_W_eq_1_2": abs(delta_w_lim - 0.5) < 1e-15,
        "QG_ma2_eq_1_2": abs(float(L["Q_G_m_a2"]) - 0.5) < 1e-12,
    }
    claim_kinds = {
        "C2": STATUS_KERNEL_EXACT,
        "E_grade2_eq_v_Delta2": STATUS_DERIVED,
        "m_gap_eq_C2_E_grade2": STATUS_CONTINUUM,
        "m_gap_in_glueball_window_anno": STATUS_EMPIRICAL,
        "Delta_W_256_eq_128_255": STATUS_KERNEL_EXACT,
        "lim_Delta_W_eq_1_2": STATUS_KERNEL_EXACT,
        "QG_ma2_eq_1_2": STATUS_KERNEL_EXACT,
    }
    return {
        "E_grade2_GeV": e_g2,
        "m_gap_RouteA_GeV": m_gap,
        "C2": float(C2),
        "Delta_W_256": delta_w,
        "Delta_W_128_over_255": 128.0 / 255.0,
        "Delta_W_limit": delta_w_lim,
        "glueball_0pp_window_GeV": (glueball_lo, glueball_hi),
        "m_gap_in_glueball_window_anno": glueball_lo <= m_gap <= glueball_hi,
        "checks": checks,
        "claim_kinds": claim_kinds,
        "all_checks_pass": all(checks.values()),
        "status": STATUS_CONTINUUM,
    }


def _landmark_table() -> List[Landmark]:
    # Frozen measured / CGM-placed energies (eV). Imports values used in trestleboard.
    mW = 80.379e9
    mZ = 91.1876e9
    mH = 125.10e9
    mT = 172.76e9
    th = 8.3557335
    deut = 2.224e6  # MeV -> eV
    # Strong bare v*Delta^3 in eV
    strong_bare = V_GEV * (DELTA**3) * 1e9
    # Barriers (MeV -> eV) from trestleboard literature placements
    vb_dt = 0.444e6
    vb_pb = 1.861e6
    return [
        Landmark("v_EW", "electroweak", V_EV, "E_EW_GEV"),
        Landmark("W", "electroweak", mW, "PDG"),
        Landmark("Z", "electroweak", mZ, "PDG"),
        Landmark("Higgs", "electroweak", mH, "PDG"),
        Landmark("Top", "electroweak", mT, "PDG"),
        Landmark("Th-229m", "nuclear", th, "Zhang 2024"),
        Landmark("Deuteron_BE", "nuclear", deut, "CODATA"),
        Landmark("strong_bare_vDelta3", "nuclear", strong_bare, "CGM grammar"),
        Landmark("Vb_DT", "fusion", vb_dt, "trestleboard"),
        Landmark("Vb_pB11", "fusion", vb_pb, "trestleboard"),
        Landmark("E_CS", "gravity", E_CS_GEV * 1e9, "Planck UV anchor"),
        Landmark("E_CS_IR", "electroweak", 6.24e9, "optical conjugacy IR"),
        Landmark("kT_310K", "chemical", 0.0267, "approx kT @ 310K eV"),
        Landmark("Ea_MTE_0.645", "chemical", 0.645, "allometry MTE band"),
    ]


def _landmark_rows(landmarks: Sequence[Landmark]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for lm in landmarks:
        if lm.E_eV <= 0:
            continue
        coord = energy_octave_coordinate(lm.E_eV, E0=V_EV)
        phase, dist_oct = nearest_octave_boundary_ticks(coord.ticks)
        half = TICKS_PER_OCTAVE / 2.0
        dist_half = min(abs(phase - half), TICKS_PER_OCTAVE - abs(phase - half))
        # distance to dyadic rational phases k/8 of an octave
        dyadic = min(
            circular_tick_distance(phase, (k / 8.0) * TICKS_PER_OCTAVE)
            for k in range(8)
        )
        rows.append(
            {
                "name": lm.name,
                "sector": lm.sector,
                "E_eV": lm.E_eV,
                "octaves_from_v": coord.octave_index + coord.residual,
                "octave_index": coord.octave_index,
                "octave_residual": coord.residual,
                "ticks": coord.ticks,
                "octave_phase_ticks": phase,
                "dist_octave_boundary_ticks": dist_oct,
                "dist_half_octave_ticks": dist_half,
                "dist_dyadic_8_ticks": dyadic,
                "source": lm.source,
                "status": STATUS_EMPIRICAL,
            }
        )
    return rows


def _clustering_audit(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Predeclared: concentration near octave / half / dyadic-8 boundaries."""
    if not rows:
        return {"status": STATUS_NULL, "n": 0}
    phases = [float(r["octave_phase_ticks"]) for r in rows]
    # mean circular distance to boundary / half / dyadic
    d_oct = [float(r["dist_octave_boundary_ticks"]) for r in rows]
    d_half = [float(r["dist_half_octave_ticks"]) for r in rows]
    d_dy = [float(r["dist_dyadic_8_ticks"]) for r in rows]
    mean_oct = sum(d_oct) / len(d_oct)
    mean_half = sum(d_half) / len(d_half)
    mean_dy = sum(d_dy) / len(d_dy)
    # Null: shuffle phases within sector-preserving random ticks uniform on [0, T)
    rng = random.Random(NULL_SEED + 2)
    n_null = 500
    null_oct = []
    for _ in range(n_null):
        fake = [rng.random() * TICKS_PER_OCTAVE for _ in phases]
        dists = [min(p, TICKS_PER_OCTAVE - p) for p in fake]
        null_oct.append(sum(dists) / len(dists))
    # p_hat: fraction of nulls with mean_oct_null <= observed (more concentrated)
    hits = sum(1 for m in null_oct if m <= mean_oct)
    p_hat = (hits + 1) / (n_null + 1)
    return {
        "n": len(rows),
        "mean_dist_octave_boundary": mean_oct,
        "mean_dist_half": mean_half,
        "mean_dist_dyadic_8": mean_dy,
        "uniform_expected_boundary": TICKS_PER_OCTAVE / 4.0,
        "null_mean_boundary_mean": sum(null_oct) / len(null_oct),
        "p_hat_more_concentrated_than_uniform": p_hat,
        "status": STATUS_HYP if p_hat < 0.05 else STATUS_NULL,
    }


def _ew_mass_rows() -> List[Dict[str, object]]:
    masses = {
        "Top": 172.76,
        "Higgs": 125.10,
        "Z": 91.1876,
        "W": 80.379,
        "v": V_GEV,
    }
    rows = []
    for name, gev in masses.items():
        octs = math.log2(V_GEV / gev)
        ticks = octs / DELTA
        nearest_48 = 48.0 * round(ticks / 48.0)
        rows.append(
            {
                "name": name,
                "GeV": gev,
                "octaves_below_v": octs,
                "ticks": ticks,
                "nearest_int_tick": round(ticks),
                "tick_residual": ticks - round(ticks),
                "nearest_48": nearest_48,
                "residual_48": ticks - nearest_48,
                "status": STATUS_EMPIRICAL,
            }
        )
    # W/Z
    log2_wz = math.log2(91.1876 / 80.379)
    rows.append(
        {
            "name": "Z_over_W",
            "GeV": 91.1876 / 80.379,
            "octaves_below_v": log2_wz,
            "ticks": log2_wz / DELTA,
            "nearest_int_tick": round(log2_wz / DELTA),
            "tick_residual": log2_wz / DELTA - round(log2_wz / DELTA),
            "code_gap_C2_C1": float(C2 - C1),
            "status": STATUS_EMPIRICAL,
        }
    )
    return rows


def _nuclear_rows() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    th = 8.3557335
    deut_eV = 2.224e6
    # ticks from EW
    for name, E in (("Th-229m", th), ("Deuteron", deut_eV)):
        ticks = ticks_of_energy(E, V_EV)
        octs = math.log2(V_EV / E)
        rows.append(
            {
                "name": name,
                "E_eV": E,
                "octaves_from_v": octs,
                "ticks": ticks,
                "status": STATUS_EMPIRICAL,
            }
        )
    if Trestleboard is not None:
        tb = Trestleboard()
        bare = tb.deuteron_bare_MeV()
        full = tb.deuteron_binding_MeV()
        rows.append(
            {
                "name": "deuteron_bare_MeV",
                "E_eV": bare * 1e6,
                "MeV": bare,
                "octaves_from_v": math.log2(V_GEV / (bare * 1e-3)),
                "ticks": ticks_of_energy(bare * 1e6, V_EV),
                "status": STATUS_EXACT,
            }
        )
        rows.append(
            {
                "name": "deuteron_full_MeV",
                "E_eV": full * 1e6,
                "MeV": full,
                "octaves_from_v": math.log2(V_GEV / (full * 1e-3)),
                "ticks": ticks_of_energy(full * 1e6, V_EV),
                "abs_err_MeV_vs_2.224": abs(full - 2.224),
                "status": STATUS_EMPIRICAL,
            }
        )
        # forced class (6,2) prediction vs Th
        try:
            from hqvm_cgm_trestleboard_1 import default_grammar
            from hqvm_cgm_trestleboard_common import ClosureClass

            cls = ClosureClass(6, 2, "Nuclear spinorial", True, True, True)
            E_pred = tb.predict_E_eV(cls)
            n_pred = tb.n_of_E(E_pred)
            n_th = tb.n_of_E(th)
            rows.append(
                {
                    "name": "Th_vs_class_6_2",
                    "E_pred_eV": E_pred,
                    "E_th_eV": th,
                    "n_pred": n_pred,
                    "n_th": n_th,
                    "tick_residual": n_th - n_pred,
                    "octaves_residual": (n_th - n_pred) * DELTA,
                    "status": STATUS_EMPIRICAL,
                }
            )
        except Exception as exc:  # pragma: no cover
            rows.append({"name": "Th_vs_class_6_2", "error": str(exc), "status": STATUS_HYP})
    return rows


def _gravity_block() -> Dict[str, object]:
    log2_qg = math.log2(Q_G)
    log2_d = math.log2(24.0)
    log2_gk = math.log2(G_KERNEL)
    oct_planck = math.log2(E_CS_GEV / V_GEV)
    n_e = 48**2
    L = foundation_lock_scalars()
    return {
        "Q_G": Q_G,
        "log2_Q_G": log2_qg,
        "log2_Q_G_over_2pi": math.log2(Q_G / (2.0 * math.pi)),
        "D_shell": 24.0,
        "log2_D": log2_d,
        "G_kernel": G_KERNEL,
        "log2_G_kernel": log2_gk,
        "log2_Q_minus_log2_D": log2_qg - log2_d,
        "octaves_EW_to_Planck": oct_planck,
        "N_e_48sq": n_e,
        "N_e_over_ln2_octaves": n_e / math.log(2.0),
        "m_a": float(M_A),
        "log2_m_a": math.log2(float(M_A)),
        "m_a_over_inv_sqrt_2pi": float(M_A) * math.sqrt(2.0 * math.pi),
        "abs_log2_m_a_vs_half_inv_sqrt_2pi": abs(
            math.log2(float(M_A) * math.sqrt(2.0 * math.pi)) - (-1.0)
        ),
        "rho5": L["rho5"],
        "m_gap_RouteA_GeV": L["m_gap_RouteA_GeV"],
        "E_grade2_GeV": L["E_grade2_GeV"],
        "C2": float(C2),
        "status": STATUS_HYP,
    }


def _allometry_rows() -> List[Dict[str, object]]:
    # Exponents as octave conversion factors (octaves_out per mass octave)
    exps = [
        ("Kleiber_3/4", 0.75),
        ("surface_2/3", 2.0 / 3.0),
        ("computational_1/2", 0.5),
        ("a_time_1/4", 0.25),
        ("a_eg_3/16", 3.0 / 16.0),
        ("heart_rate_-1/4", -0.25),
    ]
    rows = []
    for name, a in exps:
        rows.append(
            {
                "name": name,
                "exponent": a,
                "octaves_out_per_mass_octave": a,
                "ticks_out_per_mass_octave": a * TICKS_PER_OCTAVE,
                "status": STATUS_EXACT,
            }
        )
    # mu-band width 3/4 - 2/3
    width = 0.75 - (2.0 / 3.0)
    rows.append(
        {
            "name": "mu_band_width",
            "exponent": width,
            "octaves_out_per_mass_octave": width,
            "ticks_out_per_mass_octave": width * TICKS_PER_OCTAVE,
            "status": STATUS_EXACT,
        }
    )
    return rows


def _cmb_rows() -> List[Dict[str, object]]:
    ell0 = 37
    rows = []
    for n in range(1, 9):
        ell = ell0 * n
        octs = math.log2(n)
        rows.append(
            {
                "n": n,
                "ell": ell,
                "octaves_from_fundamental": octs,
                "is_integer_octave": abs(octs - round(octs)) < 1e-12,
                "is_stf_index_n5": n == 5,
                "status": STATUS_HYP,
            }
        )
    return rows


def _forty_eight_inventory() -> List[Dict[str, object]]:
    items = [
        ("48*Delta", 48.0 * DELTA, "aperture cycle"),
        ("ticks_per_octave", TICKS_PER_OCTAVE, "1/Delta"),
        ("depth4_bits", 48.0, "4*12"),
        ("N_e", float(48**2), "inflation e-folds"),
        ("|Omega|/2^6", float(OMEGA / H_CARD), "holographic ratio 64"),
        ("chirality_space", CHIRALITY_SPACE, "2/3"),
        ("C3", float(C3), "equator weight"),
    ]
    return [
        {"name": n, "value": v, "note": note, "status": STATUS_EXACT}
        for n, v, note in items
    ]


def _cross_chart(
    landmark_rows: Sequence[Dict[str, object]],
    ew_rows: Sequence[Dict[str, object]],
    grav: Dict[str, object],
) -> Dict[str, object]:
    """Shared octave residual table + S_oct vs permutation null."""
    objects: List[Dict[str, object]] = []
    for r in landmark_rows:
        objects.append(
            {
                "object_id": r["name"],
                "chart": r["sector"],
                "octave_residual": float(r["octave_residual"]),
                "octave_phase_ticks": float(r["octave_phase_ticks"]),
                "ticks": float(r["ticks"]),
            }
        )
    # Add gravity scalars as dimensionless octave coords
    objects.append(
        {
            "object_id": "log2_Q_G",
            "chart": "gravity",
            "octave_residual": float(grav["log2_Q_G"]) - math.floor(float(grav["log2_Q_G"])),
            "octave_phase_ticks": (
                (float(grav["log2_Q_G"]) % 1.0) * TICKS_PER_OCTAVE
            ),
            "ticks": float(grav["log2_Q_G"]) / DELTA,
        }
    )
    objects.append(
        {
            "object_id": "log2_m_a",
            "chart": "aperture",
            "octave_residual": abs(float(grav["log2_m_a"]))
            - math.floor(abs(float(grav["log2_m_a"]))),
            "octave_phase_ticks": (abs(float(grav["log2_m_a"])) % 1.0) * TICKS_PER_OCTAVE,
            "ticks": abs(float(grav["log2_m_a"])) / DELTA,
        }
    )

    tol = OCTAVE_TOL = 2.0  # ticks
    # Cross-chart pairs within tolerance on phase
    charts = sorted({o["chart"] for o in objects})
    hits = 0
    total = 0
    for i, a in enumerate(objects):
        for b in objects[i + 1 :]:
            if a["chart"] == b["chart"]:
                continue
            total += 1
            d = circular_tick_distance(
                float(a["octave_phase_ticks"]), float(b["octave_phase_ticks"])
            )
            if d <= tol:
                hits += 1
    s_oct = hits / total if total else float("nan")

    # Null: permute phases within each chart independently
    rng = random.Random(NULL_SEED + 3)
    null_scores = []
    by_chart: Dict[str, List[int]] = {}
    for i, o in enumerate(objects):
        by_chart.setdefault(str(o["chart"]), []).append(i)
    for _ in range(400):
        phases = [float(o["octave_phase_ticks"]) for o in objects]
        for idxs in by_chart.values():
            vals = [phases[i] for i in idxs]
            rng.shuffle(vals)
            for i, v in zip(idxs, vals):
                phases[i] = v
        h = t = 0
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                if objects[i]["chart"] == objects[j]["chart"]:
                    continue
                t += 1
                if circular_tick_distance(phases[i], phases[j]) <= tol:
                    h += 1
        null_scores.append(h / t if t else 0.0)
    null_hits = sum(1 for s in null_scores if s >= s_oct)
    p_hat = (null_hits + 1) / (len(null_scores) + 1)
    return {
        "n_objects": len(objects),
        "charts": charts,
        "tol_ticks": tol,
        "cross_pairs": total,
        "hits": hits,
        "S_oct": s_oct,
        "null_mean_S": sum(null_scores) / len(null_scores),
        "p_hat_S_ge_obs": p_hat,
        "status": STATUS_HYP if p_hat < 0.05 else STATUS_NULL,
        "objects": objects,
    }


def run_octaves_3() -> Octaves3Census:
    ap = _aperture_cents_block()
    locks = _foundation_locks_block()
    ym = _ym_shadow_locks_block()
    curv = curvature_manifestations_table()
    stf = stf_quadrupole_layer()
    lms = _landmark_rows(_landmark_table())
    clus = _clustering_audit(lms)
    ew = _ew_mass_rows()
    nuc = _nuclear_rows()
    grav = _gravity_block()
    allo = _allometry_rows()
    cmb = _cmb_rows()
    fre = _forty_eight_inventory()
    cross = _cross_chart(lms, ew, grav)

    one_third = curv["one_third_identity"]
    gates = [
        ("ticks_per_octave_positive", TICKS_PER_OCTAVE > 40),
        ("48_Delta_near_1", abs(48.0 * DELTA - 1.0) < 0.01),
        ("Delta_depth4_cents_25", abs(float(ap["Delta_depth4_cents"]) - 25.0) < 1e-9),
        ("Delta_dyadic_8_best_approx", bool(ap["Delta_dyadic_8_is_best"])),
        ("wave_normalization_axiom", bool(locks["checks"]["wave_normalization_QG_ma2_half"])),
        ("Q_G_one_octave_above_2pi", abs(float(grav["log2_Q_G_over_2pi"]) - 1.0) < 1e-12),
        (
            "m_a_is_half_inv_sqrt_2pi",
            abs(float(grav["m_a_over_inv_sqrt_2pi"]) - 0.5) < 1e-12,
        ),
        ("foundation_locks", bool(locks["all_checks_pass"])),
        ("QG_ma2_half", bool(locks["checks"]["QG_ma2_is_1_2"])),
        ("sp_ma2_4pi2", bool(locks["checks"]["sp_over_ma2_is_4pi2"])),
        ("gyro_sum_pi", bool(locks["checks"]["gyro_sum_is_pi"])),
        ("alpha0_zeta_identity", bool(locks["checks"]["alpha0_zeta_matches_rho4"])),
        ("ym_shadow_locks", bool(ym["all_checks_pass"])),
        ("stf_bulk_3968", bool(stf["all_five_shell_count"])),
        ("stf_dyadic_num_5", bool(stf["dyadic_numerator_is_5"])),
        ("mu_band_ticks_near_4", abs(allo[-1]["ticks_out_per_mass_octave"] - 4.0) < 0.1),
        ("cmb_n2_is_octave", bool(cmb[1]["is_integer_octave"])),
        ("m_gap_RouteA_positive", float(grav["m_gap_RouteA_GeV"]) > 1.0),
    ]
    gate_kinds = {
        "ticks_per_octave_positive": "internal_kernel_identity",
        "48_Delta_near_1": "derived_imported",
        "Delta_depth4_cents_25": "internal_kernel_identity",
        "Delta_dyadic_8_best_approx": "internal_kernel_identity",
        "wave_normalization_axiom": "internal_kernel_identity",
        "Q_G_one_octave_above_2pi": "internal_kernel_identity",
        "m_a_is_half_inv_sqrt_2pi": "internal_kernel_identity",
        "foundation_locks": "internal_kernel_identity",
        "QG_ma2_half": "internal_kernel_identity",
        "sp_ma2_4pi2": "internal_kernel_identity",
        "gyro_sum_pi": "internal_kernel_identity",
        "alpha0_zeta_identity": "internal_kernel_identity",
        "ym_shadow_locks": "continuum_bridge",
        "stf_bulk_3968": "internal_kernel_identity",
        "stf_dyadic_num_5": "internal_kernel_identity",
        "mu_band_ticks_near_4": "external_alignment",
        "cmb_n2_is_octave": "external_alignment",
        "m_gap_RouteA_positive": "continuum_bridge",
    }
    p_clus = float(clus.get("p_hat_more_concentrated_than_uniform", float("nan")))
    audits = [
        ("landmark_clustering", str(clus["status"]), p_clus),
        ("cross_chart_S_oct", str(cross["status"]), float(cross["p_hat_S_ge_obs"])),
        (
            "delta_BU_vs_phi_SU2_over_3",
            str(one_third["claim_status"]),
            float(one_third["abs_delta_BU_minus_phi_over_3"]),
        ),
    ]

    return Octaves3Census(
        aperture_cents=ap,
        foundation_locks=locks,
        ym_shadow_locks=ym,
        curvature_audit=curv,
        stf_layer=stf,
        landmarks=lms,
        clustering=clus,
        ew_masses=ew,
        nuclear=nuc,
        gravity=grav,
        allometry=allo,
        cmb_harmonics=cmb,
        forty_eight=fre,
        cross_chart=cross,
        audits=audits,
        gates=gates,
        gate_kinds=gate_kinds,
    )
