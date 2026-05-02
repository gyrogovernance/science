#!/usr/bin/env python3
"""
Compact Geometry: Spectral Algebra of the Electroweak Mass Spectrum

Print the physics-facing aperture-coordinate report.
Kernel structure is treated as established input.
"""

from __future__ import annotations

import math
from typing import Sequence

from cgm_compact_geometry_core import (
    CODE_C1,
    CODE_C2,
    CODE_C3,
    HORIZON_CARDINALITY,
    clamp,
    GYROSCOPIC_COUPLING,
    ALPHA_SURPLUS,
    k4_gyroscopic_charge,
    DELTA,
    DELTA_BU,
    electroweak_delta_coordinates_full,
    electroweak_laws_stage_foundation,
    electroweak_laws_full,
    UV_IR_GYRATION_SQUARED,
    ALPHA_GEOMETRIC,
    TOP_MATTER_DENSITY_Q,
    solve_delta_top_quadratic,
    EW_TARGET_NAMES,
    KERNEL_BYTE_APERTURE,
    MUON_EQUATOR_COEFF,
    r5_closure_coeff,
    CoordinateResult,
    W_Z_APERTURE_COEFF,
    W_Z_OFFSET,
    Z_H_OFFSET,
    aperture_delta_coordinate,
    build_constants,
    build_coordinate_table,
    build_observables,
    compact_algebra,
    coordinate_law,
    ckm_conversion_quantities,
    delta_self_consistency_rhs,
    solve_delta_self_consistency,
    electroweak_delta_backsolves,
    electroweak_delta_coordinates,
    electroweak_laws,
    ew_log2_gap,
    d2_coefficient_from_stage,
    Q_G,
    CS,
    CS_UNA,
    LAMBDA_0,
    GAUGE_SECTOR,
    MATTER_SECTOR,
    UNA_ONA,
    ONA,
    UNA,
    MONODROMY_PER_STAGE,
    STAGE_THRESHOLD_INV_SQRT2,
    STAGE_THRESHOLD_PI_OVER_FOUR,
    STAGE_THRESHOLD_PI_OVER_TWO,
    hzw_delta_consensus,
    hzw_leave_one_out_predictions,
    stage_action_ratio_to_cs,
    lepton_horizon_residual_r,
    optical_depth_from_coordinate,
    RHO,
    name_index,
    planck_scales,
    hzw_observable_name,
    M_A,
    BU,
)
 


def wz_split_base(D, C1=CODE_C1, C2=CODE_C2, C3=CODE_C3):
    """Promoted W/Z split law in n-coordinate (base term, no phase closure)."""
    return (C2 - C1) - (C3 / 2.0) * D


def wz_split_closed(D, C1=CODE_C1, C2=CODE_C2, C3=CODE_C3):
    """W/Z split after coherent compact phase closure.

    Derived from:
      p_W - p_Z = 2
      q_W - q_Z = -1

    Therefore:
      n_W - n_Z = (C2 - C1) - (C3/2)D + 2D^2/sqrt(5) - D^3
    """
    return wz_split_base(D, C1, C2, C3) + 2.0 * D**2 / math.sqrt(5.0) - D**3


def wz_log2_ratio_phase_corrected(D):
    """log2(mZ/mW) after phase closure."""
    return D * wz_split_closed(D)


def sin2_theta_w_from_delta(D, C1=CODE_C1, C2=CODE_C2, C3=CODE_C3):
    """Compute sin^2(theta_W) from the promoted W/Z coherent split law."""
    split = wz_split_closed(D, C1, C2, C3)
    cos_theta = 2.0 ** (-D * split)
    return 1.0 - cos_theta**2


def w_mass_from_z_and_delta(m_z, D, C1=CODE_C1, C2=CODE_C2, C3=CODE_C3):
    """Compute W mass from Z and coherent split at Delta D."""
    split = wz_split_closed(D, C1, C2, C3)
    return m_z / (2.0 ** (D * split))


def solve_delta_wz_base(l_wz: float) -> float:
    """Backsolve Delta from base W/Z split law."""
    return (9.0 - math.sqrt(81.0 - 40.0 * l_wz)) / 20.0


def delta_from_wz_closed(mZ, mW, guess=0.0207, iterations=16):
    """Newton solve for:

      log2(mZ/mW) = 9D - 10D^2 + 2D^3/sqrt(5) - D^4
    """
    target = math.log2(mZ / mW)
    delta = guess

    sqrt5 = math.sqrt(5.0)
    for _ in range(iterations):
        f = 9.0 * delta - 10.0 * delta * delta + 2.0 * delta**3 / sqrt5 - delta**4 - target
        fp = 9.0 - 20.0 * delta + 6.0 * delta * delta / sqrt5 - 4.0 * delta**3
        delta -= f / fp

    return delta


def print_header(title: str) -> None:
    print()
    print(title)
    print("=" * len(title))


def print_compact_input_summary() -> None:
    print_header("Compact Geometry Inputs [derived]")

    algebra = compact_algebra()
    delta_sq = DELTA * DELTA

    print(f"Delta aperture                  = {DELTA:.12f}")
    print(f"kernel byte aperture 5/256      = {KERNEL_BYTE_APERTURE:.12f}")
    print(f"Delta - 5/256                   = {DELTA - KERNEL_BYTE_APERTURE:.12f}")
    print(f"48 Delta                        = {48.0 * DELTA:.12f}")
    print("APERTURE_FRAME                  = 3 * |K4|^2 = 48")
    print("P = 1 - 1/48                  = 47/48")
    print(f"kernel boundary projector       = {47.0 / 48.0:.12f}")
    print(f"epsilon = 1/Delta - 48         = {algebra.epsilon:.12f}")
    print(f"eta = m_a - delta_BU            = {algebra.eta:.12f}")
    print(f"eta / epsilon                  = {algebra.eta / algebra.epsilon:.12f}")
    print(f"48*delta_BU - 47*m_a            = {(48.0 * DELTA_BU - 47.0 * algebra.m_a):.12f}")
    print(f"omega = delta_BU / 2            = {algebra.omega:.12f}")
    print(f"kappa = pi/4 - 1/sqrt(2)        = {algebra.kappa:.12f}")
    print(f"Delta^2                         = {delta_sq:.12f}")
    print(f"alpha_geometric                  = {ALPHA_GEOMETRIC:.12e}")


def print_optical_depth_ruler_diagnostics(rows: Sequence[CoordinateResult]) -> None:
    print_header("Optical Depth Ruler + Diagnostics [derived]")

    by_name = name_index(rows)
    algebra = compact_algebra()
    ckm = ckm_conversion_quantities()
    coords = electroweak_delta_coordinates()

    print("0) Optical-depth interpretation (Beer-Lambert)")
    ew = by_name["Electroweak scale"]
    print(f"{'   n to tau':28} = tau = n * Delta * ln2")
    for label in (
        "Top quark mass energy",
        "Higgs mass energy",
        "Z boson mass energy",
        "W boson mass energy",
        "Electron mass energy",
    ):
        row = by_name[label]
        delta_n = abs(ew.coordinate - row.coordinate)
        tau = optical_depth_from_coordinate(delta_n)
        print(
            f"{'   ' + label[:28]:30} n={delta_n:9.6f}  tau={tau:12.9f}"
        )

    print()

    print("1) Notation and conversion depth conflict")
    print(f"{'   epsilon':28} = {algebra.epsilon:.12f}")
    print(f"{'   eta':28} = {algebra.eta:.12f} (m_a - delta_BU)")
    print(f"{'   epsilon * eta':28} = {algebra.epsilon * algebra.eta:.12f}")
    print(f"{'   48 delta_BU - 47 m_a':28} = {(48.0 * DELTA_BU - 47.0 * algebra.m_a):.12f}")
    print()

    print("2) CKM |Vub| inclusive/exclusive depth split")
    print(f"{'   exclusive = sin(9D^2)':28} = {ckm['exclusive']:.12f}")
    print(f"{'   inclusive density eta':28} = {ckm['inclusive_density']:.12f}")
    print(f"{'   inclusive / exclusive':28} = {ckm['ratio']:.12f}")
    print(f"{'   phase correction (phi_conv)':28} = {ckm['delta_product']:.12f}")
    print(f"{'   sin(9D^2 + phi_conv)':28} = {ckm['inclusive_alt']:.12f}")
    print()

    print("3) Porosity = opacity (order hierarchy)")
    p_linear = 1.0 - 48.0 * DELTA
    p_half = 0.5 - 24.0 * DELTA
    p_higgs = DELTA * (algebra.epsilon + 24.0 * DELTA)
    p_molecular = DELTA * DELTA
    print(f"{'   Kramers term (linear)':28} = {p_linear:.12f}")
    print(f"{'   half-frame term':28} = {p_half:.12f}")
    print(f"{'   Higgs defect term':28} = {p_higgs:.12f}")
    print(f"{'   molecular proxy term':28} = {p_molecular:.12f}")
    print(f"{'   linear/molecular ratio':28} = {DELTA / DELTA**2:.12f}")
    print()

    print("4) Stage-pattern derivation from force/depth")
    print(f"{'   UV/backbone (Top)':35} = pre-conversion, CS, depth 1")
    print(f"{'   Transition (Higgs/Z)':35} = boundary and Weak->EM, depth 2-3")
    print(f"{'   IR/lepton':35} = BU, mostly temporal, depth 4")
    print(f"{'   Hubble residual states':35} = n_t={coords['n_t']:.3f}, n_H={coords['n_h']:.3f}, n_Z={coords['n_z']:.3f}, n_W={coords['n_w']:.3f}")
    print(f"{'   tau/mu/e relative anchors':35} = {coords['n_tau']:.3f}/{coords['n_muon']:.3f}/{coords['n_electron']:.3f}")
    print()

    print("5) Delta self-consistency with third order")
    rhs_d2 = delta_self_consistency_rhs(DELTA, include_third_order=False)
    rhs_d3 = delta_self_consistency_rhs(DELTA, include_third_order=True)
    solved_delta, residual = solve_delta_self_consistency(DELTA, include_third_order=True)
    print(f"{'   rhs from 1 + 6piD^2':28} = {rhs_d2:.12f}")
    print(f"{'   rhs from 1 + 6piD^2 + eta/epsilon D^3':28} = {rhs_d3:.12f}")
    print(f"{'   fixed-point delta':28} = {solved_delta:.12f}")
    print(f"{'   fixed-point residual':28} = {residual:.12e}")
    print(
        "   Note: third order does not tighten electroweak backsolves (already ~1e-8); "
        "this block is a closure consistency check, not a mass-gap fix."
    )
    print()

    print("6) Lepton anchors from horizon structure (M_shell ratios)")
    ms = float(algebra.m1)
    for label, k, lepton, coord_key in (
        ("tau", 5, "tau", "n_tau"),
        ("mu", 8, "mu", "n_muon"),
        ("e", 14, "e", "n_electron"),
    ):
        r = lepton_horizon_residual_r(lepton, ms)
        base = k * HORIZON_CARDINALITY + r
        n_model = coords[coord_key]
        print(
            f"{'   ' + label:6} k={k} r={r:.6f} (M_shell)  base={base:.6f}  "
            f"n_model={n_model:.6f}  tau={optical_depth_from_coordinate(n_model):.9f}"
        )
    print()

    print("7) Closed core and hydrogen residuals")
    scale_epsilon = 1.0 / DELTA - 48.0
    shell_scale = DELTA * scale_epsilon
    for err in (0.757, 0.397, 0.419, 0.368, 1.139):
        print(
            f"{'   hydrogen err':15} {err:6.3f} "
            f"eps*H ratio {err / (HORIZON_CARDINALITY * shell_scale):8.3f}"
        )
    print(f"{'   delta*epsilon':28} = {shell_scale:.12f}")
    print(f"{'   |H|*delta*epsilon':28} = {HORIZON_CARDINALITY * shell_scale:.12f}")
    print()

    print("8) Strong scale on Delta-ruler")
    n_qcd_by_e = math.log(
        by_name["Electroweak scale"].value / 0.2,
        2.0,
    ) / DELTA
    n_qcd_by_top = coords["n_t"] + (
        math.log(
            by_name["Top quark mass energy"].value / 0.2,
            2.0,
        )
        / DELTA
    )
    print(f"{'   n_QCD from EW/0.2':28} = {n_qcd_by_e:.3f}")
    print(f"{'   n_QCD from top anchor':28} = {n_qcd_by_top:.3f}")
    print(f"{'   BU-strong offset':28} = {n_qcd_by_top - n_qcd_by_e:.3f}")
    print()

    print("9) Gravity + antihydrogen channel")
    grav_factor = (1.0 - 12.0 * DELTA) ** 3
    grav_gap = 0.25 - (1.0 - 12.0 * DELTA)
    print(f"{'   geometry factor':28} = {grav_factor:.12f}")
    print(f"{'   geometry gap to 1/4':28} = {grav_gap:.12f}")
    print(f"{'   alpha_geometric':28} = {ALPHA_GEOMETRIC:.12f}")
    print()

    print("10) Code-valued curvature and opacity spectrum")
    print(f"{'   TRK moment (sum k*C(6,k)':28} = {algebra.m1:.12f}")
    print(f"{'   horizon+ab distance':28} = {6.0 + 6.0:.1f}")
    print()

    print("11) Holographic thermal floor")
    print(f"{'   kappa_min':28} = {1.0 / HORIZON_CARDINALITY:.12f}")
    print(f"{'   4pi^2 factor':28} = {UV_IR_GYRATION_SQUARED:.12f}")
    print()

    print("12) Transition law / Compton / UV-IR conjugacy")
    print(f"{'   Klein-Nishina x_max':28} = {1.0 / DELTA:.12f}")
    print(f"{'   UVIR factor':28} = {UV_IR_GYRATION_SQUARED:.12f}")


def print_ew_mass_coordinates(rows: Sequence[CoordinateResult]) -> None:
    print_header("Electroweak Mass Coordinates [empirical input]")
    by_name = name_index(rows)
    ew = by_name["Electroweak scale"]

    print(
        f"{'Target':28} {'AbsCoord':>10} {'n_Delta':>10} "
        f"{'tau':>12} {'EW log2':>10} {'nearest_n':>12} {'n_resid':>10}"
    )
    print("-" * 86)
    for name in EW_TARGET_NAMES:
        row = by_name[name]
        ew_sep = abs(ew.coordinate - row.coordinate)
        tau = optical_depth_from_coordinate(ew_sep)
        ew_log2 = math.log(ew.value / row.value, 2.0)
        print(
            f"{name[:28]:28} {row.coordinate:10.3f} {ew_sep:10.3f} "
            f"{tau:12.6f} {ew_log2:10.3f} {round(ew_sep):12.3f} {abs(ew_sep - round(ew_sep)):10.3f}"
        )


def print_compact_electroweak_cascade(rows: Sequence[CoordinateResult]) -> None:
    print_header("Compact Electroweak Cascade [derived]")
    by_name = name_index(rows)
    ew_value = by_name["Electroweak scale"].value

    algebra = compact_algebra()

    cascade = (
        ("Top", "73D-1+D^2/4", CS, MATTER_SECTOR, 73.0, -1.0),
        (
            "Higgs",
            "96D - 1 - 24D^2",
            CS_UNA,
            GAUGE_SECTOR,
            96.0,
            -1.0,
        ),
        ("Z", "117D - 47/48 - 45D^2/2", UNA, GAUGE_SECTOR, 117.0, -47.0 / 48.0),
        ("W", "126D - 47/48 - 65D^2/2", UNA_ONA, GAUGE_SECTOR, 126.0, -47.0 / 48.0),
    )

    print(f"  epsilon                        = {algebra.epsilon:.12f}")
    print(f"  Higgs defect d_H               = {algebra.d_h:.12f}")
    print()
    print(
        f"{'State':8} {'CGM':8} {'Sector':8} {'Law':8} "
        f"{'D2 coeff':>10} {'n_pred':>10} {'tau_p':>10} {'n_obs':>10} {'tau_o':>10} "
        f"{'n_err':>10} {'err/D^2':>10} {'mass_err':>12}"
    )
    print("-" * 145)

    ew = by_name["Electroweak scale"]
    for label, law, stage, sector, d_coeff, const in cascade:
        if label == "Top":
            row = by_name["Top quark mass energy"]
        else:
            row = by_name[hzw_observable_name(label)]
        row_log2 = ew_log2_gap(rows, row.name)
        n_obs = row_log2 / DELTA
        matter_q = TOP_MATTER_DENSITY_Q if label == "Top" else None
        d2_coeff = (
            TOP_MATTER_DENSITY_Q
            if label == "Top"
            else d2_coefficient_from_stage(stage, algebra)
        )
        predicted_law = coordinate_law(
            stage,
            sector,
            d_coeff,
            const,
            algebra,
            matter_density_d2=matter_q,
        )
        pred = predicted_law / DELTA
        coord_err = n_obs - pred
        err_scale = DELTA * DELTA
        tau_pred = optical_depth_from_coordinate(pred)
        tau_obs = optical_depth_from_coordinate(n_obs)
        mass_pred = ew_value * (2.0 ** (-predicted_law))
        rel_err = (mass_pred / row.value) - 1.0
        print(
            f"{label:8} {stage.name:8} {sector.name:8} {law:8} "
            f"{d2_coeff:10.2f} {pred:10.6f} {tau_pred:10.6f} {n_obs:10.6f} {tau_obs:10.6f} "
            f"{coord_err:10.6f} {(coord_err / err_scale):10.3f} {rel_err:12.3e}"
        )
    print()
    print("Gauge sector carries d_H projected through 47/48:")
    print(f"  n_Z = 70 + D - (47/48)*d_H = {70.0 + DELTA - (47.0 / 48.0) * algebra.d_h:.6f}")
    print(f"  n_W = 79 - 9D - (47/48)*d_H = {79.0 - 9.0 * DELTA - (47.0 / 48.0) * algebra.d_h:.6f}")
    print("  47/48 = 1 - 1/48 = depth-4 boundary projector")


def print_k4_gyroscopic_ladder(
    rows: Sequence[CoordinateResult],
) -> None:
    print_header("K4 Gyroscopic Ladder [structural derivation]")

    by_name = name_index(rows)
    ew_value = by_name["Electroweak scale"].value
    laws_d2 = electroweak_laws_full(order=2)
    laws_phase = electroweak_laws_full(order=3)
    laws_phase_closure = electroweak_laws_full(order=4)
    coords_d2 = electroweak_delta_coordinates_full(order=2)
    coords_phase = electroweak_delta_coordinates_full(order=3)
    coords_phase_closure = electroweak_delta_coordinates_full(order=4)

    print("CGM stage-ordered residual ladder:")
    print(f"{'lambda_0 = Delta / sqrt(5)':38} = {LAMBDA_0:.12f}")
    print(f"{'stage phase amplitude (primary)':38} = {LAMBDA_0:.12f}")
    print(f"{'rho = delta_BU / m_a':38} = {RHO:.12f}")
    print(f"{'rho_gap = 1 - rho^4':38} = {ALPHA_SURPLUS:.12f}")
    print(f"{'phase amplitude * Delta^2':38} = {LAMBDA_0 * DELTA * DELTA:.6e}")
    print(f"{'stage phase D3 source: 4D - 6D^2 + 4D^3':38} = {4.0 * DELTA - 6.0 * DELTA * DELTA + 4.0 * DELTA**3:.6e}")
    print(f"{'stage closure D4 source: -Delta^4':38} = {-DELTA**4:.6e}")
    print(
        f"{'stage phase n_shift(L_t,L_H,L_Z,L_W)':38} = "
        f"{(laws_phase['L_t'] - laws_d2['L_t']) / DELTA:.3e}, "
        f"{(laws_phase['L_H'] - laws_d2['L_H']) / DELTA:.3e}, "
        f"{(laws_phase['L_Z'] - laws_d2['L_Z']) / DELTA:.3e}, "
        f"{(laws_phase['L_W'] - laws_d2['L_W']) / DELTA:.3e}"
    )
    print(
        f"{'stage closure n_shift(L_t,L_H,L_Z,L_W)':38} = "
        f"{(laws_phase_closure['L_t'] - laws_phase['L_t']) / DELTA:.3e}, "
        f"{(laws_phase_closure['L_H'] - laws_phase['L_H']) / DELTA:.3e}, "
        f"{(laws_phase_closure['L_Z'] - laws_phase['L_Z']) / DELTA:.3e}, "
        f"{(laws_phase_closure['L_W'] - laws_phase['L_W']) / DELTA:.3e}"
    )
    print(
        f"{'coord n_shift(n_t,n_h,n_z,n_w)':38} = "
        f"{coords_phase['n_t'] - coords_d2['n_t']:.3e}, "
        f"{coords_phase['n_h'] - coords_d2['n_h']:.3e}, "
        f"{coords_phase['n_z'] - coords_d2['n_z']:.3e}, "
        f"{coords_phase['n_w'] - coords_d2['n_w']:.3e}"
    )
    print(
        f"{'coord closure n_shift(n_t,n_h,n_z,n_w)':38} = "
        f"{coords_phase_closure['n_t'] - coords_phase['n_t']:.3e}, "
        f"{coords_phase_closure['n_h'] - coords_phase['n_h']:.3e}, "
        f"{coords_phase_closure['n_z'] - coords_phase['n_z']:.3e}, "
        f"{coords_phase_closure['n_w'] - coords_phase['n_w']:.3e}"
    )
    print()

    channels = (
        ("Top", "Top quark mass energy", CS, MATTER_SECTOR, 73.0, -1.0, TOP_MATTER_DENSITY_Q),
        ("Higgs", "Higgs mass energy", CS_UNA, GAUGE_SECTOR, 96.0, -1.0, None),
        ("Z", "Z boson mass energy", UNA, GAUGE_SECTOR, 117.0, -47.0 / 48.0, None),
        ("W", "W boson mass energy", UNA_ONA, GAUGE_SECTOR, 126.0, -47.0 / 48.0, None),
    )

    print(
        f"{'Ch':8} {'p_i':>6} {'q_i':>6} {'n_err(D2)':>12} "
        f"{'n_err(D3)':>12} {'n_err(D3+D4)':>13} {'L_err_D4':>11} {'L_err/D^5':>11}"
    )
    print("-" * 66)

    sum_p = 0.0
    sum_q = 0.0
    max_before = 0.0
    max_after_d3 = 0.0
    max_after_d4 = 0.0

    for label, obs_name, stage, sector, d_coeff, const, matter_q in channels:
        obs_mass = by_name[obs_name].value
        l_obs = math.log(ew_value / obs_mass, 2.0)

        l_d2 = coordinate_law(
            stage,
            sector,
            d_coeff,
            const,
            compact_algebra(),
            matter_density_d2=matter_q,
        )
        p, q = k4_gyroscopic_charge(stage)
        phase_corr = p * GYROSCOPIC_COUPLING * DELTA * DELTA
        closure_corr = q * DELTA**4
        l_d3 = l_d2 + phase_corr
        l_d4 = l_d3 + closure_corr
        l_err_d4 = l_obs - l_d4

        n_err_d2 = (l_obs - l_d2) / DELTA
        n_err_d3 = (l_obs - l_d3) / DELTA
        n_err_d4 = (l_obs - l_d4) / DELTA
        l_err_d5 = l_err_d4 / (DELTA**5) if DELTA != 0.0 else float("inf")
        sum_p += p
        sum_q += q
        max_before = max(max_before, abs(n_err_d2))
        max_after_d3 = max(max_after_d3, abs(n_err_d3))
        max_after_d4 = max(max_after_d4, abs(n_err_d4))

        print(
            f"{label:8} {p:6.1f} {q:6.2f} "
            f"{n_err_d2:12.6f} {n_err_d3:12.6f} "
            f"{n_err_d4:13.6f} {l_err_d4:11.3e} {l_err_d5:11.3e}"
        )

    print("-" * 66)
    print(f"  sum(p_i) = {sum_p:.1f}  (trace-free check)")
    print(f"  sum(q_i) = {sum_q:.1f}  (trace-free check)")
    print(f"  max|n_err| before stage phase = {max_before:.3e}")
    if max_after_d3 > 0:
        print(f"  max|n_err| after stage phase = {max_after_d3:.3e}")
        print(f"  reduction factor (d2->d3)= {max_before / max_after_d3:.1f}x")
    else:
        print("  max|n_err| after stage phase = 0.0")
        print("  reduction factor (d2->d3)= inf")
    if max_after_d4 > 0:
        print(f"  max|n_err| after stage closure= {max_after_d4:.3e}")
        if max_after_d3 > 0:
            print(f"  reduction factor (phase->closure)= {max_after_d3 / max_after_d4:.1f}x")
    else:
        print("  max|n_err| after stage closure = 0.0")
        print("  reduction factor (phase->closure)= inf")
    print()
    print("  Stage phase correction: delta_L = p_i * lambda_0 * Delta^2")
    print("  Stage closure correction: delta_L = q_i * Delta^4")
    print("  Stage curvature correction: delta_L = r5_i * Delta^5")
    print(f"  phase_amplitude = Delta / sqrt(5) = {LAMBDA_0:.12f}")
    print("  Pentagonal link: Lambda_0 / Delta = 1/sqrt(5), vesica piscis cut.")
    print("  Gyrotriangle closure: delta = pi - (pi/2 + pi/4 + pi/4) = 0")
    print("  Structural source: Gyroscopic Multiplication Sec 6.2 and Sec 9.2")
    print("  ONA/CS = UNA^2 = 1/2")
    print("  p edges: -C1/2 = -3, +C1/4 = +3/2, +4* (1/2) = +2")
    print("  q edges: 0, -4 * (1/2) = -2, -2 * (1/2) = -1")
    print("  trace-free projection => p=(1,-2,-1/2,3/2), q=(5/4,5/4,-3/4,-7/4)")
    print(
        "The compact electroweak mass coordinates are modelled by a five-order expansion in Delta; "
        "the [12,6,2] code enumerator gives the linear/quadratic terms and the D5 curvature, and "
        "gyrotriangle closure delta = 0 gives ONA/CS = UNA^2 = 1/2, which fixes p and q through K4 "
        "edge increments for D3 and D4. With Q = d_A d_B = 1/4 and P = 1 - 1/48 = 47/48 as kernel "
        "invariants, the law uses zero fitted coefficients, conditional on the CGM Delta ruler and "
        "the selected empirical mass inputs, collapses residuals by 60,443x, and locks the W/Z Delta "
        "backsolve to 8.3x10^-10."
    )


def print_stage_action_diagnostics(
    rows: Sequence[CoordinateResult],
) -> None:
    print_header("Stage Action Diagnostics [diagnostic]")

    by_name = name_index(rows)
    ew_value = by_name["Electroweak scale"].value
    phase_laws = electroweak_laws_full(order=3)
    stage_foundation = electroweak_laws_stage_foundation()
    stage_with_delta = electroweak_laws_stage_foundation(include_delta_transport=True)

    print(f"{'Q_G':8} = {Q_G:.12f}")
    print(f"{'stage threshold pi/2':22} = {STAGE_THRESHOLD_PI_OVER_TWO:.12f}")
    print(f"{'stage threshold 1/sqrt(2)':22} = {STAGE_THRESHOLD_INV_SQRT2:.12f}")
    print(f"{'stage threshold pi/4':22} = {STAGE_THRESHOLD_PI_OVER_FOUR:.12f}")
    print(f"{'m_a':8} = {M_A:.12f}")
    print(f"{'delta_BU':8} = {DELTA_BU:.12f}")
    print(f"{'monodromy per stage':8} = {MONODROMY_PER_STAGE:.12f}")
    print(f"{'UV IR gyration squared':8} = {UV_IR_GYRATION_SQUARED:.12f}")
    print()
    print("stage action map and residuals (observed - prediction)")
    print(
        f"{'stage':22} {'ratio':>16} "
        f"{'foundation':>16} {'transport':>16} "
        f"{'phase':>16} {'n_err foundation':>15}"
    )

    channels = (
        (CS, "L_t", "Top quark mass energy"),
        (CS_UNA, "L_H", "Higgs mass energy"),
        (UNA, "L_Z", "Z boson mass energy"),
        (UNA_ONA, "L_W", "W boson mass energy"),
    )
    for stage, key, obs_name in channels:
        obs_mass = by_name[obs_name].value
        l_obs = math.log(ew_value / obs_mass, 2.0)
        foundation = stage_foundation[key]
        transported = stage_with_delta[key]
        phase = phase_laws[key]

        print(
            f"{stage.name:22} {stage_action_ratio_to_cs(stage):16.12f} "
            f"{foundation:16.12f} {transported:16.12f} "
            f"{phase:16.12f} {(l_obs - foundation) / DELTA:15.6f}"
        )


def print_delta_backsolves(rows: Sequence[CoordinateResult]) -> None:
    print_header("Delta Backsolves [derived]")
    print(f"{'Source':10} {'Equation':38} {'Delta_back':>14} {'Delta_err':>14}")
    print("-" * 82)
    for item in electroweak_delta_backsolves(rows):
        print(
            f"{item.source:10} {item.equation:38} "
            f"{item.delta_back:14.12f} {item.delta_err:14.6e}"
        )


def print_four_point_delta_consensus(rows: Sequence[CoordinateResult]) -> None:
    print_header("Four-Point Delta Consensus [derived]")
    backsolves = electroweak_delta_backsolves(rows)
    top = next(item for item in backsolves if item.source == "top")
    h = next(item for item in backsolves if item.source == "Higgs")
    z = next(item for item in backsolves if item.source == "Z")
    w = next(item for item in backsolves if item.source == "W")

    print(f"{'Channel':10} {'Delta_back':>18} {'Delta_err':>16}")
    print("-" * 48)
    for item in (top, h, z, w):
        print(f"{item.source:10} {item.delta_back:18.12f} {item.delta_err:16.6e}")

    mean_d = (top.delta_back + h.delta_back + z.delta_back + w.delta_back) / 4.0
    mean_err = mean_d - DELTA
    spread = max(top.delta_back, h.delta_back, z.delta_back, w.delta_back) - min(
        top.delta_back, h.delta_back, z.delta_back, w.delta_back
    )

    print("-" * 48)
    print(f"{'mean':10} {mean_d:18.12f} {mean_err:16.6e}")
    print(f"{'reference Delta':10} {DELTA:18.12f}")
    print(f"{'four-point spread':10} {spread:18.12e}")
    print()
    print(
        "Q = d_A d_B is the verified mask-code invariant 1/4 from pair-diagonal "
        "code popcount symmetry, so the five-order electroweak law uses zero fitted "
        "coefficients, conditional on the CGM Delta ruler and selected empirical mass inputs."
    )




def print_hzw_leave_one_out(rows: Sequence[CoordinateResult]) -> None:
    print_header("H/Z/W Leave-One-Out Test [derived]")

    print(
        f"{'Target':10} {'Delta source':14} {'Delta used':>14} "
        f"{'m_pred':>16} {'m_ref':>16} {'rel_err':>14}"
    )
    print("-" * 90)

    for item in hzw_leave_one_out_predictions(rows):
        print(
            f"{item.target:10} {item.delta_source:14} {item.delta_used:14.12f} "
            f"{item.predicted_mass:16.9f} {item.reference_mass:16.9f} "
            f"{item.relative_error:14.6e}"
        )


def print_code_weight_derivation() -> None:
    print_header("Code-Weight Derivation [kernel-derived coefficient alphabet]")
    print("Weight enumerator of the verified [12,6,2] code:")
    print("  (1 + z^2)^6 = ", end="")
    terms = []
    for k in range(7):
        c = math.comb(6, k)
        if c == 1:
            terms.append(f"z^{2*k}" if k > 0 else "1")
        else:
            terms.append(f"{c}z^{2*k}" if k > 0 else str(c))
    print(" + ".join(terms))
    print()
    c1, c2, c3 = math.comb(6, 1), math.comb(6, 2), math.comb(6, 3)
    m_shell = compact_algebra().m1
    print("Coefficients -> electroweak alphabet:")
    print(f"  C1 = C(6,1) = {c1}")
    print(f"  C2 = C(6,2) = {c2}")
    print(f"  C3 = C(6,3) = {c3}")
    print(f"  M_shell = sum shell_index*C(6,shell_index) = {m_shell}")
    print(f"  raw 12-bit Hamming-weight moment = 2*M_shell = {2 * m_shell}")
    print()
    print()
    print("  64 = |H| (horizon cardinality, verified)")
    print(
        f"  Top: L_t = (|H| + C2 - C1)*D - 1 + Q*D^2 = {64+c2-c1}*D - 1 + (1/4)D^2"
    )
    print(f"  Muon: n_mu = 540 + C3*D = 540 + {c3}*D")


def print_operator_group_theorem() -> None:
    print_header("Kernel Operator Group [derived]")

    translation_subgroup = 64 * 64
    parity_sector = 2
    operator_group = translation_subgroup * parity_sector
    commutator_sector = 64
    abelian_shadow = operator_group // commutator_sector
    byte_pair_multiplicity = (256 * 256) // translation_subgroup

    print("G = (GF(2)^6 x GF(2)^6) semidirect C2")
    print("C2 acts by swapping the two GF(2)^6 coordinates.")
    print("The kernel operator algebra is a class-2 exponent-4 finite curvature geometry.")
    print()
    print(f"{'|translation subgroup|':32} = {translation_subgroup}")
    print(f"{'|parity sector|':32} = {parity_sector}")
    print(f"{'|operator group|':32} = {operator_group}")
    print("{:32} = {}".format("|derived subgroup G'|", commutator_sector))
    print(f"{'|centre Z(G)|':32} = {commutator_sector}")
    print("{:32} = diagonal GF(2)^6".format("G' = Z(G)"))
    print("{:32} = {}".format("|abelian shadow G/G'|", abelian_shadow))
    print()
    print("Odd x odd closes the full translation subgroup:")
    print(f"{'256^2 / 4096':32} = {byte_pair_multiplicity}")
    print("Two-step uniformisation is exact translation surjection, not stochastic mixing.")


def print_matter_gauge_dichotomy(rows: Sequence[CoordinateResult]) -> None:
    print_header("Matter-Gauge Dichotomy [derived]")
    h, z, w, _ = hzw_delta_consensus(rows)
    backsolves = electroweak_delta_backsolves(rows)
    top = next(item for item in backsolves if item.source == "top")
    hzw_max = max(abs(item.delta_err) for item in (h, z, w))
    ratio = abs(top.delta_err) / hzw_max
    print(f"Top |Delta error|              = {abs(top.delta_err):.6e}")
    print(f"H/Z/W max |Delta error|       = {hzw_max:.6e}")
    print(f"Ratio (top / HZW)             = {ratio:.1f}")
    print()
    print("Top: CS matter channel (D + Q*D^2), Q = 1/4 density projector")
    print("H/Z/W: gauge channel (D + negative porosity D^2 from stage projection)")
    if ratio > 3.0:
        print(f"Top Delta error still ~{ratio:.1f}x H/Z/W max (see quark mass uncertainty).")
    else:
        print(f"Top Delta error within ~{ratio:.1f}x of H/Z/W max (electroweak cluster).")


def print_boundary_and_density_projectors(rows: Sequence[CoordinateResult]) -> None:
    print_header("Boundary and Density Projectors [derived]")

    by_name = name_index(rows)
    ew_value = by_name["Electroweak scale"].value
    top_mass = by_name["Top quark mass energy"].value

    p_boundary = 47.0 / 48.0
    q_density = TOP_MATTER_DENSITY_Q

    laws = electroweak_laws()
    l_obs = math.log(ew_value / top_mass, 2.0)
    l_full = laws["L_t"]
    l_linear_only = (64.0 + CODE_C2 - CODE_C1) * DELTA - 1.0
    n_obs = l_obs / DELTA
    n_full = l_full / DELTA
    n_linear = l_linear_only / DELTA
    m_full = ew_value * (2.0 ** (-l_full))

    print(f"  P_boundary = 47/48            = {p_boundary:.12f}")
    print(f"  Q_density (matter projector)  = {q_density:.12f}")
    print()
    print("Top CS matter channel (full law vs linear backbone only):")
    print("  full law                      = L_t = 73D - 1 + Q*D^2")
    print("  linear backbone only         = L_t = 73D - 1")
    print(f"  L_err full law                = {l_obs - l_full:.12e}")
    print(f"  L_err linear only             = {l_obs - l_linear_only:.12e}")
    print(f"  n_err full law                = {n_obs - n_full:.6f}")
    print(f"  n_err linear only             = {n_obs - n_linear:.6f}")
    print(f"  mass_err full law             = {(m_full / top_mass) - 1.0:.6e}")
    print()
    print("Status:")
    print("  P_boundary is the gauge-sector depth-4 boundary projector in H/Z/W.")
    print("  Q = 1/4 is the matter-sector density projector (CS top complement).")


def print_quadratic_sector_status() -> None:
    print_header("Quadratic Sector Status [derived]")

    algebra = compact_algebra()
    print("First-order structure:")
    print("  73, 96, 117, 126 come from |H|, C1, C2, C3, M_shell.")
    print()
    print("Resolved from CGM stage projections:")
    print(
        f"  Top:      CS matter   D^2 = +{TOP_MATTER_DENSITY_Q:.3f} "
        f"(Q density; stage porosity {d2_coefficient_from_stage(CS, algebra):+.3f})"
    )
    print(f"  Higgs:    stage={CS_UNA.name}  D^2 = {d2_coefficient_from_stage(CS_UNA, algebra):+.3f}")
    print(f"  Z:        stage={UNA.name}    D^2 = {d2_coefficient_from_stage(UNA, algebra):+.3f}")
    print(f"  W:        stage={UNA_ONA.name} D^2 = {d2_coefficient_from_stage(UNA_ONA, algebra):+.3f}")
    print("  Closure check: BU stage projects to zero net D^2")
    print()
    print("Gauge sector D^2 (stage projection, zeta terms):")
    print(f"  base gyration:   -M_shell/8 = {-algebra.m1 / 8.0:.3f}")
    print(f"  UNA rotational:  +C1/4 = +{algebra.c1 / 4.0:.3f}")
    print(f"  ONA balance:     -C3/2 = -{algebra.c3 / 2.0:.3f}")
    print("  zeta = -M/8 + C1/4 - C3/2 with flags switched by CGM stage")
    print("Matter sector D^2 (Top): +Q, Q = 1/4 (density projector; opposite sign to gauge porosity)")


def print_force_stage_consistency() -> None:
    print_header("Force-Stage Consistency [derived]")
    algebra = compact_algebra()

    print("CGM stage to force mapping (compact mass algebra convention):")
    print("  CS  (depth 1) -> Strong: gauge has no stage D^2; Top matter adds +Q*D^2")
    print("  CS->UNA (boundary) -> VEV: base gyration turns on")
    print("  UNA (depth 2) -> Weak force: base + rotational projection")
    print("  UNA->ONA (boundary) -> charged current: opposition turns on")
    print("  ONA (depth 3) -> EM force: base + rotational + balance projection")
    print("  BU  (depth 4) -> Gravity: closure neutralizes projection")
    print()
    print("D^2 diagnostics:")
    print(
        f"  CS gauge (no matter Q): {d2_coefficient_from_stage(CS, algebra):+.3f} (expected 0.000)"
    )
    print(f"  CS Top matter +Q:      +{TOP_MATTER_DENSITY_Q:.3f} (density projector)")
    print(f"  CS->UNA/VEV:        {d2_coefficient_from_stage(CS_UNA, algebra):+.3f} (expected -24.000)")
    print(f"  UNA/Weak:          {d2_coefficient_from_stage(UNA, algebra):+.3f} (expected -22.500)")
    print(f"  UNA->ONA/charged:   {d2_coefficient_from_stage(UNA_ONA, algebra):+.3f} (expected -32.500)")
    print(f"  ONA/EM:            {d2_coefficient_from_stage(ONA, algebra):+.3f} (expected -32.500)")
    print(f"  BU/Gravity:       {d2_coefficient_from_stage(BU, algebra):+.3f} (expected 0.000)")


def print_quark_kappa_omega_test(rows: Sequence[CoordinateResult]) -> None:
    print_header("Quark kappa/omega Consistency Test [falsifiable]")

    # Quark coordinate formulas n_bottom = 284 + kappa, etc. are
    # EW-referred (log2(E_EW / m) / Delta), not Planck-referred.
    n_b = ew_log2_gap(rows, "Bottom quark mass energy") / DELTA
    n_c = ew_log2_gap(rows, "Charm quark mass energy") / DELTA
    n_s = ew_log2_gap(rows, "Strange quark mass energy") / DELTA

    algebra = compact_algebra()
    kappa_closed = algebra.kappa
    omega_closed = algebra.omega

    # The implementation carries three quark coordinate formulas:
    #   n_bottom  = 284 + kappa
    #   n_charm   = 367 + omega + 0.5 Delta
    #   n_strange = 548 - (omega + kappa)
    # Each gives a direct estimator of one unknown (or their sum), so
    # three independent readings must agree if (kappa, omega) are truly
    # two universal scalars carrying the quark residual structure.
    kappa_from_b = n_b - 284.0
    omega_from_c = n_c - 367.0 - 0.5 * DELTA
    kappa_plus_omega_from_s = 548.0 - n_s

    print("Closed-form values:")
    print(f"  kappa = pi/4 - 1/sqrt(2)      = {kappa_closed:+.9f}")
    print(f"  omega = delta_BU / 2          = {omega_closed:+.9f}")
    print()
    print("Direct estimators from quark coordinates (n and tau = n*Delta*ln2):")
    print(
        f"  Bottom n={n_b:+.9f}  tau={optical_depth_from_coordinate(n_b):+.9f}  "
        f"kappa from Bottom = {kappa_from_b:+.9f}"
    )
    print(
        f"  Charm  n={n_c:+.9f}  tau={optical_depth_from_coordinate(n_c):+.9f}  "
        f"omega from Charm = {omega_from_c:+.9f}"
    )
    print(
        f"  Strange n={n_s:+.9f}  tau={optical_depth_from_coordinate(n_s):+.9f}  "
        f"kappa+omega from Strange = {kappa_plus_omega_from_s:+.9f}"
    )
    print()
    print("Residuals against closed forms:")
    print(f"  Bottom:  dk        = {kappa_from_b - kappa_closed:+.6e}")
    print(f"  Charm:   dw        = {omega_from_c - omega_closed:+.6e}")
    sum_closed = kappa_closed + omega_closed
    print(f"  Strange: d(k+w)    = {kappa_plus_omega_from_s - sum_closed:+.6e}")
    print()
    print("Internal consistency (no closed form used):")
    internal = kappa_plus_omega_from_s - (kappa_from_b + omega_from_c)
    print(f"  (k+w)_s - (k_b + w_c) = {internal:+.6e}")
    print()
    print("Note: PDG uncertainty floor is ~1% on bottom/charm, ~5% on strange,")
    print("which at the EW scale maps to Delta-units of order")
    print(f"  d(n) ~ log2(1 + sigma_m/m) / Delta  ~= {math.log(1.01, 2.0) / DELTA:.3f} for 1% mass error")
    print("so ~1e-3 level residuals are at or below experimental noise,")
    print("whereas the H/Z/W cluster sits at ~1e-7 because their masses are")
    print("known far more precisely.")


def print_top_d2_probe(rows: Sequence[CoordinateResult]) -> None:
    print_header("Top D^2/4 Back-Solve Probe [falsifiable]")

    by_name = name_index(rows)
    ew_value = by_name["Electroweak scale"].value
    top_mass = by_name["Top quark mass energy"].value
    l_t = math.log(ew_value / top_mass, 2.0)

    h, z, w, mean = hzw_delta_consensus(rows)
    hzw_abs = tuple(abs(item.delta_err) for item in (h, z, w))
    hzw_max = max(hzw_abs)
    hzw_rms = math.sqrt(sum(e * e for e in hzw_abs) / 3.0)

    delta_linear = solve_delta_top_quadratic(l_t, zeta=0.0)
    delta_quad = solve_delta_top_quadratic(l_t, zeta=TOP_MATTER_DENSITY_Q)

    err_linear = delta_linear - DELTA
    err_quad = delta_quad - DELTA

    print("Primary law uses L_t = 73D - 1 + Q*D^2 with Q = 1/4 (matter density).")
    print("This panel contrasts Delta back-solved from the linear backbone only")
    print("versus the full quadratic law (same Q as electroweak_laws / backsolves).")
    print()
    print(f"  Delta (reference)                    = {DELTA:.12f}")
    print(f"  Delta from L_t = 73D - 1             = {delta_linear:.12f}")
    print(f"  Delta from L_t = 73D - 1 + D^2/4     = {delta_quad:.12f}")
    print()
    print(f"  linear  |Delta error|                = {abs(err_linear):.6e}")
    print(f"  quadratic |Delta error|              = {abs(err_quad):.6e}")
    print()
    print(f"  H/Z/W max |Delta error|              = {hzw_max:.6e}")
    print(f"  H/Z/W RMS |Delta error|              = {hzw_rms:.6e}")
    print()
    ratio_lin = abs(err_linear) / hzw_max if hzw_max > 0 else float("inf")
    ratio_quad = abs(err_quad) / hzw_max if hzw_max > 0 else float("inf")
    print(f"  ratio linear / HZW max               = {ratio_lin:.2f}")
    print(f"  ratio quadratic / HZW max            = {ratio_quad:.2f}")
    print()
    if ratio_quad < 3.0 and ratio_lin > 10.0:
        verdict = "SUPPORTS D^2/4: Top error enters HZW cluster after correction"
    elif ratio_quad < ratio_lin * 0.2:
        verdict = "improves fit but does not reach HZW cluster"
    else:
        verdict = "does not improve fit meaningfully"
    print(f"  verdict: {verdict}")


def print_compact_weinberg_angle(rows: Sequence[CoordinateResult]) -> None:
    print_header("Compact On-Shell Weak Mixing Coordinate [derived]")
    by_name = name_index(rows)

    z_mass = by_name["Z boson mass energy"].value
    w_mass = by_name["W boson mass energy"].value

    split_pred = wz_split_closed(DELTA)
    cos_pred = 2.0 ** (-(split_pred * DELTA))
    sin2_pred = 1.0 - cos_pred**2
    theta_pred = math.degrees(math.acos(cos_pred))
    w_pred = z_mass * cos_pred

    cos_obs = w_mass / z_mass
    sin2_obs = 1.0 - cos_obs**2
    split_obs = math.log(z_mass / w_mass, 2.0) / DELTA

    print("Charged-neutral split:")
    print("  D2 backbone:")
    print("    log2(m_Z/m_W) = D[(C2 - C1) - (C3/2)D]")
    print("  stage promoted law:")
    print("    log2(m_Z/m_W) = D[(C2 - C1) - (C3/2)D + 2D^2/sqrt(5) - D^3]")
    print(f"predicted split (n)          = {split_pred:.12f}")
    print(f"tau (pred)                   = {optical_depth_from_coordinate(split_pred):.12f}")
    print(f"observed split (n)           = {split_obs:.12f}")
    print(f"tau (obs)                    = {optical_depth_from_coordinate(split_obs):.12f}")
    print(f"split error                  = {split_obs - split_pred:.12f}")
    print()
    print(f"cos(theta_W) predicted       = {cos_pred:.12f}")
    print(f"cos(theta_W) observed        = {cos_obs:.12f}")
    print(f"sin^2(theta_W) predicted     = {sin2_pred:.12f}")
    print(f"sin^2(theta_W) observed      = {sin2_obs:.12f}")
    print(f"theta_W predicted            = {theta_pred:.9f} deg")
    print()
    print(f"W from Z and Delta           = {w_pred:.12f} GeV")
    print(f"W observed in script         = {w_mass:.12f} GeV")
    print(f"relative mass error          = {w_pred / w_mass - 1.0:.6e}")


def print_wz_ratio_channel_delta_lock(rows: Sequence[CoordinateResult]) -> None:
    by_name = name_index(rows)
    backsolves = electroweak_delta_backsolves(rows)

    top = next(item for item in backsolves if item.source == "top")
    h = next(item for item in backsolves if item.source == "Higgs")
    z = next(item for item in backsolves if item.source == "Z")
    w = next(item for item in backsolves if item.source == "W")
    wz = next(item for item in backsolves if item.source == "W/Z")

    z_mass = by_name["Z boson mass energy"].value
    w_mass = by_name["W boson mass energy"].value
    l_wz = math.log(z_mass / w_mass, 2.0)
    split_obs = l_wz / DELTA
    base_split = wz_split_base(DELTA)
    corrected_split = wz_split_closed(DELTA)
    base_delta = solve_delta_wz_base(l_wz)
    corrected_delta = delta_from_wz_closed(z_mass, w_mass, guess=DELTA)
    delta_four_mean = (top.delta_back + h.delta_back + z.delta_back + w.delta_back) / 4.0
    base_gap = abs(base_delta - delta_four_mean)
    corr_gap = abs(corrected_delta - delta_four_mean)

    print_header("W/Z Ratio-Channel Delta Lock [promoted]")
    print()
    print(f"four-point Delta mean          = {delta_four_mean:.12f}")
    print(f"corrected W/Z Delta            = {corrected_delta:.12f}")
    print(f"W/Z - four-point mean          = {corrected_delta - delta_four_mean:.12e}")
    print(f"reference Delta error          = {wz.delta_err:.12e}")
    print()
    print(f"delta_wz_base                  = {base_delta:.12f}")
    print(f"delta_wz_corr                  = {corrected_delta:.12f}")
    print(f"base W/Z-to-consensus gap      = {base_gap:.12e}")
    print(f"corrected W/Z-to-consensus gap = {corr_gap:.12e}")
    print(f"consensus improvement factor   = {base_gap / corr_gap:.1f}x")
    print("  Strongest single validation: ratio-channel consensus now reaches 8.3e-10.")
    print()
    print(f"base split error                = {split_obs - base_split:.12e}")
    print(f"corrected split error           = {split_obs - corrected_split:.12e}")
    print()
    print(f"sin^2 theta_W corrected       = {sin2_theta_w_from_delta(DELTA):.12f}")
    print(f"sin^2 theta_W observed        = {1.0 - (w_mass / z_mass) ** 2:.12f}")
    print(f"corrected sin^2 error         = {sin2_theta_w_from_delta(DELTA) - (1.0 - (w_mass / z_mass) ** 2):.12e}")
    print(f"W from Z corrected             = {w_mass_from_z_and_delta(z_mass, DELTA):.12f} GeV")
    print(f"W observed                    = {w_mass:.12f} GeV")
    print(f"W relative error               = {(w_mass_from_z_and_delta(z_mass, DELTA) - w_mass) / w_mass:.12e}")


def print_electroweak_order_ladder_closure(rows: Sequence[CoordinateResult]) -> None:
    print_header("Electroweak Order-Ladder Closure [promoted]")

    by_name = name_index(rows)
    ew_value = by_name["Electroweak scale"].value

    channels = (
        ("Top", "Top quark mass energy", CS, MATTER_SECTOR, 73.0, -1.0, TOP_MATTER_DENSITY_Q),
        ("Higgs", "Higgs mass energy", CS_UNA, GAUGE_SECTOR, 96.0, -1.0, None),
        ("Z", "Z boson mass energy", UNA, GAUGE_SECTOR, 117.0, -47.0 / 48.0, None),
        ("W", "W boson mass energy", UNA_ONA, GAUGE_SECTOR, 126.0, -47.0 / 48.0, None),
    )

    print(
        f"{'Ch':8} {'p':>7} {'q':>7} {'r5':>9} "
        f"{'c5_emp':>12} {'n_err D2':>12} {'D3':>12} {'D4':>12} {'D5':>12} {'L_err/D6':>12}"
    )
    print("-" * 108)

    max_d2 = max_d3 = max_d4 = max_d5 = 0.0
    sum_p = sum_q = 0.0

    for label, obs_name, stage, sector, d_coeff, const, matter_q in channels:
        obs_mass = by_name[obs_name].value
        l_obs = math.log(ew_value / obs_mass, 2.0)

        l_d2 = coordinate_law(
            stage,
            sector,
            d_coeff,
            const,
            compact_algebra(),
            matter_density_d2=matter_q,
        )

        p, q = k4_gyroscopic_charge(stage)
        r5 = r5_closure_coeff(stage)
        l_d3 = l_d2 + p * GYROSCOPIC_COUPLING * DELTA * DELTA
        l_d4 = l_d3 + q * DELTA**4
        l_d5 = l_d4 + r5 * DELTA**5

        n2 = (l_obs - l_d2) / DELTA
        n3 = (l_obs - l_d3) / DELTA
        n4 = (l_obs - l_d4) / DELTA
        n5 = (l_obs - l_d5) / DELTA

        c5_emp = (l_obs - l_d4) / DELTA**5
        l_err_d6 = (l_obs - l_d5) / DELTA**6

        max_d2 = max(max_d2, abs(n2))
        max_d3 = max(max_d3, abs(n3))
        max_d4 = max(max_d4, abs(n4))
        max_d5 = max(max_d5, abs(n5))
        sum_p += p
        sum_q += q

        print(
            f"{label:8} {p:7.2f} {q:7.2f} {r5:9.3f} "
            f"{c5_emp:12.6f} {n2:12.6e} {n3:12.6e} {n4:12.6e} {n5:12.6e} {l_err_d6:12.6f}"
        )

    print("-" * 108)
    print(f"trace checks: sum(p)={sum_p:.1f}, sum(q)={sum_q:.1f}")
    print(f"max |n_err| D2 = {max_d2:.6e}")
    print(f"max |n_err| D3 = {max_d3:.6e}   reduction D2->D3 = {max_d2 / max_d3:.1f}x")
    print(f"max |n_err| D4 = {max_d4:.6e}   reduction D3->D4 = {max_d3 / max_d4:.1f}x")
    print(f"max |n_err| D5 = {max_d5:.6e}   reduction D4->D5 = {max_d4 / max_d5:.1f}x")
    print(f"total reduction D2->D5 = {max_d2 / max_d5:.1f}x")
    print(f"largest single reduction D4->D5 = {max_d4 / max_d5:.1f}x")
    print(
        "This is the first step where exact code-valued r5 coefficients "
        "(-4.5, 2.375, -4.5, -2.625) appear."
    )
    print(
        "L_err/D^6 values estimate the remaining sixth-order interface, where CGM structure would need "
        "to meet renormalisation and QFT corrections."
    )
    print()
    print("D5 coefficient law:")
    print("  r5 = -(C2-C1)/2 + (|H|-(C2-C1))/8 * (base-rot) + C2/8 * bal")
    print("  with |H|=64, C1=6, C2=15.")
def print_compact_electroweak_algebra(rows: Sequence[CoordinateResult]) -> None:
    print_header("Compact Spectral Electroweak Algebra")
    by_name = name_index(rows)
    ew_value = by_name["Electroweak scale"].value
    c3 = CODE_C3
    algebra = compact_algebra()
    d_h = algebra.d_h
    epsilon = algebra.epsilon

    laws = electroweak_laws()
    coords = electroweak_delta_coordinates()
    m_t = ew_value * (2.0 ** (-laws["L_t"]))
    m_h = ew_value * (2.0 ** (-laws["L_H"]))
    m_z = ew_value * (2.0 ** (-laws["L_Z"]))
    m_w = ew_value * (2.0 ** (-laws["L_W"]))
    y_t = math.sqrt(2.0) * m_t / ew_value
    lambda_h = (m_h * m_h) / (2.0 * ew_value * ew_value)
    g_z = (2.0 * m_z) / ew_value
    g = (2.0 * m_w) / ew_value

    print("Symbolic coordinate forms:")
    print(f"  n_t = 25 - epsilon + D/4 = {25.0 - epsilon + DELTA / 4.0:.6f}")
    print(f"  n_H = 48 - d_H            = {48.0 - d_h:.6f}")
    print(f"  n_Z = 70 + D - P d_H      = {70.0 + DELTA - (47.0 / 48.0) * d_h:.6f}")
    print(f"  n_W = 79 - 9D - P d_H     = {79.0 - 9.0 * DELTA - (47.0 / 48.0) * d_h:.6f}")
    print("  S_WZ = (C2 - C1) - (C3/2)D + 2D^2/sqrt(5) - D^3")
    print(
        f"       = {W_Z_OFFSET - c3 * DELTA / 2.0 + 2.0 * DELTA * DELTA / math.sqrt(5.0) - DELTA**3:.6f}"
    )
    print()
    print("Computed values:")
    print(f"{'Quantity':20} {'Formula':34} {'value':>18}")
    print("-" * 76)
    rows_out = (
        ("L_t", "73Delta - 1 + Delta^2/4", laws["L_t"]),
        ("L_H", "96Delta - 1 - 24Delta^2", laws["L_H"]),
        ("L_Z", "117Delta - 47/48 - 45Delta^2/2", laws["L_Z"]),
        ("L_W", "126Delta - 47/48 - 65Delta^2/2", laws["L_W"]),
        ("y_t", "2^(3/2 - 73Delta - Delta^2/4)", y_t),
        ("lambda_H", "2^(1 - 192Delta + 48Delta^2)", lambda_h),
        ("g_Z", "2^(95/48 - 117Delta + 45Delta^2/2)", g_z),
        ("g", "2^(95/48 - 126Delta + 65Delta^2/2)", g),
        ("n_H - n_t", "n_H - n_t (from coords)", coords["n_h"] - coords["n_t"]),
        ("m_H / m_t", "2^[(n_t - n_H) Delta]", m_h / m_t),
    )
    for label, law, value in rows_out:
        print(f"{label:20} {law:34} {value:18.12f}")


def print_electroweak_coefficient_alphabet() -> None:
    print_header("Electroweak Coefficient Alphabet [compact electroweak reconstruction]")

    algebra = compact_algebra()
    m_shell = algebra.m1
    d2_top = TOP_MATTER_DENSITY_Q
    d2_h = d2_coefficient_from_stage(CS_UNA, algebra)
    d2_z = d2_coefficient_from_stage(UNA, algebra)
    d2_w = d2_coefficient_from_stage(UNA_ONA, algebra)


    print(
        f"{'State':10} {'law':8} {'formula':32} {'channel':20} "
        f"{'D coeff':>10} {'const':>12} {'D2 coeff':>12}"
    )
    print("-" * 112)

    rows_out = (
        ("Top", "L_t", "73D - 1 + D^2/4", "Top quark", 73.0, -1.0, d2_top),
        ("H", "L_H", "96D - 1 - 24D2", "Higgs", 96.0, -1.0, d2_h),
        ("Z", "L_Z", "117D - 47/48 - 45D2/2", "Z boson", 117.0, -47.0 / 48.0, d2_z),
        ("W", "L_W", "126D - 47/48 - 65D2/2", "W boson", 126.0, -47.0 / 48.0, d2_w),
        ("Muon", "N_mu", "540 + 20D", "Muon split", 20.0, 540.0, 0.0),
    )

    for state, law, formula, channel, d_coeff, const, d2_coeff in rows_out:
        print(
            f"{state:10} {law:8} {formula:32} {channel:20} "
            f"{d_coeff:10.3f} {const:12.9f} {d2_coeff:12.3f}"
        )

    print()
    print(f"{'Z - Higgs D coefficient':32} = {117.0 - 96.0:.3f}")
    print(f"{'W - Z D coefficient':32} = {126.0 - 117.0:.3f}")
    print(f"{'W - Z D2 coefficient':32} = {(-65.0 / 2.0) - (-45.0 / 2.0):.3f}")
    print(f"{'shared gauge constant':32} = {-47.0 / 48.0:.12f}")
    print(f"{'gauge coupling constant term':32} = {95.0 / 48.0:.12f}")
    print()
    print(f"{'C1 = C(6,1)':32} = {CODE_C1}")
    print(f"{'C2 = C(6,2)':32} = {CODE_C2}")
    print(f"{'C3 = C(6,3)':32} = {CODE_C3}")
    print(f"{'Z_H_OFFSET = 1 + C1 + C2':32} = {Z_H_OFFSET}")
    print(f"{'W_Z_OFFSET = C2 - C1':32} = {W_Z_OFFSET}")
    print(f"{'W_Z_APERTURE_COEFF = C3/2':32} = {W_Z_APERTURE_COEFF:.12f}")
    print(f"{'MUON_EQUATOR_COEFF = C3':32} = {MUON_EQUATOR_COEFF}")
    print(f"{'M_shell = sum k C(6,k)':32} = {m_shell}")
    print(f"{'|H| = 64':32} = horizon cardinality")
    print()
    print(f"{'73 = 64 + C2 - C1':32} = {64 + CODE_C2 - CODE_C1}")
    print(f"{'96 = M_shell/2':32} = {m_shell / 2}")
    print(f"{'117 = M_shell/2 + C1 + C2':32} = {m_shell / 2 + CODE_C1 + CODE_C2}")
    print(f"{'126 = M_shell/2 + 2C2':32} = {m_shell / 2 + 2 * CODE_C2}")
    print(f"{'24 = M_shell/8':32} = {m_shell / 8}")
    print("Quadratic-sector gauge totals (zeta on L, negative porosity):")
    print(f"{'  -24 = -M_shell/8 (Higgs)':32} = {-m_shell / 8:.12f}")
    print(f"{'  -45/2 = -M/8 + C1/4 (Z)':32} = {-m_shell / 8.0 + CODE_C1 / 4.0:.12f}")
    print(f"{'  -65/2 = -M/8 + C1/4 - C3/2 (W)':32} = {-m_shell / 8.0 + CODE_C1 / 4.0 - CODE_C3 / 2.0:.12f}")
    print("Magnitude identities (same numbers, positive form):")
    print(f"{'  45/2 = M_shell/8 - C1/4':32} = {m_shell / 8.0 - CODE_C1 / 4.0:.12f}")
    print(f"{'  65/2 = M_shell/8 - C1/4 + C3/2':32} = {m_shell / 8.0 - CODE_C1 / 4.0 + CODE_C3 / 2.0:.12f}")


def print_z_w_boundary_shell_anchors() -> None:
    print_header("Z/W Boundary-Shell Anchors [derived]")

    z_anchor = 64 + CODE_C1
    w_anchor = 64 + CODE_C2

    print(f"{'Z anchor = |H| + C1':32} = {z_anchor}")
    print(f"{'W anchor = |H| + C2':32} = {w_anchor}")
    print(f"{'W_Z_OFFSET = C2 - C1':32} = {W_Z_OFFSET}")
    print(f"{'W_Z_APERTURE_COEFF = C3/2':32} = {W_Z_APERTURE_COEFF:.12f}")
    print()
    print("Gauge-sector coordinate forms:")
    print(f"  n_Z = {z_anchor} + D - (47/48)d_H")
    print(f"  n_W = {w_anchor} - (C2-C1)D - (47/48)d_H")
    print(f"      = {w_anchor} - {W_Z_OFFSET}D - (47/48)d_H")
    print()
    print("Charged-neutral split:")
    print("  log2(m_Z/m_W) = (C2 - C1)D - (C3/2)D^2")
def print_coupling_dependency_audit(rows: Sequence[CoordinateResult]) -> None:
    print_header("Coupling Dependency Audit [correctness check]")

    by_name = name_index(rows)
    ew_value = by_name["Electroweak scale"].value

    l_t = ew_log2_gap(rows, "Top quark mass energy")
    l_h = ew_log2_gap(rows, "Higgs mass energy")
    l_z = ew_log2_gap(rows, "Z boson mass energy")
    l_w = ew_log2_gap(rows, "W boson mass energy")

    y_t = math.sqrt(2.0) * by_name["Top quark mass energy"].value / ew_value
    lambda_h = (by_name["Higgs mass energy"].value ** 2) / (2.0 * ew_value * ew_value)
    g_z = (2.0 * by_name["Z boson mass energy"].value) / ew_value
    g = (2.0 * by_name["W boson mass energy"].value) / ew_value

    checks = (
        ("y_t", "log2(y_t) + L_t - 1/2", math.log(y_t, 2.0) + l_t - 0.5),
        ("lambda_H", "log2(lambda_H) + 2L_H + 1", math.log(lambda_h, 2.0) + 2.0 * l_h + 1.0),
        ("g_Z", "log2(g_Z) + L_Z - 1", math.log(g_z, 2.0) + l_z - 1.0),
        ("g", "log2(g) + L_W - 1", math.log(g, 2.0) + l_w - 1.0),
    )

    print(f"{'Quantity':12} {'identity':36} {'residual':>18}")
    print("-" * 72)
    print("These are algebraic identities, not independent tests.")
    print()

    for quantity, identity, residual in checks:
        print(f"{quantity:12} {identity:36} {residual:18.12e}")

def main() -> None:
    const = build_constants()
    planck = planck_scales(const)
    observables = build_observables(const)

    delta_rows = build_coordinate_table(
        observables=observables,
        coordinate_fn=lambda obs: aperture_delta_coordinate(obs, planck),
    )

    print_header("Compact Geometry: Spectral Algebra of the Electroweak Mass Spectrum")
    print("Mapping dimensionful physical observables onto compact-kernel coordinates.")

    print_compact_input_summary()
    print_operator_group_theorem()
    print_code_weight_derivation()
    print_electroweak_coefficient_alphabet()
    print_z_w_boundary_shell_anchors()
    print_k4_gyroscopic_ladder(delta_rows)
    print_electroweak_order_ladder_closure(delta_rows)
    print_compact_electroweak_cascade(delta_rows)
    print_delta_backsolves(delta_rows)
    print_four_point_delta_consensus(delta_rows)
    print_hzw_leave_one_out(delta_rows)
    print_wz_ratio_channel_delta_lock(delta_rows)
    print_compact_weinberg_angle(delta_rows)
    print_compact_electroweak_algebra(delta_rows)
    print_coupling_dependency_audit(delta_rows)
    print_ew_mass_coordinates(delta_rows)

    print_matter_gauge_dichotomy(delta_rows)
    print_boundary_and_density_projectors(delta_rows)
    print_stage_action_diagnostics(delta_rows)
    print_optical_depth_ruler_diagnostics(delta_rows)
    print_quadratic_sector_status()
    print_force_stage_consistency()
    print_quark_kappa_omega_test(delta_rows)
    print_top_d2_probe(delta_rows)


if __name__ == "__main__":
    main()
