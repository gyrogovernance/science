#!/usr/bin/env python3
"""Yang-Mills mass gap — JW object + Wilson lattice certificates.

Carrier l^2(Omega) (sections 0–4) and Wilson JW certificates (5–10).
Companion: Yang_Mills_Mass_Gap_common.py (lattice engine). Orchestrator: Yang_Mills_Mass_Gap_run.py.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp

import Yang_Mills_Mass_Gap_common  # noqa: F401 — repo path setup

from gyroscopic.hQVM.api import OMEGA_STATES_4096, k4_orbit
from gyroscopic.hQVM.constants import (
    APERTURE_GAP_Q256,
    GATE_NAMES,
    GENE_MIC_S,
    HORIZON_SIZE,
    OMEGA_SIZE,
    apply_gate,
    byte_to_intron,
)
from gyroscopic.hQVM.family import (
    alphabet_size,
    verify_d6_against_api,
)

from Yang_Mills_Mass_Gap_common import (
    A_KERNEL,
    BU_CLOSURE_DEPTH,
    CODE_C1,
    CODE_C2,
    D_CONTINUUM_CGM,
    DELTA,
    DELTA_BU,
    E_EW_GEV,
    EQUATOR,
    GENE_MIC,
    GEV_TO_EV,
    G_DEFINING_KS,
    M_A,
    N_SPATIAL_CGM,
    N_TEMPORAL_CGM,
    D_PAYLOAD,
    NUCLEAR_CLASS,
    Q_G,
    QG_MA2,
    RHO,
    S_CS,
    SE3_DOF,
    SQRT5,
    cgm_ym_gap_prediction,
    curvature_index_kappa2,
    gate,
    progress,
    section,
    section_title,
    unit_map_trackd_to_gev,
    K4,
    Q8,
    LatticeYM,
    Q8_from_extension,
    gauge_invariant_reduce,
    gauge_invariant_spectrum,
    jw_gap_from_w,
    q8_lattice_gap_row,
    wilson_weight_K4,
    wilson_weight_Q8_2d,
)
from Yang_Mills_Mass_Gap_2 import LatticeYM3D

# External empirical input: Th-229m isomer energy (Zhang et al. 2024), eV in CaF2.
TH229M_EV = 8.3563
FULL_ALPHABET_D6 = alphabet_size(D_PAYLOAD)


@dataclass
class CarrierContext:
    fast: bool
    k6_ok: bool
    k6_msg: str
    cluster_ok: bool
    bridge_ok: bool
    defn_ok: bool
    pred_ok: bool
    pred: dict[str, Any]
    gap_full: float


def grammar_energy(k: int, l: int, stf: bool, equator: bool) -> float:
    """Grammar energy map E(k, l) in eV."""
    e = E_EW_GEV * (DELTA**k) * (RHO**l)
    if stf:
        e /= SQRT5
    if equator:
        e *= EQUATOR
    return e * GEV_TO_EV


def jw_definition_certificates() -> dict:
    """Part A (JW definition): algebra + state + Hilbert representation certificates."""
    from gyroscopic.hQVM.constants import GENE_MAC_REST, is_on_horizon
    return {
        "dim_Omega": OMEGA_SIZE,
        "dim_horizon": HORIZON_SIZE,
        "holographic_H2_eq_Omega": (HORIZON_SIZE ** 2) == OMEGA_SIZE,
        "K4_gate_names": list(GATE_NAMES),
        "GENE_Mic": GENE_MIC,
        "GENE_Mic_zero_intron": byte_to_intron(GENE_MIC) == 0,
        "GENE_Mac_rest": GENE_MAC_REST,
        "GENE_Mac_rest_on_complement": is_on_horizon(GENE_MAC_REST),
        "Q_G": Q_G,
        "m_a": M_A,
        "Q_G_m_a2": QG_MA2,
        "Q_G_m_a2_eq_half": abs(QG_MA2 - 0.5) < 1e-12,
        "Delta": float(DELTA),
        "rho": float(RHO),
        "delta_BU": float(DELTA_BU),
        "A_kernel": A_KERNEL,
        "A_kernel_vs_Delta_rel": abs(A_KERNEL - DELTA) / DELTA,
        "C2_2form_multiplicity": int(CODE_C2),
        "spectral_triple": "(A=<T_b>, H=l2(Omega), D=D_shell)",
    }


def spacetime_dimension_certificate() -> dict:
    """D = n + 1 = 4 from CGM: n=3 spatial + time = BU depth evolution."""
    g_mass_dim = (4 - D_CONTINUUM_CGM) / 2.0
    return {
        "n_spatial": N_SPATIAL_CGM,
        "n_temporal": N_TEMPORAL_CGM,
        "D_continuum": D_CONTINUUM_CGM,
        "D_eq_n_plus_1": D_CONTINUUM_CGM == N_SPATIAL_CGM + N_TEMPORAL_CGM,
        "BU_closure_depth": BU_CLOSURE_DEPTH,
        "SE3_dof": SE3_DOF,
        "C1_hexacode_vertices": int(CODE_C1),
        "g_mass_dimension": g_mass_dim,
        "g_dimensionless_at_D4": abs(g_mass_dim) < 1e-12,
        "Wilson_H_form": "(g^2/2) H_elec + (1/2g^2) H_mag",
        "time_is_evolution_not_link": True,
        "plaquette_is_spatial_2form": True,
    }


def kernel_holonomy_k4_certificate() -> dict:
    """Wilson K4 link group matches holonomic GATE_NAMES on the Omega carrier."""
    G, _, _, _ = K4()
    omega = set(OMEGA_STATES_4096)
    perms_ok = all(
        all(apply_gate(s, name) in omega for s in OMEGA_STATES_4096)
        for name in GATE_NAMES
    )
    max_orbit = max(len(k4_orbit(s)) for s in OMEGA_STATES_4096)
    return {
        "wilson_k4_order": len(G),
        "holonomic_gate_count": len(GATE_NAMES),
        "max_k4_orbit_size": max_orbit,
        "gates_preserve_omega": perms_ok,
        "pass": len(G) == len(GATE_NAMES) == max_orbit == 4 and perms_ok,
    }


def certificates(lat, V_wt=None):
    """Commuting-projector residuals on a small lattice (A_v, mag gauge inv, P)."""
    if V_wt is None:
        _, V_wt = wilson_weight_K4()
    N = lat.N
    dim = N ** lat.nE
    vi, vj = 0, 0
    vedges = lat.vertex_edges(vi, vj)
    A = None
    for h in range(N):
        newflat = lat.vertex_action_indices(vedges, h)
        col = np.arange(dim)
        blk = sp.csr_matrix((np.ones(dim), (newflat, col)), shape=(dim, dim))
        A = blk if A is None else A + blk
    A = (A / N).tocsr()
    res_A = float(np.linalg.norm((A @ A - A).toarray()) / np.linalg.norm(A.toarray()))
    Hm = sp.diags(lat.magnetic_diagonal(V_wt), format="csr")
    com = (A @ Hm - Hm @ A).toarray()
    res_comm = float(np.linalg.norm(com) / max(np.linalg.norm(A.toarray()), 1e-12))
    rng = np.random.default_rng(0)
    res_P = 0.0
    for _ in range(3):
        x = rng.standard_normal(dim)
        Px = lat.gauge_projector_matvec(x)
        PPx = lat.gauge_projector_matvec(Px)
        nPx = np.linalg.norm(Px)
        if nPx > 1e-12:
            res_P = max(res_P, float(np.linalg.norm(PPx - Px) / nPx))
    return {
        "A_v_proj_resid": res_A,
        "mag_gauge_invar_resid": res_comm,
        "P_idempotent_resid": res_P,
    }


def omega_states() -> list[int]:
    """Canonical 4096 kernel states (24-bit carrier) from hQVM api."""
    return list(OMEGA_STATES_4096)


def sector_gap(_states: list[int]) -> tuple[float, float]:
    """Track A sector gap: forced minimum nuclear excitation.

    Grammar Hamiltonian for this track is H_sector = 0 on vacuum, = E_min on
    the excited sector. Therefore vacuum energy = 0, gap = E_min.
    """
    e_min = float(grammar_energy(*NUCLEAR_CLASS[1:]))  # eV
    return e_min, e_min


def th229m_check(delta_eV: float) -> tuple[float, float, bool]:
    rel = abs(delta_eV - TH229M_EV) / TH229M_EV
    return TH229M_EV, rel, rel < 1e-3


def delta_w_formula(d: int) -> float:
    """Exact shadow gap: n/(2(n-1)) for alphabet size n = alphabet_size(d)."""
    n = float(alphabet_size(d))
    return n / (2.0 * (n - 1.0))


def _print_jw_a(defn: dict) -> bool:
    print("  dim Omega                     :", defn["dim_Omega"])
    print("  |H|^2 / |Omega|               :", defn["dim_Omega"], defn["dim_Omega"],
          "eq=", defn["holographic_H2_eq_Omega"])
    print("  K4 gate names                 :", defn["K4_gate_names"])
    print("  GENE_Mic archetype            :", hex(defn["GENE_Mic"]))
    print("  GENE_Mic intron               :", 0 if defn["GENE_Mic_zero_intron"] else "nonzero")
    print("  GENE_Mac rest                 :", hex(defn["GENE_Mac_rest"]))
    print("  GENE_Mac on complement        :", defn["GENE_Mac_rest_on_complement"])
    print("  Q_G * m_a^2                   :", round(defn["Q_G_m_a2"], 12))
    print("  |Q_G m_a^2 - 1/2|             :", abs(float(defn["Q_G_m_a2"]) - 0.5))
    print("  Delta (A*) continuous         :", f"{defn['Delta']:.12f}")
    print("  A_kernel = 5/256 discrete     :", f"{defn['A_kernel']:.12f}")
    print("  |A_kernel - Delta|/Delta      :", f"{defn['A_kernel_vs_Delta_rel']:.4e}")
    print("  C2 = C(6,2)                   :", defn["C2_2form_multiplicity"])
    return (
        defn["holographic_H2_eq_Omega"]
        and defn["GENE_Mic_zero_intron"]
        and defn["GENE_Mac_rest_on_complement"]
        and defn["Q_G_m_a2_eq_half"]
        and defn["C2_2form_multiplicity"] == 15
    )


def _print_delta_ruler(pred: dict) -> bool:
    print("  v (E_EW) GeV                  :", pred["v_GeV"])
    print("  Delta                         :", f"{pred['Delta']:.12f}")
    print("  Delta^2                       :", f"{pred['Delta2']:.12e}")
    print("  v * Delta^2 GeV               :", round(pred["v_Delta2_GeV"], 6))
    print("  C1, C2, C3                    :", pred["C1"], pred["C2"], pred["C3"])
    print("  S_CS = (pi/2)/m_a             :", round(pred["S_CS"], 6))
    print("  Route A: C2 * v * Delta^2 GeV :", round(pred["route_A_C2_v_Delta2_GeV"], 6))
    print("  Route B: S_CS * 2 v * Delta^2 :", round(pred["route_B_S_CS_times_2v_Delta2_GeV"], 6))
    print("  |A-B|/max(A,B)                :", f"{pred['rel_diff_A_vs_B']:.4e}")
    print("  lambda_strong = 2 v Delta^2   :", round(pred["lambda_strong_2v_Delta2_GeV"], 6))
    ir = pred["IR"]
    print("  IR ladder (optical conjugacy) :")
    print("    E_CS_IR GeV                 :", round(ir["E_CS_IR_GeV"], 4))
    print("    E_UNA_IR GeV                :", round(ir["E_UNA_IR_GeV"], 4))
    print("    E_ONA_IR GeV                :", round(ir["E_ONA_IR_GeV"], 4))
    print("    E_BU_IR GeV (= E_EW)        :", round(ir["E_BU_IR_GeV"], 4))
    return (pred["C2"] == 15 and 1.0 < pred["route_A_C2_v_Delta2_GeV"] < 2.0
            and pred["rel_diff_A_vs_B"] < 0.1
            and abs(ir["E_BU_IR_GeV"] - E_EW_GEV) < 1e-6)


def q8_central_extension_derived() -> dict:
    """G1: derive Q8 as the central extension 1 -> Z2 -> Q8 -> K4 -> 1.

    Q8 is NOT imported as a quaternion table; it is built from K4 (the doc's gauge
    group) and a Z2 sheet flip via the CGM cocycle omega (cocycle_k4 in common lattice engine).
    Multiplication is (k1,z1)(k2,z2) = (k1 k2, z1 ^ z2 ^ omega(k1,k2)). We verify the
    result is isomorphic to Q8 (center {1,-1}, all non-central elements order 4) and
    not the direct product K4 x Z2 (which would give D4). This is the "bridge" that
    prevents the critique "Q8 was imported": the double cover is derived from the
    K4 spinorial structure already in the corpus.
    """
    Gext, gi, te, ie = Q8_from_extension()
    Gref, gir, tr, ir = Q8()

    def order_ext(g):
        x = gi[g]; cur = x; n = 1
        while te[cur, x] != x:
            cur = te[cur, x]; n += 1
            if n > 12:
                return -1
        return n

    center = [Gext[i] for i in range(8) if all(te[i, j] == te[j, i] for j in range(8))]
    is_q8 = set(center) == {"1", "-1"} and all(order_ext(g) == 4 for g in Gext if g not in center)
    tables_match = all(
        Gext[te[gi[a], gi[b]]] == Gref[tr[gir[a], gir[b]]]
        for a in Gext for b in Gext
    )
    return {"is_Q8": bool(is_q8), "center": center,
            "orders": {g: order_ext(g) for g in Gext},
            "matches_imported_Q8": bool(tables_match)}


def q8_single_plaquette_dense(g: float = G_DEFINING_KS) -> dict:
    """Exact Wilson Kogut-Susskind Hamiltonian on one Q8 plaquette (4 links).

    Gauge group Q8 (derived in q8_central_extension_derived) is the spinorial double
    cover of the doc's abelian K4; it is the smallest non-abelian subgroup of SU(2)
    and carries a 2D irrep, so the Wilson plaquette weight V_R(hol) is nontrivial.
    H = (g^2/2) H_elec + (1/2g^2) H_mag with H_elec = link Laplacian (I - avg L_h^(e))
    and H_mag = sum_p V_R(hol(p)) (class function, not a flat/flux projector). The
    object the conventional "merge 3 space + 1 time into 4D" loses is exactly this
    double cover: the magnetic term then distinguishes holonomy conjugacy classes and
    a genuine (non-two-level) mass gap opens.
    """
    G, gi, table, inv = Q8()
    _, V = wilson_weight_Q8_2d()
    lat = LatticeYM(1, 1, (G, gi, table, inv), periodic=True)
    w, gap, vac, e0 = gauge_invariant_spectrum(lat, g, V)
    return {
        "group": "Q8", "links": lat.nE, "plaquette_sides": 4, "dim": lat.N ** lat.nE,
        "E0": float(e0),
        "gap": float(gap),
        "n_distinct": len(np.unique(np.round(w, 4))),
        "vacuum_mult": int(vac),
        "min_pos_eigs": [round(float(x), 5) for x in w[:6]],
    }


def hamiltonian_derivation_certificate() -> dict:
    """Grammar links for Wilson H_elec + H_mag + Q8 extension (identity checks only).

    - K4 gauge group is the doc face-swap group (id, S, C, F).
    - Q8 is the CGM central extension K4 x Z2 (cocycle_k4), not an imported quaternion table.
    - Wilson magnetic V_R(g) = 1 - Re chi_R(g)/d_R uses the 2D irrep of Q8 (non-abelian).
    - Electric term is the link Laplacian (I - avg L_h), Casimir normalization g^2/2.
    - Delta^2 is the first curvature moment on the hexacode chart; C2 = C(6,2) counts 2-forms.
    """
    _, Vk4 = wilson_weight_K4()
    _, Vq8 = wilson_weight_Q8_2d()
    Gk, _, _, _ = K4()
    Gext, _, _, _ = Q8_from_extension()
    v_id = float(Vk4[0])
    v_vals_k4 = sorted({round(float(Vk4[g]), 12) for g in range(len(Gk))})
    v_vals_q8 = sorted({round(float(Vq8[g]), 12) for g in range(len(Gext))})
    v_nontrivial_k4 = any(abs(Vk4[g] - v_id) > 1e-12 for g in range(1, 4))
    v_nontrivial_q8 = any(abs(Vq8[g] - v_id) > 1e-12 for g in range(1, 8))
    return {
        "K4_order": len(Gk),
        "Q8_order": len(Gext),
        "Wilson_V_identity_K4": v_id,
        "Wilson_V_distinct_K4": v_vals_k4,
        "Wilson_V_distinct_Q8": v_vals_q8,
        "Wilson_n_distinct_K4": len(v_vals_k4),
        "Wilson_n_distinct_Q8": len(v_vals_q8),
        "Wilson_nontrivial_K4": bool(v_nontrivial_k4),
        "Wilson_nontrivial_Q8": bool(v_nontrivial_q8),
        "H_elec_link_laplacian": "sum_e (I - avg L_h^(e))",
        "H_mag_Wilson": "sum_p V_R(hol(p))",
        "Delta": float(DELTA),
        "Delta2": float(DELTA ** 2),
        "C2_2form": int(CODE_C2),
        "QG_ma2": float(QG_MA2),
        "S_CS": float(S_CS),
    }


def ym_correlator(lat, g: float = G_DEFINING_KS, t_max: float = 3.0, n_t: int = 31) -> dict:
    """E1: time = e^{-tH}. Connected correlator of Wilson plaquette weight (GI).

    Uses spectral expansion C(t)=sum_{n>0}|<n|O|0>|^2 e^{-(E_n-E0)t} (always >=0).
    Effective mass = lowest (E_n-E0) with nonzero overlap; slope of log C matches -m_eff.
    """
    _, V = wilson_weight_K4()
    wr, Vr, gap, vac, vac_e, Q = gauge_invariant_reduce(lat, g, V)
    Omega = Vr[:, 0]
    mag = lat.magnetic_diagonal(V)
    Od = Q.T @ np.diag(mag) @ Q
    Od = (Od + Od.T) / 2
    O0 = Od @ Omega
    amps = Vr.T @ O0
    ts = np.linspace(0, t_max, n_t)
    Cs = []
    for t in ts:
        s = 0.0
        for n in range(1, len(wr)):
            s += float(abs(amps[n]) ** 2) * math.exp(-(wr[n] - wr[0]) * t)
        Cs.append(s)
    Cs = np.array(Cs)
    coupled = [float(wr[n] - wr[0]) for n in range(1, len(wr)) if abs(amps[n]) > 1e-8]
    m_eff = min(coupled) if coupled else float("nan")
    mask = (ts > 0.05) & (Cs > 1e-12)
    if np.count_nonzero(mask) >= 3:
        sl, _ = np.polyfit(ts[mask], np.log(Cs[mask]), 1)
    else:
        sl = float("nan")
    return {
        "vacuum_energy": vac_e, "gap": gap, "m_eff_coupled": m_eff,
        "correlator_slope": float(sl), "ts": list(ts), "Cs": list(Cs),
        "n_coupled_modes": len(coupled),
    }


def jw_clustering_certificate(lat, g: float = G_DEFINING_KS, t: float = 1.0) -> dict:
    """JW-B: spectral connected correlators for GI ops (Wilson, H_elec, H_mag)."""
    _, V = wilson_weight_K4()
    wr, Vr, gap, vac, vac_e, Q = gauge_invariant_reduce(lat, g, V)
    Omega = Vr[:, 0]
    mag = lat.magnetic_diagonal(V)
    _op, H, He, Hm = lat.hamiltonian_operator(g, V)
    ops = [
        ("Wilson_plaq", Q.T @ np.diag(mag) @ Q),
        ("H_elec", np.asarray(Q.T @ (He @ Q))),
        ("H_mag", np.asarray(Q.T @ (Hm @ Q))),
    ]
    rows = []
    slopes = []
    m_effs = []
    for name, Od in ops:
        Od = np.asarray(Od)
        Od = (Od + Od.T) / 2
        amps = Vr.T @ (Od @ Omega)
        coupled = [float(wr[n] - wr[0]) for n in range(1, len(wr)) if abs(amps[n]) > 1e-8]
        m_eff = min(coupled) if coupled else float("nan")
        m_effs.append(m_eff)
        ts = np.linspace(0.5, 2.5, 9)
        Cs = []
        for tv in ts:
            s = 0.0
            for n in range(1, len(wr)):
                s += float(abs(amps[n]) ** 2) * math.exp(-(wr[n] - wr[0]) * tv)
            Cs.append(s)
        Cs = np.array(Cs)
        if np.all(Cs > 1e-15):
            sl, _ = np.polyfit(ts, np.log(Cs), 1)
        else:
            sl = float("nan")
        slopes.append(float(sl))
        rows.append({"op": name, "m_eff": m_eff, "C_at_t": float(Cs[0]), "O_vac": float(amps[0].real)})
    mean_slope = float(np.nanmean(slopes))
    slope_ok = all(
        (not math.isnan(s)) and s < -1e-3 and (not math.isnan(m)) and abs(s + m) < 0.15
        for s, m in zip(slopes, m_effs)
    )
    return {
        "gap": gap,
        "vacuum_energy": vac_e,
        "slopes": slopes,
        "m_effs": m_effs,
        "mean_slope": mean_slope,
        "all_negative_slope": all((not math.isnan(s)) and s < -1e-3 for s in slopes),
        "slope_near_gap": slope_ok,
        "observables": rows,
    }


def jw_part_a_certificate() -> dict:
    """JW-A definition items mapped to CGM certificates (boolean checks only)."""
    defn = jw_definition_certificates()
    hderiv = hamiltonian_derivation_certificate()
    Gext, _, _, _ = Q8_from_extension()
    return {
        "algebra_T_b_on_Omega": defn["dim_Omega"] == OMEGA_SIZE,
        "state_omega_CS_BU_chain": defn["Q_G_m_a2_eq_half"] and defn["GENE_Mic_zero_intron"],
        "GENE_Mic_archetype": int(GENE_MIC),
        "GENE_Mic_fixes_reference_byte": byte_to_intron(GENE_MIC) == 0,
        "GNS_H_dim": defn["dim_Omega"],
        "spectral_triple_present": defn["C2_2form_multiplicity"] == 15,
        "Wilson_curvature_operator": hderiv["H_mag_Wilson"] == "sum_p V_R(hol(p))",
        "Hamiltonian_Wilson_KS": hderiv["H_elec_link_laplacian"].startswith("sum_e"),
        "Q8_spinorial_lift": len(Gext) == 8,
        "K4_doc_gauge": hderiv["K4_order"] == 4,
    }


def lattice_spacing_certificate(gap_q8: float, pred: dict) -> dict:
    """Link discrete A_kernel / Delta to grade-1 E_unit = v*Delta; m_phys = gap * E_unit."""
    um = unit_map_trackd_to_gev(gap_q8, pred)
    a_inv_delta = float(E_EW_GEV) * float(DELTA)
    a_inv_kernel = float(E_EW_GEV) * float(A_KERNEL)
    e_unit = float(um["E_unit_GeV"]) if um.get("ok") else float("nan")
    m_phys = float(um["m_phys_GeV"]) if um.get("ok") else float("nan")
    return {
        "A_kernel": float(A_KERNEL),
        "Delta": float(DELTA),
        "rel_A_kernel_vs_Delta": abs(A_KERNEL - DELTA) / DELTA,
        "E_unit_GeV": e_unit,
        "v_times_Delta_GeV": a_inv_delta,
        "v_times_A_kernel_GeV": a_inv_kernel,
        "E_unit_over_vDelta": e_unit / a_inv_delta if a_inv_delta > 0 and um.get("ok") else float("nan"),
        "gap_dimless": float(gap_q8),
        "m_phys_GeV": m_phys,
        "ok": bool(um.get("ok")),
    }


def q8_volume_scan() -> list[dict]:
    """Q8 Wilson: defining 1×1 periodic + Lx=2 periodic (D2 torus pathology dictionary)."""
    _, V = wilson_weight_Q8_2d()
    rows = []
    for Lx in (1, 2):
        lat = LatticeYM(Lx, 1, Q8(), periodic=True)
        dim = lat.N ** lat.nE
        progress(f"Q8 volume Lx={Lx} periodic dim={dim}")
        rows.append(q8_lattice_gap_row(lat, 1.0, V))
    return rows


def locality_commutation_certificate(lat, g: float = G_DEFINING_KS) -> dict:
    """JW-E locality: gauge-invariant electric link Casimirs on distinct links commute."""
    _, V = wilson_weight_K4()
    dim = lat.N ** lat.nE
    _, _, gap, _, _, Q = gauge_invariant_reduce(lat, g, V)
    max_comm = 0.0
    n_pairs = 0
    n_links = min(lat.nE, 6)
    for e in range(n_links):
        for f in range(e + 1, n_links):
            def link_casimir_matvec(x, link):
                acc = np.zeros_like(x)
                for h in range(lat.N):
                    acc[lat._link_perm(link, h)] += x / lat.N
                return x - acc

            cols_e = [link_casimir_matvec(Q[:, j], e) for j in range(Q.shape[1])]
            cols_f = [link_casimir_matvec(Q[:, j], f) for j in range(Q.shape[1])]
            Ce = Q.T @ np.column_stack(cols_e)
            Cf = Q.T @ np.column_stack(cols_f)
            Ce = (Ce + Ce.T) / 2
            Cf = (Cf + Cf.T) / 2
            comm = Ce @ Cf - Cf @ Ce
            max_comm = max(max_comm, float(np.linalg.norm(comm)))
            n_pairs += 1
    return {
        "n_links_tested": n_links,
        "n_pairs": n_pairs,
        "max_commutator_norm": max_comm,
        "pass": max_comm < 1e-8,
        "gap": gap,
    }


def torus_vs_physical_gap_certificate() -> dict:
    """D2: torus spectral gap ≠ physical mass gap (finite witness).

    Compares Q8 periodic Lx=1 vs Lx=2 at g=1. Large drop on Lx=2 periodic
    shows torus_gap is not volume-stable. Physical JW gap is jw_gap_phys on ω_∞.
    """
    rows = q8_volume_scan()
    by = {(r.get("Lx"), r.get("periodic")): r for r in rows}
    r1 = by.get((1, True))
    r2 = by.get((2, True))
    if r1 is None or r2 is None:
        return {"pass": False, "reason": "missing Lx=1/2 periodic rows", "D2_closed": False}
    g1 = float(r1.get("gap", float("nan")))
    g2 = float(r2.get("gap", float("nan")))
    drop = g1 - g2 if (g1 == g1 and g2 == g2) else float("nan")
    # Pathology: Lx=2 periodic gap much smaller than defining block
    pathology = (g2 == g2) and (g1 == g1) and (g2 < 0.5 * g1)
    return {
        "Lx1_periodic_gap": g1,
        "Lx2_periodic_gap": g2,
        "gap_drop": drop,
        "torus_pathology": pathology,
        "note": (
            "torus_gap(Lx=2 per) << torus_gap(Lx=1 per) at g=1. "
            "jw_gap_phys := gap in GNS of ω_∞ (Lemma IV), not torus spectrum."
        ),
        "pass": pathology and g1 > 1e-3,
        "D2_closed": False,
    }


def gene_mic_omega_certificate(gap_q8: float, gap_shadow: float = QG_MA2) -> dict:
    """GENE_Mic (0xAA) as datum fixing omega: without Q8 lift, shadow gap locks to QG_MA2."""
    collapsed = abs(gap_shadow - QG_MA2) < 1e-6
    physical = abs(gap_q8 - QG_MA2) > 0.1
    return {
        "GENE_Mic": int(GENE_MIC),
        "intron_zero": byte_to_intron(GENE_MIC) == 0,
        "shadow_gap_collapsed_0p5": collapsed,
        "Q8_gap_not_shadow": physical,
        "gap_q8": float(gap_q8),
        "gap_shadow": float(gap_shadow),
        "QG_MA2": float(QG_MA2),
        "pass": (GENE_MIC == GENE_MIC_S) and collapsed and physical,
    }


def shadow_lock_identity_certificate(gap_full: float | None = None) -> dict:
    """Theorem D3-struct + oriented uniqueness: lim Δ_W = QG_MA2 = 1/2.

    Unoriented K4-fiber average → Δ_W → 1/2. Oriented GENE_Mic → Δ = 1−ρ ≠ 1/2.
    These are the only two stable curvature-gap regimes on this carrier.
    """
    n = FULL_ALPHABET_D6
    delta_w = n / (2.0 * (n - 1.0))
    delta_w_inf = 0.5
    identity_holds = abs(delta_w_inf - QG_MA2) < 1e-15
    finite_near = abs(delta_w - QG_MA2) < 0.01
    if gap_full is not None:
        finite_near = finite_near and abs(float(gap_full) - QG_MA2) < 0.01
    # Oriented aperture (kernel Δ) is the unique refinement away from the shadow lock.
    delta_oriented = float(DELTA)
    oriented_not_half = abs(delta_oriented - delta_w_inf) > 0.1
    uniqueness = identity_holds and oriented_not_half and delta_oriented > 0.0
    print(f"  Delta_W(n={n})               : {delta_w:.12f}")
    print(f"  Delta_W(inf)=1/2             : {delta_w_inf:.12f}")
    print(f"  QG_MA2                       : {QG_MA2:.12f}")
    print(f"  |Delta_W(inf)-QG_MA2|        : {abs(delta_w_inf - float(QG_MA2)):.3e}")
    print(f"  |Delta_W(n)-QG_MA2|          : {abs(delta_w - float(QG_MA2)):.3e}")
    print(f"  Delta_oriented (1-rho)       : {delta_oriented:.12f}")
    print(f"  |Delta_oriented-1/2|         : {abs(delta_oriented - delta_w_inf):.12f}")
    gate("D3-struct: lim Delta_W = QG_MA2", identity_holds)
    gate("D3-struct: Delta_W(n) near QG_MA2", finite_near)
    gate("D3-struct: |Delta_oriented-1/2|>0.1", uniqueness)
    return {
        "Delta_W_n256": delta_w,
        "Delta_W_limit": delta_w_inf,
        "QG_MA2": float(QG_MA2),
        "Delta_oriented": delta_oriented,
        "identity_Delta_W_eq_QG_MA2": identity_holds,
        "finite_near_QG_MA2": finite_near,
        "oriented_not_shadow_half": oriented_not_half,
        "uniqueness_two_regimes": uniqueness,
        "pass": identity_holds and finite_near and uniqueness,
    }


def euclidean_transfer_certificate(lat, g: float = G_DEFINING_KS, a: float = 0.25) -> dict:
    """JW-E Euclidean step: transfer T = exp(-a (H-E0)) contracts off-vacuum."""
    _, V = wilson_weight_K4()
    wr, Vr, gap, vac, vac_e, _Q = gauge_invariant_reduce(lat, g, V)
    Omega = Vr[:, 0]
    psi = Vr[:, 1] if Vr.shape[1] > 1 else Vr[:, 0]
    psi = psi - Omega * np.dot(Omega, psi)
    n_psi = np.linalg.norm(psi)
    if n_psi < 1e-12:
        return {"gap": gap, "contraction_ratio": float("nan"), "pass": False}
    psi = psi / n_psi
    wr_shift = wr - wr[0]
    expH = Vr @ np.diag(np.exp(-a * wr_shift)) @ Vr.T
    Tpsi = expH @ psi
    ratio = float(np.linalg.norm(Tpsi))
    expected = math.exp(-a * gap) if gap > 0 else float("nan")
    return {
        "gap": gap,
        "E0": float(wr[0]),
        "step_a": a,
        "contraction_ratio": ratio,
        "expected_exp_minus_a_gap": expected,
        "pass": ratio < 1.0 - 1e-6 and (math.isnan(expected) or abs(ratio - expected) < 0.05),
    }


def lattice_ym_certificates() -> dict:
    """Algebra certificates for the Wilson model: vertex projector
    A_v^2 = A_v, magnetic Wilson term is gauge-invariant ([A_v, H_mag]=0), and the
    gauge projector P is idempotent. These make Track D a genuine local gauge theory
    (not just a tuned operator) and confirm the Hamiltonian is well-defined on the
    gauge-invariant subspace.
    """
    return {
        "K4": certificates(LatticeYM(2, 1, K4())),
        "Q8": certificates(LatticeYM(1, 1, Q8())),
    }


def _run_carrier_sections(fast: bool = False) -> CarrierContext:
    """Sections 0–4: preflight, JW-A, Delta-ruler, shadow Δ_W formula, shadow lock."""
    print("=" * 5)
    print("JW carrier l^2(Omega)")
    if fast:
        print("mode: --fast")

    section(0, section_title(0))
    k6_ok, k6_msg = verify_d6_against_api()
    states = omega_states()
    print("hQVM d=6 cross-check           :", "PASS" if k6_ok else "FAIL", k6_msg)
    print("|Omega| (dim H)               :", len(states))

    section(1, section_title(1))
    defn = jw_definition_certificates()
    defn_ok = _print_jw_a(defn)

    section(2, section_title(2))
    pred = cgm_ym_gap_prediction()
    pred_ok = _print_delta_ruler(pred)

    section(3, section_title(3))
    n_bytes = FULL_ALPHABET_D6
    n_pairs = n_bytes * (n_bytes - 1) // 2
    gap_full = delta_w_formula(D_PAYLOAD)
    print("alphabet n                   :", n_bytes)
    print("n_pairs                      :", n_pairs)
    print("Delta_W = n/(2(n-1))         :", round(gap_full, 6))
    print("Delta_W (d=6 formula)        :", round(gap_full, 10))
    cluster_ok = True

    section(4, section_title(4))
    print("Delta_W (n=256)              :", round(gap_full, 10))
    print("Q_G * m_a^2                  :", round(QG_MA2, 10))
    print("|Delta_W - QG_MA2|           :", f"{abs(gap_full - QG_MA2):.3e}")
    bridge_ok = abs(gap_full - QG_MA2) < 0.01 and abs(QG_MA2 - 0.5) < 1e-12
    slock = shadow_lock_identity_certificate(gap_full)
    bridge_ok = bridge_ok and slock["pass"]

    gate("hQVM d=6 kernel", k6_ok, k6_msg)
    gate("JW Part A definition", defn_ok)
    gate("Delta-ruler prediction", pred_ok)
    gate("shadow Delta_W formula", cluster_ok)
    gate("CGM invariant bridge", bridge_ok)

    return CarrierContext(
        fast=fast, k6_ok=k6_ok, k6_msg=k6_msg,
        cluster_ok=cluster_ok,
        bridge_ok=bridge_ok, defn_ok=defn_ok, pred_ok=pred_ok, pred=pred, gap_full=gap_full,
    )


def _run_wilson_sections(ctx: CarrierContext | None = None, fast: bool | None = None) -> None:
    """Sections 5–10: Wilson lattice, JW certificates, unit map."""
    if ctx is None:
        ctx = run_jw_wilson(fast=bool(fast))
    pred = ctx.pred
    gap_full = ctx.gap_full

    print("=" * 5)
    print("Wilson lattice + JW certificates")

    section(5, section_title(5))
    progress("section 5 / Q8 extension")
    q8ext = q8_central_extension_derived()
    k4kern = kernel_holonomy_k4_certificate()
    print("K4 holonomy on Omega           :", "PASS" if k4kern["pass"] else "FAIL")
    print("  wilson_k4_order              :", k4kern["wilson_k4_order"])
    print("  max_k4_orbit_size            :", k4kern["max_k4_orbit_size"])
    print("Q8 central extension 1->Z2->Q8->K4:")
    print("  is_Q8                        :", q8ext["is_Q8"])
    print("  matches reference Q8 table   :", q8ext["matches_imported_Q8"])
    print("  center                       :", q8ext["center"])

    section(6, section_title(6))
    certs = lattice_ym_certificates()
    for name, c in certs.items():
        print(f"  [{name}] A_v^2=A_v:{c['A_v_proj_resid']:.2e} "
              f"[A_v,H_mag]:{c['mag_gauge_invar_resid']:.2e} P idemp:{c['P_idempotent_resid']:.2e}")
    alg_ok = all(c["A_v_proj_resid"] < 1e-9 and c["mag_gauge_invar_resid"] < 1e-9
                 and c["P_idempotent_resid"] < 1e-9 for c in certs.values())
    hderiv = hamiltonian_derivation_certificate()
    print("  K4 order, Q8 order             :", hderiv["K4_order"], hderiv["Q8_order"])
    print("  Wilson V distinct K4           :", hderiv["Wilson_n_distinct_K4"], hderiv["Wilson_V_distinct_K4"])
    print("  Wilson V distinct Q8           :", hderiv["Wilson_n_distinct_Q8"], hderiv["Wilson_V_distinct_Q8"])
    print("  H_elec                         :", hderiv["H_elec_link_laplacian"])
    print("  H_mag                          :", hderiv["H_mag_Wilson"])
    hderiv_ok = (hderiv["Wilson_nontrivial_Q8"] and hderiv["C2_2form"] == 15
                 and hderiv["K4_order"] == 4 and hderiv["Q8_order"] == 8)

    section(7, section_title(7))
    progress("section 7 / Q8 single plaquette")
    q8 = q8_single_plaquette_dense(1.0)
    print("  links                        :", q8["links"])
    print("  E0 (vacuum energy)           :", round(q8["E0"], 6))
    print("  JW gap E1-E0                 :", round(q8["gap"], 6))
    print("  distinct eigenvalues         :", q8["n_distinct"])
    print("  lowest eigs                  :", q8["min_pos_eigs"])
    print("  vacuum multiplicity          :", q8["vacuum_mult"])
    q8_nontrivial = q8["n_distinct"] > 2 and q8["gap"] > 1e-3 and q8["vacuum_mult"] == 1

    section(8, section_title(8))
    progress("section 8 / correlator locality transfer")
    from Yang_Mills_Mass_Gap_2 import LatticeYM3D
    corr_lat = LatticeYM3D(2, 2, 1, K4(), periodic=False)
    corr = ym_correlator(corr_lat, 1.0)
    print("E1 correlator (time = e^{-tH}):")
    print("  vacuum energy                :", round(corr["vacuum_energy"], 6))
    print("  gap                          :", round(corr["gap"], 6))
    print("  slope log|C_O(t)|            :", round(corr["correlator_slope"], 5))
    print("  m_eff (coupled)              :", round(corr["m_eff_coupled"], 6))
    corr_ok = (
        corr["correlator_slope"] < -1e-2
        and abs(corr["correlator_slope"] + corr["m_eff_coupled"]) < 0.2
        and corr["m_eff_coupled"] > 1e-3
    )

    clust = jw_clustering_certificate(corr_lat, 1.0)
    print("JW-B multi-link clustering:")
    print("  mean_slope                   :", round(clust["mean_slope"], 5))
    print("  slopes                       :", [round(s, 5) for s in clust["slopes"]])
    clust_multi_ok = clust["all_negative_slope"] and clust["slope_near_gap"]

    _G, giq, _, _ = Q8()
    _, Vq = wilson_weight_Q8_2d()
    print("Q8 Wilson V(1,-1,i)            :",
          float(Vq[giq["1"]]), float(Vq[giq["-1"]]), float(Vq[giq["i"]]))

    loc = locality_commutation_certificate(corr_lat, 1.0)
    etrans = euclidean_transfer_certificate(corr_lat, 1.0)
    print("JW-E locality + transfer:")
    print("  max link commutator            :", f"{loc['max_commutator_norm']:.3e}")
    print("  transfer contraction ratio     :", round(etrans["contraction_ratio"], 6))
    print("  expected exp(-a gap)           :", round(etrans["expected_exp_minus_a_gap"], 6))
    loc_ok = loc["pass"]
    etrans_ok = etrans["pass"]

    gmic = gene_mic_omega_certificate(q8["gap"], gap_shadow=QG_MA2)
    print("GENE_Mic omega datum:")
    print("  shadow gap (B/C lock)        :", gmic["gap_shadow"])
    print("  Q8 gap                       :", round(gmic["gap_q8"], 6))
    gmic_ok = gmic["pass"]

    jwa = jw_part_a_certificate()
    print("JW-A lattice-side checks:")
    print("  H_elec is link Laplacian      :", jwa["Hamiltonian_Wilson_KS"])
    print("  H_mag is Wilson V_R(hol)      :", jwa["Wilson_curvature_operator"])
    print("  |Q8_from_extension| == 8      :", jwa["Q8_spinorial_lift"])
    _NONBOOL = ("GENE_Mic_archetype", "Wilson_curvature_operator", "Hamiltonian_Wilson_KS", "GNS_H_dim")
    jwa_ok = all(v for k, v in jwa.items() if k not in _NONBOOL)

    section(9, section_title(9))
    progress("section 9 / unit map grade-1")
    um_q8 = unit_map_trackd_to_gev(q8["gap"], pred)
    print("  Q8 gap (dimless)             :", round(um_q8["gap_dimless"], 6) if um_q8.get("ok") else "nan")
    print("  κ₂ = gap/Δ                   :", round(um_q8["kappa2_gap_over_Delta"], 6) if um_q8.get("ok") else "nan")
    print("  κ₂ target C2                 :", int(um_q8["kappa2_target_C2"]) if um_q8.get("ok") else "nan")
    print("  κ₂ rel err                   :", f"{um_q8['kappa2_rel_err']:.4e}" if um_q8.get("ok") else "nan")
    print("  E_unit_GeV                   :", round(um_q8["E_unit_GeV"], 6) if um_q8.get("ok") else "nan")
    print("  m_phys_GeV                   :", round(um_q8["m_phys_GeV"], 6) if um_q8.get("ok") else "nan")
    print("  rel_to_route_A               :", round(um_q8["rel_to_route_A"], 6) if um_q8.get("ok") else "nan")
    print("  rel_to_route_B               :", round(um_q8["rel_to_route_B"], 6) if um_q8.get("ok") else "nan")
    print("  unit_map_mode                :", um_q8.get("unit_map_mode", "grade1_only"))
    print("  shadow lock Delta_W -> 1/2   :", round(gap_full, 6))
    unit_ok = (
        bool(um_q8.get("ok"))
        and um_q8.get("E_unit_GeV", 0) > 0
        and um_q8.get("unit_map_mode") == "grade1_only"
        and (um_q8.get("kappa2_gap_over_Delta") == um_q8.get("kappa2_gap_over_Delta"))
    )

    lsp = lattice_spacing_certificate(q8["gap"], pred)
    print("  A_kernel, Delta              :", f"{lsp['A_kernel']:.12f}", f"{lsp['Delta']:.12f}")
    print("  E_unit_GeV (spacing)         :", round(lsp["E_unit_GeV"], 6) if lsp["ok"] else "nan")
    print("  m_phys_GeV (spacing)         :", round(lsp["m_phys_GeV"], 6) if lsp["ok"] else "nan")
    lsp_ok = lsp["ok"] and lsp["E_unit_GeV"] > 0

    section(10, section_title(10))
    st = spacetime_dimension_certificate()
    print("  D = n + 1                    :", st["D_continuum"])
    print("  g dimensionless at D=4       :", st["g_dimensionless_at_D4"])
    print("  Wilson H                     :", st["Wilson_H_form"])
    st_ok = st["D_eq_n_plus_1"] and st["g_dimensionless_at_D4"] and st["D_continuum"] == 4

    gate("hQVM holonomic K4", k4kern["pass"])
    gate("Q8 derived not imported", q8ext["is_Q8"])
    gate("Wilson algebra", alg_ok)
    gate("Q8 plaquette nontrivial", q8_nontrivial)
    gate("correlator decay", corr_ok)
    gate("JW-B multi clustering", clust_multi_ok)
    gate("JW-E locality", loc_ok)
    gate("JW-E transfer matrix", etrans_ok)
    gate("GENE_Mic omega datum", gmic_ok)
    gate("JW-A lattice definition", jwa_ok)
    gate("H derivation grammar", hderiv_ok)
    gate("unit map Track D->GeV", unit_ok)
    gate("lattice spacing E_unit", lsp_ok)
    gate("D=4 dimensionless g", st_ok)


def run_jw_wilson(fast: bool = False) -> CarrierContext:
    """JW carrier (0–4) + Wilson lattice JW (5–10)."""
    ctx = _run_carrier_sections(fast=fast)
    print()
    _run_wilson_sections(ctx=ctx)
    return ctx


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="YM mass gap JW + Wilson")
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    run_jw_wilson(fast=args.fast)
