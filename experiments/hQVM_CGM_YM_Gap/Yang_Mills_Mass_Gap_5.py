#!/usr/bin/env python3
"""Yang-Mills mass gap — Formalism Clay checklist + H7 Formalism aggregate.

Sections 21–22. Delivery path. Companion:
  docs/Notes/drops/mass gap/Clay_via_hQVM_Hopf.md
  Yang_Mills_Mass_Gap_Solution.md
  hQVM_Specs_Formalism.md
Orchestrator: Yang_Mills_Mass_Gap_run.py.
"""

from __future__ import annotations

import argparse
import math
import sys
from math import comb

import Yang_Mills_Mass_Gap_common  # noqa: F401 — repo path setup

from gyroscopic.hQVM.api import (
    OMEGA_STATES_4096,
    q_word6,
    shadow_partner_byte,
    so3_shadow_count,
)
from gyroscopic.hQVM.constants import (
    APERTURE_GAP,
    APERTURE_GAP_Q256,
    GATE_NAMES,
    GENE_MAC_REST,
    GENE_MIC_S,
    HORIZON_SIZE,
    OMEGA_SIZE,
    apply_gate,
    byte_to_intron,
    intron_family,
    intron_micro_ref,
    is_on_equality_horizon,
    is_on_horizon,
)
from gyroscopic.hQVM.family import (
    alphabet_size,
    byte_from_family_micro,
    depth4_projection_bits,
    mean_byte_curvature_rate,
    mean_carrier_entanglement_d,
    mean_fold_disagreement_d,
    partition_Z1_coeff_d,
    phase_pairs_d,
    shell_population_d,
    verify_carrier_entanglement_exact,
    verify_f_squared_rest_d,
)

from Yang_Mills_Mass_Gap_common import (
    BU_CLOSURE_DEPTH,
    BYTE256,
    CODE_C2,
    DELTA,
    D_CONTINUUM_CGM,
    D_PAYLOAD,
    E_EW_GEV,
    K4_ORDER,
    N_SPATIAL_CGM,
    N_TEMPORAL_CGM,
    Q8_ORDER,
    Q_G,
    QG_MA2,
    RHO,
    SE3_DOF,
    SHELLS7,
    TRANSPORT_SIZE,
    gate,
    permute_payload_byte,
    progress,
    section,
    section_title,
)


D = D_PAYLOAD


def oriented_shadow_certificate() -> dict:
    """Reference-preserving shadow vs over-coarse half-gap collapse.

    Checks: GENE_Mic orients ω⋆; Δ > 0; SO(3) 2-to-1 shadow; fold holonomy
    nonzero; combinatorial Δ_W(n) → 1/2 without orientation. Companion: Solution §1.1a.
    """
    gene_ok = byte_to_intron(GENE_MIC_S) == 0
    delta = float(APERTURE_GAP)
    delta_ok = delta > 0.0
    flat, curved, _ = mean_byte_curvature_rate(D)
    fold_ok = flat == 16 and curved == 240
    shadow_n = so3_shadow_count(GENE_MAC_REST)
    partner_pairs = len({frozenset((b, shadow_partner_byte(b))) for b in range(BYTE256)})
    so3_ok = shadow_n == 128 and partner_pairs == 128
    n = BYTE256
    delta_w = n / (2.0 * (n - 1.0))
    half_lock = abs(delta_w - QG_MA2) < 2e-3
    print(f"  GENE_Mic zero intron        : {gene_ok}")
    print(f"  Delta > 0                   : {delta_ok} ({delta:.12f})")
    print(f"  fold flat/curved            : {flat}/{curved}")
    print(f"  SO3 shadow / partner pairs  : {shadow_n}/{partner_pairs}")
    print(f"  Delta_W(n={BYTE256}) / QG_MA2 lock : {delta_w:.9f} {half_lock}")
    ok = gene_ok and delta_ok and fold_ok and so3_ok and half_lock
    gate("oriented shadow: GENE_Mic", gene_ok)
    gate("oriented shadow: Delta>0", delta_ok)
    gate("oriented shadow: fold holonomy", fold_ok)
    gate("oriented shadow: SO3 2-to-1", so3_ok)
    gate("oriented shadow: D3 QG_MA2 lock", half_lock)
    return {
        "GENE_Mic_orients": gene_ok,
        "Delta": delta,
        "Delta_positive": delta_ok,
        "fold_flat": flat,
        "fold_curved": curved,
        "SO3_shadow": shadow_n,
        "shadow_partner_pairs": partner_pairs,
        "Delta_W_n256": delta_w,
        "half_gap_formula": half_lock,
        "pass": ok,
    }


def hopf_dictionary_certificate() -> dict:
    """Exact discrete↔Hopf dictionary (Clay_via_hQVM_Hopf.md)."""
    pairs = phase_pairs_d(D)
    # fwd bits 0–3 / rev bits 4–7 / fold at BU (3,4)
    ok_phases = pairs == ((0, 7), (1, 6), (2, 5), (3, 4))
    flat, curved, _ = mean_byte_curvature_rate(D)
    mean_fd = mean_fold_disagreement_d(D)
    shadow_n = so3_shadow_count(GENE_MAC_REST)
    partner_pairs = len({frozenset((b, shadow_partner_byte(b))) for b in range(BYTE256)})
    proj_bits = depth4_projection_bits(D)
    n = int(N_SPATIAL_CGM)
    ok_flat = flat == 16
    ok_curved = curved == 240
    ok_fd = abs(mean_fd - 0.5) < 1e-12
    ok_so3 = shadow_n == 128 and partner_pairs == 128
    ok_proj = proj_bits == 48
    ok_qg = abs(Q_G - 4.0 * math.pi) < 1e-12
    ok_D = D_CONTINUUM_CGM == n + 1 == 4 and n == 3
    print(f"  phase_pairs (fwd/rev/BU)   : {pairs}")
    print(f"  flat / curved / mean_fd    : {flat} {curved} {mean_fd:.6f}")
    print(f"  SO3 / partner pairs        : {shadow_n} {partner_pairs}")
    print(f"  depth4 projection bits     : {proj_bits}")
    print(f"  Q_G / D=n+1                : {Q_G:.10f} {D_CONTINUUM_CGM} (n={n})")
    gate("hopf dict: fwd/rev/fold BU", ok_phases)
    gate("hopf dict: flat=16 curved=240", ok_flat and ok_curved)
    gate("hopf dict: mean_fd=1/2", ok_fd)
    gate("hopf dict: SO3=128 pairs=128", ok_so3)
    gate("hopf dict: depth4=48", ok_proj)
    gate("hopf dict: Q_G=4pi D=n+1=4", ok_qg and ok_D)
    ok = all((ok_phases, ok_flat, ok_curved, ok_fd, ok_so3, ok_proj, ok_qg, ok_D))
    return {
        "phase_pairs": pairs,
        "flat": flat,
        "curved": curved,
        "mean_fd": mean_fd,
        "SO3_shadow": shadow_n,
        "partner_pairs": partner_pairs,
        "depth4_bits": proj_bits,
        "Q_G": Q_G,
        "D": D_CONTINUUM_CGM,
        "n": n,
        "pass": ok,
    }


def admissible_quotient_certificate(shadow: dict | None = None) -> dict:
    """Admissible shadow checks (Solution §1.1a). Quotient ladder Byte256→q6→shells7."""
    sh = shadow if shadow is not None else oriented_shadow_certificate()
    q6_classes = {q_word6(b) for b in range(BYTE256)}
    byte_n = BYTE256
    q6_n = len(q6_classes)
    chi_n = TRANSPORT_SIZE
    byte_over_q6 = byte_n // q6_n
    omega_over_chi = OMEGA_SIZE // chi_n
    ok_bq = byte_over_q6 == K4_ORDER and q6_n == TRANSPORT_SIZE
    ok_oc = omega_over_chi == TRANSPORT_SIZE and OMEGA_SIZE == TRANSPORT_SIZE ** 2
    print(f"  |Byte|/|q6|                 : {byte_over_q6} (expect {K4_ORDER})")
    print(f"  |q6| enumerated             : {q6_n} (expect {TRANSPORT_SIZE})")
    print(f"  |Omega|/|chi|               : {omega_over_chi} (expect {TRANSPORT_SIZE})")
    print(f"  OMEGA_SIZE                  : {OMEGA_SIZE}")
    gate("admissible: oriented shadow", sh["pass"])
    gate("admissible: |Byte|/|q6|=4", ok_bq)
    gate("admissible: |Omega|/|chi|=64", ok_oc)
    ok = bool(sh["pass"]) and ok_bq and ok_oc
    return {
        "oriented_shadow_pass": sh["pass"],
        "Byte_over_q6": byte_over_q6,
        "q6_enumerated": q6_n,
        "Omega_over_chi": omega_over_chi,
        "pass": ok,
    }


def gap_on_quotient_certificate() -> dict:
    """Gap on admissible/Hopf-oriented data: Δ, C2, m_A, D0-3D, IsoSupport."""
    from Yang_Mills_Mass_Gap_4 import (
        d0_3d_dark_intersection_certificate,
        d0_iso_support_certificate,
    )

    delta = float(APERTURE_GAP)
    delta_ok = delta > 0.0
    c2_ok = CODE_C2 == 15
    m_A = float(CODE_C2) * float(E_EW_GEV) * (delta**2)
    d0 = d0_3d_dark_intersection_certificate()
    iso = d0_iso_support_certificate()
    d0_ok = bool(d0.get("pass"))
    iso_ok = bool(iso.get("pass"))
    n2_ok = int(iso.get("N2_target_C2", -1)) == 15 and bool(iso.get("channels_eq_C2"))
    print(f"  Delta (APERTURE_GAP)        : {delta:.12f}")
    print(f"  C2                          : {CODE_C2}")
    print(f"  m_A = C2*v*Delta^2 (GeV)    : {m_A:.6f}")
    print(f"  D0-3D dark ∩ empty          : {d0_ok}")
    print(f"  IsoSupport transitive       : {iso_ok}")
    print(f"  N2_target=15                : {n2_ok}")
    gate("gap quotient: Delta>0", delta_ok)
    gate("gap quotient: C2=15", c2_ok)
    gate("gap quotient: D0-3D", d0_ok)
    gate("gap quotient: IsoSupport", iso_ok)
    gate("gap quotient: N2=15", n2_ok)
    ok = delta_ok and c2_ok and d0_ok and iso_ok and n2_ok
    return {
        "Delta": delta,
        "C2": CODE_C2,
        "m_A_GeV": m_A,
        "D0_3D_pass": d0_ok,
        "IsoSupport_pass": iso_ok,
        "N2_target_15": n2_ok,
        "pass": ok,
    }


def hopf_fiber_census_certificate() -> dict:
    """H0: byte fold fiber census + SO(3) shadow + shadow-partner involution."""
    flat, curved, curved_rate = mean_byte_curvature_rate(D)
    mean_fd = mean_fold_disagreement_d(D)
    mean_S = mean_carrier_entanglement_d(D)
    ent_ok, ent_got, ent_exp = verify_carrier_entanglement_exact(D)
    f_ok, f_n = verify_f_squared_rest_d(D)
    proj_bits = depth4_projection_bits(D)
    q48 = 1.0 / proj_bits
    residual = mean_fd / (4.0 * D)
    shadow_n = so3_shadow_count(GENE_MAC_REST)
    partner_invol = all(
        shadow_partner_byte(shadow_partner_byte(b)) == b for b in range(BYTE256)
    )
    partner_fixed = sum(1 for b in range(BYTE256) if shadow_partner_byte(b) == b)
    partner_pairs = len({frozenset((b, shadow_partner_byte(b))) for b in range(BYTE256)})

    print(f"  alphabet                   : {flat + curved}")
    print(f"  flat (fwd=rev)             : {flat}")
    print(f"  curved (Z2 fold)           : {curved}")
    print(f"  curved_rate                : {curved_rate:.6f}")
    print(f"  mean_fold_disagreement     : {mean_fd:.6f}")
    print(f"  mean_carrier_S/d           : {mean_S:.6f}")
    print(f"  entanglement sum           : {ent_got} (expect {ent_exp})")
    print(f"  F^2=id on rest (micros)    : {f_ok}/{f_n}")
    print(f"  depth4 projection bits     : {proj_bits}")
    print(f"  Q_48(Delta)~1/48           : {q48:.6f}")
    print(f"  spinorial residual         : {residual:.6f}")
    print(f"  SO3 shadow |next| @ rest   : {shadow_n}")
    print(f"  shadow partner involution  : {partner_invol}")
    print(f"  shadow partner fixed pts   : {partner_fixed}")
    print(f"  shadow partner pairs       : {partner_pairs}")

    ok_flat = flat == 16
    ok_curved = curved == 240
    ok_mean_fd = abs(mean_fd - 0.5) < 1e-12
    ok_mean_S = abs(mean_S - 0.5) < 1e-12
    ok_ent = ent_ok
    ok_f = f_ok == f_n
    ok_proj = proj_bits == 48
    ok_shadow = shadow_n == HORIZON_SIZE * 2  # SO(3) shadow 128 = 2×|H|
    ok_partner = partner_invol and partner_fixed == 0 and partner_pairs == 128
    gate("hopf flat=16", ok_flat)
    gate("hopf curved=240", ok_curved)
    gate("hopf mean_fd=1/2", ok_mean_fd)
    gate("hopf mean_S/d=1/2", ok_mean_S)
    gate("hopf entanglement exact", ok_ent)
    gate("hopf F^2 rest", ok_f)
    gate("hopf depth4=48 bits", ok_proj)
    gate("hopf SO3 shadow=128", ok_shadow)
    gate("hopf shadow partner 128 pairs", ok_partner)

    closed = all(
        (
            ok_flat,
            ok_curved,
            ok_mean_fd,
            ok_mean_S,
            ok_ent,
            ok_f,
            ok_proj,
            ok_shadow,
            ok_partner,
        )
    )
    return {
        "flat": flat,
        "curved": curved,
        "curved_rate": curved_rate,
        "mean_fold_disagreement": mean_fd,
        "mean_carrier_S_over_d": mean_S,
        "F2_rest_ok": f_ok,
        "F2_rest_n": f_n,
        "depth4_bits": proj_bits,
        "Q48": q48,
        "spinorial_residual": residual,
        "SO3_shadow": shadow_n,
        "shadow_partner_pairs": partner_pairs,
        "H0_closed": closed,
    }


def lemma_G_certificate() -> dict:
    """Gauge G: GENE_Mic → family(K4)×payload(GF(2)^6)=Byte256; K4 gate elements."""
    fam_counts = [0] * K4_ORDER
    fam_payloads: list[set[int]] = [set() for _ in range(K4_ORDER)]
    gene_ok = byte_to_intron(GENE_MIC_S) == 0
    roundtrip_ok = True
    for b in range(BYTE256):
        intron = byte_to_intron(b)
        fam = intron_family(intron)
        pay = intron_micro_ref(intron)
        if not (0 <= fam < K4_ORDER):
            roundtrip_ok = False
            break
        fam_counts[fam] += 1
        fam_payloads[fam].add(pay)
        if byte_from_family_micro(fam, pay, D) != b:
            roundtrip_ok = False
            break
    per_family_ok = all(c == TRANSPORT_SIZE for c in fam_counts)
    payload_cover_ok = all(len(s) == TRANSPORT_SIZE for s in fam_payloads)
    n_payload = D
    n_alphabet = alphabet_size(D)
    print(f"  GENE_Mic zero intron       : {gene_ok}")
    print(f"  payload bits (se(3) DoF)   : {n_payload} (SE3_DOF={SE3_DOF})")
    print(f"  families (K4 L0 labels)    : {K4_ORDER} counts={fam_counts}")
    print(f"  alphabet                   : {n_alphabet}")
    print(f"  family×payload roundtrip   : {roundtrip_ok}")
    print(f"  K4 gate names              : {GATE_NAMES}")
    ok = (
        gene_ok
        and roundtrip_ok
        and per_family_ok
        and payload_cover_ok
        and n_payload == SE3_DOF
        and n_alphabet == BYTE256
        and tuple(GATE_NAMES) == ("id", "S", "C", "F")
        and SHELLS7 == D + 1
    )
    gate("G GENE_Mic+family×payload", ok)
    return {
        "GENE_Mic_orients": gene_ok,
        "payload_bits": n_payload,
        "families": K4_ORDER,
        "family_counts": fam_counts,
        "alphabet": n_alphabet,
        "roundtrip_ok": roundtrip_ok,
        "G_closed": ok,
    }


def lemma_R4_certificate() -> dict:
    """3+1 packaging: spatial 3, temporal depth-4, D=4, Q_G=4π."""
    print(f"  N_spatial                  : {N_SPATIAL_CGM}")
    print(f"  N_temporal (BU depth)      : {N_TEMPORAL_CGM}")
    print(f"  BU_CLOSURE_DEPTH           : {BU_CLOSURE_DEPTH}")
    print(f"  D_continuum                : {D_CONTINUUM_CGM}")
    print(f"  Q_G                        : {Q_G:.10f}")
    ok = (
        N_SPATIAL_CGM == 3
        and N_TEMPORAL_CGM == 1
        and BU_CLOSURE_DEPTH == 4
        and D_CONTINUUM_CGM == 4
        and abs(Q_G - 4.0 * math.pi) < 1e-12
    )
    gate("R4 3+1 + Q_G=4pi", ok)
    return {
        "N_spatial": N_SPATIAL_CGM,
        "N_temporal": N_TEMPORAL_CGM,
        "D": D_CONTINUUM_CGM,
        "Q_G": Q_G,
        "R4_closed": ok,
    }


def lemma_M_certificate() -> dict:
    """QuBEC Z1(λ)=64(1+λ)^6 and shell census."""
    shells = [shell_population_d(D, k) for k in range(D + 1)]
    total = sum(shells)
    expect_shells = [comb(D, k) * (1 << D) for k in range(D + 1)]
    lam = 1.0
    z1 = partition_Z1_coeff_d(D, lam)
    z1_expect = (1 << D) * ((1.0 + lam) ** D)
    samples = []
    for lam_s in (0.5, 1.0, 2.0):
        z_poly = sum(shells[k] * (lam_s**k) for k in range(D + 1))
        z_closed = partition_Z1_coeff_d(D, lam_s)
        samples.append((lam_s, z_poly, z_closed, abs(z_poly - z_closed) < 1e-9))

    print(f"  shell populations          : {shells}")
    print(f"  sum shells                 : {total} (Omega={OMEGA_SIZE})")
    print(f"  Z1(1)                      : {z1:.6f} (expect {z1_expect:.6f})")
    for lam_s, z_poly, z_closed, ok_s in samples:
        print(f"  Z1({lam_s}) poly vs closed  : {z_poly:.6f} / {z_closed:.6f}")

    ok_shells = shells == expect_shells
    ok_total = total == OMEGA_SIZE
    ok_z = abs(z1 - z1_expect) < 1e-12
    ok_samples = all(s[3] for s in samples)
    gate("M shell census", ok_shells and ok_total)
    gate("M Z1 closed form", ok_z and ok_samples)
    return {
        "shells": shells,
        "Z1_1": z1,
        "M_closed": ok_shells and ok_total and ok_z and ok_samples,
    }


def lemma_H_certificate() -> dict:
    """Hilbert / vacuum sector: ℓ²(Ω), horizons, gate F involution, GENE_Mac rest."""
    n = len(OMEGA_STATES_4096)
    eq = sum(1 for s in OMEGA_STATES_4096 if is_on_equality_horizon(s))
    comp = sum(1 for s in OMEGA_STATES_4096 if is_on_horizon(s))
    f2_ok = 0
    f_fixed = 0
    for s in OMEGA_STATES_4096:
        fs = apply_gate(s, "F")
        if apply_gate(fs, "F") == s:
            f2_ok += 1
        if fs == s:
            f_fixed += 1
    plus_dim = OMEGA_SIZE // 2
    minus_dim = OMEGA_SIZE // 2
    rest_on_comp = is_on_horizon(GENE_MAC_REST)

    print(f"  |Omega|                    : {n}")
    print(f"  equality horizon           : {eq} (expect {HORIZON_SIZE})")
    print(f"  complement horizon         : {comp} (expect {HORIZON_SIZE})")
    print(f"  |H|^2                      : {HORIZON_SIZE ** 2}")
    print(f"  F^2=id count               : {f2_ok}/{n}")
    print(f"  F fixed points             : {f_fixed}")
    print(f"  F ±1 eigenspace dims      : {plus_dim}, {minus_dim}")
    print(f"  GENE_MAC_REST              : 0x{GENE_MAC_REST:06X}")
    print(f"  GENE_Mac rest on complement: {rest_on_comp}")

    ok = (
        n == OMEGA_SIZE
        and eq == HORIZON_SIZE
        and comp == HORIZON_SIZE
        and HORIZON_SIZE**2 == OMEGA_SIZE
        and f2_ok == n
        and f_fixed == 0
        and plus_dim == OMEGA_SIZE // 2
        and rest_on_comp
    )
    gate("H Omega+horizons", n == OMEGA_SIZE and eq == HORIZON_SIZE and comp == HORIZON_SIZE)
    gate("H |H|^2=|Omega|", HORIZON_SIZE**2 == OMEGA_SIZE)
    gate("H gate F involution", f2_ok == n and f_fixed == 0)
    gate("H GENE_Mac rest on complement", rest_on_comp)
    return {
        "Omega": n,
        "equality_horizon": eq,
        "complement_horizon": comp,
        "F2_ok": f2_ok,
        "F_fixed": f_fixed,
        "GENE_Mac_rest_on_complement": rest_on_comp,
        "H_closed": ok,
    }


def lemma_gap_certificate() -> dict:
    """Aperture Δ=1−ρ>0 and C2 ruler skeleton."""
    delta = float(DELTA)
    rho = float(RHO)
    aperture = float(APERTURE_GAP)
    q256 = APERTURE_GAP_Q256 / 256.0
    m_A = CODE_C2 * E_EW_GEV * (delta**2)
    print(f"  rho                        : {rho:.12f}")
    print(f"  Delta (common)             : {delta:.12f}")
    print(f"  APERTURE_GAP               : {aperture:.12f}")
    print(f"  Q_256(Delta)=5/256         : {q256:.12f}")
    print(f"  CODE_C2                    : {CODE_C2}")
    print(f"  m_A = C2*v*Delta^2 (GeV)   : {m_A:.6f}")

    ok_pos = delta > 0.0 and aperture > 0.0
    ok_match = abs(delta - aperture) < 1e-12
    ok_rho = abs(1.0 - rho - delta) < 1e-12
    ok_c2 = CODE_C2 == 15
    ok_q = APERTURE_GAP_Q256 == 5
    gate("gap Delta>0", ok_pos)
    gate("gap Delta=1-rho", ok_rho and ok_match)
    gate("gap C2=15", ok_c2)
    gate("gap Q256=5/256", ok_q)
    return {
        "Delta": delta,
        "rho": rho,
        "Q256": q256,
        "C2": CODE_C2,
        "m_A_GeV": m_A,
        "gap_closed": ok_pos and ok_rho and ok_match and ok_c2 and ok_q,
    }


def formalism_checklist_certificate() -> dict:
    """Aggregate H0 + oriented shadow + Hopf dict + admissible + gap-on-quotient + G/R4/M/H/gap + Schwinger."""
    section(21, section_title(21))
    sys.stdout.flush()
    progress("Hopf fiber census (H0)")
    h0 = hopf_fiber_census_certificate()
    print("-" * 5)
    progress("Oriented shadow (ω⋆ / D3 contrast)")
    sh = oriented_shadow_certificate()
    print("-" * 5)
    progress("Hopf dictionary")
    hd = hopf_dictionary_certificate()
    print("-" * 5)
    progress("Admissible quotient")
    aq = admissible_quotient_certificate(shadow=sh)
    print("-" * 5)
    progress("Gap on quotient")
    gq = gap_on_quotient_certificate()
    print("-" * 5)
    progress("Lemma G — gauge / SE(3)")
    g = lemma_G_certificate()
    print("-" * 5)
    progress("Lemma R4 — 3+1 / S2 horizon scale")
    r4 = lemma_R4_certificate()
    print("-" * 5)
    progress("Lemma M — QuBEC measure")
    m = lemma_M_certificate()
    print("-" * 5)
    progress("Lemma H — Hilbert / horizons")
    h = lemma_H_certificate()
    print("-" * 5)
    progress("Lemma gap — aperture")
    gap = lemma_gap_certificate()
    print("-" * 5)
    progress("Hopf horizon chart")
    schw = hopf_horizon_chart_certificate()
    print("-" * 5)
    progress("κ₂ spectral bound (Q8 1×1)")
    k2 = kappa2_operator_bound_certificate()
    print("-" * 5)
    progress("Γ lift commutator (defining chart)")
    gamma = gamma_physical_lift_certificate()
    print("-" * 5)

    closed = all(
        (
            h0["H0_closed"],
            sh["pass"],
            hd["pass"],
            aq["pass"],
            gq["pass"],
            g["G_closed"],
            r4["R4_closed"],
            m["M_closed"],
            h["H_closed"],
            gap["gap_closed"],
            schw["pass"],
            k2["pass"],
        )
    )
    print(f"  H0_closed                  : {h0['H0_closed']}")
    print(f"  oriented_shadow_pass       : {sh['pass']}")
    print(f"  hopf_dictionary_pass       : {hd['pass']}")
    print(f"  admissible_quotient_pass   : {aq['pass']}")
    print(f"  gap_on_quotient_pass       : {gq['pass']}")
    print(f"  G_closed                   : {g['G_closed']}")
    print(f"  R4_closed                  : {r4['R4_closed']}")
    print(f"  M_closed                   : {m['M_closed']}")
    print(f"  H_closed                   : {h['H_closed']}")
    print(f"  gap_closed                 : {gap['gap_closed']}")
    print(f"  hopf_horizon_chart_pass    : {schw['pass']}")
    print(f"  kappa2_min_bound_pass      : {k2['pass']}")
    print(f"  gamma_aut_ok (defining)    : {gamma.get('aut_ok', False)}")
    print(f"  gamma_carrier_ok           : {gamma.get('carrier_ok', False)}")
    print(f"  formalism_checklist_closed : {closed}")
    gate("formalism_checklist_closed", closed)
    return {
        "H0": h0,
        "oriented_shadow": sh,
        "hopf_dictionary": hd,
        "admissible_quotient": aq,
        "gap_on_quotient": gq,
        "G": g,
        "R4": r4,
        "M": m,
        "H": h,
        "gap": gap,
        "hopf_horizon_chart": schw,
        "kappa2_operator": k2,
        "gamma_lift": gamma,
        "formalism_checklist_closed": closed,
    }


def h7_formalism_aggregate(checklist: dict) -> dict:
    """H7 closes on Formalism checklist (not lattice continuum_gates)."""
    section(22, section_title(22))
    sys.stdout.flush()
    closed = bool(checklist.get("formalism_checklist_closed"))
    m_A = checklist.get("gap_on_quotient", {}).get("m_A_GeV")
    if m_A is None:
        m_A = checklist.get("gap", {}).get("m_A_GeV")
    print(f"  formalism_checklist_closed : {closed}")
    print(f"  H0 / shadow / hopf / adm / gq :",
          checklist.get("H0", {}).get("H0_closed"),
          checklist.get("oriented_shadow", {}).get("pass"),
          checklist.get("hopf_dictionary", {}).get("pass"),
          checklist.get("admissible_quotient", {}).get("pass"),
          checklist.get("gap_on_quotient", {}).get("pass"))
    print(f"  G / R4 / M / H / gap       :",
          checklist.get("G", {}).get("G_closed"),
          checklist.get("R4", {}).get("R4_closed"),
          checklist.get("M", {}).get("M_closed"),
          checklist.get("H", {}).get("H_closed"),
          checklist.get("gap", {}).get("gap_closed"))
    print(f"  m_A_GeV                    : {m_A}")
    print(f"  H7_closed                  : {closed}")
    gate("H7 deliverable closed", closed)
    note = (
        "H7_closed := Formalism + Hopf dictionary + admissible quotient "
        "+ gap on quotient (CGM Clay skeleton)."
    )
    return {
        "formalism_checklist": checklist,
        "pass": closed,
        "H7_closed": closed,
        "note": note,
    }


def run_formalism(fast: bool = False) -> dict:
    _ = fast
    checklist = formalism_checklist_certificate()
    print()
    progress("H7 Formalism aggregate")
    h7 = h7_formalism_aggregate(checklist)
    return {
        **checklist,
        "h7": h7,
        "H7_closed": h7["H7_closed"],
        "pass": checklist["formalism_checklist_closed"] and h7["H7_closed"],
    }


# ============================================================
# Hopf horizon + derived operator certificates
# ============================================================

def hopf_horizon_chart_certificate() -> dict:
    """SO(3) shadow, C2, Δ, m_a, Route A mass on the Hopf horizon chart."""
    print("[99] Hopf horizon chart")
    from Yang_Mills_Mass_Gap_common import M_A

    m_a = float(M_A)
    delta = float(DELTA)
    c2 = int(CODE_C2)
    m_A_GeV = float(c2) * float(E_EW_GEV) * (delta**2)
    S2 = so3_shadow_count(GENE_MAC_REST)

    so3_ok = S2 == 128
    c2_ok = c2 == 15
    delta_ok = delta > 0.0
    ma_ok = abs(Q_G * (m_a**2) - QG_MA2) < 1e-12 and abs(QG_MA2 - 0.5) < 1e-12
    route_a_ok = 1.0 < m_A_GeV < 2.0
    chart_pass = so3_ok and c2_ok and delta_ok and ma_ok and route_a_ok

    print(f"  S2 horizon (SO3 shadow)       : {S2}")
    print(f"  Lambda2 channels C2           : {c2}")
    print(f"  Delta                         : {delta:.12f}")
    print(f"  m_a = 1/(2 sqrt(2 pi))        : {m_a:.12f}")
    print(f"  Q_G * m_a^2 (=QG_MA2)         : {QG_MA2:.12f}")
    print(f"  Route A m_A = C2*v*Delta^2    : {m_A_GeV:.6f} GeV")
    print(f"  chart_pass                    : {chart_pass}")

    gate("Hopf chart: SO3 shadow=128", so3_ok)
    gate("Hopf chart: C2=15", c2_ok)
    gate("Hopf chart: Delta>0", delta_ok)
    gate("Hopf chart: Q_G m_a^2=QG_MA2=1/2", ma_ok)
    gate("Hopf chart: Route A in 1-2 GeV", route_a_ok)
    return {
        "S2_horizon_count": S2,
        "n_curvature_channels": c2,
        "Delta": delta,
        "m_a_dimensionless": m_a,
        "QG_MA2": float(QG_MA2),
        "m_A_GeV": m_A_GeV,
        "pass": chart_pass,
    }



def kappa2_operator_bound_certificate() -> dict:
    """Spectral chart: m_coupled(O_Lambda2) vs per-channel masses; selection-rule scan.

    Proven here: m_sum >= min_ab m_coupled(O_ab).
    Printed only (not a theorem): m_sum vs 15*Delta; below-threshold orthogonality scan.
    """
    import numpy as np
    from Yang_Mills_Mass_Gap_common import (
        LatticeYM,
        Q8,
        Q8_ORDER,
        G_DEFINING_KS,
        correlator_local_mass_from_spectrum,
        gauge_invariant_reduce,
        wilson_weight_Q8_2d,
    )
    from Yang_Mills_Mass_Gap_4 import q8_config_lambda2_channel_diags

    print("[K2-OP] spectral masses + below-threshold scan (Q8 1x1)")
    delta = float(APERTURE_GAP)
    fifteen_delta = float(CODE_C2) * delta
    lat = LatticeYM(1, 1, Q8(), periodic=True)
    _, Vq = wilson_weight_Q8_2d()
    wr, Vr, _gap, _vac, _e0, Q = gauge_invariant_reduce(lat, G_DEFINING_KS, Vq)
    ch = q8_config_lambda2_channel_diags(lat, packing="dual_frame")

    m_k = []
    Od_list = []
    for k in range(CODE_C2):
        Od = Q.T @ np.diag(ch[:, k]) @ Q
        Od = 0.5 * (Od + Od.T)
        Od_list.append(Od)
        m_k.append(float(correlator_local_mass_from_spectrum(wr, Vr, Od)["m_coupled"]))
    Od_sum = Q.T @ np.diag(ch.sum(axis=1)) @ Q
    Od_sum = 0.5 * (Od_sum + Od_sum.T)
    m_sum = float(correlator_local_mass_from_spectrum(wr, Vr, Od_sum)["m_coupled"])
    m_min = min(m_k)
    bound_min = m_sum >= m_min - 1e-12

    # Selection-rule scan (chart): for eigenstates with E_n-E_0 < 15*Delta,
    # measure max_ab |<n|O_ab|0>|.
    Omega = Vr[:, 0]
    E0 = float(wr[0])
    below = []
    max_ov_below = 0.0
    n_below = 0
    for n in range(1, len(wr)):
        dE = float(wr[n] - E0)
        if dE >= fifteen_delta - 1e-15:
            continue
        n_below += 1
        psi = Vr[:, n]
        ov_ab = [abs(float(psi @ (Od @ Omega))) for Od in Od_list]
        ov_sum = abs(float(psi @ (Od_sum @ Omega)))
        mx = max(ov_ab) if ov_ab else 0.0
        max_ov_below = max(max_ov_below, mx, ov_sum)
        below.append({"n": n, "dE": dE, "max_ov_ab": mx, "ov_sum": ov_sum})

    print(f"  m_coupled(O_Lambda2)         : {m_sum:.6f}")
    print(f"  min_ab m_coupled             : {m_min:.6f}")
    print(f"  m_sum >= min_ab              : {bound_min}")
    print(f"  JW_gap                       : {float(wr[1] - E0):.6f}")
    print(f"  15*Delta (reference)         : {fifteen_delta:.6f}")
    print(f"  m_sum - 15*Delta (chart)     : {m_sum - fifteen_delta:.6e}")
    print(f"  n states with dE < 15*Delta  : {n_below}")
    print(f"  max |<n|O|0>| below thresh   : {max_ov_below:.6e}")
    print(f"  selection_scan_vacuous       : {n_below == 0}")
    gate("kappa2: m_sum >= min_channel", bound_min)
    return {
        "m_coupled_sum": m_sum,
        "m_coupled_per_channel": m_k,
        "min_channel_m_coupled": m_min,
        "fifteen_delta": fifteen_delta,
        "bound_min_holds": bound_min,
        "JW_gap": float(wr[1] - E0),
        "n_below_15delta": n_below,
        "max_ov_below_15delta": max_ov_below,
        "selection_scan_vacuous": n_below == 0,
        "below_rows": below,
        "pass": bound_min,
    }


def gamma_physical_lift_certificate() -> dict:
    """Intron+GENE_Mic payload S6 on Ω + Aut(Q8)≅S4 chart symmetry on H_GI.

    Carrier Γ: intron = b⊕GENE_Mic; permute payload bits; family fixed;
    b' = intron'⊕GENE_Mic; P_σ on (u6,v6); P U_b = U_σ(b) P via byte_transition.
    Aut(Q8): configs (h,v)↦(φh,φv), U=Q.T Π_φ Q; H-preserving SO(3) finite bridge.
    pass := carrier_ok and aut_ok.
    """
    import numpy as np
    from gyroscopic.hQVM.api import (
        OMEGA_STATES_4096,
        OmegaState12,
        omega12_to_state24,
        state24_to_omega12,
    )
    from gyroscopic.hQVM.sdk import byte_transition
    from Yang_Mills_Mass_Gap_common import (
        BYTE256,
        G_DEFINING_KS,
        LatticeYM,
        Q8,
        Q8_ORDER,
        gauge_invariant_reduce,
        permute_payload_byte,
        wilson_weight_Q8_2d,
    )
    from Yang_Mills_Mass_Gap_4 import carrier_lattice_intertwiner_certificate

    print("[Gamma-LIFT] intron+GENE_Mic on Ω; Aut(Q8) lift on H_GI")
    G, _gi, table, _inv = Q8()
    lat = LatticeYM(1, 1, Q8(), periodic=True)
    _, V = wilson_weight_Q8_2d()
    _wr, Vr, _gap, _vac, _e0, Q = gauge_invariant_reduce(lat, G_DEFINING_KS, V)
    _op, H_full, _He, _Hm = lat.hamiltonian_operator(G_DEFINING_KS, V)
    H_full = np.asarray(H_full.toarray() if hasattr(H_full, "toarray") else H_full)
    Hred = 0.5 * (Q.T @ (H_full @ Q) + (Q.T @ (H_full @ Q)).T)
    dim_gi = int(Hred.shape[0])
    dim_cfg = int(lat.N ** lat.nE)
    Omega = Vr[:, 0]

    itw = carrier_lattice_intertwiner_certificate()
    W = itw.get("_W_matrix")
    if W is None:
        raise RuntimeError("intertwiner did not return _W_matrix")
    n_om, w_gi = W.shape
    dim_match = (w_gi == dim_gi) and (Hred.shape == (dim_gi, dim_gi))

    def _adj_transpositions(pairs: tuple[tuple[int, int], ...]) -> list[tuple[int, ...]]:
        out = []
        for i, j in pairs:
            p = list(range(D_PAYLOAD))
            p[i], p[j] = p[j], p[i]
            out.append(tuple(p))
        return out

    gens_s6 = _adj_transpositions(tuple((i, i + 1) for i in range(5)))

    def _perm_bits(word6: int, perm: tuple[int, ...]) -> int:
        out = 0
        for i in range(D_PAYLOAD):
            out |= ((word6 >> i) & 1) << perm[i]
        return out

    def _sigma_byte(b: int, perm: tuple[int, ...]) -> int:
        return permute_payload_byte(b, perm)

    def _act_state24(s: int, perm: tuple[int, ...]) -> int:
        st = state24_to_omega12(int(s))
        return int(omega12_to_state24(OmegaState12(
            u6=_perm_bits(st.u6, perm),
            v6=_perm_bits(st.v6, perm),
        )))

    omega_list = list(OMEGA_STATES_4096)

    byte_perm_ok = True
    for perm in gens_s6:
        if len({_sigma_byte(b, perm) for b in range(BYTE256)}) != BYTE256:
            byte_perm_ok = False
            break

    sample_states = [omega_list[i] for i in (0, 1, 17, 64, 255, 1024, 2047, 4095)]
    sample_bytes = (0x00, 0x01, GENE_MIC_S, 0x55, 0x0F, 0xF0, 0x12, 0xC3)
    intertwine_ok = True
    intertwine_checks = 0
    for perm in gens_s6:
        for s in sample_states:
            Ps = _act_state24(s, perm)
            for b in sample_bytes:
                intertwine_checks += 1
                left = _act_state24(byte_transition(s, b), perm)
                right = byte_transition(Ps, _sigma_byte(b, perm))
                if left != right:
                    intertwine_ok = False
                    break
            if not intertwine_ok:
                break
        if not intertwine_ok:
            break

    # Aut(Q8) via images of i,j
    idx = {n: i for i, n in enumerate(G)}
    i_ix, j_ix, k_ix = idx["i"], idx["j"], idx["k"]
    one_ix, m1_ix = idx["1"], idx["-1"]

    def _aut_from_ij(ip: int, jp: int) -> tuple[int, ...] | None:
        mp = {one_ix: one_ix, m1_ix: m1_ix, i_ix: ip, j_ix: jp}
        mp[int(table[i_ix, i_ix])] = int(table[ip, ip])
        mp[int(table[j_ix, j_ix])] = int(table[jp, jp])
        kp = int(table[ip, jp])
        mp[k_ix] = kp
        mp[int(table[m1_ix, i_ix])] = int(table[m1_ix, ip])
        mp[int(table[m1_ix, j_ix])] = int(table[m1_ix, jp])
        mp[int(table[m1_ix, k_ix])] = int(table[m1_ix, kp])
        if len(mp) != Q8_ORDER or len(set(mp.values())) != Q8_ORDER:
            return None
        for a in range(Q8_ORDER):
            for b in range(Q8_ORDER):
                if mp[int(table[a, b])] != int(table[mp[a], mp[b]]):
                    return None
        return tuple(mp[a] for a in range(Q8_ORDER))

    auts = sorted({
        phi for ip in range(Q8_ORDER) for jp in range(Q8_ORDER)
        if (phi := _aut_from_ij(ip, jp)) is not None
    })
    aut_sample = [auts[0]]
    for phi in auts:
        if phi != auts[0] and len(aut_sample) < 4:
            aut_sample.append(phi)

    unit_errs, comm_norms, vac_errs = [], [], []
    for phi in aut_sample:
        Pi = np.zeros((dim_cfg, dim_cfg))
        for f in range(dim_cfg):
            h, v = f % Q8_ORDER, (f // Q8_ORDER) % Q8_ORDER
            Pi[phi[h] + Q8_ORDER * phi[v], f] = 1.0
        U = Q.T @ (Pi @ Q)
        unit_errs.append(float(np.linalg.norm(U.T @ U - np.eye(dim_gi))))
        comm_norms.append(float(np.linalg.norm(U @ Hred - Hred @ U, ord="fro")))
        vac_errs.append(float(np.linalg.norm(U @ Omega - Omega)))

    aut_unit = max(unit_errs) < 1e-9
    aut_comm = max(comm_norms) < 1e-9
    aut_vac = max(vac_errs) < 1e-9
    aut_ok = bool(aut_unit and aut_comm and aut_vac and len(auts) == 24)
    carrier_ok = bool(dim_match and byte_perm_ok and intertwine_ok)

    print(f"  dim_GI / dim(U) / n_om       : {dim_gi} {dim_gi} {n_om}")
    print(f"  dim_match                    : {dim_match}")
    print(f"  byte_perm_bijective (S6 adj) : {byte_perm_ok}")
    print(f"  intertwine P U_b = U_s(b) P  : {intertwine_ok} (n={intertwine_checks})")
    print(f"  carrier_ok                   : {carrier_ok}")
    print(f"  Aut |Aut| / sample            : {len(auts)} {len(aut_sample)}")
    print(f"  Aut ||U^T U-I|| / ||[U,H]|| / ||UO-O|| : "
          f"{max(unit_errs):.6e} {max(comm_norms):.6e} {max(vac_errs):.6e}")
    print(f"  Aut unitary / [U,H]=0 / UΩ=Ω : {aut_unit} {aut_comm} {aut_vac}")
    print(f"  Aut order==24                 : {len(auts) == 24}")
    gate("Gamma lift: dim_GI match", dim_match)
    gate("Gamma lift: byte_perm bijective", byte_perm_ok)
    gate("Gamma lift: byte intertwining", intertwine_ok)
    gate("Gamma lift: Aut(Q8) |Aut|=24 H-sym", aut_ok)
    return {
        "dim_GI": dim_gi,
        "n_om": n_om,
        "dim_match": dim_match,
        "byte_perm_bijective": byte_perm_ok,
        "intertwine_ok": intertwine_ok,
        "intertwine_checks": intertwine_checks,
        "carrier_ok": carrier_ok,
        "aut_Q8_order": len(auts),
        "aut_sample": len(aut_sample),
        "aut_max_unit_err": max(unit_errs),
        "aut_max_comm_norm": max(comm_norms),
        "aut_max_vac_err": max(vac_errs),
        "aut_unitary": aut_unit,
        "aut_commutes": aut_comm,
        "aut_fixes_vac": aut_vac,
        "aut_ok": aut_ok,
        "isotropy_bridge_aut_q8_S4": aut_ok,
        "pass": bool(carrier_ok and aut_ok),
    }



if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Formalism Clay checklist + H7 aggregate")
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    out = run_formalism(fast=args.fast)
    raise SystemExit(0 if out.get("pass") else 1)
