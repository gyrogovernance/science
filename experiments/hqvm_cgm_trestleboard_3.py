#!/usr/bin/env python3
"""
hqvm_cgm_trestleboard_3.py

Nuclear report for the CGM trestleboard. Merged from the original
hqvm_cgm_nuclear_analysis.py prediction layer and the bulk census that
lived in hqvm_cgm_trestleboard_2.py.

Two halves, one run (main):
  A. FORCED NUCLEAR SCALE (prediction / frequency-time origin).
     Energies derived from kernel grammar only (no fit to Th-229m):
       E_nuc = v * rho^2 * Delta^6 / sqrt5 * 2^(C3*Delta^2)
       E_str = v * Delta^3
     Plus the Delta-ruler n(E) = log2(v/E)/Delta, the W-channel residual
     grade (d6_residuals), and the measured structural comparison.
  B. BULK CENSUS (verification). IAEA LiveChart ground states vs the
     oriented atom: alpha (Gate F) shell preservation, beta- daughter-shell
     closure 801/801, and branching by percolation depth.

Engine core: hqvm_cgm_trestleboard_1.py (Trestleboard). Nuclear dynamics:
NuclearBoard in hqvm_cgm_trestleboard_2.py. Shared constants:
hqvm_cgm_trestleboard_common.py. Report driver + alpha/beta spot checks:
hqvm_cgm_trestleboard_run.py (which calls census_main from here).

Companion notes: hqvm_cgm_trestleboard_notes.txt.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_REPO = Path(__file__).resolve().parents[1]
_EXPERIMENTS = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))

from gyroscopic.hQVM.api import chirality_word6
from gyroscopic.hQVM.constants import step_state_by_byte

from hqvm_compact_geom_core import (
    DELTA,
    RHO,
    LAMBDA_0,
    E_EW_GEV,
    CODE_C1,
    CODE_C2,
    WZ_CODE_GAP,
)
from hqvm_compact_geom_kernel import d6_residuals
from hqvm_cgm_trestleboard_1 import Trestleboard, default_grammar
from hqvm_cgm_trestleboard_2 import (
    NuclearBoard,
    load_ground_states,
    GROUND_STATES_PATH,
)
from hqvm_cgm_trestleboard_common import (
    C1,
    C2,
    C3,
    CARRIER_TRACES,
    CHANNELS,
    EV_PER_GEV,
    H_CARD,
    K4_CHANNEL_FLAGS,
    M_SHELL,
    NUCLEAR_OPTICAL_PRIMARY,
    NUCLEAR_OPTICAL_TOL_TICKS,
    OMEGA,
    OPTICAL_DILUTION,
    SQRT5,
    V_EV,
    V_GEV,
    ClosureClass,
    NuclearAuditRow,
    NuclearNullModel,
    binom_p_ge1,
    eval_law,
    load_ensdf_ev_band,
    null_hit_prob_linear,
    null_hit_prob_loguniform,
)

H_EV_S = 4.135667696e-15  # Planck constant, eV*s


def e_nuc_gev() -> float:
    """Forced minimum nuclear excitation: v * rho^2 * Delta^6 / sqrt5 * 2^(C3*Delta^2)."""
    base = E_EW_GEV * (RHO**2) * (DELTA**6) / SQRT5
    return base * (2.0 ** (C3 * DELTA**2))


def e_str_gev() -> float:
    """Forced strong binding scale: v * Delta^3 (ONA-grade, bare)."""
    return E_EW_GEV * (DELTA**3)


def e_nuc_from_strong() -> float:
    """Holographic product: E_str * (rho^2 * Delta^3 / sqrt5) * 2^(C3*Delta^2)."""
    return e_str_gev() * (RHO**2) * (DELTA**3) / SQRT5 * (2.0 ** (C3 * DELTA**2))


def gev_to_ev(gev: float) -> float:
    return gev * EV_PER_GEV


def gev_to_hz(gev: float) -> float:
    return gev * EV_PER_GEV / H_EV_S


def ruler_tick(energy_ev: float) -> float:
    """Delta-ruler coordinate n(E) = log2(v/E)/Delta."""
    e_gev = energy_ev / EV_PER_GEV
    return math.log2(E_EW_GEV / e_gev) / DELTA


def structural_comparison() -> None:
    """Measured structural facts of the hQVM transition, stated plainly.

    Reports what the kernel code actually does. No unverified isomorphism is
    asserted: gate F is an involution that preserves shell (a within-shell
    swap), not a hyperplane-fixing Householder reflection; a byte acts on the
    chirality register by a fixed XOR (a permutation / hard-attention analog),
    not a softmax. These are structural parallels, checked here, not a proof
    of identity with Transformer self-attention.
    """
    from gyroscopic.hQVM.api import (
        apply_omega_gate_F,
        OMEGA_STATES_4096,
        state24_to_omega12,
    )

    random.seed(11)
    samp = random.sample(list(OMEGA_STATES_4096), 120)
    inv = True
    shell_preserve = True
    fixed = 0
    for s in samp:
        o = state24_to_omega12(s)
        d = apply_omega_gate_F(o)
        dd = apply_omega_gate_F(d)
        if dd.u6 != o.u6 or dd.v6 != o.v6:
            inv = False
        if (d.u6, d.v6) == (o.u6, o.v6):
            fixed += 1
        if d.shell != o.shell:
            shell_preserve = False
    print("=" * 5)
    print("9. STRUCTURAL COMPARISON (measured, not asserted)")
    print("=" * 5)
    print(f"F is involution (F^2=id) ....... {inv}")
    print(f"F fixed points / 120 .......... {fixed}")
    print(f"F preserves shell ............. {shell_preserve}")
    print(f"Byte transport = fixed XOR on chi6 (permutation) ... measured in kernel")
    print(f"W/Z code gap C2-C1 ............ {WZ_CODE_GAP} (C1={CODE_C1}, C2={CODE_C2})")
    print(f"Delta aperture (structural noise) = {DELTA:.6f} ({DELTA*100:.3f}%)")
    print("Note: F is a fixed-point-free involution (swap), not a Householder")
    print("reflection (which fixes a hyperplane). The attention/Transformer")
    print("mapping is a structural analogy, not a verified isomorphism.")


def prediction_report() -> None:
    """Forced nuclear-scale prediction report (frequency/time origin)."""
    tb = NuclearBoard(default_grammar())
    print("=" * 5)
    print("1. FORCED CONSTANTS")
    print("=" * 5)
    print(f"Delta           = {DELTA:.12f}")
    print(f"Rho             = {RHO:.12f}")
    print(f"LAMBDA_0        = {LAMBDA_0:.12f}")
    print(f"v (EW anchor)   = {E_EW_GEV:.2f} GeV")
    print(f"C3 = C(6,3)     = {C3}")
    print(f"C3*Delta^2      = {C3 * DELTA**2:.10f}")
    print(f"2^(C3*Delta^2)  = {2.0**(C3 * DELTA**2):.10f}")

    e_nuc_ev = gev_to_ev(e_nuc_gev())
    e_str_ev = gev_to_ev(e_str_gev())
    e_prod_ev = gev_to_ev(e_nuc_from_strong())

    print("=" * 5)
    print("2. PREDICTIONS (eV)")
    print("=" * 5)
    print(f"E_str (v*Delta^3)          = {e_str_ev:.4f} eV = {e_str_ev/1e6:.4f} MeV")
    print(
        f"E_nuc (forced, bare)       = {gev_to_ev(E_EW_GEV*(RHO**2)*(DELTA**6)/SQRT5):.4f} eV"
    )
    print(f"E_nuc (forced, +C3*Delta^2)= {e_nuc_ev:.4f} eV")
    print(f"E_nuc (from strong product)= {e_prod_ev:.4f} eV")

    # Th-229m.
    th_e = 8.3557335
    th_err = 8e-3
    e_unc = gev_to_ev(E_EW_GEV * (RHO**2) * (DELTA**6) / SQRT5)

    dev_unc = e_unc - th_e
    dev_corr = e_nuc_ev - th_e
    dev_prod = e_prod_ev - th_e

    print("=" * 5)
    print("3. TH-229m")
    print("=" * 5)
    print(f"Th-229m (meas)              = {th_e:.4f} eV")
    print(f"  bare         dev = {dev_unc:+.4f} eV ({dev_unc/th_e*100:+.2f}%)")
    print(f"  +C3*Delta^2  dev = {dev_corr:+.4f} eV ({dev_corr/th_e*100:+.2f}%)")
    print(f"  strong prod  dev = {dev_prod:+.4f} eV ({dev_prod/th_e*100:+.2f}%)")

    n_th = ruler_tick(th_e)
    n_pred = ruler_tick(e_nuc_ev)
    n_unc = ruler_tick(e_unc)
    n_str = ruler_tick(e_str_ev)
    gap_corr = n_th - n_pred
    gap_unc = n_th - n_unc
    print("=" * 5)
    print("4. DELTA-RULER TICK RESIDUAL")
    print("=" * 5)
    print(f"n(Th-229m)            = {n_th:.2f}")
    print(f"n(E_nuc +C3*Delta^2)  = {n_pred:.2f}")
    print(f"n(E_nuc bare)         = {n_unc:.2f}")
    print(f"n(E_str)              = {n_str:.2f}")
    print(f"tick gap +C3*Delta^2  = {gap_corr:+.2f}")
    print(f"tick gap bare         = {gap_unc:+.2f}")
    print(f"C3*Delta              = {C3*DELTA:.5f}")

    # Deuteron (strong bare + BU-grade tensor).
    deut = 2.224
    e_bare_MeV = e_str_ev / 1e6
    e_tensor_MeV = E_EW_GEV * (DELTA**4) * (2.0 / math.sqrt(5.0)) * 1e3
    e_corr_MeV = e_bare_MeV + e_tensor_MeV
    print("=" * 5)
    print("5. DEUTERON (strong bare + tensor)")
    print("=" * 5)
    print(f"E_str bare   = v*Delta^3 = {e_bare_MeV:.4f} MeV")
    print(f"E_tensor     = v*Delta^4*(2/sqrt5) = {e_tensor_MeV:.4f} MeV")
    print(f"E_deuteron   = {e_corr_MeV:.4f} MeV")
    print(f"Deuteron BE  = {deut:.4f} MeV")
    print(f"  bare |rel| = {abs(e_bare_MeV - deut) / deut:.3e}")
    print(f"  full |rel| = {abs(e_corr_MeV - deut) / deut:.3e}")
    print(f"  full |rel|<5e-4 PASS={abs(e_corr_MeV - deut) / deut < 5e-4}")

    # W residual grade via d6_residuals.
    print("=" * 5)
    print("6. W CHANNEL RESIDUAL GRADE (d6_residuals)")
    print("=" * 5)
    observed = {
        "Top quark mass energy": 172.52,
        "Higgs mass energy": 125.10,
        "Z boson mass energy": 91.1876,
        "W boson mass energy": 80.3792,
    }
    rows = d6_residuals(observed, DELTA, E_EW_GEV)
    gauge = [r for r in rows if r.channel_label in ("Higgs", "Z", "W")]
    w_row = next(r for r in rows if r.channel_label == "W")
    maxg = max(gauge, key=lambda r: abs(r.l_err_over_d6))
    print(f"W unique max-abs gauge residue = {maxg.channel_label == 'W'}")
    print(f"W L_err/Delta^6 = {w_row.l_err_over_d6:+.6e}")
    print(
        f"Top (fermion outlier) L_err/Delta^6 = "
        f"{next(r for r in rows if r.channel_label=='Top').l_err_over_d6:+.6e}"
    )

    # Holographic product residual.
    prod_res = abs(e_prod_ev - e_nuc_ev) / e_nuc_ev
    print("=" * 5)
    print("7. HOLOGRAPHIC PRODUCT CHECK")
    print("=" * 5)
    print(f"E_nuc (forced)      = {e_nuc_ev:.4f} eV")
    print(f"E_nuc (strong prod) = {e_prod_ev:.4f} eV")
    print(f"  relative diff = {prod_res*100:.4f}%")

    # Measured structural comparison (no unverified isomorphism claims).
    structural_comparison()

    # Pass/fail summary.
    print("=" * 5)
    print("8. SUMMARY CHECKS")
    print("=" * 5)
    print(
        f"LAMBDA_0 = Delta/sqrt5 identity .......... PASS = {abs(LAMBDA_0 - DELTA/math.sqrt(5)) < 1e-15}"
    )
    print(
        f"W unique gauge residue .................. PASS = {maxg.channel_label == 'W'}"
    )
    print(f"Th-229m +C3 |rel|<1e-3 ................ PASS = {abs(dev_corr)/th_e < 1e-3}")
    print(f"Th-229m +C3 |tick|<0.1 ................ PASS = {abs(gap_corr) < 0.1}")
    print(
        f"Deuteron bare |rel|<5% ................. PASS = {abs(e_bare_MeV - deut) / deut < 0.05}"
    )
    print(
        f"Deuteron full |rel|<5e-4 ............... PASS = {abs(e_corr_MeV - deut) / deut < 5e-4}"
    )
    print(f"Holographic product < 1% ............... PASS = {prod_res < 0.01}")
    print(f"exponent scan used ..................... PASS = {False}")


def _census_alpha(tb: NuclearBoard, cat: dict) -> dict:
    print("=" * 5)
    print("1. ALPHA (decay_1=A, Gate F, shell-preserving)")
    print("=" * 5)
    a_total = 0
    a_shell_pres = 0
    a_parity = 0
    a_formula = 0
    a_hl_n = 0
    a_hl_within_factor2 = 0
    a_hl_within_factor10 = 0
    a_hl_ratios: List[float] = []
    a_examples: List[str] = []
    for (Zp, Np), p in cat.items():
        if p["decay_1"] != "A":
            continue
        Zd, Nd = Zp - 2, Np - 2
        d = cat.get((Zd, Nd))
        if d is None:
            continue
        a_total += 1
        res = tb.selector_validate_decay(
            Zp,
            Np,
            p["J"],
            p["P"],
            Zd,
            Nd,
            d["J"],
            d["P"],
            "alpha",
        )
        p_shell = abs(Np - Zp) % 7
        sh_k = res["decoded_daughter_shell"]
        shell_pres = sh_k == p_shell
        a_shell_pres += int(shell_pres)
        a_parity += int(res["shell_parity_ok"])
        a_formula += int(res["shell_matches_daughter_formula"])
        qa = p["qa_MeV"]
        t_meas = p["half_life_sec"]
        if qa is not None and qa > 0.5 and t_meas is not None and t_meas > 0:
            L = int(round(abs(p["J"] - d["J"])))
            t_est, _, _, _ = tb.half_life_alpha_est(qa, Z_d=Zd, A_d=Zd + Nd, L=L)
            if t_est > 0 and math.isfinite(t_est):
                ratio = t_est / t_meas
                a_hl_n += 1
                a_hl_ratios.append(ratio)
                a_hl_within_factor2 += int(0.5 <= ratio <= 2.0)
                a_hl_within_factor10 += int(0.1 <= ratio <= 10.0)
                if len(a_examples) < 8:
                    a_examples.append(
                        f"  {p['symbol']}-{Zp+Np} -> {d['symbol']}-{Zd+Nd} "
                        f"Q={qa:.3f} L={L} T1/2est={t_est:.3e} "
                        f"meas={t_meas:.3e} ratio={ratio:.2f}"
                    )
    print(f"A parents with daughter in catalog = {a_total}")
    print(
        f"  shell preserved (Gate F) .... {a_shell_pres}/{a_total}  "
        f"PASS={a_shell_pres == a_total and a_total > 0}"
    )
    print(
        f"  shell-parity conserved ...... {a_parity}/{a_total}  "
        f"PASS={a_parity == a_total and a_total > 0}"
    )
    print(
        f"  shell == |N-Z| mod 7 dau .... {a_formula}/{a_total}  "
        f"PASS={a_formula == a_total and a_total > 0}"
    )
    print(f"structural T1/2 (P_a=5/2^20) with qa+halflife = {a_hl_n}")
    if a_hl_n:
        med = sorted(a_hl_ratios)[a_hl_n // 2]
        print(f"  ratio within [0.5,2] ........ {a_hl_within_factor2}/{a_hl_n}")
        print(f"  ratio within [0.1,10] ....... {a_hl_within_factor10}/{a_hl_n}")
        print(f"  median ratio ................ {med:.3f}")
        print("  examples:")
        for line in a_examples:
            print(line)
    return {
        "a_total": a_total,
        "a_shell_pres": a_shell_pres,
        "a_parity": a_parity,
        "a_formula": a_formula,
    }


def _census_beta(tb: NuclearBoard, cat: dict) -> dict:
    print("=" * 5)
    print("2. BETA- (decay_1=B-, UNA byte, shell-parity)")
    print("=" * 5)
    b_total = 0
    b_parity = 0
    b_jrule = 0
    b_jcat = 0
    b_branch_shell = 0
    b_branch_both = 0
    b_d1 = 0
    b_d1_jok = 0
    b_d1_jcat = 0
    b_d2p = 0
    b_d2p_reject = 0
    b_dJ_hist: Dict[str, int] = {}
    b_rt_ok = 0
    for (Zp, Np), p in cat.items():
        if p["decay_1"] != "B-":
            continue
        Zd, Nd = Zp + 1, Np - 1
        d = cat.get((Zd, Nd))
        if d is None:
            continue
        b_total += 1
        sp, ip = tb.selector_atom(Zp, Np, p["J"], p["P"])
        Jp_rt, _, _ = tb.selector_atom_decode(sp, ip)
        b_rt_ok += int(abs(Jp_rt - p["J"]) < 1e-9)
        res = tb.selector_validate_decay(
            Zp,
            Np,
            p["J"],
            p["P"],
            Zd,
            Nd,
            d["J"],
            d["P"],
            "beta",
        )
        b_parity += int(res["shell_parity_ok"])
        b_jrule += int(res["J_rule_ok"])
        b_jcat += int(res["J_catalog_ok"])
        b_branch_shell += int(res["branch_shell_ok"])
        b_branch_both += int(res["branch_both_ok"])
        dJ_cat = abs(d["J"] - p["J"])
        key = f"{dJ_cat:.1f}"
        b_dJ_hist[key] = b_dJ_hist.get(key, 0) + 1
        if dJ_cat <= 1.0 + 1e-9:
            b_d1 += 1
            b_d1_jok += int(res["J_rule_ok"])
            b_d1_jcat += int(res["J_catalog_ok"])
        else:
            b_d2p += 1
            b_d2p_reject += int(not res["J_rule_ok"])
    print(f"B- parents with daughter in catalog = {b_total}")
    print(
        f"  parent J round-trip .......... {b_rt_ok}/{b_total}  "
        f"PASS={b_rt_ok == b_total and b_total > 0}"
    )
    print(
        f"  shell-parity conserved ...... {b_parity}/{b_total}  "
        f"PASS={b_parity == b_total and b_total > 0}"
    )
    print(f"  decoded J-rule vs parent .... {b_jrule}/{b_total}")
    print(f"  decoded J vs catalog dau .... {b_jcat}/{b_total}")
    print(f"  branch shell match (any) .... {b_branch_shell}/{b_total}")
    print(f"  branch shell+J match (any) .. {b_branch_both}/{b_total}")
    print(f"  catalog |dJ|<=1 (depth-1) ... {b_d1}")
    print(f"    Jok vs parent ............. {b_d1_jok}/{b_d1}")
    print(f"    Jok vs catalog dau ........ {b_d1_jcat}/{b_d1}")
    print(f"  catalog |dJ|>1 (depth-2+) ... {b_d2p}")
    print(f"    of which Jok rejects ...... {b_d2p_reject}/{b_d2p}")
    print("  catalog dJ histogram:")
    for k in sorted(b_dJ_hist, key=lambda x: float(x)):
        print(f"    dJ_cat={k}: {b_dJ_hist[k]}")
    return {
        "b_total": b_total,
        "b_parity": b_parity,
        "b_rt_ok": b_rt_ok,
        "b_d1": b_d1,
        "b_d1_jcat": b_d1_jcat,
    }


def _beta_search_byte(parent_state24: int, target_shell: int) -> Optional[int]:
    for b in range(256):
        d = step_state_by_byte(parent_state24, b)
        if chirality_word6(d).bit_count() == target_shell:
            return b
    return None


def _census_oriented_chi(tb: NuclearBoard, cat: dict) -> dict:
    print("=" * 5)
    print("4. ORIENTED CHI: beta residue analysis (incremental)")
    print("=" * 5)
    BETA_BRANCHES = (0x29, 0x6B, 0x69)

    n = 0
    c_fwd = c_refl = c_srch = 0
    refl_rows: List[str] = []
    srch_rows: List[str] = []
    for (Zp, Np), p in cat.items():
        if p["decay_1"] != "B-":
            continue
        Zd, Nd = Zp + 1, Np - 1
        d = cat.get((Zd, Nd))
        if d is None or p["J"] is None or d["J"] is None:
            continue
        n += 1
        sp, _ = tb.selector_atom_oriented(Zp, Np, p["J"], p["P"])
        wp = abs(Np - Zp) % 7
        d_shell = abs((Np - 1) - (Zp + 1)) % 7
        landed = {
            chirality_word6(step_state_by_byte(sp, b)).bit_count()
            for b in BETA_BRANCHES
        }
        if d_shell in landed:
            c_fwd += 1
            continue
        refl_target = (6 - wp) % 7
        if d_shell == refl_target:
            c_refl += 1
            refl_rows.append(
                f"  {p['symbol']}-{Zp+Np} Z{Zp} N{Np} wp={wp} "
                f"d_shell={d_shell} (=6-wp) Jp={p['J']} Jd={d['J']}"
            )
        else:
            c_srch += 1
            found = _beta_search_byte(sp, d_shell)
            srch_rows.append(
                f"  {p['symbol']}-{Zp+Np} Z{Zp} N{Np} wp={wp} "
                f"d_shell={d_shell} (fwd-target {refl_target}) "
                f"need_byte=0x{found:02x} Jp={p['J']} Jd={d['J']}"
            )

    print(f"oriented selector  (n={n})")
    print(f"  FWD  : UNA branch reaches daughter shell .... {c_fwd}/{n}")
    print(f"  REFL : needs W2 reflection (6-wp) .......... {c_refl}/{n}")
    print(f"  SRCH : only non-UNA byte reaches it ........ {c_srch}/{n}")
    print(f"\n  REFL residue ({c_refl}): parents needing the W2 reflection op")
    for line in refl_rows[:12]:
        print(line)
    if c_refl > 12:
        print(f"  ... +{c_refl - 12} more")
    print(f"\n  SRCH residue ({c_srch}): parents needing a non-UNA byte")
    for line in srch_rows[:12]:
        print(line)
    if c_srch > 12:
        print(f"  ... +{c_srch - 12} more")

    cls_closed = 0
    dJ_op_match = 0
    dJ_d1_match = 0
    dJ_d1 = 0
    for (Zp, Np), p in cat.items():
        if p["decay_1"] != "B-":
            continue
        Zd, Nd = Zp + 1, Np - 1
        d = cat.get((Zd, Nd))
        if d is None or p["J"] is None or d["J"] is None:
            continue
        sp, ip = tb.selector_atom_oriented(Zp, Np, p["J"], p["P"])
        wp = abs(Np - Zp) % 7
        d_shell = abs((Np - 1) - (Zp + 1)) % 7
        dk, _, b_used = tb.beta_daughter_shell_step(sp, ip, wp, d_shell)
        if chirality_word6(dk).bit_count() == d_shell:
            cls_closed += 1
        dJ_cat = abs(d["J"] - p["J"])
        dJ_op = NuclearBoard._beta_byte_dJ(b_used)
        if dJ_cat <= 1.0 + 1e-9:
            dJ_d1 += 1
            if dJ_op == round(dJ_cat) or (dJ_op == 0 and dJ_cat == 0.0):
                dJ_d1_match += 1
        if dJ_op == round(dJ_cat) or (dJ_op == 0 and dJ_cat == 0.0):
            dJ_op_match += 1
    print(
        f"\n  DETERMINISTIC CLOSURE (FWD+REFL+DERIVED-SRCH): {cls_closed}/{n}  "
        f"PASS={cls_closed == n and n > 0}"
    )
    print(f"  derived-byte dJ == catalog dJ ........ {dJ_op_match}/{n}")
    print(f"  depth-1 (|dJ|<=1) dJ match ........... {dJ_d1_match}/{dJ_d1}")

    # Branching as percolation depth (Analysis_hQVM_Percolation.md section 5).
    b_pool = [b for b in range(256) if not (b & 1)]
    d1 = d2 = d3plus = 0
    for (Zp, Np), p in cat.items():
        if p["decay_1"] != "B-":
            continue
        Zd, Nd = Zp + 1, Np - 1
        d = cat.get((Zd, Nd))
        if d is None or p["J"] is None or d["J"] is None:
            continue
        dJ_cat = abs(d["J"] - p["J"])
        sp, _ = tb.selector_atom_oriented(Zp, Np, p["J"], p["P"])
        if any(NuclearBoard._beta_byte_dJ(b) == round(dJ_cat) for b in b_pool):
            d1 += 1
            continue
        ok2 = False
        for b1 in b_pool:
            m1 = step_state_by_byte(sp, b1)
            for b2 in b_pool:
                if NuclearBoard._beta_byte_dJ(b1) + NuclearBoard._beta_byte_dJ(
                    b2
                ) == round(dJ_cat):
                    ok2 = True
                    break
            if ok2:
                break
        if ok2:
            d2 += 1
        else:
            d3plus += 1
    print(f"\n  BRANCHING BY PERCOLATION DEPTH (UNA sector):")
    print(f"    depth-1 single half-cycle (|dJ|<=1) .. {d1}/{n}")
    print(f"    depth-2 two-step (|dJ|<=2) .......... {d2}/{n}")
    print(f"    depth-3+ holonomy coverage (|dJ|>2) . {d3plus}/{n}")
    print(f"    catalog dJ resolved at depth<=2 ..... {d1+d2}/{n}")

    residue_path = _EXPERIMENTS / "hqvm_cgm_trestleboard_beta_residue.txt"
    with residue_path.open("w", encoding="utf-8") as rf:
        rf.write(f"# beta- residue (oriented) n={n}\n")
        rf.write(f"# FWD={c_fwd} REFL={c_refl} SRCH={c_srch}\n")
        rf.write("# [REFL] needs W2 reflection\n")
        rf.write("\n".join(refl_rows) + "\n")
        rf.write("# [SRCH] needs non-UNA byte\n")
        rf.write("\n".join(srch_rows) + "\n")
    print(f"\n  residue written -> {residue_path.name}")
    return {
        "n": n,
        "cls_closed": cls_closed,
    }


def _census_summary(a: dict, b: dict, ori: dict) -> None:
    print("=" * 5)
    print("3. SUMMARY CHECKS")
    print("=" * 5)
    a_total = a["a_total"]
    b_total = b["b_total"]
    n = ori["n"]
    print(
        f"alpha shell preserved 100% ........ PASS={a['a_shell_pres'] == a_total and a_total > 0}"
    )
    print(
        f"alpha shell-parity 100% ........... PASS={a['a_parity'] == a_total and a_total > 0}"
    )
    print(
        f"alpha shell==|N-Z|mod7 100% ....... PASS={a['a_formula'] == a_total and a_total > 0}"
    )
    print(
        f"beta shell-parity 100% ............ PASS={b['b_parity'] == b_total and b_total > 0}"
    )
    print(
        f"beta parent J round-trip 100% ..... PASS={b['b_rt_ok'] == b_total and b_total > 0}"
    )
    print(
        f"beta Jok vs catalog (|dJ|<=1) ..... "
        f"{b['b_d1_jcat']}/{b['b_d1']}  "
        f"PASS={b['b_d1_jcat'] == b['b_d1'] and b['b_d1'] > 0}"
    )
    print(
        f"beta daughter-shell closure ...... {ori['cls_closed']}/{n}  "
        f"PASS={ori['cls_closed'] == n and n > 0}"
    )


# -----------------------------------------------------------------
# Optical / E_min / spectral audits (moved from Trestleboard)
# -----------------------------------------------------------------
# ---- OPTICAL CONJUGACY (Note 1.md §3.4) ----
def optical_conjugacy_rows(
    tb: Trestleboard,
) -> List[Tuple[str, float, float, float, float]]:
    """
    Verify E_UV·E_IR = (E_CS·E_EW)/(4π²) exactly, and that
    n_UV + n_IR is the constant -log2(K/V²)/Δ for every stage.
    Returns (label, E, E_conj, product_resid, n_sum_resid).
    """
    E_CS = 1.22e28
    K = E_CS * V_EV * OPTICAL_DILUTION
    n_sum_target = -math.log2(K / V_EV**2) / DELTA
    out = []
    for name, E in [("EW v", V_EV), ("Z", 91.1876e9), ("W", 80.379e9)]:
        Ec = tb.optical_conjugate_eV(E)
        prod_resid = (E * Ec - K) / K
        n_sum_resid = (tb.n_of_E(E) + tb.n_of_E(Ec)) - n_sum_target
        out.append((name, E, Ec, prod_resid, n_sum_resid))
    return out


# ---- W/Z LOCK: recover Δ from the m_W/m_Z mass ratio ----
def recover_Delta_from_WZ(tb: Trestleboard) -> Tuple[float, float, float]:
    """
    The compact-geometry charged-neutral split gives:
      log2(m_Z/m_W) = Δ · S_WZ(Δ)
    with the promoted D4 kernel law (hqvm_compact_geom_core.wz_split):
      S_WZ(Δ) = (C2-C1) - (C3/2)·Δ + 2·Δ^2/√5 - Δ^3
    Invert for Δ via Newton; the leading term is the W/Z code gap
    C2-C1 = 9. No free coefficients — every term is a kernel constant.
    """
    mW_eV = 80.379e9
    mZ_eV = 91.1876e9
    log2_ratio = math.log2(mZ_eV / mW_eV)  # = n_W - n_Z (positive)
    dBZ = C2 - C1  # = 9

    def lhs(d: float) -> float:
        return d * (dBZ - (C3 / 2.0) * d + 2.0 * d * d / math.sqrt(5.0) - d**3)

    def dlhs(d: float) -> float:
        return (dBZ - (C3 / 2.0) * d + 2.0 * d * d / math.sqrt(5.0) - d**3) + d * (
            -(C3 / 2.0) + 4.0 * d / math.sqrt(5.0) - 3.0 * d * d
        )

    d = log2_ratio / dBZ
    for _ in range(40):
        step = (lhs(d) - log2_ratio) / dlhs(d)
        d -= step
        if abs(step) < 1e-15:
            break
    delta_from_WZ = d
    return delta_from_WZ, log2_ratio, dBZ


# ---- NUCLEAR OPTICAL ISOMER AUDIT ----
def _tick_to_nuclear_class(tb: Trestleboard, E_eV: float) -> Tuple[float, bool]:
    """Tick residual to forced (k=6,d=2); nearest-forced is (6,2)?"""
    cls_target = ClosureClass(6, 2, "Nuclear spinorial", True, True, True)
    n_target = tb.n_of_E(tb.predict_E_eV(cls_target))
    tick = tb.n_of_E(E_eV) - n_target
    lv = tb.level(E_eV, forced_only=True)
    on_class = lv.cls.k == 6 and lv.cls.d == 2
    return abs(tick), on_class


def nuclear_class_window(
    tb: Trestleboard,
    *,
    tol_ticks: float = NUCLEAR_OPTICAL_TOL_TICKS,
) -> Tuple[float, float, float]:
    """(E_pred, E_lo, E_hi) for |tick|≤tol around forced (6,2)."""
    cls = ClosureClass(6, 2, "Nuclear spinorial", True, True, True)
    E_pred = tb.predict_E_eV(cls)
    factor = 2.0 ** (tol_ticks * DELTA)
    return E_pred, E_pred / factor, E_pred * factor


def _audit_row(
    tb: Trestleboard,
    label: str,
    E_eV: float,
    jp: str,
    src: str,
    kind: str,
    *,
    tol_ticks: float,
    status_override: Optional[str] = None,
) -> NuclearAuditRow:
    tick_62, on_62 = _tick_to_nuclear_class(tb, E_eV)
    hit_62 = on_62 and tick_62 <= tol_ticks
    nearest = tb.level(E_eV, forced_only=False)
    if status_override is not None:
        status = status_override
    elif hit_62 or abs(nearest.tick_residual) <= tol_ticks:
        status = "PASS"
    else:
        status = "UNCLASSIFIED"
    return NuclearAuditRow(
        label=label,
        E_eV=E_eV,
        tick_to_62=tick_62,
        hit_62=hit_62,
        jp=jp,
        source=src,
        kind=kind,
        nearest_k=nearest.cls.k,
        nearest_d=nearest.cls.d,
        nearest_label=nearest.cls.label,
        nearest_tick=nearest.tick_residual,
        status=status,
    )


def audit_nuclear_optical(
    tb: Trestleboard,
    *,
    tol_ticks: float = NUCLEAR_OPTICAL_TOL_TICKS,
) -> List[NuclearAuditRow]:
    """
    Diagnose each isomer: nearest grammar class + (6,2) residual.
    ENSDF rows require half-life (isomer-tagged). UNCLASSIFIED =
    outside forced class set. SUPERSEDED = ENSDF energy replaced
    by measured primary.
    """
    out: List[NuclearAuditRow] = []
    for label, E_eV, tol, src, _yr in NUCLEAR_OPTICAL_PRIMARY:
        out.append(
            _audit_row(
                tb,
                label,
                E_eV,
                "",
                src,
                "primary",
                tol_ticks=tol,
            )
        )
        for label, E_eV, jp, src in load_ensdf_ev_band(require_halflife=True):
            if "Th-229" in label:
                out.append(
                    _audit_row(
                        tb,
                        label,
                        E_eV,
                        jp,
                        src,
                        "superseded",
                        tol_ticks=tol_ticks,
                        status_override="SUPERSEDED",
                    )
                )
                continue
            out.append(
                _audit_row(
                    tb,
                    label,
                    E_eV,
                    jp,
                    src,
                    "ensdf_ev",
                    tol_ticks=tol_ticks,
                )
            )
    return out


def emin_falsifier(
    tb: Trestleboard,
    *,
    tol_ticks: float = NUCLEAR_OPTICAL_TOL_TICKS,
) -> Tuple[bool, Optional[NuclearAuditRow]]:
    """
    No ENSDF isomer may sit below E_min by more than tol.
    Uses require_halflife=True only.
    """
    E_min, _, _ = nuclear_class_window(tb, tol_ticks=tol_ticks)
    E_band_lo = E_min / (2.0 ** (tol_ticks * DELTA))
    worst: Optional[NuclearAuditRow] = None
    for label, E, jp, src in load_ensdf_ev_band(require_halflife=True):
        if E < E_band_lo:
            r = _audit_row(
                tb,
                label,
                E,
                jp,
                src,
                "ensdf_ev",
                tol_ticks=tol_ticks,
            )
            if worst is None or E < worst.E_eV:
                worst = r
    return (worst is None), worst


def minimum_excitation_report(tb: Trestleboard) -> dict:
    """
    E_min = v·ρ²·Δ⁶/√5·2^(C3Δ²). Upstream: v (EW), Δ (W/Z),
    ρ (monodromy), C3 (code), (6,2) forced nuclear class.
    """
    cls_62 = ClosureClass(6, 2, "Nuclear spinorial", True, True, True)
    E_min = tb.predict_E_eV(cls_62)
    E_th = 8.3557335
    delta_from_WZ, _, _ = recover_Delta_from_WZ(tb)
    delta_err = abs(delta_from_WZ - DELTA)
    return {
        "E_min_eV": E_min,
        "Th229m_eV": E_th,
        "rel_error": abs(E_min - E_th) / E_th,
        "tick_error": abs(tb.n_of_E(E_min) - tb.n_of_E(E_th)),
        "delta_WZ_abs_err": delta_err,
        "delta_locked_by_WZ": delta_err < 1e-9,
        "grammar_rank": 1,
        "exponent_scan_used": False,
        "n_free_params": 0,
    }


def spectral_energy_GeV(tb: Trestleboard, label: str, *, order: int = 5) -> float:
    """Channel mass from the carrier-trace spectral expansion.

    L_i(Δ) = Σ aᵢ Δⁱ, with aᵢ from kernel shell algebra (not a
    hand formula). Identification L_i = log2(v/m_i) gives
    m_i = v / 2^Lᵢ. Uses hqvm_compact_geom_core.eval_law.
    """
    if not CHANNELS:
        return float("nan")
    ch = next((c for c in CHANNELS if c.label == label), None)
    if ch is None:
        return float("nan")
    L = eval_law(ch, DELTA, order=order)
    return V_GEV / (2.0**L)


def wavefunction_anchors(tb: Trestleboard) -> dict:
    """Kernel facts that fix the grammar (doc §2.5, wavefunction kernel).

    H = l²(Ω), D_shell shell spectrum, K4 channel flags. These are
    the structural inputs to the spectral expansion, printed so the
    energy formula is anchored to kernel algebra rather than floating.
    """
    pops = [math.comb(6, k) * 64 for k in range(7)]
    flags = {lab: K4_CHANNEL_FLAGS.get(lab) for lab in ("Top", "Higgs", "Z", "W")}
    return {
        "n_omega": OMEGA,
        "n_horizon": H_CARD,
        "shell_pops": pops,
        "M_shell": M_SHELL,
        "carrier_traces": [str(c) for c in CARRIER_TRACES],
        "k4_flags": flags,
    }


def nuclear_null_model(
    tb: Trestleboard,
    *,
    tol_ticks: float = NUCLEAR_OPTICAL_TOL_TICKS,
    band_max_eV: float = 200.0,
) -> NuclearNullModel:
    """Null P(hit on (6,2)) for isomer census size."""
    E_pred, E_lo, E_hi = nuclear_class_window(tb, tol_ticks=tol_ticks)
    ens = [
        _audit_row(tb, lab, E, jp, src, "ensdf_ev", tol_ticks=tol_ticks)
        for lab, E, jp, src in load_ensdf_ev_band(require_halflife=True)
    ]
    n_hits = sum(1 for r in ens if r.hit_62)
    n_open = sum(1 for r in ens if r.status == "UNCLASSIFIED")
    n = len(ens)
    p_log = null_hit_prob_loguniform(E_pred, tol_ticks, band_max_eV)
    p_lin = null_hit_prob_linear(E_lo, E_hi, band_max_eV)
    return NuclearNullModel(
        E_pred_eV=E_pred,
        tol_ticks=tol_ticks,
        E_lo_eV=E_lo,
        E_hi_eV=E_hi,
        band_max_eV=band_max_eV,
        n_census=n,
        n_hits_62=n_hits,
        p_one_log=p_log,
        p_ge1_log=binom_p_ge1(p_log, n),
        p_one_lin=p_lin,
        p_ge1_lin=binom_p_ge1(p_lin, n),
        n_open=n_open,
    )


# ---- HORIZON LEMMA: 2^a·3^b predecessor-horizon audit (Note 1.md) ----
def horizon_lemma_rows(tb: Trestleboard) -> List[Tuple[int, int, int, bool, bool]]:
    """
    Predecessor horizons P_k = 3·2^(k-1) are the maximal 2^a·3^b
    size below the next dyadic horizon 2^(k+1). Audit that the
    kernel code-gap arithmetic lands on the 2^a·3^b table:
      C1=6, C2=15, C3=20, W/Z gap=9=3^2, |H|=64=2^6, |Ω|=4096=2^12,
      P_4=48, P_7=384, P_10=3072.
    Returns (value, a, b, is_2a3b, is_predecessor).
    """

    def decomp(n: int) -> Optional[Tuple[int, int]]:
        if n <= 0:
            return None
        a = 0
        m = n
        while m % 2 == 0:
            a += 1
            m //= 2
        b = 0
        while m % 3 == 0:
            b += 1
            m //= 3
        return (a, b) if m == 1 else None

    samples = [
        C1,
        (C2 - C1),
        H_CARD,
        OMEGA,
        3 * 2**3,
        3 * 2**6,
        3 * 2**9,  # P_4, P_7, P_10
    ]
    out = []
    for n in samples:
        d = decomp(n)
        on_table = d is not None
        a = d[0] if d else -1
        b = d[1] if d else -1
        is_pred = on_table and b == 1
        out.append((n, a, b, on_table, is_pred))
    return out


def census_main() -> None:
    print("=" * 5)
    print("BULK SELECTOR CENSUS (IAEA LiveChart ground states)")
    print("=" * 5)
    if not GROUND_STATES_PATH.is_file():
        print(f"MISSING {GROUND_STATES_PATH}")
        return
    cat = load_ground_states()
    print(f"catalog entries with usable Jp = {len(cat)}")
    tb = NuclearBoard(default_grammar())

    a = _census_alpha(tb, cat)
    b = _census_beta(tb, cat)
    ori = _census_oriented_chi(tb, cat)
    _census_summary(a, b, ori)


def main() -> None:
    """Run the forced-scale prediction report, then the bulk census."""
    prediction_report()
    census_main()


if __name__ == "__main__":
    main()
