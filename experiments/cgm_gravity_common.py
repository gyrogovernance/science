"""
Shared CGM gravity kernel helpers and invariants.

Used by cgm_gravity_analysis_{1,2,exp}.py.

Convention: arch_shell = 6 - chi_shell, measuring distance from the complement
horizon (0 = complement horizon, 6 = equality horizon).

E_CS is the Planck-scale UV anchor (E_CS = E_Planck = 1.22e19 GeV; see
Analysis_Energy_Scales). Planck units presuppose G, so G = G_kernel / E_CS^2
must not be used to derive G. E_CS may appear only in consistency checks
(e.g. conjugacy depth 2 ln(E_CS/v)). The forward prediction is
G_pred = G_kernel * exp(-tau_G) / v^2 with kernel tau_G; tau_required from
G_meas is validation only.
"""

from __future__ import annotations

import math
import sys
from math import comb
from pathlib import Path

import numpy as np

from gyroscopic.aQPU.constants import (
    GENE_MAC_REST,
    GENE_MIC_S,
    CHIRALITY_MASK_6,
    step_state_by_byte,
    byte_to_intron,
    intron_family,
    intron_micro_ref,
    is_on_horizon,
    is_on_equality_horizon,
)
from gyroscopic.aQPU.api import (
    chirality_word6,
    q_word6,
    q_word6_for_items,
    state24_to_omega12,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

TR_SIGMA_SHELL = [0, 0.416667, 0.666667, 0.75, 0.666667, 0.416667, 0]
FA_STF = math.sqrt(6) / 9
CHI6_FULL = CHIRALITY_MASK_6
FAMILY_RAY_REF = 1

Q_G = 4 * np.pi
m_a = 1 / (2 * np.sqrt(2 * np.pi))
d_BU = 0.195342176580
rho = d_BU / m_a
Delta = 1 - rho
G_kernel = math.pi / 6
Omega_size = 4096
H_size = 64
W2_SHELL_DISPLACEMENT = 6  # per W2 depth-4 half-word (2 bytes); wavefunction_2 T2,T5
F_CYCLE_PATH_TRAVERSE = 12  # F-cycle path length (4 bytes); F preserves shell per T4
Z2_HOLONOMY_PATH_TRAVERSE = 24  # Z2 holonomy path length (8 bytes, 2 F-cycles); net disp 0
AF = 2 * Z2_HOLONOMY_PATH_TRAVERSE  # Z2 double-cover: holonomy cycle path x 2 for optical depth round-trip (T6: F o F = id)
v_EW = 246.22
G_meas = 6.708810e-39

# E_CS = E_Planck (Analysis_Energy_Scales 3.2). Not for deriving G (circular).
E_CS = 1.22e19

alpha_G_meas = G_meas * v_EW**2
f_ordered = 1.0 - 4.0 * rho * Delta**2

# Validation only: inverts measured G through alpha_G(v) = G_kernel * exp(-tau).
tau_required = -math.log(alpha_G_meas / G_kernel)
tau_req_meas = tau_required

# Consistency check: Z2 optical depth vs UV-IR conjugacy ladder (not a G derivation).
tau_conjugacy_depth = 2.0 * math.log(E_CS / v_EW)
tau_G_formula = Omega_size * Delta * rho**5 * f_ordered
binom_shell = [comb(6, s) / 64.0 for s in range(7)]
weights_pop = {m: binom_shell[bin(m).count("1")] for m in range(64)}
pi_eq = FA_STF * TR_SIGMA_SHELL[3]
V_EW_PDG = (246.22, 0.01)
C4_REF = -1.75

K4_CHANNEL_FLAGS = [
    ("top", 0, 0, 0),
    ("higgs", 1, 0, 0),
    ("z", 1, 1, 0),
    ("w", 1, 1, 1),
]


def configure_stdout_utf8():
    fn = getattr(sys.stdout, "reconfigure", None)
    if callable(fn):
        try:
            fn(encoding="utf-8", errors="replace")
        except Exception:
            pass


def byte_from_family_and_micro(family: int, micro_ref: int) -> int:
    family &= 0x03
    micro_ref &= 0x3F
    bit7 = (family >> 1) & 1
    bit0 = family & 1
    intron = (bit7 << 7) | (micro_ref << 1) | bit0
    return intron ^ GENE_MIC_S


def family_word_for_micro(micro_ref: int) -> list[int]:
    return [byte_from_family_and_micro(fam, micro_ref) for fam in range(4)]


def W2_word(micro_ref: int) -> list[int]:
    """Depth-4 half-word (families 00,01). W2 is an involution mapping shell s -> 6-s (T2)."""
    return [
        byte_from_family_and_micro(0, micro_ref),
        byte_from_family_and_micro(1, micro_ref),
    ]


def W2p_word(micro_ref: int) -> list[int]:
    """Depth-4 half-word (families 10,11). W2' is an involution mapping shell s -> 6-s (T3)."""
    return [
        byte_from_family_and_micro(2, micro_ref),
        byte_from_family_and_micro(3, micro_ref),
    ]


def F_cycle_word(micro_ref: int) -> list[int]:
    """One F-cycle: W2 o W2' (T6). 4 bytes; gate F on Omega. Z2 carrier flip; net shell displacement = 0 (T4)."""
    return family_word_for_micro(micro_ref)


def cycle_word_for_micro(micro_ref: int) -> list[int]:
    """Z2 holonomy cycle word: F o F = id (carrier Z2 round-trip). 8 bytes; two F-cycles completing Z2 holonomy. Carrier returns to rest, NOT to CS."""
    return family_word_for_micro(micro_ref) * 2


def apply_word_to_state(word: list[int], state24: int = GENE_MAC_REST) -> int:
    s = state24
    for b in word:
        s = step_state_by_byte(s, b)
    return s


def _shell_fields(state24: int) -> tuple[int, int]:
    chi_shell = state24_to_omega12(state24).shell
    return chi_shell, 6 - chi_shell


def trace_word_steps(
    word: list[int],
    start_state24: int = GENE_MAC_REST,
    micro_ref: int = 0,
) -> list[dict]:
    s = int(start_state24) & 0xFFFFFF
    q_acc = 0
    chi0, arch0 = _shell_fields(s)
    rows = [{
        "step": 0,
        "byte": None,
        "state24": s,
        "shell": chi0,
        "arch_shell": arch0,
        "chi6": chirality_word6(s),
        "qxor": 0,
        "family": 0,
        "micro": micro_ref & 0x3F,
        "intron": None,
        "on_horizon": is_on_horizon(s),
        "on_equality_horizon": is_on_equality_horizon(s),
    }]
    for step, byte in enumerate(word, start=1):
        s = step_state_by_byte(s, byte)
        q_acc = (q_acc ^ q_word6(byte)) & CHI6_FULL
        intron = byte_to_intron(byte)
        chi_sh, arch_sh = _shell_fields(s)
        rows.append({
            "step": step,
            "byte": byte,
            "state24": s,
            "shell": chi_sh,
            "arch_shell": arch_sh,
            "chi6": chirality_word6(s),
            "qxor": q_acc,
            "family": intron_family(intron),
            "micro": intron_micro_ref(intron),
            "intron": intron,
            "on_horizon": is_on_horizon(s),
            "on_equality_horizon": is_on_equality_horizon(s),
        })
    return rows


def poly_mul(a, b):
    out = np.zeros(len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, aj in enumerate(b):
            out[i + j] += ai * aj
    return out


def pi_norm_shell(shell_k):
    if shell_k in (0, 6) or shell_k < 0 or shell_k > 6:
        return 0.0
    return FA_STF * TR_SIGMA_SHELL[shell_k]


def build_joint_table() -> list[dict]:
    """Joint step table for one Z2 holonomy cycle (8 steps per micro_ref)."""
    table = []
    for m_ref in range(64):
        pop_m = bin(m_ref).count("1")
        weight = comb(6, pop_m) / 64.0
        word = cycle_word_for_micro(m_ref)
        for row in trace_word_steps(word, micro_ref=m_ref)[1:]:
            table.append({
                "m_ref": m_ref,
                "pop": pop_m,
                "weight": weight,
                "step": row["step"],
                "byte": row["byte"],
                "state24": row["state24"],
                "arch_shell": int(row["arch_shell"]),
                "intron": row["intron"],
                "family": row["family"],
                "micro": row["micro"],
                "qxor": row["qxor"],
                "chi6": row["chi6"],
            })
    return table


def kappa_pi_step(shell_k, chi6, chi6_full=CHI6_FULL):
    if shell_k in (0, 6) or shell_k < 0 or shell_k > 6:
        return 0.0
    pi_k = pi_norm_shell(shell_k)
    return Delta * (pi_k / pi_eq) if pi_eq > 0 else 0.0


def kappa_binom_step(shell_k, chi6, chi6_full=CHI6_FULL):
    if shell_k < 0 or shell_k > 6:
        return 0.0
    if shell_k in (0, 6):
        return 0.0
    return Delta * binom_shell[shell_k]


def tau_path_kappa(micro, shell_w=None):
    if shell_w is None:
        shell_w = binom_shell
    tau_path = 0.0
    for row in trace_word_steps(cycle_word_for_micro(micro))[1:]:
        s = row["arch_shell"]
        if s < 0 or s > 6:
            continue
        if s in (0, 6):
            continue
        tau_path += Delta * shell_w[s]
    return tau_path


def tau_path_binom(micro, chi6_full=CHI6_FULL):
    tau = 0.0
    for row in trace_word_steps(cycle_word_for_micro(micro))[1:]:
        tau += kappa_binom_step(row["arch_shell"], row["chi6"], chi6_full)
    return tau


def tau_cycle_weighted(micro_weights, shell_w=None):
    if shell_w is None:
        shell_w = binom_shell
    tau_sum = 0.0
    w_sum = 0.0
    for micro in range(64):
        w = micro_weights.get(micro, 0.0)
        if w <= 0:
            continue
        tau_sum += tau_path_kappa(micro, shell_w) * w
        w_sum += w
    return tau_sum / w_sum if w_sum > 0 else 0.0


def tau_g_with_c4(c4_val):
    f_ext = 1.0 - 4.0 * rho * Delta**2 + c4_val * Delta**4
    return Omega_size * Delta * rho**5 * f_ext


def g_pred_from_tau(tau_val):
    return G_kernel * math.exp(-tau_val) / v_EW**2


def shell_bounds_embed(k: int) -> tuple[float, float]:
    """Shell k at r_k = k/6, width 1/6, clipped to [0, 1]."""
    dr = 1.0 / 6.0
    r_c = k / 6.0
    return max(0.0, r_c - dr / 2.0), min(1.0, r_c + dr / 2.0)


def enclosed_mass_binomial(r: float) -> float:
    """M(r) = integral_0^r 4 pi r'^2 rho(r') dr' from binomial shell fractions."""
    if r <= 0.0:
        return 0.0
    total = 0.0
    for k in range(7):
        r_in, r_out = shell_bounds_embed(k)
        if r <= r_in:
            break
        w_k = binom_shell[k]
        if r >= r_out:
            total += w_k
        else:
            vol_full = r_out**3 - r_in**3
            vol_part = r**3 - r_in**3
            if vol_full > 0:
                total += w_k * (vol_part / vol_full)
    return total


def kernel_field_g(r: float, m_enc: float | None = None) -> float:
    """g(r) = -G_kernel M(r) / r^2 from kernel Gauss law."""
    if r <= 0.0:
        return float("nan")
    m = enclosed_mass_binomial(r) if m_enc is None else m_enc
    return -G_kernel * m / (r * r)


def verify_gauss_law_bridge(*, n_ext: int = 40) -> dict:
    """Discrete flux closure and exterior inverse-square checks."""
    m_total = enclosed_mass_binomial(1.0)
    flux_bnd = 4.0 * math.pi * kernel_field_g(1.0)
    flux_pred = -Q_G * G_kernel * m_total
    rs = np.linspace(1.05, 3.0, n_ext)
    gr2 = [abs(kernel_field_g(r, m_total)) * r**2 for r in rs]
    mean_gr2 = float(np.mean(gr2))
    rel_std = float(np.std(gr2)) / mean_gr2 if mean_gr2 else float("nan")
    log_r = np.log(rs)
    log_g = np.log([abs(kernel_field_g(r, m_total)) for r in rs])
    slope = float(np.polyfit(log_r, log_g, 1)[0])
    return {
        "m_total": m_total,
        "flux_boundary": flux_bnd,
        "flux_expected": flux_pred,
        "flux_ratio": flux_bnd / flux_pred if flux_pred else float("nan"),
        "exterior_gr2_mean": mean_gr2,
        "exterior_gr2_rel_std": rel_std,
        "log_slope": slope,
        "ok_flux": abs(flux_bnd / flux_pred - 1.0) < 1e-12 if flux_pred else False,
        "ok_is": rel_std < 1e-14,
        "ok_slope": abs(slope + 2.0) < 1e-10,
    }


def alpha_lab_with_transport_corrections() -> float:
    """
    alpha after AB, HC, IDE transport corrections (cgm_corrections_analysis_1).
    """
    d = d_BU
    mp_ = m_a
    r_curv = 0.993434896272
    h_hol = 4.417034
    rho_inv = 1.021137
    diff = 0.001874
    d_ap = 1.0 - d / mp_
    d2 = d_ap * d_ap
    d4 = d2 * d2
    phi = 3.0 * d + diff
    c_ab = 1.0 - (3.0 / 4.0) * r_curv * d2
    c_hc = 1.0 - (5.0 / 6.0) * (
        (phi / (3.0 * d) - 1.0)
        * (1.0 - d2 * h_hol)
        * d2
        / (4.0 * math.pi * math.sqrt(3.0))
    )
    c_ide = 1.0 + rho_inv * diff * d4
    alpha0 = d**4 / mp_
    return alpha0 * c_ab * c_hc * c_ide


def zeta_from_alpha(alpha: float) -> float:
    """zeta = rho^4 / (pi sqrt(3) alpha) from alpha*zeta identity."""
    return rho**4 / (math.pi * math.sqrt(3.0) * alpha)


def verify_alpha_zeta_product(*, alpha_codata: float | None = None) -> dict:
    """alpha_kernel * zeta = rho^4 / (pi sqrt(3))."""
    zeta_geom = 16.0 * math.sqrt(2.0 * math.pi / 3.0)
    alpha_kernel = d_BU**4 / m_a
    rhs = rho**4 / (math.pi * math.sqrt(3.0))
    lhs = alpha_kernel * zeta_geom
    out = {
        "zeta_geom": zeta_geom,
        "alpha_kernel": alpha_kernel,
        "lhs": lhs,
        "rhs": rhs,
        "exact": lhs == rhs,
    }
    if alpha_codata is not None and alpha_codata > 0:
        out["alpha_codata"] = alpha_codata
        out["zeta_from_alpha"] = rhs / alpha_codata
        out["zeta_ratio"] = out["zeta_from_alpha"] / zeta_geom - 1.0
    alpha_lab = alpha_lab_with_transport_corrections()
    out["alpha_lab"] = alpha_lab
    out["zeta_predicted_lab"] = zeta_from_alpha(alpha_lab)
    out["zeta_ratio_lab"] = out["zeta_predicted_lab"] / zeta_geom - 1.0
    return out


def c4_from_anchors(g_gev2, v_ew_gev):
    """Validation only: c4 implied by measured G. Do not use G = G_kernel/E_CS^2."""
    tau_req_loc = -math.log(g_gev2 * v_ew_gev**2 / G_kernel)
    k_need = tau_req_loc / (Omega_size * Delta * rho**5)
    x_ex = k_need - f_ordered
    return x_ex / Delta**4 if Delta > 0 else float("nan")


def k4_pq_charges():
    """EW trace-free charges (p, q) per K4 channel from gyrotriangle closure.

    Channel flags on the K4 edge walk (see cgm_compact_geom_core.CHANNELS):
      b (base): breaks CS reference frame (Higgs path)
      r (rot):  ONA reversal increment on the edge
      bal:    BU balance increment on the edge

    Formulas match _pq() in cgm_compact_geom_core: p = 1 + (-C1/2)*b + (C1/4)*r + 2*bal,
    q = 5/4 - 2*r - bal with C1=6 (CODE_C1). Returns (p, q) per channel name.
    """
    p0, q0 = 1.0, 5.0 / 4.0
    rows = []
    for name, b, r, bal in K4_CHANNEL_FLAGS:
        p = p0 + (0.0 if not b else -6.0 / 2.0)
        q = q0
        if r:
            p += 6.0 / 4.0
            q += -4.0 * 0.5
        if bal:
            p += 4.0 * 0.5
            q += -2.0 * 0.5
        rows.append((name, p, q))
    q_sum = sum(r[2] for r in rows)
    return rows, q_sum


def chi6_step_bit_stats(joint_table):
    by_step = {}
    for rec in joint_table:
        by_step.setdefault(rec["step"], []).append(int(rec["chi6"]) & 0x3F)
    for s in sorted(by_step):
        xs = by_step[s]
        n = len(xs)
        probs = [sum((x >> b) & 1 for x in xs) / n if n > 0 else 0.0 for b in range(6)]
        ps = " ".join(f"{p:.3f}" for p in probs)
        print(f"  step {s}: n={n}  P(bit=1) b0..b5 = {ps}")


def print_joint_table_condensed(table):
    n = len(table)
    print(f"  rows={n} (expected 512)")
    by_m = {}
    for rec in table:
        by_m.setdefault(rec["m_ref"], []).append(rec)

    fa_ref = [0, 1, 2, 3, 0, 1, 2, 3]
    mismatches = []
    by_pop = {}
    for m in range(64):
        rows = sorted(by_m.get(m, []), key=lambda r: r["step"])
        if not rows:
            continue
        pop = rows[0]["pop"]
        w = rows[0]["weight"]
        sh = [r["arch_shell"] for r in rows]
        mi = [r["micro"] for r in rows]
        ins = [r["intron"] for r in rows]
        fa = [r["family"] for r in rows]
        qx = [r["qxor"] for r in rows]
        exp_sh = [pop, 6, pop, 0, pop, 6, pop, 0]
        exp_mi = [m] * 8
        exp_in = [2 * m, 2 * m + 1, 128 + 2 * m, 129 + 2 * m] * 2
        exp_qx = [m, 63, m, 0] * 2
        if (sh, mi, ins, fa, qx) != (exp_sh, exp_mi, exp_in, fa_ref, exp_qx):
            mismatches.append(m)
        by_pop.setdefault(pop, []).append((m, w))

    print("  step 1..8 templates (verified on all 64 m_ref):")
    print("    fa: 0>1>2>3>0>1>2>3")
    print("    arch_sh(pop): pop>6>pop>0>pop>6>pop>0")
    print("    mi(m): m x8")
    print("    in(m): 2m>2m+1>128+2m>129+2m (half-cycle x2)")
    print("    qx(m): m>63>m>0 (half-cycle x2)")
    print("    w(pop)=binom(6,pop)/64")
    for pop in sorted(by_pop):
        ms = sorted(x[0] for x in by_pop[pop])
        w = by_pop[pop][0][1]
        m_str = ",".join(f"{x:02d}" for x in ms)
        print(f"    pop={pop} w={w:.6f} n={len(ms)} m=[{m_str}]")
    if mismatches:
        print(f"  template mismatches at m={mismatches}")
