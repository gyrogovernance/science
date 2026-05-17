"""
Shared CGM gravity kernel helpers and invariants.

Used by cgm_gravity_analysis_2.py (main) and cgm_gravity_analysis_3.py (compat).
"""

from __future__ import annotations

import math
import sys
from math import comb
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_exp = Path(__file__).resolve().parent
if str(_exp) not in sys.path:
    sys.path.insert(0, str(_exp))

TR_SIGMA_SHELL = [0, 0.416667, 0.666667, 0.75, 0.666667, 0.416667, 0]
FA_STF = math.sqrt(6) / 9
CHI6_FULL = 0x3F

Q_G = 4 * np.pi
m_a = 1 / (2 * np.sqrt(2 * np.pi))
d_BU = 0.195342176580
rho = d_BU / m_a
Delta = 1 - rho
G_kernel = math.pi / 6
Omega_size = 4096
H_size = 64
AF = 48
v_EW = 246.22
G_meas = 6.708810e-39
E_CS = math.sqrt(G_kernel / G_meas)
alpha_G_meas = G_meas * v_EW**2
f_ordered = 1.0 - 4.0 * rho * Delta**2
tau_required = 2.0 * np.log(E_CS / v_EW)
tau_req_meas = -math.log(alpha_G_meas / G_kernel)
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


def require_monodromy():
    try:
        import cgm_aqpu_monodromy as mon
        return mon
    except Exception as exc:
        print(f"  Monodromy unavailable: {exc}")
        return None


def require_compact_geom():
    try:
        from cgm_compact_geom_core import CHANNELS, HORIZON_CARDINALITY
        return CHANNELS, HORIZON_CARDINALITY
    except Exception as exc:
        print(f"  Compact geometry module unavailable: {exc}")
        return None, None


def pm1_pair(a, b):
    return 1.0 if (a ^ b) == 1 else -1.0


def trans_vector_pm1(micro):
    return np.array([
        2.0 * ((micro >> 5) & 1) - 1.0,
        2.0 * ((micro >> 4) & 1) - 1.0,
        2.0 * ((micro >> 3) & 1) - 1.0,
    ])


def rot_vector_pm1(micro):
    return np.array([
        2.0 * ((micro >> 2) & 1) - 1.0,
        2.0 * ((micro >> 1) & 1) - 1.0,
        2.0 * ((micro >> 0) & 1) - 1.0,
    ])


def v3_from_chi6_pairs_pm1(chi6):
    c = int(chi6) & 0x3F
    x0, x1 = (c >> 0) & 1, (c >> 1) & 1
    y0, y1 = (c >> 2) & 1, (c >> 3) & 1
    z0, z1 = (c >> 4) & 1, (c >> 5) & 1
    return np.array([
        pm1_pair(x0, x1),
        pm1_pair(y0, y1),
        pm1_pair(z0, z1),
    ], dtype=float)


def quadrupole_tensor(v):
    v = np.asarray(v, dtype=float)
    return np.outer(v, v) - np.eye(3) * np.dot(v, v) / 3.0


def q_stf_norm(Q):
    return float(np.sqrt(np.sum(Q * Q) / 2.0))


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


def build_joint_table(mon):
    table = []
    for m_ref in range(64):
        pop_m = bin(m_ref).count("1")
        weight = comb(6, pop_m) / 64.0
        word = mon.family_word_for_micro(m_ref) * 2
        for row in mon.trace_word(word)[1:]:
            if row.byte is None:
                continue
            intron = mon.byte_to_intron(row.byte)
            mp = mon.intron_micro_ref(intron)
            chi6 = row.obs.chi6
            v_t = v3_from_chi6_pairs_pm1(chi6)
            v_micro = trans_vector_pm1(mp)
            v_r = rot_vector_pm1(mp)
            Q = quadrupole_tensor(v_t)
            Q_micro = quadrupole_tensor(v_micro)
            shell = int(row.obs.arch_shell)
            table.append({
                "m_ref": m_ref,
                "pop": pop_m,
                "weight": weight,
                "step": row.step,
                "shell": shell,
                "arch_shell": shell,
                "intron": intron,
                "family": mon.intron_family(intron),
                "micro": mp,
                "byte": row.byte,
                "qxor": row.qxor,
                "chi6": chi6,
                "v_t": v_t,
                "v_r": v_r,
                "Q": Q,
                "Q_micro": Q_micro,
            })
    return table


def kappa_pi_step(shell_k, chi6, chi6_full=CHI6_FULL):
    if shell_k in (0, 6) or shell_k < 0 or shell_k > 6:
        return 0.0
    if chi6 == chi6_full:
        return 0.0
    pi_k = pi_norm_shell(shell_k)
    return Delta * (pi_k / pi_eq) if pi_eq > 0 else 0.0


def kappa_binom_step(shell_k, chi6, chi6_full=CHI6_FULL):
    if shell_k < 0 or shell_k > 6:
        return 0.0
    cf = 0.0 if chi6 == chi6_full else 1.0
    return Delta * binom_shell[shell_k] * cf


def tau_path_kappa(mon, micro, shell_w=None):
    if shell_w is None:
        shell_w = binom_shell
    word = mon.family_word_for_micro(micro) * 2
    tau_path = 0.0
    for row in mon.trace_word(word)[1:]:
        if row.byte is None:
            continue
        s = row.obs.arch_shell
        if s < 0 or s > 6:
            continue
        cf = 0.0 if row.obs.chi6 == CHI6_FULL else 1.0
        tau_path += Delta * shell_w[s] * cf
    return tau_path


def tau_path_binom(mon, micro, chi6_full=CHI6_FULL):
    word = mon.family_word_for_micro(micro) * 2
    tau = 0.0
    for row in mon.trace_word(word)[1:]:
        if row.byte is None:
            continue
        tau += kappa_binom_step(row.obs.arch_shell, row.obs.chi6, chi6_full)
    return tau


def tau_cycle_weighted(mon, micro_weights, shell_w=None):
    if shell_w is None:
        shell_w = binom_shell
    tau_sum = 0.0
    w_sum = 0.0
    for micro in range(64):
        w = micro_weights.get(micro, 0.0)
        if w <= 0:
            continue
        tau_sum += tau_path_kappa(mon, micro, shell_w) * w
        w_sum += w
    return tau_sum / w_sum if w_sum > 0 else 0.0


def defect_weight_per_depth8(mon, micro):
    word = mon.family_word_for_micro(micro) * 2
    rows = [r for r in mon.trace_word(word)[1:] if r.byte is not None]
    wsum = 0
    for i in range(1, len(rows)):
        d = mon.byte_chirality_increment(rows[i - 1].byte) ^ mon.byte_chirality_increment(
            rows[i].byte
        )
        wsum += int(bin(d).count("1"))
    return wsum


def measure_defect_invariant(mon):
    vals = [defect_weight_per_depth8(mon, m) for m in range(64)]
    return sorted(set(vals)), float(np.mean(vals))


def defect_opacity_scale():
    return 2.0 * Delta**2 / rho**3


def tau_cycle_from_defect(defect_weight):
    return defect_opacity_scale() * float(defect_weight)


def tau_g_with_c4(c4_val):
    f_ext = 1.0 - 4.0 * rho * Delta**2 + c4_val * Delta**4
    return Omega_size * Delta * rho**5 * f_ext


def g_pred_from_tau(tau_val):
    return G_kernel * math.exp(-tau_val) / v_EW**2


def c4_from_anchors(g_gev2, v_ew_gev):
    e_cs_loc = math.sqrt(G_kernel / g_gev2)
    tau_req_loc = 2.0 * math.log(e_cs_loc / v_ew_gev)
    k_need = tau_req_loc / (Omega_size * Delta * rho**5)
    x_ex = k_need - f_ordered
    return x_ex / Delta**4 if Delta > 0 else float("nan")


def k4_pq_charges():
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


def pi_vec_bulk_step(shell_k, frac=None):
    if frac is None:
        frac = [0.3, 0.3, 0.2, 0.1, 0.1]
    total = pi_norm_shell(shell_k)
    return np.array([total * f for f in frac], dtype=float)


def k4_gate_index(f_prev, f_next):
    return int((f_next - f_prev) % 4)


def stf_k4_coupling_matrix(canonical_path, frac=None):
    fsum = canonical_path["fsum"]
    shell = canonical_path["shell"]
    bulk_idx = [
        i for i in range(len(shell))
        if 1 <= shell[i] <= 5 and not canonical_path["H"][i]
        and not canonical_path["E_flag"][i]
    ]
    m = np.zeros((5, 4), dtype=float)
    for i in bulk_idx:
        pi = pi_vec_bulk_step(shell[i], frac)
        g_prev = fsum[i - 1] if i > 0 else fsum[i]
        g_next = fsum[i]
        g = k4_gate_index(g_prev, g_next)
        for a in range(5):
            m[a, g] += pi[a]
    if bulk_idx:
        m /= float(len(bulk_idx))
    return m, bulk_idx


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
        sh = [r["shell"] for r in rows]
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
    print("    sh(pop): pop>6>pop>0>pop>6>pop>0")
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
    qnorms = sorted({f"{q_stf_norm(rec['Q']):.4f}" for rec in table})
    qmicro = sorted({
        f"{q_stf_norm(rec['Q_micro']):.4f}"
        for rec in table if "Q_micro" in rec
    })
    print(f"  |Q(chi6)|_stf unique: {qnorms}")
    if qmicro:
        print(f"  |Q(micro)|_stf unique: {qmicro}")
