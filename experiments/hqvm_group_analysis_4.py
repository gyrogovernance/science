#!/usr/bin/env python3
"""hqvm_group_analysis_4.py — Irrep labeling, 32-bit register lift, hQVM(d) family.

Role: PASS/FAIL gates on 2016 two-dim irrep labels by (n_u, n_v), odd-coset vanishing,
character L2 / rho2 homomorphism samples, Peter-Weyl block order; 32-bit register
shadow fibers and s=1 spinor sector; G_d census, mixing, transient dim, d=6 lock.
Inputs: Clifford / register / G_d helpers via hqvm_group_analysis_common;
gyroscopic.hQVM.family build_hqvm_d.
Outputs: gates appended to ReportState; printed tables.
Companion: hqvm_group_analysis_1/2/3/5.py, hqvm_group_analysis_run.py,
hqvm_group_analysis_common.py.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from math import comb

import numpy as np

from hqvm_group_analysis_common import (
    _KERNEL_OK,
    ReportState, section, check, vprint,
    kernel_group, kernel_group_d, compose_sig_int, apply_sig_int,
    inv_sig_int,
    swap_halves, twod_char, rho2, k_reps, irrep_label,
    clifford_irrep_counts,
    linear_char, lin_reps,
    pack_register32, unpack_register32, shadow_register32, step_register32,
    step_state_by_byte, omega12_to_state24, state24_to_omega12, OmegaState12,
    GENE_MAC_REST, shadow_partner_byte,
    byte_step_set, permutation_character_d, conjugacy_class_index,
    walsh_hadamard64, byte_transition_matrix,
)

if _KERNEL_OK:
    from gyroscopic.hQVM.family import (
        build_hqvm_d, step_uv, enumerate_omega_d, alphabet_size,
        q_word_d,
    )






def _theory_orbit_count(nu, nv, d=6):
    """Orbit count for sorted bidegree bin (nu, nv) with nu <= nv."""
    if nu > nv:
        nu, nv = nv, nu
    if nu < nv:
        return comb(d, nu) * comb(d, nv)
    c = comb(d, nu)
    return (c * c - c) // 2


def _bidegree_theory_table(d=6):
    rows = []
    for nu in range(d + 1):
        for nv in range(nu, d + 1):
            rows.append(((nu, nv), _theory_orbit_count(nu, nv, d)))
    return rows


def _matrix_coeff_ip(k1, i1, j1, k2, i2, j2, gl, d=6):
    """Peter-Weyl inner product of matrix coefficients (1/|gl|) sum_g rho_ij conj(rho')."""
    n = len(gl)
    acc = 0.0 + 0.0j
    for g in gl:
        r1 = rho2(k1, g, d)
        r2 = rho2(k2, g, d)
        acc += r1[i1, j1] * np.conj(r2[i2, j2])
    return acc / n


def _rho_homomorphism_errors(k, gl_sample, d=6):
    """Count failures of rho(gh)=rho(g)rho(h) on sample pairs from gl_sample."""
    errs = 0
    n = len(gl_sample)
    trials = min(200, n * n)
    rng = np.random.RandomState(int(k) * 17 + 3)
    for _ in range(trials):
        g = int(gl_sample[rng.randint(0, n)])
        h = int(gl_sample[rng.randint(0, n)])
        lhs = rho2(k, compose_sig_int(g, h), d)
        rhs = rho2(k, g, d) @ rho2(k, h, d)
        if not np.allclose(lhs, rhs):
            errs += 1
    return errs, trials


def _character_l2_norm(k, gl, d=6):
    """||chi_k||_2^2 = (1/|G|) sum |chi|^2; expect 1 for irreps."""
    n = len(gl)
    s = sum(twod_char(k, g, d) ** 2 for g in gl)
    return s / n






def _pop(x):
    return bin(int(x)).count('1')


def _unpack_omega_index(idx):
    return (idx >> 6) & 63, idx & 63


def _omega_index(u6, v6):
    return ((u6 & 63) << 6) | (v6 & 63)


def carrier_dim():
    return 8192  # 2 * 4096


def pack_carrier(sigma, omega_idx):
    return ((sigma & 1) << 12) | (omega_idx & 4095)


def unpack_carrier(x):
    return (x >> 12) & 1, x & 4095


def apply_g_on_carrier(g, sigma, omega_idx):
    p = (g >> 12) & 1
    u, v = _unpack_omega_index(omega_idx)
    st = omega12_to_state24(OmegaState12(u, v))
    st2 = apply_sig_int(g, st)
    om2 = state24_to_omega12(st2)
    return sigma ^ p, _omega_index(om2.u6, om2.v6)


def psi_s1(a, sigma, omega_idx):
    u, v = _unpack_omega_index(omega_idx)
    chi = (u ^ v) & 63
    return ((-1) ** (sigma & 1)) * ((-1) ** (_pop(a & chi) & 1))


def omega_embed(f_omega, sigma, omega_idx):
    """Embed Omega function as sigma-even on carrier."""
    return f_omega(omega_idx)


def ip_carrier(f, h, normalize=True):
    s = 0.0
    for sigma in (0, 1):
        for w in range(4096):
            s += f(sigma, w) * h(sigma, w)
    if normalize:
        return s / carrier_dim()
    return s


def transform_check_psi(a, g, n_samples, rng):
    """Count points where g.psi_a != chi_{1,a}(g) * psi_a."""
    chi = linear_char(1, a, g)
    bad = 0
    for _ in range(n_samples):
        sigma = int(rng.randint(0, 2))
        w = int(rng.randint(0, 4096))
        s2, w2 = apply_g_on_carrier(g, sigma, w)
        lhs = psi_s1(a, s2, w2)
        rhs = chi * psi_s1(a, sigma, w)
        if lhs != rhs:
            bad += 1
    return bad


def shadow_fiber_census(start24):
    """Mac images and fiber sizes under shadow_partner collapse from start24."""
    buckets = defaultdict(list)
    for b in range(256):
        mac = step_state_by_byte(start24, b) & 0xFFFFFF
        buckets[mac].append(b)
    sizes = Counter(len(v) for v in buckets.values())
    partner_closed = 0
    for mac, bs in buckets.items():
        if len(bs) != 2:
            continue
        b0, b1 = bs
        if shadow_partner_byte(b0) == b1 and shadow_partner_byte(b1) == b0:
            partner_closed += 1
    return buckets, sizes, partner_closed


def phase_pair_distinguishability(n_mac, rng):
    """Shadow partners: same Mac, distinct intron/high-8."""
    start = GENE_MAC_REST & 0xFFFFFF
    n_pairs = 0
    same_mac_from_rest = 0
    diff_intron_same_mac = 0
    same_mac_rand = 0
    for b in range(256):
        p = shadow_partner_byte(b)
        if p <= b:
            continue
        n_pairs += 1
        m1 = step_state_by_byte(start, b) & 0xFFFFFF
        m2 = step_state_by_byte(start, p) & 0xFFFFFF
        if m1 == m2:
            same_mac_from_rest += 1
        r1 = step_register32(b, pack_register32(0, start))
        r2 = step_register32(p, pack_register32(0, start))
        i1, m1r = unpack_register32(r1)
        i2, m2r = unpack_register32(r2)
        if m1r == m2r and i1 != i2:
            diff_intron_same_mac += 1
    for _ in range(n_mac):
        mac0 = int(rng.randint(0, 1 << 24))
        b = int(rng.randint(0, 256))
        p = shadow_partner_byte(b)
        m1 = step_state_by_byte(mac0, b) & 0xFFFFFF
        m2 = step_state_by_byte(mac0, p) & 0xFFFFFF
        if m1 == m2:
            same_mac_rand += 1
    return {
        'n_pairs': n_pairs,
        'same_mac_from_rest': same_mac_from_rest,
        'diff_intron_same_mac': diff_intron_same_mac,
        'same_mac_rand': same_mac_rand,
        'n_mac': n_mac,
    }


def depth4_intron_uniqueness(n_frames, rng, start_reg):
    """Depth-4 words: intron-seq uniqueness vs Mac-seq uniqueness."""
    intron_keys = Counter()
    mac_keys = Counter()
    partner_mac_match = 0
    partner_intron_diff = 0
    partner_trials = 0
    for _ in range(n_frames):
        word = [int(x) for x in rng.randint(0, 256, 4)]
        reg = start_reg
        intron_seq = []
        mac_seq = []
        for b in word:
            reg = step_register32(b, reg)
            intron8, mac = unpack_register32(reg)
            intron_seq.append(intron8)
            mac_seq.append(mac)
        intron_keys[tuple(intron_seq)] += 1
        mac_keys[tuple(mac_seq)] += 1
        # partner twist on first byte
        b0 = word[0]
        p0 = shadow_partner_byte(b0)
        if p0 != b0:
            partner_trials += 1
            word_p = [p0] + word[1:]
            reg_p = start_reg
            intron_p = []
            mac_p = []
            for b in word_p:
                reg_p = step_register32(b, reg_p)
                i8, m = unpack_register32(reg_p)
                intron_p.append(i8)
                mac_p.append(m)
            if tuple(mac_p) == tuple(mac_seq):
                partner_mac_match += 1
            if tuple(intron_p) != tuple(intron_seq):
                partner_intron_diff += 1
    return {
        'n_frames': n_frames,
        'intron_keys': len(intron_keys),
        'mac_keys': len(mac_keys),
        'intron_collisions': sum(1 for c in intron_keys.values() if c > 1),
        'mac_collisions': sum(1 for c in mac_keys.values() if c > 1),
        'partner_mac_match': partner_mac_match,
        'partner_intron_diff': partner_intron_diff,
        'partner_trials': partner_trials,
    }


def spinor_gram_matrix_fast(as_list):
    """Gram via chi shells: 2 * sum_chi 64 * (-1)^{pop((a^b)&chi)}."""
    n = len(as_list)
    G = np.zeros((n, n), dtype=np.float64)
    for i, a in enumerate(as_list):
        for j, b in enumerate(as_list):
            s = 0.0
            ab = (a ^ b) & 63
            for chi in range(64):
                s += 64.0 * ((-1) ** (_pop(ab & chi) & 1))
            G[i, j] = 2.0 * s
    return G


def verify_gram_closed(as_list):
    """Brute-force Gram on full carrier; return max |G_ij - 8192 delta|."""
    max_err = 0.0
    for a in as_list:
        for b in as_list:
            s = 0.0
            for sigma in (0, 1):
                for w in range(4096):
                    s += psi_s1(a, sigma, w) * psi_s1(b, sigma, w)
            target = float(carrier_dim()) if a == b else 0.0
            max_err = max(max_err, abs(s - target))
    return max_err


def omega_embed_orthogonality(as_list, n_f, rng):
    """max |<psi_a, iota(f)>| for random Omega functions f."""
    mx = 0.0
    for a in as_list:
        for _ in range(n_f):
            coeffs = rng.randint(-1, 2, size=4096).astype(np.float64)

            def f_om(w, c=coeffs):
                return float(c[w])

            ip = 0.0
            for sigma in (0, 1):
                for w in range(4096):
                    ip += psi_s1(a, sigma, w) * omega_embed(f_om, sigma, w)
            mx = max(mx, abs(ip))
    return mx


def enumerate_s1_labels():
    return [a for s, a in lin_reps(6) if s == 1]


def sample_transform_table(as_list, gl, rng, n_a=8, n_g=12, n_pts=20):
    total_bad = 0
    total_n = 0
    vprint(f'  {"a":>4} {"g":>6} {"chi":>4} {"bad/N":>8}')
    for a in as_list[:n_a]:
        for g in rng.choice(gl, size=min(n_g, len(gl)), replace=False):
            g = int(g)
            bad = transform_check_psi(a, g, n_pts, rng)
            chi = linear_char(1, a, g)
            vprint(f'  {a:4d} {g:6d} {chi:4d} {bad:4d}/{n_pts}')
            total_bad += bad
            total_n += n_pts
    return total_bad, total_n


def register_pack_roundtrip_table(n, rng):
    bad = 0
    for _ in range(n):
        intron = int(rng.randint(0, 256))
        mac = int(rng.randint(0, 1 << 24))
        reg = pack_register32(intron, mac)
        i2, m2 = unpack_register32(reg)
        sh = shadow_register32(reg)
        if i2 != intron or m2 != (mac & 0xFFFFFF) or sh != (mac & 0xFFFFFF):
            bad += 1
    return bad


def step_matches_kernel_table(n, rng):
    bad = 0
    for _ in range(n):
        b = int(rng.randint(0, 256))
        mac = int(rng.randint(0, 1 << 24))
        reg = pack_register32(0, mac)
        reg2 = step_register32(b, reg)
        _, mac_r = unpack_register32(reg2)
        mac_k = step_state_by_byte(mac, b) & 0xFFFFFF
        if mac_r != mac_k:
            bad += 1
    return bad





DS_FULL = (3, 4, 5, 6)
DS_ALL = (3, 4, 5, 6, 8)


def order_g_d(d):
    """|G_d| = 2^{2d+1}."""
    return 1 << (2 * d + 1)


def order_omega_d(d):
    return 1 << (2 * d)


def conjugacy_classes_d(d):
    """Enumerate conjugacy classes of G_d via conjugacy_class_index."""
    G = kernel_group_d(d)
    gl_arr = np.array(list(G), dtype=np.uint32)
    unseen = set(int(g) for g in G)
    classes = []
    while unseen:
        g = unseen.pop()
        c = conjugacy_class_index(g, gl_arr, d)
        classes.append((g, len(c)))
        unseen -= c
    return classes, G


def character_ip_perm_lin(d, s, a, gl):
    n = len(gl)
    acc = 0.0
    for g in gl:
        acc += permutation_character_d(g, d) * linear_char(s, a, g, d)
    return acc / n


def character_ip_perm_twod(d, k, gl):
    n = len(gl)
    acc = 0.0
    for g in gl:
        acc += permutation_character_d(g, d) * twod_char(k, g, d)
    return acc / n


def multiplicity_free_report(d, gl, max_lin=None, max_2d=None, rng=None):
    """Sample or full inner products of chi_perm with linear/2d chars."""
    n_lin, n_2d, _ = clifford_irrep_counts(d)
    lin = lin_reps(d)
    reps = k_reps(d)
    if max_lin is not None and len(lin) > max_lin:
        half = max_lin // 2
        a_max = 1 << d
        step = max(1, a_max // half)
        lin = [(0, a) for a in range(0, a_max, step)][:half]
        lin += [(1, a) for a in range(0, a_max, step)][:half]
    if max_2d is not None and len(reps) > max_2d:
        if rng is None:
            rng = np.random.RandomState(0)
        reps = list(rng.choice(reps, size=max_2d, replace=False))

    rows_lin = []
    for s, a in lin:
        rows_lin.append((s, a, character_ip_perm_lin(d, s, a, gl)))
    rows_2d = []
    for k in reps:
        rows_2d.append((int(k), character_ip_perm_twod(d, k, gl)))
    return rows_lin, rows_2d, n_lin, n_2d


def build_transition_via_hqvm(d):
    eng = build_hqvm_d(d)
    n = eng.n_omega
    A = eng.n_bytes
    P = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for b in range(A):
            j = eng.transitions[i][b]
            P[i, j] += 1.0 / A
    return P, eng


def unique_row_signatures(d, eng=None):
    if eng is None:
        eng = build_hqvm_d(d)
    seen = {}
    for i in range(eng.n_omega):
        row = eng.transitions[i]
        seen.setdefault(row, []).append(i)
    return len(seen), eng, seen


def rank_of_P(d, method='auto'):
    """Full SVD for d<=6; unique-row + sampled P^2 for d=8."""
    if d <= 6 and method in ('auto', 'svd'):
        P, eng = build_transition_via_hqvm(d)
        r1 = int(np.linalg.matrix_rank(P, tol=1e-9))
        P2 = P @ P
        r2 = int(np.linalg.matrix_rank(P2, tol=1e-9))
        n = P.shape[0]
        J = np.full((n, n), 1.0 / n)
        mix_err = float(np.max(np.abs(P2 - J)))
        n_unique, _, _ = unique_row_signatures(d, eng)
        return {
            'd': d, 'n_omega': n, 'n_bytes': eng.n_bytes,
            'rank_P': r1, 'rank_P2': r2, 'n_unique_rows': n_unique,
            'mix_err': mix_err, 'method': 'svd',
        }
    n_unique, eng, _ = unique_row_signatures(d)
    rng = np.random.RandomState(d * 1009)
    n = eng.n_omega
    A = eng.n_bytes
    max_dev = 0.0
    for _ in range(32):
        i = int(rng.randint(0, n))
        counts = np.zeros(n, dtype=np.float64)
        bs = rng.choice(A, size=min(A, 256), replace=False)
        scale = 1.0 / (len(bs) ** 2)
        for b1 in bs:
            j = eng.transitions[i][int(b1)]
            for b2 in bs:
                k = eng.transitions[j][int(b2)]
                counts[k] += scale
        max_dev = max(max_dev, float(np.max(np.abs(counts - 1.0 / n))))
    return {
        'd': d, 'n_omega': n, 'n_bytes': A,
        'rank_P': n_unique, 'rank_P2': 1 if max_dev < 5e-3 else -1,
        'n_unique_rows': n_unique, 'mix_err': max_dev,
        'method': 'unique_rows+sample_P2',
    }


def fit_rank_log_model(ds, ranks):
    """Least-squares fit log2(rank) ~= a*(d-1) + b."""
    xs = np.array([d - 1 for d in ds], dtype=np.float64)
    ys = np.array([math.log2(r) for r in ranks], dtype=np.float64)
    A = np.column_stack([xs, np.ones_like(xs)])
    coef, _, _, _ = np.linalg.lstsq(A, ys, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    pred = a * xs + b
    resid = ys - pred
    return a, b, float(np.max(np.abs(resid))), list(
        zip(ds, ranks, ys.tolist(), pred.tolist(), resid.tolist()))


def q_span_rank(d):
    """GF(2)-rank of {q_d(b)} over the alphabet."""
    from gyroscopic.hQVM.family import gf2_rank
    vecs = [q_word_d(b, d) for b in range(alphabet_size(d))]
    return gf2_rank(vecs, d)


def alphabet_signature_parity_table(d):
    """Odd-signature count and distinct step-set size for alphabet."""
    from hqvm_group_analysis_common import byte_signature_d
    odd = 0
    distinct = set()
    for b in range(alphabet_size(d)):
        g = byte_signature_d(b, d)
        distinct.add(g)
        if (g >> (2 * d)) & 1 == 1:
            odd += 1
    return odd, len(distinct), alphabet_size(d)


def d6_lock_targets():
    return {
        '|G|': 8192,
        '|Omega|': 4096,
        '|A|': 256,
        'n_lin': 128,
        'n_2d': 2016,
        'n_classes': 2144,
        'rank_P': 32,
        'rank_P2': 1,
        'step_set': 128,
    }


def g_d_structure_table(ds):
    rows = []
    for d in ds:
        rows.append({
            'd': d,
            '|G|': order_g_d(d),
            '|Omega|': order_omega_d(d),
            '|A|': alphabet_size(d) if _KERNEL_OK else -1,
            'n_lin': clifford_irrep_counts(d)[0],
            'n_2d': clifford_irrep_counts(d)[1],
            'n_cls': clifford_irrep_counts(d)[2],
        })
    return rows


def print_census_table(rows):
    vprint(f'  {"d":>3} {"|G|":>10} {"|Omega|":>10} {"|A|":>6} '
          f'{"n_lin":>6} {"n_2d":>8} {"n_cls":>8}')
    for r in rows:
        vprint(f'  {r["d"]:3d} {r["|G|"]:10d} {r["|Omega|"]:10d} {r["|A|"]:6d} '
              f'{r["n_lin"]:6d} {r["n_2d"]:8d} {r["n_cls"]:8d}')


def print_class_size_histogram(classes, d):
    hist = Counter(sz for _, sz in classes)
    vprint(f'  d={d} class-size histogram: {dict(sorted(hist.items()))}')


def print_mult_table(d, rows_lin, rows_2d):
    vprint(f'  d={d} mult-free sample: n_lin={len(rows_lin)} n_2d={len(rows_2d)}')
    vprint(f'  {"s":>3} {"a":>6} {"IP":>12}')
    for s, a, ip in rows_lin[:8]:
        vprint(f'  {s:3d} {a:6d} {ip:12.6f}')
    if len(rows_lin) > 8:
        vprint(f'  ... ({len(rows_lin) - 8} more lin)')
    vprint(f'  {"k":>8} {"IP":>12}')
    for k, ip in rows_2d[:6]:
        vprint(f'  {k:8d} {ip:12.6f}')
    if len(rows_2d) > 6:
        vprint(f'  ... ({len(rows_2d) - 6} more 2d)')


def print_rank_table(results):
    vprint(f'  {"d":>3} {"rank_P":>8} {"rank_P2":>8} {"uniq":>8} {"mix_err":>12} {"method":>22}')
    for r in results:
        vprint(f'  {r["d"]:3d} {r["rank_P"]:8d} {r["rank_P2"]:8d} '
              f'{r["n_unique_rows"]:8d} {r["mix_err"]:12.3e} {r["method"]:>22}')


def _sig_unpack(g, d=6):
    m = (1 << d) - 1
    return (g >> (2 * d)) & 1, (g >> d) & m, g & m


def _sig_pack(p, u, v, d=6):
    return (p << (2 * d)) | ((u & ((1 << d) - 1)) << d) | (v & ((1 << d) - 1))


def _special2_gates(state, gl):
    """Class-2 special 2-group: Z(G)=G'=Phi(G), quadratic map q(t,x)=t*x (A1)."""
    d = 6
    zg = set()
    for dd in range(1 << d):
        zg.add(_sig_pack(0, dd, dd, d))
    squares = set()
    for g in gl:
        squares.add(compose_sig_int(g, g))
    sq_ok = squares == zg
    check(state, f'G^2 = Z(G): |squares|={len(squares)}',
          sq_ok,
          quantity='G^2 = [G,G] = Z(G) (special 2-group)',
          measured=f'|sq|={len(squares)}, |Z|={len(zg)}', threshold='equal')
    rng = np.random.RandomState(31)
    q_ok = pol_ok = 0
    trials = 300
    for _ in range(trials):
        g = gl[int(rng.randint(0, len(gl)))]
        p, u, v = _sig_unpack(g, d)
        x = u ^ v
        gs = compose_sig_int(g, g)
        _, du, dv = _sig_unpack(gs, d)
        if du == dv and du == (p * x if p else 0):
            q_ok += 1
        a = gl[int(rng.randint(0, len(gl)))]
        b = gl[int(rng.randint(0, len(gl)))]
        pa, ua, va = _sig_unpack(a, d)
        pb, ub, vb = _sig_unpack(b, d)
        xa, xb = ua ^ va, ub ^ vb
        ab = compose_sig_int(
            compose_sig_int(a, b),
            compose_sig_int(inv_sig_int(a, 6), inv_sig_int(b, 6)))
        _, du, dv = _sig_unpack(ab, d)
        pred = ((xb if pa else 0) ^ (xa if pb else 0)) & 63
        if du == dv == pred:
            pol_ok += 1
    check(state, f'q(t,x)=t*x on odd squares: {q_ok}/{trials}',
          q_ok == trials,
          quantity='quadratic square map q(t,x) = t*x on quotient',
          measured=f'{q_ok}/{trials}', threshold='all')
    check(state, f'polarization B(a,b)=t*y+s*x: {pol_ok}/{trials}',
          pol_ok == trials,
          quantity='B(a,b) polarization of q',
          measured=f'{pol_ok}/{trials}', threshold='all')


def _affine_mode_perms():
    perms = set()
    for a in ((0, 0), (1, 0), (0, 1), (1, 1)):
        for sw in (False, True):
            lut = []
            for u in (0, 1):
                for v in (0, 1):
                    if sw:
                        up, vp = v ^ a[0], u ^ a[1]
                    else:
                        up, vp = u ^ a[0], v ^ a[1]
                    lut.append(up | (vp << 1))
            perms.add(tuple(lut))
    return perms


def _d8_fiber_gates(state):
    """Per-mode D8 and global parity-locked fiber product (A2)."""
    perms = _affine_mode_perms()
    gl = list(kernel_group())
    check(state, f'|G| = 2*4^6 = {len(gl)}',
          len(gl) == 2 * (4 ** 6),
          quantity='parity-locked D8 fiber product order',
          measured=str(len(gl)), threshold=str(2 * 4 ** 6))
    check(state, f'affine GF(2)^2 maps per mode: |closure|={len(perms)}',
          len(perms) == 8,
          quantity='each mode generates D8 (order 8)',
          measured=str(len(perms)), threshold='8')
    odds = sum(1 for g in gl if (g >> 12) & 1)
    evens = len(gl) - odds
    check(state, f'global swap cosets: |even|={evens}, |odd|={odds}',
          odds == evens == 4096,
          quantity='one global reflection bit splits (D8)^6 fiber product',
          measured=f'{evens}/{odds}', threshold='4096/4096')


def _quotient32_gates(state):
    """Rank-32 transient = even-|k1| Walsh sector, k2=0 (A5)."""
    W64 = walsh_hadamard64()
    P, _ = byte_transition_matrix()
    Pd = P.toarray()
    n_surv = 0
    for k1 in range(64):
        if bin(k1).count('1') % 2:
            continue
        mode = np.kron(W64[k1], W64[0])
        if float(np.linalg.norm(Pd @ mode)) > 1e-8:
            n_surv += 1
    comp = 63
    comp_even = (bin(comp).count('1') % 2) == 0
    check(state, f'32 surviving even-|k1| k2=0 Walsh modes',
          n_surv == 32,
          quantity='image(P) spanned by 32 quotient characters',
          measured=str(n_surv), threshold='32')
    check(state, f'complement 111111 has even weight: {comp_even}',
          comp_even,
          quantity='quotient GF(2)^6/<111111> has 32 characters',
          measured=str(comp_even), threshold='True')


def run(state):
    if not _KERNEL_OK:
        print('hqvm_group_analysis_4.py requires the hQVM kernel')
        return

    d = 6
    reps = k_reps(d)
    n_lin, n_2d, n_cls = clifford_irrep_counts(d)
    labels = [irrep_label(k, d) for k in reps]

    # 1. Swap-orbit census
    section(state, 'Swap-Orbit Census')
    n_reps = len(reps)
    fixed = sum(1 for k in range(1 << (2 * d)) if k == swap_halves(k, d))
    total_k = 1 << (2 * d)
    orbit_formula = (total_k - fixed) // 2
    vprint(f'  d = {d}')
    vprint(f'  |GF(2)^{{2d}}| = {total_k}')
    vprint(f'  swap-fixed k (diagonal): {fixed}')
    vprint(f'  k < swap(k) representatives: {n_reps}')
    vprint(f'  formula (2^{{2d}} - 2^d)/2 = {orbit_formula}')
    vprint(f'  clifford_irrep_counts: n_lin={n_lin}, n_2d={n_2d}, n_classes={n_cls}')
    check(state, f'|k_reps| = {n_reps} == 2016',
          n_reps == 2016 and n_reps == orbit_formula and n_reps == n_2d,
          quantity='Swap-orbit census: 2016 two-dim irrep labels k < swap(k)',
          measured=f'|k_reps|={n_reps}, formula={orbit_formula}, clifford n_2d={n_2d}',
          threshold='2016')

    # 2. Bidegree histogram
    section(state, 'Bidegree Histogram')
    meas_ord = Counter((lab['n_u'], lab['n_v']) for lab in labels)
    meas_sorted = Counter()
    for lab in labels:
        nu, nv = lab['n_u'], lab['n_v']
        meas_sorted[(min(nu, nv), max(nu, nv))] += 1

    vprint('  ordered (n_u, n_v) of canonical k < swap(k):')
    vprint(f'  {"n_u":>4} {"n_v":>4} {"count":>8}')
    for nu in range(d + 1):
        for nv in range(d + 1):
            c = meas_ord.get((nu, nv), 0)
            if c:
                vprint(f'  {nu:4d} {nv:4d} {c:8d}')
    vprint(f'  ordered sum = {sum(meas_ord.values())}')

    vprint('  sorted-bidegree orbit table vs theory:')
    vprint(f'  {"n_u":>4} {"n_v":>4} {"measured":>10} {"theory":>10} {"delta":>8}')
    theory_rows = _bidegree_theory_table(d)
    theory_sum = 0
    max_abs_delta = 0
    mismatches = 0
    for (nu, nv), th in theory_rows:
        m = meas_sorted.get((nu, nv), 0)
        delta = m - th
        theory_sum += th
        max_abs_delta = max(max_abs_delta, abs(delta))
        if delta != 0:
            mismatches += 1
        if th or m:
            vprint(f'  {nu:4d} {nv:4d} {m:10d} {th:10d} {delta:8d}')
    vprint(f'  theory sum = {theory_sum}; measured sorted sum = {sum(meas_sorted.values())}')

    comp_ok = True
    vprint('  complementary ordered bins (nu < nv): count(nu,nv)+count(nv,nu) vs C*C')
    vprint(f'  {"n_u":>4} {"n_v":>4} {"sum_ord":>10} {"C*C":>10}')
    for nu in range(d + 1):
        for nv in range(nu + 1, d + 1):
            th = comb(d, nu) * comb(d, nv)
            m_pair = meas_ord.get((nu, nv), 0) + meas_ord.get((nv, nu), 0)
            vprint(f'  {nu:4d} {nv:4d} {m_pair:10d} {th:10d}')
            if m_pair != th:
                comp_ok = False

    eq_ok = True
    vprint('  equal bidegree (nu == nv):')
    vprint(f'  {"n":>4} {"measured":>10} {"(C^2-C)/2":>12}')
    for nu in range(d + 1):
        th = (comb(d, nu) ** 2 - comb(d, nu)) // 2
        m = meas_ord.get((nu, nu), 0)
        vprint(f'  {nu:4d} {m:10d} {th:12d}')
        if m != th:
            eq_ok = False

    check(state, f'bidegree sorted table matches theory (mismatches={mismatches})',
          mismatches == 0 and theory_sum == 2016 and sum(meas_sorted.values()) == 2016,
          quantity='Bidegree orbit counts: sorted (n_u,n_v) match C*C / diagonal formula',
          measured=f'mismatches={mismatches}, theory_sum={theory_sum}, '
                   f'max|delta|={max_abs_delta}',
          threshold='0 mismatches; sum=2016')
    check(state, f'complementary ordered bins = C*C (ok={comp_ok})',
          comp_ok,
          quantity='When nu!=nv: count(nu,nv)+count(nv,nu) = C(6,nu)*C(6,nv)',
          measured=f'comp_ok={comp_ok}', threshold='True')
    check(state, f'equal bidegree orbits (ok={eq_ok})',
          eq_ok,
          quantity='When nu==nv: count = (C(6,nu)^2 - C(6,nu))/2',
          measured=f'eq_ok={eq_ok}', threshold='True')

    # 3. Character height on swap coset
    section(state, 'Character Height on Swap Coset')
    gl = list(kernel_group())
    N = len(gl)
    gl_even = [g for g in gl if ((g >> (2 * d)) & 1) == 0]
    gl_odd = [g for g in gl if ((g >> (2 * d)) & 1) == 1]
    vprint(f'  |G|={N}, |even|={len(gl_even)}, |odd|={len(gl_odd)}')

    odd_nonzero = 0
    sample_k = reps[:: max(1, len(reps) // 30)]
    sample_odd = gl_odd[:: max(1, len(gl_odd) // 20)]
    for k in sample_k:
        for g in sample_odd:
            if twod_char(k, g, d) != 0:
                odd_nonzero += 1
    check(state, f'2d chars vanish on odd coset (nonzero hits={odd_nonzero})',
          odd_nonzero == 0,
          quantity='Two-dim characters: chi_k(odd) = 0 on sampled (k,g)',
          measured=f'nonzero={odd_nonzero} over |k|={len(sample_k)} x |g|={len(sample_odd)}',
          threshold='0')

    rng = np.random.RandomState(20260822)
    global_take = list(rng.choice(reps, size=min(60, len(reps)), replace=False))
    l2_sample = global_take[:12]
    vprint('  character L2 norms on full G (sample):')
    vprint(f'  {"k":>8} {"||chi||^2":>14}')
    l2_bad = 0
    for k in l2_sample:
        nrm = _character_l2_norm(k, gl, d)
        vprint(f'  {k:8d} {nrm:14.10f}')
        if abs(nrm - 1.0) > 1e-9:
            l2_bad += 1
    check(state, f'||chi_k||^2 = 1 on {len(l2_sample)} reps (bad={l2_bad})',
          l2_bad == 0,
          quantity='Irrep character L2-normalization on full G',
          measured=f'violations={l2_bad}', threshold='0')

    vprint('  rho homomorphism sample errors:')
    vprint(f'  {"k":>8} {"errs":>8} {"trials":>8}')
    hom_gl = gl[::32]
    hom_total_err = 0
    hom_total_trial = 0
    for k in l2_sample[:5]:
        e, t = _rho_homomorphism_errors(k, hom_gl, d)
        vprint(f'  {k:8d} {e:8d} {t:8d}')
        hom_total_err += e
        hom_total_trial += t
    check(state, f'rho(gh)=rho(g)rho(h) errs={hom_total_err}/{hom_total_trial}',
          hom_total_err == 0 and hom_total_trial > 0,
          quantity='rho2 homomorphism on sampled products',
          measured=f'errs={hom_total_err}, trials={hom_total_trial}',
          threshold='0 errors')

    # 4. Peter-Weyl block ordering
    section(state, 'Peter-Weyl Block Ordering')
    ordered = sorted(labels, key=lambda L: (L['n_u'] + L['n_v'], L['n_u'], L['n_v'], L['k']))
    vprint(f'  sorted by (n_u+n_v, n_u, n_v, k); n={len(ordered)}')
    vprint('  first 12 blocks:')
    vprint(f'  {"idx":>5} {"k":>8} {"n_u":>4} {"n_v":>4} {"r":>4} {"pop":>4}')
    for i, L in enumerate(ordered[:12]):
        vprint(f'  {i:5d} {L["k"]:8d} {L["n_u"]:4d} {L["n_v"]:4d} '
              f'{L["n_u"]+L["n_v"]:4d} {L["pop"]:4d}')

    mono_ok = True
    prev = (-1, -1, -1, -1)
    for L in ordered:
        key = (L['n_u'] + L['n_v'], L['n_u'], L['n_v'], L['k'])
        if key < prev:
            mono_ok = False
            break
        prev = key
    check(state, f'sort key monotone: {mono_ok}',
          mono_ok and len(ordered) == 2016,
          quantity='Peter-Weyl block order: (n_u+n_v, n_u, n_v, k) monotone',
          measured=f'mono={mono_ok}, n={len(ordered)}', threshold='True; n=2016')

    pick_idxs = [0, 50, 200, 800, 1600, 2015]
    pick_idxs = [i for i in pick_idxs if i < len(ordered)]
    pick = [ordered[i]['k'] for i in pick_idxs]
    gl_sub = gl[::4]
    vprint(f'  matrix-coeff orthonormality on |G_sub|={len(gl_sub)} (stride-4)')
    vprint(f'  {"k1":>8} {"k2":>8} {"ij":>5} {"i\'j\'":>5} {"Re IP":>14} {"Im IP":>14}')
    max_off = 0.0
    max_diag_err = 0.0
    for a, k1 in enumerate(pick):
        for b, k2 in enumerate(pick):
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            if k1 != k2 and (i1, j1, i2, j2) != (0, 0, 0, 0):
                                continue
                            if k1 == k2 and (i1, j1) != (i2, j2) and (i1 + j1 + i2 + j2) > 0:
                                if (i1, j1, i2, j2) not in ((0, 0, 0, 1), (0, 0, 1, 0), (1, 1, 0, 0)):
                                    continue
                            ip = _matrix_coeff_ip(k1, i1, j1, k2, i2, j2, gl_sub, d)
                            if k1 == k2 and i1 == i2 and j1 == j2:
                                max_diag_err = max(max_diag_err, abs(ip.real - 0.5))
                            else:
                                max_off = max(max_off, abs(ip))
                            if (k1 == k2 and i1 == j1 == i2 == j2) or (k1 != k2 and i1 == j1 == i2 == j2 == 0):
                                vprint(f'  {k1:8d} {k2:8d} {i1}{j1:>2} {i2}{j2:>4} '
                                      f'{ip.real:14.6e} {ip.imag:14.6e}')

    vprint(f'  max |diag IP - 1/2| = {max_diag_err:.6e}')
    vprint(f'  max |off-diag IP|   = {max_off:.6e}')
    pw_ok = max_diag_err < 0.08 and max_off < 0.08
    check(state, f'PW sample orthonormality diag_err={max_diag_err:.4e} off={max_off:.4e}',
          pw_ok,
          quantity='Peter-Weyl sample: matrix coeffs ~ orthonormal (subsample G)',
          measured=f'max|diag-1/2|={max_diag_err:.6e}, max|off|={max_off:.6e}',
          threshold='both < 0.08')

    rng = np.random.RandomState(20260822)
    gl = list(kernel_group())
    N = len(gl)
    as_s1 = enumerate_s1_labels()

    # 1. Register model
    section(state, 'Register Model')
    vprint('  pack_register32(intron8, state24); step_register32(byte, reg)')
    vprint(f'  GENE_MAC_REST = {GENE_MAC_REST & 0xFFFFFF}')
    bad_rt = register_pack_roundtrip_table(40, rng)
    check(state, f'pack/unpack/shadow roundtrip bad={bad_rt}',
          bad_rt == 0,
          quantity='pack_register32 / unpack / shadow_register32 roundtrip',
          measured=f'bad={bad_rt}/40', threshold='0')

    bad_step = step_matches_kernel_table(80, rng)
    check(state, f'step_register32 Mac == step_state_by_byte bad={bad_step}',
          bad_step == 0,
          quantity='Register Mac update matches step_state_by_byte',
          measured=f'bad={bad_step}/80', threshold='0')

    # 2. Shadow map / family-phase
    section(state, 'Shadow Map')
    buckets, sizes, partner_closed = shadow_fiber_census(GENE_MAC_REST & 0xFFFFFF)
    n_img = len(buckets)
    fiber2 = sizes.get(2, 0)
    vprint(f'  distinct Mac images={n_img}, size-2 fibers={fiber2}, '
          f'partner_closed={partner_closed}')
    check(state, f'distinct Mac images={n_img}, size-2 fibers={fiber2}',
          n_img == 128 and fiber2 == 128 and partner_closed == 128,
          quantity='Family-phase collapse: 128 Mac images, fibers size 2 (partners)',
          measured=f'images={n_img}, fiber2={fiber2}, partner_closed={partner_closed}',
          threshold='128 / 128 / 128')

    # 3. Group order lock
    section(state, 'Group Before Shadow')
    vprint(f'  |G| = {N} (expect 8192)')
    check(state, f'|kernel_group|={N}',
          N == 8192,
          quantity='|G| = 8192 = 2^13',
          measured=f'|G|={N}', threshold='8192')

    # 4. Recover s=1 linear characters
    section(state, 'Recover s=1 Linear Characters')
    vprint(f'  carrier = {{0,1}}_sigma x Omega; dim = {carrier_dim()}')
    vprint(f'  |s=1 labels a| = {len(as_s1)} (expect 64)')
    check(state, f'|s=1 labels|={len(as_s1)}',
          len(as_s1) == 64 and as_s1 == list(range(64)),
          quantity='64 parity-odd linear labels a in GF(2)^6',
          measured=f'n={len(as_s1)}, range0_63={as_s1==list(range(64))}',
          threshold='64')

    total_bad, total_n = sample_transform_table(as_s1, gl, rng, n_a=8, n_g=10, n_pts=25)
    check(state, f'transform law bad={total_bad}/{total_n}',
          total_bad == 0,
          quantity='g.psi_a = linear_char(1,a,g) * psi_a on sampled (a,g,x)',
          measured=f'bad={total_bad}/{total_n}', threshold='0')

    # All 64: transform sample over every a
    all64_bad = 0
    g_samp = [int(x) for x in rng.choice(gl, size=20, replace=False)]
    for a in as_s1:
        for g in g_samp:
            all64_bad += transform_check_psi(a, g, 4, rng)
    check(state, f'all 64 psi_a transform bad={all64_bad}',
          all64_bad == 0,
          quantity='All 64 psi_a transform as chi_{1,a} (20 g x 4 pts)',
          measured=f'bad={all64_bad}', threshold='0')

    # Gram: closed-form check on sample + fast diagonal identity
    gram_sample = list(rng.choice(as_s1, size=8, replace=False))
    gram_err = verify_gram_closed(gram_sample)
    vprint(f'  Gram closed-form max err on {len(gram_sample)}x{len(gram_sample)} = {gram_err:.3e}')
    Gfast = spinor_gram_matrix_fast(as_s1)
    check(state, 'Gram = 8192 * I_64',
          gram_err < 1e-9 and np.allclose(Gfast, np.eye(64) * carrier_dim()),
          quantity='Spinor Gram matrix equals 8192 * I_64',
          measured=f'sample_err={gram_err:.3e}, diag0={Gfast[0,0]}',
          threshold='8192 * I')

    # 5. Spinor sector isolation
    section(state, 'Spinor Sector Isolation')
    orth = omega_embed_orthogonality(as_s1[::8], n_f=2, rng=rng)
    vprint(f'  max |<psi_a, iota(f)>| on sample = {orth:.3e}')
    check(state, f'orthogonal to Omega-embed (max={orth:.3e})',
          orth < 1e-8,
          quantity='Spinor psi_a orthogonal to Omega-embedded functions',
          measured=f'max_abs_ip={orth:.3e}', threshold='< 1e-8')

    # 6. Phase recovery
    section(state, 'Phase Recovery')
    ph = phase_pair_distinguishability(200, rng)
    vprint(f'  shadow-partner pairs = {ph["n_pairs"]}')
    vprint(f'  same Mac from rest = {ph["same_mac_from_rest"]}/{ph["n_pairs"]}')
    vprint(f'  diff intron, same Mac = {ph["diff_intron_same_mac"]}/{ph["n_pairs"]}')
    check(state, 'partners: same Mac, different high-8',
          ph['same_mac_from_rest'] == ph['n_pairs']
          and ph['diff_intron_same_mac'] == ph['n_pairs']
          and ph['same_mac_rand'] == ph['n_mac'],
          quantity='Shadow partners: identical Mac, distinct register high-8',
          measured=(f'same_mac={ph["same_mac_from_rest"]}, '
                    f'diff_intron={ph["diff_intron_same_mac"]}, '
                    f'rand={ph["same_mac_rand"]}/{ph["n_mac"]}'),
          threshold='all pairs')

    start_reg = pack_register32(0, GENE_MAC_REST & 0xFFFFFF)
    d4 = depth4_intron_uniqueness(400, rng, start_reg)
    vprint(f'  depth-4 frames = {d4["n_frames"]}')
    vprint(f'  unique intron-seq={d4["intron_keys"]} mac-seq={d4["mac_keys"]}')
    vprint(f'  partner mac_match={d4["partner_mac_match"]}/{d4["partner_trials"]} '
          f'intron_diff={d4["partner_intron_diff"]}')
    check(state, 'depth-4 intron sequences refine Mac; partners collapse Mac only',
          d4['intron_keys'] >= d4['mac_keys']
          and d4['mac_collisions'] >= d4['intron_collisions']
          and d4['partner_mac_match'] == d4['partner_trials']
          and d4['partner_intron_diff'] == d4['partner_trials'],
          quantity='Depth-4: intron path refines Mac path; partners collapse Mac only',
          measured=(f'intron_keys={d4["intron_keys"]}, mac_keys={d4["mac_keys"]}, '
                    f'intron_col={d4["intron_collisions"]}, mac_col={d4["mac_collisions"]}, '
                    f'partner_mac={d4["partner_mac_match"]}, '
                    f'partner_intron_diff={d4["partner_intron_diff"]}'),
          threshold='intron>=mac uniqueness; partner Mac match & intron diff')

    rng = np.random.RandomState(20260822)

    # 1. G_d census
    section(state, 'G_d Census')
    rows = g_d_structure_table(DS_ALL)
    print_census_table(rows)

    order_ok = True
    measured_orders = {}
    for d in DS_ALL:
        G = kernel_group_d(d)
        nG, nTh = len(G), order_g_d(d)
        measured_orders[d] = (nG, nTh, G)
        vprint(f'  d={d}: |G_meas|={nG} theory={nTh} step_set={len(byte_step_set(d))}')
        if nG != nTh:
            order_ok = False

    check(state, f'|G_d|=2^(2d+1) for d in {DS_ALL}',
          order_ok and all(measured_orders[d][0] == measured_orders[d][1] for d in DS_ALL),
          quantity='|G_d| = 2^{2d+1} matches kernel_group_d for all listed d',
          measured=str({d: measured_orders[d][0] for d in DS_ALL}),
          threshold=str({d: order_g_d(d) for d in DS_ALL}))

    for d in (3, 4, 5):
        classes, G = conjugacy_classes_d(d)
        _, _, n_cls = clifford_irrep_counts(d)
        print_class_size_histogram(classes, d)
        check(state, f'd={d}: n_classes={len(classes)} vs theory {n_cls}',
              len(classes) == n_cls and sum(sz for _, sz in classes) == len(G),
              quantity=f'Conjugacy class count at d={d} equals clifford n_classes',
              measured=f'n_classes={len(classes)}, |G|={len(G)}, '
                       f'sum_sizes={sum(sz for _, sz in classes)}',
              threshold=f'n_classes={n_cls}')

    # Alphabet odd signature: one gate covering all d
    sig_ok = True
    sig_meas = {}
    for d in DS_ALL:
        odd, dist, A = alphabet_signature_parity_table(d)
        sig_meas[d] = (odd, dist, A)
        if not (odd == A and dist == (A // 2)):
            sig_ok = False
    vprint(f'  alphabet signature (odd, distinct, |A|): {sig_meas}')
    check(state, f'alphabet odd signatures; |step_set|=|A|/2 for d in {DS_ALL}',
          sig_ok,
          quantity='All byte signatures odd; distinct step-set size |A|/2',
          measured=str(sig_meas),
          threshold='odd=|A|; distinct=|A|/2 for each d')

    # 2. Multiplicity-free L2(Omega_d)
    section(state, 'Multiplicity-Free L2(Omega_d)')
    for d in (3, 4):
        gl = list(measured_orders[d][2])
        rows_lin, rows_2d, n_lin, n_2d = multiplicity_free_report(d, gl)
        print_mult_table(d, rows_lin, rows_2d)
        s0_ok = all(abs(ip - 1.0) < 1e-9 for s, a, ip in rows_lin if s == 0)
        s1_ok = all(abs(ip - 0.0) < 1e-9 for s, a, ip in rows_lin if s == 1)
        t2_ok = all(abs(ip - 1.0) < 1e-9 for _, ip in rows_2d)
        check(state, f'd={d} mult-free sample s0={s0_ok} s1={s1_ok} 2d={t2_ok}',
              s0_ok and s1_ok and t2_ok,
              quantity=f'L2(Omega_{d}): <chi_perm,lin_s0>=1, <.,lin_s1>=0, <.,2d>=1',
              measured=f'n_lin_checked={len(rows_lin)}, n_2d_checked={len(rows_2d)}',
              threshold='s0:1 / s1:0 / 2d:1')

    # Thin sample for d=5,6
    for d, max_lin, max_2d in ((5, 24, 24), (6, 16, 16)):
        gl = list(measured_orders[d][2])
        rows_lin, rows_2d, _, _ = multiplicity_free_report(
            d, gl, max_lin=max_lin, max_2d=max_2d, rng=rng)
        print_mult_table(d, rows_lin, rows_2d)
        s0_ok = all(abs(ip - 1.0) < 1e-9 for s, a, ip in rows_lin if s == 0)
        s1_ok = all(abs(ip - 0.0) < 1e-9 for s, a, ip in rows_lin if s == 1)
        t2_ok = all(abs(ip - 1.0) < 1e-9 for _, ip in rows_2d)
        check(state, f'd={d} mult-free thin sample s0={s0_ok} s1={s1_ok} 2d={t2_ok}',
              s0_ok and s1_ok and t2_ok,
              quantity=f'L2(Omega_{d}) thin sample: s0=1, s1=0, 2d=1',
              measured=f'n_lin={len(rows_lin)}, n_2d={len(rows_2d)}',
              threshold='s0:1 / s1:0 / 2d:1')

    # 3. Byte-walk P_d
    section(state, 'Byte-Walk P_d')
    rank_results = []
    for d in DS_FULL:
        vprint(f'  computing rank(P) for d={d} ...')
        r = rank_of_P(d)
        rank_results.append(r)
        eng = build_hqvm_d(d)
        bad = 0
        omega = enumerate_omega_d(d)
        for i in list(range(0, len(omega), max(1, len(omega) // 20)))[:20]:
            u, v = omega[i]
            for b in range(0, alphabet_size(d), max(1, alphabet_size(d) // 8)):
                nu, nv = step_uv(u, v, b, d)
                j = eng.uv_to_idx[(nu, nv)]
                if eng.transitions[i][b] != j:
                    bad += 1
        if bad != 0:
            vprint(f'  d={d}: build_hqvm_d vs step_uv mismatches={bad}')

    print_rank_table(rank_results)

    rank_p2_ok = True
    for r in rank_results:
        if not (r['rank_P2'] == 1 and r['mix_err'] < 1e-8):
            rank_p2_ok = False
    check(state, f'rank(P^2)=1 and mix for d in {DS_FULL}',
          rank_p2_ok,
          quantity='Byte walk d<=6: rank(P^2)=1 and P^2 ~ J/|Omega|',
          measured=str({r['d']: (r['rank_P'], r['rank_P2'], r['mix_err'])
                        for r in rank_results}),
          threshold='rank_P2=1; mix_err<1e-8')

    # 4. Transient subspace dim
    section(state, 'Transient Subspace Dim')
    ds_fit = [r['d'] for r in rank_results]
    ranks_fit = [r['rank_P'] for r in rank_results]
    print(f'  {"d":>3} {"rank_P":>8} {"theory 2^(d-1)":>14} {"log2(r)/(d-1)":>14}')
    for d, rk in zip(ds_fit, ranks_fit):
        theory = 1 << (d - 1)
        ratio = math.log2(rk) / (d - 1) if d > 1 and rk > 0 else float('nan')
        print(f'  {d:3d} {rk:8d} {theory:14d} {ratio:14.6f}')

    a_fit, b_fit, max_resid, fit_detail = fit_rank_log_model(ds_fit, ranks_fit)
    vprint(f'  LS fit log2(rank) ~= {a_fit:.6f}*(d-1) + {b_fit:.6f}; '
          f'max|resid|={max_resid:.3e}')
    for d, rk, y, pred, res in fit_detail:
        vprint(f'    d={d}: rank={rk} log2={y:.6f} pred={pred:.6f} resid={res:.3e}')
    theory_match = all(rk == (1 << (d - 1)) for d, rk in zip(ds_fit, ranks_fit))
    check(state, f'rank(d)=2^(d-1); log2 LS max|resid|={max_resid:.3e}',
          theory_match and max_resid < 1e-9,
          quantity='Measured rank(P_d) = 2^{d-1}; exact affine log2 fit',
          measured=f'ranks={dict(zip(ds_fit, ranks_fit))}, a={a_fit:.6f}, b={b_fit:.6f}',
          threshold='rank=2^(d-1); max|resid|<1e-9')

    q_ok = True
    q_meas = {}
    for d in DS_ALL:
        rq = q_span_rank(d)
        q_meas[d] = rq
        if rq != d:
            q_ok = False
    vprint(f'  q-span ranks: {q_meas}')
    check(state, f'q-span rank = d for d in {DS_ALL}',
          q_ok,
          quantity='GF(2)-rank of {q_d(b)} equals d',
          measured=str(q_meas), threshold=str({d: d for d in DS_ALL}))

    # 5. d=6 lock (one bundled gate)
    section(state, 'd=6 Lock')
    tgt = d6_lock_targets()
    G = measured_orders[6][2]
    n_lin, n_2d, n_cls = clifford_irrep_counts(6)
    r6 = next(r for r in rank_results if r['d'] == 6)
    ss = len(byte_step_set(6))
    measured = {
        '|G|': len(G),
        '|Omega|': order_omega_d(6),
        '|A|': alphabet_size(6),
        'n_lin': n_lin,
        'n_2d': n_2d,
        'n_classes': n_cls,
        'rank_P': r6['rank_P'],
        'rank_P2': r6['rank_P2'],
        'step_set': ss,
    }
    vprint(f'  {"qty":>16} {"measured":>12} {"target":>12} {"ok":>6}')
    all_lock = True
    for name in tgt:
        ok = measured[name] == tgt[name]
        all_lock = all_lock and ok
        vprint(f'  {name:>16} {measured[name]:12d} {tgt[name]:12d} {str(ok):>6}')
    check(state, f'd=6 lock all targets ok={all_lock}',
          all_lock,
          quantity='d=6 lock: 8192, 4096, 256, 128+2016, rank 32/1, step_set 128',
          measured=str(measured),
          threshold=str(tgt))

    section(state, 'Special 2-Group Certificate')
    _special2_gates(state, list(kernel_group()))
    section(state, 'D8 Fiber Product')
    _d8_fiber_gates(state)
    section(state, 'Transient Quotient GF(2)^6/<111111>')
    _quotient32_gates(state)


if __name__ == '__main__':
    st = ReportState()
    run(st)
    passed = sum(1 for _, ok in st.gates if ok)
    failed = sum(1 for _, ok in st.gates if not ok)
    print(f'\nSUMMARY: {passed} passed, {failed} failed out of {len(st.gates)}')
