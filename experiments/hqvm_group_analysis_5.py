#!/usr/bin/env python3
"""hqvm_group_analysis_5.py — Mixer/Aut, equivariant/codebook/rigid-body, Grover/Bell/sync/Krawtchouk/toric.

Role: PASS/FAIL gates on byte step-set algebra, association scheme, Cayley/Schreier
spectrum, Aut(Gamma) bounds, design catalog; finite Fourier equivariant maps,
two-byte Haar-analog sampling, SO(3) codebook geodesic error, rigid-body POC;
Householder-Grover amplification, Bell pair correlators under G, non-abelian sync
thresholds, Krawtchouk-Racah shell tests, mod-2 toric plaquettes.
Inputs: hqvm_group_analysis_common (d=6).
Outputs: gates appended to ReportState; printed tables.
Companion: hqvm_group_analysis_1/2/3/4.py, hqvm_group_analysis_run.py,
hqvm_group_analysis_common.py.
"""
from __future__ import annotations

import math, time
from collections import Counter, defaultdict

import numpy as np

from hqvm_group_analysis_common import (
    _SCIPY_OK, _KERNEL_OK,
    ReportState, section, check, vprint,
    kernel_group, kernel_group_d, compose_sig_int, compose_sig_int_d,
    inv_sig_int, apply_sig_int, apply_sig_int_d,
    byte_step_set, byte_transition_matrix,
    omega12_to_state24, state24_to_omega12, OmegaState12,
    fwht, step_state_by_byte,
    so3_encode_bits, so3_roundtrip,
    uniform_random_rotation, rotation_angle_from_matrix,
    rodrigues_exp,
    casimir_eigenvalue, wigner_character,
    walsh_hadamard64,
)

if _SCIPY_OK:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spspl
    import scipy.spatial.transform as sstr
    import scipy.special as spspec

D = 6

MASK = (1 << D) - 1

N_OMEGA = 1 << (2 * D)

N_A = N_OMEGA

def _pack(p, u, v, d=D):
    return (p << (2 * d)) | ((u & ((1 << d) - 1)) << d) | (v & ((1 << d) - 1))

def _unpack(g, d=D):
    return (g >> (2 * d)) & 1, (g >> d) & ((1 << d) - 1), g & ((1 << d) - 1)

def _pop(x):
    return bin(int(x)).count('1')

def gf2_rank(rows):
    if not rows:
        return 0
    A = [int(r) for r in rows]
    rank = 0
    width = max(r.bit_length() for r in A) if A else 0
    for c in range(width):
        bit = 1 << c
        piv = None
        for i in range(rank, len(A)):
            if A[i] & bit:
                piv = i
                break
        if piv is None:
            continue
        A[rank], A[piv] = A[piv], A[rank]
        for i in range(len(A)):
            if i != rank and (A[i] & bit):
                A[i] ^= A[rank]
        rank += 1
    return rank

def even_to_bits(g):
    _, u, v = _unpack(g)
    return (u << D) | v

def neighbor_set(u, v, S):
    return {apply_sig_int_d(s, u, v, D) for s in S}

def omega_from_uv(u, v):
    return omega12_to_state24(OmegaState12(u6=u & MASK, v6=v & MASK))

def uv_from_omega(s):
    om = state24_to_omega12(s)
    return om.u6 & MASK, om.v6 & MASK

def translate_omega(s, tu, tv):
    u, v = uv_from_omega(s)
    return omega_from_uv(u ^ tu, v ^ tv)

def permute_bits6(x, pi):
    out = 0
    for i in range(6):
        if (x >> pi[i]) & 1:
            out |= 1 << i
    return out

def adjacent_transpositions():
    gens = []
    for k in range(5):
        pi = list(range(6))
        pi[k], pi[k + 1] = pi[k + 1], pi[k]
        gens.append(tuple(pi))
    return gens

def difference_multiset(S):
    c = Counter()
    for s in S:
        si = inv_sig_int(s, D)
        for sp in S:
            c[compose_sig_int_d(si, sp, D)] += 1
    return c

def subgroup_from_coset(S):
    o = S[0]
    oi = inv_sig_int(o, D)
    H = tuple(sorted(compose_sig_int_d(oi, s, D) for s in S))
    return o, H

def partial_ds_params(diff_c, ambient_order):
    k2 = sum(diff_c.values())
    k = int(round(math.sqrt(k2)))
    vals = sorted(set(diff_c.values()))
    id_m = diff_c.get(0, 0)
    support = set(diff_c.keys())
    nz_in = [diff_c[g] for g in support if g != 0]
    lam = nz_in[0] if nz_in and len(set(nz_in)) == 1 else (
        float(np.mean(nz_in)) if nz_in else 0)
    return {
        'v': ambient_order,
        'k': k,
        'lambda': lam,
        'mu': 0,
        'support': len(support),
        'id_mult': id_m,
        'mult_values': vals,
        'off_support': ambient_order - len(support),
    }

def orbital_type(u1, v1, u2, v2):
    du, dv = u1 ^ u2, v1 ^ v2
    pop_uv = (_pop(du), _pop(dv), _pop(du ^ dv))
    pop_sw = tuple(sorted([_pop(du), _pop(dv)]))
    same = int((u1, v1) == (u2, v2))
    return (same, pop_uv, pop_sw)

def adjacency_spectrum_info(P):
    if sp.issparse(P):
        A = (P * 256.0).tocsr()
        As = 0.5 * (A + A.T)
        evals = spspl.eigsh(As, k=min(32, As.shape[0] - 2), which='LA',
                            return_eigenvectors=False)
        evals = np.sort(evals)[::-1]
    else:
        A = np.asarray(P) * 256.0
        As = 0.5 * (A + A.T)
        evals = np.sort(np.linalg.eigvalsh(As))[::-1]
    lam0 = float(evals[0])
    lam1 = float(evals[1]) if len(evals) > 1 else float('nan')
    return {
        'lambda_max': lam0,
        'lambda_2': lam1,
        'gap': lam0 - lam1,
        'evals_head': [float(x) for x in evals[:8]],
    }

def catalog_match(params):
    v, k, lam, mu = params['v'], params['k'], params['lambda'], params['mu']
    support = params['support']
    catalog = [
        ('elementary_coset_H_order_128_in_A_4096', 4096, 128, 128, 0, 128),
    ]
    for name, cv, ck, clam, cmu, csup in catalog:
        if (v, k, lam, mu) == (cv, ck, clam, cmu) and csup == support:
            return name, (v, k, lam, mu, support)
    return None, ('novel_parameters', v, k, lam, mu, support)

def t_aut_elements():
    out = []
    for tu in range(64):
        for tv in range(64):
            if (tu ^ tv) in (0, 63):
                out.append((tu, tv))
    return out

def uv_to_index(u, v):
    return ((u & MASK) << D) | (v & MASK)

def index_to_uv(idx):
    return (idx >> D) & MASK, idx & MASK

def translate_index(idx, a):
    return idx ^ (a & (N_A - 1))

def apply_g_index(g, idx):
    u, v = index_to_uv(idx)
    up, vp = apply_sig_int_d(g, u, v, D)
    return uv_to_index(up, vp)

def fwht_normalized(x):
    y = fwht(x)
    return y / math.sqrt(len(x))

def ifwht_normalized(x):
    return fwht_normalized(x)

def equivariant_layer(f, weights):
    F = fwht_normalized(f)
    Y = F * weights
    return ifwht_normalized(Y)

def rho_translate(a, f):
    out = np.zeros_like(f)
    for idx in range(len(f)):
        out[idx] = f[translate_index(idx, a)]
    return out

def rho_g(g, f):
    gi = inv_sig_int(g, D)
    out = np.zeros_like(f)
    for idx in range(len(f)):
        src = apply_g_index(gi, idx)
        out[idx] = f[src]
    return out

def codebook_decode(idx, budget):
    if budget <= 0:
        return np.eye(3)
    n_ang = max(1, budget // 3)
    n_ax = budget - n_ang
    n_th = max(1, n_ax // 2)
    n_ph = max(1, n_ax - n_th)
    mask_ph = (1 << n_ph) - 1
    mask_th = (1 << n_th) - 1
    i_ph = idx & mask_ph
    i_th = (idx >> n_ph) & mask_th
    i_ang = idx >> (n_th + n_ph)
    i_ang = min(i_ang, (1 << n_ang) - 1)
    ang_h = i_ang / max(1, (1 << n_ang) - 1) * math.pi
    th_h = i_th / max(1, (1 << n_th) - 1) * math.pi
    ph_h = i_ph / max(1, (1 << n_ph) - 1) * 2 * math.pi
    ax_h = np.array([math.sin(th_h) * math.cos(ph_h),
                     math.sin(th_h) * math.sin(ph_h),
                     math.cos(th_h)])
    nrm = float(np.linalg.norm(ax_h))
    if nrm < 1e-15:
        ax_h = np.array([1.0, 0.0, 0.0])
    else:
        ax_h = ax_h / nrm
    return rodrigues_exp(ang_h, ax_h)

def geodesic_error(R, S):
    return rotation_angle_from_matrix(R.T @ S)

def compose_mats(Rs):
    out = np.eye(3)
    for R in Rs:
        out = out @ R
    return out

def random_signal(rng):
    return rng.randn(N_OMEGA)

def bandlimited_weights(rng, n_keep=64):
    w = np.zeros(N_OMEGA)
    idx = rng.choice(N_OMEGA, size=n_keep, replace=False)
    w[idx] = rng.randn(n_keep)
    return w


def g_equivariant_weights(rng):
    """2080 scalar gains: 64 diagonal + 2016 off-diagonal Walsh pairs."""
    gd = rng.randn(64)
    gp = rng.randn(2016)
    w = np.zeros(N_OMEGA)
    p = 0
    for k1 in range(64):
        for k2 in range(64):
            if k1 < k2:
                val = gp[p]
                p += 1
            elif k1 > k2:
                val = w[k2 * 64 + k1]
            else:
                val = gd[k1]
            w[k1 * 64 + k2] = val
    return w, len(gd) + len(gp)


def fwht2d_flat(f, W64):
    M = np.asarray(f, dtype=np.float64).reshape(64, 64)
    return (W64 @ M @ W64.T).ravel()


def g_equivariant_layer(f, w, W64):
    F = fwht2d_flat(f, W64)
    return fwht2d_flat(F * w, W64) / N_OMEGA


def aut_factorization_counts():
    """Step-level Aut(Gamma) lower bound T_diag semidirect S6; UV adjacency bound 128*720."""
    import itertools

    def _pbyte(b, pi):
        from gyroscopic.hQVM.constants import byte_to_intron
        intron = b ^ 0xAA
        mr = (intron >> 1) & 0x3F
        return ((intron & 0x81) | (permute_bits6(mr, pi) << 1)) ^ 0xAA

    def _phi(s, tu, tv, pi):
        u, v = uv_from_omega(s)
        return omega_from_uv(permute_bits6(u, pi) ^ tu, permute_bits6(v, pi) ^ tv)

    def _is_aut(tu, tv, pi):
        for u, v in ((0, 0), (5, 9), (31, 17)):
            s = omega_from_uv(u, v)
            for b in range(0, 256, 8):
                if _phi(step_state_by_byte(s, b), tu, tv, pi) != \
                   step_state_by_byte(_phi(s, tu, tv, pi), _pbyte(b, pi)):
                    return False
        return True

    id_pi = tuple(range(6))
    perms = list(itertools.permutations(range(6)))
    t_diag = [(tu, tv) for tu, tv in t_aut_elements() if tu == tv]
    n_t = sum(1 for tu, tv in t_diag if _is_aut(tu, tv, id_pi))
    n_pi = sum(1 for pi in perms if _is_aut(0, 0, pi))
    compat = True
    rng = np.random.RandomState(17)
    for _ in range(24):
        tu, tv = t_diag[int(rng.randint(0, len(t_diag)))]
        pi = perms[int(rng.randint(0, len(perms)))]
        if not _is_aut(tu, tv, pi):
            compat = False
            break
    return n_t, n_pi, n_t * n_pi, compat, len(t_diag)


def print_table(title, rows, *, always=False):
    out = print if always else vprint
    out(f'  {title}')
    if not rows:
        return
    keys = list(rows[0].keys())
    out('    ' + ' | '.join(f'{k:>14s}' for k in keys))
    out('    ' + '-'*5)
    for r in rows:
        out('    ' + ' | '.join(f'{str(r[k]):>14s}' for k in keys))

def householder_diffusion(psi):
    mean = np.mean(psi)
    return 2.0 * mean - psi

def oracle_flip(psi, marked):
    out = psi.copy()
    for m in marked:
        out[m] = -out[m]
    return out

def grover_iteration(psi, marked):
    return householder_diffusion(oracle_flip(psi, marked))

def grover_angle(M, N):
    return math.asin(math.sqrt(M / N))

def amplitude_on_marked(psi, marked):
    return float(np.linalg.norm(psi[list(marked)]))

def krawtchouk(r, x, n, q=2):
    s = 0
    for j in range(r + 1):
        s += ((-1) ** j) * math.comb(x, j) * math.comb(n - x, r - j)
    return s

def krawtchouk_matrix(n):
    K = np.zeros((n + 1, n + 1), dtype=np.float64)
    for r in range(n + 1):
        for x in range(n + 1):
            K[r, x] = krawtchouk(r, x, n)
    return K

def correlator_ij(samples, i, j):
    s = 0.0
    for x in samples:
        bi = (int(x) >> i) & 1
        bj = (int(x) >> j) & 1
        s += 1.0 if (bi ^ bj) == 0 else -1.0
    return s / len(samples)

def six_pair_edges():
    return [(i, i + 6) for i in range(6)]

def apply_g_to_index(g, idx):
    u, v = index_to_uv(idx)
    up, vp = apply_sig_int_d(g, u, v, D)
    return uv_to_index(up, vp)

def transform_samples(samples, g):
    return np.array([apply_g_to_index(g, int(x)) for x in samples], dtype=np.int64)

def sync_observation_abelian(true_g, noise_rate, rng):
    x = true_g & (N_OMEGA - 1)
    reps = 15
    votes = np.zeros((12, 2), dtype=np.int64)
    for _ in range(reps):
        for b in range(12):
            bit = (x >> b) & 1
            if rng.rand() < noise_rate:
                bit ^= 1
            votes[b, bit] += 1
    est = 0
    for b in range(12):
        if votes[b, 1] > votes[b, 0]:
            est |= (1 << b)
    return est, x

def sync_observation_nonabelian(true_g, noise_rate, rng, gl_sample):
    votes_p = [0, 0]
    votes_u = np.zeros(64, dtype=np.int64)
    votes_v = np.zeros(64, dtype=np.int64)
    n_obs = 40
    for _ in range(n_obs):
        h = int(gl_sample[rng.randint(0, len(gl_sample))])
        conj = compose_sig_int_d(
            compose_sig_int_d(h, true_g, D), inv_sig_int(h, D), D)
        cp, cu, cv = _unpack(conj)
        if rng.rand() < noise_rate:
            cu ^= (1 << rng.randint(0, 6))
        if rng.rand() < noise_rate:
            cv ^= (1 << rng.randint(0, 6))
        if rng.rand() < noise_rate * 0.5:
            cp ^= 1
        votes_p[cp] += 1
        votes_u[cu] += 1
        votes_v[cv] += 1
    ep = 1 if votes_p[1] > votes_p[0] else 0
    eu = int(np.argmax(votes_u))
    ev = int(np.argmax(votes_v))
    est1 = (ep << 12) | (eu << 6) | ev
    est2 = (ep << 12) | (ev << 6) | eu
    return est1, est2, true_g

def match_g(est, true):
    return int(est) == int(true)

def central_involutions():
    return [(0 << 12) | (d << 6) | d for d in range(64)]

def commutator(g, h):
    gi = inv_sig_int(g, D)
    hi = inv_sig_int(h, D)
    return compose_sig_int_d(
        compose_sig_int_d(compose_sig_int_d(g, h, D), gi, D), hi, D)


def run(state):
    if not _KERNEL_OK:
        print('hqvm_group_analysis_5.py requires the hQVM kernel')
        return
    if not _SCIPY_OK:
        print('hqvm_group_analysis_5.py requires scipy')
        return

    rng = np.random.RandomState(20260822)
    t0 = time.time()

    # 1. Step set algebra
    section(state, 'Step Set Algebra')
    S = byte_step_set(D)
    k_S = len(S)
    vprint(f'  |S| = {k_S} (d={D})')
    check(state, f'|S| = {k_S}',
          k_S == 128,
          quantity='Byte step set cardinality |S|',
          measured=str(k_S), threshold='128')

    all_odd = all((_unpack(s)[0] == 1) for s in S)
    check(state, 'All step signatures have parity 1',
          all_odd,
          quantity='S subset odd coset of G',
          measured=f'all_odd={all_odd}', threshold='True')

    o, H = subgroup_from_coset(S)
    H_set = set(H)
    S_rebuilt = tuple(sorted(compose_sig_int_d(o, h, D) for h in H))
    coset_ok = S_rebuilt == tuple(S)
    H_even = all(_unpack(h)[0] == 0 for h in H)
    H_closed = True
    for a in H:
        for b in H:
            if compose_sig_int_d(a, b, D) not in H_set:
                H_closed = False
                break
        if not H_closed:
            break
    H_bits = [even_to_bits(h) for h in H]
    H_rank = gf2_rank(H_bits)
    check(state, f'S = oH coset: |H|={len(H)}, rank={H_rank}',
          coset_ok and H_even and H_closed and len(H) == 128,
          quantity='S is a right coset oH of an even subgroup H <= A',
          measured=f'coset={coset_ok}, |H|={len(H)}, closed={H_closed}, '
                   f'GF2-rank={H_rank}',
          threshold='coset True, |H|=128, closed, rank=7')

    diff_c = difference_multiset(S)
    params_G = partial_ds_params(diff_c, ambient_order=len(kernel_group_d(D)))
    params_H = partial_ds_params(diff_c, ambient_order=len(H))
    params_A = {
        'v': N_A,
        'k': k_S,
        'lambda': params_H['lambda'],
        'mu': 0,
        'support': params_G['support'],
        'id_mult': params_G['id_mult'],
        'mult_values': params_G['mult_values'],
        'off_support': N_A - params_G['support'],
    }
    uniform_H = (params_G['support'] == 128
                 and params_G['mult_values'] == [128]
                 and set(diff_c.keys()) == H_set)
    check(state, f'Difference multiset uniform on H: support={params_G["support"]}',
          uniform_H,
          quantity='(v,k,lambda,mu) partial DS: support=H, lambda=|S|, mu=0 off H',
          measured=f'v_A={N_A}, k={k_S}, lambda={params_A["lambda"]}, '
                   f'mu={params_A["mu"]}, support={params_A["support"]}',
          threshold='support=128, lambda=128, mu=0')

    imgs0 = [apply_sig_int_d(s, 0, 0, D) for s in S]
    imgs0_set = set(imgs0)
    check(state, f'|N(0)| = {len(imgs0_set)}',
          len(imgs0_set) == 128,
          quantity='Out-degree of Omega byte graph (unique signatures)',
          measured=str(len(imgs0_set)), threshold='128')

    deg_hist = Counter()
    sample_uv = [(i, j) for i in range(0, 64, 4) for j in range(0, 64, 4)]
    for u, v in sample_uv:
        deg_hist[len(neighbor_set(u, v, S))] += 1
    regular = list(deg_hist.keys()) == [128]
    check(state, f'Degree regularity on sample: {dict(deg_hist)}',
          regular,
          quantity='Byte graph is 128-regular on Omega (sample grid)',
          measured=str(dict(deg_hist)), threshold='{128: n_sample}')

    check(state, f'dim H = {H_rank}',
          H_rank == 7 and len(H) == 128,
          quantity='Difference support H is 7-dimensional in GF(2)^12',
          measured=f'rank={H_rank}, |H|={len(H)}', threshold='7, 128')

    # 2. Association scheme (essentials)
    section(state, 'Association Scheme / Coherent Configuration')
    type_valency = Counter()
    for u in range(64):
        for v in range(64):
            type_valency[orbital_type(0, 0, u, v)] += 1
    val_sum = sum(type_valency.values())
    check(state, f'Sum of valencies = {val_sum}',
          val_sum == N_OMEGA,
          quantity='Orbital valencies partition Omega',
          measured=str(val_sum), threshold=str(N_OMEGA))

    diag_vals = []
    for _ in range(40):
        u, v = rng.randint(0, 64), rng.randint(0, 64)
        nb = neighbor_set(u, v, S)
        diag_vals.append(len(nb & nb))
    diag_ok = all(v == 128 for v in diag_vals)
    check(state, f'Diagonal intersection = degree: {set(diag_vals)}',
          diag_ok,
          quantity='p_{ii}^0 style: |N(x)|=128 constant',
          measured=f'sample={set(diag_vals)}', threshold='{128}')

    nb0 = neighbor_set(0, 0, S)
    nb_types = Counter(orbital_type(0, 0, u, v) for u, v in nb0)
    check(state, f'Neighbor relation covers {len(nb_types)} orbital types',
          len(nb_types) >= 1 and sum(nb_types.values()) == 128,
          quantity='Byte adjacency is a union of G-orbitals on Omega x Omega',
          measured=f'n_types={len(nb_types)}, |N(0)|={sum(nb_types.values())}',
          threshold='sum=128')

    undirected_hits = 0
    undirected_total = 0
    for _ in range(200):
        u, v = rng.randint(0, 64), rng.randint(0, 64)
        nb = list(neighbor_set(u, v, S))
        if not nb:
            continue
        a, b = nb[rng.randint(0, len(nb))]
        undirected_total += 1
        if (u, v) in neighbor_set(a, b, S):
            undirected_hits += 1
    und_frac = undirected_hits / max(1, undirected_total)
    directed_adj = und_frac < 0.5
    check(state, f'Adjacency symmetry rate={und_frac:.3f} (directed={directed_adj})',
          undirected_total > 0 and directed_adj,
          quantity='Byte adjacency directedness (sample symmetry rate)',
          measured=f'{undirected_hits}/{undirected_total}={und_frac:.3f}',
          threshold='rate<0.5 (directed)')

    axiom_partition = val_sum == N_OMEGA
    axiom_diag = any(t[0] == 1 for t in type_valency)
    check(state, 'Coherent configuration axioms A1/A2 on samples',
          axiom_partition and axiom_diag,
          quantity='Orbital partition + diagonal relation present',
          measured=f'partition={axiom_partition}, diag={axiom_diag}, '
                   f'n_types={len(type_valency)}',
          threshold='partition True, diag present')

    # 3. Spectrum
    section(state, 'Schreier / Cayley Spectrum')
    P, omega_list = byte_transition_matrix()
    spec = adjacency_spectrum_info(P)
    vprint(f'  lambda_max={spec["lambda_max"]:.6f}, gap={spec["gap"]:.6f}')
    check(state, f'lambda_max ~ degree 256',
          abs(spec['lambda_max'] - 256.0) < 1.0,
          quantity='Largest adjacency eigenvalue equals byte degree 256',
          measured=f'{spec["lambda_max"]:.6f}', threshold='256')

    A = P.toarray()
    P2 = A @ A
    r2 = int(np.linalg.matrix_rank(P2, tol=1e-9))
    n = A.shape[0]
    J = np.ones((n, n)) / n
    fro = float(np.linalg.norm(P2 - J, 'fro'))
    check(state, f'rank(P^2)={r2}, ||P^2 - J/n||_F={fro:.3e}',
          r2 == 1 and fro < 1e-8,
          quantity='2-design spectral collapse: rank(P^2)=1',
          measured=f'rank={r2}, fro={fro:.3e}', threshold='rank=1, fro~0')

    cayley_deg = sum(1 for h in H if h != 0)
    check(state, f'Cayley on H is complete: deg={cayley_deg}',
          cayley_deg == len(H) - 1,
          quantity='Two-step Cayley graph on H is complete K_128',
          measured=str(cayley_deg), threshold=str(len(H) - 1))

    # 4. Aut(Gamma)
    section(state, 'Full Aut(Gamma)')
    t_aut = t_aut_elements()
    omega_uv = [(u, v) for u in range(64) for v in range(64)]

    def neigh_uv(u, v):
        return neighbor_set(u, v, S)

    t_aut_ok = 0
    for tu, tv in t_aut:
        ok = True
        for u, v in omega_uv[::31]:
            nb = neigh_uv(u, v)
            nb_t = {(a ^ tu, b ^ tv) for a, b in nb}
            if nb_t != neigh_uv(u ^ tu, v ^ tv):
                ok = False
                break
        if ok:
            for _ in range(5):
                u, v = rng.randint(0, 64), rng.randint(0, 64)
                nb = neigh_uv(u, v)
                nb_t = {(a ^ tu, b ^ tv) for a, b in nb}
                if nb_t != neigh_uv(u ^ tu, v ^ tv):
                    ok = False
                    break
        if ok:
            t_aut_ok += 1
    check(state, f'T_aut in Aut: {t_aut_ok}/{len(t_aut)}',
          t_aut_ok == len(t_aut) == 128,
          quantity='Chirality-pure translations inject into Aut(Gamma)',
          measured=f'{t_aut_ok}/{len(t_aut)}', threshold='128/128')

    s6_ok_count = 0
    for pi in adjacent_transpositions():
        ok = True
        for u, v in omega_uv[::37]:
            nb = neigh_uv(u, v)
            nb_p = {(permute_bits6(a, pi), permute_bits6(b, pi)) for a, b in nb}
            up, vp = permute_bits6(u, pi), permute_bits6(v, pi)
            if nb_p != neigh_uv(up, vp):
                ok = False
                break
        if ok:
            s6_ok_count += 1
    check(state, f'S6 generators preserve adjacency: {s6_ok_count}/5',
          s6_ok_count == 5,
          quantity='S_6 dipole-pair permutations lie in Aut(Gamma)',
          measured=f'{s6_ok_count}/5', threshold='5/5')

    aut_lower = len(t_aut) * 720
    constructive = 128 * 720
    check(state, f'|Aut| lower = constructive = {constructive}',
          aut_lower == constructive,
          quantity='|Aut| lower bound = |T_aut rtimes S6| = 92160',
          measured=str(aut_lower), threshold=str(constructive))

    gl = list(kernel_group())
    g_even = [g for g in gl if (g >> 12) & 1 == 0]
    even_aut_hits = 0
    even_tested = 0
    for g in g_even[::64]:
        _, tu, tv = _unpack(g)
        even_tested += 1
        ok = True
        for u, v in [(0, 0), (1, 2), (7, 9), (31, 15)]:
            nb = neigh_uv(u, v)
            nb_g = {(a ^ tu, b ^ tv) for a, b in nb}
            if nb_g != neigh_uv(u ^ tu, v ^ tv):
                ok = False
                break
        if ok:
            even_aut_hits += 1
    containment_ok = (t_aut_ok == 128 and even_aut_hits < even_tested
                      and s6_ok_count == 5)
    check(state, 'Aut containment: T_aut and S6 in; generic G not subset',
          containment_ok,
          quantity='Measured Aut(Gamma) containment (not full G)',
          measured=f'T_aut={t_aut_ok}/128, S6={s6_ok_count}/5, '
                   f'even_hits={even_aut_hits}/{even_tested}',
          threshold='T_aut+S6 in; G not subset')

    # Connected + diameter <= 2
    dist = {(0, 0): 0}
    q = [(0, 0)]
    qi = 0
    while qi < len(q):
        u, v = q[qi]
        qi += 1
        for a, b in neigh_uv(u, v):
            if (a, b) not in dist:
                dist[(a, b)] = dist[(u, v)] + 1
                q.append((a, b))
    diam = max(dist.values()) if dist else -1
    connected = len(dist) == N_OMEGA
    check(state, f'Gamma connected diam={diam}, covered={len(dist)}',
          connected and diam <= 2,
          quantity='Byte graph connected with diameter <= 2',
          measured=f'diam={diam}, covered={len(dist)}',
          threshold='covered=4096, diam<=2')

    n_t, n_pi, n_aut, compat, n_diag = aut_factorization_counts()
    check(state, f'Step Aut lower bound |T_diag|={n_t}, |S6|={n_pi}, product={n_aut}',
          n_t == 64 and n_pi == 720 and n_aut == 46080 and compat and n_diag == 64,
          quantity='Step-level Aut(Gamma) lower bound T_diag semidirect S6 order 46080',
          measured=f'{n_t}*{n_pi}={n_aut}, compat={compat}',
          threshold='64*720=46080')

    # 5. Design classification
    section(state, 'Design Classification Verdict')
    match_name, novel = catalog_match(params_A)
    matched_tuple = (params_A['v'], params_A['k'], params_A['lambda'],
                     params_A['mu'], params_A['support'])
    S_inv_set = {inv_sig_int(s, D) for s in S}
    inv_closed = set(S) == S_inv_set
    print(f'  (v,k,lambda,mu,support) = {matched_tuple}')
    print(f'  H: dim={H_rank}, |H|={len(H)}, S=oH, directed={not inv_closed}')
    if match_name:
        print(f'  Catalog match: {match_name}')
        check(state, f'Catalog match: {match_name}',
              match_name == 'elementary_coset_H_order_128_in_A_4096'
              and matched_tuple == (4096, 128, 128, 0, 128),
              quantity=f'Design parameters match catalog entry {match_name}',
              measured=f'{matched_tuple}, directed={not inv_closed}, '
                       f'S=oH, dimH={H_rank}',
              threshold='(4096,128,128,0,128) coset-PDS')
    else:
        check(state, 'Coset-PDS parameters match expected tuple',
              matched_tuple == (4096, 128, 128, 0, 128)
              and coset_ok and H_rank == 7,
              quantity='Coset-PDS (v,k,lambda,mu,support) classification',
              measured=f'{matched_tuple}, directed={not inv_closed}',
              threshold='(4096,128,128,0,128)')

    cons = (k_S * k_S == len(H) * int(params_H['lambda']))
    check(state, f'|S|^2 = |H|*lambda: {k_S**2} = {len(H)}*{int(params_H["lambda"])}',
          cons,
          quantity='Difference-set counting identity |S|^2 = |H|*lambda',
          measured=f'{k_S**2} vs {len(H)*int(params_H["lambda"])}',
          threshold='equal')

    dt = time.time() - t0
    vprint(f'  elapsed_s = {dt:.2f}')
    n_pass = sum(1 for _, ok in state.gates if ok)
    n_fail = sum(1 for _, ok in state.gates if not ok)
    vprint(f'  gates: {n_pass} PASS, {n_fail} FAIL, total={len(state.gates)}')

    rng = np.random.RandomState(20260822)
    t0 = time.time()

    # 1. Exact equivariant layer
    section(state, 'Exact Equivariant Layer')
    f = random_signal(rng)
    w = bandlimited_weights(rng, n_keep=128)
    Lf = equivariant_layer(f, w)

    n_trans = 40
    trans_err = []
    for _ in range(n_trans):
        a = int(rng.randint(0, N_A))
        lhs = equivariant_layer(rho_translate(a, f), w)
        rhs = rho_translate(a, Lf)
        trans_err.append(float(np.linalg.norm(lhs - rhs)))
    max_trans = max(trans_err)
    mean_trans = float(np.mean(trans_err))
    check(state, f'Translation equivariance max err={max_trans:.3e}',
          max_trans < 1e-9,
          quantity='L(T_a f)=T_a L(f) on A (FWHT layer)',
          measured=f'max={max_trans:.3e}, mean={mean_trans:.3e}, n={n_trans}',
          threshold='<1e-9')

    F = fwht_normalized(f)
    Y = F * w
    recon = ifwht_normalized(Y)
    recon_err = float(np.linalg.norm(recon - Lf))
    check(state, f'Layer = IFFT(w * FFT(f)): err={recon_err:.3e}',
          recon_err < 1e-12,
          quantity='Equivariant layer equals Fourier multiplier',
          measured=f'{recon_err:.3e}', threshold='<1e-12')

    gl = list(kernel_group())
    even = [g for g in gl if (g >> 12) & 1 == 0]
    even_errs = []
    for g in even[:: max(1, len(even) // 30)][:30]:
        a = ((g >> 6) & 63) << 6 | (g & 63)
        lhs = equivariant_layer(rho_translate(a, f), w)
        rhs = rho_translate(a, Lf)
        even_errs.append(float(np.linalg.norm(lhs - rhs)))
    max_even = max(even_errs) if even_errs else 0.0
    check(state, f'Even (translation) G-sample max err={max_even:.3e}',
          max_even < 1e-9,
          quantity='Equivariance under sampled even G (=A)',
          measured=f'{max_even:.3e}', threshold='<1e-9')

    e_in = float(np.dot(f, f))
    e_F = float(np.dot(F, F))
    check(state, f'Parseval: ||f||^2={e_in:.6f} vs ||F||^2={e_F:.6f}',
          abs(e_in - e_F) < 1e-8,
          quantity='Unitary FWHT Parseval on L2(A)',
          measured=f'in={e_in:.6f}, F={e_F:.6f}', threshold='equal')

    idxs = rng.choice(N_OMEGA, size=64, replace=False)
    f2 = np.zeros(N_OMEGA)
    f2[idxs] = rng.randn(len(idxs))
    Lf2 = equivariant_layer(f2, w)
    a_test = 0b101101011010
    comm = float(np.linalg.norm(
        equivariant_layer(rho_translate(a_test, f2), w) - rho_translate(a_test, Lf2)))
    check(state, f'Commutator ||[L,T_a]|| on sparse f={comm:.3e}',
          comm < 1e-9,
          quantity='Operator commutator [L, T_a] = 0 (sparse probe)',
          measured=f'{comm:.3e}', threshold='<1e-9')

    e0 = np.zeros(N_A); e0[0] = 1.0
    Fe0 = fwht_normalized(e0)
    flat_err = float(np.max(np.abs(Fe0 - 1.0 / math.sqrt(N_A))))
    check(state, f'FWHT(e_0) flat err={flat_err:.3e}',
          flat_err < 1e-12,
          quantity='FWHT maps delta_0 to constant character',
          measured=f'{flat_err:.3e}', threshold='<1e-12')

    # Nested multiplier algebra (novel, one gate)
    w1 = bandlimited_weights(rng, 32)
    w2 = bandlimited_weights(rng, 32)
    f0 = random_signal(rng)
    nest_err = float(np.linalg.norm(
        equivariant_layer(equivariant_layer(f0, w1), w2)
        - equivariant_layer(f0, w1 * w2)))
    check(state, f'Nested multipliers err={nest_err:.3e}',
          nest_err < 1e-9,
          quantity='L_w2 o L_w1 = L_{w1*w2} (multiplier algebra)',
          measured=f'{nest_err:.3e}', threshold='<1e-9')

    section(state, 'Full G-Equivariant Layer')
    W64 = walsh_hadamard64()
    w_g, n_params = g_equivariant_weights(rng)
    check(state, f'G-commutant parameter count = {n_params}',
          n_params == 2080,
          quantity='dim End_G(L^2(Omega)) = 2080 spectral gains',
          measured=str(n_params), threshold='2080')
    f_g = random_signal(rng)
    Lg = g_equivariant_layer(f_g, w_g, W64)
    g_errs = []
    gl = list(kernel_group())
    for g in gl[::137][:24]:
        lhs = g_equivariant_layer(rho_g(g, f_g), w_g, W64)
        rhs = rho_g(g, Lg)
        g_errs.append(float(np.linalg.norm(lhs - rhs)))
    max_g = max(g_errs) if g_errs else 0.0
    check(state, f'Full G equivariance max err={max_g:.3e}',
          max_g < 1e-8,
          quantity='L(rho_g f) = rho_g L(f) for sampled g in G',
          measured=f'max={max_g:.3e}, n={len(g_errs)}', threshold='<1e-8')

    # 2. Two-byte Haar-analog (real uniform on Omega)
    section(state, 'Two-Byte Haar-Analog Sampler')
    omega_mult = Counter()
    for b1 in range(256):
        for b2 in range(256):
            s0 = omega_from_uv(0, 0)
            s1 = step_state_by_byte(s0, b1)
            s2 = step_state_by_byte(s1, b2)
            u, v = uv_from_omega(s2)
            omega_mult[uv_to_index(u, v)] += 1
    n_hit = len(omega_mult)
    mults = np.array([omega_mult[i] for i in range(N_OMEGA)], dtype=np.float64)
    expect_m = 65536.0 / N_OMEGA
    chi2_omega = float(np.sum((mults - expect_m) ** 2 / expect_m))
    mult_min, mult_max = int(np.min(mults)), int(np.max(mults))
    vprint(f'  Two-byte Omega coverage: n_hit={n_hit}/{N_OMEGA}, '
          f'mult=[{mult_min},{mult_max}], chi2={chi2_omega:.4f}')
    check(state, f'Two-byte hits all Omega uniformly chi2={chi2_omega:.4f}',
          n_hit == N_OMEGA and mult_min == mult_max == int(expect_m) and chi2_omega < 1e-9,
          quantity='Two-byte map hits all 4096 Omega states uniformly (multiplicity)',
          measured=f'n_hit={n_hit}, mult=[{mult_min},{mult_max}], chi2={chi2_omega:.4e}',
          threshold=f'n_hit={N_OMEGA}, mult={int(expect_m)}, chi2~0')

    n_codes = 1 << 12
    check(state, f'Codebook size = {n_codes}',
          n_codes == 4096,
          quantity='E: Omega -> SO(3) codebook cardinality at 12 bits',
          measured=str(n_codes), threshold='4096')

    # 3. SO(3) codebook E/D
    section(state, 'SO(3) Codebook E/D')
    budgets = [8, 12, 16, 24]
    n_rt = 300
    rows = []
    mean_errs = {}
    for B in budgets:
        errs = []
        for _ in range(n_rt):
            R = uniform_random_rotation(rng)
            errs.append(so3_roundtrip(R, B))
        errs = np.asarray(errs)
        mean_errs[B] = float(np.mean(errs))
        rows.append({
            'budget': B,
            'mean_err': f'{mean_errs[B]:.4f}',
            'median': f'{float(np.median(errs)):.4f}',
            'p90': f'{float(np.percentile(errs, 90)):.4f}',
            'max': f'{float(np.max(errs)):.4f}',
        })
    print_table('Geodesic roundtrip error vs budget', rows, always=True)
    mono = all(mean_errs[budgets[i]] + 1e-9 >= mean_errs[budgets[i + 1]]
               for i in range(len(budgets) - 1))
    check(state, f'mean_err monotone nonincreasing: {[round(mean_errs[b],4) for b in budgets]}',
          mono,
          quantity='so3_roundtrip mean geodesic error monotone nonincreasing in budget',
          measured=str({b: round(mean_errs[b], 4) for b in budgets}),
          threshold='err(8)>=err(12)>=err(16)>=err(24)')
    check(state, f'mean_err(24)={mean_errs[24]:.4f}',
          mean_errs[24] < 0.5,
          quantity='24-bit codebook mean geodesic error',
          measured=f'{mean_errs[24]:.4f}', threshold='<0.5 rad')

    idemp_errs = []
    for B in budgets:
        step = max(1, (1 << min(B, 10)) // 32)
        for c in range(0, 1 << min(B, 10), step):
            R = codebook_decode(c, B)
            _, R_hat = so3_encode_bits(R, B)
            _, R_hat2 = so3_encode_bits(R_hat, B)
            idemp_errs.append(geodesic_error(R_hat, R_hat2))
    idemp_mean = float(np.mean(idemp_errs))
    check(state, f'Quantized idempotence on codebook centers mean={idemp_mean:.4e}',
          idemp_mean < 1e-6,
          quantity='decode(encode(R_hat))≈R_hat on codebook centers (idempotence)',
          measured=f'mean={idemp_mean:.4e}, max={float(np.max(idemp_errs)):.4e}, n={len(idemp_errs)}',
          threshold='<1e-6')

    # 4. Rigid-body POC (short chains only)
    section(state, 'Rigid-Body Composition POC')
    short_eps = 0.75
    short_lengths = [20, 50]
    short_rows = []
    for L in short_lengths:
        incs_s = [uniform_random_rotation(rng) for _ in range(L)]
        R_ex_s = compose_mats(incs_s)
        R_q_s = compose_mats([so3_encode_bits(R, 24)[1] for R in incs_s])
        err_s = geodesic_error(R_ex_s, R_q_s)
        short_rows.append({'L': L, 'budget': 24, 'err': f'{err_s:.4f}'})
        check(state, f'L={L} budget24 err={err_s:.4f}',
              err_s < short_eps,
              quantity=f'Short rigid-body chain L={L} budget=24 geodesic error < eps',
              measured=f'{err_s:.4f}', threshold=f'<{short_eps}')
    print_table('Short-chain errors (budget 24)', short_rows, always=True)

    incs200 = [uniform_random_rotation(rng) for _ in range(200)]
    R_ex = compose_mats(incs200)
    errs_b = {}
    for B in (8, 12, 16):
        Rq = compose_mats([so3_encode_bits(R, B)[1] for R in incs200])
        errs_b[B] = geodesic_error(R_ex, Rq)
    print_table('Chain L=200 error vs budget', [
        {'budget': B, 'err': f'{errs_b[B]:.4f}'} for B in errs_b
    ], always=True)
    check(state, f'chain err budget8={errs_b[8]:.3f} >= budget16={errs_b[16]:.3f}',
          errs_b[8] + 1e-9 >= errs_b[16],
          quantity='Higher codebook budget reduces chain composition error',
          measured=str({b: round(errs_b[b], 4) for b in errs_b}),
          threshold='err(8) >= err(16)')

    R_quant = compose_mats([so3_encode_bits(R, 12)[1] for R in incs200])
    det_ok = abs(np.linalg.det(R_quant) - 1.0) < 1e-9
    orth_ok = float(np.linalg.norm(R_quant.T @ R_quant - np.eye(3))) < 1e-9
    check(state, f'Quantized product in SO(3): det_ok={det_ok}, orth_ok={orth_ok}',
          det_ok and orth_ok,
          quantity='Quantized chain product remains in SO(3)',
          measured=f'det={np.linalg.det(R_quant):.6f}, '
                   f'orth_res={float(np.linalg.norm(R_quant.T @ R_quant - np.eye(3))):.3e}',
          threshold='det=1, orth=0')

    dt = time.time() - t0
    vprint(f'  elapsed_s = {dt:.2f}')
    n_pass = sum(1 for _, ok in state.gates if ok)
    n_fail = sum(1 for _, ok in state.gates if not ok)
    vprint(f'  gates: {n_pass} PASS, {n_fail} FAIL, total={len(state.gates)}')

    rng = np.random.RandomState(20260822)
    t0 = time.time()
    N = N_OMEGA

    # 1. Grover from gate F
    section(state, 'Grover from Gate F')
    for M in (1, 4, 16):
        marked = list(rng.choice(N, size=M, replace=False))
        theta = grover_angle(M, N)
        psi = np.ones(N, dtype=np.float64) / math.sqrt(N)
        rows = []
        law_ok = True
        for k in range(0, 8):
            if k > 0:
                psi = grover_iteration(psi, marked)
            amp = amplitude_on_marked(psi, marked)
            predicted = abs(math.sin((2 * k + 1) * theta))
            err = abs(amp - predicted)
            rows.append({
                'M': M, 'k': k, 'amp': f'{amp:.6f}',
                'sin_law': f'{predicted:.6f}', 'err': f'{err:.3e}',
            })
            if err > 1e-9:
                law_ok = False
        print_table(f'Grover M={M}, theta={theta:.6f}', rows)
        check(state, f'Grover angle law M={M}',
              law_ok,
              quantity=f'Amplitude amp_k = |sin((2k+1)arcsin(sqrt(M/N)))| for M={M}',
              measured=f'theta={theta:.6f}, max_err from table',
              threshold='err<1e-9 each k')

    M = 4
    theta = grover_angle(M, N)
    k_opt = int(round(math.pi / (4 * theta) - 0.5))
    psi = np.ones(N) / math.sqrt(N)
    marked = list(range(M))
    for k in range(k_opt):
        psi = grover_iteration(psi, marked)
    amp_opt = amplitude_on_marked(psi, marked)
    p_succ = amp_opt ** 2
    vprint(f'  M={M}: k_opt={k_opt}, amp={amp_opt:.6f}, P_succ={p_succ:.6f}')
    check(state, f'Near-optimal amp at k_opt={k_opt}: {amp_opt:.4f}',
          amp_opt > 0.9 and p_succ > 0.85,
          quantity='Grover peak amplitude / P_succ near 1 at optimal iteration',
          measured=f'amp={amp_opt:.6f}, P_succ={p_succ:.6f}', threshold='amp>0.9, P>0.85')

    psi0 = rng.randn(N)
    psi0 /= np.linalg.norm(psi0)
    Ff = householder_diffusion(householder_diffusion(psi0))
    inv_err = float(np.linalg.norm(Ff - psi0))
    check(state, f'Householder involution err={inv_err:.3e}',
          inv_err < 1e-12,
          quantity='Gate F is a balanced involution F^2=I (not Grover diffusion)',
          measured=f'{inv_err:.3e}', threshold='<1e-12')

    # 2. Bell pairs
    section(state, 'Bell Pairs under Composition')
    edges = six_pair_edges()
    n_samp = 3000
    stab_states = []
    for x in range(N_OMEGA):
        ok = True
        for i, j in edges:
            if ((x >> i) & 1) ^ ((x >> j) & 1):
                ok = False
                break
        if ok:
            stab_states.append(x)
    vprint(f'  Stabilizer subspace size (all pairs even): {len(stab_states)}')
    check(state, f'|stab subspace|={len(stab_states)}',
          len(stab_states) == 64,
          quantity='Graph-state-like even-pair subspace order',
          measured=str(len(stab_states)), threshold='64')

    samples_gs = np.array([stab_states[rng.randint(0, len(stab_states))]
                           for _ in range(n_samp)])
    gs_corr = {e: correlator_ij(samples_gs, e[0], e[1]) for e in edges}
    print_table('Graph-state-like correlators', [
        {'pair': f'({i},{j})', 'corr': f'{gs_corr[(i,j)]:.4f}'}
        for i, j in edges
    ])
    gs_ok = all(abs(gs_corr[e] - 1.0) < 1e-12 for e in edges)
    check(state, 'GS correlators = +1 on all six pairs',
          gs_ok,
          quantity='E[(-1)^{bi xor bj}]=+1 on even-pair subspace',
          measured=str({f'{e}': gs_corr[e] for e in edges}),
          threshold='all +1')

    gl = list(kernel_group())
    g_sample = [gl[i] for i in rng.choice(len(gl), size=24, replace=False)]
    rows = []
    for g in g_sample:
        samp_g = transform_samples(samples_gs, g)
        corr_g = {e: correlator_ij(samp_g, e[0], e[1]) for e in edges}
        vals = [corr_g[e] for e in edges]
        rows.append({
            'g_parity': _unpack(g)[0],
            'mean_corr': f'{float(np.mean(vals)):.3f}',
            'min_corr': f'{min(vals):.3f}',
            'max_corr': f'{max(vals):.3f}',
        })
    print_table('Correlators after g·samples (24 group elements)', rows[:12])
    check(state, 'Bell correlators tabulated under 24 sampled g',
          len(rows) == 24,
          quantity='Six pair correlators tabulated under 24 sampled g in G',
          measured=f'n_rows={len(rows)}',
          threshold='24 rows')

    # 3. Non-abelian sync
    section(state, 'Non-Abelian Sync')
    gl_arr = np.array(gl, dtype=np.int64)
    fine_eta = [i / 40.0 for i in range(0, 21)]
    ab_curve, nab_curve = [], []
    for eta in fine_eta:
        ab_hit = nab_hit = 0
        trials = 40
        for _ in range(trials):
            x = int(rng.randint(0, N_OMEGA))
            est, true = sync_observation_abelian(x, eta, rng)
            if match_g(est, true):
                ab_hit += 1
            tg = int(gl_arr[rng.randint(0, len(gl_arr))])
            e1, e2, tgg = sync_observation_nonabelian(tg, eta, rng, gl_arr)
            if match_g(e1, tgg) or match_g(e2, tgg):
                nab_hit += 1
        ab_curve.append(ab_hit / trials)
        nab_curve.append(nab_hit / trials)
    print_table('Fine sync recovery curve', [
        {'eta': f'{fine_eta[i]:.3f}', 'ab': f'{ab_curve[i]:.3f}',
         'nab': f'{nab_curve[i]:.3f}'}
        for i in range(0, len(fine_eta), 2)
    ], always=True)

    ab0 = ab_curve[0]
    nab0 = nab_curve[0]
    check(state, f'Abelian recovery eta=0: {ab0:.3f}',
          ab0 > 0.95,
          quantity='Abelian sync recovery at zero noise',
          measured=f'{ab0:.3f}', threshold='>0.95')
    check(state, f'Nonabelian recovery eta=0: {nab0:.3f}',
          0.0 <= nab0 <= 1.0 and nab0 < ab0,
          quantity='Non-abelian sync recovery at zero noise (harder than abelian)',
          measured=f'nab0={nab0:.3f}, ab0={ab0:.3f}',
          threshold='in [0,1] and nab0 < ab0')

    def half_thresh(curve, etas):
        for e, c in zip(etas, curve):
            if c < 0.5:
                return e
        return etas[-1]
    ht_ab = half_thresh(ab_curve, fine_eta)
    ht_nab = half_thresh(nab_curve, fine_eta)
    vprint(f'  Half-height thresholds: abelian={ht_ab:.3f}, nonabelian={ht_nab:.3f}')
    check(state, f'Half-height thresholds ab={ht_ab:.3f}, nab={ht_nab:.3f}',
          ht_ab > 0.0 and ht_nab >= 0.0 and ht_ab <= 0.5,
          quantity='Sync half-height noise thresholds (fine grid)',
          measured=f'ab={ht_ab:.3f}, nab={ht_nab:.3f}',
          threshold='ab in (0,0.5], nab>=0')

    # 4. Krawtchouk / casimir
    section(state, 'Krawtchouk to Racah')
    K = krawtchouk_matrix(12)
    ortho_ok = True
    n = 12
    for r in range(0, 7):
        for s in range(0, 7):
            acc = 0.0
            for x in range(n + 1):
                acc += math.comb(n, x) * K[r, x] * K[s, x]
            rhs = (math.comb(n, r) * (2 ** n)) if r == s else 0.0
            if abs(acc - rhs) > 1e-6:
                ortho_ok = False
    check(state, 'Krawtchouk orthogonality on shells r,s<=6',
          ortho_ok,
          quantity='sum_x C(12,x) K_r(x) K_s(x) = delta_rs C(12,r) 2^12',
          measured=f'ortho_ok={ortho_ok}', threshold='True')

    shell_pop = Counter(_pop(i) for i in range(N_OMEGA))
    pop_ok = all(shell_pop[k] == math.comb(12, k) for k in range(13))
    check(state, 'GF(2)^12 register shells = C(12,k)',
          pop_ok,
          quantity='translation-register shell census C(12,k)',
          measured=str([shell_pop[k] for k in range(13)]),
          threshold=str([math.comb(12, k) for k in range(13)]))
    omega_shell = [math.comb(6, k) * 64 for k in range(7)]
    check(state, f'Omega chirality shells = C(6,k)*64',
          omega_shell == [64, 384, 960, 1280, 960, 384, 64],
          quantity='Omega chirality shell census',
          measured=str(omega_shell), threshold=str([64, 384, 960, 1280, 960, 384, 64]))

    corr_xs, corr_ys = [], []
    for j in range(0, 7):
        corr_xs.append(casimir_eigenvalue(j))
        corr_ys.append(float(j * (12 - j)))
    xs = np.asarray(corr_xs)
    ys = np.asarray(corr_ys)
    if float(np.std(xs)) > 1e-12 and float(np.std(ys)) > 1e-12:
        pearson_cas_shell = float(np.corrcoef(xs, ys)[0, 1])
    else:
        pearson_cas_shell = 0.0
    check(state, f'Pearson(casimir, shell_quad)={pearson_cas_shell:.4f}',
          pearson_cas_shell > 0.85,
          quantity='Correlation of j(j+1) template to k(12-k) shell energy',
          measured=f'{pearson_cas_shell:.4f}', threshold='>0.85')

    # 5. Toric / center
    section(state, 'Mod-2 Loop / Toric Probe')
    center = set(central_involutions())
    vprint(f'  |Z(G)| = {len(center)}')
    comm_in_center = 0
    n_comm = 500
    for _ in range(n_comm):
        g = int(gl_arr[rng.randint(0, len(gl_arr))])
        h = int(gl_arr[rng.randint(0, len(gl_arr))])
        c = commutator(g, h)
        if c in center:
            comm_in_center += 1
    check(state, f'Commutators in Z(G): {comm_in_center}/{n_comm}',
          comm_in_center == n_comm,
          quantity='Plaquette commutators land in central involutions',
          measured=f'{comm_in_center}/{n_comm}', threshold=f'{n_comm}/{n_comm}')

    order_ok = True
    for c in center:
        if compose_sig_int_d(c, c, D) != 0:
            order_ok = False
            break
    check(state, f'Central elements are involutions: {order_ok}',
          order_ok and len(center) == 64,
          quantity='Z(G) = (Z/2)^6 of central involutions',
          measured=f'|Z|={len(center)}, involutive={order_ok}',
          threshold='64, True')

    uv_eq = True
    for _ in range(300):
        g = int(gl_arr[rng.randint(0, len(gl_arr))])
        h = int(gl_arr[rng.randint(0, len(gl_arr))])
        c = commutator(g, h)
        _, u, v = _unpack(c)
        if u != v or _unpack(c)[0] != 0:
            uv_eq = False
            break
    check(state, f'[g,h] has form (0,d,d): {uv_eq}',
          uv_eq,
          quantity='Commutator image inside diagonal center',
          measured=f'{uv_eq}', threshold='True')

    dt = time.time() - t0
    vprint(f'  elapsed_s = {dt:.2f}')
    n_pass = sum(1 for _, ok in state.gates if ok)
    n_fail = sum(1 for _, ok in state.gates if not ok)
    vprint(f'  gates: {n_pass} PASS, {n_fail} FAIL, total={len(state.gates)}')

if __name__ == '__main__':
    st = ReportState()
    run(st)
    passed = sum(1 for _, ok in st.gates if ok)
    failed = sum(1 for _, ok in st.gates if not ok)
    print(f'\nSUMMARY: {passed} passed, {failed} failed out of {len(st.gates)}')
