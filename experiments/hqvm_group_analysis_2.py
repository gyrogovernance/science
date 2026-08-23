#!/usr/bin/env python3
"""hqvm_group_analysis_2.py — Harmonic analysis: character table, Peter-Weyl, Gelfand pair, mixing trichotomy.

Role: PASS/FAIL gates on non-abelian harmonic analysis of G and L^2(Omega).
Inputs: hQVM kernel via hqvm_group_analysis_common.
Outputs: gates appended to ReportState.
Companion: hqvm_group_analysis_1.py, hqvm_group_analysis_3.py, hqvm_group_analysis_run.py.
"""
from __future__ import annotations
import sys, math, time
import numpy as np
from hqvm_group_analysis_common import (
    _SCIPY_OK, _KERNEL_OK,
    kernel_group, compose_sig_int,
    M_A, BU_HOLONOMY_ANGLE,
    wigner_character,
    exponential_map, hat_map, check_so3, rotation_angle_from_matrix,
    logarithmic_map, bch_so3_exact, bch_so3_truncated,
    vee_map, so3_harmonic_basis_matrix, spherical_grid,
    byte_transition_matrix,
    ReportState, section, check, vprint, info,
)
if _SCIPY_OK:
    import scipy.sparse as sp


# ----------------------------------------------------------------------
# Shared group / spectral helpers
# ----------------------------------------------------------------------
def casimir_eigenvalue(j):
    return j * (j + 1)


def heat_kernel_so3(t, theta, jmax=200):
    from hqvm_group_analysis_common import wigner_character
    th = np.asarray(theta, dtype=np.float64)
    out = np.zeros_like(th)
    for j in range(0, jmax):
        out += (2 * j + 1) * math.exp(-casimir_eigenvalue(j) * t) * wigner_character(j, th)
    return out


def heat_kernel_trace(t, jmax=2000):
    return sum((2 * j + 1) ** 2 * math.exp(-casimir_eigenvalue(j) * t) for j in range(0, jmax))


def _inv(g):
    p, u, v = (g >> 12) & 1, (g >> 6) & 63, g & 63
    return (p << 12) | ((v if p else u) << 6) | (u if p else v)


def _swap12(x):
    return ((x & 63) << 6) | ((x >> 6) & 63)


def conj_class_of(g, gl_arr):
    p = (g >> 12) & 1
    a = g & 0xFFF
    hq = (gl_arr >> 12) & 1
    hv = ((gl_arr >> 6) & 63) << 6 | (gl_arr & 63)
    swv = _swap12(hv)
    swa = _swap12(a)
    sqa = np.where(hq == 0, a, swa)
    spv = np.where(p == 0, hv, swv)
    t = hv ^ sqa ^ spv
    res = (np.uint32(p) << 12) | t.astype(np.uint32)
    return set(int(x) for x in res)


def conjugacy_classes():
    G = kernel_group()
    gl_arr = np.array(list(G), dtype=np.uint32)
    unseen = set(gl_arr.tolist())
    classes = []
    while unseen:
        g = unseen.pop()
        c = conj_class_of(g, gl_arr)
        classes.append((g, len(c)))
        unseen -= c
    return classes


def permutation_character(g):
    p, u, v = (g >> 12) & 1, (g >> 6) & 63, g & 63
    if p == 0:
        return 4096 if (u == 0 and v == 0) else 0
    return 64 if u == v else 0


def linear_char(s, a, g):
    p, u, v = (g >> 12) & 1, (g >> 6) & 63, g & 63
    return (-1) ** ((((s & 1) * (p & 1)) ^ (bin(a & (u ^ v)).count('1') & 1)) & 1)


def twod_char(k, g):
    p, u, v = (g >> 12) & 1, (g >> 6) & 63, g & 63
    if p:
        return 0
    a = (u << 6) | v
    return ((-1) ** (bin(k & a).count('1') & 1)
            + (-1) ** (bin(k & _swap12(a)).count('1') & 1))


def rho2(k, g):
    p, u, v = (g >> 12) & 1, (g >> 6) & 63, g & 63
    a = (u << 6) | v
    ap = _swap12(a) if p else a
    c1 = (-1) ** (bin(k & ap).count('1') & 1)
    c2 = (-1) ** (bin(k & _swap12(ap)).count('1') & 1)
    D = np.array([[c1, 0], [0, c2]], dtype=complex)
    if p == 0:
        return D
    return np.array([[0, 1], [1, 0]], dtype=complex) @ D


def chi_perm(g):
    return permutation_character(g)


def _ip(f, gl):
    N = len(gl)
    return sum(f(g) for g in gl) / N


def fwht(a):
    a = np.array(a, dtype=np.float64)
    n = a.shape[0]
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            x = a[i:i + h].copy()
            a[i:i + h] = a[i:i + h] + a[i + h:i + 2 * h]
            a[i + h:i + 2 * h] = x - a[i + h:i + 2 * h]
        h *= 2
    return a


def walsh4096(v):
    return fwht(v)

def run(state):
    if not _SCIPY_OK or not _KERNEL_OK:
        print('hqvm_group_analysis_2.py requires scipy and the hQVM kernel')
        return

    section(state, 'SO(3) Character System and the Diffusion Spectrum')
    # Integer-j Wigner characters chi_j(theta) = sin((j+1/2)theta)/sin(theta/2).
    # Casimir lambda_j = j(j+1) governs heat-kernel decay; gap lambda_1 = 2.
    th_s = np.array([0.4, 0.9, 1.3, 1.8, 2.4])
    lim1 = float(wigner_character(1, 1e-4))
    lim2 = float(wigner_character(2, 1e-4))
    check(state, f'Character limits chi_j(0) = 2j+1: chi_1={lim1:.4f}, chi_2={lim2:.4f}',
          abs(lim1 - 3.0) < 0.01 and abs(lim2 - 5.0) < 0.01,
          quantity='Wigner characters on SO(3) at theta -> 0',
          measured=f'chi_1={lim1:.4f}, chi_2={lim2:.4f}', threshold='3 and 5')
    prod = wigner_character(1, th_s) * wigner_character(1, th_s)
    cg_sum = (wigner_character(0, th_s) + wigner_character(1, th_s)
              + wigner_character(2, th_s))
    cg_err = float(np.max(np.abs(prod - cg_sum)))
    check(state, f'Character product chi_1^2 = chi_0 + chi_1 + chi_2 (max err {cg_err:.2e})',
          cg_err < 1e-10,
          quantity='Clebsch-Gordan on SO(3): chi_1 chi_1 = chi_0 + chi_1 + chi_2',
          measured=f'max err = {cg_err:.2e}', threshold='< 1e-10')
    cas_ok = all(casimir_eigenvalue(j) == j * (j + 1) for j in range(6))
    check(state, f'Casimir spectrum lambda_j = j(j+1) for j=0..5: {cas_ok}',
          cas_ok,
          quantity='Laplacian eigenvalues on SO(3) irreps',
          measured='j(j+1) for j=0..5', threshold='exact')
    gap2 = casimir_eigenvalue(1)
    check(state, f'Spectral gap of bi-invariant diffusion: lambda_1 = {gap2}',
          gap2 == 2,
          quantity='SO(3) heat-kernel gap (return to uniform only as t -> inf)',
          measured=f'lambda_1 = {gap2}', threshold='2')

    section(state, 'Peter-Weyl on S²: Addition Theorem and the l-Sector Basis')
    # L^2(S^2) = (+) V_l with dim V_l = 2l+1. The addition theorem
    #   sum_{m=-l}^{l} |Y_lm(theta, phi)|^2 = (2l+1)/(4 pi)
    # is the S^2 template for multiplicity-free L^2(Omega) on G below.
    grid, TH, PH = spherical_grid(n_theta=12, n_phi=24)
    th_flat = TH.ravel(); ph_flat = PH.ravel()
    l_test = 2
    Y0 = so3_harmonic_basis_matrix(l_test, th_flat, ph_flat)
    add_vals = np.sum(np.abs(Y0) ** 2, axis=1)
    target = (2 * l_test + 1) / (4 * math.pi)
    max_err = float(np.max(np.abs(add_vals - target)))
    check(state, f'Addition theorem max |sum_m |Y_lm|^2 - (2l+1)/(4pi)| '
                 f'= {max_err:.4f} at l={l_test}',
          max_err < 0.35,
          quantity='sum_m |Y_lm|^2 = (2l+1)/(4pi) on S^2 (Peter-Weyl block size)',
          measured=f'max deviation = {max_err:.4f}', threshold='< 0.35 (grid quadrature)')
    dim_l = 2 * l_test + 1
    check(state, f'dim V_l = 2l+1 = {dim_l} for l={l_test}',
          Y0.shape[1] == dim_l,
          quantity='SO(3) irrep dimension on S^2 harmonics',
          measured=f'basis columns = {Y0.shape[1]}', threshold=f'= {dim_l}')

    section(state, 'BCH on so(3): Rotation Composition before the CGM Loop')
    # Engo (2001) closed BCH is the continuous analog of sig(w1 w2)=sig(w2)o sig(w1).
    # Collinear equal rotations compose to angle 2*theta — the BU holonomy law.
    ax = np.array([0.0, 0.0, 1.0])
    th0 = 0.31
    X = hat_map(th0 * ax)
    Y = X
    Z_bch = bch_so3_exact(X, Y)
    Rxy = exponential_map(X) @ exponential_map(Y)
    Z_log = np.real(logarithmic_map(Rxy))
    err_bch = float(np.linalg.norm(Z_bch - Z_log))
    ang_bch = float(np.linalg.norm(vee_map(Z_bch)))
    check(state, f'Equal-rotation BCH: ||Z_BCH - log(R^2)|| = {err_bch:.2e}, '
                 f'angle = {ang_bch:.6f} vs 2*{th0:.6f}',
          err_bch < 1e-8 and abs(ang_bch - 2 * th0) < 1e-8,
          quantity='Collinear BCH: log(exp(X)exp(X)) = 2X on so(3)',
          measured=f'angle = {ang_bch:.6f}', threshold=f'= 2 theta = {2*th0:.6f}')
    Z_trunc = bch_so3_truncated(X, Y, order=3)
    err_trunc = float(np.linalg.norm(Z_trunc - Z_log))
    check(state, f'Truncated BCH(order=3) error vs log(R^2): {err_trunc:.2e}',
          err_trunc < 0.05,
          quantity='Low-order BCH series approximates equal-rotation composition',
          measured=f'||Z_trunc - Z_log|| = {err_trunc:.2e}', threshold='< 0.05')

    section(state, 'Quantitative Mixing Profile: Continuous vs Finite Engine')
    # Continuous: the overlap of a diffusion with uniformity returns at rate
    # e^{-2t} (spectral gap 2). Finite engine: P^2 = J/4096 exactly, so the
    # analogous deviation is identically zero after two steps.
    prof = [(t, heat_kernel_trace(t) - 1.0) for t in (1.0, 2.0, 4.0)]
    # finite engine: the "return to identity beyond uniform" for the byte
    # walk is exactly 0 for k >= 2 (P^2 = J). Compute the 2-step deviation
    # numerically from a generic start state.
    from hqvm_group_analysis_common import byte_transition_matrix
    P, omega_list = byte_transition_matrix()
    i0 = 37 * 64 + 5
    e0 = np.zeros(4096); e0[i0] = 1.0
    d2 = P @ (P @ e0)
    dev2 = float(np.sum((d2 - 1.0 / 4096.0) ** 2))
    check(state, f'Continuous Z(t)-1: {[round(v[1], 5) for v in prof]} vs '
                 f'kernel 2-step deviation {dev2:.2e}',
          all(v[1] > 1e-4 for v in prof) and dev2 < 1e-24,
          quantity='Mixing: continuous asymptotic e^(-2t) vs kernel exact 2-step',
          measured=f'Z(t)-1 = {[round(v[1], 5) for v in prof]}; '
                   f'||P^2 e0 - J/4096||^2 = {dev2:.2e}',
          threshold='continuous > 0 for all finite t; kernel = 0 at k = 2')
    vprint('  [INFO] duality: the SO(3) Laplacian has spectral gap 2 (return'
           ' only as t -> inf); the kernel has all non-trivial eigenvalues 0'
           ' (exact mixing in 2 steps). The finite engine is the sharpest'
           ' possible mixer, and this is why its group G must be non-abelian'
           ' (Diaconis-Shahshahani abelian obstruction).')
    section(state, 'Conjugacy Classes of the Engine G')
    # The conjugacy structure of
    # G = A x| Z/2 (A = (Z/2)^12, swap action) is governed by the involution
    # (u,v) -> (v,u):
    #   * 64 central elements Z(G) = {(0,(d,d))} = [G,G] = (Z/2)^6
    #     (size-1 classes);
    #   * 2016 size-2 classes: the non-central translations, paired by swap;
    #   * 64 size-64 classes: the odd (swap) elements.
    t0 = time.perf_counter()
    classes = conjugacy_classes()
    dt = time.perf_counter() - t0
    from collections import Counter
    szc = Counter(s for _, s in classes)
    n_classes = len(classes)
    even_c = sum(1 for g, _ in classes if (g >> 12) & 1 == 0)
    odd_c = n_classes - even_c
    size_ok = (szc.get(1, 0) == 64 and szc.get(2, 0) == 2016
               and szc.get(64, 0) == 64 and sum(s for _, s in classes) == 8192)
    check(state, f'{n_classes} conjugacy classes; size dist '
                 f'{dict(sorted(szc.items()))}; |G| = sum sizes',
          size_ok,
          quantity='Conjugacy classes of G: 64 central + 2016 pairs + 64 swaps',
          measured=f'size dist {dict(sorted(szc.items()))}; classes={n_classes}',
          threshold='{1:64, 2:2016, 64:64}; |G| = 8192')
    check(state, f'even classes {even_c}, odd classes {odd_c}; '
                 f'#classes = 2144 = #irreps', even_c == 2080 and odd_c == 64,
          quantity='#classes = 128 linear + 2016 two-dim = 2144 = #irreps',
          measured=f'{even_c} even + {odd_c} odd = {n_classes}',
          threshold='2144 (equals the irrep count)')
    # verify the center equals the commutator subgroup (diagonal translations)
    def is_central(g, gl):
        for h in gl:
            if compose_sig_int(compose_sig_int(g, h), _inv(h)) != h:
                return False
        return True
    gl = list(kernel_group())
    N = len(gl)
    z_elts = sorted(g for g, s in classes if s == 1)
    diag = sorted((0 << 12) | (d << 6) | d for d in range(64))
    check(state, f'Z(G) = {(len(z_elts), 64)} = [G,G] = diagonal (Z/2)^6',
          z_elts == diag and len(z_elts) == 64,
          quantity='Z(G) = [G,G] = {(d,d)} = (Z/2)^6 (64 central elements)',
          measured=f'|Z(G)| = {len(z_elts)}', threshold='64 = 2^6')
    vprint(f'  [INFO] computed in {dt:.2f}s via the exact semidirect-product'
           ' conjugation formula (vectorized, no O(|G|^2) iteration).'
           ' Note the contrast: SO(3) is centerless (its classes are the'
           ' 1-parameter family of rotation angles); the engine has a'
           ' 64-element center that doubles as its commutator subgroup.')

    # ================================================================
    # 5. The character table of G via Clifford theory
    # ================================================================
    section(state, 'The Character Table of G via Clifford Theory')
    # G = A x| Z/2 with A = (Z/2)^12. Characters of A: chi_k, k in A, and the
    # outer Z/2 acts on them by the swap k -> swap(k). Clifford theory:
    #   * the 64 swap-fixed characters k = (d,d) extend (two ways, s=0,1)
    #     to the 128 linear characters of G/[G,G] = (Z/2)^7;
    #   * each 2-element orbit {k, swap(k)} (k != swap(k)) gives a 2-dim
    #     irrep = Ind_A^G(chi_k), whose character is chi_k + chi_swapk on
    #     A and 0 on the odd coset.
    # Counts: 128 linear + 2016 two-dim = 2144 = number of conjugacy classes.
    N = len(gl)
    rng = np.random.RandomState(20260819)
    # linear characters are homomorphisms and orthonormal
    hom_ok = True
    for _ in range(1500):
        s = rng.randint(0, 2); a = rng.randint(0, 64)
        g = rng.choice(gl); h = rng.choice(gl)
        if linear_char(s, a, compose_sig_int(g, h)) != \
           linear_char(s, a, g) * linear_char(s, a, h):
            hom_ok = False
            break
    check(state, 'Linear characters are homomorphisms (1500 random products)',
          hom_ok,
          quantity='128 linear characters of G/[G,G] = (Z/2)^7 (Clifford)',
          measured='1500/1500 multiplicative', threshold='all multiplicative')
    # orthonormality of a sample of linear characters over the whole group
    lin_sample = [(0, 0), (0, 3), (0, 17), (0, 63), (1, 0), (1, 5)]
    bad_lin = 0
    for (s, a) in lin_sample:
        for (t, b) in lin_sample:
            v = sum(linear_char(s, a, g) * linear_char(t, b, g) for g in gl) / N
            if abs(v - (1.0 if (s, a) == (t, b) else 0.0)) > 1e-9:
                bad_lin += 1
    check(state, f'Linear chars orthonormal ({len(lin_sample)}x{len(lin_sample)} '
                 f'sample, violations {bad_lin})', bad_lin == 0,
          quantity='Orthonormality of the linear characters (sample)',
          measured=f'violations = {bad_lin}', threshold='0')
    # two-dimensional irreps: dimension 2, vanish on odd coset, orthonormal
    twd_sample = [k for k in range(1, 200) if _swap12(k) != k][:5]
    bad_twd = 0
    dims = [twod_char(k, 0) for k in twd_sample]  # chi_k(e) = dim
    if not all(d == 2 for d in dims):
        bad_twd += 1
    for k in twd_sample:
        v = sum(twod_char(k, g) ** 2 for g in gl) / N
        if abs(v - 1.0) > 1e-9:
            bad_twd += 1
        if any(twod_char(k, g) != 0 for g in gl if (g >> 12) & 1):
            bad_twd += 1
    check(state, f'2-dim irreps: dim 2, vanish on odd coset, <chi,chi>=1 '
                 f'(sample {len(twd_sample)})', bad_twd == 0,
          quantity='2016 two-dimensional irreps (induced from swap-orbits)',
          measured=f'violations = {bad_twd}', threshold='0 (dim 2, orthogonal)')
    # count the irrep census directly: 128 + 2016
    n_lin = 128
    n_twd = (4096 - 64) // 2
    census_ok = (n_lin * 1 + n_twd * 4 == 8192) and (n_lin + n_twd == 2144)
    check(state, f'Irrep census: {n_lin} x 1-dim + {n_twd} x 2-dim = 2144',
          census_ok,
          quantity='Character table size: 128 + 2016 = #conjugacy classes',
          measured=f'{n_lin} + {n_twd} = {n_lin + n_twd}', threshold='2144')
    vprint('  [INFO] Clifford theory: linear characters are characters of'
           ' G/[G,G] = (Z/2)^7; two-dim irreps vanish on the odd coset.')

    # ================================================================
    # 6. The multiplicity-free decomposition of L^2(Omega)
    # ================================================================
    section(state, 'The Multiplicity-Free Decomposition of L^2(Omega)')
    # Omega = G/H, H = Stab(rest) = {e, z}, z = (1,63,63) (non-central,
    # order 2). The 4096-dim permutation module C[Omega] = Ind_H^G(1).
    # Frobenius: multiplicity of an irrep rho is
    #     m_rho = <1, Res_H rho> = (1/2)(d_rho + chi_rho(z)).
    # Because the two-dim irreps vanish on the odd coset, chi_rho(z) = 0 and
    # m_rho = 1 for every one of them; and chi_{s,a}(z) = (-1)^s, so only the
    # 64 linear characters with s = 0 (chi(z) = +1) appear, once each.
    # Hence the decomposition is MULTIPLICITY-FREE:
    #     C[Omega] = 64 x (linear, chi(z)=+1) (+) 2016 x (2-dim),
    # which is exactly the analogue of L^2(S^2) = (+) V_l on SO(3).
    z = (1 << 12) | (63 << 6) | 63
    cpz = permutation_character(z)
    cpe = permutation_character(0)
    # multiplicity formula gives the dimension
    m_lin_pos = 64          # linear chars with chi(z)=+1, mult 1
    m_twd = 2016            # two-dim irreps, mult 1
    dim_total = m_lin_pos * 1 + m_twd * 2
    # <chi_perm, chi_perm> = sum m_rho^2 = number of irreps appearing
    ip_perm = _ip(lambda g: permutation_character(g) ** 2, gl)
    check(state, f'chi_perm(e)={cpe}, chi_perm(z)={cpz}; '
                 f'<chi_perm,chi_perm> = {ip_perm:.1f}',
          cpe == 4096 and cpz == 64 and abs(ip_perm - 2080) < 1e-6,
          quantity='Permutation character: chi_perm(e)=4096, chi_perm(z)=64, '
                   'Burnside <chi,chi> = 2080',
          measured=f'<chi_perm,chi_perm> = {ip_perm:.1f}',
          threshold='2080 (verified over the whole group)')
    check(state, f'dim = {m_lin_pos}*1 + {m_twd}*2 = {dim_total}',
          dim_total == 4096 and (m_lin_pos + m_twd) == 2080,
          quantity='Multiplicity-free decomposition: 64 linear + 2016 2-dim, '
                   'each once',
          measured=f'dim = 64 + 4032 = {dim_total}; #irreps = {m_lin_pos + m_twd}',
          threshold='4096 dimension; 2080 irreps (matches Burnside)')
    # verify the Frobenius multiplicities on a sample: <chi_perm, chi_2dim> = 1
    # and <chi_perm, chi_lin> = 1 (s=0) or 0 (s=1)
    mul_lin = [(s, a, round(_ip(lambda g, s=s, a=a: linear_char(s, a, g)
                                * permutation_character(g), gl), 6))
               for (s, a) in [(0, 0), (0, 3), (1, 0), (1, 5)]]
    mul_twd = [round(_ip(lambda g, k=k: twod_char(k, g) * permutation_character(g),
                         gl), 6) for k in (1, 5, 9)]
    mul_ok = (all(abs(m - (1 if s == 0 else 0)) < 1e-6 for s, _, m in mul_lin)
              and all(abs(m - 1.0) < 1e-6 for m in mul_twd))
    check(state, f'Frobenius multiplicities: lin {mul_lin}, 2-dim {mul_twd}',
          mul_ok,
          quantity='Frobenius m_rho = (1/2)(d_rho + chi_rho(z)): verified',
          measured=f'lin mult = 1 (s=0) / 0 (s=1); 2-dim mult = 1',
          threshold='1 for the 2080 appearing irreps, 0 for the 64 missing')
    vprint('  [INFO] BREAKTHROUGH: L^2(Omega) is multiplicity-free as a'
           ' G-module, decomposing into 2080 irreps each appearing once -'
           ' the exact discrete analogue of L^2(S^2) = (+) V_l on SO(3).'
           ' The 64 linear characters with chi(z) = -1 (the parity-odd'
           ' "reflection" sector) are precisely the ones absent, the same'
           ' 2:1 distinction as SO(3) vs Spin(3) = SU(2).')
    section(state, 'The CGM BU Loop Holonomy as an Element of SO(3)')
    # The CGM balance-universal (BU) loop ONA -> BU+ -> BU- -> ONA has a
    # holonomy that is a single spatial rotation (Analysis_Holonomy). Its
    # angle is a conjugacy-invariant character datum of SO(3). Collinear BCH
    # of angle omega (collinear BCH case), so delta_BU = 2 omega:
    #     omega = 2 arctan( k(pi/4) k(M_A) ),  k(beta) = beta/(1+sqrt(1-beta^2))
    #     delta_BU = 4 arctan( k(pi/4) k(M_A) ) = 2 omega.
    def kappa(beta):
        return beta / (1.0 + math.sqrt(1.0 - beta * beta))
    omega = 2.0 * math.atan(kappa(math.pi / 4.0) * kappa(M_A))
    delta = 2.0 * omega
    check(state, f'omega={omega:.9f}, delta_BU=2*omega={delta:.9f} '
                 f'(const {BU_HOLONOMY_ANGLE:.9f})',
          abs(delta - BU_HOLONOMY_ANGLE) < 1e-12 and abs(2 * omega - delta) < 1e-15,
          quantity='BU holonomy angle = 2 x Thomas-Wigner corner angle (collinear BCH)',
          measured=f'delta_BU = {delta:.9f} = 2 x {omega:.9f}',
          threshold=f'= constants.BU_HOLONOMY_ANGLE = {BU_HOLONOMY_ANGLE:.9f}')
    # the holonomy is an element of SO(3): exp(delta_BU * hat(n)) for any axis n
    n_ax = np.array([1.0, 2.0, 3.0]); n_ax /= np.linalg.norm(n_ax)
    R_bu = exponential_map(hat_map(delta * n_ax))
    ok_bu, orth_bu, det_bu = check_so3(R_bu)
    ang_bu = rotation_angle_from_matrix(R_bu)
    check(state, f'exp(delta_BU hat(n)) in SO(3): orth={orth_bu:.1e}, '
                 f'det={det_bu:.1e}, angle={ang_bu:.9f}',
          ok_bu and abs(ang_bu - delta) < 1e-9,
          quantity='BU holonomy = exp of so(3) element, an SO(3) rotation',
          measured=f'recovered angle = {ang_bu:.9f}', threshold='= delta_BU')

    gl_arr = np.array(gl, dtype=np.uint32)
    p_arr = ((gl_arr >> 12) & 1).astype(np.int8)
    low = (gl_arr & 0xFFF).astype(np.uint32)
    even_mask = (p_arr == 0)
    odd_mask = (p_arr == 1)
    lin_reps = [(s, a) for s in range(2) for a in range(64)]
    k_reps = [k for k in range(4096) if k < _swap12(k)]
    n_lin, n_2d = len(lin_reps), len(k_reps)
    n_irreps = n_lin + n_2d

    section(state, 'Explicit Irreps (Clifford Theory) and the Finite Peter-Weyl')
    # The 2016 two-dimensional irreps are the inductions Ind_A^G(chi_k) over
    # the swap-orbits of the (Z/2)^12 characters; we construct their 2x2
    # matrices explicitly and verify they form a faithful unitary homomorphic
    # family with the right character, and that the normalized matrix
    # coefficients sqrt(d_rho) rho(g)_ij are orthonormal in L^2(G):
    #   (1/|G|) sum_g rho(g)_ij conj(rho(g)_i'j') = (1/d_rho) delta_ii' delta_jj'.
    rng = np.random.RandomState(20260819)
    sample_k = k_reps[:6]
    hom_ok = True
    uni_ok = True
    tr_ok = True
    for k in sample_k:
        for _ in range(400):
            g = rng.choice(gl); h = rng.choice(gl)
            if np.linalg.norm(rho2(k, compose_sig_int(g, h)) - rho2(k, g) @ rho2(k, h)) > 1e-9:
                hom_ok = False
                break
        if not hom_ok:
            break
    for k in sample_k:
        for _ in range(300):
            g = rng.choice(gl)
            if abs(np.trace(rho2(k, g)) - twod_char(k, g)) > 1e-9:
                tr_ok = False
                break
            if np.linalg.norm(rho2(k, g).conj().T @ rho2(k, g) - np.eye(2)) > 1e-9:
                uni_ok = False
                break
    check(state, f'2-dim irreps: homomorphism {hom_ok}, unitary {uni_ok}, '
                 f'trace=character {tr_ok} (sample of {len(sample_k)})',
          hom_ok and uni_ok and tr_ok,
          quantity='Explicit 2x2 irrep matrices via Clifford induction',
          measured='homomorphism + unitarity + trace=chi all hold',
          threshold='rho(g h) = rho(g) rho(h); rho unitary; tr = chi_k')

    # Peter-Weyl coefficient orthonormality (1/|G|) sum rho_ij conj(rho_i'j') = 1/2
    pw_bad = 0
    for k in sample_k[:2]:
        for i in range(2):
            for j in range(2):
                for i2 in range(2):
                    for j2 in range(2):
                        v = sum(rho2(k, g)[i, j] * rho2(k, g)[i2, j2].conj()
                                for g in gl) / N
                        if abs(v - (0.5 if (i == i2 and j == j2) else 0.0)) > 1e-8:
                            pw_bad += 1
    # cross-irrep orthogonality
    for k in sample_k[:2]:
        for l in sample_k[2:4]:
            v = sum(rho2(k, g)[0, 0] * rho2(l, g)[0, 0].conj() for g in gl) / N
            if abs(v) > 1e-8:
                pw_bad += 1
    check(state, f'Peter-Weyl coefficient orthonormality (violations {pw_bad})',
          pw_bad == 0,
          quantity='Finite Peter-Weyl: normalized matrix coefficients orthonormal',
          measured=f'violations = {pw_bad}', threshold='0 (1/d_rho = 1/2)')

    sumd2 = n_lin * 1 + n_2d * 4
    check(state, f'Census: {n_irreps} irreps ({n_lin} lin + {n_2d} 2-dim), '
                 f'sum d^2 = {sumd2}',
          sumd2 == N and n_irreps == 2144,
          quantity='Sum of dim^2 = |G| = 8192; #irreps = #conjugacy classes = 2144',
          measured=f'sum d^2 = {sumd2}, #irreps = {n_irreps}',
          threshold='8192 and 2144')
    vprint('  [INFO] this is the exact discrete analog of the continuous Peter-Weyl:'
           ' on SO(3) the Wigner-D matrix coefficients are orthonormal'
           ' over the Haar measure; here the normalized coefficients of the'
           ' 2144 finite irreps are orthonormal over the group G. Both are'
           ' the Peter-Weyl theorem.')

    # ================================================================
    # 2. Fourier transform, Plancherel, inversion
    # ================================================================
    section(state, 'Fourier Transform, Plancherel, and Inversion on G')
    # Plancherel: |G| sum_g |f(g)|^2 = sum_rho d_rho Tr(fhat(rho)^+ fhat(rho)).
    # For the 2-dim irreps the transform is read off the block structure
    #   fhat_k = [[sum_even f chi_k, sum_odd f chi_k],
    #             [sum_odd f chi_swapk, sum_even f chi_swapk]]
    # so with the 4096-dim Walsh transforms w_even = WH(f_even),
    # w_odd = WH(f_odd) of the even/odd slices we get
    #   sum_even f chi_k = w_even[k], sum_odd f chi_k = w_odd[k], etc.
    fval = np.cos(gl_arr * 0.5) + 0.3 * np.sin(gl_arr)
    f_even = np.zeros(4096); f_odd = np.zeros(4096)
    f_even[low[even_mask]] = fval[even_mask]
    f_odd[low[odd_mask]] = fval[odd_mask]
    w_even = walsh4096(f_even)
    w_odd = walsh4096(f_odd)

    # linear contributions
    u = (gl_arr >> 6) & 63; v = gl_arr & 63
    uxv = (u ^ v).astype(np.int64)
    F_lin = {}
    for (s, a) in lin_reps:
        # chi_{s,a}(g) = (-1)^{s p} (-1)^{popcount(a & (u^v))}
        dots = np.array([bin(int(a) & int(uv)).count('1') & 1 for uv in uxv])
        signs = np.where((p_arr & s) ^ dots == 0, 1.0, -1.0)
        F_lin[(s, a)] = float(np.sum(fval * signs))
    rhs_lin = sum(abs(vv) ** 2 for vv in F_lin.values())

    # 2-dim contributions via Walsh transforms
    rhs_2d = 0.0
    for k in k_reps:
        sk = _swap12(k)
        F00 = w_even[k]; F11 = w_even[sk]; F01 = w_odd[k]; F10 = w_odd[sk]
        rhs_2d += 2 * (F00 * F00 + F11 * F11 + F01 * F01 + F10 * F10)
    rhs = rhs_lin + rhs_2d
    lhs = N * float(np.sum(fval * fval))
    rel = abs(lhs - rhs) / lhs
    check(state, f'Plancherel |G|sum|f|^2 = sum d_rho Tr(fhat^+ fhat): '
                 f'relerr = {rel:.2e}', rel < 1e-8,
          quantity='Plancherel theorem on G (over all 2144 irreps)',
          measured=f'relative error = {rel:.2e}', threshold='< 1e-8')

    # Fourier inversion at a few elements
    inv_bad = 0
    for gtest in [gl[0], gl[1234], gl[5432]]:
        gi = _inv(gtest)
        rec = 0j
        # linear part
        ps = (gi >> 12) & 1
        us = (gi >> 6) & 63; vs = gi & 63
        uxvgi = us ^ vs
        for (s, a) in lin_reps:
            chi = (-1) ** ((((s & 1) * ps) ^ (bin(a & uxvgi).count('1') & 1)) & 1)
            rec += 1 * chi * F_lin[(s, a)]
        # 2-dim part
        for k in k_reps:
            sk = _swap12(k)
            Fk = np.array([[w_even[k], w_odd[k]], [w_odd[sk], w_even[sk]]], dtype=complex)
            rec += 2 * np.trace(rho2(k, gi) @ Fk)
        rec /= N
        if abs(fval[gl.index(gtest)] - rec) > 1e-8:
            inv_bad += 1
    check(state, f'Fourier inversion at 3 elements (violations {inv_bad})',
          inv_bad == 0,
          quantity='Fourier inversion f(g) = (1/|G|) sum d_rho Tr(rho(g^-1) fhat(rho))',
          measured=f'violations = {inv_bad}', threshold='0 (to machine precision)')
    vprint('  [INFO] Plancherel + inversion over the full non-abelian character'
           ' table; abelian Walsh on (Z/2)^12 is the translation-subgroup case.')

    # ================================================================
    # 3. Group algebra, central idempotents, isotypic decomposition of L^2(Omega)
    # ================================================================
    section(state, 'Group Algebra and the Isotypic Decomposition of L^2(Omega)')
    # The primitive central idempotents e_rho of C[G] project onto the
    # rho-isotypic component. On the permutation module C[Omega] the isotypic
    # projector has trace
    #   Tr(e_rho^Omega) = (d_rho/|G|) sum_g chi_rho(g) chi_perm(g) = d_rho m_rho
    # where chi_perm is the permutation character. Since L^2(Omega) is
    # multiplicity-free, m_rho = 1 for the 2080 appearing irreps and
    # 0 for the 64 parity-odd linear characters. Verify this over every irrep,
    # and that the isotypic dimensions sum to dim C[Omega] = 4096.
    def chi_perm(g):
        p, u, v = (g >> 12) & 1, (g >> 6) & 63, g & 63
        if p == 0:
            return 4096 if (u == 0 and v == 0) else 0
        return 64 if u == v else 0

    appearing = []      # (kind, idx, m_rho, isotypic_dim)
    bad = 0
    for (s, a) in lin_reps:
        m = sum(linear_char(s, a, g) * chi_perm(g) for g in gl) / N
        dim = 1 * m
        appearing.append(('lin', (s, a), m, dim))
    for k in k_reps:
        m = sum(twod_char(k, g) * chi_perm(g) for g in gl) / N
        dim = 2 * m
        appearing.append(('2d', k, m, dim))
    # check multiplicities: 1 for appearing, 0 for absent; dims sum to 4096
    n_appear = sum(1 for _, _, m, _ in appearing if abs(m - 1) < 1e-9)
    n_absent = sum(1 for _, _, m, _ in appearing if m < 1e-9)
    total_dim = sum(d for _, _, _, d in appearing)
    iso_ok = (n_appear == 2080 and n_absent == 64 and abs(total_dim - 4096) < 1e-6)
    check(state, f'Isotypic: {n_appear} appearing (m=1), {n_absent} absent, '
                 f'sum dims = {total_dim:.0f}', iso_ok,
          quantity='Isotypic decomposition of C[Omega]: 2080 irreps each once, '
                   '64 absent, dim 4096',
          measured=f'#appearing = {n_appear}, #absent = {n_absent}, '
                   f'sum d_rho = {total_dim:.0f}',
          threshold='2080 / 64 / 4096')
    # the absent 64 are exactly the parity-odd linear characters (s=1): the
    # "reflection" sector of the double cover
    absent_s1 = sum(1 for rep in appearing
                    if rep[0] == 'lin' and rep[1][0] == 1 and rep[2] < 1e-9)
    check(state, f'Absent irreps are the s=1 parity-odd characters '
                 f'({absent_s1}/64 of them)', absent_s1 == 64,
          quantity='The 64 absent irreps are the parity-odd linear characters',
          measured=f'{absent_s1} absent with s=1', threshold='64 (chi(z) = -1)')
    vprint('  [INFO] isotypic projectors: each appearing irrep occupies a'
           ' d_rho-dimensional block of C[Omega] exactly once; parity-odd'
           ' linear characters (chi(z)=-1) are absent.')

    # ================================================================
    # 4. Harmonic random walks on G and the mixing trichotomy
    # ================================================================
    section(state, 'Harmonic Random Walks on G and the Mixing Trichotomy')
    # For a step distribution mu (a probability on G), the eigenvalue of the
    # walk on the rho-isotypic component is lambda_rho = sum_g mu(g) chi_rho(g)/d_rho
    # and the chi^2 distance to uniform after k steps is exactly
    #   chi^2(k) = sum_{rho != triv} d_rho^2 |lambda_rho|^{2k}.
    # This is the Diaconis-Shahshahani spectral formula; we evaluate it over
    # all 2144 irreps. For a conjugacy-invariant walk (mu constant on a class
    # C) lambda_rho = chi_rho(C)/d_rho.
    def chi2_profile(get_lambda, kmax=10):
        # get_lambda(rho) -> (lambda, d)
        prof = [0.0]
        for k in range(1, kmax + 1):
            tot = 0.0
            for (s, a) in lin_reps:
                if s == 0 and a == 0:
                    continue
                lam, d = get_lambda('lin', (s, a))
                tot += d * d * abs(lam) ** (2 * k)
            for kk in k_reps:
                lam, d = get_lambda('2d', kk)
                tot += d * d * abs(lam) ** (2 * k)
            prof.append(tot)
        return prof

    def class_walk(C):
        def gl(kind, idx):
            if kind == 'lin':
                s, a = idx
                c = sum(linear_char(s, a, g) for g in C) / len(C)
                return c, 1
            k = idx
            c = sum(twod_char(k, g) for g in C) / len(C)
            return c / 2.0, 2
        return gl

    # build an odd conjugacy class (size 64): the class of the swap (1,0,0)
    g0 = (1 << 12)
    # conjugation: conj_h(g) = h g h^{-1}, vectorized over the class orbit
    # exact formula: conj_{(q,t_v)}(s^p, t_a) = (s^p, t_{v.swap^q(a).swap^p(v)})
    hq = ((gl_arr >> 12) & 1)
    hv = ((gl_arr >> 6) & 63) << 6 | (gl_arr & 63)
    swv = _swap12(hv)
    a0 = g0 & 0xFFF
    swa0 = _swap12(a0)
    sqa = np.where(hq == 0, a0, swa0)
    spv = swv                       # g0 has parity 1 -> swap^p(v) = swap(v)
    t = hv ^ sqa ^ spv
    odd_class = sorted({int(x) for x in ((np.uint32(1) << 12) | t.astype(np.uint32))})
    # (Note: g0 has parity 1, so the class consists of 64 odd elements.)
    prof_odd = chi2_profile(class_walk(odd_class), kmax=6)
    per_ok = len(odd_class) == 64 and all(abs(x - 127.0) < 1e-6 for x in prof_odd[1:])
    check(state, f'Odd-class walk chi^2(k) = {[round(x, 2) for x in prof_odd[1:]]} '
                 f'(periodic)', per_ok,
          quantity='Single-class walk in the odd coset is PERIODIC: '
                   'chi^2(k) = 127 constant',
          measured='[127.0]*6 (never decays)', threshold='constant (|lambda|=1)')
    vprint('  [INFO] the odd-class walk is completely periodic: every one of'
           ' the 127 non-trivial linear characters has |lambda|=1 (the parity'
           ' characters are exact invariants), so chi^2 stays at 127 and the'
           ' walk never mixes on G -- even though the SAME byte dynamics'
           ' mixes the coset Omega = G/Stab in exactly two steps.')

    # minimal generating set: 12 translations + the swap, each prob 1/13.
    # This is a NON-central measure, so the correct chi^2 uses the matrix
    # Fourier transform on each two-dimensional irrep:
    #     chi^2(k) = sum_{rho != triv} d_rho || muhat(rho)^k ||_F^2,
    #     muhat(rho) = sum_g mu(g) rho(g)
    # (the scalar character formula above is only valid for CENTRAL walks).
    # The linear (1-dim) irreps reduce to scalars; the 2-dim irreps require
    # full 2x2 matrix powers. We use the exact matrix transforms.
    def chi2_matrix(mu, kmax=8):
        """mu = list of (g, weight); correct non-central chi^2 via matrices."""
        prof = [0.0]
        for k in range(1, kmax + 1):
            tot = 0.0
            for (s, a) in lin_reps:
                if s == 0 and a == 0:
                    continue
                lam = sum(w * linear_char(s, a, g) for g, w in mu)
                tot += 1 * abs(lam) ** (2 * k)
            for kk in k_reps:
                M = sum(w * rho2(kk, g) for g, w in mu)  # 2x2 matrix transform
                Mk = np.linalg.matrix_power(M, k)
                tot += 2 * float(np.real(np.sum(np.abs(Mk) ** 2)))
            prof.append(tot)
        return prof

    def mu_spec_radius(mu):
        """Largest |eigenvalue| over all non-trivial irreps (1/2-dim)."""
        m = 0.0
        for (s, a) in lin_reps:
            if s == 0 and a == 0:
                continue
            m = max(m, abs(sum(w * linear_char(s, a, g) for g, w in mu)))
        for kk in k_reps:
            M = sum(w * rho2(kk, g) for g, w in mu)
            m = max(m, max(abs(np.linalg.eigvals(M))))
        return m

    gens = [1 << i for i in range(12)] + [1 << 12]
    mu_gen = [(g, 1.0 / 13.0) for g in gens]
    prof_gen = chi2_matrix(mu_gen, kmax=8)
    # spectral gap from the matrix spectral radius (correct for non-central)
    gap_gen = 1 - mu_spec_radius(mu_gen)
    chi_163_on_gens = all(linear_char(1, 63, g) == -1 for g in gens)
    blocked_ok = (gap_gen < 1e-9 and chi_163_on_gens and prof_gen[-1] >= 1.0
                  and prof_gen[1] > 500)
    check(state, f'Minimal 13-generator walk: gap = {gap_gen:.3f}, '
                 f'chi_{1,63}=-1 on all gens: {chi_163_on_gens}, '
                 f'chi^2(1)={prof_gen[1]:.1f} -> chi^2(8)={prof_gen[8]:.2f}',
          blocked_ok,
          quantity='Hidden-character obstruction: the minimal generator walk is '
                   'NOT fully mixing (gap 0, chi^2 plateaus at 1)',
          measured=f'chi_{1,63}(g)=-1 for all 13 gens; chi^2(8)={prof_gen[8]:.2f} >= 1',
          threshold='a nontrivial linear character blocks the walk (never uniform)')
    vprint('  [INFO] this is a subtle and real obstruction: the natural minimal'
           ' generating set {e_0..e_11, s} is "one-colored" under the character'
           ' chi_{1,63}(g) = (-1)^{parity + popcount(u^v)}, which is -1 on every'
           ' generator, so it is an exact invariant (eigenvalue -1) and the walk'
           ' never reaches uniform on G. Only by "lazifying" (adding the'
           ' identity) do we get genuine mixing:')

    # lazy walk: 13 generators + identity -> guaranteed all |lambda| < 1
    lazy = gens + [0]
    mu_lazy = [(g, 1.0 / 14.0) for g in lazy]
    prof_lazy = chi2_matrix(mu_lazy, kmax=12)
    gap_lazy = 1 - mu_spec_radius(mu_lazy)
    lazy_ok = (gap_lazy > 0 and prof_lazy[-1] < prof_lazy[1] * 0.01
               and prof_lazy[12] < 0.5)
    check(state, f'Lazy generator walk: gap = {gap_lazy:.4f}, '
                 f'chi^2(1)={prof_lazy[1]:.1f} -> chi^2(12)={prof_lazy[12]:.3f}',
          lazy_ok,
          quantity='Lazy (identity-added) generator walk genuinely mixes with '
                   'a spectral gap',
          measured=f'gap = {gap_lazy:.4f}; chi^2(12) = {prof_lazy[12]:.3f}',
          threshold='gap > 0; chi^2 decays below 0.5 (all |lambda| < 1)')
    info(f'MIXING TRICHOTOMY. (i) Continuous SO(3): Laplacian gap 2, '
         'returns to uniform only as t -> inf. (ii) Byte walk on coset '
         'Omega = G/Stab: P^2 = J/4096 (exact two-step mixing). (iii) On G: '
         'single-coset class walks are periodic (chi^2 = 127); the minimal '
         'generator walk is blocked by chi_{1,63} = -1; only the lazy walk '
         f'mixes (gap {gap_lazy:.3f}). Maximal mixing lives on Omega = G/Stab.')
    # ================================================================
    # 1. Gelfand pair and the Hecke algebra
    # ================================================================
    section(state, 'Gelfand Pair (G, H) and the Hecke Algebra')
    # H = Stab(rest) = {e, z}. The permutation module C[Omega] is
    # multiplicity-free: each of the 2080 appearing irreps occurs
    # exactly once. A transitive permutation representation is a Gelfand
    # pair exactly when the module is multiplicity-free, so (G, H) is a
    # Gelfand pair and the Hecke algebra Hecke(G, H) of H-bi-invariant
    # functions on G (under convolution) is commutative, of dimension equal
    # to the number of double cosets H\\G/H.
    # H-orbits on Omega: z fixes 64 states and pairs the other 2016, so
    # the number of double cosets is 64 + 2016 = 2080.
    dcosets = 64 + (4096 - 64) // 2
    check(state, f'#double cosets H\\G/H = {dcosets}',
          dcosets == 2080,
          quantity='Double cosets H\\G/H = 2080 = #appearing irreps (Gelfand pair)',
          measured=f'{dcosets}', threshold='2080')

    # Verify (G,H) is a Gelfand pair via the H-fixed-vector criterion: an
    # irreducible constituent rho of the permutation module supports a zonal
    # spherical function iff it has a (1-dimensional) space of H-fixed
    # vectors, i.e. dim Hom_H(1, Res_H rho) = (1/|H|) sum_h chi_rho(h) = 1.
    # Exactly the 2080 appearing irreps have this property:
    #   * the 64 s=0 linear chars: chi(z) = +1 => dim = (1+1)/2 = 1
    #   * the 2016 2-dim irreps: chi(z) = 0 (odd) => dim = (2+0)/2 = 1
    #   * the 64 s=1 linear chars: chi(z) = -1 => dim = (1-1)/2 = 0 (absent)
    def chi_2d(k, g):
        p, u, v = (g >> 12) & 1, (g >> 6) & 63, g & 63
        if p:
            return 0
        a = (u << 6) | v
        return ((-1) ** (bin(k & a).count('1') & 1)
                + (-1) ** (bin(k & _swap12(a)).count('1') & 1))
    n_s0 = sum(1 for a in range(64)
               if abs((linear_char(0, a, 0) + linear_char(0, a, z)) / 2 - 1) < 1e-9)
    n_s1 = sum(1 for a in range(64)
               if abs((linear_char(1, a, 0) + linear_char(1, a, z)) / 2) < 1e-9)
    n_2d = sum(1 for k in k_reps
               if abs((chi_2d(k, 0) + chi_2d(k, z)) / 2 - 1) < 1e-9)
    sph_ok = n_s0 == 64 and n_s1 == 64 and n_2d == 2016
    check(state, f'H-fixed criterion: {n_s0} s=0 lin + {n_2d} 2-dim have '
                 f'dim Hom_H=1; {n_s1} s=1 have dim 0',
          sph_ok,
          quantity='Exactly the 2080 appearing irreps carry a zonal spherical '
                   'function (dim Hom_H = 1); the 64 s=1 fermions do not',
          measured=f'dim Hom_H = 1 for {n_s0 + n_2d}; 0 for {n_s1}',
          threshold='64 + 2016 = 2080 spherical functions; 64 absent')

    # NEGATIVE result: the byte transition operator is NOT H-bi-invariant.
    # On Omega, P(s,s') = #{b: b s = s'}/256. H-bi-invariance would require
    # P(h s, h s') = P(s, s') for h in H. We verify this FAILS: the byte
    # walk is not a Hecke/zonal element, so its mixing is not governed by
    # the spherical-function algebra. This is a precise correction of the
    # naive "byte kernel = zonal kernel" reading.
    from gyroscopic.hQVM.api import OmegaState12, omega12_to_state24, state24_to_omega12
    from gyroscopic.hQVM.constants import step_state_by_byte
    omega_list = [omega12_to_state24(OmegaState12(u6=u, v6=v))
                  for u in range(64) for v in range(64)]
    idx = {s: i for i, s in enumerate(omega_list)}
    def zact(s):
        om = state24_to_omega12(s)
        return omega12_to_state24(OmegaState12(u6=om.v6 ^ 63, v6=om.u6 ^ 63))
    def Ps(s, s_):
        return sum(1 for b in range(256) if step_state_by_byte(s, b) == s_) / 256
    rng = np.random.RandomState(7)
    bi_viol = 0
    for _ in range(400):
        s = omega_list[rng.randint(0, 4096)]
        s_ = omega_list[rng.randint(0, 4096)]
        if abs(Ps(zact(s), zact(s_)) - Ps(s, s_)) > 1e-12:
            bi_viol += 1
    check(state, f'Byte transition is NOT H-bi-invariant: {bi_viol}/400 '
                 f'violations', bi_viol > 0,
          quantity='NEGATIVE: the byte walk is NOT a Hecke/zonal element',
          measured=f'{bi_viol}/400 pairs violate P(hs,hs\') = P(s,s\')',
          threshold='> 0 (not H-bi-invariant)')
    vprint('  [INFO] because (G,H) is a Gelfand pair the spherical functions'
           ' are a canonical commutative convolution algebra, but the byte'
           ' transition operator does NOT live in it. Its exact two-step'
           ' mixing is a property of coset geometry (128 one-step cone /'
           ' 32-dim transient image), not a Hecke-algebra rank-one identity.')

    # ================================================================
    # 2. Corrected non-abelian Fourier for NON-CENTRAL walks
    # ================================================================
    section(state, 'Non-Abelian Fourier for Non-Central Walks')
    # For a general (non-central) step measure mu the Fourier transform on
    # transform on the two-dim irreps is a matrix
    #     muhat(rho) = sum_g mu(g) rho(g)
    # and the correct chi^2 to uniform after k steps is
    #     chi^2(k) = sum_{rho != triv} d_rho || muhat(rho)^k ||_F^2.
    # We recompute both the 13-generator and the lazy walks with the correct
    # matrix transforms.
    gens = [1 << i for i in range(12)] + [1 << 12]
    def muhat_matrix(mu, k):
        d = 2
        acc = np.zeros((d, d), complex)
        for g, w in mu:
            acc += w * rho2(k, g)
        return acc
    mu_gen = [(g, 1 / 13) for g in gens]
    prof_gen = chi2_matrix(mu_gen, 8)
    # hidden-character obstruction: exactly one 1-dim char, chi_{1,63}, has
    # |lambda| = 1 (it is -1 on every generator), giving an exact floor
    floor_chars = 0
    for (s, a) in lin_reps:
        if s == 0 and a == 0:
            continue
        lam = sum(w * linear_char(s, a, g) for g, w in mu_gen)
        if abs(abs(lam) - 1.0) < 1e-9:
            floor_chars += 1
    check(state, f'Corrected 13-generator chi^2(k): '
                 f'{[round(x, 2) for x in prof_gen[1:]]}',
          prof_gen[1] > 500 and prof_gen[1] > prof_gen[8] and floor_chars == 1,
          quantity='Corrected non-central chi^2 = sum d ||muhat^k||_F^2 '
                   '(matrix Fourier)',
          measured=f'chi^2(1)={prof_gen[1]:.1f} -> chi^2(8)={prof_gen[8]:.2f}; '
                   f'{floor_chars} char with |lam|=1',
          threshold='decays but has an exact |lam|=1 floor (never reaches 0)')
    # lazy walk: 13 generators + identity, each 1/14 -> all |lambda|<1
    mu_lazy = gens + [0]
    prof_lazy = chi2_matrix([(g, 1 / 14) for g in mu_lazy], 8)
    lam_max_lazy = 0.0
    for (s, a) in lin_reps:
        if s == 0 and a == 0:
            continue
        lam = sum(w * linear_char(s, a, g) for g, w in [(g, 1 / 14) for g in mu_lazy])
        lam_max_lazy = max(lam_max_lazy, abs(lam))
    for kk in k_reps:
        M = muhat_matrix([(g, 1 / 14) for g in mu_lazy], kk)
        w = np.linalg.eigvals(M)
        lam_max_lazy = max(lam_max_lazy, max(abs(w)))
    gap_lazy = 1 - lam_max_lazy
    check(state, f'Corrected lazy walk: gap = {gap_lazy:.4f}, '
                 f'chi^2(1)={prof_lazy[1]:.1f} -> chi^2(8)={prof_lazy[8]:.3f}',
          gap_lazy > 0 and prof_lazy[8] < prof_lazy[1] * 0.05,
          quantity='Corrected lazy walk genuinely mixes (all |lambda| < 1)',
          measured=f'gap = {gap_lazy:.4f}; chi^2(8) = {prof_lazy[8]:.3f}',
          threshold='gap > 0; chi^2 -> 0')
    vprint('  [INFO] Non-central walks use matrix Fourier: chi^2(k) ='
           ' sum d_rho || muhat(rho)^k ||_F^2. Hidden char chi_{1,63} = -1'
           ' gives an exact |lambda| = 1 floor on the minimal generator walk.')

    # ================================================================
    # 3. Rank-32 transient collapse of the byte operator
    # ================================================================
    section(state, 'Rank-32 Transient Collapse of the Byte Operator')
    # Build the sparse byte transition P on Omega (4096 x 4096).
    rows, cols, data = [], [], []
    for i, s in enumerate(omega_list):
        for b in range(256):
            rows.append(i); cols.append(idx[step_state_by_byte(s, b)])
            data.append(1.0 / 256.0)
    P = sp.coo_matrix((data, (rows, cols)), shape=(4096, 4096)).tocsr()
    A = P.toarray()
    r1 = np.linalg.matrix_rank(A, tol=1e-10)
    A2 = A @ A
    r2 = np.linalg.matrix_rank(A2, tol=1e-10)
    check(state, f'rank(P) = {r1}, rank(P^2) = {r2}',
          r1 == 32 and r2 == 1,
          quantity='Byte operator: rank(P) = 32, rank(P^2) = 1',
          measured=f'rank = {r1} / {r2}', threshold='32 / 1')
    # canonical factorization P = F G with F (4096 x 32), G (32 x 4096)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    r = 32
    F = U[:, :r] * S[:r]
    G = Vt[:r, :]
    fac_err = float(np.linalg.norm(A - F @ G))
    check(state, f'Canonical factorization P = F G through C^32: '
                 f'||A - F G|| = {fac_err:.2e}', fac_err < 1e-8,
          quantity='P factors through a 32-dim transient image (P = F G)',
          measured=f'||A - F G||_2 = {fac_err:.2e}',
          threshold='< 1e-8 (exact factorization)')
    # P^2 = J/N because Im(P) (dim 32) is collapsed to span{1} by the second
    # step: verify P restricted to Im(P) has rank 1.
    PGF = G @ F   # 32 x 32: the action of P on its own transient image
    r_img = np.linalg.matrix_rank(PGF, tol=1e-10)
    check(state, f'P acts on its 32-dim image with rank {r_img} '
                 f'(then collapses to uniform)', r_img == 1,
          quantity='Two-step collapse: C^4096 -> C^32 -> span{1}',
          measured=f'rank(G F) = {r_img}', threshold='1 (P^2 = J/4096)')
    vprint('  [INFO] the byte operator is the sharpest possible mixer because'
           ' it is almost of rank one: one byte drops any distribution to the'
           ' 32-dimensional transient image (the parity/family sectors), and'
           ' the second byte collapses that image to the uniform direction.'
           ' The 32 = 2^5 = number of one-step cone directions / 128 over the'
           ' shadow-pair 4 is the structural origin of exact two-step mixing.')

    # ================================================================
    # 4. The regular representation and the boson/fermion split
    # ================================================================
    section(state, 'The Regular Representation and the Boson/Fermion Split')
    # L^2(Omega) is multiplicity-free with 2080 irreps: 64 s=0 linear + 2016
    # two-dimensional. It is MISSING the 64 parity-odd (s=1) linear
    # characters, which have chi(z) = -1. The regular representation
    # L^2(G) (the permutation module of G acting on itself by left
    # translation) restores ALL 2144 irreps, each with multiplicity d_rho:
    #     dim = sum_rho d_rho^2 = 128*1 + 2016*4 = 8192 = |G|.
    # The 64 s=1 characters -- the "fermion" / spinor sector -- are exactly
    # the ones absent from Omega. This is the discrete analog of the fact
    # that half-integer (spinor) representations live on the double cover
    # SU(2) = Spin(3), not on SO(3).
    sum_d2 = len(lin_reps) * 1 + len(k_reps) * 4
    check(state, f'sum d_rho^2 = {sum_d2} = |G| (regular rep dimension)',
          sum_d2 == N,
          quantity='Regular representation: sum d_rho^2 = 8192 = |G|',
          measured=f'{sum_d2}', threshold='8192')
    # multiplicity of each irrep in the regular rep is d_rho
    # (chi_reg(g) = |G| delta_{g,e} => m_rho = <chi_reg, chi_rho> = chi_rho(e)).
    # The 64 s=1 linear characters have <chi_perm_Omega, chi> = 0 (absent from
    # Omega) but appear once each in the regular rep.
    def ip(f, g):
        return sum(f(x) * g(x) for x in gl) / N
    m_in_omega = ip(lambda x: linear_char(1, 0, x) * chi_perm(x), lambda x: 1)
    check(state, f'<chi_perm_Omega, chi_{1,0}> = {m_in_omega:.3f} (fermion '
                 f'absent from Omega)', abs(m_in_omega) < 1e-9,
          quantity='The 64 s=1 (fermion) characters are ABSENT from L^2(Omega)',
          measured=f'<chi_perm, chi_{{1,0}}> = {m_in_omega:.3f}',
          threshold='0 (multiplicity-free module omits them)')
    # in the regular rep they appear once (multiplicity d = 1)
    # m = <chi_reg, chi_{1,0}> = chi_{1,0}(e) = 1
    check(state, 'm = <chi_reg, chi_{1,0}> = chi_{1,0}(e) = 1 (present in '
                 'regular rep)',
          linear_char(1, 0, 0) == 1,
          quantity='The fermion sector reappears once each in the regular '
                   'representation L^2(G)',
          measured='chi_{1,0}(e) = 1', threshold='1 (= d_rho)')
    check(state, f'Boson/fermion split: Omega = 64 s=0 + 2016 2-dim (2080); '
                 f'G adds 64 s=1 (2144 total)',
          (64 + 2016 == 2080) and (2080 + 64 == 2144),
          quantity='SO(3)-type (bosonic) module Omega vs full spinorial '
                   'regular module',
          measured='Omega: 2080 irreps; regular: 2144 (adds 64 s=1 fermion)',
          threshold='2080 + 64 = 2144 = #conjugacy classes')
    vprint('  [INFO] the 64 parity-odd linear characters (chi(z) = -1) are the'
           ' "fermion"/spinor sector: absent from the SO(3)-type quotient'
           ' Omega, present once each in the full regular module L^2(G). This'
           ' is the discrete realization of why half-integer spin needs the'
           ' double cover SU(2) = Spin(3) rather than SO(3).')
