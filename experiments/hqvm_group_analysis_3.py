#!/usr/bin/env python3
"""hqvm_group_analysis_3.py — Structural limits, 2-design, central holonomy.

Role: PASS/FAIL gates on no-go theorems, combinatorial 2-design, GF(2)^6 holonomy.
Inputs: hQVM kernel via hqvm_group_analysis_common.
Outputs: gates appended to ReportState.
Companion: hqvm_group_analysis_1.py, hqvm_group_analysis_2.py, hqvm_group_analysis_run.py.
"""
from __future__ import annotations
import sys, math
import numpy as np
from hqvm_group_analysis_common import (
    _SCIPY_OK, _KERNEL_OK,
    kernel_group, compose_sig_int, apply_sig_int,
    omega12_to_state24, state24_to_omega12, OmegaState12,
    byte_transition_matrix,
    rp_n_boundary_map, haar_mean_angle, uniform_random_rotation,
    rotation_angle_from_matrix,
    ReportState, section, check, vprint, info,
)
if _SCIPY_OK:
    import scipy.sparse as sp


def _inv(g):
    p, u, v = (g >> 12) & 1, (g >> 6) & 63, g & 63
    return (p << 12) | ((v if p else u) << 6) | (u if p else v)


def _swap12(x):
    return ((x & 63) << 6) | ((x >> 6) & 63)

def run(state):
    if not _SCIPY_OK or not _KERNEL_OK:
        print('hqvm_group_analysis_3.py requires scipy and the hQVM kernel')
        return

    gl = list(kernel_group())
    N = len(gl)

    section(state, 'No-go: G Does Not Embed Faithfully into SO(3)')
    # Classification of finite subgroups of SO(3) (Coxeter): cyclic C_n,
    # dihedral D_2n, tetrahedral A_4, octahedral S_4, icosahedral A_5. The
    # Sylow 2-subgroups are therefore cyclic 2-groups or dihedral 2-groups,
    # each of 2-rank at most 2. G contains the elementary abelian group
    # (Z/2)^12 (all even words) of 2-rank 12, so no faithful homomorphism
    # G -> SO(3) exists.
    #
    # We verify the structural facts numerically: G contains an elementary
    # abelian subgroup of order 4096 = 2^12, and it has many involutions
    # (far more than a cyclic/dihedral 2-group allows).
    even_words = [g for g in gl if (g >> 12) & 1 == 0]
    # even words = (Z/2)^12: abelian, every nonzero element order 2
    is_eab = True
    for _ in range(1500):
        a = np.random.choice(even_words); b = np.random.choice(even_words)
        if compose_sig_int(a, b) != compose_sig_int(b, a):
            is_eab = False
            break
    n_inv = sum(1 for g in gl if compose_sig_int(g, g) == 0)
    check(state, f'G contains elementary abelian (Z/2)^12 ({len(even_words)} '
                 f'even words); #involutions = {n_inv}',
          is_eab and len(even_words) == 4096 and n_inv > 4096,
          quantity='G contains (Z/2)^12 (2-rank 12) and >4096 involutions',
          measured=f'|even| = {len(even_words)}, #involution = {n_inv}',
          threshold='2-rank 12 > 2 => not a subgroup of SO(3)')
    info('Coxeter classification: finite 2-subgroups of SO(3) have 2-rank '
         'at most 2, so (Z/2)^12 cannot embed faithfully in SO(3).')
    info('this is a boundary, not a failure: it fixes the only legitimate '
         'bridges between G and SO(3) -- approximation (a codebook), quotient '
         '(Omega = G/H), higher-dimensional unitary representations, or a '
         'continuum limit. The engine is a quantized rotation-composition '
         'engine, never a faithful subgroup.')

    # ================================================================
    # 2. Failure certificate for the se(3) bracket claim
    # ================================================================
    section(state, 'Failure Certificate: The 6 Payload Bits are not se(3)')
    # The reading "six payload bits = six se(3) generators (3 rotation + 3
    # translation)" is a coordinate count. A genuine se(3) = su(2) x| R^3
    # bracket realization requires [J_i, J_j] = eps_ijk J_k,
    # [J_i, P_j] = eps_ijk P_k, [P_i, P_j] = 0. Here the translation
    # subgroup (Z/2)^12 is abelian (all [P_i, P_j] = 0 trivially, but also
    # no [J,P] = P coupling), and the outer Z/2 acts by the SWAP (u,v)->(v,u),
    # which is not the SO(3) vector action on R^3.
    # Verify: the 12-bit translation group is abelian; the swap is an
    # involution that does not reproduce the so(3) vector-action structure.
    transl_abelian = True
    for _ in range(2000):
        a = np.random.choice(even_words); b = np.random.choice(even_words)
        if compose_sig_int(a, b) != compose_sig_int(b, a):
            transl_abelian = False
            break
    # a nonzero "translation" generator and the swap: does [swap, J] = J
    # (vector-action) hold? swap is order 2; the bracket of the swap with a
    # translation is a translation (conjugation action swap(a)), which IS
    # nonzero - but it is the Z/2 swap action, giving (Z/2)^12 x| Z/2, not
    # se(3) (whose translation algebra is R^3 over the reals, with a
    # 3-dim vector action, not a binary swap).
    swap_ok = True
    odd_sample = [g for g in gl if (g >> 12) & 1 == 1][:500]
    for g in odd_sample:
        u, v = (g >> 6) & 63, g & 63
        for test_u, test_v in ((17, 42), (0, 63), (63, 0)):
            om = OmegaState12(u6=test_u, v6=test_v)
            out = state24_to_omega12(apply_sig_int(g, omega12_to_state24(om)))
            if out.u6 != (test_v ^ u) or out.v6 != (test_u ^ v):
                swap_ok = False
                break
        if not swap_ok:
            break
    check(state, f'Translation subgroup abelian: {transl_abelian}',
          transl_abelian,
          quantity='Translations (Z/2)^12 are abelian (no [J,P]=P coupling)',
          measured=f'abelian = {transl_abelian}', threshold='True')
    check(state, f'Odd coset is swap-XOR action (500 samples): {swap_ok}',
          swap_ok,
          quantity='Outer Z/2 is (u,v)->(v,u) with translation, not SO(3) on R^3',
          measured=f'swap-XOR law on odd elements = {swap_ok}',
          threshold='True on samples')
    info('the six payload bits are six carrier coordinates, not six se(3) '
         'generators. A genuine se(3) bracket realization would need the '
         'translations to carry a 3-dim vector action with [J,P]=P; the '
         'finite engine instead carries the binary swap action. The "6 DoF '
         '/ 12-bit = 2 x 6" identity is a coordinate count, not a '
         'derivation of rigid-body kinematics.')

    # ================================================================
    # 3. Combinatorial design and the strict-expander property
    # ================================================================
    section(state, 'Haar Measure on SO(3) and Exact Mixing on Omega')
    # Haar density on rotation angle: f(theta) = (2/pi) sin^2(theta/2).
    # E[theta] = pi/2 + 2/pi. The byte walk reaches uniform on Omega in two
    # steps — the finite Haar analog on the 4096-state codebook.
    rng = np.random.RandomState(11)
    samp = [rotation_angle_from_matrix(uniform_random_rotation(rng))
            for _ in range(4000)]
    mean_emp = float(np.mean(samp))
    mean_th = haar_mean_angle()
    check(state, f'Haar mean rotation angle: empirical {mean_emp:.4f} vs '
                 f'analytic {mean_th:.4f}',
          abs(mean_emp - mean_th) < 0.04,
          quantity='E[theta] = pi/2 + 2/pi under SO(3) Haar',
          measured=f'empirical = {mean_emp:.4f}', threshold=f'analytic = {mean_th:.4f}')
    vprint('  [INFO] two uniform byte draws give uniform on Omega (P^2 = J/4096):'
           ' the discrete analog of Haar mixing on a finite rotation codebook.')

    section(state, 'Combinatorial Design and the Strict-Expander Property')
    # Exact two-step mixing P^2 = J/4096 is a sharply-uniform 2-design: from
    # any start, the 256^2 = 65536 ordered byte pairs reach each of the 4096
    # target states exactly 16 = 65536/4096 times.
    from gyroscopic.hQVM.constants import step_state_by_byte
    def two_step_multiplicities(start):
        counts = {}
        for b1 in range(256):
            s1 = step_state_by_byte(start, b1)
            for b2 in range(256):
                s2 = step_state_by_byte(s1, b2)
                counts[s2] = counts.get(s2, 0) + 1
        return counts
    c1 = two_step_multiplicities(omega12_to_state24(OmegaState12(u6=0, v6=63)))
    c2 = two_step_multiplicities(omega12_to_state24(OmegaState12(u6=37, v6=5)))
    des_ok = (len(c1) == 4096 and sorted(set(c1.values())) == [16]
              and len(c2) == 4096 and sorted(set(c2.values())) == [16])
    check(state, f'2-step design: from any start each of 4096 targets reached '
                 f'exactly 16 times (256^2/4096)',
          des_ok,
          quantity='Sharply-uniform 2-design: 2-step multiplicity = 16 for '
                   'every ordered state pair',
          measured='multiplicity {16} over all 4096 targets, from 2 starts',
          threshold='65536/4096 = 16 (exact combinatorial origin of P^2 = J)')
    # lambda_2(P)=0 <=> rank(P^2)=1 on the 4096-dim state space
    P, _omega_list = byte_transition_matrix()
    A = P.toarray()
    r2 = np.linalg.matrix_rank(A @ A, tol=1e-10)
    check(state, f'rank(P^2) = {r2} (lambda_2 = 0 on 1-perp)',
          r2 == 1,
          quantity='Non-trivial spectrum collapsed: rank(P^2) = 1 (exact 2-step uniformization)',
          measured=f'rank(P^2) = {r2}', threshold='1 (uniform after two steps)')
    d_eff = 128
    ram = 2 * math.sqrt(d_eff - 1)
    vprint(f'  [INFO] P is non-normal; sharply-uniform 2-design above is the'
           f' mixing statement. Ramanujan bound 2 sqrt(d-1) = {ram:.2f} applies'
           f' to undirected d-regular adjacency spectra.')

    section(state, 'Homology of RP³ and Continuous Z/2 Template')
    # SO(3) ~= RP^3. With one cell per dimension, d_k = 1+(-1)^k on Z gives
    # d_2 = 2, so H_1 = Z/2 and H_2 = 0; H^2(SO(3); Z) = Z/2 is the spinorial
    # torsion echoed by Z/2 plaquette commutators on G below.
    d2_z = rp_n_boundary_map(3, 2, 'Z')
    d2_z2 = rp_n_boundary_map(3, 2, 'Z2')
    check(state, f'RP^3 boundary d_2 = {d2_z} (Z), d_2 mod 2 = {d2_z2}',
          d2_z == 2 and d2_z2 == 0,
          quantity='Cellular d_2 = 2 => H_1 = Z/2, H_2 = 0 for RP^3',
          measured=f'd_2 = {d2_z}, d_2 mod 2 = {d2_z2}', threshold='2 and 0')
    info('H^2(SO(3); Z)=Z/2: closed loops on G carry only a Z/2 gauge '
         'phase, not a continuous U(1) Berry angle.')

    section(state, 'Central Holonomy in GF(2)^6')
    # Native holonomy: plaquette commutators land in Z(G) = GF(2)^6.
    # Scalar +/-1 appears only after choosing a central character (1-dim probe).
    rng = np.random.RandomState(0)
    diag = set((0 << 12) | (d << 6) | d for d in range(64))
    comm_central = True; comm_involution = True; comm_in_diag = True
    for _ in range(4000):
        g = rng.choice(gl); h = rng.choice(gl)
        c = compose_sig_int(compose_sig_int(compose_sig_int(g, h), _inv(g)), _inv(h))
        if c not in diag:
            comm_in_diag = False
        if compose_sig_int(c, c) != 0:
            comm_involution = False
        for x in gl[:20]:
            if compose_sig_int(compose_sig_int(c, x), _inv(x)) != c:
                comm_central = False
                break
    check(state, f'Plaquette commutators central involutions: '
                 f'central={comm_central}, involutive={comm_involution}, '
                 f'in [G,G]={comm_in_diag}',
          comm_central and comm_involution and comm_in_diag,
          quantity='Plaquette commutators land in central GF(2)^6',
          measured='[g,h] in [G,G]=Z(G)=(Z/2)^6, order 1 or 2, central',
          threshold='central, involutive, in (Z/2)^6')
    # In a 2-dim irrep a central involution (0,(d,d)) acts as scalar
    # chi_{2d}(0,(d,d))/2 = (-1)^{k.(d,d)} in {+1,-1}. Verify the Z/2 nature.
    def lin_char(s, a, g):
        p, u, v = (g >> 12) & 1, (g >> 6) & 63, g & 63
        return (-1) ** ((((s & 1) * (p & 1)) ^ (bin(a & (u ^ v)).count('1') & 1)) & 1)
    def chi_2d(k, g):
        p, u, v = (g >> 12) & 1, (g >> 6) & 63, g & 63
        if p:
            return 0
        a = (u << 6) | v
        return ((-1) ** (bin(k & a).count('1') & 1)
                + (-1) ** (bin(k & _swap12(a)).count('1') & 1))
    phases = set()
    for k in [1, 5, 9, 34]:
        for d in range(64):
            gd = (0 << 12) | (d << 6) | d
            phases.add(chi_2d(k, gd) / 2.0)
    check(state, f'Central elements act as scalar phases {sorted(phases)} '
                 f'in the 2-dim irreps',
          set(phases) == {-1.0, 1.0},
          quantity='Central character readout is Z/2-valued (+/-1), not continuous angle',
          measured=f'phases {sorted(phases)}', threshold='{-1, +1}')
    info('native holonomy is GF(2)^6-valued; scalar +/-1 is a chosen central '
         'character readout. H^2(SO(3);Z)=Z/2 is a continuous template only.')
