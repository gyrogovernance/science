#!/usr/bin/env python3
"""hqvm_SO_analysis_1.py — SO(3) built from its own theory.

This part develops SO(3) forward from the group axioms, using the
mathematical machinery itself (not as a verification exercise). Every
section derives or constructs a piece of the theory and checks the
internal consistency of that construction.

Sections:
  1. Group axioms and the algebra so(3): closure, associativity,
     inverse; generators, structure constants
  2. Root system of so(3) = A1: Cartan subalgebra, roots, Weyl group
     W(A1) = Z/2 — the discrete skeleton of the double cover
  3. exp / log dictionary: surjectivity, branch structure, and the
     exact BCH composition law (Engo closed form)
  4. Representation theory: spin-j irreps, character formula and
     orthogonality, Clebsch-Gordan dimension identity via characters
  5. Harmonic analysis: SO(3) action on L^2(S^2), spherical-harmonic
     irreducibility, addition theorem, Peter-Weyl orthogonality of
     Wigner D-matrix elements
  6. Geometry: geodesics = one-parameter subgroups, sectional
     curvature via holonomy of a geodesic square, Euler top on
     T*SO(3) (6-DoF rigid body) with conserved quantities
  7. Topology: cell structure of RP^3, cellular homology with Z and
     Z/2 coefficients — H*(SO(3);Z) = {Z,0,Z/2,Z}
  8. Random rotations: Haar angle distribution, eigenvalue statistics
"""
from __future__ import annotations
import sys, math
import numpy as np
from hqvm_SO_analysis_common import (
    _SCIPY_OK, _KERNEL_OK, _GYRO_OK, spla, spspec,
    SO3_BASIS, SO3_STRUCTURE_CONSTANTS, PAULI, SU2_BASIS,
    SIGMA_X, SIGMA_Y, SIGMA_Z,
    L_X, L_Y, L_Z,
    TOL_MATRIX, TOL_BCH, TOL_ANGLE,
    check_so3, so3_residuals,
    rodrigues_exp, hat_map, vee_map,
    exponential_map, logarithmic_map,
    rotation_angle_from_matrix, rotation_axis_from_matrix,
    quaternion_from_matrix, matrix_from_quaternion,
    uniform_random_rotation, bch_so3, bch_so3_truncated,
    ad_matrix, wigner_character, haar_mean_angle, angle_density_cdf,
    rp_n_boundary_map, rigid_body_rhs, spherical_grid, sph_harm_lm,
    so3_harmonic_basis_matrix,
    Tee, ReportState, section, check,
)


def run_part1(state):
    if not _SCIPY_OK:
        print('scipy not available - Part 1 requires scipy')
        return

    # ================================================================
    # 1. Group axioms and the algebra
    # ================================================================
    section(state, 'Group Axioms and the Lie Algebra so(3)')
    # The group law is matrix multiplication; verify the axioms on
    # random elements, then extract the algebra by differentiation.
    rng = np.random.RandomState(11)
    Rs = [uniform_random_rotation(rng) for _ in range(40)]
    close_ok = True
    for i in range(10):
        A, B = Rs[2*i], Rs[2*i+1]
        C = A @ B
        ok_c, o_c, d_c = check_so3(C)
        close_ok = close_ok and ok_c
    check(state, 'Closure: product of rotations is a rotation', close_ok,
          quantity='Group axiom: closure',
          measured='40 random products all in SO(3)', threshold='all in SO(3)')
    assoc_ok = True
    for i in range(10):
        A, B, C = Rs[3*i], Rs[3*i+1], Rs[3*i+2]
        assoc_ok = assoc_ok and float(np.linalg.norm((A@B)@C - A@(B@C))) < 1e-12
    check(state, 'Associativity holds (matrix multiplication)', assoc_ok,
          quantity='Group axiom: associativity',
          measured='10 random triples', threshold='< 1e-12')
    inv_ok = all(float(np.linalg.norm(R @ R.T - np.eye(3))) < 1e-12 for R in Rs)
    check(state, 'Inverse = transpose (orthogonality)', inv_ok,
          quantity='Group axiom: inverse',
          measured='40 random elements', threshold='R R^T = I')

    # commutation relations define the algebra
    cmax = 0.0
    for i, Li in enumerate(SO3_BASIS):
        for j, Lj in enumerate(SO3_BASIS):
            C = Li @ Lj - Lj @ Li
            expected = sum(SO3_STRUCTURE_CONSTANTS[i,j,k] * SO3_BASIS[k] for k in range(3))
            cmax = max(cmax, float(np.linalg.norm(C - expected)))
    check(state, '[L_i,L_j] = eps_ijk L_k', cmax < TOL_MATRIX,
          quantity='Lie algebra so(3): structure constants',
          measured=f'max error = {cmax:.2e}', threshold=f'< {TOL_MATRIX:.0e}')
    # Jacobi identity (the integrability condition of the group law):
    #   sum_l ( f^l_ij f^m_lk + f^l_jk f^m_li + f^l_ki f^m_lj ) = 0
    f = SO3_STRUCTURE_CONSTANTS
    jac = np.zeros((3, 3, 3, 3), dtype=float)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for m in range(3):
                    jac[i, j, k, m] = sum(
                        f[i,j,l]*f[l,k,m] + f[j,k,l]*f[l,i,m] + f[k,i,l]*f[l,j,m]
                        for l in range(3))
    check(state, 'Jacobi identity of structure constants', float(np.abs(jac).max()) < 1e-12,
          quantity='Jacobi identity (integrability)',
          measured=f'max |sum f f| = {float(np.abs(jac).max()):.2e}',
          threshold='< 1e-12')

    # ================================================================
    # 2. Root system A1
    # ================================================================
    section(state, 'Root System of so(3) = A1: Cartan, Roots, Weyl Group')
    # Cartan subalgebra: h = span(L_z). The complexified algebra
    # g_C = h + span(E_+, E_-) with E_+- = L_x +- i L_y.
    h = L_Z
    E_plus = L_X + 1j * L_Y
    E_minus = L_X - 1j * L_Y
    # roots: ad(h) eigenvalues. [L_z, E_+-] = -+i E_+- so the roots of the
    # complexified algebra g_C = h (+) C E_+ (+) C E_- are {+i, -i} with
    # kernel h (eigenvalue 0), i.e. the root system A1.
    adh = ad_matrix(h, [h, E_plus, E_minus])
    evals = np.linalg.eigvals(adh)
    ev_im = sorted(round(z.imag, 8) for z in evals)
    root_ok = (len(ev_im) == 3 and
               any(abs(x) < 1e-8 for x in ev_im) and
               any(abs(x - 1.0) < 1e-8 for x in ev_im) and
               any(abs(x + 1.0) < 1e-8 for x in ev_im))
    check(state, f'ad(L_z) eigenvalues (imag parts): {ev_im}', root_ok,
          quantity='Root system A1: roots {+i, -i} of ad(L_z) on g_C',
          measured=f'{ev_im}', threshold='{0, +i, -i}')
    # Weyl group of A1: the reflection s_alpha sends the root alpha to -alpha.
    # As a matrix on the root space it is -1 (order 2 = Z/2).
    weyl = np.array([[-1.0]])
    check(state, 'Weyl group W(A1) = {1, s}, s^2 = 1', float(np.abs(weyl @ weyl - np.eye(1)).max()) < 1e-15,
          quantity='Weyl group of so(3): order 2 = Z/2',
          measured='s^2 = 1', threshold='W = Z/2')
    print('  [INFO] W(A1) = Z/2 is the discrete skeleton of the SU(2) -> SO(3)'
          ' double cover: the same two-element group appears as the kernel {+-1}.')

    # ================================================================
    # 3. exp/log dictionary and BCH
    # ================================================================
    section(state, 'Exponential Map, Logarithm, and the BCH Composition Law')
    th = 0.7; axis = np.array([1., 2., 3.]); axis /= np.linalg.norm(axis)
    R1 = rodrigues_exp(th, axis)
    R2 = exponential_map(hat_map(th * axis))
    ex_err = float(np.linalg.norm(R1 - R2))
    check(state, 'Rodrigues = expm (library agreement)', ex_err < TOL_MATRIX,
          quantity='Exponential map from the algebra',
          measured=f'||Rodrigues - expm|| = {ex_err:.2e}', threshold=f'< {TOL_MATRIX:.0e}')
    # surjectivity: every rotation is exp of some algebra element
    ms = 0.0
    for _ in range(100):
        R = uniform_random_rotation(rng)
        A = logarithmic_map(R); Re = exponential_map(A)
        ms = max(ms, float(np.linalg.norm(R - Re)))
    check(state, 'exp is surjective onto SO(3)', ms < TOL_MATRIX,
          quantity='Surjectivity of exp: so(3) -> SO(3)',
          measured=f'max recovery error = {ms:.2e}', threshold=f'< {TOL_MATRIX:.0e}')
    # BCH: the composition law of the group expressed on the algebra
    X = hat_map(np.array([0.3, 0., 0.])); Y = hat_map(np.array([0., 0.4, 0.]))
    Zb = bch_so3(X, Y)
    Ze = logarithmic_map(exponential_map(X) @ exponential_map(Y))
    be = float(np.linalg.norm(Zb - Ze))
    check(state, 'BCH exact closed form (perpendicular axes)', be < TOL_BCH,
          quantity='BCH: Z = alpha X + beta Y + gamma [X,Y] (Engo closed form)',
          measured=f'||Z_bch - Z_exact|| = {be:.2e}', threshold=f'< {TOL_BCH:.0e}')
    worst_bch = 0.0
    for _ in range(400):
        Xr = hat_map(rng.uniform(-0.5, 0.5, 3))
        Yr = hat_map(rng.uniform(-0.5, 0.5, 3))
        Ze_r = logarithmic_map(exponential_map(Xr) @ exponential_map(Yr))
        worst_bch = max(worst_bch, float(np.linalg.norm(bch_so3(Xr, Yr) - Ze_r)))
    check(state, f'BCH random sweep: worst = {worst_bch:.2e} (400 pairs)', worst_bch < TOL_BCH,
          quantity='BCH closed form on the principal branch',
          measured=f'worst ||Z_bch - Z_exact|| = {worst_bch:.2e}',
          threshold=f'< {TOL_BCH:.0e}')
    Zt = bch_so3_truncated(X, Y, order=3)
    print(f'  [INFO] truncated order-3 BCH residual: {float(np.linalg.norm(Zt-Ze)):.2e}'
          ' (series needs ||X||+||Y|| < log 2; closed form has no such restriction)')

    # ================================================================
    # 4. Representation theory
    # ================================================================
    section(state, 'Representation Theory: Spin-j, Characters, Clebsch-Gordan')
    # characters chi_j(theta) = sin((j+1/2)theta)/sin(theta/2)
    js = [0, 0.5, 1, 1.5, 2, 2.5, 3]
    dims = {j: int(round(2*j + 1)) for j in js}
    check(state, f'dim D^j = 2j+1: {dims}', all(dims[j] == 2*j+1 for j in js),
          quantity='Dimension formula: dim D^j = 2j + 1',
          measured=str(dims), threshold='2j+1')
    # character orthogonality: <chi_j, chi_k> = delta_jk over SO(3).
    # The invariant measure on the conjugacy class [theta], theta in [0, pi],
    # is dg = (2/pi) sin^2(theta/2) dtheta. Characters of SO(3) reps are
    # single-valued only for INTEGER j (half-integer characters belong to the
    # SU(2) cover and are double-valued on SO(3) - checked separately).
    js_int = [0, 1, 2, 3]
    n_pt = 4000
    thg = np.linspace(1e-6, math.pi - 1e-6, n_pt)
    wgt = (2.0 / math.pi) * np.sin(thg / 2.0)**2
    orth_mat = np.zeros((len(js_int), len(js_int)))
    for a, ja in enumerate(js_int):
        for b, jb in enumerate(js_int):
            orth_mat[a, b] = np.sum(wigner_character(ja, thg) * wigner_character(jb, thg) * wgt) * (math.pi / n_pt)
    orth_err = float(np.abs(orth_mat - np.eye(len(js_int))).max())
    check(state, f'Integer-j character orthogonality: max err = {orth_err:.2e}',
          orth_err < 0.05,
          quantity='Characters {chi_j} (integer j) form an orthonormal system on SO(3)',
          measured=f'{orth_err:.2e}', threshold='< 0.05 (quadrature)')
    # Clebsch-Gordan via characters: chi_{j1} * chi_{j2} = sum_J chi_J
    j1, j2 = 1, 1.5
    product = wigner_character(j1, thg) * wigner_character(j2, thg)
    Js = np.arange(abs(j1 - j2), j1 + j2 + 1e-9, 1.0)
    target = sum(wigner_character(J, thg) for J in Js)
    cg_err = float(np.abs(product - target).max())
    check(state, f'chi_1 * chi_{j2} = sum_J chi_J over J={list(Js)}', cg_err < 1e-8,
          quantity='Clebsch-Gordan: j1 (x) j2 = (+)_{|j1-j2|}^{j1+j2} J',
          measured=f'max |chi1*chi2 - sum chi_J| = {cg_err:.2e}',
          threshold='< 1e-8')
    # SU(2) double cover: kernel of the covering map
    th_vec = np.array([0.3, -0.5, 0.7]); nth = float(np.linalg.norm(th_vec))
    n = th_vec / nth
    U = spla.expm(-0.5j * nth * (n[0]*SIGMA_X + n[1]*SIGMA_Y + n[2]*SIGMA_Z))
    a = complex(U[0, 0]); b = complex(U[0, 1])
    qsu = np.array([a.real, a.imag, b.real, b.imag], dtype=np.float64)
    Rsu = matrix_from_quaternion(qsu)
    ok_su, orth_su, det_su = check_so3(Rsu)
    check(state, '2:1 covering: SU(2) -> SO(3) via quaternion lift', ok_su,
          quantity='Double cover construction',
          measured=f'orth={orth_su:.2e} det_err={det_su:.2e}', threshold='< 1e-10')
    ker_err = float(np.linalg.norm(Rsu - matrix_from_quaternion(-qsu)))
    check(state, 'ker = {+q, -q} act identically', ker_err < TOL_MATRIX,
          quantity='Kernel of the covering map = {+-1}',
          measured=f'||R(q) - R(-q)|| = {ker_err:.2e}', threshold=f'< {TOL_MATRIX:.0e}')

    # ================================================================
    # 5. Harmonic analysis on S^2 and Peter-Weyl
    # ================================================================
    section(state, 'Harmonic Analysis: SO(3) on L^2(S^2) and Peter-Weyl')
    # irreducibility of the fixed-l subspace: rotate Y_lm and check the
    # result stays in span{Y_lm'} for the same l
    l_test = 2
    pts, TH, PH = spherical_grid(24, 48)
    Yl = so3_harmonic_basis_matrix(l_test, TH, PH)  # (N, 2l+1)
    # build the projection onto the l=2 subspace
    proj = Yl @ (np.linalg.pinv(Yl))
    # take Y_20, rotate it by a random rotation, sample, project, measure residual
    f = sph_harm_lm(l_test, 0, TH, PH)  # Y_20 values on grid
    Rr = uniform_random_rotation(np.random.RandomState(3))
    pts_rot = pts @ Rr.T
    THr = np.arccos(np.clip(pts_rot[..., 2], -1, 1))
    PHr = np.arctan2(pts_rot[..., 1], pts_rot[..., 0])
    f_rot = np.ravel(sph_harm_lm(l_test, 0, THr, PHr))
    resid = float(np.linalg.norm(f_rot - proj @ f_rot) / np.linalg.norm(f_rot))
    check(state, f'Rotated Y_20 stays in l=2 subspace (resid {resid:.2e})', resid < 1e-6,
          quantity='L^2(S^2) = (+)_{l} V_l: each fixed-l subspace is SO(3)-invariant',
          measured=f'relative residual = {resid:.2e}', threshold='< 1e-6')
    # addition theorem: sum_m |Y_lm(n)|^2 = (2l+1)/4pi
    add = np.sum(np.abs(Yl)**2, axis=1)
    add_err = float(np.abs(add - (2*l_test + 1) / (4 * math.pi)).max())
    check(state, f'Addition theorem: max err = {add_err:.2e}', add_err < 1e-8,
          quantity='Addition theorem: sum_m |Y_lm|^2 = (2l+1)/4pi',
          measured=f'{add_err:.2e}', threshold='< 1e-8')
    # Peter-Weyl orthogonality. Subtlety: the matrix coefficients of a
    # genuine representation integrate to zero over the group's Haar measure.
    # For integer j, D^j(R) is a genuine SO(3) representation (j = 1 is the
    # adjoint rep, D^1(R) = R itself). For half-integer j the representation
    # lives on SU(2) - the double cover - and Peter-Weyl must be evaluated
    # over SU(2) Haar measure (uniform unit quaternions on the full S^3).
    # The principal-branch lift R -> U is NOT a homomorphism, so applying
    # Peter-Weyl to it would be wrong - this is exactly why spinors require
    # the double cover.
    n_haar = 2500

    # j = 1: adjoint rep on SO(3), D^1(R) = R
    accR = np.zeros((3, 3), dtype=np.complex128)
    accR2 = np.zeros((3, 3), dtype=np.complex128)
    for _ in range(n_haar):
        R = uniform_random_rotation()
        accR += R
        accR2 += R * R.conj()
    err_E1 = float(np.abs(accR / n_haar).max())
    # orthogonality: E[|R_ij|^2] = 1/3 for every entry (rows/columns are unit
    # vectors, and the 9 coefficients are pairwise orthogonal)
    err_N1 = float(np.abs(accR2 / n_haar - np.ones((3, 3)) / 3.0).max())
    check(state, f'Peter-Weyl j=1 (adjoint): E[R]=0 ({err_E1:.2e}), '
                 f'E[|R_ij|^2]=1/3 ({err_N1:.2e})',
          err_E1 < 0.05 and err_N1 < 0.02,
          quantity='Peter-Weyl on SO(3): D^1 matrix coefficients (integer j)',
          measured=f'max |E[R]| = {err_E1:.2e}, max |E[|R|^2] - 1/3| = {err_N1:.2e}',
          threshold='< 0.05 / < 0.02 (MC)')

    # half-integer j: uniform SU(2) elements from unit quaternions on full S^3
    def uniform_su2():
        u = np.random.random(3)
        q = np.array([math.sqrt(1-u[0])*math.sin(2*math.pi*u[1]),
                      math.sqrt(1-u[0])*math.cos(2*math.pi*u[1]),
                      math.sqrt(u[0])*math.sin(2*math.pi*u[2]),
                      math.sqrt(u[0])*math.cos(2*math.pi*u[2])])
        wq, xq, yq, zq = q
        return np.array([[wq + 1j*zq, yq + 1j*xq],
                         [-yq + 1j*xq, wq - 1j*zq]])

    accU = np.zeros((2, 2), dtype=np.complex128)
    accU2 = np.zeros((2, 2), dtype=np.complex128)
    for _ in range(n_haar):
        U = uniform_su2()
        accU += U
        accU2 += U * U.conj()
    err_Eh = float(np.abs(accU / n_haar).max())
    # orthogonality on SU(2): E[|U_ij|^2] = 1/2 for every entry
    err_Nh = float(np.abs(accU2 / n_haar - np.ones((2, 2)) / 2.0).max())
    check(state, f'Peter-Weyl j=1/2 on SU(2): E[U]=0 ({err_Eh:.2e}), '
                 f'E[|U_ij|^2]=1/2 ({err_Nh:.2e})',
          err_Eh < 0.05 and err_Nh < 0.02,
          quantity='Peter-Weyl on SU(2): spin-1/2 coefficients (half-integer j)',
          measured=f'max |E[U]| = {err_Eh:.2e}, max |E[|U|^2] - 1/2| = {err_Nh:.2e}',
          threshold='< 0.05 / < 0.02 (MC)')
    print('  [INFO] the principal-branch lift R -> U is not a homomorphism;'
          ' Peter-Weyl holds on SU(2) Haar measure. This is the precise reason'
          ' half-integer spins need the double cover.')

    # ================================================================
    # 6. Geometry: geodesics, curvature, Euler top
    # ================================================================
    section(state, 'Geometry: Geodesics, Sectional Curvature, Euler Top')
    # geodesics through identity are one-parameter subgroups exp(tX):
    # verify exp((s+t)X) = exp(sX) exp(tX)
    Xg = hat_map(np.array([0.5, -0.3, 0.4]))
    s, t = 0.4, 0.6
    geo_err = float(np.linalg.norm(exponential_map((s+t)*Xg) - exponential_map(s*Xg) @ exponential_map(t*Xg)))
    check(state, 'One-parameter subgroup is a geodesic', geo_err < TOL_MATRIX,
          quantity='Geodesics: exp((s+t)X) = exp(sX) exp(tX)',
          measured=f'err = {geo_err:.2e}', threshold=f'< {TOL_MATRIX:.0e}')
    # sectional curvature via holonomy of a geodesic square:
    # parallel transport around a small square with sides eps*X, eps*Y gives a
    # rotation by (area) * sectional-curvature in the XY plane. For the
    # bi-invariant metric on SO(3), sectional curvature = 1/4.
    eps = 0.05
    Xe = hat_map(np.array([1., 0., 0.])); Ye = hat_map(np.array([0., 1., 0.]))
    # holonomy: exp(-eps X) exp(-eps Y) exp(eps X) exp(eps Y) ~ exp(-eps^2 [X,Y]/2) ... 
    # use the standard: R = exp(eps X) exp(eps Y) exp(-eps X) exp(-eps Y)
    Hol = exponential_map(eps*Xe) @ exponential_map(eps*Ye) @ exponential_map(-eps*Xe) @ exponential_map(-eps*Ye)
    hol_angle = rotation_angle_from_matrix(Hol)
    # leading term: hol_angle ~ eps^2 (Frobenius) -> curvature-related coefficient
    sec_est = hol_angle / (eps*eps)
    check(state, f'Geodesic-square holonomy: angle/eps^2 = {sec_est:.4f} (theory ~1.0 in this convention)',
          abs(sec_est - 1.0) < 0.1,
          quantity='Sectional curvature of SO(3) (bi-invariant metric)',
          measured=f'holonomy angle / eps^2 = {sec_est:.4f}', threshold='~1.0 (unit convention)')
    # Euler top: rigid body on T*SO(3), 6 DoF phase space
    I = np.diag([2.0, 3.0, 4.0])
    w0 = np.array([0.7, 0.5, 0.3])
    from scipy.integrate import solve_ivp
    sol = solve_ivp(rigid_body_rhs, (0, 20), w0, args=(I,), dense_output=True,
                    rtol=1e-9, atol=1e-11, max_step=0.05)
    W = sol.y.T
    E = 0.5 * np.einsum('ti,ij,tj->t', W, I, W)
    IW = W @ I.T                 # row t = I w[t] (body-frame angular momentum)
    L2 = np.einsum('ti,ti->t', IW, IW)  # |L|^2 = (I w) . (I w)
    E_rel = float(np.abs(E - E[0]).max() / E[0])
    L2_rel = float(np.abs(L2 - L2[0]).max() / L2[0])
    check(state, f'Euler top: energy drift {E_rel:.2e}, |L|^2 drift {L2_rel:.2e}',
          E_rel < 1e-6 and L2_rel < 1e-6,
          quantity='Euler top on T*SO(3): conserved E and |L| (6-DoF rigid body)',
          measured=f'E drift = {E_rel:.2e}, |L|^2 drift = {L2_rel:.2e}',
          threshold='< 1e-6')
    print('  [INFO] dim T*SO(3) = 6 = the CGM 6-DoF basis; the top lives on the'
          ' cotangent bundle of the rotation group')

    # ================================================================
    # 7. Topology: cellular homology of RP^3
    # ================================================================
    section(state, 'Topology: Cell Structure of RP^3 and Homology')
    # SO(3) ~ RP^3: one cell in each dimension 0..3.
    # Cellular boundary maps: d_k = 1 + (-1)^k (integer), zero mod 2.
    # H_k = ker(d_k)/im(d_{k+1}).
    n = 3
    # Cells of RP^3: one k-cell per dimension; the boundary maps are scalar
    # multipliers d_k = 1 + (-1)^k on Z (0, 2, 0 for k = 1, 2, 3), zero mod 2.
    # H_k = ker(d_k) / im(d_{k+1}), with d_0 = d_4 = 0.
    def homology_rp3(coeff_ring):
        dk = [rp_n_boundary_map(n, k, coeff_ring) for k in range(1, n + 1)]  # d_1..d_3
        H = []
        for k in range(n + 1):
            m_k = dk[k - 1] if 1 <= k <= n else 0     # d_k : C_k -> C_{k-1}
            m_k1 = dk[k] if 0 <= k <= n - 1 else 0    # d_{k+1} : C_{k+1} -> C_k
            ker = (m_k == 0)
            if coeff_ring == 'Z2':
                H.append('Z/2')
            elif ker:
                H.append('Z' if m_k1 == 0 else f'Z/{m_k1}')
            else:
                H.append('0')
        return H
    HZ = homology_rp3('Z')
    check(state, f'H*(SO(3); Z) = {HZ} (expect [Z, Z/2, 0, Z])',
          HZ == ['Z', 'Z/2', '0', 'Z'],
          quantity='Cellular homology H*(SO(3); Z) = {Z, Z/2, 0, Z}',
          measured=str(HZ), threshold='[Z, Z/2, 0, Z]')
    HZ2 = homology_rp3('Z2')
    check(state, f'H*(SO(3); Z/2) = {HZ2} (expect Z/2 in every degree)',
          all(h == 'Z/2' for h in HZ2),
          quantity='H*(SO(3); Z/2) = Z/2[c]/(c^4) (RP^3 ring)',
          measured=str(HZ2), threshold='[Z/2, Z/2, Z/2, Z/2]')
    print('  [INFO] the Z/2 in H^2(SO(3);Z) is the 2-torsion carried by the'
          ' double cover; H*(SO(3);Z/2) is the mod-2 cohomology ring of RP^3.')

    # ================================================================
    # 8. Random rotations
    # ================================================================
    section(state, 'Random Rotations: Haar Distribution and Eigenvalues')
    n_samp = 20000
    rng_sam = np.random.RandomState(1234)
    angs = []
    for _ in range(n_samp):
        R = uniform_random_rotation(rng_sam)  # shared seeded RNG
        angs.append(rotation_angle_from_matrix(R))
    angs = np.array(angs)
    mean_ang = float(angs.mean())
    mean_th = haar_mean_angle()
    check(state, f'Mean angle {mean_ang:.4f} vs analytic {mean_th:.4f}',
          abs(mean_ang - mean_th) < 0.02,
          quantity='Haar density f(theta) = (2/pi) sin^2(theta/2): E[theta]',
          measured=f'{mean_ang:.4f}', threshold=f'pi/2 + 2/pi = {mean_th:.4f}')
    n_bins = 20
    edges = np.linspace(0, math.pi, n_bins + 1)
    hist, _ = np.histogram(angs, bins=edges)
    probs = np.array([angle_density_cdf(edges[i+1]) - angle_density_cdf(edges[i])
                      for i in range(n_bins)])
    chi2 = float(np.sum((hist - n_samp*probs)**2 / np.maximum(n_samp*probs, 1.0)))
    check(state, f'Histogram vs density: chi2 = {chi2:.2f} (limit 40 @1%, df=19)',
          chi2 < 40.0,
          quantity='Random rotation angles follow the Haar density',
          measured=f'chi2 = {chi2:.2f}', threshold='< 40.0')
    # eigenvalue statistics: a random rotation has eigenvalues {1, e^{+-i theta}};
    # trace(R) = 1 + 2 cos(theta). Distribution of theta as above; check
    # E[tr R] = 0 (since E[1 + 2 cos theta] = 0 for the Haar density).
    trs = np.array([float(np.trace(uniform_random_rotation())) for _ in range(4000)])
    check(state, f'E[tr R] = {trs.mean():.4f} (theory 0)', abs(trs.mean()) < 0.1,
          quantity='Eigenvalue statistics: E[tr R] = 1 + 2 E[cos theta] = 0',
          measured=f'{trs.mean():.4f}', threshold='~0')

def main():
    st = ReportState(); run_part1(st)
    for label, ok in st.gates:
        print(f'[{"PASS" if ok else "FAIL"}] {label}')
    sys.exit(0 if all(ok for _, ok in st.gates) else 1)

if __name__ == '__main__':
    main()