#!/usr/bin/env python3
"""hqvm_SO_analysis_common.py — Shared infrastructure.
All linear algebra delegated to scipy/numpy/mpmath.
Integrates with hQVM kernel and CGM theory.
"""
from __future__ import annotations
import math, sys
from pathlib import Path
from typing import Any
import mpmath as mp
import numpy as np

_EXP = Path(__file__).resolve().parent
_REPO = _EXP.parent
RESULTS_PATH = _EXP / "hqvm_SO_analysis_results.txt"
WORKNOTES_PATH = _EXP / "hqvm_SO_analysis_temp_worknotes.txt"
if str(_EXP) not in sys.path: sys.path.insert(0, str(_EXP))
if str(_REPO) not in sys.path: sys.path.insert(0, str(_REPO))

_SCIPY_OK, _KERNEL_OK, _GYRO_OK = True, True, True
try:
    import scipy.linalg as spla
    import scipy.special as spspec
    import scipy.sparse as sp
    import scipy.sparse.linalg as spspl
    import scipy.spatial.transform as sstr
except ImportError: _SCIPY_OK = False
if not _SCIPY_OK:
    spla = None  # type: ignore

try:
    from gyroscopic.hQVM.api import (
        OmegaState12, OmegaSignature12, compose_omega_signatures,
        state24_to_omega12, omega12_to_state24,
        chirality_word6, q_word6, q_word6_for_items,
        step_omega12_by_byte, omega_word_signature,
        shadow_partner_byte,
        shell_transition_probability, shell_transition_matrix_for_q_weight,
        shell_markov_step, shell_krawtchouk_transform_exact,
        shell_population, k4_orbit, k4_stabilizer, fixed_locus,
        walsh_hadamard64,
    )
    from gyroscopic.hQVM.constants import (
        GENE_MAC_REST, MASK_STATE24, LAYER_MASK_12,
        step_state_by_byte, unpack_state, pack_state,
        GENE_MAC_A12, M_A, BU_HOLONOMY_ANGLE, APERTURE_GAP,
    )
    from gyroscopic.hQVM.sdk import (
        state_charts, moment_from_ledger,
        future_cone_measure, future_entropy_bits,
        directional_derivative, byte_derivative_table,
        witness_from_rest, SpectralOps, StateOps, MomentOps,
    )
except ImportError: _KERNEL_OK = False

# Fallback stubs so importing modules always succeed even without the kernel
if not _KERNEL_OK:
    OmegaState12 = None  # type: ignore
    OmegaSignature12 = None  # type: ignore
    compose_omega_signatures = None  # type: ignore
    step_state_by_byte = None  # type: ignore
    omega12_to_state24 = None  # type: ignore
    chirality_word6 = None  # type: ignore
    q_word6 = None  # type: ignore
    shadow_partner_byte = None  # type: ignore
    shell_population = None  # type: ignore
    shell_transition_matrix_for_q_weight = None  # type: ignore
    state_charts = None  # type: ignore
    future_cone_measure = None  # type: ignore
    optical_coordinates = None  # type: ignore

try:
    from functions.gyrovector_ops import GyroVectorSpace
except ImportError: _GYRO_OK = False

mp.mp.dps = 80
TOL_SO3 = 1e-12; TOL_MATRIX = 1e-10; TOL_ANGLE = 1e-12
TOL_BCH = 1e-10; TOL_SPECTRAL = 1e-10; TOL_KERNEL = 1e-12
TOL_MP = mp.mpf("1e-60")

# Canonical so(3) Lie algebra generators [L_i, L_j] = eps_ijk L_k
L_X = np.array([[0,0,0],[0,0,-1],[0,1,0]], dtype=np.float64)
L_Y = np.array([[0,0,1],[0,0,0],[-1,0,0]], dtype=np.float64)
L_Z = np.array([[0,-1,0],[1,0,0],[0,0,0]], dtype=np.float64)
SO3_BASIS = (L_X, L_Y, L_Z)

SO3_STRUCTURE_CONSTANTS = np.zeros((3,3,3), dtype=np.float64)
SO3_STRUCTURE_CONSTANTS[0,1,2] = 1.0; SO3_STRUCTURE_CONSTANTS[1,2,0] = 1.0
SO3_STRUCTURE_CONSTANTS[2,0,1] = 1.0; SO3_STRUCTURE_CONSTANTS[1,0,2] = -1.0
SO3_STRUCTURE_CONSTANTS[2,1,0] = -1.0; SO3_STRUCTURE_CONSTANTS[0,2,1] = -1.0

# Pauli matrices (physicist convention, hermitian)
SIGMA_X = np.array([[0,1],[1,0]], dtype=np.complex128)
SIGMA_Y = np.array([[0,-1j],[1j,0]], dtype=np.complex128)
SIGMA_Z = np.array([[1,0],[0,-1]], dtype=np.complex128)
PAULI = (SIGMA_X, SIGMA_Y, SIGMA_Z)
SU2_BASIS = tuple(-0.5j * p for p in PAULI)

# ---- Quaternion helpers ----
# Deterministic shared RNG for all Haar sampling (no per-call entropy
# seeding; reproducible across runs).
_GLOBAL_RNG = np.random.RandomState(20260819)


def quaternion_from_matrix(R):
    """SO(3) matrix -> unit quaternion (w, x, y, z), w >= 0."""
    R = np.asarray(R, dtype=float)
    tr = float(np.trace(R))
    if tr > 0:
        s = math.sqrt(tr+1.0)*2.0
        w=0.25*s; x=(R[2,1]-R[1,2])/s
        y=(R[0,2]-R[2,0])/s; z=(R[1,0]-R[0,1])/s
    else:
        idx=int(np.argmax(np.diag(R)))
        if idx==0:
            s=math.sqrt(1.0+R[0,0]-R[1,1]-R[2,2])*2.0
            w=(R[2,1]-R[1,2])/s; x=0.25*s
            y=(R[0,1]+R[1,0])/s; z=(R[0,2]+R[2,0])/s
        elif idx==1:
            s=math.sqrt(1.0+R[1,1]-R[0,0]-R[2,2])*2.0
            w=(R[0,2]-R[2,0])/s; x=(R[0,1]+R[1,0])/s
            y=0.25*s; z=(R[1,2]+R[2,1])/s
        else:
            s=math.sqrt(1.0+R[2,2]-R[0,0]-R[1,1])*2.0
            w=(R[1,0]-R[0,1])/s; x=(R[0,2]+R[2,0])/s
            y=(R[1,2]+R[2,1])/s; z=0.25*s
    q=np.array([w,x,y,z],dtype=float)
    n=float(np.linalg.norm(q))
    if n>1e-15: q/=n
    if q[0]<0: q=-q
    return q


def uniform_random_rotation(rng=None):
    """Uniformly random SO(3) matrix (Shoemake quaternion, Haar measure).

    If rng is None, uses the module-level deterministic RNG.
    """
    if rng is None:
        rng = _GLOBAL_RNG
    u1, u2, u3 = rng.random(3)
    q = np.array([math.sqrt(1-u1)*math.sin(2*math.pi*u2),
                  math.sqrt(1-u1)*math.cos(2*math.pi*u2),
                  math.sqrt(u1)*math.sin(2*math.pi*u3),
                  math.sqrt(u1)*math.cos(2*math.pi*u3)], dtype=np.float64)
    return matrix_from_quaternion(q)

def matrix_from_quaternion(q):
    w,x,y,z=(float(q[0]),float(q[1]),float(q[2]),float(q[3]))
    return np.array([[1-2*y*y-2*z*z,2*x*y-2*z*w,2*x*z+2*y*w],
                     [2*x*y+2*z*w,1-2*x*x-2*z*z,2*y*z-2*x*w],
                     [2*x*z-2*y*w,2*y*z+2*x*w,1-2*x*x-2*y*y]],dtype=np.float64)

def rotation_angle_from_matrix(R):
    return math.acos((float(np.clip(np.trace(R),-1.0,3.0))-1.0)/2.0)

def rotation_axis_from_matrix(R):
    a=rotation_angle_from_matrix(R)
    if a<1e-12: return np.array([1.0,0.0,0.0])
    rx=R[2,1]-R[1,2]; ry=R[0,2]-R[2,0]; rz=R[1,0]-R[0,1]
    n=math.sqrt(rx*rx+ry*ry+rz*rz)
    if n<1e-15: return np.array([1.0,0.0,0.0])
    return np.array([rx/n,ry/n,rz/n])

def hat_map(v):
    x,y,z=float(v[0]),float(v[1]),float(v[2])
    return np.array([[0,-z,y],[z,0,-x],[-y,x,0]],dtype=np.float64)

def vee_map(A):
    return np.array([A[2,1],A[0,2],A[1,0]],dtype=np.float64)

def rodrigues_exp(theta, axis):
    kx,ky,kz=axis[0],axis[1],axis[2]
    K=np.array([[0,-kz,ky],[kz,0,-kx],[-ky,kx,0]],dtype=np.float64)
    c=math.cos(theta); s=math.sin(theta)
    return np.eye(3)+s*K+(1-c)*(K@K)

def exponential_map(A):
    try: return spla.expm(A)
    except:
        theta=math.sqrt(max(0,-0.5*np.trace(A@A)))
        ax=vee_map(A)/(theta+1e-30)
        return rodrigues_exp(theta,ax)

def logarithmic_map(R):
    return spla.logm(R)

def so3_residuals(R):
    orth=float(np.linalg.norm(R.T@R-np.eye(3)))
    det=abs(float(np.linalg.det(R))-1.0)
    return orth,det

def check_so3(R, tol=1e-10):
    o,d=so3_residuals(R)
    return o<tol and d<tol, o, d

# ---- BCH formula ----
def bch_so3_truncated(X, Y, order=3):
    """Truncated BCH series: Z = X+Y + 1/2[X,Y] + 1/12([X,[X,Y]]-[Y,[X,Y]]) + ...
    Converges only when ||X||+||Y|| < log(2); kept for series comparison.
    """
    Z = X + Y
    comm = X@Y - Y@X
    if order >= 2: Z += 0.5*comm
    if order >= 3: Z += (1/12)*(X@comm - comm@X - Y@comm + comm@Y)
    return Z


def bch_so3_exact(X, Y):
    """Exact closed-form BCH for so(3) (Engo 2001):
    Z = alpha*X + beta*Y + gamma*[X,Y] with trigonometric coefficients.

    Verified against log(exp(X)exp(Y)) to machine precision whenever the
    composite rotation angle is below pi (principal branch); beyond that the
    formula tracks the continuous branch while logm returns the principal
    value, a known branch-of-logarithm effect.

    Reference: Engo, K. (2001), 'On the BCH-formula in so(3)',
    BIT Numerical Mathematics 41(3):629-632.
    """
    ux = vee_map(X); uy = vee_map(Y)
    th = float(np.linalg.norm(ux)); ph = float(np.linalg.norm(uy))
    if th < 1e-300 or ph < 1e-300:
        return X + Y
    cos_ang = float(np.dot(ux, uy) / (th * ph))
    cos_ang = float(np.clip(cos_ang, -1.0, 1.0))
    ang = math.acos(cos_ang)
    c = 0.5*math.sin(th)*math.sin(ph) - 2.0*math.sin(th/2)**2*math.sin(ph/2)**2*cos_ang
    a = c * (math.cos(ph/2) / math.sin(ph/2))
    b = c * (math.cos(th/2) / math.sin(th/2))
    d2 = a*a + b*b + 2*a*b*cos_ang + c*c*math.sin(ang)**2
    d = math.sqrt(max(0.0, d2))
    gamma = (math.asin(min(1.0, d)) / d) * c / (th * ph)
    alpha = ph * (math.cos(ph/2) / math.sin(ph/2)) * gamma
    beta = th * (math.cos(th/2) / math.sin(th/2)) * gamma
    return alpha*X + beta*Y + gamma*(X@Y - Y@X)


def bch_so3(X, Y, order=3):
    """BCH Z = log(exp(X)exp(Y)); exact closed form (Engo 2001) by default."""
    return bch_so3_exact(X, Y)


# ----------------------------------------------------------------------
# Kernel engine: the finite group generated by byte transitions
# ----------------------------------------------------------------------
def sig_int(sig):
    """Pack an OmegaSignature12 into an int: (parity << 12) | (tau_u << 6) | tau_v."""
    return (sig.parity << 12) | (sig.tau_u6 << 6) | sig.tau_v6


def sig_from_int(g):
    """Unpack an int into an OmegaSignature12."""
    return OmegaSignature12(
        parity=(g >> 12) & 1,
        tau_u6=(g >> 6) & 63,
        tau_v6=g & 63,
    )


def compose_sig_int(a, b):
    """Compose two packed signatures: result = a o b (apply b first).
    Uses the kernel's exact compose_omega_signatures (api.py)."""
    return sig_int(compose_omega_signatures(sig_from_int(a), sig_from_int(b)))


def byte_signature_ints():
    """All 256 single-byte packed signatures (128 distinct)."""
    out = set()
    for b in range(256):
        out.add(sig_int(omega_word_signature(bytes([b]))))
    return tuple(sorted(out))


def kernel_group():
    """The group generated by the 128 distinct byte signatures.

    Structural theorem (verified in the analysis): the generators are all
    parity-1 maps with tau_u in {0, 63}, tau_v arbitrary; pair products
    generate all 4096 translations, and odd products all 4096 swap-maps,
    so the generated group is the full affine double cover
    G = (Z/2)^12 x| Z/2 of order 8192 = 2^13.
    """
    gens = byte_signature_ints()
    # even words: pair products -> all translations (4096)
    T = set()
    for a in gens:
        for b in gens:
            T.add(compose_sig_int(a, b))
    # odd words: (parity-1 generator) o (any translation)
    O = set()
    for g in gens:
        for t in T:
            O.add(compose_sig_int(g, t))
    return tuple(sorted(T | O))


def apply_sig_int(g, state24):
    """Apply packed signature g to a 24-bit Omega state."""
    om = state24_to_omega12(state24)
    p, u, v = (g >> 12) & 1, (g >> 6) & 63, g & 63
    if p == 0:
        return omega12_to_state24(OmegaState12(u6=om.u6 ^ u, v6=om.v6 ^ v))
    return omega12_to_state24(OmegaState12(u6=om.v6 ^ u, v6=om.u6 ^ v))


def stabilizer_of(state24, group):
    """Elements of the group fixing a state (order-2 discrete cover kernel)."""
    return [g for g in group if apply_sig_int(g, state24) == state24]


def orbit_of(state24, group):
    """Orbit of a state under the group.

    Fast exact path: the group G = (Z/2)^12 x| Z/2 contains all 4096
    parity-0 translations (verified structurally in the analysis), and the
    translations act by (u,v) -> (u ^ t_u, v ^ t_v), so the orbit of any
    state is Omega itself, computed in 4096 exact applications.
    """
    om = state24_to_omega12(state24)
    out = set()
    for g in group:
        if (g >> 12) & 1 != 0:
            continue
        t_u, t_v = (g >> 6) & 63, g & 63
        out.add(omega12_to_state24(OmegaState12(u6=om.u6 ^ t_u, v6=om.v6 ^ t_v)))
    return out


# ----------------------------------------------------------------------
# Spectral analysis of the byte random walk on Omega
# ----------------------------------------------------------------------
def byte_transition_matrix():
    """Sparse 4096x4096 doubly-stochastic matrix P = uniform over 256 bytes,
    restricted to Omega. Returns (P, omega_list)."""
    omega_list = sorted(
        omega12_to_state24(OmegaState12(u6=u, v6=v)) for u in range(64) for v in range(64)
    )
    idx = {s: i for i, s in enumerate(omega_list)}
    n = len(omega_list)
    rows, cols, data = [], [], []
    w = 1.0 / 256.0
    for i, s in enumerate(omega_list):
        for b in range(256):
            nxt = step_state_by_byte(s, b)
            rows.append(i)
            cols.append(idx[nxt])
            data.append(w)
    P = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    return P, omega_list


# ----------------------------------------------------------------------
# SO(3) theory helpers
# ----------------------------------------------------------------------
def ad_matrix(X, basis):
    """Matrix of ad_X on the algebra with respect to `basis` (list of arrays).

    Column j holds the coordinates of [X, basis_j] in the basis, solved by
    least squares over the flattened 9-dim embedding space (robust for
    complexified bases like {h, E_+, E_-}).
    """
    n = len(basis)
    A = np.stack([B.ravel() for B in basis], axis=1).astype(np.complex128)  # 9 x n
    M = np.zeros((n, n), dtype=np.complex128)
    for j, Y in enumerate(basis):
        adXY = (X @ Y - Y @ X).ravel()
        coef, *_ = np.linalg.lstsq(A, adXY, rcond=None)
        M[:, j] = coef
    return M


def wigner_character(j, theta):
    """Character of the spin-j rep: chi_j(theta) = sin((j+1/2)theta)/sin(theta/2)."""
    th = np.asarray(theta, dtype=float)
    return np.sin((j + 0.5) * th) / np.sin(th / 2.0 + 1e-300)


def haar_mean_angle():
    """Analytic E[theta] = pi/2 + 2/pi under the Haar density (2/pi) sin^2(theta/2)."""
    return math.pi / 2 + 2.0 / math.pi


def angle_density_cdf(t):
    """CDF of the Haar rotation-angle distribution: int_0^t (2/pi) sin^2(x/2) dx."""
    return (t - math.sin(t)) / math.pi


def rp_n_boundary_map(n, k, coeff_ring='Z'):
    """Boundary map d_k of the standard CW structure of RP^n:
    d_k = 1 + (-1)^k  (mod 2 for Z/2 coefficients). Returns int (0, 1, or 2)."""
    d = 1 + ((-1) ** k)
    if coeff_ring == 'Z2':
        d = d % 2
    return d if 0 <= k <= n else 0


def rigid_body_rhs(t, y, I):
    """Body-frame Euler top: omega_dot = I^-1 ((I omega) x omega)."""
    w = np.asarray(y, dtype=float)
    Iw = I @ w
    return np.linalg.solve(I, np.cross(Iw, w))


def spherical_grid(n_theta=24, n_phi=48):
    """Equiangular grid on S^2 avoiding the poles (theta in (0, pi))."""
    eps_g = 1e-4
    thetas = np.linspace(eps_g, math.pi - eps_g, n_theta)
    phis = np.linspace(0, 2 * math.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(thetas, phis, indexing='ij')
    x = np.sin(TH) * np.cos(PH)
    y = np.sin(TH) * np.sin(PH)
    z = np.cos(TH)
    return np.stack([x, y, z], axis=-1), TH, PH


# scipy.special spherical harmonic compatibility:
#   scipy < 1.17: sph_harm(m, l, phi, theta)
#   scipy >= 1.17: sph_harm_y(n=l, m, theta, phi)
_SPH_OLD = getattr(spspec, 'sph_harm', None)   # (m, l, phi, theta)
_SPH_NEW = getattr(spspec, 'sph_harm_y', None)  # (l, m, theta, phi)


def sph_harm_lm(l, m, theta, phi):
    """Spherical harmonic Y_lm(theta, phi) via scipy.special (compat shim)."""
    th = np.asarray(theta)
    ph = np.asarray(phi)
    shape = th.shape
    if _SPH_NEW is not None:
        val = _SPH_NEW(int(l), int(m), th.ravel(), ph.ravel())
    else:
        val = _SPH_OLD(int(m), int(l), ph.ravel(), th.ravel())
    return val.reshape(shape) if val.shape != shape else val


def so3_harmonic_basis_matrix(l, theta, phi):
    """Matrix Y[l, m] = Y_lm(theta, phi) for m = -l..l (rows = grid points)."""
    th = np.asarray(theta).ravel()
    ph = np.asarray(phi).ravel()
    Y = np.zeros((th.size, 2 * l + 1), dtype=np.complex128)
    for mi, m in enumerate(range(-l, l + 1)):
        Y[:, mi] = sph_harm_lm(l, m, th, ph)
    return Y

# ---- Report helpers ----
class Tee:
    def __init__(self, *streams): self._streams=streams
    def write(self,d): [s.write(d) for s in self._streams]; return len(d)
    def flush(self): [s.flush() for s in self._streams]

class ReportState:
    def __init__(self): self.gates=[]; self.section_n=0

def section(state, title):
    state.section_n+=1
    print(f"\n{state.section_n}. {title}")
    print("="*5)

def check(state, label, ok, quantity=None, measured=None, threshold=None):
    st="PASS" if ok else "FAIL"
    if quantity:
        print(f"  [{st}] {quantity}")
        if measured: print(f"         measured: {measured}")
        if threshold: print(f"         threshold: {threshold}")
        state.gates.append((quantity, ok))
    else:
        print(f"  check  {label:56s} {st}")
        state.gates.append((label, ok))