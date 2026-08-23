#!/usr/bin/env python3
"""hqvm_group_analysis_common.py — Shared infrastructure for hQVM group analysis.

Role: kernel group helpers, SO(3) bridge maps (exp/hat for BU holonomy), reporting.
Inputs: gyroscopic.hQVM, scipy/numpy.
Outputs: constants and functions imported by hqvm_group_analysis_1/2/3.
Companion: hqvm_group_analysis_run.py.
"""
from __future__ import annotations
import math, sys
from pathlib import Path
from typing import Any
import mpmath as mp
import numpy as np

_EXP = Path(__file__).resolve().parent
_REPO = _EXP.parent
RESULTS_PATH = _EXP / "hqvm_group_analysis_results.txt"
WORKNOTES_PATH = _EXP / "hqvm_group_analysis_temp_worknotes.txt"
if str(_EXP) not in sys.path: sys.path.insert(0, str(_EXP))
if str(_REPO) not in sys.path: sys.path.insert(0, str(_REPO))

_SCIPY_OK, _KERNEL_OK, _GYRO_OK = True, True, True


def _kernel_missing(name: str):
    """Stub for kernel symbols when gyroscopic.hQVM is unavailable."""
    def _stub(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(f"hQVM kernel not available ({name})")
    return _stub


class _KernelMissingType:
    """Stub type for kernel dataclasses when gyroscopic.hQVM is unavailable."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("hQVM kernel not available")


spla: Any
spspec: Any
try:
    import scipy.linalg as _spla
    import scipy.special as _spspec
    import scipy.sparse as sp
    import scipy.sparse.linalg as spspl
    import scipy.spatial.transform as sstr
    spla = _spla
    spspec = _spspec
except ImportError: _SCIPY_OK = False
if not _SCIPY_OK:
    spla = _kernel_missing("scipy.linalg")
    spspec = _kernel_missing("scipy.special")

OmegaState12: Any
OmegaSignature12: Any
compose_omega_signatures: Any
state24_to_omega12: Any
omega12_to_state24: Any
step_state_by_byte: Any
chirality_word6: Any
q_word6: Any
shadow_partner_byte: Any
shell_population: Any
shell_transition_matrix_for_q_weight: Any
state_charts: Any
future_cone_measure: Any
optical_coordinates: Any
try:
    from gyroscopic.hQVM.api import (
        OmegaState12 as _OmegaState12,
        OmegaSignature12 as _OmegaSignature12,
        compose_omega_signatures as _compose_omega_signatures,
        state24_to_omega12 as _state24_to_omega12,
        omega12_to_state24 as _omega12_to_state24,
        chirality_word6 as _chirality_word6,
        q_word6 as _q_word6,
        q_word6_for_items,
        step_omega12_by_byte, omega_word_signature,
        shadow_partner_byte as _shadow_partner_byte,
        shell_transition_probability, shell_transition_matrix_for_q_weight as _shell_transition_matrix_for_q_weight,
        shell_markov_step, shell_krawtchouk_transform_exact,
        shell_population as _shell_population,
        k4_orbit, k4_stabilizer, fixed_locus,
        walsh_hadamard64,
    )
    from gyroscopic.hQVM.constants import (
        GENE_MAC_REST, MASK_STATE24, LAYER_MASK_12,
        step_state_by_byte as _step_state_by_byte,
        unpack_state, pack_state,
        GENE_MAC_A12, M_A, BU_HOLONOMY_ANGLE, APERTURE_GAP,
    )
    from gyroscopic.hQVM.sdk import (
        state_charts as _state_charts,
        moment_from_ledger,
        future_cone_measure as _future_cone_measure,
        future_entropy_bits,
        directional_derivative, byte_derivative_table,
        witness_from_rest, SpectralOps, StateOps, MomentOps,
    )
    OmegaState12 = _OmegaState12
    OmegaSignature12 = _OmegaSignature12
    compose_omega_signatures = _compose_omega_signatures
    state24_to_omega12 = _state24_to_omega12
    omega12_to_state24 = _omega12_to_state24
    step_state_by_byte = _step_state_by_byte
    chirality_word6 = _chirality_word6
    q_word6 = _q_word6
    shadow_partner_byte = _shadow_partner_byte
    shell_population = _shell_population
    shell_transition_matrix_for_q_weight = _shell_transition_matrix_for_q_weight
    state_charts = _state_charts
    future_cone_measure = _future_cone_measure
except ImportError: _KERNEL_OK = False

# Fallback stubs so importing modules always succeed even without the kernel
if not _KERNEL_OK:
    OmegaState12 = _KernelMissingType  # type: ignore[misc,assignment]
    OmegaSignature12 = _KernelMissingType  # type: ignore[misc,assignment]
    compose_omega_signatures = _kernel_missing("compose_omega_signatures")
    state24_to_omega12 = _kernel_missing("state24_to_omega12")
    step_state_by_byte = _kernel_missing("step_state_by_byte")
    omega12_to_state24 = _kernel_missing("omega12_to_state24")
    chirality_word6 = _kernel_missing("chirality_word6")
    q_word6 = _kernel_missing("q_word6")
    shadow_partner_byte = _kernel_missing("shadow_partner_byte")
    shell_population = _kernel_missing("shell_population")
    shell_transition_matrix_for_q_weight = _kernel_missing(
        "shell_transition_matrix_for_q_weight")
    state_charts = _kernel_missing("state_charts")
    future_cone_measure = _kernel_missing("future_cone_measure")
    optical_coordinates = _kernel_missing("optical_coordinates")

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
    if _SCIPY_OK:
        return spla.expm(A)
    theta=math.sqrt(max(0,-0.5*np.trace(A@A)))
    ax=vee_map(A)/(theta+1e-30)
    return rodrigues_exp(theta,ax)

def logarithmic_map(R):
    if _SCIPY_OK:
        return spla.logm(R)
    raise RuntimeError("scipy.linalg required for logarithmic_map")

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


def compose_sig_batch(a, b):
    """Vectorized composition of packed signatures (uint16 arrays).

    The affine map composition needs only bit-parallel integer ops
    (XOR/select) - no floating point, no normalization. This is the
    'matrix multiplication' of the finite rotation group G: it composes
    rotations in a few integer instructions instead of 27 float
    multiply-adds, and packs 4 signatures per 64-bit register.
    """
    a = np.asarray(a, dtype=np.uint16)
    b = np.asarray(b, dtype=np.uint16)
    pa = (a >> 12) & 1; ua = (a >> 6) & 63; va = a & 63
    pb = (b >> 12) & 1; ub = (b >> 6) & 63; vb = b & 63
    p = pa ^ pb
    u = np.where(pa == 0, ua ^ ub, ua ^ vb)
    v = np.where(pa == 0, va ^ vb, va ^ ub)
    return (p.astype(np.uint16) << 12) | (u.astype(np.uint16) << 6) | v.astype(np.uint16)


def omega_transition_table():
    """Omega-restricted byte transition table (4096 x 256, uint32).

    Stepping a state by a byte is one table lookup (fits in L3 cache,
    ~4.2 MB), instead of a per-call bitwise computation.
    """
    omega = [omega12_to_state24(OmegaState12(u6=u, v6=v))
             for u in range(64) for v in range(64)]
    idx = {s: i for i, s in enumerate(omega)}
    tbl = np.zeros((4096, 256), dtype=np.uint32)
    for i, s in enumerate(omega):
        for b in range(256):
            tbl[i, b] = idx[step_state_by_byte(s, b)]
    return tbl


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
if _SCIPY_OK:
    _SPH_OLD = getattr(spspec, 'sph_harm', None)   # (m, l, phi, theta)
    _SPH_NEW = getattr(spspec, 'sph_harm_y', None)  # (l, m, theta, phi)
else:
    _SPH_OLD = None
    _SPH_NEW = None


def sph_harm_lm(l, m, theta, phi):
    """Spherical harmonic Y_lm(theta, phi) via scipy.special (compat shim)."""
    th = np.asarray(theta)
    ph = np.asarray(phi)
    shape = th.shape
    if _SPH_NEW is not None:
        val = _SPH_NEW(int(l), int(m), th.ravel(), ph.ravel())
    elif _SPH_OLD is not None:
        val = _SPH_OLD(int(m), int(l), ph.ravel(), th.ravel())
    else:
        raise RuntimeError("scipy.special required for sph_harm_lm")
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
# Default quiet: PASS/FAIL + measured/threshold; sample dumps via vprint only.
# Runner --verbose sets VERBOSE True for full tables.
VERBOSE = False


def set_verbose(flag: bool) -> None:
    global VERBOSE
    VERBOSE = bool(flag)


def vprint(*args, **kwargs) -> None:
    """Print only when VERBOSE (sample tables, flat histograms, helper dumps)."""
    if VERBOSE:
        print(*args, **kwargs)


def info(msg: str) -> None:
    """Always-on short note (LIMIT / trichotomy / kill-switch). Prefer one line."""
    print(f'  [INFO] {msg}')


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
        if measured: print(f'         measured: {measured}')
        if threshold: print(f'         threshold: {threshold}')
        state.gates.append((quantity, ok))
    else:
        print(f"  check  {label:56s} {st}")
        state.gates.append((label, ok))


# ----------------------------------------------------------------------
# Character / Clifford helpers (shared by _2 and _4+)
# ----------------------------------------------------------------------
def swap_halves(x, d=6):
    """Swap the two d-bit halves of a 2d-bit word."""
    m = (1 << d) - 1
    return ((x & m) << d) | ((x >> d) & m)


def _swap12(x):
    return swap_halves(x, 6)


def inv_sig_int(g, d=6):
    """Inverse of a packed signature in G_d."""
    p = (g >> (2 * d)) & 1
    u = (g >> d) & ((1 << d) - 1)
    v = g & ((1 << d) - 1)
    return (p << (2 * d)) | ((v if p else u) << d) | (u if p else v)


def linear_char(s, a, g, d=6):
    """Linear character chi_{s,a} of G_d on even coset stabilizer a in GF(2)^d."""
    p = (g >> (2 * d)) & 1
    u = (g >> d) & ((1 << d) - 1)
    v = g & ((1 << d) - 1)
    return (-1) ** ((((s & 1) * (p & 1)) ^ (bin(a & (u ^ v)).count('1') & 1)) & 1)


def twod_char(k, g, d=6):
    """Character of Ind_A^G(chi_k) for swap-orbit of k in GF(2)^{2d}."""
    p = (g >> (2 * d)) & 1
    if p:
        return 0
    a = g & ((1 << (2 * d)) - 1)
    return ((-1) ** (bin(k & a).count('1') & 1)
            + (-1) ** (bin(k & swap_halves(a, d)).count('1') & 1))


def rho2(k, g, d=6):
    """Explicit 2x2 matrix for Ind_A^G(chi_k)."""
    p = (g >> (2 * d)) & 1
    a = g & ((1 << (2 * d)) - 1)
    ap = swap_halves(a, d) if p else a
    c1 = (-1) ** (bin(k & ap).count('1') & 1)
    c2 = (-1) ** (bin(k & swap_halves(ap, d)).count('1') & 1)
    D = np.array([[c1, 0], [0, c2]], dtype=complex)
    if p == 0:
        return D
    return np.array([[0, 1], [1, 0]], dtype=complex) @ D


def lin_reps(d=6):
    """(s, a) labels for the 2^{d+1} linear characters."""
    return [(s, a) for s in range(2) for a in range(1 << d)]


def k_reps(d=6):
    """Canonical swap-orbit representatives k < swap(k) in GF(2)^{2d}."""
    n = 1 << (2 * d)
    return [k for k in range(n) if k < swap_halves(k, d)]


def irrep_label(k, d=6):
    """Canonical label for a 2-dim irrep: (k, n_u, n_v, pop).

    n_u / n_v are popcounts of the upper / lower d-bit halves of k.
    """
    m = (1 << d) - 1
    ku, kv = (k >> d) & m, k & m
    return {
        'k': int(k),
        'n_u': bin(ku).count('1'),
        'n_v': bin(kv).count('1'),
        'pop': bin(k).count('1'),
        'ku': ku,
        'kv': kv,
    }


def conjugacy_class_index(g, gl_arr, d=6):
    """Conjugacy class of g under G_d (vectorized over gl_arr)."""
    p = (g >> (2 * d)) & 1
    a = g & ((1 << (2 * d)) - 1)
    hq = (gl_arr >> (2 * d)) & 1
    hv = gl_arr & ((1 << (2 * d)) - 1)
    swv = np.array([swap_halves(int(x), d) for x in hv], dtype=np.uint32)
    swa = swap_halves(a, d)
    sqa = np.where(hq == 0, a, swa)
    spv = np.where(p == 0, hv, swv)
    t = hv ^ sqa ^ spv
    res = (np.uint32(p) << (2 * d)) | t.astype(np.uint32)
    return set(int(x) for x in res)


def casimir_eigenvalue(j):
    return j * (j + 1)


# ----------------------------------------------------------------------
# G_d family and step-set helpers
# ----------------------------------------------------------------------
def compose_sig_int_d(a, b, d=6):
    """Compose packed signatures in G_d: result = a o b (apply b first)."""
    m = (1 << d) - 1
    pa = (a >> (2 * d)) & 1
    ua = (a >> d) & m
    va = a & m
    pb = (b >> (2 * d)) & 1
    ub = (b >> d) & m
    vb = b & m
    p = pa ^ pb
    u = (ua ^ ub) if pa == 0 else (ua ^ vb)
    v = (va ^ vb) if pa == 0 else (va ^ ub)
    return (p << (2 * d)) | (u << d) | v


def byte_signature_d(byte, d=6):
    """Packed signature of a single byte in G_d (always parity-1).

    Affine law: (u,v) -> (v xor eps_a, u xor micro xor eps_b).
    """
    from gyroscopic.hQVM.family import (
        intron_from_byte, eps_a_d, eps_b_d, intron_micro_ref_d, mask_d,
    )
    intron = intron_from_byte(byte, d)
    m = mask_d(d)
    tau_u = eps_a_d(intron, d) & m
    tau_v = (intron_micro_ref_d(intron, d) ^ eps_b_d(intron, d)) & m
    return (1 << (2 * d)) | (tau_u << d) | tau_v


def byte_step_set(d=6):
    """Distinct packed signatures of the alphabet A_d (subset of odd coset)."""
    from gyroscopic.hQVM.family import alphabet_size
    out = set()
    for b in range(alphabet_size(d)):
        out.add(byte_signature_d(b, d))
    return tuple(sorted(out))


def kernel_group_d(d=6):
    """G_d = (Z/2)^{2d} x| Z/2 generated by byte signatures.

    At d=6 this equals kernel_group() (order 8192).
    """
    gens = byte_step_set(d)
    T = set()
    for a in gens:
        for b in gens:
            T.add(compose_sig_int_d(a, b, d))
    O = set()
    for g in gens:
        for t in T:
            O.add(compose_sig_int_d(g, t, d))
    return tuple(sorted(T | O))


def apply_sig_int_d(g, u, v, d=6):
    """Apply packed signature g to (u,v) on Omega_d."""
    m = (1 << d) - 1
    p = (g >> (2 * d)) & 1
    tu = (g >> d) & m
    tv = g & m
    if p == 0:
        return (u ^ tu) & m, (v ^ tv) & m
    return (v ^ tu) & m, (u ^ tv) & m


def permutation_character_d(g, d=6):
    """Permutation character of G_d on Omega_d."""
    n = 1 << (2 * d)
    half = 1 << d
    p = (g >> (2 * d)) & 1
    u = (g >> d) & (half - 1)
    v = g & (half - 1)
    if p == 0:
        return n if (u == 0 and v == 0) else 0
    return half if u == v else 0


def clifford_irrep_counts(d=6):
    """(n_linear, n_2d, n_classes) from Clifford theory for G_d."""
    n_lin = 1 << (d + 1)          # 2 * 2^d swap-fixed extensions
    n_2d = ((1 << (2 * d)) - (1 << d)) // 2  # swap-orbits of size 2
    return n_lin, n_2d, n_lin + n_2d


# ----------------------------------------------------------------------
# Fourier / codebook helpers
# ----------------------------------------------------------------------
def fwht(a):
    """In-place-style fast Walsh–Hadamard transform (returns new array)."""
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


def fourier_matrix_linear(d=6):
    """Unitary character table of the translation subgroup A=(Z/2)^{2d}.

    Rows indexed by frequency k, columns by group element a; entries
    (-1)^{<k,a>} / 2^d.
    """
    n = 1 << (2 * d)
    F = np.empty((n, n), dtype=np.float64)
    scale = 1.0 / (1 << d)
    for k in range(n):
        for a in range(n):
            F[k, a] = scale * ((-1) ** (bin(k & a).count('1') & 1))
    return F


def omega_index(u, v, d=6):
    return (u << d) | v


def so3_encode_bits(R, budget):
    """Encode SO(3) matrix into an integer codebook index at given bit budget.

    Uses axis-angle quantized to 2^{budget} bins on a product chart
    (angle x hemisphere). Returns (code_int, R_hat) round-trip matrix.
    """
    ang = rotation_angle_from_matrix(R)
    ax = rotation_axis_from_matrix(R)
    if budget <= 0:
        return 0, np.eye(3)
    # Split bits: ~1/3 for angle in [0,pi], rest for axis on S^2 via spherical
    n_ang = max(1, budget // 3)
    n_ax = budget - n_ang
    n_th = max(1, n_ax // 2)
    n_ph = max(1, n_ax - n_th)
    i_ang = int(np.clip(round(ang / math.pi * ((1 << n_ang) - 1)), 0, (1 << n_ang) - 1))
    th = math.acos(float(np.clip(ax[2], -1.0, 1.0)))
    ph = math.atan2(ax[1], ax[0]) % (2 * math.pi)
    i_th = int(np.clip(round(th / math.pi * ((1 << n_th) - 1)), 0, (1 << n_th) - 1))
    i_ph = int(np.clip(round(ph / (2 * math.pi) * ((1 << n_ph) - 1)), 0, (1 << n_ph) - 1))
    code = (i_ang << (n_th + n_ph)) | (i_th << n_ph) | i_ph
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
    R_hat = rodrigues_exp(ang_h, ax_h)
    return code, R_hat


def so3_roundtrip(R, budget):
    """Encode/decode R at bit budget; return geodesic angle error (radians)."""
    _, R_hat = so3_encode_bits(R, budget)
    return rotation_angle_from_matrix(R.T @ R_hat)


# ----------------------------------------------------------------------
# 32-bit register atom helpers
# ----------------------------------------------------------------------
def pack_register32(intron8, state24):
    """32-bit atom: high 8 bits = intron, low 24 bits = Mac state."""
    return ((int(intron8) & 0xFF) << 24) | (int(state24) & 0xFFFFFF)


def unpack_register32(reg):
    return (int(reg) >> 24) & 0xFF, int(reg) & 0xFFFFFF


def shadow_register32(reg):
    """Projection pi: reg32 -> state24."""
    return int(reg) & 0xFFFFFF


def step_register32(byte, reg):
    """Step a 32-bit register by one byte (Mac via kernel, intron = byte).

    The intron slot stores the latest action byte (family phase carrier).
    Mac updates by the 24-bit kernel step.
    """
    _, mac = unpack_register32(reg)
    new_mac = step_state_by_byte(mac, byte) & 0xFFFFFF
    return pack_register32(byte & 0xFF, new_mac)