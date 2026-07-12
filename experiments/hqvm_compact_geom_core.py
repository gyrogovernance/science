"""
hqvm_compact_geom_core.py

Pure algebra layer for the compact geometry electroweak analysis.
No printing, no I/O, no side effects.

Layer 0: Named constants
Layer 1: Compact algebra (epsilon, eta, d_H, shell moments)
Layer 2: Electroweak laws (D2 through D5, all channels)
Layer 3: Backsolves (top, Higgs, Z, W, W/Z)
Layer 4: Shell transition algebra (exact fractions, carrier traces)
Layer 5: Lepton ladder coordinates
Layer 6: Observable definitions
"""
from __future__ import annotations
import math
import sys
from dataclasses import dataclass
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from fractions import Fraction
from math import comb
from typing import Sequence, TypedDict
try:
    import scipy.constants as sc
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
M_A: float = 1.0 / (2.0 * math.sqrt(2.0 * math.pi))
DELTA_BU: float = 0.19534217658
RHO: float = DELTA_BU / M_A
DELTA: float = 1.0 - RHO
E_EW_GEV: float = 246.22
ELECTRON_MASS_GEV: float = 0.000510998950714
F_CS_133_HZ: float = 9192631770.0
APERTURE_FRAME: float = 48.0
P_BOUNDARY: float = 47.0 / 48.0
KERNEL_BYTE_APERTURE: float = 5.0 / 256.0
_PHI_SU2: float = 2.0 * math.acos((1.0 + 2.0 * math.sqrt(2.0)) / 4.0)
SU2_RESIDUAL: float = (_PHI_SU2 - 3.0 * DELTA_BU) / M_A
LAMBDA_0: float = DELTA / math.sqrt(5.0)
STF_DIMENSION: int = 5
W2_ARCH_SHELL_TRAVERSE: int = 6
GYROTRIANGLE_G: float = 0.5
K4_CARDINALITY: int = 4
ALPHA_GEOMETRIC: float = DELTA_BU ** 4 / M_A
Q_DENSITY: float = 0.25
UV_IR_GYRATION_SQ: float = (2.0 * math.pi) ** 2
CODE_C1: int = comb(6, 1)
CODE_C2: int = comb(6, 2)
CODE_C3: int = comb(6, 3)
CODE_C4: int = comb(6, 4)
CODE_C5: int = comb(6, 5)
M_SHELL: int = sum((k * comb(6, k) for k in range(7)))
HORIZON_CARDINALITY: int = 64
OMEGA_SIZE: int = 4096
MONODROMY_PER_STAGE: float = DELTA_BU / 2.0
WZ_CODE_GAP: int = CODE_C2 - CODE_C1
WZ_OFFSET: int = WZ_CODE_GAP
C4_GRAVITY: Fraction = Fraction(-7, 4)
WZ_APERTURE_COEFF: float = CODE_C3 / 2.0
Z_H_OFFSET: int = 1 + CODE_C1 + CODE_C2
MUON_EQUATOR_COEFF: int = CODE_C3

@dataclass(frozen=True)
class CompactAlgebra:
    """All derived scalar quantities from DELTA."""
    delta: float
    epsilon: float
    eta: float
    d_h: float
    omega: float
    kappa: float
    sigma: float

def compact_algebra(delta: float=DELTA) -> CompactAlgebra:
    """Construct the full compact algebra from a given delta."""
    epsilon = 1.0 / delta - 48.0
    eta = M_A - DELTA_BU
    d_h = epsilon + 24.0 * delta
    omega = DELTA_BU / 2.0
    kappa = math.pi / 4.0 - 1.0 / math.sqrt(2.0)
    return CompactAlgebra(delta=delta, epsilon=epsilon, eta=eta, d_h=d_h, omega=omega, kappa=kappa, sigma=SU2_RESIDUAL)

def delta_self_consistency_rhs(delta: float, *, include_third_order: bool=True) -> float:
    """
    RHS of the aperture self-consistency equation:
        Delta = (5/256) * 2^(1/12) * (1 + (sqrt6/pi)*Delta^2 [+ (eta/eps)*Delta^3])
    """
    eps = 1.0 / delta - 48.0
    eta = M_A - DELTA_BU
    correction = 1.0 + math.sqrt(6.0) / math.pi * delta * delta
    if include_third_order and eps != 0.0:
        correction += eta / eps * delta ** 3
    return KERNEL_BYTE_APERTURE * 2.0 ** (1.0 / 12.0) * correction

def solve_delta_self_consistency(delta_guess: float=DELTA, *, include_third_order: bool=True, max_iter: int=30) -> tuple[float, float]:
    """
    Fixed-point iteration for the aperture self-consistency equation.
    Returns (delta_solution, residual).
    """
    delta = delta_guess
    for _ in range(max_iter):
        rhs = delta_self_consistency_rhs(delta, include_third_order=include_third_order)
        delta_next = 0.5 * delta + 0.5 * rhs
        if abs(delta_next - delta) <= 1e-15:
            delta = delta_next
            break
        delta = delta_next
    residual = delta - delta_self_consistency_rhs(delta, include_third_order=include_third_order)
    return (delta, residual)

@dataclass(frozen=True)
class ChannelCoeffs:
    """
    Coefficients for one electroweak channel in the carrier-trace expansion:
        L_i(D) = a*D + b + c*D^2 + p*(D/sqrt(5))*D^2 + q*D^4 + r5*D^5
    Identification: L_i(D) = log2(v/m_i).
    """
    label: str
    observable: str
    a: float
    b: float
    c: float
    p: float
    q: float
    r5: float
EW_CHANNEL_Q_WEIGHT: dict[str, int] = {'Top': 2, 'Higgs': 3, 'Z': 3, 'W': 4}
EW_CHANNEL_ORDER: tuple[str, ...] = ('Top', 'Higgs', 'Z', 'W')
K4_ELEMENTS: tuple[str, ...] = ('id', 'W2', "W2'", 'F')

def k4_element_for_channel(label: str) -> str:
    """K4 operator for an electroweak channel in CGM stage order."""
    return K4_ELEMENTS[EW_CHANNEL_ORDER.index(label)]

def derive_k4_flags_from_element(element: str) -> tuple[bool, bool, bool]:
    """
    Binary flags on the K4 edge walk from operator depth.

    base: egress half-word (W2) beyond CS identity
    rot:  ingress half-word (W2') at depth-four closure
    bal:  full holonomy cycle (F) at depth-eight Z2 return
    """
    idx = K4_ELEMENTS.index(element)
    return (idx >= 1, idx >= 2, idx >= 3)

def derive_k4_flags_from_path(label: str) -> tuple[bool, bool, bool]:
    """Flags from Omega byte-path length (2, 3, 4, 8 bytes for Top..W)."""
    n = len(ew_channel_path_word(label, 0))
    return (n >= 3, n >= 4, n >= 8)

def k4_channel_flags(label: str) -> tuple[bool, bool, bool]:
    """K4 edge walk flags (base, rot, bal) for channel label."""
    return derive_k4_flags_from_element(k4_element_for_channel(label))

def k4_channel_flags_dict() -> dict[str, tuple[bool, bool, bool]]:
    return {label: k4_channel_flags(label) for label in EW_CHANNEL_ORDER}

K4_CHANNEL_FLAGS: dict[str, tuple[bool, bool, bool]] = k4_channel_flags_dict()

@dataclass(frozen=True)
class ShellPathAStep:
    """Leading-order shell-path coordinate a_i from kernel grammar."""
    label: str
    k4_element: str
    formula: str
    value: int

def shell_path_a_ladder() -> tuple[ShellPathAStep, ...]:
    """Explicit shell-path ladder for a_i (horizon + code enumerator)."""
    return (
        ShellPathAStep('Top', 'id', '|H| + C2 - C1', HORIZON_CARDINALITY + CODE_C2 - CODE_C1),
        ShellPathAStep('Higgs', 'W2', 'M_shell/2', M_SHELL // 2),
        ShellPathAStep('Z', "W2'", 'M_shell/2 + C1 + C2', M_SHELL // 2 + CODE_C1 + CODE_C2),
        ShellPathAStep('W', 'F', 'M_shell/2 + 2*C2', M_SHELL // 2 + 2 * CODE_C2),
    )

def lambda_0_from_stf_dimension(delta: float = DELTA) -> float:
    """Third-order amplitude Delta/sqrt(n_STF); n_STF = 5 bulk STF shells."""
    return delta / math.sqrt(float(STF_DIMENSION))

def plaquette_defect_count(popcount_k: int) -> int:
    """Ordered byte-pair count with defect popcount k (gravity plaquette census)."""
    return 1024 * comb(6, popcount_k)

def plaquette_mean_defect() -> Fraction:
    """Mean popcount(d) over all 65536 byte-pair plaquettes."""
    total = sum(k * plaquette_defect_count(k) for k in range(7))
    return Fraction(total, 256 * 256)

def r5_constant_from_wz_gap() -> float:
    """Constant r5 offset -(C2-C1)/2 from the W/Z code-gap enumerator."""
    return -(CODE_C2 - CODE_C1) / 2.0

def channel_a_from_horizon(label: str) -> float:
    """Leading shell-population weight a_i from |H| and code enumerator."""
    if label == 'Top':
        return float(HORIZON_CARDINALITY + CODE_C2 - CODE_C1)
    if label == 'Higgs':
        return float(M_SHELL) / 2.0
    if label == 'Z':
        return float(M_SHELL) / 2.0 + CODE_C1 + CODE_C2
    if label == 'W':
        return float(M_SHELL) / 2.0 + 2.0 * CODE_C2
    raise KeyError(label)

def channel_b_from_boundary(label: str) -> float:
    """Depth-4 boundary projector contribution (T6: K4 composition)."""
    if label in ('Top', 'Higgs'):
        return -1.0
    if label in ('Z', 'W'):
        return -P_BOUNDARY
    raise KeyError(label)

def channel_c_from_shell_projection(label: str) -> float:
    """D^2 carrier-trace coefficient from gauge stage projection on shell algebra."""
    if label == 'Top':
        return Q_DENSITY
    base = -float(M_SHELL) / 8.0
    base_rot, ona_bal = k4_channel_flags(label)[1], k4_channel_flags(label)[2]
    c = base
    if base_rot:
        c += CODE_C1 / 4.0
    if ona_bal:
        c -= CODE_C3 / 2.0
    return c

def _pq_from_flags(base: bool, rot: bool, bal: bool) -> tuple[float, float]:
    """
    Gyrotriangle closure (delta=0) on the K4 edge walk.

    g = GYROTRIANGLE_G = 1/2 (ONA/CS). Trace-free: sum_i p_i = 0, sum_i q_i = 0.
    D^3 term in the law is p * LAMBDA_0 * D^2 (W2 pole-swap / STF dim 5).
    """
    g = GYROTRIANGLE_G
    b, r, bal_f = (float(base), float(rot), float(bal))
    p = 1.0 + -CODE_C1 / 2.0 * b + CODE_C1 / 4.0 * r + 4.0 * g * bal_f
    q = 5.0 / 4.0 - 4.0 * g * r - 2.0 * g * bal_f
    return (p, q)

def _r5_from_flags(base: bool, rot: bool, bal: bool) -> float:
    """
    D^5 code curvature from [12,6,2] enumerator and |H|=64 holographic correction.

    r5 = -(C2-C1)/2 + (|H|-(C2-C1))/8*(base-rot) + C2/8*bal
    """
    wz = float(WZ_CODE_GAP)
    return -wz / 2.0 + (HORIZON_CARDINALITY - wz) / 8.0 * (float(base) - float(rot)) + CODE_C2 / 8.0 * float(bal)

def channel_p_from_gyrotriangle(label: str) -> float:
    """D^3 W2 pole-swap phase charge (gyrotriangle + K4 edge walk)."""
    return _pq_from_flags(*k4_channel_flags(label))[0]

def channel_q_from_k4_closure(label: str) -> float:
    """D^4 F-gate Z2 holonomy charge (K4 closure, trace-free)."""
    return _pq_from_flags(*k4_channel_flags(label))[1]

def channel_r5_from_code_curvature(label: str) -> float:
    """D^5 [12,6,2] weight-enumerator curvature."""
    return _r5_from_flags(*k4_channel_flags(label))
_OBSERVABLE_BY_LABEL = {
    "Top": "Top quark mass energy",
    "Higgs": "Higgs mass energy",
    "Z": "Z boson mass energy",
    "W": "W boson mass energy",
}


def derive_channel_coeffs() -> tuple[ChannelCoeffs, ...]:
    """Build all channel coefficients from kernel algebra."""
    return tuple(
        ChannelCoeffs(
            label,
            _OBSERVABLE_BY_LABEL[label],
            a=channel_a_from_horizon(label),
            b=channel_b_from_boundary(label),
            c=channel_c_from_shell_projection(label),
            p=channel_p_from_gyrotriangle(label),
            q=channel_q_from_k4_closure(label),
            r5=channel_r5_from_code_curvature(label),
        )
        for label in ("Top", "Higgs", "Z", "W")
    )
CHANNELS: tuple[ChannelCoeffs, ...] = derive_channel_coeffs()
_pq = _pq_from_flags
_r5 = _r5_from_flags

def eval_law(ch: ChannelCoeffs, delta: float, *, order: int = 5) -> float:
    """
    Evaluate the carrier-trace expansion L_i(D) for channel i.

    L_i(D) = a*D + b + c*D^2 + p*(D/sqrt(5))*D^2 + q*D^4 + r5*D^5.
    Identification: L_i(D) = log2(v/m_i).
    """
    L = ch.a * delta + ch.b
    if order >= 2:
        L += ch.c * delta ** 2
    if order >= 3:
        L += ch.p * LAMBDA_0 * delta ** 2
    if order >= 4:
        L += ch.q * delta ** 4
    if order >= 5:
        L += ch.r5 * delta ** 5
    return L

def all_laws(delta: float=DELTA, *, order: int=5) -> dict[str, float]:
    """Return {label: L_i} for all four channels at given order."""
    return {ch.label: eval_law(ch, delta, order=order) for ch in CHANNELS}

@dataclass(frozen=True)
class LawTermBreakdown:
    """Per-order contributions to L_i(D) at fixed Delta."""
    d1: float
    d0: float
    d2: float
    d3: float
    d4: float
    d5: float

    @property
    def total(self) -> float:
        return self.d1 + self.d0 + self.d2 + self.d3 + self.d4 + self.d5

def eval_law_terms(ch: ChannelCoeffs, delta: float, *, order: int = 5) -> LawTermBreakdown:
    """Carrier-trace expansion split by polynomial order."""
    d1 = ch.a * delta
    d0 = ch.b
    d2 = ch.c * delta ** 2 if order >= 2 else 0.0
    d3 = ch.p * LAMBDA_0 * delta ** 2 if order >= 3 else 0.0
    d4 = ch.q * delta ** 4 if order >= 4 else 0.0
    d5 = ch.r5 * delta ** 5 if order >= 5 else 0.0
    return LawTermBreakdown(d1=d1, d0=d0, d2=d2, d3=d3, d4=d4, d5=d5)

def _import_gravity_path():
    exp = Path(__file__).resolve().parent
    if str(exp) not in sys.path:
        sys.path.insert(0, str(exp))
    import hqvm_gravity_common as g
    return g

def ew_channel_path_word(label: str, micro_ref: int) -> list[int]:
    """
    Byte word on Omega for channel label (K4 / stage structure).

    Top:   W2 (egress half-word, q=2)
    Higgs: W2 + family-2 byte (base flag path)
    Z:     F = W2 o W2' (4 bytes)
    W:     Z2 holonomy cycle (8 bytes, F o F)
    """
    g = _import_gravity_path()
    m = micro_ref & 63
    if label == 'Top':
        return g.W2_word(m)
    if label == 'Higgs':
        return g.W2_word(m) + [g.byte_from_family_and_micro(2, m)]
    if label == 'Z':
        return g.F_cycle_word(m)
    if label == 'W':
        return g.cycle_word_for_micro(m)
    raise KeyError(label)

def ew_path_step_kappa(arch_shell: int) -> float:
    """Per-step bulk carrier-trace weight (horizon steps contribute 0)."""
    if arch_shell in (0, 6) or arch_shell < 0 or arch_shell > 6:
        return 0.0
    return DELTA * comb(6, arch_shell) / 64.0

def bulk_carrier_sum_for_micro(label: str, micro_ref: int) -> float:
    """Bulk carrier-trace sum along channel path for one micro-ref."""
    g = _import_gravity_path()
    word = ew_channel_path_word(label, micro_ref)
    bulk = 0.0
    for row in g.trace_word_steps(word, micro_ref=micro_ref)[1:]:
        bulk += ew_path_step_kappa(int(row["arch_shell"]))
    return bulk


def bulk_carrier_sum_weighted(label: str) -> float:
    """64-micro binomial average of per-path bulk carrier trace (gravity Lemma A analog)."""
    g = _import_gravity_path()
    w_sum = 0.0
    bulk_sum = 0.0
    for m in range(64):
        w = g.weights_pop[m]
        bulk_sum += w * bulk_carrier_sum_for_micro(label, m)
        w_sum += w
    return bulk_sum / w_sum if w_sum > 0 else 0.0

@dataclass(frozen=True)
class EwOmegaPathTrace:
    """Explicit shell-path trace on Omega for one EW channel."""
    label: str
    micro_ref: int
    word_len: int
    arch_path: tuple[int, ...]
    bulk_carrier_sum: float
    l_algebra: float
    l_from_path: float
    path_matches_algebra: bool
    template_ok: bool

def _path_template_ok(label: str, arch_path: tuple[int, ...], micro_ref: int) -> bool:
    k = bin(micro_ref).count('1')
    if label == 'W':
        return arch_path == (k, 6, k, 0, k, 6, k, 0)
    if label == 'Z':
        return arch_path == (k, 6, k, 0)
    if label == 'Top':
        return len(arch_path) == 2
    if label == 'Higgs':
        return len(arch_path) == 3
    return True
# Complement-horizon shell (P_6 projector support).
P6_SHELL_INDEX: int = 6
P6_SUPPORT_STATES: int = HORIZON_CARDINALITY

@dataclass(frozen=True)
class CarrierTracePolynomialBreakdown:
    """
    Five-order polynomial L_i(D) assembled from carrier-trace / shell-path data.

    Each term is a carrier-trace order.
    bulk_path_sum certifies the byte word; L is the carrier-trace polynomial.
    """
    label: str
    q_weight: int
    carrier_trace_q: Fraction
    byte_count: int
    bulk_path_sum: float
    d1: float
    d0: float
    d2: float
    d3: float
    d4: float
    d5: float
    source_d1: str
    source_d0: str
    source_d2: str
    source_d3: str
    source_d4: str
    source_d5: str

    @property
    def total(self) -> float:
        return self.d1 + self.d0 + self.d2 + self.d3 + self.d4 + self.d5
def carrier_trace_polynomial_breakdown(
    label: str, delta: float = DELTA, *, order: int = 5
) -> CarrierTracePolynomialBreakdown:
    """
    Build L_i(D) from carrier traces C(q), shell-path byte count, and K4 flags.

    Carrier-trace polynomial (kernel-derived, not fitted):
      L = a*D + b + c*D^2 + p*Lambda_0*D^2 + q*D^4 + r5*D^5
    """
    ch = channel_by_label(label)
    terms = eval_law_terms(ch, delta, order=order)
    q = EW_CHANNEL_Q_WEIGHT[label]
    ct = carrier_trace(q)
    byte_count = 4 * comb(6, q)
    return CarrierTracePolynomialBreakdown(
        label=label,
        q_weight=q,
        carrier_trace_q=ct,
        byte_count=byte_count,
        bulk_path_sum=bulk_carrier_sum_weighted(label),
        d1=terms.d1,
        d0=terms.d0,
        d2=terms.d2,
        d3=terms.d3,
        d4=terms.d4,
        d5=terms.d5,
        source_d1=f"a={ch.a:.4g} horizon; 4*C(6,{q})={byte_count} path bytes",
        source_d0=f"b={ch.b:.6g} boundary (T_F_composition)",
        source_d2=f"c={ch.c:.6g}*D^2; C({q})={float(ct):.6g} (T_carrier_traces)",
        source_d3=f"p={ch.p:.4g}*Lambda_0*D^2 (T_W2_swap)",
        source_d4=f"q={ch.q:.4g}*D^4 (T_F_preserve)",
        source_d5=f"r5={ch.r5:.4g}*D^5 (T_code_curvature_r5)",
    )

def eval_law_from_omega_path(label: str, delta: float=DELTA, *, order: int=5) -> float:
    """
    L_i(D) from Omega path + carrier-trace polynomial assembly.

    Coefficients come from shell-path derivation (horizon, C(q), K4 flags on the
    channel byte word). bulk_carrier_sum_weighted() supplies the bulk path check;
    L is the carrier-trace polynomial, not the bulk path sum.
    """
    return carrier_trace_polynomial_breakdown(label, delta, order=order).total

def ew_omega_path_trace(label: str, micro_ref: int=63, delta: float=DELTA) -> EwOmegaPathTrace:
    """Trace channel path on Omega and compare to path-certified algebraic law."""
    g = _import_gravity_path()
    word = ew_channel_path_word(label, micro_ref)
    arch = tuple((int(r['arch_shell']) for r in g.trace_word_steps(word, micro_ref=micro_ref)[1:]))
    bulk = bulk_carrier_sum_for_micro(label, micro_ref)
    l_alg = eval_law(channel_by_label(label), delta)
    l_path = eval_law_from_omega_path(label, delta)
    return EwOmegaPathTrace(
        label=label,
        micro_ref=micro_ref,
        word_len=len(word),
        arch_path=arch,
        bulk_carrier_sum=bulk,
        l_algebra=l_alg,
        l_from_path=l_path,
        path_matches_algebra=abs(l_alg - l_path) < 1e-12,
        template_ok=_path_template_ok(label, arch, micro_ref),
    )

def ew_omega_path_traces_all(micro_ref: int=63, delta: float=DELTA) -> tuple[EwOmegaPathTrace, ...]:
    return tuple((ew_omega_path_trace(lbl, micro_ref, delta) for lbl in ('Top', 'Higgs', 'Z', 'W')))

def verify_omega_path_ladder(delta: float = DELTA) -> dict[str, bool | float]:
    """Omega path traces, gravity cycle sum, and carrier-trace polynomial checks."""
    g = _import_gravity_path()
    bulk_w = bulk_carrier_sum_weighted("W")
    gravity_cycle_sum = sum(
        g.weights_pop[m]
        * sum(
            ew_path_step_kappa(int(r["arch_shell"]))
            for r in g.trace_word_steps(g.cycle_word_for_micro(m), micro_ref=m)[1:]
        )
        for m in range(64)
    ) / sum(g.weights_pop.values())
    traces = {
        t.label: t
        for t in (ew_omega_path_trace(lbl, 63, delta) for lbl in ("Top", "Higgs", "Z", "W"))
    }
    ok_poly = all(
        abs(carrier_trace_polynomial_breakdown(ch.label, delta).total - eval_law(ch, delta))
        < 1e-12
        for ch in CHANNELS
    )
    return {
        "bulk_W_matches_gravity_cycle": abs(bulk_w - gravity_cycle_sum) < 1e-12,
        "bulk_W": bulk_w,
        "gravity_cycle_sum": gravity_cycle_sum,
        "all_L_from_path": all(t.path_matches_algebra for t in traces.values()),
        "all_L_polynomial_matches_eval_law": ok_poly,
        "W_template_ok": traces["W"].template_ok,
        "Z_template_ok": traces["Z"].template_ok,
    }

def mass_ppm_err(predicted: float, observed: float) -> float:
    """Parts-per-million relative error (predicted/observed - 1) * 1e6."""
    if observed == 0.0:
        return float('nan')
    return (predicted / observed - 1.0) * 1000000.0

@dataclass(frozen=True)
class EwMassAnchor:
    """Path-predicted mass vs PDG anchor (L_i(D) = log2(v/m_i) identification)."""
    label: str
    observable: str
    mass_obs_gev: float
    mass_pred_gev: float
    l_obs: float
    l_pred: float
    n_obs: float
    n_pred: float
    shell_residual_ticks: float
    mass_ppm: float

    @property
    def shell_dist_obs(self) -> float:
        return self.n_obs

    @property
    def shell_dist_pred(self) -> float:
        return self.n_pred

def predicted_mass_from_path(label: str, delta: float=DELTA, v: float=E_EW_GEV, *, order: int=5) -> float:
    """m_pred = v * 2^(-L_i(D)) with L from Omega path-certified expansion."""
    l_pred = eval_law_from_omega_path(label, delta)
    if order < 5:
        ch = channel_by_label(label)
        l_pred = eval_law(ch, delta, order=order)
    return v * 2.0 ** (-l_pred)

def ew_mass_anchor_rows(observed: dict[str, float], delta: float=DELTA, v: float | None=None, *, order: int=5) -> tuple[EwMassAnchor, ...]:
    """Validate L_i(D) = log2(v/m_i) using carrier-path decomposition rows."""
    if v is None:
        v = observed.get('Electroweak scale', E_EW_GEV)
    rows: list[EwMassAnchor] = []
    for d in ew_carrier_path_decomposition(observed, delta, v, order=order):
        m_pred = v * 2.0 ** (-d.l_pred)
        rows.append(EwMassAnchor(label=d.label, observable=d.observable, mass_obs_gev=observed[d.observable], mass_pred_gev=m_pred, l_obs=d.l_obs, l_pred=d.l_pred, n_obs=d.shell_dist, n_pred=d.l_pred / delta, shell_residual_ticks=d.residual_ticks, mass_ppm=mass_ppm_err(m_pred, observed[d.observable])))
    return tuple(rows)

def verify_ew_mass_identification(observed: dict[str, float], delta: float=DELTA, v: float | None=None) -> dict[str, bool | float | tuple[EwMassAnchor, ...]]:
    """
    Anchor checks: path-predicted masses vs PDG, alpha_geometric vs coupling alpha.
    """
    if v is None:
        v = observed.get('Electroweak scale', E_EW_GEV)
    rows = ew_mass_anchor_rows(observed, delta, v, order=5)
    by_label = {r.label: r for r in rows}
    m = model_masses_from_path(delta, v)
    mc = ew_couplings_from_masses(m['m_top'], m['m_higgs'], m['m_z'], m['m_w'], v)
    m_obs = observed
    rc = ew_couplings_from_masses(m_obs['Top quark mass energy'], m_obs['Higgs mass energy'], m_obs['Z boson mass energy'], m_obs['W boson mass energy'], v)
    ppm_top = abs(by_label['Top'].mass_ppm) if 'Top' in by_label else float('nan')
    ppm_higgs = abs(by_label['Higgs'].mass_ppm) if 'Higgs' in by_label else float('nan')
    ppm_z = abs(by_label['Z'].mass_ppm) if 'Z' in by_label else float('nan')
    ppm_w = abs(by_label['W'].mass_ppm) if 'W' in by_label else float('nan')
    ppm_wz = max(ppm_z, ppm_w)
    alpha_rel = mc.alpha_ew_inv / rc.alpha_ew_inv - 1.0 if rc.alpha_ew_inv else float('nan')
    return {'rows': rows, 'sub_ppm_top': ppm_top < 1.0, 'sub_ppm_wz': ppm_wz < 1.0, 'ppm_top': ppm_top, 'ppm_higgs': ppm_higgs, 'ppm_z': ppm_z, 'ppm_w': ppm_w, 'alpha_geometric': ALPHA_GEOMETRIC, 'alpha_inv_pred': mc.alpha_ew_inv, 'alpha_inv_pdg': rc.alpha_ew_inv, 'alpha_rel_err': alpha_rel, 'alpha_digits_match': abs(alpha_rel) < 1e-06, 'max_shell_ticks': max((abs(r.shell_residual_ticks) for r in rows)), 'shell_ticks_ok': all((abs(r.shell_residual_ticks) < 0.2 for r in rows))}

def model_masses_from_path(delta: float=DELTA, v: float=E_EW_GEV) -> dict[str, float]:
    """Masses from Omega path-certified five-order law."""
    return {'m_top': predicted_mass_from_path('Top', delta, v), 'm_higgs': predicted_mass_from_path('Higgs', delta, v), 'm_z': predicted_mass_from_path('Z', delta, v), 'm_w': predicted_mass_from_path('W', delta, v)}

def verify_ew_ladder_derivations() -> dict[str, bool | float]:
    """Consistency checks for the five-order EW ladder derivations."""
    sum_p = sum(ch.p for ch in CHANNELS)
    sum_q = sum(ch.q for ch in CHANNELS)
    q_w = channel_q_from_k4_closure("W")
    c4_w = float(C4_GRAVITY)
    coeffs_ok = all(
        ch.a == channel_a_from_horizon(ch.label)
        and ch.b == channel_b_from_boundary(ch.label)
        and ch.c == channel_c_from_shell_projection(ch.label)
        and ch.p == channel_p_from_gyrotriangle(ch.label)
        and ch.q == channel_q_from_k4_closure(ch.label)
        and ch.r5 == channel_r5_from_code_curvature(ch.label)
        for ch in CHANNELS
    )
    ladder_ok = all(
        int(channel_a_from_horizon(step.label)) == step.value
        for step in shell_path_a_ladder()
    )
    return {
        "all_channels_coeffs": coeffs_ok,
        "shell_path_a_ladder": ladder_ok,
        "sum_p_trace_free": abs(sum_p) < 1e-12,
        "sum_q_trace_free": abs(sum_q) < 1e-12,
        "q_W_equals_c4_gravity": abs(q_w - c4_w) < 1e-12,
        "lambda_0_stf": abs(LAMBDA_0 - lambda_0_from_stf_dimension()) < 1e-15,
        "r5_top_matches_wz_gap": abs(channel_r5_from_code_curvature("Top") - r5_constant_from_wz_gap()) < 1e-12,
        "plaquette_mean_defect_is_3": plaquette_mean_defect() == Fraction(3, 1),
        "sum_p": sum_p,
        "sum_q": sum_q,
        "q_W": q_w,
        "c4_gravity": c4_w,
    }

def verify_k4_channel_derivation() -> dict[str, bool]:
    """K4 operator assignment and path-length flag derivation."""
    flags_from_element = {
        label: derive_k4_flags_from_element(k4_element_for_channel(label))
        for label in EW_CHANNEL_ORDER
    }
    flags_from_path = {
        label: derive_k4_flags_from_path(label) for label in EW_CHANNEL_ORDER
    }
    return {
        "element_matches_path": flags_from_element == flags_from_path,
        "top_is_id": k4_element_for_channel("Top") == "id",
        "w_is_F": k4_element_for_channel("W") == "F",
        "flags_match_declared": flags_from_element == k4_channel_flags_dict(),
    }

@dataclass(frozen=True)
class EwAssignmentAudit:
    """Uniqueness audit of the algebraically derived K4 channel assignment."""
    raw_assignments: int
    trace_free_candidates: int
    declared_rank: int
    rank1_max_tick_error: float
    rank2_max_tick_error: float | None

def electroweak_assignment_uniqueness_audit(
    observed: dict[str, float],
    delta: float = DELTA,
    v: float | None = None,
) -> EwAssignmentAudit:
    """
    Rank trace-free flag assignments by max tick error.

    The declared assignment follows the K4 operator walk; this audit verifies
  uniqueness among grammar-consistent candidates.
    """
    if v is None:
        v = observed.get("Electroweak scale", E_EW_GEV)

    def _obs_n(mass_key: str) -> float:
        return math.log2(v / observed[mass_key]) / delta

    observed_n = {
        "Top": _obs_n("Top quark mass energy"),
        "Higgs": _obs_n("Higgs mass energy"),
        "Z": _obs_n("Z boson mass energy"),
        "W": _obs_n("W boson mass energy"),
    }
    base = {ch.label: (ch.a, ch.b, ch.c) for ch in CHANNELS}
    trace_rows: list[float] = []
    declared_flags = k4_channel_flags_dict()
    declared_err: float | None = None

    for f_top in range(8):
        a_t, r_t, b_t = (bool((f_top >> 2) & 1), bool((f_top >> 1) & 1), bool(f_top & 1))
        p_t, q_t, r5_t = _pq_from_flags(a_t, r_t, b_t)[0], _pq_from_flags(a_t, r_t, b_t)[1], _r5_from_flags(a_t, r_t, b_t)
        for f_h in range(8):
            a_h, r_h, b_h = (bool((f_h >> 2) & 1), bool((f_h >> 1) & 1), bool(f_h & 1))
            p_h, q_h, r5_h = _pq_from_flags(a_h, r_h, b_h)[0], _pq_from_flags(a_h, r_h, b_h)[1], _r5_from_flags(a_h, r_h, b_h)
            for f_z in range(8):
                a_z, r_z, b_z = (bool((f_z >> 2) & 1), bool((f_z >> 1) & 1), bool(f_z & 1))
                p_z, q_z, r5_z = _pq_from_flags(a_z, r_z, b_z)[0], _pq_from_flags(a_z, r_z, b_z)[1], _r5_from_flags(a_z, r_z, b_z)
                for f_w in range(8):
                    a_w, r_w, b_w = (bool((f_w >> 2) & 1), bool((f_w >> 1) & 1), bool(f_w & 1))
                    p_w, q_w, r5_w = _pq_from_flags(a_w, r_w, b_w)[0], _pq_from_flags(a_w, r_w, b_w)[1], _r5_from_flags(a_w, r_w, b_w)
                    p_sum = p_t + p_h + p_z + p_w
                    q_sum = q_t + q_h + q_z + q_w
                    if abs(p_sum) > 1e-12 or abs(q_sum) > 1e-12:
                        continue
                    max_err = 0.0
                    for label, obs_n, (a_ch, b_ch, c_ch), p_val, q_val, r5_val in [
                        ("Top", observed_n["Top"], base["Top"], p_t, q_t, r5_t),
                        ("Higgs", observed_n["Higgs"], base["Higgs"], p_h, q_h, r5_h),
                        ("Z", observed_n["Z"], base["Z"], p_z, q_z, r5_z),
                        ("W", observed_n["W"], base["W"], p_w, q_w, r5_w),
                    ]:
                        l_pred = eval_law(
                            ChannelCoeffs(label, "", a_ch, b_ch, c_ch, p_val, q_val, r5_val),
                            delta,
                        )
                        max_err = max(max_err, abs(l_pred / delta - obs_n))
                    trace_rows.append(max_err)
                    flags = {
                        "Top": (bool(a_t), bool(r_t), bool(b_t)),
                        "Higgs": (bool(a_h), bool(r_h), bool(b_h)),
                        "Z": (bool(a_z), bool(r_z), bool(b_z)),
                        "W": (bool(a_w), bool(r_w), bool(b_w)),
                    }
                    if flags == declared_flags:
                        declared_err = max_err

    trace_rows.sort()
    declared_rank = trace_rows.index(declared_err) + 1 if declared_err is not None else -1
    return EwAssignmentAudit(
        raw_assignments=4096,
        trace_free_candidates=len(trace_rows),
        declared_rank=declared_rank,
        rank1_max_tick_error=trace_rows[0] if trace_rows else float("nan"),
        rank2_max_tick_error=trace_rows[1] if len(trace_rows) > 1 else None,
    )

@dataclass(frozen=True)
class EwCarrierPathDecomposition:
    """Observed mass gap decomposed by carrier-trace order (modeling + theorem)."""
    label: str
    observable: str
    l_obs: float
    shell_dist: float
    terms: LawTermBreakdown
    l_pred: float
    residual_ticks: float
    residual_d6: float

def ew_carrier_path_decomposition(observed: dict[str, float], delta: float=DELTA, v: float | None=None, *, order: int=5) -> tuple[EwCarrierPathDecomposition, ...]:
    """
    Decompose log2(v/m) into carrier-trace orders for each EW channel.

    shell_dist = L_obs / Delta (mass-coordinate gap in aperture units).
    """
    if v is None:
        v = observed.get('Electroweak scale', E_EW_GEV)
    rows: list[EwCarrierPathDecomposition] = []
    for ch in CHANNELS:
        mass = observed.get(ch.observable)
        if mass is None:
            continue
        l_obs = math.log2(v / mass)
        shell_dist = l_obs / delta
        terms = eval_law_terms(ch, delta, order=order)
        l_pred = terms.total
        rows.append(EwCarrierPathDecomposition(label=ch.label, observable=ch.observable, l_obs=l_obs, shell_dist=shell_dist, terms=terms, l_pred=l_pred, residual_ticks=(l_obs - l_pred) / delta, residual_d6=(l_obs - l_pred) / delta ** 6))
    return tuple(rows)

def channel_by_label(label: str) -> ChannelCoeffs:
    for ch in CHANNELS:
        if ch.label == label:
            return ch
    raise KeyError(f'No channel with label {label!r}')

def channel_by_observable(name: str) -> ChannelCoeffs:
    for ch in CHANNELS:
        if ch.observable == name:
            return ch
    raise KeyError(f'No channel for observable {name!r}')

@dataclass(frozen=True)
class DeltaBacksolve:
    source: str
    equation: str
    delta_back: float

    @property
    def delta_err(self) -> float:
        return self.delta_back - DELTA

def _solve_top(l_t: float) -> float:
    """L_t = 73D - 1 + (1/4)D^2  =>  quadratic in D."""
    a, b, c = (Q_DENSITY, 73.0, -(1.0 + l_t))
    disc = b * b - 4.0 * a * c
    return (-b + math.sqrt(disc)) / (2.0 * a)

def _solve_higgs(l_h: float) -> float:
    """L_H = 96D - 1 - 24D^2  =>  quadratic in D."""
    a, b, c = (24.0, -96.0, 1.0 + l_h)
    disc = b * b - 4.0 * a * c
    return (-b - math.sqrt(disc)) / (2.0 * a)

def _solve_z(l_z: float) -> float:
    """L_Z = 117D - 47/48 - (45/2)D^2  =>  quadratic in D."""
    a, b, c = (45.0 / 2.0, -117.0, P_BOUNDARY + l_z)
    disc = b * b - 4.0 * a * c
    return (-b - math.sqrt(disc)) / (2.0 * a)

def _solve_w(l_w: float) -> float:
    """L_W = 126D - 47/48 - (65/2)D^2  =>  quadratic in D."""
    a, b, c = (65.0 / 2.0, -126.0, P_BOUNDARY + l_w)
    disc = b * b - 4.0 * a * c
    return (-b - math.sqrt(disc)) / (2.0 * a)

def _solve_wz_phase_corrected(l_wz: float, *, guess: float=0.0207, iters: int=20) -> float:
    """
    Promoted W/Z split law (Newton's method):
        log2(mZ/mW) = D * [9 - 10D + 2D^2/sqrt(5) - D^3]
    """
    D = guess
    s5 = math.sqrt(5.0)
    for _ in range(iters):
        f = 9.0 * D - 10.0 * D ** 2 + 2.0 * D ** 3 / s5 - D ** 4 - l_wz
        fp = 9.0 - 20.0 * D + 6.0 * D ** 2 / s5 - 4.0 * D ** 3
        D -= f / fp
    return D

def electroweak_backsolves(observed: dict[str, float]) -> tuple[DeltaBacksolve, ...]:
    """
    Compute Delta backsolves from a dict of {observable_name: mass_GeV}.
    Expected keys: 'Top quark mass energy', 'Higgs mass energy',
                   'Z boson mass energy', 'W boson mass energy',
                   'Electroweak scale'.
    """
    v = observed['Electroweak scale']
    l_t = math.log2(v / observed['Top quark mass energy'])
    l_h = math.log2(v / observed['Higgs mass energy'])
    l_z = math.log2(v / observed['Z boson mass energy'])
    l_w = math.log2(v / observed['W boson mass energy'])
    l_wz = math.log2(observed['Z boson mass energy'] / observed['W boson mass energy'])
    return (DeltaBacksolve('Top', '73D-1+(1/4)D^2', _solve_top(l_t)), DeltaBacksolve('Higgs', '96D-1-24D^2', _solve_higgs(l_h)), DeltaBacksolve('Z', '117D-47/48-(45/2)D^2', _solve_z(l_z)), DeltaBacksolve('W', '126D-47/48-(65/2)D^2', _solve_w(l_w)), DeltaBacksolve('W/Z', 'D[9-10D+2D^2/sqrt(5)-D^3]', _solve_wz_phase_corrected(l_wz)))

def four_point_consensus(backsolves: tuple[DeltaBacksolve, ...]) -> DeltaBacksolve:
    """Mean of Top + Higgs + Z + W backsolves."""
    sources = ('Top', 'Higgs', 'Z', 'W')
    vals = [bs.delta_back for bs in backsolves if bs.source in sources]
    return DeltaBacksolve('4-pt mean', 'mean(Top,Higgs,Z,W)', sum(vals) / len(vals))

def shell_transition_prob(w_src: int, q_weight: int, w_dst: int) -> Fraction:
    """
    Exact transition probability P(w_src --(q_weight)--> w_dst)
    in the Hamming-scheme on GF(2)^6.
    """
    w, j, wp = (int(w_src), int(q_weight), int(w_dst))
    delta = w + j - wp
    if delta < 0 or delta % 2:
        return Fraction(0)
    t = delta // 2
    if t < 0 or t > min(w, j):
        return Fraction(0)
    if j - t < 0 or j - t > 6 - w:
        return Fraction(0)
    return Fraction(comb(w, t) * comb(6 - w, j - t), comb(6, j))

def shell_transition_matrix(q: int) -> tuple[tuple[Fraction, ...], ...]:
    return tuple((tuple((shell_transition_prob(w, q, wp) for wp in range(7))) for w in range(7)))

def shell_trace(q: int) -> Fraction:
    M = shell_transition_matrix(q)
    return sum((M[i][i] for i in range(7)), Fraction(0))

def shell_return_trace(q: int) -> Fraction:
    M = shell_transition_matrix(q)
    return sum((M[i][k] * M[k][i] for i in range(7) for k in range(7)), Fraction(0))

def carrier_trace(q: int) -> Fraction:
    """Tr(M) if non-zero, else Tr(M^2)."""
    t = shell_trace(q)
    return t if t != 0 else shell_return_trace(q)
CARRIER_TRACES: tuple[Fraction, ...] = tuple((carrier_trace(q) for q in range(7)))

def lepton_base_n(k: int, lepton: str) -> float:
    """
    Base mass-coordinate n = k * |H| + r(lepton) from horizon/shell decomposition.

    r(tau)      = M_shell / 8           = 24
    r(muon)     = M_shell / 8 + M_shell / 48  = 28
    r(electron) = M_shell / 8 - M_shell / 24  = 16
    """
    m = float(M_SHELL)
    if lepton == 'tau':
        r = m / 8.0
    elif lepton in ('mu', 'muon'):
        r = m / 8.0 + m / 48.0
    elif lepton in ('e', 'electron'):
        r = m / 8.0 - m / 24.0
    else:
        raise ValueError(f'Unknown lepton: {lepton!r}')
    return k * HORIZON_CARDINALITY + r

def lepton_delta_coords(n_h_law: float, delta: float=DELTA) -> dict[str, float]:
    """
    Lepton EW mass-coordinate gaps.
    n_h_law: the Higgs n-coordinate (L_H/delta) from the five-order law.

    Returns dict with keys: n_tau, n_mu, n_electron.
    """
    higgs_memory = KERNEL_BYTE_APERTURE / n_h_law
    return {'n_tau': lepton_base_n(5, 'tau') - (DELTA_BU + 5.0 * delta), 'n_mu': lepton_base_n(8, 'mu') + MUON_EQUATOR_COEFF * delta, 'n_electron': lepton_base_n(14, 'electron') + SU2_RESIDUAL + delta / (n_h_law * delta / KERNEL_BYTE_APERTURE)}

@dataclass(frozen=True)
class LeptonLadderRow:
    """One lepton in the temporal ladder with order-by-order residuals."""
    label: str
    k: int
    r: float
    base_n: float
    n_model_d1: float
    n_model_d2: float
    n_model_d3: float
    n_observed: float
    resid_d1: float
    resid_d2: float
    resid_d3: float
    law_d1: str
    law_d2: str
    law_d3: str

@dataclass(frozen=True)
class LeptonD3TransitionCost:
    label: str
    q_from: int
    q_to: int
    carrier_from: Fraction
    carrier_to: Fraction
    carrier_delta: Fraction
    dyadic: Fraction
    coeff_delta3: Fraction
    normalized_64_cost: Fraction
    numerator_rule: str
    numerator_rule_value: Fraction
    numerator_rule_matches: bool

@dataclass(frozen=True)
class LeptonD3SelectionRule:
    label: str
    rule: str
    value: Fraction
    target: Fraction

    @property
    def matches(self) -> bool:
        return self.value == self.target

@dataclass(frozen=True)
class ShellSpectralWeight:
    q: int
    byte_count: int
    trace: Fraction
    return_trace: Fraction
    carrier: Fraction
    shell_frobenius_sq: Fraction
    full_operator_hs_sq: Fraction

@dataclass(frozen=True)
class K4PhaseSelector:
    label: str
    base: bool
    rot: bool
    bal: bool
    p: Fraction
    q: Fraction
    r5: Fraction
    normalized_64_cost: Fraction
    rule: str

@dataclass(frozen=True)
class QSupportDuality:
    q_a: int
    q_b: int
    size_a: int
    size_b: int
    direct_intersection: int
    complement_pair_hits: int

    @property
    def complement_duality_complete(self) -> bool:
        return self.size_a == self.size_b == self.complement_pair_hits

@dataclass(frozen=True)
class ByteArchetypeProbe:
    gene_mic: int
    intron: int
    family: int
    micro_ref: int
    q_weight: int
    archetype_shadow: Fraction
    carrier_delta: Fraction
    shadow_carrier_coeff: Fraction
    residual_carrier_sum: Fraction
    matches_residual: bool

@dataclass(frozen=True)
class HorizonGateSelection:
    byte: int
    gate: str
    intron: int
    family: int
    micro_ref: int
    q_weight: int
    is_s_gate: bool
    zero_intron: bool
    zero_payload: bool
    zero_family: bool
    selected_by_electron_reset: bool

@dataclass(frozen=True)
class SourceTraceabilityProbe:
    """
    Finite audit for the terminal lepton source-reset rule.

    The source-traceability condition selects the unique q=0 S-gate with zero
    intron, zero payload mutation, and zero spinorial family displacement.
    """
    selected_byte: int
    selected_count: int
    q_kernel_size: int
    selected_is_gene_mic: bool
    dyadic_atom: Fraction
    carrier_delta: Fraction
    carrier_shadow: Fraction
    electron_neutral: Fraction
    electron_implemented: Fraction
    electron_derived: Fraction
    closes_electron_dyadic: bool

@dataclass(frozen=True)
class StrongScaleBulkProbe:
    """
    Strong-scale mass-coordinate placement on the EW aperture axis.
    Lambda_QCD remains an external input; this probe only audits where it lands
    in the compact kernel coordinate system.
    """
    lambda_qcd_gev: float
    n_qcd: float
    n_phase_64: float
    n_phase_48: float
    bulk_states: int
    boundary_states: int
    bulk_fraction: Fraction
    boundary_fraction: Fraction
    placed_in_relational_bulk: bool

@dataclass(frozen=True)
class LeptonHorizonWrapExhaustionProbe:
    """
    Exhaustive audit of the proposed lepton horizon-wrap anchors.

    The broad constraint set deliberately contains only ordering and the
    tau/muon D3 cost budget. If this admits multiple candidates, the result
    marks the horizon-wrap rule as a selection rule still needing proof.
    """
    max_k: int
    q_path: tuple[int, int, int]
    expected_k_path: tuple[int, int, int]
    broad_valid_count: int
    broad_valid_examples: tuple[tuple[int, int, int], ...]
    horizon_rule_valid_count: int
    horizon_rule_valid_paths: tuple[tuple[int, int, int], ...]
    tau_muon_budget: Fraction
    tau_muon_budget_matches: bool
    delta_tau_muon: int
    delta_muon_electron: int
    expected_delta_tau_muon: Fraction
    expected_delta_muon_electron: Fraction
    unique_without_horizon_rule: bool
    unique_with_horizon_rule: bool

@dataclass(frozen=True)
class GrammarResidualCandidate:
    label: str
    value: float
    residual_minus_value: float
    abs_error: float
    matches: bool

@dataclass(frozen=True)
class QCDApertureCycleResidualProbe:
    """
    Test whether the conventional Lambda_QCD coordinate matches a closed
    aperture-cycle expression in the declared compact grammar.
    """
    lambda_qcd_gev: float
    n_qcd: float
    aperture_cycles: int
    electron_offset: int
    candidate_base: float
    residual: float
    candidates: tuple[GrammarResidualCandidate, ...]
    best_label: str
    best_abs_error: float
    closes_to_grammar: bool
    remains_external_input: bool

class LeptonD3CarrierPath(TypedDict):
    edges: tuple[tuple[int, int], ...]
    edge_labels: dict[tuple[int, int], list[str]]
    starts: tuple[int, ...]
    sinks: tuple[int, ...]
    path: tuple[int, ...]
    complete_directed_chain: bool

class LeptonCarrierPathProbe(TypedDict):
    q: int
    Tr_M: Fraction
    Tr_M2: Fraction
    carrier: Fraction
    byte_count: int

class LeptonD3SpectralObstruction(TypedDict):
    weights: tuple[ShellSpectralWeight, ...]
    q2_q4_full_hs_equal: bool
    q2_q4_return_trace_equal: bool
    q2_q4_shell_frobenius_equal: bool
    q5_over_q4_full_hs_ratio: Fraction
    q5_over_q4_byte_support_ratio: Fraction
    muon_electron_dyadic_ratio: Fraction
    hs_ratio_splits_muon_electron: bool

class LeptonD3PathBreakingProbe(TypedDict):
    selectors: tuple[K4PhaseSelector, ...]
    implemented_muon_electron_split: Fraction
    implemented_muon_electron_split_dyadic: Fraction
    simplified_electron_64_cost: Fraction
    simplified_muon_electron_split: Fraction
    simplified_muon_electron_split_dyadic: Fraction
    claim_25_over_64_matches_implemented: bool
    claim_25_over_64_matches_simplified: bool
    electron_reset_offset_from_simplified: Fraction
    carrier_path: LeptonD3CarrierPath
    history_path: tuple[int, ...]
    history_xor_moment: Fraction
    history_split_64: Fraction
    history_split_dyadic: Fraction
    history_split_matches_simplified: bool
    byte_interior_roles: Fraction
    byte_horizon: Fraction
    byte_phase_reset: Fraction
    byte_phase_reset_rule: str
    byte_phase_reset_64: Fraction
    byte_phase_reset_matches_rule: bool
    electron_from_history_64: Fraction
    electron_from_history_dyadic: Fraction
    electron_from_history_matches: bool
    carrier_conservation_coeff_sum: Fraction
    carrier_conservation_electron_dyadic: Fraction
    carrier_conservation_electron_64: Fraction
    carrier_conservation_current_offset: Fraction
    carrier_conservation_current_matches: bool

def lepton_ladder_residuals(observed: dict[str, float], delta: float=DELTA, v: float=E_EW_GEV) -> tuple[LeptonLadderRow, ...]:
    """
    Full order-by-order residual breakdown for the lepton temporal ladder.
    Returns one row per lepton (tau, muon, electron).
    """
    n_h_law = all_laws(delta, order=5)['Higgs'] / delta
    higgs_memory = KERNEL_BYTE_APERTURE / n_h_law
    specs = [('tau', 5, 'Tau mass energy', -(DELTA_BU + 5.0 * delta), CODE_C1 / 4.0 * delta ** 2, Fraction(-27, 64) * float(CARRIER_TRACES[4] - CARRIER_TRACES[5]) * delta ** 3, '-(dBU+5D)', '+(C1/4)D^2', '+(-27/64)*(C4-C5)*D^3'), ('muon', 8, 'Muon mass energy', MUON_EQUATOR_COEFF * delta, -(CODE_C1 / 16.0) * delta ** 2, Fraction(-37, 64) * float(CARRIER_TRACES[2] - CARRIER_TRACES[4]) * delta ** 3, '+C3*D', '-(C1/16)D^2', '+(-37/64)*(C2-C4)*D^3'), ('electron', 14, 'Electron mass energy', SU2_RESIDUAL + higgs_memory, 0.0, Fraction(-51, 256) * float(CARRIER_TRACES[2] - CARRIER_TRACES[4]) * delta ** 3, '+SU2+(5/256)/n_H', '', '+(-51/256)*(C2-C4)*D^3')]
    rows: list[LeptonLadderRow] = []
    for label, k, obs_name, d1, d2, d3, law1, law2, law3 in specs:
        obs_mass = observed.get(obs_name)
        if obs_mass is None:
            continue
        n_obs = ew_delta_n(obs_mass, ew_scale_gev=v, delta=delta)
        base = lepton_base_n(k, label)
        r = base - k * HORIZON_CARDINALITY
        n_d1 = base + d1
        n_d2 = n_d1 + d2
        n_d3 = n_d2 + d3
        rows.append(LeptonLadderRow(label=label, k=k, r=r, base_n=base, n_model_d1=n_d1, n_model_d2=n_d2, n_model_d3=n_d3, n_observed=n_obs, resid_d1=n_obs - n_d1, resid_d2=n_obs - n_d2, resid_d3=n_obs - n_d3, law_d1=law1, law_d2=law2, law_d3=law3))
    return tuple(rows)

def lepton_d3_transition_costs() -> tuple[LeptonD3TransitionCost, ...]:
    wz = CODE_C2 - CODE_C1
    specs = (('tau', 5, 4, Fraction(-27, 64), '-3*(C2-C1)', Fraction(-3 * wz, 1)), ('muon', 4, 2, Fraction(-37, 64), '-(|H|-3*(C2-C1))', Fraction(-(HORIZON_CARDINALITY - 3 * wz), 1)), ('electron', 4, 2, Fraction(-51, 256), '-(3*|K4|+C1/8)', -(Fraction(12, 1) + Fraction(CODE_C1, 8))))
    rows: list[LeptonD3TransitionCost] = []
    for label, q_from, q_to, dyadic, rule, rule_value in specs:
        carrier_from = CARRIER_TRACES[q_from]
        carrier_to = CARRIER_TRACES[q_to]
        carrier_delta = carrier_to - carrier_from
        normalized_64 = dyadic * 64
        rows.append(LeptonD3TransitionCost(label=label, q_from=q_from, q_to=q_to, carrier_from=carrier_from, carrier_to=carrier_to, carrier_delta=carrier_delta, dyadic=dyadic, coeff_delta3=dyadic * carrier_delta, normalized_64_cost=normalized_64, numerator_rule=rule, numerator_rule_value=rule_value, numerator_rule_matches=normalized_64 == rule_value))
    return tuple(rows)

def lepton_d3_selection_rule_probe() -> tuple[LeptonD3SelectionRule, ...]:
    """
    Mass-blind candidate rules for the three lepton D3 dyadics.

    These rules use only kernel constants and the temporal-ladder topology:
    |H|, C1, C2, C3, and WZ_OFFSET=C2-C1. They are not yet a proof that the
    lepton paths must select these rules; they make that selection problem
    precise and directly auditable.
    """
    wz = Fraction(WZ_OFFSET, 1)
    horizon = Fraction(HORIZON_CARDINALITY, 1)
    k4_cardinality = Fraction(4, 1)
    tau_cost = -Fraction(CODE_C1, 2) * wz
    muon_cost = -(horizon - abs(tau_cost))
    electron_cost = -(3 * k4_cardinality + Fraction(CODE_C1, 8))
    return (LeptonD3SelectionRule('tau', '-(C1/2)*(C2-C1)', tau_cost, Fraction(-27, 1)), LeptonD3SelectionRule('muon', '-(|H|-|(tau 64-cost)|)', muon_cost, Fraction(-37, 1)), LeptonD3SelectionRule('electron', '-(3*|K4|+C1/8)', electron_cost, Fraction(-51, 4)))

def _k4_phase_selector(label: str, base: bool, rot: bool, bal: bool, normalized_64_cost: Fraction, rule: str) -> K4PhaseSelector:
    p_float, q_float = _pq(base, rot, bal)
    return K4PhaseSelector(label=label, base=base, rot=rot, bal=bal, p=Fraction(p_float).limit_denominator(), q=Fraction(q_float).limit_denominator(), r5=Fraction(_r5(base, rot, bal)).limit_denominator(), normalized_64_cost=normalized_64_cost, rule=rule)

def lepton_d3_carrier_path() -> LeptonD3CarrierPath:
    """
    Derive the connected temporal carrier path from the D3 transition edges.

    The lepton D3 layer uses two distinct carrier edges:
      tau             q=5 -> q=4
      muon/electron   q=4 -> q=2

    With the empirical labels removed, these directed edges have one source
    endpoint and one sink endpoint, so they force the chain 5 -> 4 -> 2.
    This is the candidate canonical path used by the q-history moment below.
    """
    costs = lepton_d3_transition_costs()
    edge_labels: dict[tuple[int, int], list[str]] = {}
    for row in costs:
        edge_labels.setdefault((row.q_from, row.q_to), []).append(row.label)
    edges = tuple(edge_labels)
    sources = {src for src, _ in edges}
    targets = {dst for _, dst in edges}
    starts = sources - targets
    sinks = targets - sources
    path: list[int] = []
    complete = False
    if len(starts) == 1 and len(sinks) == 1:
        current = next(iter(starts))
        path.append(current)
        unused = set(edges)
        while unused:
            next_edges = [edge for edge in unused if edge[0] == current]
            if len(next_edges) != 1:
                break
            edge = next_edges[0]
            unused.remove(edge)
            current = edge[1]
            path.append(current)
        complete = not unused and current in sinks
    return {'edges': edges, 'edge_labels': edge_labels, 'starts': tuple(sorted(starts)), 'sinks': tuple(sorted(sinks)), 'path': tuple(path), 'complete_directed_chain': complete}

def lepton_d3_path_breaking_probe() -> LeptonD3PathBreakingProbe:
    """
    Candidate temporal path-breaking probe for the q=4 -> q=2 degeneracy.

    The K4 flags are tested as path labels, not as a proven transition operator.
    This records the important distinction between the simplified electron
    coefficient -3/16 and the implemented closure coefficient -51/256.
    """
    tau = _k4_phase_selector('tau memory', True, False, False, Fraction(-27, 1), '-(C1/2)*(C2-C1)')
    muon = _k4_phase_selector('muon hard path', True, True, False, Fraction(-37, 1), '-(|H|-|tau|)')
    electron = _k4_phase_selector('electron reset path', False, False, True, Fraction(-51, 4), '-(3*|K4|+C1/8)')
    simplified_electron_64 = Fraction(-12, 1)
    carrier_path = lepton_d3_carrier_path()
    history_path = carrier_path['path']
    history_xor_moment = q_history_xor_moment(history_path)
    history_split_64 = -(history_xor_moment * WZ_OFFSET)
    byte_interior_roles = Fraction(CODE_C1, 2)
    byte_phase_reset = -byte_interior_roles / 256
    byte_phase_reset_64 = byte_phase_reset * 64
    electron_from_history_64 = muon.normalized_64_cost - history_split_64 + byte_phase_reset_64
    tau_coeff = Fraction(-27, 64) * (CARRIER_TRACES[4] - CARRIER_TRACES[5])
    muon_coeff = Fraction(-37, 64) * (CARRIER_TRACES[2] - CARRIER_TRACES[4])
    electron_coeff = Fraction(-51, 256) * (CARRIER_TRACES[2] - CARRIER_TRACES[4])
    carrier_delta_muon_electron = CARRIER_TRACES[2] - CARRIER_TRACES[4]
    conservation_electron_dyadic = -(tau_coeff + muon_coeff) / carrier_delta_muon_electron
    conservation_electron_64 = conservation_electron_dyadic * 64
    conservation_sum_coeff = tau_coeff + muon_coeff + electron_coeff
    return {'selectors': (tau, muon, electron), 'implemented_muon_electron_split': muon.normalized_64_cost - electron.normalized_64_cost, 'implemented_muon_electron_split_dyadic': Fraction(-37, 64) - Fraction(-51, 256), 'simplified_electron_64_cost': simplified_electron_64, 'simplified_muon_electron_split': muon.normalized_64_cost - simplified_electron_64, 'simplified_muon_electron_split_dyadic': Fraction(-37, 64) - Fraction(-3, 16), 'claim_25_over_64_matches_implemented': Fraction(-37, 64) - Fraction(-51, 256) == Fraction(-25, 64), 'claim_25_over_64_matches_simplified': Fraction(-37, 64) - Fraction(-3, 16) == Fraction(-25, 64), 'electron_reset_offset_from_simplified': Fraction(-51, 256) - Fraction(-3, 16), 'carrier_path': carrier_path, 'history_path': history_path, 'history_xor_moment': history_xor_moment, 'history_split_64': history_split_64, 'history_split_dyadic': history_split_64 / 64, 'history_split_matches_simplified': history_split_64 == Fraction(-25, 1), 'byte_interior_roles': byte_interior_roles, 'byte_horizon': Fraction(256, 1), 'byte_phase_reset': byte_phase_reset, 'byte_phase_reset_rule': '-(C1/2)/256', 'byte_phase_reset_64': byte_phase_reset_64, 'byte_phase_reset_matches_rule': byte_phase_reset == Fraction(-3, 256), 'electron_from_history_64': electron_from_history_64, 'electron_from_history_dyadic': electron_from_history_64 / 64, 'electron_from_history_matches': electron_from_history_64 == electron.normalized_64_cost, 'carrier_conservation_coeff_sum': conservation_sum_coeff, 'carrier_conservation_electron_dyadic': conservation_electron_dyadic, 'carrier_conservation_electron_64': conservation_electron_64, 'carrier_conservation_current_offset': Fraction(-51, 256) - conservation_electron_dyadic, 'carrier_conservation_current_matches': Fraction(-51, 256) == conservation_electron_dyadic}

def q_support_words(q: int) -> tuple[int, ...]:
    return tuple((word for word in range(64) if word.bit_count() == q))

def q_history_xor_moment(q_path: tuple[int, ...]) -> Fraction:
    supports = tuple((q_support_words(q) for q in q_path))
    total = 1
    for support in supports:
        total *= len(support)
    weighted_sum = 0

    def visit(index: int, acc: int) -> None:
        nonlocal weighted_sum
        if index == len(supports):
            weighted_sum += acc.bit_count()
            return
        for word in supports[index]:
            visit(index + 1, acc ^ word)
    visit(0, 0)
    return Fraction(weighted_sum, total)

def q_support_duality_probe() -> tuple[QSupportDuality, ...]:
    """
    Exact q-support relations inside the 6-bit chirality register.

    q=2 and q=4 have no direct set intersection, but they are perfectly paired
    by complement x -> x XOR 0x3f. This is the finite-algebra source of their
    isospectrality; a generation split therefore needs path/phase information
    beyond static q-support volume.
    """
    rows: list[QSupportDuality] = []
    for q_a, q_b in ((2, 4), (5, 4)):
        a = set(q_support_words(q_a))
        b = set(q_support_words(q_b))
        rows.append(QSupportDuality(q_a=q_a, q_b=q_b, size_a=len(a), size_b=len(b), direct_intersection=len(a & b), complement_pair_hits=sum((1 for word in a if word ^ 63 in b))))
    return tuple(rows)

def byte_archetype_shadow_probe() -> ByteArchetypeProbe:
    """
    Exact byte-horizon audit for the final lepton carrier offset.

    In the hQVM byte formalism, transcription is intron = byte XOR 0xAA.
    Hence 0xAA is the unique zero-intron source. The implemented electron
    D3 coefficient differs from exact carrier neutrality by -1/256, and
    multiplying that single byte atom by the shared q=4 -> q=2 carrier delta
    gives the residual carrier coefficient exactly.
    """
    gene_mic = 170
    intron = gene_mic ^ 170
    family = (intron >> 7 & 1) << 1 | intron & 1
    micro_ref = intron >> 1 & 63
    q_weight = 0
    archetype_shadow = Fraction(1, 256)
    carrier_delta = CARRIER_TRACES[2] - CARRIER_TRACES[4]
    shadow_carrier_coeff = -archetype_shadow * carrier_delta
    residual_carrier_sum = lepton_d3_path_breaking_probe()['carrier_conservation_coeff_sum']
    return ByteArchetypeProbe(gene_mic=gene_mic, intron=intron, family=family, micro_ref=micro_ref, q_weight=q_weight, archetype_shadow=archetype_shadow, carrier_delta=carrier_delta, shadow_carrier_coeff=shadow_carrier_coeff, residual_carrier_sum=residual_carrier_sum, matches_residual=shadow_carrier_coeff == residual_carrier_sum)

def horizon_gate_selection_probe() -> tuple[HorizonGateSelection, ...]:
    """
    Selection audit over the four q-kernel horizon gate bytes.

    The electron reset is tested against the source-traceability criterion:
    it must be an S-gate with no payload mutation and no nontrivial family
    phase. Among q^{-1}(0) = {0xAA, 0x54, 0xD5, 0x2B}, only 0xAA satisfies
    all three conditions.
    """
    gates = ((170, 'S'), (84, 'S'), (213, 'C'), (43, 'C'))
    rows: list[HorizonGateSelection] = []
    for byte, gate in gates:
        intron = byte ^ 170
        family = (intron >> 7 & 1) << 1 | intron & 1
        micro_ref = intron >> 1 & 63
        q_weight = 0
        is_s_gate = gate == 'S'
        zero_intron = intron == 0
        zero_payload = micro_ref == 0
        zero_family = family == 0
        rows.append(HorizonGateSelection(byte=byte, gate=gate, intron=intron, family=family, micro_ref=micro_ref, q_weight=q_weight, is_s_gate=is_s_gate, zero_intron=zero_intron, zero_payload=zero_payload, zero_family=zero_family, selected_by_electron_reset=is_s_gate and zero_intron and zero_payload and zero_family))
    return tuple(rows)

def source_traceability_probe() -> SourceTraceabilityProbe:
    """
    Verify the finite source-traceability rule used by the terminal electron
    branch and its dyadic consequence.
    """
    selected = tuple((row for row in horizon_gate_selection_probe() if row.selected_by_electron_reset))
    carrier_delta = CARRIER_TRACES[2] - CARRIER_TRACES[4]
    dyadic_atom = Fraction(1, 256)
    electron_neutral = Fraction(-25, 128)
    electron_implemented = Fraction(-51, 256)
    electron_derived = electron_neutral - dyadic_atom
    return SourceTraceabilityProbe(selected_byte=selected[0].byte if selected else -1, selected_count=len(selected), q_kernel_size=len(horizon_gate_selection_probe()), selected_is_gene_mic=len(selected) == 1 and selected[0].byte == 170, dyadic_atom=dyadic_atom, carrier_delta=carrier_delta, carrier_shadow=-dyadic_atom * carrier_delta, electron_neutral=electron_neutral, electron_implemented=electron_implemented, electron_derived=electron_derived, closes_electron_dyadic=electron_derived == electron_implemented)

def lepton_horizon_wrap_exhaustion_probe(max_k: int=16) -> LeptonHorizonWrapExhaustionProbe:
    """
    Exhaustively test the current lepton anchor conditions.

    Broad conditions:
      - k_tau < k_mu < k_e <= max_k
      - the tau/muon 64-normalised D3 budget closes to |H|

    Horizon-wrap candidate rule:
      - k_tau equals the source q of the carrier path, q=5
      - k_mu - k_tau = C1/2
      - k_e - k_mu = C1

    The broad conditions intentionally do not force a unique path. If only
    the horizon-wrap rule gives uniqueness, the probe identifies that rule as
    the missing theorem target.
    """
    q_path = (5, 4, 2)
    expected = (5, 8, 14)
    costs = {row.label: row for row in lepton_d3_transition_costs()}
    tau_mu_budget = abs(costs['tau'].normalized_64_cost) + abs(costs['muon'].normalized_64_cost)
    broad: list[tuple[int, int, int]] = []
    horizon_rule: list[tuple[int, int, int]] = []
    expected_delta_tau_mu = Fraction(CODE_C1, 2)
    expected_delta_mu_e = Fraction(CODE_C1, 1)
    for k_tau in range(max_k + 1):
        for k_mu in range(k_tau + 1, max_k + 1):
            for k_e in range(k_mu + 1, max_k + 1):
                seq = (k_tau, k_mu, k_e)
                if tau_mu_budget == HORIZON_CARDINALITY:
                    broad.append(seq)
                if k_tau == q_path[0] and Fraction(k_mu - k_tau, 1) == expected_delta_tau_mu and (Fraction(k_e - k_mu, 1) == expected_delta_mu_e):
                    horizon_rule.append(seq)
    return LeptonHorizonWrapExhaustionProbe(max_k=max_k, q_path=q_path, expected_k_path=expected, broad_valid_count=len(broad), broad_valid_examples=tuple(broad[:8]), horizon_rule_valid_count=len(horizon_rule), horizon_rule_valid_paths=tuple(horizon_rule), tau_muon_budget=tau_mu_budget, tau_muon_budget_matches=tau_mu_budget == HORIZON_CARDINALITY, delta_tau_muon=expected[1] - expected[0], delta_muon_electron=expected[2] - expected[1], expected_delta_tau_muon=expected_delta_tau_mu, expected_delta_muon_electron=expected_delta_mu_e, unique_without_horizon_rule=len(broad) == 1 and broad[0] == expected, unique_with_horizon_rule=len(horizon_rule) == 1 and horizon_rule[0] == expected)

def shell_spectral_weights() -> tuple[ShellSpectralWeight, ...]:
    """
    Exact Hilbert-Schmidt/Frobenius diagnostics for the q-weight operators.

    shell_frobenius_sq is computed from the 7x7 shell transition matrix.
    full_operator_hs_sq is the exact Frobenius norm squared of the normalized
    q-step Hamming operator on the full 64-point chirality register:
        ||T_q||_F^2 = 64 / C(6,q)
    because each row has C(6,q) entries of weight 1/C(6,q).
    """
    rows: list[ShellSpectralWeight] = []
    for q in range(7):
        M = shell_transition_matrix(q)
        shell_frob = sum((M[i][j] * M[i][j] for i in range(7) for j in range(7)), Fraction(0))
        rows.append(ShellSpectralWeight(q=q, byte_count=4 * comb(6, q), trace=shell_trace(q), return_trace=shell_return_trace(q), carrier=carrier_trace(q), shell_frobenius_sq=shell_frob, full_operator_hs_sq=Fraction(64, comb(6, q))))
    return tuple(rows)

def lepton_d3_spectral_obstruction() -> LeptonD3SpectralObstruction:
    """
    Test whether spectral weights alone force the lepton D3 dyadics.

    Result: q=2 and q=4 are isospectral at the full Hamming-operator level,
    so a rule depending only on q-shell Hilbert-Schmidt weights cannot split
    muon and electron, which share q=4 -> q=2 but require different dyadics.
    """
    weights = {row.q: row for row in shell_spectral_weights()}
    q2 = weights[2]
    q4 = weights[4]
    q5 = weights[5]
    dyadic_ratio = Fraction(-37, 64) / Fraction(-51, 256)
    hs_ratio_q5_q4 = q5.full_operator_hs_sq / q4.full_operator_hs_sq
    return {'weights': tuple((weights[q] for q in range(7))), 'q2_q4_full_hs_equal': q2.full_operator_hs_sq == q4.full_operator_hs_sq, 'q2_q4_return_trace_equal': q2.return_trace == q4.return_trace, 'q2_q4_shell_frobenius_equal': q2.shell_frobenius_sq == q4.shell_frobenius_sq, 'q5_over_q4_full_hs_ratio': hs_ratio_q5_q4, 'q5_over_q4_byte_support_ratio': Fraction(q4.byte_count, q5.byte_count), 'muon_electron_dyadic_ratio': dyadic_ratio, 'hs_ratio_splits_muon_electron': dyadic_ratio == Fraction(1, 1)}

def quark_delta_coords(delta: float=DELTA) -> dict[str, float]:
    """
    Quark EW mass-coordinate gaps.
    Selectors sample all four vertices of the {kappa, omega} Boolean lattice.

    PDG uncertainty floor (~1% b/c, ~5% s) maps to ~0.7 tick uncertainty;
    these are compact selectors, not scale-independent mass predictions.
    """
    alg = compact_algebra(delta)
    return {'n_bottom': 284.0 + alg.kappa, 'n_charm': 367.0 + alg.omega + 0.5 * delta, 'n_strange': 548.0 - (alg.omega + alg.kappa)}

def strong_scale_bulk_probe(lambda_qcd_gev: float=0.2, delta: float=DELTA, ew_scale_gev: float=E_EW_GEV) -> StrongScaleBulkProbe:
    """
    Place the conventional strong scale on the mass-coordinate axis and audit whether
    the corresponding finite-geometric interpretation is bulk, not horizon.
    """
    n_qcd = math.log2(ew_scale_gev / lambda_qcd_gev) / delta
    boundary_states = 2 * HORIZON_CARDINALITY
    bulk_states = OMEGA_SIZE - boundary_states
    return StrongScaleBulkProbe(lambda_qcd_gev=lambda_qcd_gev, n_qcd=n_qcd, n_phase_64=n_qcd % HORIZON_CARDINALITY, n_phase_48=n_qcd % APERTURE_FRAME, bulk_states=bulk_states, boundary_states=boundary_states, bulk_fraction=Fraction(bulk_states, OMEGA_SIZE), boundary_fraction=Fraction(boundary_states, OMEGA_SIZE), placed_in_relational_bulk=bulk_states > boundary_states)

def qcd_aperture_cycle_residual_probe(lambda_qcd_gev: float=0.2, delta: float=DELTA, ew_scale_gev: float=E_EW_GEV, tolerance: float=1e-12) -> QCDApertureCycleResidualProbe:
    """
    Test the proposed 10*APERTURE_FRAME + r(electron) QCD coordinate.

    This is a grammar probe only. Lambda_QCD remains an external input unless
    the residual closes to an exact compact expression at the declared
    tolerance.
    """
    n_qcd = math.log2(ew_scale_gev / lambda_qcd_gev) / delta
    electron_offset = int(M_SHELL / 8 - M_SHELL / 24)
    base = 10.0 * APERTURE_FRAME + electron_offset
    residual = base - n_qcd
    alg = compact_algebra(delta)
    grammar = (('3*Delta', 3.0 * delta), ('epsilon-eta', alg.epsilon - alg.eta), ('epsilon/5', alg.epsilon / 5.0), ('Delta/5*APERTURE_FRAME', delta * APERTURE_FRAME / 5.0), ('3*Delta-eta/2', 3.0 * delta - alg.eta / 2.0), ('3*Delta-eta/2+Delta^2/8', 3.0 * delta - alg.eta / 2.0 + delta ** 2 / 8.0), ('Delta/APERTURE_FRAME', delta / APERTURE_FRAME))
    rows = tuple((GrammarResidualCandidate(label=label, value=value, residual_minus_value=residual - value, abs_error=abs(residual - value), matches=abs(residual - value) <= tolerance) for label, value in grammar))
    best = min(rows, key=lambda row: row.abs_error)
    closes = any((row.matches for row in rows))
    return QCDApertureCycleResidualProbe(lambda_qcd_gev=lambda_qcd_gev, n_qcd=n_qcd, aperture_cycles=10, electron_offset=electron_offset, candidate_base=base, residual=residual, candidates=rows, best_label=best.label, best_abs_error=best.abs_error, closes_to_grammar=closes, remains_external_input=not closes)

def ew_delta_n(
    observable_mass_gev: float, ew_scale_gev: float = E_EW_GEV, delta: float = DELTA
) -> float:
    """Mass-coordinate gap n = log2(v/m) / delta."""
    return math.log2(ew_scale_gev / observable_mass_gev) / delta

@dataclass(frozen=True)
class ElectroweakCoords:
    """EW mass-coordinate gaps n = L_i(D)/Delta at a given law order."""
    order: int
    delta: float
    n_top: float
    n_higgs: float
    n_z: float
    n_w: float
    n_tau: float
    n_mu: float
    n_electron: float
    n_bottom: float
    n_charm: float
    n_strange: float

def electroweak_coords(delta: float=DELTA, *, order: int=5) -> ElectroweakCoords:
    """Compute all EW mass-coordinate gaps at the requested law order."""
    laws = all_laws(delta, order=order)
    ch_top = channel_by_label('Top')
    ch_h = channel_by_label('Higgs')
    n_top = laws['Top'] / delta
    n_higgs = laws['Higgs'] / delta
    n_z = laws['Z'] / delta
    n_w = laws['W'] / delta
    lep = lepton_delta_coords(n_higgs, delta)
    qrk = quark_delta_coords(delta)
    return ElectroweakCoords(order=order, delta=delta, n_top=n_top, n_higgs=n_higgs, n_z=n_z, n_w=n_w, n_tau=lep['n_tau'], n_mu=lep['n_mu'], n_electron=lep['n_electron'], n_bottom=qrk['n_bottom'], n_charm=qrk['n_charm'], n_strange=qrk['n_strange'])

@dataclass(frozen=True)
class EWCouplings:
    lambda_H: float
    g: float
    g_z: float
    g_prime: float
    e: float
    alpha_ew_inv: float
    y_t: float

def ew_couplings_from_masses(m_t: float, m_h: float, m_z: float, m_w: float, v: float=E_EW_GEV) -> EWCouplings:
    """Tree-level couplings from mass inputs (GeV)."""
    lambda_h = m_h ** 2 / (2.0 * v ** 2)
    g = 2.0 * m_w / v
    g_z = 2.0 * m_z / v
    g_prime = math.sqrt(max(g_z ** 2 - g ** 2, 0.0))
    e_charge = g * g_prime / g_z if g_z != 0.0 else float('nan')
    alpha_inv = 4.0 * math.pi / e_charge ** 2 if e_charge != 0.0 else float('inf')
    y_t = math.sqrt(2.0) * m_t / v
    return EWCouplings(lambda_H=lambda_h, g=g, g_z=g_z, g_prime=g_prime, e=e_charge, alpha_ew_inv=alpha_inv, y_t=y_t)

def model_masses(delta: float=DELTA, v: float=E_EW_GEV) -> dict[str, float]:
    """Masses predicted by the Omega path-certified five-order law."""
    return model_masses_from_path(delta, v)

def model_couplings_d2(delta: float=DELTA, v: float=E_EW_GEV) -> EWCouplings:
    """Couplings from D2-order model masses (law without gyroscopic corrections)."""
    laws = all_laws(delta, order=2)
    m_t = v * 2.0 ** (-laws['Top'])
    m_h = v * 2.0 ** (-laws['Higgs'])
    m_z = v * 2.0 ** (-laws['Z'])
    m_w = v * 2.0 ** (-laws['W'])
    return ew_couplings_from_masses(m_t, m_h, m_z, m_w, v)

def wz_split(delta: float, *, promoted: bool=True) -> float:
    """
    Charged-neutral split S_WZ such that log2(mZ/mW) = delta * S_WZ.

    promoted=False: D2 backbone only:   (C2-C1) - (C3/2)*D
    promoted=True:  full D4 law:        + 2D^2/sqrt(5) - D^3
    """
    base = WZ_OFFSET - WZ_APERTURE_COEFF * delta
    if not promoted:
        return base
    return base + 2.0 * delta ** 2 / math.sqrt(5.0) - delta ** 3

def sin2_theta_w(delta: float, *, promoted: bool=True) -> float:
    """sin^2(theta_W) from compact promoted split law."""
    split = wz_split(delta, promoted=promoted)
    cos_w = 2.0 ** (-delta * split)
    return 1.0 - cos_w ** 2

def w_mass_from_z(m_z: float, delta: float) -> float:
    """Predict m_W from m_Z and delta using promoted split law."""
    split = wz_split(delta, promoted=True)
    return m_z / 2.0 ** (delta * split)

def ckm_ansatz(delta: float=DELTA) -> dict[str, float]:
    """
    Compact CKM angle ansatz (empirical selector).
    V_us ~ sin(delta_BU + 3D/2)
    V_cb ~ sin(2D)
    V_ub exclusive ~ sin(9D^2)
    V_ub inclusive ~ sin(9D^2 + phi_conv)  where phi_conv = eta - sin(9D^2)
    """
    alg = compact_algebra(delta)
    v_us = math.sin(DELTA_BU + 1.5 * delta)
    v_cb = math.sin(2.0 * delta)
    excl = math.sin(9.0 * delta ** 2)
    phi_conv = alg.eta - excl
    v_ub_incl = math.sin(9.0 * delta ** 2 + phi_conv)
    return {'V_us': v_us, 'V_cb': v_cb, 'V_ub_excl': excl, 'V_ub_incl': v_ub_incl, 'phi_conv': phi_conv, 'delta_CKM': math.degrees(math.pi / 2.0 - 18.0 * delta)}

@dataclass(frozen=True)
class LeaveOneOutResult:
    target: str
    delta_source: str
    delta_used: float
    predicted_mass: float
    reference_mass: float

    @property
    def relative_error(self) -> float:
        return self.predicted_mass / self.reference_mass - 1.0

def _law_for(label: str, delta: float) -> float:
    return eval_law(channel_by_label(label), delta, order=2)

def hzw_leave_one_out(observed: dict[str, float], backsolves: tuple[DeltaBacksolve, ...]) -> tuple[LeaveOneOutResult, ...]:
    """
    For each of H, Z, W: predict mass using Delta averaged from the other two.
    """
    v = observed['Electroweak scale']
    bs = {b.source: b.delta_back for b in backsolves}
    results: list[LeaveOneOutResult] = []
    for target in ('Higgs', 'Z', 'W'):
        others = [s for s in ('Higgs', 'Z', 'W') if s != target]
        delta_used = sum((bs[s] for s in others)) / 2.0
        log_gap = _law_for(target, delta_used)
        pred = v * 2.0 ** (-log_gap)
        ref = observed[channel_by_label(target).observable]
        results.append(LeaveOneOutResult(target=target, delta_source='+'.join(others), delta_used=delta_used, predicted_mass=pred, reference_mass=ref))
    return tuple(results)
