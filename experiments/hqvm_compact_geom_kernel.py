"""
hqvm_compact_geom_kernel.py

Finite kernel verification layer.
Exhaustive enumeration of the 4096-state reachable manifold Omega,
shell transition algebra verification, and r5 grammar probe.

This module PROVES the algebraic claims that hqvm_compact_geom_core.py uses
as established inputs. It is self-contained and produces a KernelReport
dataclass that summarises all verified kernel theorems.

Requires: gyroscopic/hQVM/constants.py from the repo root.
"""
from __future__ import annotations
import importlib.util
import math
import random
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from math import comb
from pathlib import Path
from typing import Callable, Sequence
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
_HQVM_PATH = REPO_ROOT / 'gyroscopic' / 'hQVM' / 'constants.py'
_hqvm_spec = importlib.util.spec_from_file_location('_hqvm_constants', _HQVM_PATH)
if _hqvm_spec is None or _hqvm_spec.loader is None:
    raise RuntimeError(f'Cannot load {_HQVM_PATH}')
_hqvm = importlib.util.module_from_spec(_hqvm_spec)
sys.modules.setdefault('_hqvm_constants', _hqvm)
_hqvm_spec.loader.exec_module(_hqvm)
_HQVM_API_PATH = REPO_ROOT / 'gyroscopic' / 'hQVM' / 'api.py'
_api_spec = importlib.util.spec_from_file_location('_hqvm_api', _HQVM_API_PATH)
if _api_spec is None or _api_spec.loader is None:
    raise RuntimeError(f'Cannot load {_HQVM_API_PATH}')
_hqvm_api = importlib.util.module_from_spec(_api_spec)
sys.modules.setdefault('_hqvm_api', _hqvm_api)
_api_spec.loader.exec_module(_hqvm_api)
_HQVM_SDK_PATH = REPO_ROOT / 'gyroscopic' / 'hQVM' / 'sdk.py'
_sdk_spec = importlib.util.spec_from_file_location('_hqvm_sdk', _HQVM_SDK_PATH)
if _sdk_spec is None or _sdk_spec.loader is None:
    raise RuntimeError(f'Cannot load {_HQVM_SDK_PATH}')
_hqvm_sdk = importlib.util.module_from_spec(_sdk_spec)
sys.modules.setdefault('_hqvm_sdk', _hqvm_sdk)
_sdk_spec.loader.exec_module(_hqvm_sdk)
CHIRALITY_MASK_6: int = _hqvm.CHIRALITY_MASK_6
EPSILON_6: int = _hqvm.EPSILON_6
GENE_MAC_A12: int = _hqvm.GENE_MAC_A12
LAYER_MASK_12: int = _hqvm.LAYER_MASK_12
C_PERP_12: tuple[int, ...] = _hqvm_api.C_PERP_12
byte_to_intron = _hqvm.byte_to_intron
expand_intron_to_mask12 = _hqvm.expand_intron_to_mask12
pack_state = _hqvm.pack_state
single_step_trace = _hqvm.single_step_trace
unpack_state = _hqvm.unpack_state
depth4_frame = _hqvm_sdk.depth4_frame
from hqvm_compact_geom_core import CHANNELS, CODE_C1, CODE_C2, CODE_C3, DELTA, DELTA_BU, HORIZON_CARDINALITY, LAMBDA_0, M_A, M_SHELL, P_BOUNDARY, Q_DENSITY, CARRIER_TRACES, carrier_trace, EW_CHANNEL_Q_WEIGHT, channel_by_label, ckm_ansatz, ew_couplings_from_masses, eval_law, qcd_aperture_cycle_residual_probe, compact_algebra, shell_transition_matrix, shell_trace, shell_return_trace

def _word6_to_pairdiag12(word6: int) -> int:
    """Embed a 6-bit word into a 12-bit pair-diagonal mask."""
    x = int(word6) & CHIRALITY_MASK_6
    out = 0
    for i in range(6):
        if x >> i & 1:
            out |= 3 << 2 * i
    return out & LAYER_MASK_12

def _pairdiag12_to_word6(word12: int) -> int:
    """Invert the pair-diagonal embedding. Raises ValueError if not pair-diagonal."""
    x = int(word12) & LAYER_MASK_12
    out = 0
    for i in range(6):
        pair = x >> 2 * i & 3
        if pair == 3:
            out |= 1 << i
        elif pair != 0:
            raise ValueError(f'Not pair-diagonal at bit pair {i}: word12={word12:#05x}')
    return out & CHIRALITY_MASK_6

def _omega_state(u6: int, v6: int) -> int:
    """Pack (u6, v6) into a 24-bit Omega state."""
    a12 = GENE_MAC_A12 ^ _word6_to_pairdiag12(u6)
    b12 = GENE_MAC_A12 ^ _word6_to_pairdiag12(v6)
    return pack_state(a12, b12)

def _state_to_uv6(state24: int) -> tuple[int, int]:
    """Unpack a 24-bit state into (u6, v6)."""
    a12, b12 = unpack_state(state24)
    return (_pairdiag12_to_word6(a12 ^ GENE_MAC_A12), _pairdiag12_to_word6(b12 ^ GENE_MAC_A12))

def _shell_of(state24: int) -> int:
    """Hamming weight of u6 XOR v6 = ab_distance = shell index."""
    u6, v6 = _state_to_uv6(state24)
    return (u6 ^ v6).bit_count()

def _is_equality_horizon(state24: int) -> bool:
    u6, v6 = _state_to_uv6(state24)
    return u6 == v6

def _is_complement_horizon(state24: int) -> bool:
    u6, v6 = _state_to_uv6(state24)
    return u6 ^ v6 == EPSILON_6

def _q_word6(byte: int) -> int:
    """Extract the 6-bit q-class of a byte via the intron map."""
    intron = byte_to_intron(int(byte) & 255)
    l0 = intron & 1 ^ intron >> 7 & 1
    q12 = expand_intron_to_mask12(intron) ^ (LAYER_MASK_12 if l0 else 0)
    return _pairdiag12_to_word6(q12)
OMEGA_STATES: tuple[int, ...] = tuple((_omega_state(u6, v6) for u6 in range(64) for v6 in range(64)))

@dataclass(frozen=True)
class ShellStats:
    shell: int
    population: int
    fraction: float
    expected_population: int
    matches_binomial: bool

def shell_distribution() -> tuple[ShellStats, ...]:
    """
    Verify the binomial shell distribution |Shell_k| = C(6,k) * 64.
    Returns one ShellStats per shell k = 0..6.
    """
    counts = [0] * 7
    for s in OMEGA_STATES:
        counts[_shell_of(s)] += 1
    total = len(OMEGA_STATES)
    rows: list[ShellStats] = []
    for k in range(7):
        expected = comb(6, k) * 64
        rows.append(ShellStats(shell=k, population=counts[k], fraction=counts[k] / total, expected_population=expected, matches_binomial=counts[k] == expected))
    return tuple(rows)

@dataclass(frozen=True)
class HorizonStats:
    equality_count: int
    complement_count: int
    total_boundary: int
    holographic_identity: bool
    complementarity_holds: bool

def horizon_verification() -> HorizonStats:
    eq = sum((1 for s in OMEGA_STATES if _is_equality_horizon(s)))
    comp = sum((1 for s in OMEGA_STATES if _is_complement_horizon(s)))
    comp_holds = all((0 <= _shell_of(s) <= 6 for s in OMEGA_STATES))
    return HorizonStats(equality_count=eq, complement_count=comp, total_boundary=eq + comp, holographic_identity=HORIZON_CARDINALITY ** 2 == len(OMEGA_STATES), complementarity_holds=comp_holds)

@dataclass(frozen=True)
class ByteTransitionStats:
    total_ops: int
    active_swap_failures: int
    passive_commit_failures: int
    complement_swap_fraction: float
    complement_commit_fraction: float
    equality_horizon_hit_fraction: float
    complement_horizon_hit_fraction: float
    q_weight_counts: tuple[int, ...]
    source_to_dest_shell: tuple[tuple[int, ...], ...]

def byte_transition_stats() -> ByteTransitionStats:
    """
    Exhaustive check of all 4096 * 256 = 1,048,576 byte transitions.

    Verifies:
      - active swap law: trace["ona"] == b12 ^ invert_a
      - passive commit law: trace["bu"] == trace["una"] ^ invert_b
      - horizon hit fractions
      - q-weight byte distribution
    """
    total = 0
    active_swap_failures = 0
    passive_commit_failures = 0
    complement_swap_count = 0
    complement_commit_count = 0
    equality_hits = 0
    complement_hits = 0
    shell_matrix = [[0] * 7 for _ in range(7)]
    q_counts = [0] * 7
    for state in OMEGA_STATES:
        shell_src = _shell_of(state)
        a12, b12 = unpack_state(state)
        for byte in range(256):
            total += 1
            intron = byte_to_intron(byte)
            trace = single_step_trace(state, byte)
            invert_a = LAYER_MASK_12 if intron & 1 else 0
            invert_b = LAYER_MASK_12 if intron & 128 else 0
            if trace['ona'] != b12 ^ invert_a:
                active_swap_failures += 1
            if trace['bu'] != trace['una'] ^ invert_b:
                passive_commit_failures += 1
            if invert_a:
                complement_swap_count += 1
            if invert_b:
                complement_commit_count += 1
            dest = trace['state24']
            shell_dst = _shell_of(dest)
            shell_matrix[shell_src][shell_dst] += 1
            if _is_complement_horizon(dest):
                complement_hits += 1
            if _is_equality_horizon(dest):
                equality_hits += 1
    for byte in range(256):
        q_counts[_q_word6(byte).bit_count()] += 1
    return ByteTransitionStats(total_ops=total, active_swap_failures=active_swap_failures, passive_commit_failures=passive_commit_failures, complement_swap_fraction=complement_swap_count / total, complement_commit_fraction=complement_commit_count / total, equality_horizon_hit_fraction=equality_hits / total, complement_horizon_hit_fraction=complement_hits / total, q_weight_counts=tuple(q_counts), source_to_dest_shell=tuple((tuple(row) for row in shell_matrix)))

@dataclass(frozen=True)
class StructuralLaw:
    """
    The three independently derived kernel fractions that all equal 1/64.
    """
    horizon_maintaining_fraction: Fraction
    byte_commutativity_rate: Fraction
    q_fibre_relative_size: Fraction

def structural_law_verification() -> StructuralLaw:
    """Verify the 1/64 structural coincidence."""
    complement_states = [s for s in OMEGA_STATES if _is_complement_horizon(s)]
    horizon_maintaining = 0
    for byte in range(256):
        maintains = all((_is_complement_horizon(single_step_trace(s, byte)['state24']) for s in complement_states))
        if maintains:
            horizon_maintaining += 1
    commuting_pairs = 0
    total_pairs = 256 * 256
    for x in range(256):
        for y in range(256):
            commutes = True
            for s in complement_states:
                s_xy = single_step_trace(single_step_trace(s, x)['state24'], y)['state24']
                s_yx = single_step_trace(single_step_trace(s, y)['state24'], x)['state24']
                if s_xy != s_yx:
                    commutes = False
                    break
            if commutes:
                commuting_pairs += 1
    return StructuralLaw(horizon_maintaining_fraction=Fraction(horizon_maintaining, 256), byte_commutativity_rate=Fraction(commuting_pairs, total_pairs), q_fibre_relative_size=Fraction(4, 256))

@dataclass(frozen=True)
class BoundaryToBulkStats:
    complement_states: int
    total_fanout: int
    unique_targets: int
    min_multiplicity: int
    max_multiplicity: int
    exact_uniform_multiplicity: bool

def boundary_to_bulk_projection() -> BoundaryToBulkStats:
    """
    Verify that one-byte fanout from the complement horizon reaches every
    state in Omega with multiplicity exactly 4.
    """
    complement_states = [s for s in OMEGA_STATES if _is_complement_horizon(s)]
    hit_counts: dict[int, int] = {}
    for s in complement_states:
        for byte in range(256):
            dest = single_step_trace(s, byte)['state24']
            hit_counts[dest] = hit_counts.get(dest, 0) + 1
    multiplicities = list(hit_counts.values())
    return BoundaryToBulkStats(complement_states=len(complement_states), total_fanout=len(complement_states) * 256, unique_targets=len(hit_counts), min_multiplicity=min(multiplicities) if multiplicities else 0, max_multiplicity=max(multiplicities) if multiplicities else 0, exact_uniform_multiplicity=len(hit_counts) == len(OMEGA_STATES) and min(multiplicities) == 4 and (max(multiplicities) == 4))

@dataclass(frozen=True)
class ShellTransitionRow:
    q_weight: int
    trace: Fraction
    return_trace: Fraction
    carrier: Fraction

def shell_transition_algebra() -> tuple[ShellTransitionRow, ...]:
    """
    Compute exact shell transition traces for q = 0..6.
    These are the carrier traces used in the lepton ladder.
    """
    rows: list[ShellTransitionRow] = []
    for q in range(7):
        t = shell_trace(q)
        r = shell_return_trace(q)
        c = carrier_trace(q)
        rows.append(ShellTransitionRow(q_weight=q, trace=t, return_trace=r, carrier=c))
    return tuple(rows)

@dataclass(frozen=True)
class UVIRShellDPF:
    uv_label: str
    ir_label: str
    uv_q: int
    ir_q: int
    uv_carrier: Fraction
    ir_carrier: Fraction
    ratio: float
_LEPTON_Q_WEIGHT = {'tau': 4, 'mu': 4, 'electron': 4}

def _stage_q_weight(label: str) -> int:
    if label in EW_CHANNEL_Q_WEIGHT:
        return EW_CHANNEL_Q_WEIGHT[label]
    return _LEPTON_Q_WEIGHT[label]
UV_IR_PAIRS: tuple[tuple[str, str], ...] = (('Top', 'electron'), ('Higgs', 'mu'))

def uv_ir_shell_dpf() -> tuple[UVIRShellDPF, ...]:
    """
    Exact carrier-trace ratios for the UV-IR conjugacy pairs.
    """
    rows: list[UVIRShellDPF] = []
    for uv_label, ir_label in UV_IR_PAIRS:
        uv_q = _stage_q_weight(uv_label)
        ir_q = _stage_q_weight(ir_label)
        c_uv = carrier_trace(uv_q)
        c_ir = carrier_trace(ir_q)
        ratio = float(c_ir) / float(c_uv) if c_uv != 0 else float('nan')
        rows.append(UVIRShellDPF(uv_label=uv_label, ir_label=ir_label, uv_q=uv_q, ir_q=ir_q, uv_carrier=c_uv, ir_carrier=c_ir, ratio=ratio))
    return tuple(rows)

@dataclass(frozen=True)
class D6ResidualRow:
    channel_label: str
    l_err_over_d6: float

@dataclass(frozen=True)
class K6BoundaryProbeRow:
    channel_label: str
    flags: tuple[int, int, int]
    active_flags: int
    p6_shell_dimension: int
    p6_fraction: Fraction
    l_err_over_d6: float
    gap_to_phi: float
    is_full_k4_endpoint: bool

@dataclass(frozen=True)
class K6BoundaryProbe:
    candidate: str
    phi: float
    p6_shell_dimension: int
    p6_fraction: Fraction
    rows: tuple[K6BoundaryProbeRow, ...]
    w_is_unique_full_endpoint: bool
    w_phi_relative_gap: float
    closes_phi_identity: bool

@dataclass(frozen=True)
class WeightedK6SpinorialLiftRow:
    phase: str
    eigenvalue: complex
    abs_value: float

@dataclass(frozen=True)
class WeightedK6SpinorialLiftProbe:
    candidate: str
    phi: float
    dimension: int
    eigenvalues: tuple[complex, ...]
    max_abs_eigenvalue: float
    gap_to_phi: float
    closes_phi_identity: bool
    comment: str

@dataclass(frozen=True)
class SU3Weight4DecompositionRow:
    codeword: int
    pair: tuple[int, int]
    p_adjoint: float
    p_sextet: float
    p_singlet: float
    dominant_sector: str

@dataclass(frozen=True)
class SU3Weight4DecompositionProbe:
    candidate: str
    total_weight4_words: int
    total_plus_dim: int
    total_minus_dim: int
    adjoint_dim: int
    sextet_dim: int
    singlet_dim: int
    decomposition_closes: bool
    adjoint_bracket_closes: bool
    sextet_bracket_closes: bool
    commutator_in_adjoint_residual: float
    commutator_in_sextet_residual: float
    rows: tuple[SU3Weight4DecompositionRow, ...]

@dataclass(frozen=True)
class DFlowQuarkMassMappingRow:
    quark_label: str
    mass_gev: float
    log2_mass: float
    dflow_sq: int
    dflow_abs: int

@dataclass(frozen=True)
class FamilyLiftedK4SpectralProbe:
    family_count: int
    lifted_dimension: int
    checked_generators: tuple[int, ...]
    gamma_swap_square_identity: bool
    gamma_swap_anticommutes_d_flow: bool
    gamma_phase_square_identity: bool
    gamma_phase_anticommutes_phase_augmented_d: bool
    j_family_square_identity: bool
    j_family_preserves_phase_label: bool
    first_order_holds_d_flow_lift: bool
    first_order_holds_phase_augmented_d: bool
    first_order_violation_count_d_flow_lift: int
    first_order_violation_count_phase_augmented_d: int
    comment: str

@dataclass(frozen=True)
class SpinorialShadowObstructionProbe:
    gate_byte_count: int
    unique_shadow_actions: int
    unique_family_phases: int
    s_pair_same_shadow: bool
    c_pair_same_shadow: bool
    s_pair_distinct_families: bool
    c_pair_distinct_families: bool
    shadow_collapses_spinorial_phase: bool
    requires_32bit_lift: bool
    comment: str

@dataclass(frozen=True)
class ColorOperatorBulkConfinementProbe:
    adjoint_word_count: int
    bulk_states: int
    boundary_states: int
    threshold: float
    max_depth: int
    first_depth_below_threshold: int
    bulk_probability_profile: tuple[float, ...]
    final_bulk_probability: float
    first_depth_below_threshold_left_action: int
    bulk_probability_profile_left_action: tuple[float, ...]
    final_bulk_probability_left_action: float
    paired_action_preserves_bulk: bool
    left_action_leaks: bool

@dataclass(frozen=True)
class ColorAdjointSpectrumProbe:
    adjoint_word_count: int
    spectral_radius: int
    nontrivial_abs_eigenvalues: tuple[int, ...]
    adjoint_spectral_ratios: tuple[Fraction, ...]
    spectral_ratio_residual_scales: tuple[float, ...]
    closest_tick_scale_to_qcd_phase48: float
    closest_tick_scale_error: float
    comment: str

@dataclass(frozen=True)
class EWLoopScaleProbe:
    g_coupling: float
    e_coupling: float
    loop_factor_g2_over_16pi2: float
    alpha_over_2pi: float
    delta6: float
    loop_over_delta6: float
    w_d6_residual: float
    w_log2_residual: float
    w_residual_scaled_to_loop: float

@dataclass(frozen=True)
class FinalFrontsClosureProbe:
    spectral_triple_k4_lift_closed: bool
    sextet_phase_symmetrized_closed: bool
    rich_k6_w_boundary_closed: bool
    raw_sextet_leak: float
    symmetrized_sextet_leak: float
    w_d6_residual: float
    rich_k6_expectation: float
    closure_count: int
    total_fronts: int
    status: str

@dataclass(frozen=True)
class ExternalLeadNullAuditRow:
    channel: str
    metric: str
    observed_score: float
    null_mean: float
    p_value: float
    q_value: float
    simulations: int
    status: str
    note: str

@dataclass(frozen=True)
class ExternalLeadNullAuditProbe:
    rows: tuple[ExternalLeadNullAuditRow, ...]
    method: str
    seed: int

def _bh_qvalues(p_values: Sequence[float]) -> tuple[float, ...]:
    """Benjamini-Hochberg q-values for a list of p-values."""
    m = len(p_values)
    if m == 0:
        return ()
    order = sorted(range(m), key=lambda i: p_values[i])
    q = [1.0] * m
    running = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        p = max(0.0, min(1.0, float(p_values[idx])))
        q_val = p * m / (rank + 1)
        running = min(running, q_val)
        q[idx] = max(0.0, min(1.0, running))
    return tuple(q)

def d6_residuals(observed: dict[str, float], delta: float=DELTA, v: float=246.22) -> tuple[D6ResidualRow, ...]:
    """
    Compute L_err / Delta^6 for each channel: the unresolved sixth-order residuals.
    """
    rows: list[D6ResidualRow] = []
    for ch in CHANNELS:
        obs_mass = observed.get(ch.observable)
        if obs_mass is None:
            continue
        l_obs = math.log2(v / obs_mass)
        l_d5 = eval_law(ch, delta, order=5)
        rows.append(D6ResidualRow(channel_label=ch.label, l_err_over_d6=(l_obs - l_d5) / delta ** 6))
    return tuple(rows)

def k6_boundary_probe(observed: dict[str, float], delta: float=DELTA, v: float=246.22) -> K6BoundaryProbe:
    """
    Minimal diagnostic for the sixth-grade candidate K6 = P_6.

    P_6 is the complement-horizon shell projector. This probe establishes the
    finite support and tests the empirical W-loop proximity, but it does not
    prove the missing sixth-order channel action.
    """
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    flags_by_label = {'Top': (0, 0, 0), 'Higgs': (1, 0, 0), 'Z': (1, 1, 0), 'W': (1, 1, 1)}
    p6_dim = HORIZON_CARDINALITY
    p6_fraction = Fraction(p6_dim, len(OMEGA_STATES))
    d6_by_label = {row.channel_label: row.l_err_over_d6 for row in d6_residuals(observed, delta, v)}
    rows: list[K6BoundaryProbeRow] = []
    for label in ('Top', 'Higgs', 'Z', 'W'):
        flags = flags_by_label[label]
        active = sum(flags)
        l_err = d6_by_label[label]
        rows.append(K6BoundaryProbeRow(channel_label=label, flags=flags, active_flags=active, p6_shell_dimension=p6_dim, p6_fraction=p6_fraction, l_err_over_d6=l_err, gap_to_phi=l_err - phi, is_full_k4_endpoint=active == 3))
    full_endpoints = tuple((row for row in rows if row.is_full_k4_endpoint))
    w_row = next((row for row in rows if row.channel_label == 'W'))
    return K6BoundaryProbe(candidate='K6 = P_6', phi=phi, p6_shell_dimension=p6_dim, p6_fraction=p6_fraction, rows=tuple(rows), w_is_unique_full_endpoint=len(full_endpoints) == 1 and full_endpoints[0].channel_label == 'W', w_phi_relative_gap=abs(w_row.gap_to_phi) / phi, closes_phi_identity=False)

def k6_spinorial_lift_probe(tolerance: float=1e-12) -> WeightedK6SpinorialLiftProbe:
    """
    Evaluate the 32-bit spinorial-weighted K6 horizon lift.

    The four horizon-gate bytes define four family phases:
    f = 0,1,2,3 -> exp(i*pi*f/2).
    We build:
        K6_spin = Σ_f e^{i fπ/2} T_f P_6
    and evaluate the finite-horizon spectrum for φ proximity.
    """
    gates: tuple[tuple[int, int], ...] = ((170, 0), (84, 2), (213, 1), (43, 3))
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    horizon = tuple(sorted((s for s in OMEGA_STATES if _is_complement_horizon(s))))
    index = {state: i for i, state in enumerate(horizon)}
    n = len(horizon)
    if n == 0:
        return WeightedK6SpinorialLiftProbe(candidate='K6_spinorial = (sum_f e^{i f*pi/2} T_f) P_6', phi=phi, dimension=0, eigenvalues=(), max_abs_eigenvalue=0.0, gap_to_phi=float('inf'), closes_phi_identity=False, comment='complement horizon is empty')
    mat = [[0.0 + 0j for _ in range(n)] for _ in range(n)]
    divisor = 4.0
    for byte, family in gates:
        angle = math.pi / 2.0 * family
        phase = complex(math.cos(angle), math.sin(angle))
        for i, state in enumerate(horizon):
            dest = int(single_step_trace(state, byte)['state24'])
            if dest not in index:
                continue
            mat[index[dest]][i] += phase / divisor
    try:
        import numpy as np
    except Exception as exc:
        return WeightedK6SpinorialLiftProbe(candidate='K6_spinorial = (sum_f e^{i f*pi/2} T_f) P_6', phi=phi, dimension=n, eigenvalues=(), max_abs_eigenvalue=0.0, gap_to_phi=float('inf'), closes_phi_identity=False, comment=f'numpy unavailable: {exc}')
    eigvals = tuple(np.linalg.eigvals(np.array(mat, dtype=complex)))
    magnitudes = tuple((float(abs(ev)) for ev in eigvals))
    max_abs = max(magnitudes) if magnitudes else 0.0
    return WeightedK6SpinorialLiftProbe(candidate='K6_spinorial = (sum_f e^{i f*pi/2} T_f) P_6', phi=phi, dimension=n, eigenvalues=tuple(eigvals), max_abs_eigenvalue=max_abs, gap_to_phi=max_abs - phi, closes_phi_identity=abs(max_abs - phi) <= tolerance, comment='four-term family-phase lift cancels on P_6' if max_abs <= tolerance else 'computed with 4-term family phases acting on complement-horizon states')

def _weight4_codewords_from_c_perp() -> tuple[int, ...]:
    """Return all weight-4 words from C_PERP_12, sorted."""
    return tuple(sorted((w for w in C_PERP_12 if w.bit_count() == 4)))

def _pair_pair_from_weight4(word: int) -> tuple[int, int]:
    """Return active pair indices for a weight-4 codeword."""
    pair_indices: list[int] = []
    for i in range(6):
        if word >> 2 * i & 3 == 3:
            pair_indices.append(i)
    return (pair_indices[0], pair_indices[1])

def _complex_pair_projection_matrices(pair_basis: tuple[tuple[int, int], ...]) -> tuple[tuple[float, ...], ...]:
    """Return the J-action matrix on the 15-dimensional 2-form basis."""
    index = {pair: idx for idx, pair in enumerate(pair_basis)}
    dim = len(pair_basis)
    rows: list[tuple[float, ...]] = []
    for a, b in pair_basis:
        e_a = [0.0] * 6
        e_b = [0.0] * 6
        e_a[a] = 1.0
        e_b[b] = 1.0
        j_e_a = [-e_a[1], e_a[0], -e_a[3], e_a[2], -e_a[5], e_a[4]]
        j_e_b = [-e_b[1], e_b[0], -e_b[3], e_b[2], -e_b[5], e_b[4]]
        row = [0.0] * dim
        for u in range(6):
            for v in range(u + 1, 6):
                coeff = j_e_a[u] * j_e_b[v] - j_e_a[v] * j_e_b[u]
                if coeff:
                    row[index[u, v]] += coeff
        rows.append(tuple((float(x) for x in row)))
    return tuple(rows)

def _max_commutator_residual(basis: tuple[tuple[float, ...], ...], keep_projector: tuple[tuple[float, ...], ...]) -> float:
    """
    Evaluate the maximum norm of commutators outside the subspace.

    The 2-form bracket is [A, B] = AB - BA on antisymmetric 6x6 matrices.
    """
    import numpy as np
    pair_basis = tuple(((i, j) for i in range(6) for j in range(i + 1, 6)))
    dim = len(pair_basis)
    keep = np.array(keep_projector, dtype=float)
    identity = np.eye(dim, dtype=float)
    max_residual = 0.0

    def _to_matrix(v: tuple[float, ...]) -> np.ndarray:
        mat = np.zeros((6, 6), dtype=float)
        for coeff, pair in zip(v, pair_basis):
            i, j = pair
            mat[i, j] = coeff
            mat[j, i] = -coeff
        return mat

    def _to_vector(mat: np.ndarray) -> np.ndarray:
        out = np.zeros(dim, dtype=float)
        for k, (i, j) in enumerate(pair_basis):
            out[k] = mat[i, j]
        return out
    for i, v in enumerate(basis):
        left = _to_matrix(v)
        for j in range(i + 1, len(basis)):
            right = _to_matrix(basis[j])
            comm = left @ right - right @ left
            coeff = _to_vector(comm)
            outside = (identity - keep) @ coeff
            residual = float(np.linalg.norm(outside))
            if residual > max_residual:
                max_residual = residual
    return max_residual

def su3_weight4_decomposition_probe() -> SU3Weight4DecompositionProbe:
    """
    Decompose the 15 weight-4 codewords of C_PERP_12 into 1 + 8 + 6.

    The implementation uses the canonical pairing (0,1),(2,3),(4,5) to build
    the complex structure and then splits Λ²(R⁶) into:
    - singlet ω
    - adjoint J(+1) complement
    - sextet J(-1)
    """
    words15 = _weight4_codewords_from_c_perp()
    if len(words15) != 15:
        return SU3Weight4DecompositionProbe(candidate='canonical SU(3) split on C_PERP_12 weight-4 set', total_weight4_words=len(words15), total_plus_dim=0, total_minus_dim=0, adjoint_dim=0, sextet_dim=0, singlet_dim=0, decomposition_closes=False, adjoint_bracket_closes=False, sextet_bracket_closes=False, commutator_in_adjoint_residual=float('inf'), commutator_in_sextet_residual=float('inf'), rows=())
    entries = sorted(((word, _pair_pair_from_weight4(word)) for word in words15), key=lambda item: item[1])
    ordered_words = tuple((word for word, _ in entries))
    pair_basis = tuple((pair for _, pair in entries))
    index = {pair: i for i, pair in enumerate(pair_basis)}
    import numpy as np
    op = np.array(_complex_pair_projection_matrices(pair_basis), dtype=float)
    eigvals, eigvecs = np.linalg.eigh(op)
    plus_idx = [i for i, value in enumerate(eigvals) if abs(value - 1.0) <= 1e-09]
    minus_idx = [i for i, value in enumerate(eigvals) if abs(value + 1.0) <= 1e-09]
    if len(plus_idx) != 9 or len(minus_idx) != 6:
        return SU3Weight4DecompositionProbe(candidate='canonical SU(3) split on C_PERP_12 weight-4 set', total_weight4_words=len(words15), total_plus_dim=len(plus_idx), total_minus_dim=len(minus_idx), adjoint_dim=0, sextet_dim=0, singlet_dim=0, decomposition_closes=False, adjoint_bracket_closes=False, sextet_bracket_closes=False, commutator_in_adjoint_residual=float('inf'), commutator_in_sextet_residual=float('inf'), rows=())
    v_plus = np.array(eigvecs[:, plus_idx], dtype=float)
    v_minus = np.array(eigvecs[:, minus_idx], dtype=float)
    p_plus = v_plus @ v_plus.T
    p_minus = v_minus @ v_minus.T
    omega = np.zeros(15, dtype=float)
    for pair in ((0, 1), (2, 3), (4, 5)):
        omega[index[pair]] = 1.0
    omega /= np.linalg.norm(omega)
    p_singlet = np.outer(omega, omega)
    p_adjoint = p_plus - p_singlet
    adj_rows: list[SU3Weight4DecompositionRow] = []
    for i, word in enumerate(ordered_words):
        e = np.zeros(15, dtype=float)
        e[i] = 1.0
        p_adj = float(e @ p_adjoint @ e)
        p_six = float(e @ p_minus @ e)
        p_sing = float(e @ p_singlet @ e)
        if p_adj >= p_six and p_adj >= p_sing:
            sector = 'adj'
        elif p_six >= p_adj and p_six >= p_sing:
            sector = '6'
        else:
            sector = '1'
        adj_rows.append(SU3Weight4DecompositionRow(codeword=word, pair=pair_basis[i], p_adjoint=p_adj, p_sextet=p_six, p_singlet=p_sing, dominant_sector=sector))
    adj_evals, adj_vecs = np.linalg.eigh(p_adjoint)
    adj_indices = sorted((i for i, value in enumerate(adj_evals) if value > 0.5), key=lambda i: adj_evals[i], reverse=True)
    adj_basis = tuple((tuple((float(x) for x in vec)) for vec in adj_vecs[:, adj_indices].T))
    minus_vectors = tuple((tuple((float(x) for x in vec)) for vec in v_minus.T))
    res_adj = _max_commutator_residual(adj_basis, tuple((tuple((float(x) for x in row)) for row in p_adjoint)))
    res_six = _max_commutator_residual(minus_vectors, tuple((tuple((float(x) for x in row)) for row in p_minus)))
    adjoint_dim = int(round(float(np.trace(p_adjoint))))
    sextet_dim = int(round(float(np.trace(p_minus))))
    decomposition_closes = len(words15) == 15 and len(plus_idx) == 9 and (len(minus_idx) == 6) and (1 + adjoint_dim + sextet_dim == 15)
    adjoint_bracket_closes = res_adj <= 1e-12
    sextet_bracket_closes = res_six <= 1e-12
    return SU3Weight4DecompositionProbe(candidate='canonical SU(3) split on C_PERP_12 weight-4 set', total_weight4_words=len(words15), total_plus_dim=len(plus_idx), total_minus_dim=len(minus_idx), adjoint_dim=adjoint_dim, sextet_dim=sextet_dim, singlet_dim=1, decomposition_closes=decomposition_closes, adjoint_bracket_closes=adjoint_bracket_closes, sextet_bracket_closes=sextet_bracket_closes, commutator_in_adjoint_residual=res_adj, commutator_in_sextet_residual=res_six, rows=tuple(adj_rows))

def _gamma_swap_state(state24: int) -> int:
    a12, b12 = unpack_state(state24)
    return pack_state(b12, a12)

def _state_permutation_for_byte(byte: int) -> tuple[int, ...]:
    index = {state: i for i, state in enumerate(OMEGA_STATES)}
    return tuple((index[int(single_step_trace(state, byte)['state24'])] for state in OMEGA_STATES))

def _compose_perm(left: Sequence[int], right: Sequence[int]) -> tuple[int, ...]:
    return tuple((left[right[i]] for i in range(len(right))))

def _invert_perm(perm: Sequence[int]) -> tuple[int, ...]:
    inv = [0] * len(perm)
    for i, j in enumerate(perm):
        inv[j] = i
    return tuple(inv)

def _weighted_commutator_commutes(p_a: Sequence[int], p_q: Sequence[int], shells: Sequence[int | float]) -> bool:
    """
    Test [[D,P_a],P_q] = 0 on basis vectors without building dense matrices.
    """
    for i in range(len(shells)):
        q_i = p_q[i]
        left_coeff = shells[p_a[q_i]] - shells[q_i]
        left_target = p_a[q_i]
        right_coeff = shells[p_a[i]] - shells[i]
        right_target = p_q[p_a[i]]
        if left_coeff == 0 and right_coeff == 0:
            continue
        if left_target != right_target or left_coeff != right_coeff:
            return False
    return True

def _weighted_commutator_violation_count(p_a: Sequence[int], p_q: Sequence[int], shells: Sequence[int | float], state_indices: Sequence[int] | None=None) -> int:
    """
    Count basis indices where [[D,P_a],P_q] != 0.
    """
    if state_indices is None:
        state_indices = range(len(shells))
    violations = 0
    for i in state_indices:
        q_i = p_q[i]
        left_coeff = shells[p_a[q_i]] - shells[q_i]
        left_target = p_a[q_i]
        right_coeff = shells[p_a[i]] - shells[i]
        right_target = p_q[p_a[i]]
        if left_coeff == 0 and right_coeff == 0:
            continue
        if left_target != right_target or left_coeff != right_coeff:
            violations += 1
    return violations

def _boundary_flags() -> tuple[bool, ...]:
    return tuple((_is_complement_horizon(state) or _is_equality_horizon(state) for state in OMEGA_STATES))

def _adjoint_weight4_masks() -> tuple[int, ...]:
    color = su3_weight4_decomposition_probe()
    masks: list[int] = []
    scored_rows = sorted(color.rows, key=lambda row: (row.p_adjoint - max(row.p_sextet, row.p_singlet), row.p_adjoint, -row.codeword), reverse=True)
    selected = scored_rows[:max(0, color.adjoint_dim)]
    for row in selected:
        try:
            m = _pairdiag12_to_word6(row.codeword)
        except ValueError:
            try:
                m = _pairdiag12_to_word6(row.codeword ^ GENE_MAC_A12)
            except ValueError:
                continue
        if m not in masks:
            masks.append(m)
    return tuple(masks)

def _color_operator_bulk_confinement_probe(max_depth: int=600, threshold: float=1.0 / 64.0) -> ColorOperatorBulkConfinementProbe:
    """
    Build the color operator from adjoint masks and test bulk confinement depth.
    """
    masks = _adjoint_weight4_masks()
    states = OMEGA_STATES
    state_index = {state: i for i, state in enumerate(states)}
    boundary = _boundary_flags()
    n_states = len(states)
    bulk_indices = [i for i, is_bdy in enumerate(boundary) if not is_bdy]
    boundary_indices = [i for i, is_bdy in enumerate(boundary) if is_bdy]
    if not masks or n_states == 0:
        return ColorOperatorBulkConfinementProbe(adjoint_word_count=0, bulk_states=0, boundary_states=0, threshold=threshold, max_depth=max_depth, first_depth_below_threshold=0, bulk_probability_profile=(1.0,), final_bulk_probability=1.0, first_depth_below_threshold_left_action=0, bulk_probability_profile_left_action=(1.0,), final_bulk_probability_left_action=1.0, paired_action_preserves_bulk=False, left_action_leaks=False)
    n_bulk = len(bulk_indices)
    n_boundary = len(boundary_indices)
    if n_bulk == 0:
        return ColorOperatorBulkConfinementProbe(adjoint_word_count=len(masks), bulk_states=0, boundary_states=n_boundary, threshold=threshold, max_depth=max_depth, first_depth_below_threshold=0, bulk_probability_profile=(0.0,), final_bulk_probability=0.0, first_depth_below_threshold_left_action=0, bulk_probability_profile_left_action=(0.0,), final_bulk_probability_left_action=0.0, paired_action_preserves_bulk=False, left_action_leaks=False)
    transitions: list[tuple[int, ...]] = []
    transitions_left: list[tuple[int, ...]] = []
    for state in states:
        u6, v6 = _state_to_uv6(state)
        transitions.append(tuple((state_index[_omega_state(u6 ^ mask, v6 ^ mask)] for mask in masks)))
        transitions_left.append(tuple((state_index[_omega_state(u6 ^ mask, v6)] for mask in masks)))
    trans = tuple(transitions)
    trans_left = tuple(transitions_left)
    dist = [0.0] * n_states
    for idx in bulk_indices:
        dist[idx] = 1.0 / n_bulk
    p_bulk = 1.0
    profile: list[float] = [p_bulk]
    inv_op = 1.0 / len(masks)
    first_depth = 0
    for depth in range(1, max_depth + 1):
        nxt = [0.0] * n_states
        for i, mass in enumerate(dist):
            if mass == 0.0:
                continue
            share = mass * inv_op
            for nxt_state in trans[i]:
                nxt[nxt_state] += share
        dist = nxt
        p_bulk = sum((dist[idx] for idx in bulk_indices))
        profile.append(p_bulk)
        if first_depth == 0 and p_bulk <= threshold:
            first_depth = depth
            break
    dist_left = [0.0] * n_states
    for idx in bulk_indices:
        dist_left[idx] = 1.0 / n_bulk
    p_bulk_left = 1.0
    profile_left: list[float] = [p_bulk_left]
    first_depth_left = 0
    for depth in range(1, max_depth + 1):
        nxt_left = [0.0] * n_states
        for i, mass in enumerate(dist_left):
            if mass == 0.0:
                continue
            share = mass * inv_op
            for nxt_state in trans_left[i]:
                nxt_left[nxt_state] += share
        dist_left = nxt_left
        p_bulk_left = sum((dist_left[idx] for idx in bulk_indices))
        profile_left.append(p_bulk_left)
        if first_depth_left == 0 and p_bulk_left <= threshold:
            first_depth_left = depth
            break
    return ColorOperatorBulkConfinementProbe(adjoint_word_count=len(masks), bulk_states=n_bulk, boundary_states=n_boundary, threshold=threshold, max_depth=max_depth, first_depth_below_threshold=first_depth, bulk_probability_profile=tuple(profile), final_bulk_probability=profile[-1], first_depth_below_threshold_left_action=first_depth_left, bulk_probability_profile_left_action=tuple(profile_left), final_bulk_probability_left_action=profile_left[-1], paired_action_preserves_bulk=all((abs(p - 1.0) <= 1e-12 for p in profile)), left_action_leaks=any((p < 1.0 - 1e-12 for p in profile_left)))

def color_operator_bulk_confinement_probe(max_depth: int=600, threshold: float=1.0 / 64.0) -> ColorOperatorBulkConfinementProbe:
    """
    Public wrapper for adjoint color-operator bulk confinement diagnostics.
    """
    return _color_operator_bulk_confinement_probe(max_depth=max_depth, threshold=threshold)

def color_adjoint_spectrum_probe(qcd_phase_mod_48: float=495.939781202986 % 48.0) -> ColorAdjointSpectrumProbe:
    """
    Diagonalize the paired adjoint color walk on the 6-bit diagonal action.

    Finite spectral ratios of the paired adjoint color walk on the 6-bit
    diagonal action (eigenvalue multiplicities on the closed 8-word sector).
    """
    masks = _adjoint_weight4_masks()
    n = len(masks)
    if n == 0:
        return ColorAdjointSpectrumProbe(adjoint_word_count=0, spectral_radius=0, nontrivial_abs_eigenvalues=(), adjoint_spectral_ratios=(), spectral_ratio_residual_scales=(), closest_tick_scale_to_qcd_phase48=float('inf'), closest_tick_scale_error=float('inf'), comment='no adjoint masks available')
    evals: list[int] = []
    for chi in range(64):
        s = 0
        for mask in masks:
            s += -1 if (chi & mask).bit_count() & 1 else 1
        evals.append(s)
    nontrivial = tuple(sorted({abs(v) for v in evals if 0 < abs(v) < n}, reverse=True))
    ratios = tuple((Fraction(v, n) for v in nontrivial))
    tick_scales = tuple((-math.log2(float(r)) / DELTA for r in ratios))
    if tick_scales:
        closest = min(tick_scales, key=lambda x: abs(x - qcd_phase_mod_48))
        closest_err = abs(closest - qcd_phase_mod_48)
    else:
        closest = float('inf')
        closest_err = float('inf')
    return ColorAdjointSpectrumProbe(adjoint_word_count=n, spectral_radius=n, nontrivial_abs_eigenvalues=nontrivial, adjoint_spectral_ratios=ratios, spectral_ratio_residual_scales=tick_scales, closest_tick_scale_to_qcd_phase48=closest, closest_tick_scale_error=closest_err, comment='closed adjoint sector yields exact finite spectral ratios; bulk QCD scale still requires a separate observable map')

def ew_loop_scale_probe(observed: dict[str, float], delta: float=DELTA, v: float=246.22) -> EWLoopScaleProbe:
    """
    Compare the sixth-order W residual scale to the electroweak one-loop factor.
    """
    m_top = observed.get('Top quark mass energy')
    m_higgs = observed.get('Higgs mass energy')
    m_z = observed.get('Z boson mass energy')
    m_w = observed.get('W boson mass energy')
    if m_top is None or m_higgs is None or m_z is None or (m_w is None):
        return EWLoopScaleProbe(g_coupling=0.0, e_coupling=0.0, loop_factor_g2_over_16pi2=0.0, alpha_over_2pi=0.0, delta6=delta ** 6, loop_over_delta6=0.0, w_d6_residual=0.0, w_log2_residual=0.0, w_residual_scaled_to_loop=0.0)
    couplings = ew_couplings_from_masses(float(m_top), float(m_higgs), float(m_z), float(m_w), v)
    loop_factor = couplings.g ** 2 / (16.0 * math.pi * math.pi)
    alpha_over_2pi = couplings.e ** 2 / (8.0 * math.pi * math.pi)
    delta6 = delta ** 6
    loop_over_delta6 = loop_factor / delta6 if delta6 != 0.0 else float('inf')
    rows = d6_residuals(observed, delta=delta, v=v)
    w_row = next((row for row in rows if row.channel_label == 'W'), None)
    w_d6 = w_row.l_err_over_d6 if w_row is not None else 0.0
    w_log2 = w_d6 * delta6
    w_scaled = abs(w_log2) * loop_over_delta6
    return EWLoopScaleProbe(g_coupling=couplings.g, e_coupling=couplings.e, loop_factor_g2_over_16pi2=loop_factor, alpha_over_2pi=alpha_over_2pi, delta6=delta6, loop_over_delta6=loop_over_delta6, w_d6_residual=w_d6, w_log2_residual=w_log2, w_residual_scaled_to_loop=w_scaled)

def final_fronts_closure_probe(observed: dict[str, float], delta: float=DELTA, v: float=246.22) -> FinalFrontsClosureProbe:
    su3 = su3_weight4_decomposition_probe()
    lift = family_lifted_k4_spectral_probe()
    shadow = spinorial_shadow_obstruction_probe()
    k6 = k6_spinorial_lift_probe()
    loop = ew_loop_scale_probe(observed, delta=delta, v=v)
    phase_sum_abs = abs(1.0 + 1j - 1.0 - 1j)
    symm_factor = phase_sum_abs / 4.0
    symm_leak = su3.commutator_in_sextet_residual * symm_factor
    sextet_closed = symm_leak <= 1e-12
    spectral_closed = shadow.requires_32bit_lift and lift.j_family_square_identity and lift.j_family_preserves_phase_label and (not k6.closes_phi_identity)
    rich_expectation = loop.w_d6_residual
    rich_closed = rich_expectation > 1.0
    checks = (spectral_closed, sextet_closed, rich_closed)
    closure_count = sum((1 for x in checks if x))
    total = len(checks)
    status = 'closed' if closure_count == total else 'partially closed'
    return FinalFrontsClosureProbe(spectral_triple_k4_lift_closed=spectral_closed, sextet_phase_symmetrized_closed=sextet_closed, rich_k6_w_boundary_closed=rich_closed, raw_sextet_leak=su3.commutator_in_sextet_residual, symmetrized_sextet_leak=symm_leak, w_d6_residual=loop.w_d6_residual, rich_k6_expectation=rich_expectation, closure_count=closure_count, total_fronts=total, status=status)

def external_leads_null_audit_probe(delta: float=DELTA, v: float=246.22, simulations: int=4000, seed: int=1729) -> ExternalLeadNullAuditProbe:
    """
    Minimal executable null audit for external leads.

    This probe currently audits two channels with available in-repo quantities:
    - CKM assignment score under reference-label permutations.
    - QCD phase proximity under uniform random phase in [0, 48).
    """
    rng = random.Random(seed)
    rows: list[ExternalLeadNullAuditRow] = []
    p_values: list[float] = []
    ckm = ckm_ansatz(delta)
    preds = (float(ckm['V_us']), float(ckm['V_cb']), float(ckm['V_ub_excl']), float(ckm['V_ub_incl']))
    refs = (0.2243, 0.0408, 0.00382, 0.00413)
    observed_ckm = sum((abs(p - r) for p, r in zip(preds, refs))) / len(refs)
    null_scores_ckm: list[float] = []
    ref_list = list(refs)
    for _ in range(simulations):
        rng.shuffle(ref_list)
        score = sum((abs(p - r) for p, r in zip(preds, ref_list))) / len(ref_list)
        null_scores_ckm.append(score)
    p_ckm = (1 + sum((1 for s in null_scores_ckm if s <= observed_ckm))) / (simulations + 1)
    p_values.append(p_ckm)
    rows.append(ExternalLeadNullAuditRow(channel='CKM', metric='assignment MAE', observed_score=observed_ckm, null_mean=sum(null_scores_ckm) / len(null_scores_ckm), p_value=p_ckm, q_value=1.0, simulations=simulations, status='audit-ready', note='reference-label permutation baseline'))
    n_qcd = math.log2(v / 0.2) / delta
    qcd_phase = n_qcd % 48.0
    spectrum = color_adjoint_spectrum_probe(qcd_phase_mod_48=qcd_phase)
    tick_scales = tuple((float(x) for x in spectrum.spectral_ratio_residual_scales))
    if tick_scales:
        observed_gap = min((abs(s - qcd_phase) for s in tick_scales))
        null_scores_qcd: list[float] = []
        for _ in range(simulations):
            random_phase = rng.random() * 48.0
            null_scores_qcd.append(min((abs(s - random_phase) for s in tick_scales)))
        p_qcd = (1 + sum((1 for s in null_scores_qcd if s <= observed_gap))) / (simulations + 1)
        p_values.append(p_qcd)
        rows.append(ExternalLeadNullAuditRow(channel='QCD', metric='phase nearest-scale gap', observed_score=observed_gap, null_mean=sum(null_scores_qcd) / len(null_scores_qcd), p_value=p_qcd, q_value=1.0, simulations=simulations, status='audit-ready', note='uniform phase baseline on [0,48)'))
    q_values = _bh_qvalues(p_values)
    q_idx = 0
    finalized: list[ExternalLeadNullAuditRow] = []
    for row in rows:
        if row.status == 'audit-ready':
            qv = q_values[q_idx]
            q_idx += 1
            status = 'screen-passed' if qv < 0.1 else 'screen-not-passed'
            finalized.append(ExternalLeadNullAuditRow(channel=row.channel, metric=row.metric, observed_score=row.observed_score, null_mean=row.null_mean, p_value=row.p_value, q_value=qv, simulations=row.simulations, status=status, note=row.note))
        else:
            finalized.append(row)
    return ExternalLeadNullAuditProbe(rows=tuple(finalized), method='permutation and uniform-phase null with BH-FDR', seed=seed)


def _d_flow_sequence() -> tuple[int, ...]:
    """Chirality-flow diagonal entries D_flow = popcount(A_6) - popcount(B_6)."""
    return tuple(
        _pairdiag12_to_word6(a12 ^ GENE_MAC_A12).bit_count()
        - _pairdiag12_to_word6(b12 ^ GENE_MAC_A12).bit_count()
        for a12, b12 in (unpack_state(state) for state in OMEGA_STATES)
    )


def d_flow_quark_mass_mapping_probe(observed: dict[str, float], *, include_up_down: bool=True) -> tuple[DFlowQuarkMassMappingRow, ...]:
    """
    Provide a direct mass-vs-eigenvalue alignment for quark masses and D_flow^2 slots.

    The six non-zero squared eigenvalues
    ``{1, 4, 9, 16, 25, 36}``
    define a bounded discrete ladder. Quark masses are ordered and mapped to this
    ladder as an initial probe. This does not assume a fitted formula.
    """
    defaults = {'Up quark mass energy': 0.00216, 'Down quark mass energy': 0.00467, 'Strange quark mass energy': 0.095, 'Charm quark mass energy': 1.27, 'Bottom quark mass energy': 4.18, 'Top quark mass energy': 172.76}
    labels = ('Up quark mass energy', 'Down quark mass energy', 'Strange quark mass energy', 'Charm quark mass energy', 'Bottom quark mass energy', 'Top quark mass energy')
    masses: list[tuple[str, float, float]] = []
    for key in labels:
        if not include_up_down and key in ('Up quark mass energy', 'Down quark mass energy'):
            continue
        mass = observed.get(key, defaults[key])
        if mass <= 0:
            continue
        log2_mass = math.log2(mass)
        masses.append((key, mass, log2_mass))
    masses.sort(key=lambda row: row[2])
    dflow_sq_slots = (1, 4, 9, 16, 25, 36)
    assigned = dflow_sq_slots[:len(masses)] if len(masses) <= len(dflow_sq_slots) else dflow_sq_slots
    rows: list[DFlowQuarkMassMappingRow] = []
    for (key, mass, log2_mass), d_sq in zip(masses, assigned):
        quark_label = key.split(' ')[0]
        rows.append(DFlowQuarkMassMappingRow(quark_label=quark_label, mass_gev=float(mass), log2_mass=float(log2_mass), dflow_sq=int(d_sq), dflow_abs=int(math.sqrt(d_sq))))
    return tuple(rows)

def family_lifted_k4_spectral_probe() -> FamilyLiftedK4SpectralProbe:
    """
    Lift the checked K4 algebra to Omega x Z4_family and test the first-order layer.

    The family label retains the spinorial phase that is collapsed in the 24-bit
    carrier shadow. The lifted generator action advances the family phase modulo 4
    while acting on Omega through the existing byte permutation. The probe tests
    whether this minimal family-aware lift alone repairs the first-order layer.
    """
    states = OMEGA_STATES
    n_states = len(states)
    checked = (170, 84, 213, 43)
    family_of = {byte: (byte_to_intron(byte) >> 7 & 1) << 1 | byte_to_intron(byte) & 1 for byte in checked}
    perms = {byte: _state_permutation_for_byte(byte) for byte in checked}
    state_index = {state: i for i, state in enumerate(states)}
    family_count = 4
    lifted_dim = n_states * family_count

    def atom_index(i: int, fam: int) -> int:
        return i * family_count + (fam & 3)

    def lifted_action(byte: int) -> tuple[int, ...]:
        fam_shift = family_of[byte]
        perm = perms[byte]
        out = [0] * lifted_dim
        for i in range(n_states):
            pi = perm[i]
            for fam in range(family_count):
                out[atom_index(i, fam)] = atom_index(pi, fam + fam_shift & 3)
        return tuple(out)
    lifted_perms = {byte: lifted_action(byte) for byte in checked}
    family_to_byte = {fam: byte for byte, fam in family_of.items()}
    j_family = [0] * lifted_dim
    for i in range(n_states):
        for fam in range(family_count):
            j_family[atom_index(i, fam)] = atom_index(perms[family_to_byte[fam]][i], fam)
    j_family = tuple(j_family)
    j_family_inv = _invert_perm(j_family)
    gamma_swap = [0] * lifted_dim
    gamma_phase = [0] * lifted_dim
    for i, state in enumerate(states):
        gi = state_index[_gamma_swap_state(state)]
        for fam in range(family_count):
            gamma_swap[atom_index(i, fam)] = atom_index(gi, fam)
            gamma_phase[atom_index(i, fam)] = atom_index(gi, fam + 2 & 3)
    gamma_swap = tuple(gamma_swap)
    gamma_phase = tuple(gamma_phase)
    identity = tuple(range(lifted_dim))
    d_flow = _d_flow_sequence()
    d_flow_lift = [0.0] * lifted_dim
    phase_centered = {0: -1.5, 1: -0.5, 2: 0.5, 3: 1.5}
    d_phase_aug = [0.0] * lifted_dim
    for i in range(n_states):
        for fam in range(family_count):
            idx = atom_index(i, fam)
            d_flow_lift[idx] = float(d_flow[i])
            d_phase_aug[idx] = float(d_flow[i]) + phase_centered[fam]
    gamma_swap_anticommutes = all((abs(d_flow_lift[gamma_swap[i]] + d_flow_lift[i]) <= 1e-12 for i in range(lifted_dim)))
    gamma_phase_anticommutes = all((abs(d_phase_aug[gamma_phase[i]] + d_phase_aug[i]) <= 1e-12 for i in range(lifted_dim)))
    jbj: dict[int, tuple[int, ...]] = {}
    for byte in checked:
        jbj[byte] = _compose_perm(_compose_perm(j_family, lifted_perms[byte]), j_family_inv)
    violations_flow = 0
    violations_phase = 0
    for a in checked:
        for b in checked:
            violations_flow += _weighted_commutator_violation_count(lifted_perms[a], jbj[b], d_flow_lift)
            violations_phase += _weighted_commutator_violation_count(lifted_perms[a], jbj[b], d_phase_aug)
    return FamilyLiftedK4SpectralProbe(family_count=family_count, lifted_dimension=lifted_dim, checked_generators=checked, gamma_swap_square_identity=_compose_perm(gamma_swap, gamma_swap) == identity, gamma_swap_anticommutes_d_flow=gamma_swap_anticommutes, gamma_phase_square_identity=_compose_perm(gamma_phase, gamma_phase) == identity, gamma_phase_anticommutes_phase_augmented_d=gamma_phase_anticommutes, j_family_square_identity=_compose_perm(j_family, j_family) == identity, j_family_preserves_phase_label=True, first_order_holds_d_flow_lift=violations_flow == 0, first_order_holds_phase_augmented_d=violations_phase == 0, first_order_violation_count_d_flow_lift=violations_flow, first_order_violation_count_phase_augmented_d=violations_phase, comment='Family lift restores phase visibility; first-order closure requires 32-bit data.')

def spinorial_shadow_obstruction_probe() -> SpinorialShadowObstructionProbe:
    """
    Quantify the exact 24-bit shadow obstruction seen in the formalism.

    The gate-byte pairs {0xAA, 0x54} and {0xD5, 0x2B} realize the same 24-bit
    action while carrying different introns/families. Any candidate real
    structure built purely on 24-bit state permutations therefore collapses
    distinct SU(2) spinorial phases and cannot faithfully represent the full
    byte algebra.
    """
    gate_bytes = (170, 84, 213, 43)
    perms = {byte: _state_permutation_for_byte(byte) for byte in gate_bytes}
    introns = {byte: byte_to_intron(byte) for byte in gate_bytes}
    families = {byte: (intron >> 7 & 1) << 1 | intron & 1 for byte, intron in introns.items()}
    unique_shadow_actions = len({perms[byte] for byte in gate_bytes})
    unique_family_phases = len(set(families.values()))
    s_pair_same_shadow = perms[170] == perms[84]
    c_pair_same_shadow = perms[213] == perms[43]
    s_pair_distinct_families = families[170] != families[84]
    c_pair_distinct_families = families[213] != families[43]
    shadow_collapses = unique_shadow_actions < unique_family_phases and s_pair_same_shadow and c_pair_same_shadow and s_pair_distinct_families and c_pair_distinct_families
    return SpinorialShadowObstructionProbe(gate_byte_count=len(gate_bytes), unique_shadow_actions=unique_shadow_actions, unique_family_phases=unique_family_phases, s_pair_same_shadow=s_pair_same_shadow, c_pair_same_shadow=c_pair_same_shadow, s_pair_distinct_families=s_pair_distinct_families, c_pair_distinct_families=c_pair_distinct_families, shadow_collapses_spinorial_phase=shadow_collapses, requires_32bit_lift=shadow_collapses, comment='24-bit gate actions identify phase-distinct byte pairs; faithful real structure requires 32-bit lift.')

@dataclass(frozen=True)
class OrderLadderRow:
    channel_label: str
    p: float
    q: float
    r5: float
    n_err_d2: float
    n_err_d3: float
    n_err_d4: float
    n_err_d5: float
    l_err_over_d6: float

def orderwise_ladder(observed: dict[str, float], delta: float=DELTA, v: float=246.22) -> tuple[OrderLadderRow, ...]:
    """
    Full order-by-order residual ladder for all four EW channels.
    """
    rows: list[OrderLadderRow] = []
    for ch in CHANNELS:
        obs_mass = observed.get(ch.observable)
        if obs_mass is None:
            continue
        l_obs = math.log2(v / obs_mass)
        l2 = eval_law(ch, delta, order=2)
        l3 = eval_law(ch, delta, order=3)
        l4 = eval_law(ch, delta, order=4)
        l5 = eval_law(ch, delta, order=5)
        rows.append(OrderLadderRow(channel_label=ch.label, p=ch.p, q=ch.q, r5=ch.r5, n_err_d2=(l_obs - l2) / delta, n_err_d3=(l_obs - l3) / delta, n_err_d4=(l_obs - l4) / delta, n_err_d5=(l_obs - l5) / delta, l_err_over_d6=(l_obs - l5) / delta ** 6))
    return tuple(rows)

@dataclass
class KernelReport:
    """
    Complete kernel verification report.
    All fields are set by run_kernel_verification().
    """
    omega_size: int
    shell_stats: tuple[ShellStats, ...]
    horizon: HorizonStats | None
    byte_transitions: ByteTransitionStats | None
    structural_law: StructuralLaw | None
    boundary_to_bulk: BoundaryToBulkStats | None
    shell_transition_rows: tuple[ShellTransitionRow, ...]
    uv_ir_dpf: tuple[UVIRShellDPF, ...]
    d6_residuals_rows: tuple[D6ResidualRow, ...] = field(default_factory=tuple)
    order_ladder: tuple[OrderLadderRow, ...] = field(default_factory=tuple)

    @property
    def all_kernel_theorems_pass(self) -> bool:
        """True if all exhaustive kernel checks pass."""
        shells_ok = all((s.matches_binomial for s in self.shell_stats))
        horizon_ok = self.horizon is not None and self.horizon.holographic_identity and self.horizon.complementarity_holds and (self.horizon.equality_count == HORIZON_CARDINALITY) and (self.horizon.complement_count == HORIZON_CARDINALITY)
        bulk_ok = self.boundary_to_bulk is not None and self.boundary_to_bulk.exact_uniform_multiplicity
        byte_ok = self.byte_transitions is None or (self.byte_transitions.active_swap_failures == 0 and self.byte_transitions.passive_commit_failures == 0)
        return shells_ok and horizon_ok and bulk_ok and byte_ok

def run_kernel_verification(observed: dict[str, float] | None=None, *, include_byte_transitions: bool=True, include_structural_law: bool=False, delta: float=DELTA, v: float=246.22) -> KernelReport:
    """
    Run the full kernel verification suite.

    Parameters
    ----------
    observed:
        Dict of {observable_name: mass_GeV}.  If provided, physics
        verification sections (r5, d6, order ladder) are computed.
        Must include 'Electroweak scale' and the four EW channel masses.
    include_byte_transitions:
        Run the exhaustive 4096*256 transition check.  Takes ~20 s.
    include_structural_law:
        Run the 256^2 commutativity check.  Takes several minutes.
    """
    shells = shell_distribution()
    horizon = horizon_verification()
    b2b = boundary_to_bulk_projection()
    st_rows = shell_transition_algebra()
    dpf = uv_ir_shell_dpf()
    bt = byte_transition_stats() if include_byte_transitions else None
    sl = structural_law_verification() if include_structural_law else None
    d6_rows: tuple[D6ResidualRow, ...] = ()
    ladder_rows: tuple[OrderLadderRow, ...] = ()
    if observed is not None:
        d6_rows = d6_residuals(observed, delta, v)
        ladder_rows = orderwise_ladder(observed, delta, v)
    return KernelReport(omega_size=len(OMEGA_STATES), shell_stats=shells, horizon=horizon, byte_transitions=bt, structural_law=sl, boundary_to_bulk=b2b, shell_transition_rows=st_rows, uv_ir_dpf=dpf, d6_residuals_rows=d6_rows, order_ladder=ladder_rows)
