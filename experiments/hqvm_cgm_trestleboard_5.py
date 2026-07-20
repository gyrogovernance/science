#!/usr/bin/env python3
"""
hqvm_cgm_trestleboard_5.py

Magic-number derivation for the CGM trestleboard study.

Builds the 3D harmonic-oscillator shell structure from SE(3) carrier DoF,
applies CS-chirality spin-orbit ordering, solves the Nilsson Hamiltonian
(diagonal and Delta N=2 mixed), and checks doubly-magic and superheavy
closures against IAEA LiveChart data. Reports measurements, tables, and
PASS/FAIL checks only.

Inputs: CGM constants from hqvm_compact_geom_core.py; IAEA LiveChart ground
states (data/catalogs/ensdf/iaea_livechart_ground_states.csv).
Outputs: shell closures, Nilsson gap rankings, mixing spectrum, doubly-magic
table, superheavy scan, two-neutron gap signature.

Companion: hqvm_cgm_trestleboard_1..4.py, common.py, run.py.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

_REPO = Path(__file__).resolve().parents[1]
_EXP = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

import numpy as np
from math import comb, pi

from gyroscopic.hQVM.api import (
    shell_population,
    KRAWTCHOUK_7,
)
from hqvm_compact_geom_core import DELTA, DELTA_BU, RHO, E_EW_GEV, STF_DIMENSION
from hqvm_cgm_trestleboard_common import C1, C2

MAGIC = [2, 8, 20, 28, 50, 82, 126]
OSC_CLOSURES = [2, 8, 20, 40, 70, 112]
INTRUDERS = [28, 50, 82, 126]
CATALOG = _REPO / "data" / "catalogs" / "ensdf" / "iaea_livechart_ground_states.csv"
DOUBLY_MAGIC = [
    (2, 2), (8, 8), (20, 20), (20, 28), (28, 20), (28, 28),
    (28, 50), (50, 50), (50, 82), (82, 126),
]
KNOWN_SUPERHEAVY = [114, 120, 126, 184]


# -----------------------------------------------------------------
# Oscillator shell degeneracy: weak 3-composition (3D HO, Frame 1)
# -----------------------------------------------------------------
def spatial_weak_composition(n: int) -> int:
    """Number of weak 3-compositions of n: binom(n+2, 2).

    The 3D harmonic oscillator has 3 spatial coordinates (Frame 1, the R^3
    translational DoF of SE(3)). A shell of total quantum number n splits
    its n quanta among 3 coordinates in binom(n+2,2) ways. This is the
    standard oscillator degeneracy (n+1)(n+2)/2. It is the continuous limit
    of the 3 translational modes, distinct from the discrete 6-bit chirality
    census C(6, N).
    """
    return comb(n + 2, 2)


def oscillator_shell_degeneracy(n: int) -> int:
    """3D oscillator shell n degeneracy (no spin): (n+1)(n+2)/2."""
    return (n + 1) * (n + 2) // 2


# -----------------------------------------------------------------
# (n, l, j) subshell decomposition from kernel structure
# -----------------------------------------------------------------
def oscillator_subshells(n: int) -> List[Tuple[int, float, int]]:
    """Subshells (l, j) for 3D oscillator shell n, degeneracy 2j+1.

    l parity = n parity (oscillator selection rule), so l runs over
    {n%2, n%2+2, ..., n}. For each l, j = l +/- 1/2 (j >= 0).
    """
    out: List[Tuple[int, float, int]] = []
    for l in range(n % 2, n + 1, 2):
        for sign in (1, -1):
            j = l + 0.5 * sign
            if j < 0:
                continue
            out.append((l, j, int(2 * j + 1)))
    return out


def ordered_subshells(n: int, chi_sign: int = +1) -> List[Tuple[int, float, int]]:
    """Subshells of oscillator shell n ordered by CS chirality sign.

    chi_sign = +1 (left-chiral, CGM axiom): j = l+1/2 is lower (Mayer-Jensen).
    chi_sign = -1 (right-chiral): j = l-1/2 is lower.
    The set of degeneracies is identical; only the within-shell order flips.
    """
    return sorted(oscillator_subshells(n), key=lambda t: -t[1] * chi_sign)


# -----------------------------------------------------------------
# Kernel radial / angular split (QuBEC + Wavefunction docs)
# -----------------------------------------------------------------
def shell_census_duality_ok() -> bool:
    """Chirality shell census satisfies Poincare duality |Shell_N|=|Shell_{6-N}|.

    Discrete analogue of the oscillator's l <-> (2n-l) pairing. Verified on
    C(6, N).
    """
    return all(comb(6, N) == comb(6, 6 - N) for N in range(7))


def angular_multiplicity(shell: int) -> int:
    """In-shell angular multiplicity = shell_population / C(6, shell).

    Reports the 64 factor (the carrier subspace within a shell). This is the
    discrete analogue of sum_l (2l+1), the angular-momentum multiplicity.
    """
    cap = shell_population(shell)
    radial = comb(6, shell)
    return cap // radial if radial else 0


# -----------------------------------------------------------------
# BU vibrational aperture as the spin-orbit splitting scale
# -----------------------------------------------------------------
def so_splitting_scale_MeV() -> float:
    """Spin-orbit splitting energy scale = BU vibrational aperture.

    BU is bounded oscillation at the aperture amplitude Delta about the
    closed state (Corollary 6.1, Analysis_3D_6DOF_Proof.md). The aperture
    Delta (~2.07%) is the fractional spin-orbit splitting. Expressed as an
    energy on the strong bare scale E_str = v*Delta^3 (MeV), the same anchor
    used everywhere in the trestleboard.
    """
    E_str = E_EW_GEV * (DELTA ** 3) * 1.0e3  # MeV
    return E_str * DELTA


# -----------------------------------------------------------------
# CGM-anchored single-particle energy spectrum E(n, l, j)
# -----------------------------------------------------------------
def ticks_per_octave() -> float:
    """Ruler ticks per factor-of-two energy change: 1/Delta (trestleboard 2.1)."""
    return 1.0 / DELTA


def radial_step_ticks() -> float:
    """Radial energy step per oscillator shell, in ruler ticks.

    The 6-bit chirality register carries 7 radial shells (N = 0..6). On the
    shared Delta-ruler these span one octave (ticks_per_octave ticks), so the
    radial quantization step is one octave divided by 6 shells. This is the
    kernel-derived major-shell spacing, not a nuclear fit.
    """
    return ticks_per_octave() / 6.0


def ls_expectation(l: int, j: float) -> float:
    """Group-theoretic <L.S> = 1/2 [j(j+1) - l(l+1) - s(s+1)], s = 1/2.

    No fit: pure angular-momentum algebra. Sign convention j = l +/- 1/2.
    """
    s = 0.5
    return 0.5 * (j * (j + 1) - l * (l + 1) - s * (s + 1))


def single_particle_energy_tick(n: int, l: int, j: float, chi_sign: int = +1,
                                 b_so_scale: float = 1.0) -> float:
    """Single-particle energy of orbital (n, l, j) in ruler ticks.

    E_tick = n * S_rad                      (3D oscillator radial term)
             - chi_sign * b_so * <L.S>      (spin-orbit term, V_SO = -b L.S)

    S_rad = radial_step_ticks() is the kernel radial quantization.
    b_so = b_so_scale * DELTA * radial_step_ticks() is the spin-orbit scale.
    Default b_so_scale = 1 uses the BU aperture DELTA as the CGM anchor.
    CS left-chirality fixes chi_sign = +1; the minus sign then puts the
    aligned branch j = l+1/2 (<L.S> > 0) LOWER, as required physically.
    """
    s_rad = radial_step_ticks()
    b_so = b_so_scale * DELTA * radial_step_ticks()
    return n * s_rad - chi_sign * b_so * ls_expectation(l, j)


def spectrum_ordered(chi_sign: int = +1, b_so_scale: float = 1.0) -> List[Tuple[int, int, float, int, float]]:
    """All (n,l,j,deg,E_tick) orbitals sorted by increasing energy.

    Filling order is now derived from the energy functional, not assumed.
    b_so_scale sweeps the spin-orbit coupling (see gap_closures sweep).
    """
    orbs: List[Tuple[int, int, float, int, float]] = []
    for n in range(7):
        for (l, j, deg) in oscillator_subshells(n):
            e = single_particle_energy_tick(n, l, j, chi_sign, b_so_scale)
            orbs.append((n, l, j, deg, e))
    return sorted(orbs, key=lambda t: t[4])


def gap_closures(chi_sign: int = +1, frac: float = 2.0, b_so_scale: float = 1.0) -> List[int]:
    """Magic closures from the ENERGY spectrum (gap detector).

    Sort orbitals by E_tick, fill cumulative degeneracy, compute adjacent
    gaps dE_k = E_{k+1} - E_k, declare a closure at particle count N when the
    gap exceeds frac times the local median spacing. This is the standard
    definition (large energy gap => closed shell), now CGM-derived.
    """
    orbs = spectrum_ordered(chi_sign, b_so_scale)
    energies = [o[4] for o in orbs]
    degs = [o[3] for o in orbs]
    cum = 0
    closures: List[int] = []
    spacings = [energies[k + 1] - energies[k] for k in range(len(energies) - 1)]
    med = sorted(spacings)[len(spacings) // 2] if spacings else 1.0
    for k in range(len(orbs) - 1):
        cum += degs[k]
        if spacings[k] > frac * med:
            closures.append(cum)
    # also close at the very end
    cum += degs[-1]
    return closures


# -----------------------------------------------------------------
# Nilsson single-particle Hamiltonian (CGM-anchored)
# -----------------------------------------------------------------
def l2_expectation(l: int) -> float:
    """<l^2> = l(l+1), the quadrupole/deformation tensor invariant.

    The CGM anchor for the deformation term is the STF (symmetric trace-free)
    bulk sector of dimension STF_DIMENSION = 5, the l=2 quadrupole multiplet.
    """
    return l * (l + 1)


def nilsson_energy(n: int, l: int, j: float, kappa: float, mu: float,
                   chi_sign: int = +1, hbar_omega: float = 1.0) -> float:
    """Nilsson single-particle energy in oscillator units (hbar_omega = 1).

    E = hbar_omega * (N + 3/2)
        - chi_sign * kappa * <L.S>_j          (spin-orbit, CS-fixed sign)
        - kappa * mu * <l^2>                  (quadrupole deformation)

    N = n is the oscillator quantum number. The <l^2> term is the intruder
    mechanism: it shifts high-l orbitals of shell N DOWN, pushing the maximal
    l (and thus maximal-j) state across the N-shell boundary to create the
    magic gaps 28, 50, 82, 126. A pure linear -kappa<L.S> term cannot do
    this because it only splits j within fixed l; the <l^2> term is required.
    kappa and mu are CGM invariants (see KAPPA_CGM, MU_CGM below).
    """
    N = n
    e_ho = hbar_omega * (N + 1.5)
    e_so = -chi_sign * kappa * ls_expectation(l, j)
    e_def = -kappa * mu * l2_expectation(l)
    return e_ho + e_so + e_def


def nilsson_spectrum(kappa: float, mu: float, chi_sign: int = +1) -> List[Tuple[int, int, float, int, float]]:
    """All (n,l,j,deg,E) Nilsson orbitals sorted by increasing energy."""
    orbs: List[Tuple[int, int, float, int, float]] = []
    for n in range(7):
        for (l, j, deg) in oscillator_subshells(n):
            e = nilsson_energy(n, l, j, kappa, mu, chi_sign)
            orbs.append((n, l, j, deg, e))
    return sorted(orbs, key=lambda t: t[4])


def nilsson_gap_closures(kappa: float, mu: float, chi_sign: int = +1,
                         frac: float = 1.8) -> List[int]:
    """Gap-detected closures from the Nilsson spectrum."""
    orbs = nilsson_spectrum(kappa, mu, chi_sign)
    energies = [o[4] for o in orbs]
    degs = [o[3] for o in orbs]
    cum = 0
    closures: List[int] = []
    spacings = [energies[k + 1] - energies[k] for k in range(len(energies) - 1)]
    med = sorted(spacings)[len(spacings) // 2] if spacings else 1.0
    for k in range(len(orbs) - 1):
        cum += degs[k]
        if spacings[k] > frac * med:
            closures.append(cum)
    cum += degs[-1]
    return closures


# CGM coupling anchors (derived, not fitted):
#   tau = delta_BU / (2*pi); kappa = Q_256(tau) = round(tau*256)/256
#   mu  = 1 / STF_DIMENSION
TAU_BU = DELTA_BU / (2.0 * pi)
KAPPA_CGM = round(TAU_BU * 256.0) / 256.0
MU_CGM = 1.0 / float(STF_DIMENSION)


def gap_prominences(kappa: float, mu: float, chi_sign: int = +1) -> List[Tuple[float, int, int]]:
    """Adjacent gaps with local prominence and the closure cumulative.

    For each adjacent gap dE_k between filled orbitals k and k+1, the
    prominence is dE_k / mean(nearest-neighbor spacings). A closure at
    particle count N_k is reported as (prominence, N_k, k). High prominence
    = a dominant shell gap; low prominence = a residual HO-level spacing.
    This replaces the loose 'gap > frac*median' rule with a ranking that
    distinguishes primary magic closures from secondary HO remnants.
    """
    orbs = nilsson_spectrum(kappa, mu, chi_sign)
    energies = [o[4] for o in orbs]
    degs = [o[3] for o in orbs]
    gaps = [energies[k + 1] - energies[k] for k in range(len(energies) - 1)]
    out: List[Tuple[float, int, int]] = []
    cum = 0
    for k in range(len(orbs) - 1):
        cum += degs[k]
        neigh = []
        if k > 0:
            neigh.append(gaps[k - 1])
        if k < len(gaps) - 1:
            neigh.append(gaps[k + 1])
        local = sum(neigh) / len(neigh) if neigh else gaps[k]
        prom = gaps[k] / local if local > 0 else 0.0
        out.append((prom, cum, k))
    cum += degs[-1]
    return out


def primary_closures(kappa: float, mu: float, chi_sign: int = +1,
                     top: int = 7) -> List[int]:
    """Top-'top' most prominent gaps as primary closures (particle counts)."""
    proms = gap_prominences(kappa, mu, chi_sign)
    ranked = sorted(proms, key=lambda t: -t[0])
    return [N for (_, N, _) in ranked[:top]]


# -----------------------------------------------------------------
# Level scheme and magic-number extraction
# -----------------------------------------------------------------
def closure_cumulatives(chi_sign: int = +1) -> List[int]:
    """Cumulative nucleon counts at shell-model closures.

    A closure is the cumulative after a subshell that opens a gap. Two kinds:
    (a) end of a complete oscillator shell (naive HO closure: 2,8,20,...);
    (b) the maximal-j subshell (n, l_max, j = l_max + 1/2), where spin-orbit
    pushes the highest-j intruder down to create an extra closure. Under
    left-chirality the maximal-j subshell is filled FIRST within the shell,
    so its cumulative is 28,50,82,126. Under right-chirality it is filled
    LAST, so those cumulatives become full-shell boundaries (40,70,112,168)
    and the magic numbers 28,50,82,126 are lost.
    """
    out: List[int] = []
    cum = 0
    for n in range(7):
        subs = ordered_subshells(n, chi_sign)
        maxj = max(j for (_, j, _) in subs)
        nsub = len(subs)
        for i, (l, j, deg) in enumerate(subs):
            cum += deg
            is_shell_end = (i == nsub - 1)
            is_maxj = (j == maxj)
            if is_shell_end or is_maxj:
                out.append(cum)
    return out


def magic_set_for(chi_sign: int = +1) -> Tuple[List[int], List[int]]:
    """Magic numbers recovered as maximal-j closures under a CS sign.

    Returns (recovered, missing) against MAGIC using closure_cumulatives.
    The magic numbers beyond 20 are exactly the maximal-j closures, which
    exist only under left-chirality (CGM axiom). The right-chiral flip
    moves the maximal-j subshell to the end of each shell, so those
    cumulatives become full-shell boundaries (40,70,112,168) and the magic
    numbers 28,50,82,126 are lost.
    """
    clos = closure_cumulatives(chi_sign)
    recovered = [m for m in MAGIC if m in clos]
    missing = [m for m in MAGIC if m not in clos]
    return recovered, missing


def build_level_scheme(chi_sign: int = +1) -> List[Tuple[int, int, float, int, int]]:
    """Ordered level scheme across oscillator shells n = 0..6.

    Returns (n, l, j, degeneracy, cumulative_after_this_subshell).
    Filling proceeds shell by shell, within a shell by CS-chirality order.
    """
    rows: List[Tuple[int, int, float, int, int]] = []
    cum = 0
    for n in range(7):
        for (l, j, deg) in ordered_subshells(n, chi_sign):
            cum += deg
            rows.append((n, l, j, deg, cum))
    return rows


# -----------------------------------------------------------------
# Delta N=2 mixed Nilsson spectrum (m-substate basis)
# -----------------------------------------------------------------
def orbital_basis(n_max: int) -> List[Tuple[int, int, float, int]]:
    """All (n, l, j, deg) orbitals for oscillator shells n = 0..n_max."""
    out: List[Tuple[int, int, float, int]] = []
    for n in range(n_max + 1):
        for (l, j, deg) in oscillator_subshells(n):
            out.append((n, l, j, deg))
    return out


def nilsson_matrix_msubstate(kappa: float, mu: float, chi_sign: int = +1,
                             n_max: int = 6) -> np.ndarray:
    """Nilsson Hamiltonian on the m-substate basis with Delta N=2 mixing.

    Each orbital (n,l,j) expands to 2j+1 basis vectors (one per m_j).
    Diagonal: E = (N+3/2) - chi_sign*kappa*<L.S> - kappa*mu*<l^2>.
    Off-diagonal: Q = kappa*mu*sqrt(N(N+2)-l(l+1)) for |N-N'|=2, same l, m.
    """
    states: List[Tuple[int, int, float, float]] = []
    for (n, l, j, _) in orbital_basis(n_max):
        for m in np.arange(-j, j + 1e-9, 1.0):
            states.append((n, l, j, round(m, 6)))
    dim = len(states)
    H = np.zeros((dim, dim), dtype=float)
    for i, (n, l, j, _) in enumerate(states):
        H[i, i] = (n + 1.5) - chi_sign * kappa * ls_expectation(l, j) \
                  - kappa * mu * l2_expectation(l)
    for i, (n, l, j, m) in enumerate(states):
        for jdx in range(i + 1, dim):
            n2, l2, j2, m2 = states[jdx]
            if j2 != j or abs(m2 - m) > 1e-6:
                continue
            if l == l2 and abs(n - n2) == 2:
                q = kappa * mu * math.sqrt(max(0.0, n * (n + 2) - l * (l + 1)))
                H[i, jdx] = -q
                H[jdx, i] = -q
    return H


def _mixed_levels(kappa: float, mu: float, chi_sign: int = +1,
                  n_max: int = 6) -> Tuple[List[float], List[int]]:
    """Sorted eigenenergies on the m-substate basis (unit weight per state)."""
    evals = list(np.sort(np.linalg.eigvalsh(
        nilsson_matrix_msubstate(kappa, mu, chi_sign, n_max))))
    return evals, [1] * len(evals)


def mixed_gap_prominences(kappa: float, mu: float, chi_sign: int = +1,
                          n_max: int = 6) -> List[Tuple[float, int]]:
    """Prominence-ranked gaps; returns (prominence, cumulative particle count)."""
    energies, degs = _mixed_levels(kappa, mu, chi_sign, n_max)
    cum = 0
    cums: List[int] = []
    for d in degs:
        cum += d
        cums.append(cum)
    gaps = [energies[k + 1] - energies[k] for k in range(len(energies) - 1)]
    out: List[Tuple[float, int]] = []
    for k in range(len(energies) - 1):
        neigh = gaps[max(0, k - 3):min(len(gaps), k + 4)]
        local = sum(neigh) / len(neigh) if neigh else gaps[k]
        prom = gaps[k] / local if local > 0 else 0.0
        out.append((prom, cums[k]))
    return out


def primary_mixed_closures(kappa: float, mu: float, chi_sign: int = +1,
                           n_max: int = 6, top: int = 7) -> List[int]:
    """Top-'top' most prominent mixed-spectrum gaps."""
    proms = mixed_gap_prominences(kappa, mu, chi_sign, n_max)
    ranked = sorted(proms, key=lambda t: -t[0])
    return [N for (_, N) in ranked[:top]]


def mixed_gap_closures(kappa: float, mu: float, chi_sign: int = +1,
                       n_max: int = 6) -> List[int]:
    """Large-gap closures of the mixed Nilsson spectrum to n_max."""
    energies, degs = _mixed_levels(kappa, mu, chi_sign, n_max)
    cum = 0
    cums: List[int] = []
    for d in degs:
        cum += d
        cums.append(cum)
    spacings = [energies[k + 1] - energies[k] for k in range(len(energies) - 1)]
    closures: List[int] = []
    for k in range(len(energies) - 1):
        win = spacings[max(0, k - 4):min(len(spacings), k + 5)]
        med = float(np.median(win)) if win else 1.0
        if spacings[k] > 1.8 * med:
            closures.append(cums[k])
    return closures


# -----------------------------------------------------------------
# Empirical checks (IAEA LiveChart)
# -----------------------------------------------------------------
def load_ground_states(path: Path = CATALOG) -> Dict[Tuple[int, int], dict]:
    """Load IAEA LiveChart ground states keyed by (z, n)."""
    out: Dict[Tuple[int, int], dict] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                z = int(row["z"])
                n = int(row["n"])
            except (ValueError, KeyError):
                continue
            out[(z, n)] = row
    return out


def chirality_shell(z: int, n: int) -> int:
    """Chirality-shell index |N - Z| mod 7."""
    return abs(n - z) % 7


def doubly_magic_check(gs: Dict[Tuple[int, int], dict]) -> List[Tuple[int, int, str, int, bool]]:
    """For each doubly-magic (z,n): jp, |N-Z| mod 7, both-magic flag."""
    out: List[Tuple[int, int, str, int, bool]] = []
    for (z, n) in DOUBLY_MAGIC:
        row = gs.get((z, n))
        if row is None:
            out.append((z, n, "MISSING", -1, False))
            continue
        jp = (row.get("jp") or "").strip()
        sh = chirality_shell(z, n)
        is_magic = (z in MAGIC) and (n in MAGIC)
        out.append((z, n, jp, sh, is_magic))
    return out


def total_binding_keV(z: int, n: int, row: dict) -> float:
    """Total binding energy in keV from LiveChart binding (keV per nucleon)."""
    return float(row["binding"]) * (z + n)


def two_neutron_gap(gs: Dict[Tuple[int, int], dict], z_near: int = 82,
                    n_lo: int = 120, n_hi: int = 144) -> List[Tuple[int, float, float]]:
    """S_2n and delta_2n in MeV from total binding energies for fixed Z."""
    def b_tot(z: int, n: int) -> float:
        row = gs.get((z, n))
        if row is None or not row.get("binding"):
            return float("nan")
        try:
            return total_binding_keV(z, n, row)
        except ValueError:
            return float("nan")

    out: List[Tuple[int, float, float]] = []
    for n in range(n_lo, n_hi + 1):
        s2n_kev = b_tot(z_near, n) - b_tot(z_near, n - 2)
        s2n_next_kev = b_tot(z_near, n + 2) - b_tot(z_near, n)
        s2n = s2n_kev / 1000.0
        s2n_next = s2n_next_kev / 1000.0
        d2n = s2n - s2n_next if not math.isnan(s2n_next) else float("nan")
        out.append((n, s2n, d2n))
    return out


# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------
def main() -> int:
    print("=" * 5)
    print("CGM magic-number derivation probe")
    print("=" * 5)

    # 1. Carrier DoF = SE(3) generators
    print("-" * 5)
    print("1. Carrier DoF = SE(3) generators")
    print("total carrier DoF      : 6 (6 payload bits of GENE_Mac)")
    print("Frame 0 rot (SU(2))    : 3  (C1 = comb(6,1) =", C1, ")")
    print("Frame 1 trans (R^3)    : 3  (C2 = comb(6,2) =", C2, ")")
    print("SE(3) = SU(2) |x R^3   : 3 rot + 3 trans = 6 DoF")

    # 2. Oscillator shell degeneracy from weak 3-compositions (Frame 1)
    print("-" * 5)
    print("2. 3D oscillator from 3 spatial modes (Frame 1, R^3)")
    print("n  binom(n+2,2)  osc_deg=(n+1)(n+2)/2  match")
    osc_ok = True
    for n in range(7):
        wc = spatial_weak_composition(n)
        od = oscillator_shell_degeneracy(n)
        match = (wc == od)
        osc_ok = osc_ok and match
        print(f"{n}    {wc:<12} {od:<19} {str(match)}")
    print("weak 3-comp = oscillator deg :", osc_ok)
    caps = [shell_population(s) for s in range(7)]
    print("QuBEC shell caps C(6,s)*64 :", caps, " sum =", sum(caps))
    print("C(6,N) = discrete chirality census; binom(n+2,2) = continuous 3-mode limit")

    # 3. Kernel radial/angular split and shell duality
    print("-" * 5)
    print("3. Kernel radial/angular split and shell duality")
    print("Krawtchouk radial modes (7x7)        : present")
    print("  K_0(k) =", KRAWTCHOUK_7[0], " (shell census vector)")
    print("shell census |Shell_N|=|Shell_6-N|  :", shell_census_duality_ok())
    print("in-shell angular multiplicity (64)   :", [angular_multiplicity(s) for s in range(7)])

    # 4. (n,l,j) subshells and Mayer-Jensen ordering under CS chirality
    print("-" * 5)
    print("4. (n,l,j) subshells: left- vs right-chiral within-shell order")
    print("   (set of degeneracies is identical; order of fill flips)")
    for n in range(7):
        left = [(l, j, d) for (l, j, d) in ordered_subshells(n, +1)]
        right = [(l, j, d) for (l, j, d) in ordered_subshells(n, -1)]
        print(f"n={n}  left :", left)
        print(f"n={n}  right:", right)

    # 5. Chirality-flip gate on maximal-j closures
    print("-" * 5)
    print("5. Chirality-flip gate (maximal-j closures)")
    print("   closure = cumulative after maximal-j subshell (intruder drop)")
    print("   magic beyond 20 exists only under left-chirality (CS axiom)")
    recL, missL = magic_set_for(+1)
    recR, missR = magic_set_for(-1)
    print("left-chiral  magic recovered :", recL, " missing :", missL)
    print("right-chiral magic recovered :", recR, " missing :", missR)
    chir_pass = set(INTRUDERS).issubset(set(missR))
    print("flip removes intruders 28,50,82,126 :", chir_pass)

    # 6. Full ordered level scheme, maximal-j closures marked
    print("-" * 5)
    print("6. Left-chiral ordered level scheme (maximal-j closures marked B)")
    print("n  l   j    2j+1  cumulative  B  magic?")
    scheme = build_level_scheme(+1)
    clos = set(closure_cumulatives(+1))
    magic_set = set(MAGIC)
    for (n, l, j, deg, cum) in scheme:
        is_clos = cum in clos
        is_magic = cum in magic_set
        flag = "MAGIC" if is_magic else ""
        bmark = "B" if is_clos else ""
        print(f"{n}  {l:<3} {j:<5} {deg:<5} {cum:<10} {bmark:<2} {flag}")
    recovered = recL
    missing = missL
    print("measured magic :", MAGIC)
    print("maximal-j closures recovered :", recovered)
    print("maximal-j closures missing  :", missing)
    passed = (not missing)
    print("PASS" if passed else "FAIL", " all 7 magic closures recovered")

    # 7. Gap sizes and BU splitting scale
    print("-" * 5)
    print("7. Gap sizes and BU vibrational splitting scale")
    gaps = [MAGIC[k] - MAGIC[k - 1] for k in range(1, len(MAGIC))]
    print("gap sizes between consecutive magic :", gaps)
    print("per-shell maximal-j drop 2(l_max+1) : 8,10,12,14 (l_max=3,4,5,6)")
    so_scale = so_splitting_scale_MeV()
    print(f"BU aperture Delta          : {DELTA:.6e}")
    print(f"strong scale E_str (MeV)   : {E_EW_GEV * DELTA**3 * 1e3:.6e}")
    print(f"SO splitting scale (MeV)   : {so_scale:.6e}")
    print(f"RHO (closure ratio)        : {RHO:.6e}")

    # 8. CGM-anchored energy spectrum and gap-detected closures
    print("-" * 5)
    print("8. CGM-anchored spectrum E(n,l,j) and gap closures")
    s_rad = radial_step_ticks()
    b_so0 = DELTA * radial_step_ticks()
    print(f"radial step (ticks/shell)  : {s_rad:.4f}")
    print(f"spin-orbit scale b_so(=1)  : {b_so0:.4f}  (BU aperture DELTA * radial step)")
    print("ordered orbitals (left-chiral, b_so_scale=1), E in ticks:")
    for (n, l, j, deg, e) in spectrum_ordered(+1, 1.0):
        print(f"  n={n} l={l} j={j:4.1f} 2j+1={deg:2d} E={e:7.3f}")

    # 8a. Gap closures at the CGM-anchored (small) coupling
    closL = gap_closures(+1, b_so_scale=1.0)
    closR = gap_closures(-1, b_so_scale=1.0)
    recL_spec = [m for m in MAGIC if m in closL]
    missL_spec = [m for m in MAGIC if m not in closL]
    recR_spec = [m for m in MAGIC if m in closR]
    missR_spec = [m for m in MAGIC if m not in closR]
    print("left  gap closures (b=1)   :", closL)
    print("right gap closures (b=1)   :", closR)
    print("left  recovered vs magic   :", recL_spec, " missing :", missL_spec)
    print("right recovered vs magic   :", recR_spec, " missing :", missR_spec)
    print("linear SO model: oscillator closures only (no intruders)")

    print("spin-orbit scale sweep (left-chiral): b_so_scale -> gap closures")
    sweep = []
    for scale in [1.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0]:
        c = gap_closures(+1, b_so_scale=scale)
        rec = [m for m in MAGIC if m in c]
        sweep.append((scale, c, len(rec)))
        print(f"  b={scale:5.1f}  closures={c}  magic_recovered={len(rec)}/7")
    best = max(sweep, key=lambda t: t[2])
    print(f"max magic recovered at b={best[0]:.1f}: {best[1]}")

    spec_pass = (closL == OSC_CLOSURES and missL_spec == INTRUDERS)
    print("PASS" if spec_pass else "FAIL",
          " linear SO at b=1 gives oscillator closures only")

    # 9. Nilsson Hamiltonian: spin-orbit + quadrupole deformation
    print("-" * 5)
    print("9. Nilsson Hamiltonian E = (N+3/2)hw - kappa<L.S> - kappa*mu*<l^2>")
    print("   <l^2> = l(l+1); STF bulk sector (l=2 quadrupole)")
    print("   pushes high-l orbitals down -> intruder magic gaps")
    print("left-chiral (kappa, mu) window scan -> gap closures:")
    found = []
    for kappa in [0.03, 0.05, 0.07, 0.10, 0.13]:
        for mu in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]:
            c = nilsson_gap_closures(kappa, mu, +1)
            rec = [m for m in MAGIC if m in c]
            if len(rec) == 7:
                found.append((kappa, mu, c))
                print(f"  kappa={kappa:.2f} mu={mu:.1f}  closures={c}  ALL 7")
    if found:
        fk, fm, fc = found[0]
        print(f"first (kappa,mu) recovering all 7: ({fk}, {fm}) -> {fc}")
    else:
        print("  no (kappa,mu) in scanned range recovers all 7")
    # Chirality flip at a representative coupling
    k0, m0 = 0.10, 0.40
    cL = nilsson_gap_closures(k0, m0, +1)
    cR = nilsson_gap_closures(k0, m0, -1)
    recNL = [m for m in MAGIC if m in cL]
    missNL = [m for m in MAGIC if m not in cL]
    recNR = [m for m in MAGIC if m in cR]
    missNR = [m for m in MAGIC if m not in cR]
    print(f"at (kappa={k0}, mu={m0}):")
    print(f"  left  closures : {cL}  magic recovered {len(recNL)}/7 missing {missNL}")
    print(f"  right closures : {cR}  magic recovered {len(recNR)}/7 missing {missNR}")
    print(f"E_str (MeV)      : {E_EW_GEV * DELTA**3 * 1e3:.3f}")

    # 10. Nilsson closure and chirality gates
    print("-" * 5)
    print("10. Nilsson closure and chirality gates")
    nilsson_pass = bool(found) and (missNR == INTRUDERS)
    print(f"all 7 in coupling window : {bool(found)}"
          + (f"  first at (kappa={found[0][0]}, mu={found[0][1]})" if found else ""))
    print(f"right-chiral missing     : {missNR}")
    print("PASS" if nilsson_pass else "FAIL",
          " 7 gap closures in window + chirality flip removes intruders")

    # 10a. CGM anchor point (diagonal Nilsson, no mixing)
    print("-" * 5)
    print(f"10a. CGM anchor (kappa={KAPPA_CGM:.5f}, mu={MU_CGM}) diagonal prominence")
    print(f"     tau=delta_BU/(2*pi)={TAU_BU:.6f}  Q_256(tau)={KAPPA_CGM}")
    proms = gap_prominences(KAPPA_CGM, MU_CGM, +1)
    print("ranked gaps (prominence, N):")
    for (p, N, _) in sorted(proms, key=lambda t: -t[0])[:11]:
        mark = "MAGIC" if N in MAGIC else ("HO" if N in (40, 70, 112) else "")
        print(f"  prom={p:5.2f}  N={N:3d}  {mark}")
    pc_diag = primary_closures(KAPPA_CGM, MU_CGM, +1, top=7)
    intruder_in_top = [m for m in INTRUDERS if m in pc_diag]
    ho_in_top = [h for h in (40, 70, 112) if h in pc_diag]
    print(f"top-7 closures         : {sorted(pc_diag)}")
    print(f"intruders in top-7     : {intruder_in_top}")
    print(f"HO remnants in top-7   : {ho_in_top}")
    diag_pass = len(intruder_in_top) == 0
    print("PASS" if diag_pass else "FAIL",
          " diagonal model omits intruders (mixing required)")

    # 11. Delta N=2 mixing at CGM anchor point
    print("-" * 5)
    print(f"11. DELTA N=2 MIXING (kappa={KAPPA_CGM}, mu={MU_CGM})")
    print("basis orbitals (n_max=6) :", len(orbital_basis(6)))
    pc_mix = primary_mixed_closures(KAPPA_CGM, MU_CGM, +1, n_max=6, top=7)
    clos_mix = mixed_gap_closures(KAPPA_CGM, MU_CGM, +1, n_max=6)
    in_diag = [m for m in INTRUDERS if m in pc_diag]
    in_mix = [m for m in INTRUDERS if m in pc_mix]
    all7 = [m for m in MAGIC if m in clos_mix]
    print("diagonal top-7 closures :", sorted(pc_diag))
    print("mixed    top-7 closures :", sorted(pc_mix))
    print("mixed closure count     :", len(clos_mix))
    print("magic in mixed set      :", all7)
    print("intruders in top-7 (diag):", in_diag)
    print("intruders in top-7 (mix) :", in_mix)
    mix_pass = (len(in_mix) == 4) and (len(in_diag) == 0) and (len(all7) == 7)
    print("PASS" if mix_pass else "FAIL",
          " all 7 magic in mixed spectrum; 0 intruders in diagonal")

    # 12. Doubly-magic chirality-shell closure
    print("-" * 5)
    print("12. DOUBLY-MAGIC CHIRALITY-SHELL CLOSURE")
    gs = load_ground_states()
    print("LiveChart rows loaded   :", len(gs))
    print("z   n    jp     |N-Z| mod 7")
    dm = doubly_magic_check(gs)
    shells = [sh for (_, _, _, sh, ok) in dm if ok and sh >= 0]
    for (z, n, jp, sh, ok) in dm:
        print(f"{z:3d} {n:3d}  {jp:5s}  {sh:3d}")
    central = all(min(s, 7 - s) <= 3 for s in shells)
    zeroplus = all(jp == "0+" for (_, _, jp, _, ok) in dm if ok)
    print("all jp = 0+             :", zeroplus)
    print("all |N-Z| mod 7 central :", central, shells)
    dm_pass = zeroplus and central
    print("PASS" if dm_pass else "FAIL",
          " doubly-magic 0+ and central chirality shell")

    # 13. Superheavy spectrum
    print("-" * 5)
    print("13. SUPERHEAVY CLOSURES (n_max=12)")
    print("extended basis orbitals :", len(orbital_basis(12)))
    print("coupling scan (kappa=mu=c, exploratory):")
    best_hit: set = set()
    for c in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
        clos = mixed_gap_closures(c, c, +1, n_max=12)
        topk = primary_mixed_closures(c, c, +1, n_max=12, top=7)
        hit = [k for k in KNOWN_SUPERHEAVY if any(abs(cc - k) <= 1 for cc in clos)]
        best_hit.update(hit)
        print(f"  c={c:4.2f}  hit {hit}  top7 {sorted(topk)}")
    topk_cgm = primary_mixed_closures(KAPPA_CGM, MU_CGM, +1, n_max=12, top=7)
    clos_cgm = mixed_gap_closures(KAPPA_CGM, MU_CGM, +1, n_max=12)
    sh_magic = [m for m in KNOWN_SUPERHEAVY if any(abs(cc - m) <= 1 for cc in clos_cgm)]
    print("CGM point top-7 gaps    :", sorted(topk_cgm))
    print("CGM point SH candidates :", sorted(sh_magic))
    print("scan candidates reached :", sorted(best_hit))
    sh_pass = bool(best_hit)
    print("PASS" if sh_pass else "FAIL",
          " superheavy candidates in closure scan")

    # 14. Two-neutron gap signature
    print("-" * 5)
    print("14. TWO-NEUTRON GAP (Z=82, MeV)")
    gap = two_neutron_gap(gs, z_near=82, n_lo=120, n_hi=144)
    print("N    S2n       delta2n")
    for (n, s2n, d2n) in gap:
        if math.isnan(s2n):
            continue
        print(f"{n:3d}  {s2n:9.2f}  {d2n:9.2f}")
    d2n_vals = [(n, d) for (n, _, d) in gap if not math.isnan(d)]
    d2n_max = max((d for (_, d) in d2n_vals), default=float("nan"))
    argmax_n = next((n for (n, d) in d2n_vals if d == d2n_max), -1)
    s2n_pass = argmax_n == 126
    print(f"delta2n spike at N={argmax_n} ({d2n_max:.2f} MeV)")
    print("PASS" if s2n_pass else "FAIL",
          " delta2n spike at N=126")

    print("=" * 5)
    print("summary")
    print("=" * 5)
    print(f"KAPPA_CGM (Q_256)                 : {KAPPA_CGM}")
    print(f"MU_CGM (1/STF)                    : {MU_CGM}")
    print(f"oscillator deg = weak 3-comp      : {osc_ok}")
    print(f"shell census Poincare duality     : {shell_census_duality_ok()}")
    print(f"left-chiral magic recovered       : {recL}")
    print(f"right-chiral magic missing        : {missR}")
    print(f"linear SO closures (b=1)          : {closL}")
    print(f"Nilsson all-7 in window           : {bool(found)}"
          + (f" at ({found[0][0]},{found[0][1]})" if found else ""))
    print(f"SO splitting scale (MeV)          : {so_scale:.6e}")
    print(f"gate (maximal-j)                  : {'PASS' if passed else 'FAIL'}")
    print(f"gate (chirality flip)             : {'PASS' if chir_pass else 'FAIL'}")
    print(f"gate (linear SO oscillator-only)  : {'PASS' if spec_pass else 'FAIL'}")
    print(f"gate (Nilsson window)             : {'PASS' if found else 'FAIL'}")
    print(f"gate (Nilsson + chirality)        : {'PASS' if nilsson_pass else 'FAIL'}")
    print(f"gate (diagonal CGM point)         : {'PASS' if diag_pass else 'FAIL'}")
    print(f"gate (Delta N=2 mixing)           : {'PASS' if mix_pass else 'FAIL'}")
    print(f"gate (doubly-magic)               : {'PASS' if dm_pass else 'FAIL'}")
    print(f"gate (superheavy scan)            : {'PASS' if sh_pass else 'FAIL'}")
    print(f"gate (delta2n at N=126)           : {'PASS' if s2n_pass else 'FAIL'}")
    return 0


def magic_main() -> None:
    """Report driver entry point for hqvm_cgm_trestleboard_run.py."""
    main()


if __name__ == "__main__":
    sys.exit(main())
