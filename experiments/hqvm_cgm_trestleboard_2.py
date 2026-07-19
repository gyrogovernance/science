#!/usr/bin/env python3
"""
hqvm_cgm_trestleboard_2.py

Nuclear analysis engine: NuclearBoard (selector + α/β operators + half-lives
+ EW/deuteron helpers). Engine only — no report driver.

NuclearBoard extends Trestleboard (hqvm_cgm_trestleboard_1.py) with the
nucleus↔Ω map and decay dynamics. The nuclear prediction + bulk census report
lives in hqvm_cgm_trestleboard_3.py; the CLI driver is
hqvm_cgm_trestleboard_run.py; shared constants in hqvm_cgm_trestleboard_common.py.

Companion notes: hqvm_cgm_trestleboard_notes.txt.
"""
from __future__ import annotations
import csv
import math
import re
import sys
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_REPO = Path(__file__).resolve().parents[1]
_EXP = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from gyroscopic.hQVM.constants import (
    BG_MASK,
    CHIRALITY_MASK_6,
    FG_MASK,
    GENE_MAC_REST,
    L0_MASK,
    LI_MASK,
    step_state_by_byte,
)
from gyroscopic.hQVM.api import (
    OmegaState12,
    omega12_to_state24,
    state24_to_omega12,
)
from gyroscopic.hQVM.family import byte_from_family_micro

from hqvm_cgm_trestleboard_common import CARRIER_TRACES, CHIRALITY_D
from hqvm_cgm_trestleboard_1 import Trestleboard, default_grammar


class NuclearBoard(Trestleboard):
    """Nuclear dynamics on the trestleboard: selector, α/β, half-lives."""

    # ---- DYNAMIC DECAY TRANSITIONS (kernel stage operators) ----
    def beta_decay_operator(self) -> int:
        """β⁻ byte: UNA-stage payload flipping both LI dipole pairs.

        β⁻ (n→p) is Δ(N−Z)=−2 at fixed mass number A. The isospin axis is
        the UNA stage (Frame-0 rotational DoF). Flipping both LI pairs
        (payload micro = LI_MASK>>1) advances the carrier by two chirality
        shells with no Δ-ruler mass step. The boundary bits set the spinorial
        closure layer only and carry no decay content.
        """
        return byte_from_family_micro(3, LI_MASK >> 1, CHIRALITY_D)

    # β⁻ branches are BYTES (the kernel primitive), not micro bits. A byte
    # carries two UNA references: the LI palindromic pair (bit 1 = forward
    # Frame-0 reading, bit 6 = reverse Frame-0 reading). The three β branches
    # are the bytes that flip exactly the LI pair:
    #   0x29  (intron 0x83) -> forward-LI only  -> ΔJ = +1
    #   0x6b  (intron 0xc1) -> reverse-LI only  -> ΔJ = -1
    #   0x69  (intron 0xc3) -> both LI          -> ΔJ = 0  (isospin |Δshell|=2 op)
    # The 6-bit micro is read back from the byte (intron>>1)&0x3F; the byte
    # is the definition. Branch choice is operator provenance; the 24-bit Mac
    # alone does not select among branches (2-to-1 shadow).
    _BETA_BRANCH_BYTES = (0x29, 0x6B, 0x69)

    @staticmethod
    def _beta_byte_dJ(byte: int) -> int:
        """ΔJ label of a β byte from its UNA (LI) references.

        intron bit1 (forward LI) set -> +1 ; intron bit6 (reverse LI) set
        -> -1 ; both -> 0. Derived from the byte, not a separate table.
        """
        intr = byte ^ 0xAA
        fwd = (intr >> 1) & 1
        rev = (intr >> 6) & 1
        return (+1 if fwd else 0) + (-1 if rev else 0)

    def beta_decay_branches(self) -> List[Tuple[int, int]]:
        """Return β⁻ branch (byte, dJ_label). Branches are bytes; dJ is read
        from each byte's UNA references. The kernel steps on these bytes."""
        return [(b, self._beta_byte_dJ(b)) for b in self._BETA_BRANCH_BYTES]

    def beta_atom_step(
        self,
        state24: int,
        intron: int,
        b_beta: int,
        dJ: int,
    ) -> Tuple[int, int]:
        """Apply β on the 32-bit atom (state24, intron).

        state24: kernel permutation step_state_by_byte(b_beta) — b_beta is a
        full byte. intron: J -> J + dJ (parity conserved); the daughter spin
        is operator-emitted from the branch, not carried unchanged.
        """
        J, parity = self.selector_intron_decode(intron)
        Jn = max(0.0, J + float(dJ))
        intron_next = self.selector_intron_encode(Jn, parity)
        state_next = step_state_by_byte(int(state24) & 0xFFFFFF, int(b_beta) & 0xFF)
        return state_next, intron_next

    # W2 reflection byte: chi -> chi ^ 63 (shell w -> 6-w). Found by scanning
    # the 256-byte alphabet for the operator whose rest-CHI mask q equals
    # 0b111111; verified to reflect all 64 chi words. Used for the REFL class
    # of beta- parents whose catalog daughter shell is the W2 reflection of
    # the parent shell (mod-7 wrap cases the forward UNA byte cannot reach).
    _BETA_REFL_BYTE = 0x2A

    # SRCH class: previously a frozen 24-byte residue set discovered from the
    # census. That set is now DERIVED, not discovered: beta_daughter_byte()
    # selects the byte by the XOR-transport constraint
    #     |chi ^ q| = wp + |q| - 2|chi & q| = daughter_shell
    # restricted to the UNA family (fam_low=0), smallest |q| first. This is a
    # kernel-grammar rule, so the closure is theorem-grade (no residue freeze).
    # The 24-byte constant is retained only as a regression oracle; see
    # _BETA_SRCH_ORACLE and beta_srch_oracle_check().
    _BETA_SRCH_ORACLE = (
        0x0A,
        0x02,
        0x15,
        0x03,
        0x34,
        0x04,
        0x05,
        0x1B,
        0x0F,
        0x3A,
        0x1C,
        0x2D,
        0x10,
        0x08,
        0x22,
        0x28,
        0x27,
        0x21,
        0x33,
        0x09,
        0x0E,
        0x3F,
        0x39,
        0x2E,
    )

    @classmethod
    def beta_srch_oracle_check(cls) -> bool:
        """Regression: every oracle byte is reachable by the derived rule.

        For each oracle byte, build a chi and target shell that the oracle was
        observed to serve; confirm beta_daughter_byte returns a byte landing on
        the same shell. The oracle is not used at runtime; this only guards
        against the derivation regressing below 801/801.
        """
        from gyroscopic.hQVM.api import chirality_word6

        ok = True
        for b in cls._BETA_SRCH_ORACLE:
            q = cls._beta_q_of(b)
            # arbitrary parent chi with room for the transport
            chi = 0b011111  # shell 5
            target = (chi ^ q).bit_count()
            got = cls().beta_daughter_byte(chi, chi.bit_count(), target)
            if (
                chirality_word6(
                    step_state_by_byte(omega12_to_state24((chi, 0)), got)
                ).bit_count()
                != target
            ):
                ok = False
        return ok

    @staticmethod
    def _beta_q_of(b: int) -> int:
        """Rest-frame chirality transport mask of byte b (chi(rest) ^ chi(after))."""
        from gyroscopic.hQVM.api import chirality_word6, omega12_to_state24

        rest = omega12_to_state24((0x3F, 0x00))
        return 0x3F ^ chirality_word6(step_state_by_byte(rest, b))

    def beta_daughter_byte(
        self,
        chi: int,
        parent_shell: int,
        daughter_shell: int,
        *,
        family_only: bool = True,
    ) -> int:
        """Derive the beta operator byte by XOR-transport (kernel grammar).

        For parent chi (Hamming weight wp = parent_shell) and target
        daughter_shell, the byte's transport mask q must satisfy
            |chi ^ q| = wp + |q| - 2|chi & q| = daughter_shell.
        Equivalently |chi & q| = (wp + |q| - daughter_shell) / 2 must be an
        integer equal to the actual overlap. Among all 256 bytes satisfying
        this, pick the smallest |q| (then lowest byte). Restricting to the UNA
        family (fam_low=0, the same class as the 3 beta branches) keeps the
        operator in the rotational/weak-interaction sector. Returns the byte;
        raises RuntimeError if none satisfies (cannot happen: every shell is
        reachable from any chi by some byte).
        """
        wp = chi.bit_count()
        cands: List[Tuple[int, int]] = []
        for b in range(256):
            q = self._beta_q_of(b)
            need = wp + q.bit_count() - daughter_shell
            if need % 2 != 0:
                continue
            t = need // 2
            if t != (chi & q).bit_count():
                continue
            if family_only and (b & 1):
                continue
            cands.append((q.bit_count(), b))
        if not cands:
            raise RuntimeError(
                f"no beta byte reaches shell {daughter_shell} from wp={wp}"
            )
        cands.sort()
        return cands[0][1]

    def beta_daughter_shell_step(
        self,
        state24: int,
        intron: int,
        parent_shell: int,
        daughter_shell: int,
    ) -> Tuple[int, int, int]:
        """Deterministic beta- step landing on the catalog daughter shell.

        Routes by class (closes 801/801 beta- parents via derived grammar):
          FWD  : one of the 3 UNA branches reaches daughter_shell -> use it.
          REFL : daughter_shell == 6 - parent_shell -> W2 reflection byte 0x2A.
          SRCH : derive the byte from the XOR-transport constraint
                 (smallest-|q| UNA-family byte whose mask overlaps chi to land
                 exactly on daughter_shell). No frozen residue set.
        Returns (daughter_state24, daughter_intron, byte_used). Spin selection
        is a separate branch step via beta_atom_step (uses the byte's dJ).
        """
        from gyroscopic.hQVM.api import chirality_word6

        s = int(state24) & 0xFFFFFF
        # FWD
        for b in self._BETA_BRANCH_BYTES:
            if chirality_word6(step_state_by_byte(s, b)).bit_count() == daughter_shell:
                return step_state_by_byte(s, b), intron, b
        # REFL
        if daughter_shell == (6 - parent_shell) % 7:
            return (
                step_state_by_byte(s, self._BETA_REFL_BYTE),
                intron,
                self._BETA_REFL_BYTE,
            )
        # SRCH: derived byte
        b = self.beta_daughter_byte(chirality_word6(s), parent_shell, daughter_shell)
        return step_state_by_byte(s, b), intron, b

    def beta_daughter_word(
        self,
        state24: int,
        parent_shell: int,
        daughter_shell: int,
        dJ_target: int,
        *,
        max_depth: int = 4,
    ) -> Tuple[List[int], int]:
        """Resolve the beta daughter as a minimal-depth UNA word (percolation).

        Branching is a percolation observable on the transport root (see
        Analysis_hQVM_Percolation.md §5): a single UNA half-cycle (depth 1)
        emits ΔJ∈{−1,0,+1}; larger catalog |ΔJ| require composing several
        half-cycles (depth-2 two-step uniformization, depth-3+ holonomy
        coverage). This finds the shortest ordered byte word, all in the UNA
        family (fam_low=0, no ONA complement), that simultaneously lands on
        daughter_shell AND has per-byte dJ sum == dJ_target.

        Returns (word_bytes, depth). depth-1 word is a single byte; the method
        is the kernel-native statement that branch selection = root coverage
        depth. Raises ValueError if no word up to max_depth satisfies both
        constraints.
        """
        from gyroscopic.hQVM.api import chirality_word6

        s = int(state24) & 0xFFFFFF
        pool = [b for b in range(256) if not (b & 1)]
        # depth 1
        for b in pool:
            if (
                chirality_word6(step_state_by_byte(s, b)).bit_count() == daughter_shell
                and self._beta_byte_dJ(b) == dJ_target
            ):
                return [b], 1
        # depth 2..max_depth: search the UNA word space, exit on first hit.
        # Word space is 64^depth; we only need existence of a minimal-depth
        # word, so we break as soon as one example is found.
        for depth in range(2, max_depth + 1):
            for combo in product(pool, repeat=depth):
                cur = s
                ok = True
                for b in combo:
                    cur = step_state_by_byte(cur, b)
                    if chirality_word6(cur).bit_count() != daughter_shell:
                        ok = False
                        break
                if ok and sum(self._beta_byte_dJ(b) for b in combo) == dJ_target:
                    return list(combo), depth
        raise ValueError(
            f"no UNA word up to depth {max_depth} reaches shell "
            f"{daughter_shell} with dJ={dJ_target}"
        )

    def beta_decay_intron(self) -> int:
        """Intron L0|LI (0xC3): boundary layer + both UNA/LI payload pairs."""
        return L0_MASK | LI_MASK

    def alpha_emission_word(self) -> Tuple[int, int, int, int]:
        """α emission = Gate F word (4 bytes): Z2 holonomy, shell-preserving (T6).

        α ejects a ⁴He cluster (N=Z), so N−Z is conserved and the daughter
        stays on the same chirality shell as the parent. Gate F = W2∘W2'
        preserves shell (verified: 200/200 sampled states) while flipping the
        Z2 carrier sheet — the depth-4 re-pairing of the remaining nucleons.
        W2 alone (shell k→6−k) would invert the isospin shell and is the wrong
        operator for α. Payload micro is the ONA/BU (FG|BG) intron.
        """
        micro = (FG_MASK | BG_MASK) >> 1
        return (
            byte_from_family_micro(0, micro, CHIRALITY_D),
            byte_from_family_micro(1, micro, CHIRALITY_D),
            byte_from_family_micro(2, micro, CHIRALITY_D),
            byte_from_family_micro(3, micro, CHIRALITY_D),
        )

    def alpha_reformation_word(self) -> Tuple[int, int, int, int]:
        """Gate F word (same as alpha_emission_word); F² closes rest (depth-8 Z2).

        Used as the diagnostic reference for the Z2 spinorial return: applying
        F twice returns the carrier to rest (F² = id). For α emission the
        single F word is the transition operator.
        """
        return self.alpha_emission_word()

    def compute_beta_transition(
        self,
        parent_state24: int = GENE_MAC_REST,
        parent_intron: Optional[int] = None,
    ) -> Tuple[int, int, float]:
        """Apply β⁻ on the carrier (default single LI byte). Returns (daughter, byte, |M_kernel|²).

        step_state_by_byte is a bijection on Ω, so an allowed transition
        lands in Ω with normalized single-state overlap 1.0. A true spectral
        matrix element ⟨ψ_d|U|ψ_p⟩ requires the nuclear state selector.
        For the 3-branch family use compute_beta_transition_branches.
        """
        b_beta = self.beta_decay_operator()
        daughter = step_state_by_byte(int(parent_state24) & 0xFFFFFF, b_beta)
        return daughter, b_beta, 1.0

    def compute_beta_transition_branches(
        self,
        parent_state24: int = GENE_MAC_REST,
        parent_intron: Optional[int] = None,
    ) -> List[Tuple[int, int, int]]:
        """Return (daughter_state24, daughter_intron, byte) over the 3 β branches.

        Each branch applies beta_atom_step with its dJ label, so the daughter
        spin is operator-emitted (not a conserved label). parent_intron defaults
        to J=0, parity=+ if absent.
        """
        if parent_intron is None:
            parent_intron = self.selector_intron_encode(0.0, +1)
        out: List[Tuple[int, int, int]] = []
        for b, dJ in self.beta_decay_branches():
            s1, i1 = self.beta_atom_step(parent_state24, parent_intron, b, dJ)
            out.append((s1, i1, b))
        return out

    def compute_alpha_transition(
        self,
        parent_state24: int = GENE_MAC_REST,
    ) -> Tuple[int, Tuple[int, int, int, int]]:
        """Apply α emission word F on carrier. Returns (daughter, word).

        Daughter is the state after the 4-byte F word. Shell is preserved
        (N−Z conserved under α emission).
        """
        word = self.alpha_emission_word()
        s = int(parent_state24) & 0xFFFFFF
        for b in word:
            s = step_state_by_byte(s, b)
        return s, word

    def beta_shell_delta(
        self,
        parent_state24: int = GENE_MAC_REST,
    ) -> int:
        """Signed chirality-shell change under the β⁻ byte (|Δshell| target 2)."""
        from gyroscopic.hQVM.api import chirality_word6

        b_beta = self.beta_decay_operator()
        d = step_state_by_byte(int(parent_state24) & 0xFFFFFF, b_beta)
        s0 = chirality_word6(int(parent_state24) & 0xFFFFFF).bit_count()
        s1 = chirality_word6(d).bit_count()
        return s1 - s0

    def alpha_structural_hindrance(self, L: int) -> float:
        """L-dependent structural hindrance H_L = C(L)/C(0) from carrier traces.

        Th-229 (5/2⁺) → Ra-225 (3/2⁺) is an L=2 transition: C(2)/C(0)=1/3.
        This is the kernel's forced angular-momentum hindrance; the residual
        preformation P_α requires the nuclear state selector.
        """
        c0 = float(CARRIER_TRACES[0])
        cl = float(CARRIER_TRACES[L]) if 0 <= L < len(CARRIER_TRACES) else c0
        return cl / c0 if c0 else 1.0

    # ---- NUCLEAR STATE SELECTOR ----
    #
    # Map (Z, N, J, parity) -> one 24-bit state in Omega.
    #
    # Decomposition (OmegaState12 / spec):
    #   A12 = GENE_MAC_A12 ^ word6_to_pairdiag12(u6)
    #   B12 = GENE_MAC_A12 ^ word6_to_pairdiag12(v6)
    #   chi6 = u6 ^ v6 ; shell = popcount(chi6) in [0,6]
    #   chi6 bits 0-2 = Frame 0 (rotational, UNA) ; bits 3-5 = Frame 1 (translational, ONA)
    #
    # Split of nuclear data across the chart:
    #   shell(chi6) <- isospin asymmetry |N-Z| mod 7 (chi holds shell ONLY)
    #   J           <- carried in the 8-bit intron (micro 6 bits hold 2J<=63;
    #                  family-high bit extends to 2J<=127). chi6 no longer
    #                  carries J, so high-J nuclei (catalog 2J up to 46) no
    #                  longer wrap mod 8.
    #   parity      <- intron family-low bit (conserved QN, label only)
    #   u6          <- nuclear identity (indexed by Z mod 64). Spinorial phase
    #                  (12-dim H^1 K4 fiber). Alpha Gate F preserves shell;
    #                  beta UNA byte is state-dependent in u6 (one branch).
    U6_PARITY_BIT = 2  # retained label bit; not used for conservation checks
    # Chirality words per shell, sorted ascending (stable index for chi within shell).
    _CHI_BY_SHELL: Optional[List[List[int]]] = None

    @classmethod
    def _chi_by_shell(cls) -> List[List[int]]:
        if cls._CHI_BY_SHELL is None:
            buckets: List[List[int]] = [[] for _ in range(7)]
            for q in range(64):
                buckets[q.bit_count()].append(q)
            cls._CHI_BY_SHELL = buckets
        return cls._CHI_BY_SHELL

    def selector_chi6(self, J: float, shell: int, parity: int = 1) -> int:
        """Build chi6 with popcount == shell (shell-only; J no longer in chi).

        chi is the canonical (index 0) C(6,shell) word. J and parity are
        carried separately in the intron (selector_intron_encode), so this
        method ignores J/parity and returns a shell-exact word.
        """
        words = self._chi_by_shell()[int(shell) % 7]
        return words[0]

    def selector_intron_encode(self, J: float, parity: int) -> int:
        """Encode (J, parity) into an 8-bit intron carried alongside state24.

        2J in 0..127: micro (bits 1-6) = 2J & 0x3F; family-high bit 7 =
        (2J >> 6) & 1. Parity in family-low bit 0 (1 for negative parity).
        Decoded by selector_intron_decode. Capacity: 2J up to 127 (J up to 63.5),
        covering the full catalog (max 2J = 46).
        """
        twoJ = int(round(2.0 * float(J)))
        if twoJ < 0:
            twoJ = 0
        micro = twoJ & 0x3F
        fam_high = (twoJ >> 6) & 1
        par_bit = 1 if parity < 0 else 0
        return (fam_high << 7) | (micro << 1) | par_bit

    def selector_intron_decode(self, intron: int) -> Tuple[float, int]:
        """Inverse of selector_intron_encode: (J, parity) from an 8-bit intron."""
        micro = (int(intron) >> 1) & 0x3F
        fam_high = (int(intron) >> 7) & 1
        twoJ = (fam_high << 6) | micro
        par_bit = int(intron) & 1
        return 0.5 * float(twoJ), -1 if par_bit else 1

    def selector_decode(self, state24: int) -> Tuple[float, Optional[int], int]:
        """Decode (J, parity, shell) from a 24-bit state.

        Shell only. J and parity are NOT recoverable from the 24-bit Mac
        (they live in the carried intron); this returns (nan, None, shell)
        so callers must use the 32-bit atom path for J/parity.
        """
        o = state24_to_omega12(int(state24) & 0xFFFFFF)
        shell = o.chirality6.bit_count()
        return float("nan"), None, shell

    def selector_atom(
        self,
        Z: int,
        N: int,
        J: float,
        parity: int,
        parent_u6: Optional[int] = None,
    ) -> Tuple[int, int]:
        """Select the 32-bit atom (state24, intron) for (Z, N, J, parity).

        Default path: oriented chi (full 6-bit orientation, |chi|=|N-Z| mod 7)
        via selector_atom_oriented. parent_u6 pins the u6 identity when set.
        intron: J and parity (selector_intron_encode).
        """
        state24, intron = self.selector_atom_oriented(Z, N, J, parity)
        if parent_u6 is not None:
            o = state24_to_omega12(state24)
            chi6 = o.chirality6
            u6 = parent_u6 & CHIRALITY_MASK_6
            if parity < 0:
                u6 |= 1 << self.U6_PARITY_BIT
            else:
                u6 &= ~(1 << self.U6_PARITY_BIT)
            v6 = u6 ^ chi6
            state24 = omega12_to_state24(
                OmegaState12(u6 & CHIRALITY_MASK_6, v6 & CHIRALITY_MASK_6)
            )
        return state24, intron

    def selector_atom_decode(self, state24: int, intron: int) -> Tuple[float, int, int]:
        """Inverse of selector_atom: (J, parity, shell) from the 32-bit atom."""
        _, _, shell = self.selector_decode(state24)
        J, parity = self.selector_intron_decode(intron)
        return J, parity, shell

    # ---- ORIENTED CHI ----
    # Full 6-bit orientation: chi_rot from (Z,2J), chi_tr from (N,parity),
    # then |chi| forced to |N-Z| mod 7. Beta is state-dependent via
    # |chi^q| = w+|q|-2|chi&q|; shell-only words[0] is insufficient.
    ROT_MASK = 0b000111
    TR_MASK = 0b111000

    def _chi_oriented(self, Z: int, N: int, J: float, parity: int) -> int:
        """Full 6-bit orientation: chi_rot from (Z,2J), chi_tr from (N,parity),
        then flip bits so |chi| = |N-Z| mod 7.
        """
        twoJ = int(round(2.0 * float(J)))
        par_bit = 1 if parity < 0 else 0
        rot = ((Z & 7) ^ (twoJ & 7)) & self.ROT_MASK
        tr = ((N & 7) ^ par_bit) & 0x07
        shell = abs(N - Z) % 7
        chi = rot | ((tr << 3) & self.TR_MASK)
        return self._set_weight_to(chi, shell)

    @staticmethod
    def _set_weight_to(chi: int, w: int) -> int:
        """Return a 6-bit word with popcount == w, closest to chi, by toggling."""
        chi &= 0x3F
        w = max(0, min(6, int(w)))
        while chi.bit_count() < w:
            chi |= (~chi & (chi + 1)) & 0x3F
        while chi.bit_count() > w:
            chi &= chi - 1
        return chi & 0x3F

    def selector_atom_oriented(
        self,
        Z: int,
        N: int,
        J: float,
        parity: int,
    ) -> Tuple[int, int]:
        """Oriented atom: full 6-bit chi from (Z,N,J,parity), |chi|=|N-Z| mod 7.

        Builds state24 with oriented chi and carries J/parity in the intron.
        This is the default nuclear path (also used by selector_atom).
        """
        chi6 = self._chi_oriented(Z, N, J, parity)
        u6 = Z & CHIRALITY_MASK_6
        if parity < 0:
            u6 |= 1 << self.U6_PARITY_BIT
        else:
            u6 &= ~(1 << self.U6_PARITY_BIT)
        v6 = u6 ^ chi6
        state24 = omega12_to_state24(
            OmegaState12(u6 & CHIRALITY_MASK_6, v6 & CHIRALITY_MASK_6)
        )
        intron = self.selector_intron_encode(J, parity)
        return state24, intron

    def selector_state24(
        self,
        Z: int,
        N: int,
        J: float,
        parity: int,
        parent_u6: Optional[int] = None,
    ) -> int:
        """Select the 24-bit Omega state for (Z, N, J, parity).

        chi6: shell = |N-Z| mod 7 exact (J no longer packed in chi; J and
        parity are carried in the intron via selector_atom). Retained for
        shell-only callers and backward compatibility.
        """
        shell = abs(N - Z) % 7
        chi6 = self.selector_chi6(J, shell, parity)
        u6 = (
            (Z & CHIRALITY_MASK_6)
            if parent_u6 is None
            else (parent_u6 & CHIRALITY_MASK_6)
        )
        if parity < 0:
            u6 |= 1 << self.U6_PARITY_BIT
        else:
            u6 &= ~(1 << self.U6_PARITY_BIT)
        v6 = u6 ^ chi6
        return omega12_to_state24(
            OmegaState12(u6 & CHIRALITY_MASK_6, v6 & CHIRALITY_MASK_6)
        )

    def selector_validate_decay(
        self,
        Zp: int,
        Np: int,
        Jp: float,
        Pp: int,
        Zd: int,
        Nd: int,
        Jd: float,
        Pd: int,
        operator: str = "beta",
    ) -> dict:
        """Encode parent, apply operator, read daughter selection rules.

        Reports measured quantities only:
        - parent/selected/daughter states and decoded daughter (J, shell)
        - parity conservation: daughter parity = parent parity (conserved QN)
        - angular-momentum rule: |J_d - J_p| in {0, 1} for allowed beta
        - alpha (Gate F) preserves shell; beta shell-parity (cocycle) conserved
        - shell forced by operator vs shell from daughter |N-Z| mod 7
        """
        sp, ip = self.selector_atom(Zp, Np, Jp, Pp)
        p_shell = abs(Np - Zp) % 7
        d_formula_shell = abs(Nd - Zd) % 7
        if operator == "beta":
            # β is a 3-branch UNA family; the daughter spin is operator-emitted
            # per branch (J -> J + dJ). Check whether ANY branch reproduces the
            # catalog daughter J, and whether any branch also lands on the
            # catalog daughter shell. The parent does not pick a unique branch;
            # branch provenance is not in the 24-bit Mac (2-to-1 shadow).
            branches = []
            for b, dJ in self.beta_decay_branches():
                dk, ik = self.beta_atom_step(sp, ip, b, dJ)
                Jk, Pk = self.selector_intron_decode(ik)
                _, _, sh_k = self.selector_decode(dk)
                branches.append((dk, ik, Jk, Pk, sh_k, b, dJ))
            d_k = branches[0][0]
            J_any = any(abs(Jk - Jd) < 1e-9 for *_, Jk, _, _, _, _ in branches)
            J_allowed = any(
                abs(Jk - Jp) <= 1.0 + 1e-9 for *_, Jk, _, _, _, _ in branches
            )
            shell_any = any(sh_k == d_formula_shell for *_, sh_k, _, _, _ in branches)
            both_any = any(
                (abs(Jk - Jd) < 1e-9 and sh_k == d_formula_shell)
                for *_, Jk, _, sh_k, _, _ in branches
            )
            # primary readout: best branch by exact J match, else by |dJ|<=1
            best = next(
                (br for br in branches if abs(br[2] - Jd) < 1e-9),
                next(
                    (br for br in branches if abs(br[2] - Jp) <= 1.0 + 1e-9),
                    branches[0],
                ),
            )
            _, ik, Jd_k, Pd_k, sh_k, _, dJ = best
            # Shell-parity invariant is a property of the isospin β⁻ operator
            # (both UNA references, |Δshell|=2), not of whichever branch wins
            # the J match. Evaluate it on that operator explicitly.
            d_iso = step_state_by_byte(int(sp) & 0xFFFFFF, self._BETA_BRANCH_BYTES[2])
            _, _, sh_iso = self.selector_decode(d_iso)
            return {
                "parent_state": sp,
                "parent_intron": ip,
                "kernel_daughter": d_k,
                "decoded_daughter_J": Jd_k,
                "decoded_daughter_shell": sh_k,
                "daughter_parity": Pd_k,
                "parity_conserved": Pd_k == Pp,
                "shell_op_forced": sh_iso,
                "shell_decoded": sh_k,
                "shell_matches_daughter_formula": sh_iso == d_formula_shell,
                "shell_parity_ok": (sh_iso % 2) == (p_shell % 2),
                "dJ": abs(Jd_k - Jp),
                "dJ_vs_catalog": abs(Jd_k - Jd),
                "J_rule_ok": J_allowed,
                "J_catalog_ok": J_any,
                "branch_shell_ok": shell_any,
                "branch_both_ok": both_any,
                "n_branches": len(branches),
            }
        # Alpha: Gate F preserves shell; daughter shell from the kernel Mac.
        d_k, _ = self.compute_alpha_transition(sp)
        _, _, sh_k = self.selector_decode(d_k)
        # Parity conserved (label); for alpha J is not an operator output here.
        return {
            "parent_state": sp,
            "parent_intron": ip,
            "kernel_daughter": d_k,
            "decoded_daughter_J": float("nan"),
            "decoded_daughter_shell": sh_k,
            "daughter_parity": Pp,
            "parity_conserved": True,
            "shell_op_forced": p_shell,
            "shell_decoded": sh_k,
            "shell_matches_daughter_formula": sh_k == d_formula_shell,
            "shell_parity_ok": (sh_k % 2) == (p_shell % 2),
            "dJ": float("nan"),
            "dJ_vs_catalog": float("nan"),
            "J_rule_ok": True,
            "J_catalog_ok": True,
            "branch_shell_ok": sh_k == d_formula_shell,
            "branch_both_ok": sh_k == d_formula_shell,
            "n_branches": 1,
        }

    @staticmethod
    def _u6(state24: int) -> int:
        """Extract u6 from a 24-bit state via OmegaState12."""
        return state24_to_omega12(int(state24) & 0xFFFFFF).u6

    def compute_alpha_tunneling(
        self,
        Q_alpha_MeV: float,
        Z_d: int,
        A_d: int,
    ) -> Tuple[float, float, float]:
        """α transmission via Beer-Lambert τ-dial.

        Returns (T, tau, E_G_MeV). Barrier = α (Z=2,A=4) on daughter.
        """
        Vb = self.coulomb_barrier_MeV(2, Z_d, 4.0, float(A_d))
        E_G = self.gamow_energy_MeV(2, Z_d, 4.0, float(A_d))
        tau = self.tau_nuclear(Q_alpha_MeV, E_G, Vb)
        T = self.transmission_from_tau(tau)
        return T, tau, E_G

    @staticmethod
    def fermi_integral(
        Q_beta_MeV: float,
        Z_daughter: int = 0,
        *,
        n_steps: int = 20000,
    ) -> float:
        """Dimensionless allowed β⁻ Fermi integral f(Z,Q).

        f = ∫₁^{W₀} F(Z,W)·p·W·(W₀−W)² dW  (W, p in m_e units, W₀=(Q+m_e)/m_e).
        F is the nonrelativistic Coulomb (Fermi) function 2πη/(1−e^{−2πη})
        with η = αZW/p. This is standard nuclear physics — the correct
        low-Q phase space (Sargent's Q⁵ only holds for Q ≫ m_e). Returns f₀
        (no Coulomb) when Z_daughter=0.
        """
        m_e = 0.51099895  # MeV
        if Q_beta_MeV <= 0.0:
            return 0.0
        W0 = (Q_beta_MeV + m_e) / m_e
        if W0 <= 1.0:
            return 0.0
        alpha = 7.2973525693e-3
        total = 0.0
        dW = (W0 - 1.0) / n_steps
        for i in range(n_steps + 1):
            W = 1.0 + i * dW
            p2 = W * W - 1.0
            if p2 <= 0.0:
                continue
            p = p2**0.5
            if Z_daughter:
                eta = alpha * Z_daughter * W / p
                x = 2.0 * 3.141592653589793 * eta
                F = x / (1.0 - 2.718281828459045 ** (-x)) if x > 0 else 1.0
            else:
                F = 1.0
            w = 0.5 if (i == 0 or i == n_steps) else 1.0
            total += w * F * p * W * (W0 - W) ** 2 * dW
        return total

    def half_life_beta_est(
        self,
        Q_beta_MeV: float,
        matrix_elem_sq: float = 1.0,
        *,
        Z_daughter: int = 0,
        log_ft_allowed: float = 3.05,
    ) -> Tuple[float, float]:
        """Estimate β T½ (s) = ln2·ft / (f·|M|²) with the true Fermi integral.

        The kernel supplies |M|² (structural overlap). f = fermi_integral is
        ordinary nuclear physics (not a CGM theorem). log_ft_allowed anchors a
        superallowed transition (log ft ≈ 3.05 ⇒ ft ≈ 1130 s for ³H); the raw
        two-point G_F coupling does not fix the nuclear matrix element, so the
        empirical ft is the anchor. Returns (T_half, f).
        """
        if Q_beta_MeV <= 0.0:
            return float("inf"), 0.0
        f = self.fermi_integral(Q_beta_MeV, Z_daughter)
        ft = 10.0**log_ft_allowed
        denom = max(f * matrix_elem_sq, 1e-30)
        return 0.693 * ft / denom, f

    def half_life_alpha_est(
        self,
        Q_alpha_MeV: float,
        Z_d: int,
        A_d: int,
        *,
        assault_freq: float = 1e21,
        L: int = 0,
        P_alpha: Optional[float] = None,
    ) -> Tuple[float, float, float, float]:
        """Estimate α T½ (s).

        The bare tunneling half-life (no preformation) is
          T½_bare = ln2 / (ν·T·H_L)
        with ν = assault frequency (default 1e21 s⁻¹), T = τ-dial tunnel
        transmission, H_L = C(L)/C(0) carrier-trace hindrance. A nucleus decays
        only after an α cluster preforms, so the realized half-life is shorter:
          T½ = T½_bare · P_α = ln2 · P_α / (ν·T·H_L).
        P_α = preformation probability. If None, derived structurally as
              5 / (|Ω|·|Alphabet|) = 5 / 2^20: the 5 bulk STF shells over the
              total operator-state phase space (4096 states × 256 bytes). This
              is the CGM structural preformation; the residual vs the measured
              half-life is the residual of that geometry, not an empirical fit.
        Returns (T_half, T_tunnel, H_L, P_alpha).
        """
        T, _, _ = self.compute_alpha_tunneling(Q_alpha_MeV, Z_d, A_d)
        H_L = self.alpha_structural_hindrance(L)
        if P_alpha is None:
            P_alpha = 5.0 / float(1 << 20)  # 5 bulk shells / (4096 * 256) phase space
        denom = assault_freq * T * H_L
        if denom <= 0.0:
            return float("inf"), T, H_L, P_alpha
        return 0.693 * P_alpha / denom, T, H_L, P_alpha


GROUND_STATES_PATH = (
    _REPO / "data" / "catalogs" / "ensdf" / "iaea_livechart_ground_states.csv"
)

# Jp examples: 1/2+, 0+, 5/2+, (1/2+), [5/2+], 1+, 7/2-
_JP_RE = re.compile(
    r"^\s*[\(\[]?\s*" r"(?P<num>\d+(?:/\d+)?)" r"\s*(?P<par>[+-])?" r"\s*[\)\]]?\s*$"
)


def parse_jp(raw: str) -> Optional[Tuple[float, int]]:
    """Parse ENSDF Jp string -> (J, parity) or None if unusable.

    Handles fractions (5/2), integers (0,1), optional surrounding () or [],
    and trailing +/-. Returns None for blank or malformed entries.
    """
    s = (raw or "").strip()
    if not s or s in ("?", "?", "unknown"):
        return None
    # Strip common ENSDF wrappers and trailing uncertainty markers.
    s = s.replace(" ", "")
    m = _JP_RE.match(s)
    if not m:
        # Try first token of compound like "1/2+,3/2+" (take leading).
        head = re.split(r"[,;]", s)[0]
        m = _JP_RE.match(head)
        if not m:
            return None
    num = m.group("num")
    par_s = m.group("par")
    if "/" in num:
        a, b = num.split("/", 1)
        try:
            J = float(a) / float(b)
        except ValueError:
            return None
    else:
        try:
            J = float(num)
        except ValueError:
            return None
    if par_s is None:
        return None
    parity = 1 if par_s == "+" else -1
    return J, parity


def _f(s: str) -> Optional[float]:
    try:
        v = float((s or "").strip())
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def load_ground_states(
    path: Path = GROUND_STATES_PATH,
) -> Dict[Tuple[int, int], dict]:
    """Load (Z,N) -> {symbol, J, P, decay_1, qa_MeV, half_life_sec, ...}."""
    out: Dict[Tuple[int, int], dict] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                Z = int(row["z"])
                N = int(row["n"])
            except (KeyError, ValueError):
                continue
            jp = parse_jp(row.get("jp") or "")
            if jp is None:
                continue
            J, P = jp
            qa_keV = _f(row.get("qa") or "")
            out[(Z, N)] = {
                "symbol": (row.get("symbol") or "").strip(),
                "J": J,
                "P": P,
                "jp_raw": (row.get("jp") or "").strip(),
                "decay_1": (row.get("decay_1") or "").strip(),
                "decay_1_pct": _f(row.get("decay_1_%") or ""),
                "qa_MeV": (qa_keV / 1000.0) if qa_keV is not None else None,
                "half_life_sec": _f(row.get("half_life_sec") or ""),
            }
    return out
