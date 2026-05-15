#!/usr/bin/env python3
"""
cgm_aqpu_monodromy.py

aQPU archetype runtime + monodromy diagnostics.

It measures:
1. archetype-start trajectories
2. horizon/equality crossings
3. coordinate recurrence versus swapped recurrence
4. closure residue on repeated words
5. loop signatures for downstream monodromy analysis
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Final


# ---------------------------------------------------------------------------
# Core constants
# ---------------------------------------------------------------------------

GENE_MIC_S: Final[int] = 0xAA

ARCHETYPE_A12: Final[int] = 0xAAA
ARCHETYPE_B12: Final[int] = 0x555
GENE_MAC_REST: Final[int] = (ARCHETYPE_A12 << 12) | ARCHETYPE_B12
GENE_MAC_SWAPPED: Final[int] = (ARCHETYPE_B12 << 12) | ARCHETYPE_A12

MASK12: Final[int] = 0xFFF
PAIR_COUNT: Final[int] = 6

FAMILY_RAY_REF: Final[int] = 1
FAMILY_RAY_STEPS: Final[int] = 4

WORD_REPEATS_VISIBLE: Final[int] = 4

# ---------------------------------------------------------------------------
# Precomputed LUTs for Efficiency
# ---------------------------------------------------------------------------

MASK12_LUT: Final[list[int]] = [0] * 64
for _i in range(64):
    _mask = 0
    for _j in range(PAIR_COUNT):
        if (_i >> _j) & 1:
            _mask |= 0b11 << (2 * _j)
    MASK12_LUT[_i] = _mask & MASK12


# ---------------------------------------------------------------------------
# Byte / intron / family / mask formalism
# ---------------------------------------------------------------------------

def byte_to_intron(byte: int) -> int:
    return (int(byte) & 0xFF) ^ GENE_MIC_S


def intron_family(intron: int) -> int:
    intron &= 0xFF
    return (((intron >> 7) & 1) << 1) | (intron & 1)


def intron_micro_ref(intron: int) -> int:
    return (int(intron) >> 1) & 0x3F


def payload_popcount(micro_ref: int) -> int:
    return int(micro_ref & 0x3F).bit_count()


def intron_l0_parity(intron: int) -> int:
    x = int(intron) & 0xFF
    return (x & 1) ^ ((x >> 7) & 1)


def byte_chirality_increment(byte: int) -> int:
    intron = byte_to_intron(byte)
    micro = intron_micro_ref(intron)
    return micro ^ 0x3F if intron_l0_parity(intron) else micro


def intron_cgm_parities(intron: int) -> tuple[int, int, int, int]:
    x = int(intron) & 0xFF
    l0 = (x & 1) ^ ((x >> 7) & 1)
    li = ((x >> 1) & 1) ^ ((x >> 6) & 1)
    fg = ((x >> 2) & 1) ^ ((x >> 5) & 1)
    bg = ((x >> 3) & 1) ^ ((x >> 4) & 1)
    return l0, li, fg, bg


def intron_to_mask12(intron: int) -> int:
    """Expand the six payload bits into six 12-bit dipole-pair flips using LUT."""
    return MASK12_LUT[intron_micro_ref(intron)]


def byte_from_family_and_micro(family: int, micro_ref: int) -> int:
    family &= 0x03
    micro_ref &= 0x3F

    bit7 = (family >> 1) & 1
    bit0 = family & 1
    intron = (bit7 << 7) | (micro_ref << 1) | bit0
    return intron ^ GENE_MIC_S


def family_word_for_micro(micro_ref: int) -> list[int]:
    return [byte_from_family_and_micro(fam, micro_ref) for fam in range(4)]


# ---------------------------------------------------------------------------
# State transition
# ---------------------------------------------------------------------------

def split_state24(state24: int) -> tuple[int, int]:
    state24 &= 0xFFFFFF
    return (state24 >> 12) & MASK12, state24 & MASK12


def join_state24(a12: int, b12: int) -> int:
    return ((a12 & MASK12) << 12) | (b12 & MASK12)


def step_state24_by_byte(state24: int, byte: int) -> int:
    a12, b12 = split_state24(state24)
    intron = byte_to_intron(byte)
    mask = intron_to_mask12(intron)

    bit0 = intron & 1
    bit7 = (intron >> 7) & 1

    a_mut = a12 ^ mask
    next_a = b12 ^ (MASK12 if bit0 else 0)
    next_b = a_mut ^ (MASK12 if bit7 else 0)

    return join_state24(next_a, next_b)


# ---------------------------------------------------------------------------
# Observables
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Obs:
    state24: int
    a12: int
    b12: int
    diff12: int
    chi6: int
    bit_partition_ok: bool
    pair_diagonal: bool
    ab_bits: int
    arch_bits: int
    ab_shell: int
    arch_shell: int
    complement_horizon: bool
    equality_horizon: bool
    invariant_ok: bool
    rest_coordinate: bool
    swapped_coordinate: bool


def collapse_chi6(diff12: int) -> tuple[int, bool]:
    chi = 0
    diag = True

    for i in range(PAIR_COUNT):
        pair = (diff12 >> (2 * i)) & 0b11
        if pair == 0b11:
            chi |= 1 << i
        elif pair == 0b00:
            pass
        else:
            diag = False

    return chi & 0x3F, diag


def observe(state24: int) -> Obs:
    a12, b12 = split_state24(state24)
    diff = (a12 ^ b12) & MASK12
    chi, diag = collapse_chi6(diff)

    ab_bits = diff.bit_count()
    arch_bits = (MASK12 ^ diff).bit_count()

    ab_shell = ab_bits // 2 if diag else -1
    arch_shell = arch_bits // 2 if diag else -1

    return Obs(
        state24=state24 & 0xFFFFFF,
        a12=a12,
        b12=b12,
        diff12=diff,
        chi6=chi,
        bit_partition_ok=((ab_bits + arch_bits) == 12),
        pair_diagonal=diag,
        ab_bits=ab_bits,
        arch_bits=arch_bits,
        ab_shell=ab_shell,
        arch_shell=arch_shell,
        complement_horizon=(diff == MASK12),
        equality_horizon=(diff == 0),
        invariant_ok=diag,
        rest_coordinate=((state24 & 0xFFFFFF) == GENE_MAC_REST),
        swapped_coordinate=((state24 & 0xFFFFFF) == GENE_MAC_SWAPPED),
    )


def phase_name(obs: Obs, step: int) -> str:
    """
    Generates a phase string combining all active boundary flags.
    Fixes the original issue where 'rest' or 'swapped' masked the 'complement_horizon'.
    """
    if step == 0:
        return "archetype"
    
    parts = []
    if obs.complement_horizon:
        parts.append("C")
    if obs.equality_horizon:
        parts.append("E")
    if obs.rest_coordinate:
        parts.append("R")
    if obs.swapped_coordinate:
        parts.append("S")
        
    if parts:
        return "+".join(parts)
        
    if obs.arch_shell in (1, 2, 3):
        return "outward"
    return "bulk"


def bin6(value: int) -> str:
    return format(int(value) & 0x3F, "06b")


def yn(value: bool) -> str:
    return "Y" if value else "N"


@dataclass
class StepRow:
    step: int
    byte: int | None
    obs: Obs
    ixor: int
    pxor: int
    fxor: int
    fsum: int
    qxor: int
    chi_transport_ok: bool


def trace_word(word: list[int], start_state: int = GENE_MAC_REST) -> list[StepRow]:
    rows: list[StepRow] = []

    state = start_state
    initial_obs = observe(state)
    ixor = 0
    pxor = 0
    fxor = 0
    fsum = 0
    qxor = 0
    expected_chi = initial_obs.chi6

    rows.append(StepRow(0, None, initial_obs, ixor, pxor, fxor, fsum, qxor, True))

    for step, byte in enumerate(word, start=1):
        intron = byte_to_intron(byte)
        family = intron_family(intron)
        micro = intron_micro_ref(intron)

        ixor ^= intron
        pxor ^= micro
        fxor ^= family
        fsum = (fsum + family) & 0x03
        qxor ^= byte_chirality_increment(byte)

        state = step_state24_by_byte(state, byte)
        obs = observe(state)
        expected_chi ^= byte_chirality_increment(byte)
        rows.append(StepRow(step, byte, obs, ixor, pxor, fxor, fsum, qxor, obs.chi6 == expected_chi))

    return rows


# ---------------------------------------------------------------------------
# Summary & Horizons
# ---------------------------------------------------------------------------

@dataclass
class Summary:
    final_state: int
    start_state: int
    steps: int
    closure_defect: int
    terminal_coordinate: str
    first_revisit_of_start_step: int | None
    complement_horizon_hits: int
    equality_horizon_hits: int
    rest_hits: int
    swapped_hits: int
    horizon_sequence_ok: bool
    horizon_sequence_steps: list[int]
    carrier_revisits_rest: bool
    carrier_revisits_swapped_rest: bool
    on_complement_horizon: bool
    on_equality_horizon: bool
    invariant_ok: bool
    pair_diagonal: bool
    bit_partition_ok: bool
    chirality_transport_ok: bool
    final_qxor: int
    complement_horizon_steps: list[int]
    equality_horizon_steps: list[int]
    rest_coordinate_steps: list[int]
    swapped_coordinate_steps: list[int]
    final_cumulative_intron_xor: int
    final_cumulative_payload_xor: int
    final_cumulative_family_xor: int
    final_family_sum_mod4: int


def final_summary(rows: list[StepRow]) -> Summary:
    final = rows[-1].obs
    start_state = rows[0].obs.state24
    
    complement_steps = [r.step for r in rows if r.obs.complement_horizon]
    equality_steps = [r.step for r in rows if r.obs.equality_horizon]
    rest_steps = [r.step for r in rows if r.step > 0 and r.obs.rest_coordinate]
    swapped_steps = [r.step for r in rows if r.step > 0 and r.obs.swapped_coordinate]

    first_revisit_of_start = None
    for row in rows[1:]:
        if row.obs.state24 == start_state:
            first_revisit_of_start = row.step
            break

    if final.state24 == GENE_MAC_REST:
        terminal_coordinate = "rest"
    elif final.state24 == GENE_MAC_SWAPPED:
        terminal_coordinate = "swapped"
    else:
        terminal_coordinate = "other"

    horizon_sequence_ok, horizon_sequence_steps = horizon_sequence_check(rows)

    return Summary(
        final_state=final.state24,
        start_state=start_state,
        steps=len(rows) - 1,
        closure_defect=final.state24 ^ start_state,
        terminal_coordinate=terminal_coordinate,
        first_revisit_of_start_step=first_revisit_of_start,
        complement_horizon_hits=len(complement_steps),
        equality_horizon_hits=len(equality_steps),
        rest_hits=len(rest_steps),
        swapped_hits=len(swapped_steps),
        horizon_sequence_ok=horizon_sequence_ok,
        horizon_sequence_steps=horizon_sequence_steps,
        carrier_revisits_rest=any(rest_steps),
        carrier_revisits_swapped_rest=any(swapped_steps),
        on_complement_horizon=final.complement_horizon,
        on_equality_horizon=final.equality_horizon,
        invariant_ok=all(r.obs.invariant_ok for r in rows),
        pair_diagonal=all(r.obs.pair_diagonal for r in rows),
        bit_partition_ok=all(r.obs.bit_partition_ok for r in rows),
        chirality_transport_ok=all(r.chi_transport_ok for r in rows),
        final_qxor=rows[-1].qxor,
        complement_horizon_steps=complement_steps,
        equality_horizon_steps=equality_steps,
        rest_coordinate_steps=rest_steps,
        swapped_coordinate_steps=swapped_steps,
        final_cumulative_intron_xor=rows[-1].ixor,
        final_cumulative_payload_xor=rows[-1].pxor,
        final_cumulative_family_xor=rows[-1].fxor,
        final_family_sum_mod4=rows[-1].fsum,
    )


def horizon_sequence_check(rows: list[StepRow]) -> tuple[bool, list[int]]:
    complement_steps = [r.step for r in rows if r.obs.complement_horizon]
    equality_steps = [r.step for r in rows if r.obs.equality_horizon]
    swapped_steps = [r.step for r in rows if r.obs.swapped_coordinate]

    def next_after(values: list[int], after: int) -> int | None:
        for value in values:
            if value > after:
                return value
        return None

    if not complement_steps:
        return False, []

    c0 = complement_steps[0]
    e0 = next_after(equality_steps, c0)
    if e0 is None:
        return False, [c0]

    c_prime = next_after(swapped_steps, e0)
    if c_prime is None:
        return False, [c0, e0]

    e1 = next_after(equality_steps, c_prime)
    if e1 is None:
        return False, [c0, e0, c_prime]

    c1 = next_after(complement_steps, e1)
    if c1 is None:
        return False, [c0, e0, c_prime, e1]

    return True, [c0, e0, c_prime, e1, c1]


def format_horizon_sequence(steps: list[int]) -> str:
    if len(steps) < 5:
        return "C -> E -> C' -> E -> C"

    labels = ["C", "E", "C'", "E", "C"]
    return " -> ".join(f"{label}({step})" for label, step in zip(labels, steps))


# ---------------------------------------------------------------------------
# Generic Printers
# ---------------------------------------------------------------------------

def print_table(headers: list[str], rows: list[list[str]], title: str = "") -> None:
    """Utility to dynamically align and print tables, avoiding hidden/wrapped columns."""
    if title:
        print(f"\n{title}")
        print("-" * len(title))

    if not rows:
        print("(no data)\n")
        return

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("  ".join("-" * w for w in col_widths))

    for row in rows:
        row_line = "  ".join(str(cell).ljust(w) for cell, w in zip(row, col_widths))
        print(row_line)
    print()


def print_summary(summary: Summary, title: str = "Summary") -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for field in fields(summary):
        val = getattr(summary, field.name)
        if isinstance(val, int):
            if "state" in field.name or "defect" in field.name:
                val_str = f"0x{val:06X}"
            elif "xor" in field.name:
                val_str = f"0x{val:02X}"
            else:
                val_str = str(val)
        else:
            val_str = str(val)
        print(f"  {field.name:36s} {val_str}")
    print()


def describe_byte(byte: int) -> str:
    intron = byte_to_intron(byte)
    l0, li, fg, bg = intron_cgm_parities(intron)
    return (
        f"byte=0x{byte:02X} "
        f"intron=0x{intron:02X} "
        f"family={intron_family(intron):02b} "
        f"parities=L0{l0} LI{li} FG{fg} BG{bg} "
        f"micro_ref={intron_micro_ref(intron):02d} "
        f"payload_pop={payload_popcount(intron_micro_ref(intron))} "
        f"mask12=0x{intron_to_mask12(intron):03X}"
    )


def print_trace(rows: list[StepRow], title: str = "") -> None:
    headers = [
        "step", "phase", "byte", "state24", "A12", "B12", "chi6",
        "arch", "ab", "H", "E", "R", "S", "qx", "chOK", "ixor", "pxor", "fxor", "fsum"
    ]
    formatted_rows = []
    for row in rows:
        obs = row.obs
        byte_s = "--" if row.byte is None else f"{row.byte:02X}"
        formatted_rows.append([
            f"{row.step}",
            phase_name(obs, row.step),
            byte_s,
            f"0x{obs.state24:06X}",
            f"0x{obs.a12:03X}",
            f"0x{obs.b12:03X}",
            bin6(obs.chi6),
            f"{obs.arch_shell}",
            f"{obs.ab_shell}",
            yn(obs.complement_horizon),
            yn(obs.equality_horizon),
            yn(obs.rest_coordinate),
            yn(obs.swapped_coordinate),
            f"{row.qxor}",
            yn(row.chi_transport_ok),
            f"0x{row.ixor:02X}",
            f"{row.pxor:02d}",
            f"{row.fxor}",
            f"{row.fsum}",
        ])
    print_table(headers, formatted_rows, title)


# ---------------------------------------------------------------------------
# Diagnostic sections
# ---------------------------------------------------------------------------

def run_intrinsic_gate_checks() -> None:
    gates = [
        ("S reference", 0xAA),
        ("S comp-shadow", 0x54),
        ("C comp-shadow", 0xD5),
        ("C reference", 0x2B),
    ]

    headers = ["gate", "byte", "arche_shells", "H", "final", "rest", "ixor", "pxor", "fsum"]
    rows = []
    for label, byte in gates:
        word_rows = trace_word([byte] * 4)
        shells = [r.obs.arch_shell for r in word_rows]
        h_steps = [r.obs.complement_horizon for r in word_rows]
        final = word_rows[-1].obs
        rows.append([
            label,
            f"0x{byte:02X}",
            str(shells),
            str(h_steps),
            f"0x{final.state24:06X}",
            yn(final.rest_coordinate),
            f"0x{word_rows[-1].ixor:02X}",
            f"{word_rows[-1].pxor:02d}",
            f"{word_rows[-1].fsum}"
        ])
        
    print_table(headers, rows, "Intrinsic horizon-preserving byte checks")


def run_family_rays() -> dict[int, list[int]]:
    shell_paths: dict[int, list[int]] = {}

    for family in range(4):
        byte = byte_from_family_and_micro(family, FAMILY_RAY_REF)
        word = [byte] * FAMILY_RAY_STEPS
        word_rows = trace_word(word)
        shell_paths[family] = [r.obs.arch_shell for r in word_rows]
        
        print(f"Family {family:02b}: {describe_byte(byte)}")
        print_trace(word_rows, title=f"Family {family:02b} Trace")
        print_summary(final_summary(word_rows), title=f"Family {family:02b} Summary")

    headers = ["family", "arche_shells"]
    comp_rows = [[f"{f:02b}", str(shell_paths[f])] for f in range(4)]
    print_table(headers, comp_rows, "Family shell comparison")
    
    print(f"Family 00 == Family 11: {shell_paths[0] == shell_paths[3]}")
    print(f"Family 01 == Family 10: {shell_paths[1] == shell_paths[2]}")
    print(
        "Family 00 + Family 01: "
        f"{[a + b for a, b in zip(shell_paths[0], shell_paths[1])]}"
    )

    return shell_paths


def first_front_scan_exhaustive() -> tuple[dict[int, dict[int, int]], bool, bool, bool]:
    """
    One pass over 64 payloads x 4 families: shell histogram + law checks.

    Returns (family_counts, family_shell_law_ok, same_byte_depth4_ok, invariant_ok).
    """
    family_counts: dict[int, dict[int, int]] = {
        family: {shell: 0 for shell in range(7)}
        for family in range(4)
    }
    family_shell_law_ok = True
    same_byte_depth4_ok = True
    invariant_ok = True

    for family in range(4):
        for micro in range(64):
            byte = byte_from_family_and_micro(family, micro)

            first = trace_word([byte])[1].obs
            family_counts[family][first.arch_shell] += 1

            pop = payload_popcount(micro)
            expected = pop if family in (0, 3) else 6 - pop
            if first.arch_shell != expected:
                family_shell_law_ok = False

            rows4 = trace_word([byte] * 4)
            if rows4[-1].obs.state24 != GENE_MAC_REST:
                same_byte_depth4_ok = False

            if not all(r.obs.invariant_ok for r in rows4):
                invariant_ok = False

    return family_counts, family_shell_law_ok, same_byte_depth4_ok, invariant_ok


def run_first_front_scan() -> None:
    family_counts, family_shell_law_ok, same_byte_depth4_ok, invariant_ok = (
        first_front_scan_exhaustive()
    )

    headers = ["family"] + [f"shell_{i}" for i in range(7)]
    rows = []
    for family in range(4):
        rows.append([f"{family:02b}"] + [str(family_counts[family][s]) for s in range(7)])
        
    print_table(headers, rows, "First-expansion shell counts by family")

    print("Exhaustive checks across 64 payloads x 4 families")
    print("  Family 00/11 first shell = popcount(micro_ref)")
    print("  Family 01/10 first shell = 6 - popcount(micro_ref)")
    print(f"  family shell law holds:           {family_shell_law_ok}")
    print(f"  same-byte depth-4 carrier return: {same_byte_depth4_ok}")
    print(f"  invariant holds:                  {invariant_ok}")
    print("  shell spectrum:                   1, 6, 15, 20, 15, 6, 1")


def run_canonical_words() -> None:
    canonical = family_word_for_micro(FAMILY_RAY_REF)

    print("Canonical 4-family word")
    for i, byte in enumerate(canonical):
        print(f"{i}: {describe_byte(byte)}")

    rows = trace_word(canonical)
    print_trace(rows, title="Canonical 4-family word trace")
    print_summary(final_summary(rows), title="Canonical 4-family word summary")

    rows2 = trace_word(canonical * 2)
    print_trace(rows2, title="Canonical 4-family word repeated twice trace")
    print_summary(final_summary(rows2), title="Canonical 4-family word repeated twice summary")

    rows_loop = trace_word(canonical * WORD_REPEATS_VISIBLE)
    print_trace(rows_loop, title=f"Canonical 4-family word repeated {WORD_REPEATS_VISIBLE} times trace")
    print_summary(final_summary(rows_loop), title=f"Canonical x{WORD_REPEATS_VISIBLE} summary")

    headers = [
        "turn", "step", "state24", "arch", "ab", "H", "E", "R", "S", 
        "ixor", "pxor", "fxor", "fsum", "equality_hits", "complement_hits"
    ]
    chk_rows = []
    for turn in range(WORD_REPEATS_VISIBLE + 1):
        step = turn * 4
        row = rows_loop[step]
        obs = row.obs
        equality_hits = [r.step for r in rows_loop[: step + 1] if r.obs.equality_horizon]
        complement_hits = [r.step for r in rows_loop[: step + 1] if r.obs.complement_horizon]

        chk_rows.append([
            str(turn), str(step), f"0x{obs.state24:06X}",
            str(obs.arch_shell), str(obs.ab_shell),
            yn(obs.complement_horizon), yn(obs.equality_horizon),
            yn(obs.rest_coordinate), yn(obs.swapped_coordinate),
            f"0x{row.ixor:02X}", f"{row.pxor:02d}", f"{row.fxor}", f"{row.fsum}",
            str(equality_hits), str(complement_hits)
        ])
    print_table(headers, chk_rows, "Loop checkpoints at word boundaries")


def monodromy_probe_specs() -> list[tuple[str, list[int]]]:
    """Named probe words used by run_monodromy_probe and external callers."""
    micro = FAMILY_RAY_REF
    canonical = family_word_for_micro(micro)
    phase_shuffle = [canonical[0], canonical[2], canonical[1], canonical[3]]
    return [
        ("canonical", canonical),
        ("canonical x2", canonical * 2),
        ("canonical x4", canonical * 4),
        ("reverse", list(reversed(canonical))),
        ("phase shuffle", phase_shuffle),
        ("zero payload", family_word_for_micro(0)),
        ("full payload", family_word_for_micro(63)),
        ("same family 00", [canonical[0]] * 4),
        ("same family 11", [canonical[3]] * 4),
    ]


def monodromy_probe_summaries() -> list[tuple[str, Summary]]:
    """Summaries for the standard monodromy probe suite (no printing)."""
    out: list[tuple[str, Summary]] = []
    for name, word in monodromy_probe_specs():
        word_rows = trace_word(list(word))
        out.append((name, final_summary(word_rows)))
    return out


def verify_family_shell_law_exhaustive() -> tuple[bool, bool, bool]:
    """
    Same checks as run_first_front_scan (64 payloads x 4 families).

    Returns (family_shell_law_ok, same_byte_depth4_ok, invariant_ok).
    """
    _, family_shell_law_ok, same_byte_depth4_ok, invariant_ok = first_front_scan_exhaustive()
    return family_shell_law_ok, same_byte_depth4_ok, invariant_ok


def run_payload_population_words() -> None:
    headers = [
        "micro", "pop", "word_bytes", "final", "R", "S", "H", "E", 
        "ixor", "pxor", "fxor", "fsum", "shell_path"
    ]
    rows = []
    selected = [0, 1, 2, 3, 7, 15, 31, 63]

    for micro in selected:
        word4 = family_word_for_micro(micro)
        word8 = word4 * 2
        trace_rows = trace_word(word8)

        final = trace_rows[-1].obs
        shell_path = [r.obs.arch_shell for r in trace_word(word4)]
        byte_text = " ".join(f"{b:02X}" for b in word4)

        rows.append([
            str(micro), str(payload_popcount(micro)), byte_text,
            f"0x{final.state24:06X}", yn(final.rest_coordinate), yn(final.swapped_coordinate),
            yn(final.complement_horizon), yn(final.equality_horizon),
            f"0x{trace_rows[-1].ixor:02X}", f"{trace_rows[-1].pxor:02d}",
            f"{trace_rows[-1].fxor}", f"{trace_rows[-1].fsum}", str(shell_path)
        ])

    print_table(headers, rows, "Canonical word by payload population")


def run_monodromy_probe() -> None:
    headers = ["name", "steps", "defect", "term", "return", "eq", "co", "rest", "swap", "hseq"]
    rows = []
    for name, summary in monodromy_probe_summaries():
        rows.append([
            name, str(summary.steps), f"0x{summary.closure_defect:06X}",
            summary.terminal_coordinate, str(summary.first_revisit_of_start_step),
            str(summary.equality_horizon_hits), str(summary.complement_horizon_hits),
            str(summary.rest_hits), str(summary.swapped_hits),
            "PASS" if summary.horizon_sequence_ok else "FAIL"
        ])
    print_table(headers, rows, "Monodromy probe suite")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("aQPU archetype runtime diagnostic")
    print("=================================")
    print(f"GENE_MIC_S:       0x{GENE_MIC_S:02X}")
    print(f"GENE_MAC_REST:    0x{GENE_MAC_REST:06X}")
    print(f"GENE_MAC_SWAPPED: 0x{GENE_MAC_SWAPPED:06X}")
    print(f"ARCHETYPE_A12:    0x{ARCHETYPE_A12:03X}")
    print(f"ARCHETYPE_B12:    0x{ARCHETYPE_B12:03X}")
    print(f"family_ray_ref:   {FAMILY_RAY_REF}")
    print(f"family_ray_steps: {FAMILY_RAY_STEPS}")
    print(f"word_repeats:     {WORD_REPEATS_VISIBLE}")
    print()

    print("Measured columns & Phase Legend")
    print("-------------------------------")
    print("arch  = shell distance from complement-horizon coordinate")
    print("ab    = shell distance from equality-horizon coordinate")
    print("Phase flags: C=Complement horizon, E=Equality horizon, R=Rest coordinate, S=Swapped coordinate")
    print("Note: States can combine flags (e.g. C+R means on Complement horizon AND at Rest coordinate)")
    print("ixor  = cumulative intron xor")
    print("pxor  = cumulative payload xor")
    print("fxor  = cumulative family xor")
    print("fsum  = cumulative family sum modulo 4")
    print("qxor  = cumulative chirality increment xor")
    print()

    run_intrinsic_gate_checks()
    run_family_rays()
    run_first_front_scan()
    run_canonical_words()
    run_payload_population_words()
    run_monodromy_probe()


if __name__ == "__main__":
    main()