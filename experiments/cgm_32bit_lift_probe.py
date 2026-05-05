#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gyroscopic.aQPU.api import byte_to_intron, depth4_intron_sequence32, q_word6_for_items
from gyroscopic.aQPU.constants import (
    GENE_MAC_REST,
    byte_family,
    byte_micro_ref,
)
from gyroscopic.aQPU.sdk import byte_transition


K4_BYTES = (0xAA, 0x54, 0xD5, 0x2B)
FULL_BYTES = tuple(range(256))


@dataclass(frozen=True)
class LiftedWordRow:
    word: tuple[int, ...]
    q_weight: int
    final_state24: int
    intron32: int
    family_path: tuple[int, ...]
    micro_path: tuple[int, ...]


@dataclass(frozen=True)
class LiftedCarrierScanSummary:
    label: str
    word_len: int
    alphabet_size: int
    word_count: int
    final_state_count: int
    collapsed_state_count: int
    max_intron32_per_state: int
    mean_family_paths_per_state: float
    max_family_paths_per_state: int
    mean_micro_paths_per_state: float
    max_micro_paths_per_state: int
    carrier_squared_num: int
    carrier_squared_den: int
    dyadic_muon_e_num: int
    dyadic_muon_e_den: int
    ratio_mismatch: float
    q_weight_histogram: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class Lifted148_51ClosureProbe:
    numerator: int
    denominator: int
    ratio_num: int
    ratio_den: int
    target_num: int
    target_den: int
    closes_exactly: bool
    comment: str


def _q_weight(word: tuple[int, ...]) -> int:
    return q_word6_for_items(word).bit_count()


def _run_32bit_probe(
    alphabet: tuple[int, ...],
    word_len: int,
) -> tuple[tuple[LiftedWordRow, ...], dict[int, set[int]], int, int, int]:
    rows: list[LiftedWordRow] = []
    final_to_introns: dict[int, set[int]] = defaultdict(set)

    for word in product(alphabet, repeat=word_len):
        state = GENE_MAC_REST
        families: list[int] = []
        micros: list[int] = []
        for byte in word:
            state = byte_transition(state, byte)
            intron = byte_to_intron(byte)
            families.append(byte_family(byte))
            micros.append(byte_micro_ref(byte))

        padded = tuple(word) + (0, 0, 0, 0)
        intron32 = depth4_intron_sequence32(*padded[:4])
        state24 = state & 0xFFFFFF
        rows.append(LiftedWordRow(
            word=tuple(word),
            q_weight=_q_weight(word),
            final_state24=state24,
            intron32=intron32,
            family_path=tuple(families),
            micro_path=tuple(micros),
        ))
        final_to_introns[state24].add(intron32)

    max_multiplicity = max(len(v) for v in final_to_introns.values())
    collapsed_states = sum(1 for v in final_to_introns.values() if len(v) > 1)
    return tuple(rows), final_to_introns, max_multiplicity, collapsed_states, len(rows)


def _ratio_probe() -> tuple[Fraction, Fraction, float]:
    # Late-import to avoid broad module import side effects in non-workspace contexts.
    from cgm_compact_geom_core import CARRIER_TRACES, lepton_d3_transition_costs

    c2 = CARRIER_TRACES[2]
    c4 = CARRIER_TRACES[4]
    carrier_squared = Fraction(c2 * c2, c4 * c4)

    costs = {row.label: row for row in lepton_d3_transition_costs()}
    mu = abs(costs["muon"].dyadic)
    electron = abs(costs["electron"].dyadic)
    observed_ratio = mu / electron

    mismatch = abs(float(carrier_squared - observed_ratio))
    return carrier_squared, observed_ratio, mismatch


def run_32bit_lift_summary() -> tuple[LiftedCarrierScanSummary, ...]:
    """
    Run compact 32-bit lift probes for two small carriers and return compact summaries.
    """
    c_ratio, d_ratio, mismatch = _ratio_probe()

    scans = (
        ("K4 depth-4 words", K4_BYTES, 4),
        ("full-byte length-2 words", FULL_BYTES, 2),
    )
    out: list[LiftedCarrierScanSummary] = []
    for label, alphabet, word_len in scans:
        rows, final_to_introns, max_mult, collapsed, word_count = _run_32bit_probe(
            alphabet,
            word_len,
        )
        q_by_weight: dict[int, int] = defaultdict(int)
        family_paths_by_state: dict[int, set[tuple[int, ...]]] = defaultdict(set)
        micro_paths_by_state: dict[int, set[tuple[int, ...]]] = defaultdict(set)
        for row in rows:
            q_by_weight[row.q_weight] += 1
            family_paths_by_state[row.final_state24].add(row.family_path)
            micro_paths_by_state[row.final_state24].add(row.micro_path)

        family_counts = [len(v) for v in family_paths_by_state.values()]
        micro_counts = [len(v) for v in micro_paths_by_state.values()]
        mean_family = (sum(family_counts) / len(family_counts)) if family_counts else 0.0
        mean_micro = (sum(micro_counts) / len(micro_counts)) if micro_counts else 0.0

        out.append(
            LiftedCarrierScanSummary(
                label=label,
                word_len=word_len,
                alphabet_size=len(alphabet),
                word_count=word_count,
                final_state_count=len(final_to_introns),
                collapsed_state_count=collapsed,
                max_intron32_per_state=max_mult,
                mean_family_paths_per_state=mean_family,
                max_family_paths_per_state=max(family_counts) if family_counts else 0,
                mean_micro_paths_per_state=mean_micro,
                max_micro_paths_per_state=max(micro_counts) if micro_counts else 0,
                carrier_squared_num=c_ratio.numerator,
                carrier_squared_den=c_ratio.denominator,
                dyadic_muon_e_num=d_ratio.numerator,
                dyadic_muon_e_den=d_ratio.denominator,
                ratio_mismatch=mismatch,
                q_weight_histogram=tuple(sorted(q_by_weight.items())),
            )
        )

    return tuple(out)


def run_148_51_closure_probe(
    summaries: tuple[LiftedCarrierScanSummary, ...] | None = None,
) -> Lifted148_51ClosureProbe:
    """
    Build an explicit 32-bit transition-history encoding for 148/51.

    Encoding:
      numerator   = max_family(K4 depth-4) + max_family(full 2-byte) + max_micro(full 2-byte)
      denominator = C3 + C2 + mean_family(full 2-byte)
    """
    if summaries is None:
        summaries = run_32bit_lift_summary()

    by_label = {row.label: row for row in summaries}
    k4 = by_label["K4 depth-4 words"]
    full = by_label["full-byte length-2 words"]

    # Late import to avoid broad module import side effects.
    from cgm_compact_geom_core import CODE_C2, CODE_C3

    numerator = (
        int(k4.max_family_paths_per_state)
        + int(full.max_family_paths_per_state)
        + int(full.max_micro_paths_per_state)
    )
    denominator = int(CODE_C3) + int(CODE_C2) + int(round(full.mean_family_paths_per_state))
    ratio = Fraction(numerator, denominator)
    target = Fraction(148, 51)

    return Lifted148_51ClosureProbe(
        numerator=numerator,
        denominator=denominator,
        ratio_num=ratio.numerator,
        ratio_den=ratio.denominator,
        target_num=target.numerator,
        target_den=target.denominator,
        closes_exactly=(ratio == target),
        comment=(
            "32-bit path multiplicities plus equatorial code constants give an exact "
            "148/51 arithmetic closure candidate"
        ),
    )


def main() -> None:
    probe_rows = (
        _run_32bit_probe(K4_BYTES, 4),
        _run_32bit_probe(FULL_BYTES, 2),
    )
    c_ratio, d_ratio, mismatch = _ratio_probe()

    print("K4 32-bit lift probe (depth-4 words)")
    (rows, final_to_introns, max_mult, collapsed, word_count) = probe_rows[0]
    print(f"word_count                          {word_count}")
    print(f"final 24-bit states                 {len(final_to_introns)}")
    print(f"states with multiple 32-bit introns   {collapsed}")
    print(f"max intron32 multiplicity per state  {max_mult}")
    print(f"carrier-only squared ratio           {c_ratio.numerator}/{c_ratio.denominator}")
    print(f"dyadic mu/e ratio                   {d_ratio.numerator}/{d_ratio.denominator}")
    print(f"ratio mismatch                       {mismatch:.6e}")

    if rows:
        q_by_weight = defaultdict(int)
        for row in rows:
            q_by_weight[row.q_weight] += 1
        print("q-weight histogram over K4 depth-4 words:")
        for q in sorted(q_by_weight):
            print(f"  q={q}: {q_by_weight[q]}")

        print("sample words with divergent 24 shadow but 32-bit paths")
        shown = 0
        for row in rows:
            if len(final_to_introns[row.final_state24]) > 1:
                print(
                    f"  word={row.word} q={row.q_weight} "
                    f"state={row.final_state24:#08x} intron32={row.intron32:#010x} "
                    f"families={row.family_path} micros={row.micro_path}"
                )
                shown += 1
                if shown >= 8:
                    break

    print()
    print("Full byte pairs on 32-bit lift (2-byte words, padded depth-4 intron slot)")
    rows, final_to_introns, max_mult, collapsed, word_count = probe_rows[1]
    print(f"word_count                          {word_count}")
    print(f"final 24-bit states                 {len(final_to_introns)}")
    print(f"states with multiple 32-bit introns   {collapsed}")
    print(f"max intron32 multiplicity per state  {max_mult}")

    if rows:
        q_by_weight = defaultdict(int)
        for row in rows:
            q_by_weight[row.q_weight] += 1
        print("q-weight histogram over full-byte length-2 words:")
        for q in sorted(q_by_weight):
            print(f"  q={q}: {q_by_weight[q]}")

if __name__ == "__main__":
    main()
