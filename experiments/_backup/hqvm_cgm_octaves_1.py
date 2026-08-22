#!/usr/bin/env python3
"""
hqvm_cgm_octaves_1.py

Dyadic atlas, word-doubling holonomy, shadow/projection equivocation,
and speculative probes (comma-aperture residuals, shell-pair just ratios,
byte palindrome frame, 12-vs-19 / 2/3 fingerprints).

No printing. Invoked by hqvm_cgm_octaves_run.py.
"""
from __future__ import annotations

import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

_REPO = Path(__file__).resolve().parents[1]
_EXP = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from gyroscopic.hQVM.api import (
    q_word6,
    shadow_partner_byte,
)
from gyroscopic.hQVM.constants import GENE_MAC_REST, step_state_by_byte
from gyroscopic.hQVM.api import state24_to_omega12

from hqvm_cgm_octaves_common import (
    CHIRALITY_SPACE,
    DELTA,
    DELTA_CONT,
    DELTA_DEPTH4,
    DELTA_DYADIC_8,
    H_CARD,
    NULL_SEED,
    OMEGA,
    STAGE_OF_BIT,
    STATUS_EMPIRICAL,
    STATUS_EXACT,
    STATUS_HYP,
    W2,
    W2p,
    Wfull,
    aperture_comma_table,
    best_dyadic_denom256,
    byte_charts,
    conditional_entropy_bits,
    exact_gates,
    interval_by_name,
    kernel_manifest,
    min_mean_max,
    octave_aperture_residues,
    octave_primitives_from_wavefunction,
    omega12_word_vs_signature_disagreement,
    one_step_shadow_from_rest,
    predecessor_horizon_ladder,
    probe_word,
    q6_fiber_sizes,
    random_word,
    shell_fiber_sizes,
    shell_pair_ratios,
    shadow_fiber_sizes,
    signature_composition_exact,
    z2_holonomy_word,
)


@dataclass(frozen=True)
class ScaleNode:
    name: str
    cardinality: int
    log2_card: float
    role: str
    kind: str  # "card" | "bits" | "word_len"


@dataclass
class Octaves1Census:
    gates: List[Tuple[str, bool]]
    manifest: Dict[str, object]
    aperture_residues: Dict[str, object]
    wavefunction_octave_primitives: Dict[str, object]
    predecessor_ladder: List[Dict[str, object]]
    atlas: List[ScaleNode]
    doubling_edges: List[Dict[str, object]]
    word_probes: List[Dict[str, object]]
    doubling_defects: List[Dict[str, object]]
    random_defect_summary: Dict[str, float]
    omega_sig_disagree: List[Dict[str, object]]
    projection_entropy: List[Dict[str, object]]
    one_step_shadow: Dict[str, object]
    comma_rows: List[Dict[str, float]]
    comma_best: List[Dict[str, object]]
    shell_ratio_hits: List[Dict[str, object]]
    palindrome: Dict[str, object]
    fifths_fingerprint: Dict[str, object]
    twelve_vs_nineteen: Dict[str, object]


def _atlas() -> List[ScaleNode]:
    card_nodes = [
        ("faces_2", 2, "carrier_faces"),
        ("cgm_stages_4", 4, "stage"),
        ("family_4", 4, "gauge"),
        ("horizon_64", H_CARD, "horizon"),
        ("q6_transport_64", 64, "payload"),
        ("shadow_128", 128, "single_step_shadow"),
        ("byte_alphabet_256", 256, "alphabet"),
        ("omega_4096", OMEGA, "carrier"),
        ("shells_7", 7, "radial"),
    ]
    bit_nodes = [
        ("chirality_bits_6", 6, "register_width"),
        ("byte_bits_8", 8, "instruction_width"),
        ("face_bits_12", 12, "component_width"),
        ("carrier_bits_24", 24, "state_width"),
        ("depth4_bits_48", 48, "closure_width"),
    ]
    word_nodes = [
        ("word_len_1", 1, "word_length"),
        ("word_len_2", 2, "word_length"),
        ("word_len_4", 4, "word_length"),
        ("word_len_8", 8, "word_length"),
        ("word_len_16", 16, "word_length"),
    ]
    out: List[ScaleNode] = []
    for name, card, role in card_nodes:
        lg = math.log2(card) if card > 0 and (card & (card - 1)) == 0 else (
            math.log2(card) if card > 0 else float("nan")
        )
        out.append(ScaleNode(name, card, lg, role, "card"))
    for name, width, role in bit_nodes:
        out.append(ScaleNode(name, width, float("nan"), role, "bits"))
    for name, L, role in word_nodes:
        out.append(ScaleNode(name, L, math.log2(L) if L > 0 else float("nan"), role, "word_len"))
    return out


def _doubling_edges(atlas: Sequence[ScaleNode]) -> List[Dict[str, object]]:
    by_kind_card: Dict[Tuple[str, int], List[ScaleNode]] = defaultdict(list)
    for n in atlas:
        by_kind_card[(n.kind, n.cardinality)].append(n)
    edges: List[Dict[str, object]] = []
    for (kind, card), nodes in sorted(by_kind_card.items()):
        nxt = by_kind_card.get((kind, 2 * card), [])
        for a in nodes:
            for b in nxt:
                edges.append(
                    {
                        "from": a.name,
                        "to": b.name,
                        "kind": kind,
                        "from_card": a.cardinality,
                        "to_card": b.cardinality,
                        "from_role": a.role,
                        "to_role": b.role,
                        "same_role": a.role == b.role,
                    }
                )
    return edges


def _canonical_words() -> List[Tuple[str, Tuple[int, ...]]]:
    m0 = 0
    return [
        ("byte_AA", (0xAA,)),
        ("byte_AB", (0xAB,)),
        ("W2_m0", W2(m0)),
        ("W2p_m0", W2p(m0)),
        ("Wfull_m0", Wfull(m0)),
        ("F2_m0", z2_holonomy_word(m0)),
        ("Wfull_m1", Wfull(1)),
        ("Wfull_m63", Wfull(63)),
        ("W2x2_m0", W2(m0) + W2(m0)),
        ("W2p_x2_m0", W2p(m0) + W2p(m0)),
    ]


def _word_probe_rows() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for name, word in _canonical_words():
        p = probe_word(name, word)
        rows.append(
            {
                "name": p.name,
                "length": p.length,
                "parity": p.parity,
                "tau_u6": p.tau_u6,
                "tau_v6": p.tau_v6,
                "packed_sig": p.packed_sig,
                "shell_tau": p.shell_tau,
                "is_identity": p.is_identity,
                "bytes_hex": ",".join(f"{b:02X}" for b in p.bytes),
            }
        )
    return rows


def _doubling_defect_rows() -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    """Signature monoid homomorphism check (not a holonomy defect)."""
    rows: List[Dict[str, object]] = []
    for name, word in _canonical_words():
        d = signature_composition_exact(word)
        d["name"] = name
        d["family"] = "canonical"
        rows.append(d)

    rng = random.Random(NULL_SEED)
    rand_ham: List[float] = []
    rand_compose_ok = 0
    n_rand = 64
    for L in (1, 2, 4, 8):
        for i in range(n_rand // 4):
            w = random_word(L, rng)
            d = signature_composition_exact(w)
            d["name"] = f"rand_L{L}_{i}"
            d["family"] = "random"
            rows.append(d)
            rand_ham.append(float(d["hamming_to_lift"]))
            if d["compose_exact"]:
                rand_compose_ok += 1

    mn, mean, mx = min_mean_max(rand_ham)
    summary = {
        "n_random": float(len(rand_ham)),
        "compose_exact_frac": rand_compose_ok / max(len(rand_ham), 1),
        "hamming_to_lift_min": mn,
        "hamming_to_lift_mean": mean,
        "hamming_to_lift_max": mx,
        "check": "signature_monoid_homomorphism",
    }
    return rows, summary


def _omega_sig_disagree_rows() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for name, word in _canonical_words():
        d = omega12_word_vs_signature_disagreement(word)
        d["name"] = name
        rows.append(d)
    return rows


def _projection_entropy_rows() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    shadow = shadow_fiber_sizes()
    q6 = q6_fiber_sizes()
    shell = shell_fiber_sizes()
    rows.append(
        {
            "chart": "byte->shadow_partner",
            "n_classes": len(shadow),
            "fiber_min": min(shadow),
            "fiber_max": max(shadow),
            "H_proj_bits": conditional_entropy_bits(shadow, 256),
            "status": STATUS_EXACT,
        }
    )
    rows.append(
        {
            "chart": "byte->q6",
            "n_classes": len(set(q_word6(b) for b in range(256))),
            "fiber_min": min(q6),
            "fiber_max": max(q6),
            "H_proj_bits": conditional_entropy_bits(q6, 256),
            "status": STATUS_EXACT,
        }
    )
    rows.append(
        {
            "chart": "byte->shell",
            "n_classes": 7,
            "fiber_min": min(shell),
            "fiber_max": max(shell),
            "H_proj_bits": conditional_entropy_bits(shell, 256),
            "fiber_sizes": list(shell),
            "status": STATUS_EXACT,
        }
    )

    shell1 = Counter()
    for b in range(256):
        s = step_state_by_byte(GENE_MAC_REST, b)
        shell1[state24_to_omega12(s).shell] += 1
    sizes1 = tuple(shell1[k] for k in range(7))
    rows.append(
        {
            "chart": "history1->shell_from_rest",
            "n_classes": 7,
            "fiber_min": min(sizes1) if sizes1 else 0,
            "fiber_max": max(sizes1) if sizes1 else 0,
            "H_proj_bits": conditional_entropy_bits(sizes1, 256),
            "status": STATUS_EMPIRICAL,
        }
    )

    shell2 = Counter()
    for b0 in range(256):
        for b1 in (b0, shadow_partner_byte(b0), (b0 + 1) & 0xFF, (b0 ^ 0xAA) & 0xFF):
            s = step_state_by_byte(GENE_MAC_REST, b0)
            s = step_state_by_byte(s, b1)
            shell2[state24_to_omega12(s).shell] += 1
    n2 = sum(shell2.values())
    sizes2 = tuple(shell2[k] for k in range(7))
    rows.append(
        {
            "chart": "history2_probe->shell_from_rest",
            "n_classes": 7,
            "n_histories": n2,
            "fiber_min": min(sizes2) if sizes2 else 0,
            "fiber_max": max(sizes2) if sizes2 else 0,
            "H_proj_bits": conditional_entropy_bits(sizes2, n2),
            "status": STATUS_EMPIRICAL,
        }
    )
    return rows


def _comma_best_hits(rows: Sequence[Dict[str, float]]) -> List[Dict[str, object]]:
    by_comma: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    for r in rows:
        by_comma[str(r["comma"])].append(r)
    out: List[Dict[str, object]] = []
    for comma, items in by_comma.items():
        best = min(items, key=lambda r: float(r["abs_diff_octaves"]))
        reject = best["aperture"] == "delta_BU_over_pi"
        out.append(
            {
                "comma": comma,
                "nearest_aperture": best["aperture"],
                "abs_diff_octaves": best["abs_diff_octaves"],
                "abs_diff_cents": best["abs_diff_cents"],
                "abs_diff_ticks": best["abs_diff_ticks"],
                "sanity_rejected_holonomy": reject,
                "status": STATUS_HYP,
            }
        )
    # Exact dyadic certificate: Delta_dyadic_8 is best k/256 to Delta_cont
    k, err = best_dyadic_denom256(DELTA_CONT)
    out.append(
        {
            "comma": "Delta_cont_best_dyadic_256",
            "nearest_aperture": "Delta_dyadic_8",
            "best_k": k,
            "best_approx": k / 256.0,
            "Delta_dyadic_8": DELTA_DYADIC_8,
            "abs_err": err,
            "is_k_equals_5": k == 5,
            "matches_Delta_dyadic_8": abs(k / 256.0 - DELTA_DYADIC_8) < 1e-15,
            "status": STATUS_EXACT,
        }
    )
    return out


def _shell_ratio_hit_table(tol_cents: float = 5.0) -> List[Dict[str, object]]:
    rows = shell_pair_ratios()
    declared = {"15/8", "4/3", "5/2", "10/3", "6/1", "20/15", "15/6"}
    out = []
    for r in rows:
        ratio = str(r["ratio"])
        keep = abs(float(r["residual_cents"])) <= tol_cents or ratio in declared
        if keep:
            out.append({**r, "status": STATUS_HYP, "tol_cents": tol_cents})
    # Declared non-pair probe: C(6,2)/2^3 = 15/8 major seventh
    just7 = interval_by_name("major_seventh_just")
    lg = math.log2(15.0 / 8.0)
    out.append(
        {
            "i": 2,
            "j": -3,
            "Ci": 15,
            "Cj": 8,
            "ratio": "15/8",
            "log2": lg,
            "cents": 1200.0 * lg,
            "nearest": just7.name,
            "nearest_cents": just7.cents,
            "residual_cents": 1200.0 * (lg - just7.log2),
            "status": STATUS_HYP,
            "tol_cents": tol_cents,
            "note": "C(6,2)/2^3",
        }
    )
    return out


def _palindrome_census() -> Dict[str, object]:
    consonant = 0
    dissonant = 0
    fold_zero = 0
    for b in range(256):
        ch = byte_charts(b)
        intron = ch["intron"]
        pc = intron.bit_count()
        fd = ch["fold_d"]
        if fd == 0:
            fold_zero += 1
        if (pc % 2 == 0) and fd == 0:
            consonant += 1
        if (pc % 2 == 1) and fd > 0:
            dissonant += 1
    nobles = (2, 10, 18, 36, 54, 86)
    hits = []
    for z in nobles:
        pos = z % 8
        stage = STAGE_OF_BIT[pos]
        hits.append({"Z": z, "pos_mod8": pos, "stage": stage, "on_CS": stage == "CS"})
    return {
        "frame": list(STAGE_OF_BIT),
        "fold_zero_bytes": fold_zero,
        "consonant_even_fold0": consonant,
        "dissonant_odd_fold_gt0": dissonant,
        "ratio_consonant_dissonant": (
            consonant / dissonant if dissonant else float("nan")
        ),
        "newlands_nobles": hits,
        "newlands_CS_hits": sum(1 for h in hits if h["on_CS"]),
        "newlands_n": len(nobles),
        "status": STATUS_HYP,
    }


def _fifths_and_12_19() -> Tuple[Dict[str, object], Dict[str, object]]:
    pc = interval_by_name("pythagorean_comma")
    fifth = interval_by_name("fifth")
    twelve_fifths = 12.0 * fifth.log2
    residual = twelve_fifths - 7.0
    fifths = {
        "log2_fifth": fifth.log2,
        "12_fifths_minus_7_oct": residual,
        "PC_log2": pc.log2,
        "abs_err": abs(residual - pc.log2),
        "log2_fifth_over_Delta_ticks": fifth.log2 / DELTA,
        "chirality_space": CHIRALITY_SPACE,
        "status": STATUS_EXACT if abs(residual - pc.log2) < 1e-12 else STATUS_EMPIRICAL,
    }
    face_bits = 12
    n_shell_idx = 7
    expr_a = 12.0 * math.log2(3.0) - (face_bits + n_shell_idx)
    expr_b = 12.0 * math.log2(3.0) - (face_bits + (n_shell_idx - 1))
    twelve = {
        "PC": pc.log2,
        "12_log2_3_minus_19": expr_a,
        "12_log2_3_minus_18": expr_b,
        "Delta": DELTA,
        "Delta_cont": DELTA_CONT,
        "Delta_depth4": DELTA_DEPTH4,
        "Delta_dyadic_8": DELTA_DYADIC_8,
        "abs_PC_minus_Delta": abs(pc.log2 - DELTA),
        "abs_PC_minus_Delta_dyadic_8": abs(pc.log2 - DELTA_DYADIC_8),
        "abs_PC_minus_Delta_depth4": abs(pc.log2 - DELTA_DEPTH4),
        "face_bits": face_bits,
        "n_shell_idx": n_shell_idx,
        "status": STATUS_HYP,
    }
    return fifths, twelve


def run_octaves_1() -> Octaves1Census:
    gates = list(exact_gates())
    man = kernel_manifest()
    residues = octave_aperture_residues()
    wf_prim = octave_primitives_from_wavefunction()
    pred = predecessor_horizon_ladder()
    atlas = _atlas()
    edges = _doubling_edges(atlas)
    words = _word_probe_rows()
    defects, rand_sum = _doubling_defect_rows()
    omega_dis = _omega_sig_disagree_rows()
    proj = _projection_entropy_rows()
    one_step = one_step_shadow_from_rest()
    comma_rows = aperture_comma_table()
    comma_best = _comma_best_hits(comma_rows)
    shell_hits = _shell_ratio_hit_table()
    pal = _palindrome_census()
    fifths, twelve = _fifths_and_12_19()

    dyadic_row = next(r for r in comma_best if r.get("comma") == "Delta_cont_best_dyadic_256")
    wp = wf_prim["word_periods"]
    gates.extend(
        [
            (
                "sig_compose_homomorphism",
                float(rand_sum["compose_exact_frac"]) == 1.0
                and all(bool(r["compose_exact"]) for r in defects if r["family"] == "canonical"),
            ),
            (
                "omega12_sig_action_exact",
                all(bool(r["exact"]) for r in omega_dis),
            ),
            (
                "one_step_shadow_128",
                int(one_step["n_unique_next"]) == 128
                and bool(one_step["all_fibres_size_2"])
                and bool(one_step["H_expected_1bit"]),
            ),
            (
                "Delta_dyadic_8_best_k5",
                bool(dyadic_row["is_k_equals_5"])
                and bool(dyadic_row["matches_Delta_dyadic_8"]),
            ),
            (
                "atlas_kinds_split",
                all(n.kind in ("card", "bits", "word_len") for n in atlas)
                and all(e["kind"] in ("card", "bits", "word_len") for e in edges),
            ),
            ("aperture_scale_ordering", bool(residues["ordering_dyadic_lt_cont_lt_depth4"])),
            ("Q_G_one_octave_above_2pi", abs(float(residues["log2_Q_G_over_2pi"]) - 1.0) < 1e-12),
            ("carrier_horizon_squared", bool(wf_prim["carrier_is_horizon_squared"])),
            ("word_W2_sq_id", bool(wp["W2_sq_is_id_rest"])),
            ("word_F2_id", bool(wp["F2_is_id_rest"])),
            ("wf_k4_w2_import", bool(wf_prim["k4_w2_all_pass"])),
            ("predecessor_48_in_ladder", any(r["is_48"] and not r["is_dyadic"] for r in pred)),
        ]
    )

    return Octaves1Census(
        gates=gates,
        manifest=man,
        aperture_residues=residues,
        wavefunction_octave_primitives=wf_prim,
        predecessor_ladder=pred,
        atlas=atlas,
        doubling_edges=edges,
        word_probes=words,
        doubling_defects=defects,
        random_defect_summary=rand_sum,
        omega_sig_disagree=omega_dis,
        projection_entropy=proj,
        one_step_shadow=one_step,
        comma_rows=comma_rows,
        comma_best=comma_best,
        shell_ratio_hits=shell_hits,
        palindrome=pal,
        fifths_fingerprint=fifths,
        twelve_vs_nineteen=twelve,
    )
