#!/usr/bin/env python3
"""
hqvm_cgm_octaves_2.py

Walsh/Krawtchouk dyadic bands, QuBEC octave renormalization defect,
and percolation thresholds in dyadic/octave coordinates.

No printing. Invoked by hqvm_cgm_octaves_run.py.
"""
from __future__ import annotations

import itertools
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

_REPO = Path(__file__).resolve().parents[1]
_EXP = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from gyroscopic.hQVM.family import (
    bfs_reach,
    bisect_p_c_rank_micro_ref,
    build_hqvm_d,
    fiber_complete,
    gf2_rank,
    holonomy_micro_cov,
    predicted_cluster_size,
    theta_micro_ref_exact,
)

from hqvm_cgm_octaves_common import (
    C6,
    CHIRALITY_D,
    DELTA,
    DELTA_CONT,
    DELTA_DEPTH4,
    DELTA_DYADIC_8,
    H_CARD,
    NULL_SEED,
    STATUS_EMPIRICAL,
    STATUS_EXACT,
    STATUS_HYP,
    STATUS_KERNEL_EXACT,
    isotropic_radial_velocity_gate,
    k4_interference_map,
    kl_divergence,
    krawtchouk,
    min_mean_max,
    normalize_dist,
    phi_isotropic_bit,
    popcount6,
    projected_chi_commutator_census,
    standing_wave_shell_table,
    total_variation,
    walsh_band_of_mode,
    wave_dispersion_table,
    wht64,
)

try:
    from hqvm_cgm_trestleboard_common import _kernel_percolation_board
except Exception:  # pragma: no cover
    _kernel_percolation_board = None  # type: ignore


@dataclass
class Octaves2Census:
    walsh_band_sizes: Dict[str, List[int]]
    walsh_energy_rows: List[Dict[str, object]]
    walsh_phi_audit: List[Dict[str, object]]
    wave_dispersion: List[Dict[str, object]]
    radial_velocity: Dict[str, object]
    standing_wave: Dict[str, object]
    k4_interference: List[Dict[str, object]]
    chi_commutator: Dict[str, object]
    krawtchouk_rows: List[Dict[str, object]]
    compatibility: List[Dict[str, object]]
    rg_rows: List[Dict[str, object]]
    percolation: Dict[str, object]
    dyadic_pairs: List[Dict[str, object]]
    octave_response: List[Dict[str, object]]
    word_law: List[Dict[str, object]]
    even_harmonics: Dict[str, object]
    square_root_audit: List[Dict[str, object]]
    gates: List[Tuple[str, bool]]
    gate_kinds: Dict[str, str]


def _mode_bands(perm: Sequence[int] | None = None) -> List[int]:
    return [walsh_band_of_mode(u, perm) for u in range(64)]


def _band_energy(hat: Sequence[float], bands: Sequence[int]) -> List[float]:
    e = [0.0, 0.0, 0.0, 0.0]
    dc = float(hat[0]) ** 2
    for u in range(1, 64):
        b = bands[u]
        if 0 <= b <= 3:
            e[b] += float(hat[u]) ** 2
    tot = dc + sum(e)
    if tot <= 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [x / tot for x in e]


def _ensemble_climate(name: str, rng: random.Random) -> Tuple[str, List[float]]:
    if name == "uniform":
        return name, [1.0 / 64.0] * 64
    if name == "shell_weight":
        # weight proportional to inverse shell population of popcount
        w = []
        for q in range(64):
            k = popcount6(q)
            w.append(1.0 / max(C6[k], 1))
        return name, normalize_dist(w)
    if name == "equator":
        w = [1.0 if popcount6(q) == 3 else 0.0 for q in range(64)]
        return name, normalize_dist(w)
    if name == "independent_bit":
        # product Bern(0.3) on bits
        p = 0.3
        w = []
        for q in range(64):
            pr = 1.0
            for i in range(6):
                bit = (q >> i) & 1
                pr *= p if bit else (1.0 - p)
            w.append(pr)
        return name, normalize_dist(w)
    if name == "gauge_biased":
        # favor low fold / even weight
        w = [1.5 if (popcount6(q) % 2 == 0) else 0.5 for q in range(64)]
        return name, normalize_dist(w)
    if name == "random_dirichlet":
        raw = [rng.random() + 1e-9 for _ in range(64)]
        return name, normalize_dist(raw)
    raise KeyError(name)


def _phi_multipliers(nu: Sequence[float]) -> List[float]:
    """Exact WHT multipliers for XOR climate transport: phi = WHT(nu)."""
    return wht64(nu)


def _evolve_xor(climate: Sequence[float], phi: Sequence[float], steps: int) -> List[float]:
    hat = wht64(climate)
    for _ in range(steps):
        hat = [hat[u] * phi[u] for u in range(64)]
    # inverse WHT is WHT/64
    back = wht64(hat)
    return normalize_dist([x / 64.0 for x in back])


def _coarse_grain(climate: Sequence[float], level: int, perm: Sequence[int]) -> List[float]:
    """Average climate over coordinates beyond the first `cuts[level]` bits."""
    cuts = (1, 2, 4, 6)
    cut = cuts[min(max(level, 0), 3)]
    keep = 0
    for i in range(cut):
        keep |= 1 << perm[i]
    buckets: Dict[int, List[float]] = {}
    for q, p in enumerate(climate):
        key = q & keep
        buckets.setdefault(key, []).append(float(p))
    # Lift coarse distribution back to 64 by uniform fill of free bits
    free = [i for i in range(6) if ((1 << perm[i]) & keep) == 0] if cut < 6 else []
    n_free = 1 << (6 - cut)
    out = [0.0] * 64
    for key, masses in buckets.items():
        mass = sum(masses)
        share = mass / n_free
        if cut == 6:
            out[key] = mass
            continue
        for bits in range(n_free):
            q = key
            for j, bi in enumerate(free):
                if (bits >> j) & 1:
                    q |= 1 << perm[bi]
            out[q] += share
    return normalize_dist(out)


def _walsh_census() -> Tuple[Dict[str, List[int]], List[Dict[str, object]]]:
    bands0 = _mode_bands()
    sizes = [0, 0, 0, 0]
    for u in range(1, 64):
        sizes[bands0[u]] += 1
    size_map = {"default_perm": sizes, "dc": [1]}

    # Stability under a few coordinate permutations
    perms = [tuple(range(6))]
    perms += list(itertools.islice(itertools.permutations(range(6)), 1, 13))
    size_rows = []
    for perm in perms:
        b = _mode_bands(perm)
        s = [0, 0, 0, 0]
        for u in range(1, 64):
            s[b[u]] += 1
        size_rows.append({"perm": list(perm), "sizes": s})
    size_map["perm_size_unique"] = sorted({tuple(r["sizes"]) for r in size_rows})  # type: ignore

    rng = random.Random(NULL_SEED)
    energy_rows: List[Dict[str, object]] = []
    for ename in (
        "uniform",
        "shell_weight",
        "equator",
        "independent_bit",
        "gauge_biased",
        "random_dirichlet",
    ):
        name, clim = _ensemble_climate(ename, rng)
        # ensemble = climate itself as nu for self-transport demo; also unit-step
        phi = _phi_multipliers(clim)
        # isotropic bit-flip ensemble: nu uniform on weight-1 generators
        nu_iso = [0.0] * 64
        for i in range(6):
            nu_iso[1 << i] = 1.0 / 6.0
        phi_iso = _phi_multipliers(nu_iso)
        for steps, label in ((0, "t0"), (1, "t1"), (2, "t2")):
            c = _evolve_xor(clim, phi_iso, steps)
            hat = wht64(c)
            e = _band_energy(hat, bands0)
            energy_rows.append(
                {
                    "ensemble": name,
                    "transport": "isotropic_bits",
                    "t": label,
                    "E0": e[0],
                    "E1": e[1],
                    "E2": e[2],
                    "E3": e[3],
                    "status": STATUS_EMPIRICAL,
                }
            )
        # self-transport multipliers
        for steps, label in ((0, "t0"), (1, "t1"), (2, "t2")):
            c = _evolve_xor(clim, phi, steps)
            hat = wht64(c)
            e = _band_energy(hat, bands0)
            energy_rows.append(
                {
                    "ensemble": name,
                    "transport": "self_phi",
                    "t": label,
                    "E0": e[0],
                    "E1": e[1],
                    "E2": e[2],
                    "E3": e[3],
                    "status": STATUS_EMPIRICAL,
                }
            )
    return size_map, energy_rows


def _walsh_phi_exact_audit() -> List[Dict[str, object]]:
    """Walsh–Krawtchouk certificate: hat_t(u)=hat_0(u)*phi(u)^t with phi=1-2*wt/6."""
    rng = random.Random(NULL_SEED + 7)
    nu_iso = [0.0] * 64
    for i in range(6):
        nu_iso[1 << i] = 1.0 / 6.0
    phi_iso = _phi_multipliers(nu_iso)
    phi_closed = [phi_isotropic_bit(u) for u in range(64)]
    phi_match = all(abs(phi_iso[u] - phi_closed[u]) < 1e-12 for u in range(64))

    rows: List[Dict[str, object]] = [
        {
            "check": "phi_closed_form_vs_WHT",
            "match": phi_match,
            "status": STATUS_EXACT if phi_match else STATUS_EMPIRICAL,
        }
    ]
    # Weight-bin damping: compare measured hat ratio to phi(wt)^t
    for ename in ("equator", "independent_bit", "shell_weight"):
        _, clim = _ensemble_climate(ename, rng)
        hat0 = wht64(clim)
        for t in (1, 2, 3):
            c_t = _evolve_xor(clim, phi_iso, t)
            hat_t = wht64(c_t)
            max_err = 0.0
            by_wt: Dict[int, Dict[str, float]] = {}
            for u in range(64):
                wt = popcount6(u)
                pred = hat0[u] * (phi_closed[u] ** t)
                err = abs(hat_t[u] - pred)
                if err > max_err:
                    max_err = err
                slot = by_wt.setdefault(
                    wt, {"n": 0.0, "mean_abs_hat": 0.0, "phi": 1.0 - (2.0 * wt) / 6.0}
                )
                slot["n"] += 1.0
                slot["mean_abs_hat"] += abs(hat_t[u])
            for wt, slot in by_wt.items():
                slot["mean_abs_hat"] /= max(slot["n"], 1.0)
            rows.append(
                {
                    "check": "damping",
                    "ensemble": ename,
                    "t": t,
                    "max_abs_err": max_err,
                    "exact": max_err < 1e-10,
                    "phi_by_wt": {str(k): v["phi"] for k, v in sorted(by_wt.items())},
                    "status": STATUS_EXACT if max_err < 1e-10 else STATUS_EMPIRICAL,
                }
            )
    return rows


def _square_root_rank_audit() -> List[Dict[str, object]]:
    """Structured fiber-complete SRCT audit (weight / even-odd shells).

    Scope matches percolation_4 / allometry_3: arbitrary fiber-complete q-subsets
    are not claimed; weight and parity shells are the verified families.
    """
    eng = build_hqvm_d(CHIRALITY_D)
    cases: List[Tuple[str, List[int], int | None]] = []
    for w in range(7):
        alphabet = [b for b in range(eng.n_bytes) if eng.q_weight[b] == w]
        cases.append((f"q_weight={w}", alphabet, None))
    even = [b for b in range(eng.n_bytes) if eng.q_weight[b] % 2 == 0]
    odd = [b for b in range(eng.n_bytes) if eng.q_weight[b] % 2 == 1]
    cases.append(("even_q_weight", even, 5))
    cases.append(("odd_q_weight", odd, 6))
    # Nested weight-at-most for full-rank ladder endpoint
    cases.append(
        (
            "q_weight_le_1",
            [b for b in range(eng.n_bytes) if eng.q_weight[b] <= 1],
            6,
        )
    )

    rows: List[Dict[str, object]] = []
    ranks_seen: set = set()
    for name, alphabet, expect_r in cases:
        if not alphabet:
            continue
        rank = gf2_rank([eng.q_by_byte[b] for b in alphabet], eng.d)
        fc = fiber_complete(alphabet, eng)
        reach, _, _, e_full = bfs_reach(eng, alphabet)
        pred = predicted_cluster_size(rank)
        rank_ok = expect_r is None or rank == expect_r
        match = reach == pred and fc and rank_ok
        ranks_seen.add(rank)
        rows.append(
            {
                "case": name,
                "r": rank,
                "n_bytes": len(alphabet),
                "fiber_complete": fc,
                "n_reach": reach,
                "predicted": pred,
                "E_full": e_full,
                "expect_r": expect_r,
                "match": match,
                "status": STATUS_EXACT if match else STATUS_EMPIRICAL,
            }
        )
    rows.append(
        {
            "case": "ranks_covered",
            "r": -1,
            "ranks_seen": sorted(ranks_seen),
            "covers_0_1_5_6": set(ranks_seen) >= {0, 1, 5, 6},
            "plateau_r5_1024": any(
                r["r"] == 5 and r["n_reach"] == 1024 and r["match"] for r in rows if "n_reach" in r
            ),
            "all_structured_match": all(r["match"] for r in rows if "match" in r),
            "match": all(r.get("match", True) for r in rows if "match" in r)
            and set(ranks_seen) >= {0, 1, 5, 6},
            "status": STATUS_EXACT,
        }
    )
    return rows


def _krawtchouk_census() -> List[Dict[str, object]]:
    # Matrix K_r(N) on shells; row energy vs harmonic 1/(r+1) shape
    rows: List[Dict[str, object]] = []
    mat = [[krawtchouk(r, n, 6) for n in range(7)] for r in range(7)]
    for r in range(7):
        row = mat[r]
        energy = sum(abs(x) for x in row)
        rows.append(
            {
                "mode_r": r,
                "row": row,
                "l1": energy,
                "harmonic_1_over_r1": 1.0 / (r + 1),
                "status": STATUS_EXACT,
            }
        )
    # Shape correlation of l1 pattern vs 1/(r+1)
    l1 = [float(r["l1"]) for r in rows]
    harm = [float(r["harmonic_1_over_r1"]) for r in rows]
    mu_l, mu_h = sum(l1) / 7, sum(harm) / 7
    num = sum((a - mu_l) * (b - mu_h) for a, b in zip(l1, harm))
    den = math.sqrt(
        sum((a - mu_l) ** 2 for a in l1) * sum((b - mu_h) ** 2 for b in harm)
    )
    corr = num / den if den > 0 else float("nan")
    rows.append(
        {
            "mode_r": -1,
            "row": [],
            "l1": float("nan"),
            "harmonic_1_over_r1": float("nan"),
            "corr_l1_vs_harmonic": corr,
            "status": STATUS_HYP,
        }
    )
    return rows


def _compatibility_matrix() -> List[Dict[str, object]]:
    """Correlation between Walsh band indicator and shell popcount mode."""
    bands = _mode_bands()
    # For each mode u, band ell and shell r=popcount(u)
    # Build joint over u=1..63
    joint = [[0.0 for _ in range(7)] for _ in range(4)]
    for u in range(1, 64):
        ell = bands[u]
        r = popcount6(u)
        joint[ell][r] += 1.0
    # normalize
    tot = sum(sum(row) for row in joint)
    joint = [[x / tot for x in row] for row in joint]
    out = []
    for ell in range(4):
        for r in range(7):
            out.append(
                {
                    "band": ell,
                    "shell": r,
                    "mass": joint[ell][r],
                    "status": STATUS_EXACT,
                }
            )
    return out


def _rg_defect_rows() -> List[Dict[str, object]]:
    rng = random.Random(NULL_SEED + 1)
    perm = tuple(range(6))
    rows: List[Dict[str, object]] = []
    nu_iso = [0.0] * 64
    for i in range(6):
        nu_iso[1 << i] = 1.0 / 6.0
    phi_iso = _phi_multipliers(nu_iso)

    for ename in (
        "uniform",
        "shell_weight",
        "equator",
        "independent_bit",
        "gauge_biased",
        "random_dirichlet",
    ):
        name, clim = _ensemble_climate(ename, rng)
        for level in (0, 1, 2):
            # A = coarse_{level+1}( evolve_2 (climate) )
            e2 = _evolve_xor(clim, phi_iso, 2)
            A = _coarse_grain(e2, level + 1, perm)
            # B = evolve_1_coarse ( coarse_level(climate) )
            # Coarse climate at level, then one isotropic step in ambient, re-coarse
            c_coarse = _coarse_grain(clim, level, perm)
            e1 = _evolve_xor(c_coarse, phi_iso, 1)
            B = _coarse_grain(e1, level + 1, perm)
            tv = total_variation(A, B)
            kl = kl_divergence(A, B)
            rows.append(
                {
                    "ensemble": name,
                    "level": level,
                    "cut_bits": (1, 2, 4, 6)[min(level + 1, 3)],
                    "diagram": "pi_{l+1} K^2 vs pi_{l+1} K pi_l",
                    "TV": tv,
                    "KL": kl,
                    "commutes_exact": tv < 1e-12,
                    "exact_zero": tv < 1e-12,
                    "status": STATUS_EXACT if tv < 1e-12 else STATUS_EMPIRICAL,
                }
            )
    return rows


def _percolation_block() -> Tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    p_rank = float(bisect_p_c_rank_micro_ref(CHIRALITY_D))
    board: Dict[str, float] = {"p_c_rank": p_rank}
    if _kernel_percolation_board is not None:
        pb = _kernel_percolation_board()
        board.update(
            {
                "p_c_span": float(pb.p_c_span),
                "p_c_full": float(pb.p_c_full),
                "p_c_spectrum": float(pb.p_c_spectrum),
                "p_c_word": float(pb.p_c_word),
            }
        )
    else:
        board.update(
            {
                "p_c_span": float("nan"),
                "p_c_full": float("nan"),
                "p_c_spectrum": float("nan"),
                "p_c_word": float("nan"),
            }
        )

    refs = {
        "Delta_cont": DELTA_CONT,
        "Delta": DELTA,
        "Delta_depth4": DELTA_DEPTH4,
        "Delta_dyadic_8": DELTA_DYADIC_8,
        "1/32": 1.0 / 32.0,
        "1/64": 1.0 / 64.0,
        "1/256": 1.0 / 256.0,
    }
    # Dyadic coordinates for each threshold
    thresh_rows = []
    for name, p in board.items():
        if not (p == p) or p <= 0:
            continue
        thresh_rows.append(
            {
                "name": name,
                "p": p,
                "z_p": -math.log2(p),
                "octaves_from_Delta": math.log2(p / DELTA) if DELTA > 0 else float("nan"),
                "nearest_ref": min(refs.keys(), key=lambda k: abs(math.log2(p / refs[k]))),
                "nearest_ref_oct": min(abs(math.log2(p / refs[k])) for k in refs),
            }
        )

    # Pairwise log2(pi/pj) vs {0, ±1, ±1/2}
    names = [r["name"] for r in thresh_rows]
    pair_rows: List[Dict[str, object]] = []
    targets = (0.0, 1.0, -1.0, 0.5, -0.5)
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            if i >= j:
                continue
            pi = float(board[ni])
            pj = float(board[nj])
            if pi <= 0 or pj <= 0:
                continue
            o = math.log2(pi / pj)
            best_t = min(targets, key=lambda t: abs(o - t))
            pair_rows.append(
                {
                    "i": ni,
                    "j": nj,
                    "log2_pi_pj": o,
                    "nearest_target": best_t,
                    "abs_err": abs(o - best_t),
                    "hit_tol_0.05": abs(o - best_t) <= 0.05,
                    "status": STATUS_HYP,
                }
            )

    # Octave response R_E(p)=theta(2p)-theta(p)
    resp: List[Dict[str, object]] = []
    for p in (0.05, 0.1, p_rank, 0.2, 0.3, 0.5):
        if 2 * p > 1:
            continue
        th = theta_micro_ref_exact(p, CHIRALITY_D)
        th2 = theta_micro_ref_exact(min(2 * p, 1.0), CHIRALITY_D)
        th_h = theta_micro_ref_exact(p / 2.0, CHIRALITY_D)
        resp.append(
            {
                "p": p,
                "theta_p": th,
                "theta_2p": th2,
                "theta_p_half": th_h,
                "R_double": th2 - th,
                "R_half": th - th_h,
                "status": STATUS_EXACT,
            }
        )

    # Word completion law vs length controls
    word_rows: List[Dict[str, object]] = []
    for L in (1, 2, 4, 8):
        for p in (0.1, 0.2, 0.3, 0.5, p_rank):
            pred = holonomy_micro_cov(p, CHIRALITY_D, word_len=L)
            # compare to fourth-power reference shape at same p
            ref4 = holonomy_micro_cov(p, CHIRALITY_D, word_len=4)
            word_rows.append(
                {
                    "L": L,
                    "p": p,
                    "P_word": pred,
                    "P_word_L4": ref4,
                    "ratio_to_L4": pred / ref4 if ref4 > 0 else float("nan"),
                    "formula": f"1-(1-p^{L})^{H_CARD}",
                    "status": STATUS_EXACT,
                }
            )

    summary = {
        "board": board,
        "thresholds": thresh_rows,
        "status": STATUS_EMPIRICAL,
    }
    return summary, pair_rows, resp, word_rows


def _even_harmonics_five_octaves() -> Dict[str, object]:
    """Four readings of 'even' on H={2,4,...,64} + F cycle-index table.

    Keeps all 32 even modes (not only the dyadic spine). Imports reachability
    formula and verifies even-alphabet BFS when cheap.
    """
    from hqvm_cgm_octaves_common import (
        DYADIC_SPINE,
        EVEN_H,
        PREDECESSOR_48,
        bytes_by_q_parity,
        even_weight_q6,
        f_cycle_index_table,
        fold_flat_curved_census,
        foundation_lock_scalars,
        harmonic_to_even_q_iso,
        k4_horizon_cycle_types,
        odd_weight_q6,
    )

    N = H_CARD
    mid = N // 2
    even_ns = list(EVEN_H)
    octave_ladder = list(DYADIC_SPINE)

    mid_amps = [abs(math.sin(n * math.pi * mid / N)) for n in even_ns]
    odd_ns = list(range(1, N, 2))
    odd_mid = [abs(math.sin(n * math.pi * mid / N)) for n in odd_ns]

    ev = even_weight_q6()
    od = odd_weight_q6()
    fold = fold_flat_curved_census()
    k4 = k4_horizon_cycle_types()
    iso = harmonic_to_even_q_iso()
    cycle_table = f_cycle_index_table()

    step_octaves = [
        math.log2(octave_ladder[i + 1] / octave_ladder[i])
        for i in range(len(octave_ladder) - 1)
    ]
    span_octaves = math.log2(octave_ladder[-1] / octave_ladder[0])

    # Reading 2.2 / 5.3: even/odd alphabets + square-root reachability
    A_even = bytes_by_q_parity(True)
    A_odd = bytes_by_q_parity(False)
    reach_even_formula = (2**5) ** 2
    reach_full_formula = (2**6) ** 2
    reach_even_bfs = None
    try:
        from gyroscopic.hQVM.family import bfs_reach, build_hqvm_d

        eng = build_hqvm_d(CHIRALITY_D)
        n_reach, _, _, e_full = bfs_reach(eng, A_even)
        reach_even_bfs = {
            "n_reach": n_reach,
            "E_full": e_full,
            "matches_1024": n_reach == reach_even_formula,
        }
    except Exception as exc:  # pragma: no cover
        reach_even_bfs = {"error": str(exc)}

    # Non-dyadic evens must be retained (6,10,12,...,62) — bulk of even subspace
    nondyadic = [n for n in even_ns if n not in octave_ladder]
    locks = foundation_lock_scalars()

    scale_map = {
        2: "faces / minimal dyadic",
        4: "CGM stages / family K4",
        8: "byte bit-width",
        16: "flat fold classes (2^4)",
        32: "half-horizon / even-q / F two-cycles",
        64: "horizon |H| / q6 alphabet",
        48: "predecessor horizon (depth-4 · 3 axes)",
    }

    return {
        "N_horizon": N,
        "midpoint_index": mid,
        "H": even_ns,
        "n_H": len(even_ns),
        "D": octave_ladder,
        "nondyadic_evens": nondyadic,
        "n_nondyadic": len(nondyadic),
        "predecessor_48_in_H": PREDECESSOR_48 in even_ns,
        "span_octaves_2_to_64": span_octaves,
        "all_steps_exact_octave": all(abs(s - 1.0) < 1e-15 for s in step_octaves),
        "ladder_terminates_at_H": octave_ladder[-1] == N,
        "ladder_rungs_eq_chirality_dim": len(octave_ladder) == CHIRALITY_D,
        "scale_map": scale_map,
        # reading 2.1 node / fold
        "fold": fold,
        "even_modes_node_at_midpoint": max(mid_amps) < 1e-12,
        "odd_modes_antinode_at_midpoint": min(odd_mid) > 0.5,
        "k4_horizon": {g: {k: v for k, v in info.items() if k != "all_pairs"} for g, info in k4.items()},
        "F_two_cycles": k4["F"]["two_cycles"],
        # reading 2.2 weight / percolation
        "even_q_count": len(ev),
        "odd_q_count": len(od),
        "A_even_n": len(A_even),
        "A_odd_n": len(A_odd),
        "reach_even_formula": reach_even_formula,
        "reach_full_formula": reach_full_formula,
        "reach_even_bfs": reach_even_bfs,
        # reading 2.3 cover
        "byte_shadow_128": 256 // 2,
        "hologram_4096": N * N,
        # reading 2.4 shells
        "shell_pops": [c * N for c in C6],
        "bulk_sum": sum(c * N for c in C6[1:6]),
        "bulk_is_3968": sum(c * N for c in C6[1:6]) == 3968,
        "horizon_sum": C6[0] * N + C6[6] * N,
        "rho5": locks["rho5"],
        "atlas_note": (
            "E32 even-index atlas (32) contains dyadic spine O5={2..64} "
            "(five octave intervals) and predecessor 48"
        ),
        # iso + cycle index (the concrete chart)
        "iso_n": len(iso),
        "cycle_index_n": len(cycle_table),
        "cycle_index_head": cycle_table[:8],
        "cycle_index_48": [r for r in cycle_table if r["harmonic_n"] == 48],
        "all_f_xor_epsilon": all(r["f_xor_is_epsilon"] for r in cycle_table),
        "status": STATUS_HYP,
    }


def run_octaves_2() -> Octaves2Census:
    sizes, energy = _walsh_census()
    phi_audit = _walsh_phi_exact_audit()
    disp = wave_dispersion_table()
    vel = isotropic_radial_velocity_gate()
    stand = standing_wave_shell_table()
    interf = k4_interference_map()
    chi_comm = projected_chi_commutator_census()
    kraw = _krawtchouk_census()
    compat = _compatibility_matrix()
    rg = _rg_defect_rows()
    perc, pairs, resp, word = _percolation_block()
    even_h = _even_harmonics_five_octaves()
    sr_audit = _square_root_rank_audit()

    l4 = sorted([r for r in word if r["L"] == 4], key=lambda r: float(r["p"]))
    mono = all(l4[i]["P_word"] <= l4[i + 1]["P_word"] + 1e-15 for i in range(len(l4) - 1))
    bfs = even_h.get("reach_even_bfs") or {}
    phi_ok = all(
        bool(r.get("match", r.get("exact", False)))
        for r in phi_audit
        if r.get("check") in ("phi_closed_form_vs_WHT", "damping")
    )
    sr_ok = bool(next(r for r in sr_audit if r["case"] == "ranks_covered")["match"])
    disp_ok = all(bool(r["phi_closed"]) for r in disp)
    gates = [
        ("walsh_bands_partition_63", sum(sizes["default_perm"]) == 63),
        ("walsh_phi_exact_damping", phi_ok),
        ("dispersion_phi_r_eq_1_minus_r_over_3", disp_ok),
        ("radial_vp_eq_vg_damping_channel", bool(vel["all_vp_eq_vg"])),
        ("standing_wave_bulk_3968", bool(stand["bulk_is_3968"])),
        ("projected_chi_commutator_identity", bool(chi_comm["projected_commutator_is_identity"])),
        (
            "uniform_rg_tv_near0",
            any(r["ensemble"] == "uniform" and r["exact_zero"] for r in rg),
        ),
        ("p_c_rank_in_(0,1)", 0.0 < float(perc["board"]["p_c_rank"]) < 1.0),
        ("word_law_L4_monotonic", mono),
        ("even_H_has_32", even_h["n_H"] == 32),
        ("even_harm_five_octaves", abs(float(even_h["span_octaves_2_to_64"]) - 5.0) < 1e-15),
        ("even_harm_midpoint_node", bool(even_h["even_modes_node_at_midpoint"])),
        ("even_q_eq_32", even_h["even_q_count"] == 32),
        ("F_two_cycles_32", even_h["F_two_cycles"] == 32),
        ("fold_flat_16", even_h["fold"]["flat"] == 16),
        ("48_in_H", bool(even_h["predecessor_48_in_H"])),
        ("nondyadic_retained", even_h["n_nondyadic"] == 26),
        ("cycle_index_32", even_h["cycle_index_n"] == 32),
        ("reach_even_1024", bool(bfs.get("matches_1024", False))),
        ("square_root_rank_0to6", sr_ok),
        ("octave_ladder_ends_at_H", bool(even_h["ladder_terminates_at_H"])),
        ("octave_ladder_rungs_eq_d", bool(even_h["ladder_rungs_eq_chirality_dim"])),
    ]
    gate_kinds = {name: "internal_kernel_identity" for name, _ in gates}

    return Octaves2Census(
        walsh_band_sizes=sizes,
        walsh_energy_rows=energy,
        walsh_phi_audit=phi_audit,
        wave_dispersion=disp,
        radial_velocity=vel,
        standing_wave=stand,
        k4_interference=interf,
        chi_commutator=chi_comm,
        krawtchouk_rows=kraw,
        compatibility=compat,
        rg_rows=rg,
        percolation=perc,
        dyadic_pairs=pairs,
        octave_response=resp,
        word_law=word,
        even_harmonics=even_h,
        square_root_audit=sr_audit,
        gates=gates,
        gate_kinds=gate_kinds,
    )
