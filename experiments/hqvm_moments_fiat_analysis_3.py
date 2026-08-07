#!/usr/bin/env python3
"""
hQVM Moments unified analysis (part 3/3): the (tick, fiber, balance) parameter
datatype, sections 17-30.

Role: measures the native coordinate format for 64-wide parameter blocks.
Sections 17-23: native projection energies, application fidelity, tick
trajectory, quantization, balance sector, and model-scale accounting.
Sections 24-25: event-conditioned prediction imprint on the hQVM ledger stream.
Sections 26-27: generalization bound and depth law.
Sections 28-30: temporal compilation of the native law, the (u,v) gyration split,
and a learned context-to-byte controller that drives the kernel on held-out
data. The final sections test the Gyroscopic inversion: intelligence is the
learned byte schedule, the computer is the kernel that executes it.

Inputs: none (deterministic, fixed seeds).
Outputs: printed tables, counts, and PASS/FAIL checks only.

Companions:
  hqvm_moments_fiat_analysis_1.py  -- shared library and sections 1-8
  hqvm_moments_fiat_analysis_2.py  -- sections 9-16
  hqvm_moments_fiat_analysis_run.py  -- full study runner
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gyroscopic.hQVM.constants import OMEGA_SIZE  # noqa: E402
from gyroscopic.hQVM.family import (  # noqa: E402
    byte_from_family_micro,
    chirality_uv,
    intron_family_d,
    intron_from_byte,
    mask_d,
    q_word_d,
    rest_uv,
    step_uv,
)

from hqvm_moments_fiat_analysis_1 import _pass  # noqa: E402

N = 64  # chirality register width, fiber count

POPCOUNT = np.array([bin(i).count("1") for i in range(N)])
XOR_INDEX = np.arange(N)[:, None] ^ np.arange(N)[None, :]

MODEL_PARAMS = 10**12
BLOCK_PARAMS = N * N


def wht64(x: np.ndarray) -> np.ndarray:
    h = x.astype(np.float64).copy()
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            a = h[i : i + step].copy()
            b = h[i + step : i + 2 * step].copy()
            h[i : i + step] = a + b
            h[i + step : i + 2 * step] = a - b
        step *= 2
    return h


def circulant_projection(W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    us = np.arange(N)
    c = np.array([W[us, us ^ d].mean() for d in range(N)])
    return c[XOR_INDEX], c


def radial_projection(W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = np.zeros(7)
    for k in range(7):
        r[k] = W[POPCOUNT[XOR_INDEX] == k].mean()
    return r[POPCOUNT[XOR_INDEX]], r


def apply_circulant(c: np.ndarray, x: np.ndarray) -> np.ndarray:
    return wht64(wht64(x) * wht64(c)) / N


def frob(W: np.ndarray) -> float:
    return float(np.sqrt((W**2).sum()))


def energy_ratio(P: np.ndarray, W: np.ndarray) -> float:
    return frob(P) / frob(W)


def block_archetypes(seed: int = 2026) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    out: dict[str, np.ndarray] = {}
    out["gaussian"] = rng.standard_normal((N, N))
    a = rng.standard_normal((N, 8))
    b = rng.standard_normal((N, 8))
    out["rank-8"] = a @ b.T
    q = rng.standard_normal((N, 8))
    k = rng.standard_normal((N, 8))
    s = (q @ k.T) / np.sqrt(8.0)
    s = np.exp(s - s.max(axis=1, keepdims=True))
    out["attention-8"] = s / s.sum(axis=1, keepdims=True)
    out["permutation"] = np.eye(N)[rng.permutation(N)]
    c = rng.standard_normal(N)
    out["circulant-exact"] = c[XOR_INDEX]
    r = rng.standard_normal(7)
    out["radial-exact"] = r[POPCOUNT[XOR_INDEX]]
    out["lowrank+noise"] = (a @ b.T) + 0.1 * rng.standard_normal((N, N))
    out["identity"] = np.eye(N)
    return out


# ---------------------------------------------------------------------------
# 17. Datatype coordinates
# ---------------------------------------------------------------------------


def section_coordinates() -> None:
    print("\n17. DATATYPE COORDINATES")
    print("=" * 5)
    print(f"  block grain: {N}x{N} = {BLOCK_PARAMS} parameters")
    print(f"  fiber coordinate (space): {N} spectral multipliers on GF(2)^6")
    print(f"  balance coordinate (curve): 7 Krawtchouk eigenvalues, bulk sector shells 1-5")
    print(f"  tick coordinate (time): depth integer, composition is multiplier exponentiation")
    native_scalars = N + 7
    print(f"  native scalars per block: {N} + 7 = {native_scalars}")
    print(f"  scalar ratio block/native: {BLOCK_PARAMS / native_scalars:.2f}x")
    xi = np.arange(N, dtype=np.float64)
    roundtrip = wht64(wht64(xi)) / N
    print(f"  WHT64 self-inverse (unnormalized x64)  {_pass(np.allclose(roundtrip, xi, atol=1e-12))}")
    r = np.zeros(7)
    r[3] = 1.0
    radial = r[POPCOUNT[XOR_INDEX]]
    P, c = circulant_projection(radial)
    print(f"  radial subset of circulant class  {_pass(np.allclose(P, radial))}")
    rng = np.random.default_rng(7)
    W = rng.standard_normal((N, N))
    P_t, _ = circulant_projection(W)
    expect = np.sqrt(N / BLOCK_PARAMS)
    r_t = energy_ratio(P_t, W)
    print(f"  gaussian fiber energy R={r_t:.4f}  expect sqrt(64/4096)={expect:.4f}  {_pass(abs(r_t - expect) < 0.01)}")


# ---------------------------------------------------------------------------
# 18. Native projections on block archetypes
# ---------------------------------------------------------------------------


def section_projections() -> None:
    print("\n18. NATIVE PROJECTIONS (BLOCK ARCHETYPES)")
    print("=" * 5)
    blocks = block_archetypes()
    print(f"  {'archetype':16} {'R_fiber':>8} {'R_balance':>9} {'defect%':>8}")
    for name, W in blocks.items():
        P_t, _ = circulant_projection(W)
        P_r, _ = radial_projection(W)
        r_t = energy_ratio(P_t, W)
        r_r = energy_ratio(P_r, W)
        defect = 100.0 * (1.0 - r_t**2)
        print(f"  {name:16} {r_t:8.4f} {r_r:9.4f} {defect:8.2f}")
    for name in ("circulant-exact", "radial-exact"):
        P_t, _ = circulant_projection(blocks[name])
        print(f"  {name} fully native  {_pass(abs(energy_ratio(P_t, blocks[name]) - 1.0) < 1e-12)}")


# ---------------------------------------------------------------------------
# 19. Application fidelity
# ---------------------------------------------------------------------------


def section_application() -> None:
    print("\n19. APPLICATION FIDELITY (NATIVE PART ONLY)")
    print("=" * 5)
    blocks = block_archetypes()
    rng = np.random.default_rng(99)
    print(f"  {'archetype':16} {'cos_mean':>9} {'cos_min':>9} {'relF_err':>9}")
    for name, W in blocks.items():
        P_t, c = circulant_projection(W)
        cos_vals = []
        for _ in range(50):
            x = rng.standard_normal(N)
            y_full = W @ x
            y_nat = apply_circulant(c, x)
            num = float(y_full @ y_nat)
            den = float(np.linalg.norm(y_full) * np.linalg.norm(y_nat))
            cos_vals.append(num / den if den > 0 else 1.0)
        rel_err = frob(W - P_t) / frob(W)
        print(
            f"  {name:16} {np.mean(cos_vals):9.4f} {np.min(cos_vals):9.4f} {rel_err:9.4f}"
        )
    P_t, c = circulant_projection(blocks["circulant-exact"])
    x = rng.standard_normal(N)
    exact = np.allclose(blocks["circulant-exact"] @ x, apply_circulant(c, x), atol=1e-10)
    print(f"  circulant application exact via WHT  {_pass(exact)}")


# ---------------------------------------------------------------------------
# 20. Tick trajectory (depth composition)
# ---------------------------------------------------------------------------


def section_tick_trajectory() -> None:
    print("\n20. TICK TRAJECTORY (DEPTH COMPOSITION)")
    print("=" * 5)
    rng = np.random.default_rng(4242)
    c = rng.standard_normal(N)
    phi = wht64(c)
    scale = 0.9 / np.abs(phi).max()
    c = c * scale
    phi = phi * scale
    W = c[XOR_INDEX]
    print(f"  spectral radius scaled to 0.9")
    print(f"  {'depth n':>8} {'max|err|':>12} {'dense ops':>12} {'native ops':>11}")
    ok = True
    for n in (2, 4, 8, 16, 64, 256):
        Wn_dense = np.linalg.matrix_power(W, n)
        cn = wht64((phi**n) * wht64(np.eye(N)[0])) / N
        Wn_native = cn[XOR_INDEX]
        err = float(np.abs(Wn_dense - Wn_native).max())
        ok = ok and err < 1e-8
        dense_ops = int(np.ceil(np.log2(n))) * N**3
        native_ops = 2 * 384 + int(np.ceil(np.log2(n))) * N + N
        print(f"  {n:8d} {err:12.3e} {dense_ops:12d} {native_ops:11d}")
    print(f"  depth powers exact (n<=256)  {_pass(ok)}")
    stored_dense = 256 * BLOCK_PARAMS
    stored_native = N + 8
    print(f"  256-deep trajectory storage: dense {stored_dense} params vs native {stored_native} scalars")
    print(f"  trajectory compression {stored_dense / stored_native:.1f}x  {_pass(stored_native < stored_dense // 1000)}")


# ---------------------------------------------------------------------------
# 21. Multiplier quantization
# ---------------------------------------------------------------------------


def section_quantization() -> None:
    print("\n21. MULTIPLIER QUANTIZATION")
    print("=" * 5)
    rng = np.random.default_rng(555)
    c = rng.standard_normal(N)
    phi = wht64(c)
    phi_max = np.abs(phi).max()
    print(f"  phi max={phi_max:.2f}")
    for label, levels in (("int8", 127), ("int4", 7)):
        q = np.round(phi / phi_max * levels) / levels * phi_max
        c_hat = wht64(q) / N
        W = c[XOR_INDEX]
        W_hat = c_hat[XOR_INDEX]
        rel = frob(W_hat - W) / frob(W)
        print(f"  {label}: {N} multipliers -> {N * (8 if label == 'int8' else 4)} bits  relF_err={rel:.6f}")
        if label == "int8":
            print(f"  int8 relative error < 0.01  {_pass(rel < 0.01)}")
    r = rng.standard_normal(7)
    r_max = np.abs(r).max()
    q8 = np.round(r / r_max * 127) / 127 * r_max
    rel_r = float(np.abs(q8 - r).max() / r_max)
    print(f"  balance int8: 7 eigenvalues -> 56 bits  max_rel_err={rel_r:.6f}  {_pass(rel_r < 0.01)}")


# ---------------------------------------------------------------------------
# 22. Balance sector (STF bulk shells)
# ---------------------------------------------------------------------------


def section_balance_sector() -> None:
    print("\n22. BALANCE SECTOR (BULK SHELLS 1-5)")
    print("=" * 5)
    blocks = block_archetypes()
    bulk = (POPCOUNT[XOR_INDEX] >= 1) & (POPCOUNT[XOR_INDEX] <= 5)
    print(f"  shell populations: {[64 * math.comb(6, k) for k in range(7)]}")
    print(f"  bulk share of Omega: {3968 / 4096:.4f}")
    print(f"  {'archetype':16} {'R_balance':>9} {'R_bulk':>8} {'horizon%':>9}")
    for name, W in blocks.items():
        P_r, _ = radial_projection(W)
        r_r = energy_ratio(P_r, W)
        horizon_energy = float((P_r[~bulk] ** 2).sum())
        total_energy = float((P_r**2).sum())
        bulk_r = float(np.sqrt(1.0 - horizon_energy / total_energy)) * r_r if total_energy > 0 else 0.0
        h_pct = 100.0 * horizon_energy / total_energy if total_energy > 0 else 0.0
        print(f"  {name:16} {r_r:9.4f} {bulk_r:8.4f} {h_pct:9.2f}")
    P_r, _ = radial_projection(blocks["identity"])
    h_energy = float((P_r[~bulk] ** 2).sum())
    t_energy = float((P_r**2).sum())
    print(
        f"  identity radial horizon share={100.0 * h_energy / t_energy:.2f}%  "
        f"{_pass(h_energy / t_energy > 0.5)}"
    )


# ---------------------------------------------------------------------------
# 23. Model-scale accounting
# ---------------------------------------------------------------------------


def section_model_scale() -> None:
    print("\n23. MODEL-SCALE ACCOUNTING (1e12 PARAMETERS)")
    print("=" * 5)
    n_blocks = MODEL_PARAMS / BLOCK_PARAMS
    print(f"  blocks of {BLOCK_PARAMS}: {n_blocks:.6e}")
    fp32 = MODEL_PARAMS * 4
    print(f"  fp32 flat: {fp32:.6e} B = {fp32 / 1e12:.2f} TB")
    rows = [
        ("native int8 (64 fibers + 2B tick)", N + 2),
        ("native int4 (32 fibers + 2B tick)", N // 2 + 2),
        ("native int8 + balance (64+7+2)", N + 7 + 2),
        ("native int4 + balance", (N + 7) // 2 + 2),
    ]
    for label, per_block in rows:
        total = n_blocks * per_block
        print(
            f"  {label}: {per_block} B/block -> {total:.6e} B = {total / 1e9:.2f} GB  "
            f"ratio {fp32 / total:.0f}x"
        )
    per_block = N + 2
    total = n_blocks * per_block
    print(f"  int8 native fits one consumer drive  {_pass(total < 5e12)}")
    print(f"  compression > 100x vs fp32  {_pass(fp32 / total > 100)}")
    blocks = block_archetypes()
    unstructured = ["gaussian", "rank-8", "lowrank+noise"]
    structured = ["attention-8", "permutation", "circulant-exact", "radial-exact", "identity"]
    def mean_r(names: list[str]) -> float:
        vals = [energy_ratio(circulant_projection(blocks[k])[0], blocks[k]) for k in names]
        return float(np.mean(vals))
    print(f"  mean R_fiber unstructured: {mean_r(unstructured):.4f}")
    print(f"  mean R_fiber structured:   {mean_r(structured):.4f}")


# ---------------------------------------------------------------------------
# 24. Event-conditioned prediction imprint
# ---------------------------------------------------------------------------

FRAME = 4


def ledger_stream(corpus: bytes, d: int = 6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u, v = rest_uv(d)
    n = len(corpus)
    chi = np.empty(n + 1, dtype=np.int64)
    fam = np.empty(n, dtype=np.int64)
    q6 = np.empty(n, dtype=np.int64)
    chi[0] = chirality_uv(u, v, d)
    for i, b in enumerate(corpus):
        fam[i] = intron_family_d(intron_from_byte(b, d), d)
        q6[i] = q_word_d(b, d)
        u, v = step_uv(u, v, b, d)
        chi[i + 1] = chirality_uv(u, v, d)
    return chi, fam, q6


def train_family_ops(
    chi_in: np.ndarray,
    fam: np.ndarray,
    target: np.ndarray,
    steps: int = 1500,
    lr: float = 0.1,
) -> tuple[np.ndarray, list[float]]:
    W = np.zeros((4, N, N))
    mom = np.zeros_like(W)
    vel = np.zeros_like(W)
    n = len(target)
    losses: list[float] = []
    for step in range(1, steps + 1):
        grad = np.zeros_like(W)
        loss = 0.0
        for ff in range(4):
            m = fam == ff
            xm, ym = chi_in[m], target[m]
            logits = W[ff][:, xm].T
            logits -= logits.max(axis=1, keepdims=True)
            p = np.exp(logits)
            p /= p.sum(axis=1, keepdims=True)
            loss += float(-np.log(p[np.arange(len(ym)), ym] + 1e-12).sum())
            d = p
            d[np.arange(len(ym)), ym] -= 1.0
            np.add.at(grad[ff].T, xm, d)
        loss /= n
        losses.append(loss)
        mom = 0.9 * mom + 0.1 * grad
        vel = 0.999 * vel + 0.001 * grad * grad
        W -= lr * (mom / (1 - 0.9**step)) / (np.sqrt(vel / (1 - 0.999**step)) + 1e-8)
    return W, losses


def analytic_family_ops(
    chi_in: np.ndarray, fam: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, float]:
    A = np.zeros((4, N, N))
    floor = 0.0
    for ff in range(4):
        m = fam == ff
        counts = np.zeros((N, N))
        np.add.at(counts, (target[m], chi_in[m]), 1.0)
        cols = counts.sum(axis=0, keepdims=True)
        cols[cols == 0] = 1.0
        A[ff] = counts / cols
        n_f = float(m.sum())
        total = counts.sum()
        if total > 0:
            p_col = counts.sum(axis=1) / total
            cond = counts / cols
            h = 0.0
            for c in range(N):
                col_total = counts[:, c].sum()
                if col_total > 0:
                    p = cond[:, c]
                    h += (col_total / total) * float(-(p[p > 0] * np.log(p[p > 0])).sum())
            floor += (n_f / len(target)) * h
    return A, floor


def centered_r_fiber(ops: np.ndarray) -> float:
    vals = []
    for W in ops:
        Wc = W - W.mean(axis=0, keepdims=True)
        nf = frob(Wc)
        if nf > 0:
            vals.append(energy_ratio(circulant_projection(Wc)[0], Wc))
    return float(np.mean(vals)) if vals else 0.0


def heldout_loss(ops: np.ndarray, chi_in: np.ndarray, fam: np.ndarray, target: np.ndarray) -> float:
    loss = 0.0
    for ff in range(4):
        m = fam == ff
        if not m.any():
            continue
        xm, ym = chi_in[m], target[m]
        logits = ops[ff][:, xm].T
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        loss += float(-np.log(p[np.arange(len(ym)), ym] + 1e-12).sum())
    return loss / len(target)


def stochastic_ops(W: np.ndarray) -> np.ndarray:
    z = W - W.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def analytic_heldout(A: np.ndarray, chi_in: np.ndarray, fam: np.ndarray, target: np.ndarray) -> float:
    loss = 0.0
    for ff in range(4):
        m = fam == ff
        if not m.any():
            continue
        p = A[ff][target[m], chi_in[m]]
        loss += float(-np.log(p + 1e-12).sum())
    return loss / len(target)


def section_event_imprint(
    chi: np.ndarray,
    fam: np.ndarray,
    q6: np.ndarray,
    corpus: bytes,
) -> dict[str, np.ndarray]:
    print("\n24. EVENT-CONDITIONED PREDICTION IMPRINT")
    print("=" * 5)
    chi_in, chi_out = chi[:-1], chi[1:]
    transport = bool(np.all(chi_out == (chi_in ^ q6)))
    print(f"  transport law chi' = chi XOR q6(b) over {len(corpus)} steps  {_pass(transport)}")
    n = len(chi_in)
    split = int(0.8 * n)
    rng_split = np.random.default_rng(3)
    perm_idx = rng_split.permutation(n)
    tr, te = perm_idx[:split], perm_idx[split:]
    rng = np.random.default_rng(5150)
    perms = np.array([rng.permutation(N) for _ in range(256)])
    target_ctrl = perms[np.frombuffer(corpus, dtype=np.uint8), chi_in]
    W_real, loss_real = train_family_ops(chi_in[tr], fam[tr], chi_out[tr])
    W_ctrl, loss_ctrl = train_family_ops(chi_in[tr], fam[tr], target_ctrl[tr])
    A_real, _ = analytic_family_ops(chi_in, fam, chi_out)
    A_ctrl, _ = analytic_family_ops(chi_in, fam, target_ctrl)
    ho_real = heldout_loss(W_real, chi_in[te], fam[te], chi_out[te])
    ho_ctrl = heldout_loss(W_ctrl, chi_in[te], fam[te], target_ctrl[te])
    ho_an_real = analytic_heldout(A_real, chi_in[te], fam[te], chi_out[te])
    ho_an_ctrl = analytic_heldout(A_ctrl, chi_in[te], fam[te], target_ctrl[te])
    print(f"  real law: train loss {loss_real[-1]:.3f}  held-out={ho_real:.3f}  analytic held-out={ho_an_real:.3f}")
    print(f"  random law: train loss {loss_ctrl[-1]:.3f}  held-out={ho_ctrl:.3f}  analytic held-out={ho_an_ctrl:.3f}")
    r_real, r_an = centered_r_fiber(W_real), centered_r_fiber(A_real)
    r_ctrl, r_an_ctrl = centered_r_fiber(W_ctrl), centered_r_fiber(A_ctrl)
    print(f"  centered R_fiber learned real={r_real:.4f}  analytic={r_an:.4f}")
    print(f"  centered R_fiber learned random={r_ctrl:.4f}  analytic={r_an_ctrl:.4f}")
    print(f"  real law imprints fiber-native operators (analytic R>0.80)  {_pass(r_an > 0.80)}")
    print(f"  learned real exceeds learned random (margin>0.30)  {_pass(r_real - r_ctrl > 0.30)}")
    print(f"  random law does not (R<0.30)  {_pass(r_ctrl < 0.30 and r_an_ctrl < 0.30)}")
    print(f"  real learned within 0.5 bits of full-corpus analytic held-out  {_pass(ho_real < ho_an_real + 0.5)}")
    print(f"  random learned within 0.5 bits of full-corpus analytic held-out  {_pass(ho_ctrl < ho_an_ctrl + 0.5)}")
    return {"W_real": W_real, "A_real": A_real, "chi": chi, "fam": fam, "q6": q6}


# ---------------------------------------------------------------------------
# 25. One clock: frame composition and balance profile
# ---------------------------------------------------------------------------


def translation_op(s: int) -> np.ndarray:
    T = np.zeros((N, N))
    T[np.arange(N) ^ s, np.arange(N)] = 1.0
    return T


def radial_profile(W: np.ndarray) -> np.ndarray:
    _, c = circulant_projection(W)
    r = np.zeros(7)
    for k in range(7):
        r[k] = c[POPCOUNT == k].sum()
    total = np.abs(r).sum()
    return r / total if total > 0 else r


def section_one_clock(data: dict[str, np.ndarray], corpus: bytes) -> None:
    print("\n25. ONE CLOCK (FRAME COMPOSITION) AND BALANCE PROFILE")
    print("=" * 5)
    chi, fam, q6 = data["chi"], data["fam"], data["q6"]
    W_real, A_real = data["W_real"], data["A_real"]
    n_frames = (len(corpus) // FRAME) * FRAME
    frames_f = fam[:n_frames].reshape(-1, FRAME)
    frames_q = q6[:n_frames].reshape(-1, FRAME)
    sums = np.bitwise_xor.reduce(frames_q, axis=1)
    chi_frames = chi[: n_frames + 1]
    closed = bool(
        np.all(chi_frames[FRAME::FRAME][: len(sums)] == (chi_frames[0:-FRAME:FRAME][: len(sums)] ^ sums))
    )
    print(f"  frame sum S = XOR q6 over {FRAME} ticks; chi(t+4) = chi(t) XOR S over {len(sums)} frames  {_pass(closed)}")
    rng = np.random.default_rng(9)
    sample = rng.integers(0, len(sums), size=min(500, len(sums)))
    P = stochastic_ops(W_real)
    err_learned, err_analytic, r_comp = [], [], []
    for i in sample:
        M = np.eye(N)
        Ma = np.eye(N)
        for t in range(FRAME):
            M = P[frames_f[i, t]] @ M
            Ma = A_real[frames_f[i, t]] @ Ma
        T = translation_op(int(sums[i]))
        err_learned.append(frob(M - T) / frob(T))
        err_analytic.append(frob(Ma - T) / frob(T))
        r_comp.append(energy_ratio(circulant_projection(M)[0], M))
    print(f"  composed frame operator: R_fiber mean={np.mean(r_comp):.4f}")
    print(f"  composed vs exact translation: learned relF={np.mean(err_learned):.4f}  analytic relF={np.mean(err_analytic):.4f}")
    print(f"  composition stays fiber-native (R>0.85)  {_pass(np.mean(r_comp) > 0.85)}")
    anchor_plus_sums = 1 + len(sums)
    raw_chi = len(corpus)
    print(f"  trajectory storage: anchor 1B + 1B/frame = {anchor_plus_sums} B vs raw chi stream {raw_chi} B  ratio={anchor_plus_sums / raw_chi:.4f}")
    print(f"  anchor+deltas reconstruct chi stream exactly  {_pass(closed)}")
    print(f"  {'family':>7} {'true radial (popcount q6)':>34} {'learned radial':>34} {'L2':>7}")
    for ff in range(4):
        m = fam == ff
        true_r = np.array([np.mean(POPCOUNT[q6[m]] == k) for k in range(7)])
        learned_r = radial_profile(P[ff])
        learned_r = np.abs(learned_r)
        learned_r = learned_r / learned_r.sum() if learned_r.sum() > 0 else learned_r
        l2 = float(np.linalg.norm(true_r - learned_r))
        print(
            f"  {ff:>7} {np.array2string(true_r, precision=3, separator=' '):>34} "
            f"{np.array2string(learned_r, precision=3, separator=' '):>34} {l2:7.4f}"
        )
    true_all = np.array([np.mean(POPCOUNT[q6] == k) for k in range(7)])
    bulk = float(true_all[1:6].sum())
    print(f"  bulk shells 1-5 share of q6 popcounts: {bulk:.4f}")
    print(f"  balance profile matches binomial bulk (share>0.9)  {_pass(bulk > 0.9)}")


# ---------------------------------------------------------------------------
# 26. Generalization bound (non-XOR Markov chain)
# ---------------------------------------------------------------------------


def section_generalization_bound() -> None:
    print("\n26. GENERALIZATION BOUND (NON-XOR MARKOV CHAIN)")
    print("=" * 5)
    rng = np.random.default_rng(2024)
    T = rng.random((N, N))
    T /= T.sum(axis=1, keepdims=True)
    s = 0
    traj = np.empty(40000, dtype=np.int64)
    for i in range(40000):
        s = rng.choice(N, p=T[s])
        traj[i] = s
    chi_in, chi_out = traj[:-1], traj[1:]
    n = len(chi_in)
    split = int(0.8 * n)
    rng_split = np.random.default_rng(3)
    perm_idx = rng_split.permutation(n)
    tr, te = perm_idx[:split], perm_idx[split:]
    W_gen, loss_gen = train_family_ops(chi_in[tr], np.zeros(n, dtype=np.int64)[tr], chi_out[tr])
    A_gen, _ = analytic_family_ops(chi_in, np.zeros(n, dtype=np.int64), chi_out)
    r_gen = centered_r_fiber(W_gen)
    r_an_gen = centered_r_fiber(A_gen)
    print(f"  non-XOR Markov chain on 64 states, 40000-step trajectory")
    print(f"  centered R_fiber learned={r_gen:.4f}  analytic={r_an_gen:.4f}")
    print(f"  XOR law analytic R_fiber (sec 24) = 0.8447")
    print(f"  non-XOR law analytic R_fiber = {r_an_gen:.4f}")
    print(f"  imprint generalizes beyond XOR (R>0.40)  {_pass(r_an_gen > 0.40)}")
    print(f"  imprint is XOR-specific (R<0.20)  {_pass(r_an_gen < 0.20)}")
    T_circ = circulant_projection(T)[0]
    r_T = energy_ratio(T_circ, T)
    print(f"  true transition matrix R_fiber={r_T:.4f} (circulant content of the law itself)")


# ---------------------------------------------------------------------------
# 27. Depth law (4-phase vs linear)
# ---------------------------------------------------------------------------


def section_depth_law() -> None:
    print("\n27. DEPTH LAW (4-PHASE VS LINEAR)")
    print("=" * 5)
    corpus = (_REPO_ROOT / "docs" / "Findings" / "Analysis_Gravity_Note.md").read_bytes()[:40000]
    chi, fam, q6 = ledger_stream(corpus)
    chi_in, chi_out = chi[:-1], chi[1:]
    W_real, _ = train_family_ops(chi_in, fam, chi_out)
    P = stochastic_ops(W_real)
    phis = [block_spectrum(P[ff])[1] for ff in range(4)]
    print(f"  4 family operators -> 4 spectral vectors (the K4 phases)")
    pair_corr = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            den = float(np.linalg.norm(phis[i]) * np.linalg.norm(phis[j]))
            pair_corr[i, j] = float(phis[i] @ phis[j]) / den if den > 0 else 0.0
    print(f"  pairwise spectral cosine matrix:")
    for i in range(4):
        print(f"    {np.array2string(pair_corr[i], precision=3, separator=' ')}")
    off_diag = pair_corr[~np.eye(4, dtype=bool)]
    print(f"  mean off-diagonal corr={off_diag.mean():.4f}  std={off_diag.std():.4f}")
    phi_sum = sum(phis)
    print(f"  sum of 4 spectra L2={float(np.linalg.norm(phi_sum)):.4f}  (zero = 4-phase cancellation)")
    print(f"  4-phase cancellation (sum L2 < 0.5 x individual)  {_pass(float(np.linalg.norm(phi_sum)) < 0.5 * float(np.mean([np.linalg.norm(p) for p in phis])))}")
    diffs = [phis[(ff + 1) % 4] - phis[ff] for ff in range(4)]
    diff_corr = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            den = float(np.linalg.norm(diffs[i]) * np.linalg.norm(diffs[j]))
            diff_corr[i, j] = float(diffs[i] @ diffs[j]) / den if den > 0 else 0.0
    print(f"  delta correlation matrix (rows = delta_i = phi_(i+1) - phi_i):")
    for i in range(4):
        print(f"    {np.array2string(diff_corr[i], precision=3, separator=' ')}")
    cyclic = float(np.mean([diff_corr[ff, (ff + 1) % 4] for ff in range(4)]))
    print(f"  mean cyclic delta corr (delta_i vs delta_(i+1))={cyclic:.4f}")
    print(f"  depth deltas show 4-phase structure (cyclic corr > 0.3)  {_pass(cyclic > 0.3)}")


def block_spectrum(W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, c = circulant_projection(W)
    return c, wht64(c)


def section_temporal_compilation(
    chi: np.ndarray, fam: np.ndarray, corpus: bytes, d: int = 6
) -> None:
    print("\n28. TEMPORAL COMPILATION OF THE NATIVE LAW")
    print("=" * 5)
    q = np.array([q_word_d(int(b), d) for b in corpus], dtype=np.int64)
    chi_in, chi_out = chi[:-1], chi[1:]
    n = len(q)

    law_ok = int(np.sum(chi_out == (chi_in ^ q)))
    print(f"  translation law chi' = chi ^ q(byte): {law_ok}/{n} steps")

    q_all = {q_word_d(b, d) for b in range(256)}
    print(f"  distinct q over 256 bytes: {len(q_all)}/{N}")
    full_cov = True
    for ff in range(4):
        fam_bytes = [b for b in range(256) if intron_family_d(intron_from_byte(b, d), d) == ff]
        cov = len({q_word_d(b, d) for b in fam_bytes})
        full_cov = full_cov and cov == N
        print(f"  family {ff}: {len(fam_bytes)} bytes -> {cov}/{N} translations")
    print(f"  every family covers all translations one-step  {_pass(full_cov)}")

    rng = np.random.default_rng(11)
    violations = 0
    for depth in range(1, 9):
        reached: set[int] = set()
        for _ in range(200):
            word = rng.integers(0, 256, size=depth)
            diffs = set()
            for x in range(N):
                u, v = x, 0
                for b in word:
                    u, v = step_uv(u, v, int(b), d)
                diffs.add(chirality_uv(u, v, d) ^ x)
                if len(diffs) > 1:
                    break
            if len(diffs) != 1:
                violations += 1
            else:
                reached.add(diffs.pop())
        print(f"  depth {depth}: translations reached {len(reached)}/{N}")
    print(f"  non-translation fixed words: {violations}")
    print(f"  fixed words collapse to translations on chi  {_pass(violations == 0)}")

    cnt_all = np.bincount(q, minlength=N) / n
    shared = float(cnt_all.max())
    ent_all = float(-(cnt_all[cnt_all > 0] * np.log2(cnt_all[cnt_all > 0])).sum())
    fam_fid = 0.0
    ent_resid = 0.0
    bayes = 0.0
    for ff in range(4):
        m = fam == ff
        w = float(m.mean())
        cnt = np.bincount(q[m], minlength=N) / m.sum()
        fam_fid += w * float(cnt.max())
        nz = cnt[cnt > 0]
        ent_resid += w * float(-(nz * np.log2(nz)).sum())
        joint = np.zeros((N, N))
        np.add.at(joint, (chi_in[m], chi_out[m]), 1)
        bayes += float(joint.max(axis=1).sum())
    bayes /= n
    print(f"  fidelity shared byte (8 bits):          {shared:.4f}")
    print(f"  fidelity family controller (32 bits):   {fam_fid:.4f}")
    print(f"  fidelity deterministic Bayes (chi,fam): {bayes:.4f}")
    print(f"  fidelity true-byte steering:            {law_ok / n:.4f}")
    print(f"  chance:                                 {1 / N:.4f}")
    print(f"  H(q)={ent_all:.4f} bits  H(q|fam)={ent_resid:.4f} bits")
    print(f"  Bayes - fam controller gap (chi-dependence of q within family): {bayes - fam_fid:.4f}")
    print(f"  fam controller near Bayes (gap < 0.01)  {_pass(bayes - fam_fid < 0.01)}")
    print(f"  fam controller beats shared byte  {_pass(fam_fid > shared)}")
    print(f"  description operator bank 4x64x64 fp32: {4 * N * N * 32} bits")
    print(f"  description native law: 4x{N} q-categoricals; fam controller: 32 bits + kernel")


def section_uv_routing(
    chi: np.ndarray, fam: np.ndarray, q6: np.ndarray, corpus: bytes, d: int = 6
) -> None:
    print("\n29. DYNAMIC TAPE ON U X V: FIDELITY AND DEPTH")
    print("=" * 5)
    u, v = rest_uv(d)
    n = len(corpus)
    U = np.empty(n + 1, dtype=np.int64)
    V = np.empty(n + 1, dtype=np.int64)
    U[0], V[0] = u, v
    for i, b in enumerate(corpus):
        u, v = step_uv(u, v, int(b), d)
        U[i + 1], V[i + 1] = u, v
    chi_in, chi_out = chi[:-1], chi[1:]
    q = np.array([q_word_d(int(b), d) for b in corpus], dtype=np.int64)
    s_in, s_out = POPCOUNT[chi_in], POPCOUNT[chi_out]

    def fixed_tape_fid(starts: tuple, depth: int) -> float:
        rng = np.random.default_rng(100 + depth)
        hits = 0
        for i in range(n):
            ui, vi = U[i], V[i]
            for b in rng.integers(0, 256, size=depth):
                ui, vi = step_uv(ui, vi, int(b), d)
            hits += int((ui ^ vi) == chi_out[i])
        return hits / n

    def controller_tape_fid(starts: tuple, ctrl_bits: int) -> float:
        if ctrl_bits == 0:
            q6_best = np.zeros(n, dtype=np.int64)
        elif ctrl_bits == 2:
            q6_best = q6
        elif ctrl_bits == 6:
            q6_best = q6
        elif ctrl_bits == 8:
            q6_best = q6
        else:
            q6_best = q6
        hits = 0
        for i in range(n):
            ui, vi = U[i], V[i]
            if ctrl_bits == 0:
                b = 0
            elif ctrl_bits == 2:
                b = byte_from_family_micro(fam[i], q6[i], d)
            elif ctrl_bits == 8:
                b = int(corpus[i])
            else:
                b = byte_from_family_micro(fam[i], q6[i], d)
            ui, vi = step_uv(ui, vi, b, d)
            hits += int((ui ^ vi) == chi_out[i])
        return hits / n

    for depth in (1, 2, 3, 4):
        fid = fixed_tape_fid((U, V), depth)
        print(f"  fixed tape depth {depth}: chi-prediction {fid:.4f}")
    for ctrl in (0, 2, 8):
        fid = controller_tape_fid((U, V), ctrl)
        print(f"  controller bits {ctrl}: chi-prediction {fid:.4f}")
    print(f"  chi-only one-step (translation): chi-prediction {np.mean((chi_in ^ q) == chi_out):.4f}")

    print("\n  shell-target (u,v) controller ladders")
    for target, label in ((s_out, "shell"), (U[1:], "u"), (V[1:], "v")):
        print(f"  -- target={label}")
        for ctrl in (0, 2, 8):
            hits = 0
            for i in range(n):
                ui, vi = U[i], V[i]
                if ctrl == 0:
                    b = 0
                elif ctrl == 2:
                    b = byte_from_family_micro(fam[i], 0, d)
                elif ctrl == 8:
                    b = int(corpus[i])
                ui, vi = step_uv(ui, vi, b, d)
                if label == "shell":
                    hits += int(POPCOUNT[ui ^ vi] == target[i])
                elif label == "u":
                    hits += int(ui == target[i])
                else:
                    hits += int(vi == target[i])
            print(f"    ctrl bits {ctrl}: {hits / n:.4f}")

    print("\n  percolation depth test: reach full u-state from a single anchor")
    m = mask_d(d)
    rng = np.random.default_rng(50)
    for depth in range(1, 13):
        covered = set()
        for _ in range(1000):
            word = rng.integers(0, 256, size=depth)
            ui, vi = 0, 0
            for b in word:
                ui, vi = step_uv(ui, vi, int(b), d)
            covered.add(ui)
        print(f"    depth {depth:2d}: reached {len(covered)}/{m + 1} u-states")

    print("\n  shell-as-readout: does (u,v) gyration produce non-trivial shell mappings")
    shell_target = s_out
    for ctrl in (0, 2, 8):
        hits = 0
        for i in range(n):
            ui, vi = U[i], V[i]
            if ctrl == 0:
                b = 0
            elif ctrl == 2:
                b = byte_from_family_micro(fam[i], 0, d)
            else:
                b = int(corpus[i])
            ui, vi = step_uv(ui, vi, b, d)
            hits += int(POPCOUNT[ui ^ vi] == shell_target[i])
        print(f"    ctrl bits {ctrl}: shell-prediction {hits / n:.4f}")


def section_learned_controller(corpus: bytes, d: int = 6) -> None:
    print("\n30. LEARNED CONTROLLER DRIVES THE KERNEL")
    print("=" * 5)
    u, v = rest_uv(d)
    n = len(corpus)
    U = np.empty(n + 1, dtype=np.int64)
    V = np.empty(n + 1, dtype=np.int64)
    U[0], V[0] = u, v
    for i, b in enumerate(corpus):
        u, v = step_uv(u, v, int(b), d)
        U[i + 1], V[i + 1] = u, v

    states = (U[:-1] << d) | V[:-1]
    targets = np.array(list(corpus), dtype=np.int64)
    split = n // 2
    train_s, test_s = states[:split], states[split:]
    train_t, test_t = targets[:split], targets[split:]
    m = mask_d(d)

    lut = np.zeros((OMEGA_SIZE, 256), dtype=np.int64)
    np.add.at(lut, (train_s, train_t), 1)
    lut_byte = lut.argmax(axis=1)
    print(f"  LUT byte accuracy train={np.mean(lut_byte[train_s] == train_t):.4f} test={np.mean(lut_byte[test_s] == test_t):.4f}")

    def featurize(s: np.ndarray, ctx: np.ndarray | None = None) -> np.ndarray:
        uu, vv = s >> d, s & m
        x = np.zeros((len(s), 2 * d), dtype=np.float32)
        for i in range(d):
            x[:, i] = (uu >> i) & 1
            x[:, d + i] = (vv >> i) & 1
        if ctx is None:
            return x
        onehot = np.zeros((len(s), ctx.shape[1] * 256), dtype=np.float32)
        for j in range(ctx.shape[1]):
            onehot[np.arange(len(s)), ctx[:, j] + 256 * j] = 1
        return np.concatenate([x, onehot], axis=1)

    def train_mlp(X: np.ndarray, y: np.ndarray, hidden: int, epochs: int, lr0: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        X = np.asarray(X, dtype=np.float32)
        W1 = rng.normal(0, 0.5, size=(X.shape[1], hidden)).astype(np.float32)
        b1 = np.zeros(hidden, dtype=np.float32)
        W2 = rng.normal(0, 0.1, size=(hidden, 256)).astype(np.float32)
        b2 = np.zeros(256, dtype=np.float32)
        for epoch in range(epochs):
            lr = lr0 * (1 - epoch / epochs)
            perm = rng.permutation(len(X))
            for i in range(0, len(X), batch):
                idx = perm[i:i + batch]
                x = X[idx]
                yy = y[idx]
                h = np.maximum(0, x @ W1 + b1)
                logits = h @ W2 + b2
                logp = logits - logits.max(axis=1, keepdims=True)
                p = np.exp(logp)
                p /= p.sum(axis=1, keepdims=True)
                grad = p.copy()
                grad[np.arange(len(yy)), yy] -= 1
                grad /= len(yy)
                dW2 = h.T @ grad
                db2 = grad.sum(axis=0)
                dh = grad @ W2.T
                dh[h <= 0] = 0
                dW1 = x.T @ dh
                db1 = dh.sum(axis=0)
                W2 -= lr * dW2
                b2 -= lr * db2
                W1 -= lr * dW1
                b1 -= lr * db1
            if epoch % 10 == 0:
                acc = np.mean((np.maximum(0, X @ W1 + b1) @ W2 + b2).argmax(axis=1) == y)
                print(f"    epoch {epoch:2d}: train acc={acc:.4f}")
        return W1, b1, W2, b2

    batch = 256
    X_train, X_test = featurize(train_s), featurize(test_s)
    W1, b1, W2, b2 = train_mlp(X_train, train_t, 64, 30, 0.8, 123)
    h_test = np.maximum(0, X_test @ W1 + b1)
    mlp_byte = (h_test @ W2 + b2).argmax(axis=1)
    mlp_acc = np.mean(mlp_byte == test_t)
    print(f"  state-only MLP byte accuracy test={mlp_acc:.4f}")

    def make_ctx(k: int, start: int, end: int) -> np.ndarray:
        arr = np.zeros((end - start, k), dtype=np.int64)
        for i in range(start, end):
            for j in range(k):
                pos = i - 1 - j
                if pos >= 0:
                    arr[i - start, j] = targets[pos]
        return arr

    ctx4_train = make_ctx(4, 0, split)
    ctx4_test = make_ctx(4, split, n)
    X4_train = np.zeros((split, 4 * 256), dtype=np.float32)
    X4_test = np.zeros((n - split, 4 * 256), dtype=np.float32)
    for j in range(4):
        X4_train[np.arange(split), ctx4_train[:, j] + 256 * j] = 1
        X4_test[np.arange(n - split), ctx4_test[:, j] + 256 * j] = 1
    W14, b14, W24, b24 = train_mlp(X4_train, train_t, 128, 40, 0.8, 789)
    h4_test = np.maximum(0, X4_test @ W14 + b14)
    ctx4_byte = (h4_test @ W24 + b24).argmax(axis=1)
    ctx4_acc = np.mean(ctx4_byte == test_t)
    print(f"  4-byte context-only MLP byte accuracy test={ctx4_acc:.4f}")

    def kernel_fidelity(pred_bytes: np.ndarray, label: str) -> None:
        hits = {"byte": 0, "u": 0, "v": 0, "chi": 0, "shell": 0}
        for i in range(split, n):
            ui, vi = U[i], V[i]
            b = int(pred_bytes[i - split])
            ui, vi = step_uv(ui, vi, b, d)
            hits["byte"] += int(b == targets[i])
            hits["u"] += int(ui == U[i + 1])
            hits["v"] += int(vi == V[i + 1])
            hits["chi"] += int(chirality_uv(ui, vi, d) == chirality_uv(U[i + 1], V[i + 1], d))
            hits["shell"] += int(POPCOUNT[chirality_uv(ui, vi, d)] == POPCOUNT[chirality_uv(U[i + 1], V[i + 1], d)])
        print(f"  {label}:")
        for k, v in hits.items():
            print(f"    {k}={v / (n - split):.4f}")

    kernel_fidelity(lut_byte[test_s], "LUT controller")
    kernel_fidelity(mlp_byte, "state-only MLP controller")
    kernel_fidelity(ctx4_byte, "4-byte context-only MLP controller")
    lut_byte_acc = float(np.mean(lut_byte[test_s] == test_t))
    chi_fid_mlp = float(np.mean([chirality_uv(*step_uv(U[i], V[i], int(mlp_byte[i - split]), d), d) == chirality_uv(U[i + 1], V[i + 1], d) for i in range(split, n)]))
    chi_fid_ctx4 = float(np.mean([chirality_uv(*step_uv(U[i], V[i], int(ctx4_byte[i - split]), d), d) == chirality_uv(U[i + 1], V[i + 1], d) for i in range(split, n)]))
    print(f"  context-only MLP beats LUT on byte  {_pass(ctx4_acc > lut_byte_acc)}")
    print(f"  context-only MLP beats state-only on byte  {_pass(ctx4_acc > mlp_acc)}")
    print(f"  4-byte context-only MLP beats state-only on chi  {_pass(chi_fid_ctx4 > chi_fid_mlp)}")
    shell_fid_ctx4 = float(np.mean([POPCOUNT[chirality_uv(*step_uv(U[i], V[i], int(ctx4_byte[i - split]), d), d)] == POPCOUNT[chirality_uv(U[i + 1], V[i + 1], d)] for i in range(split, n)]))
    print(f"  4-byte context-only MLP shell > 0.5  {_pass(shell_fid_ctx4 > 0.5)}")


# ---------------------------------------------------------------------------
# Main (part 3)
# ---------------------------------------------------------------------------


def main() -> None:
    print("hQVM MOMENTS UNIFIED ANALYSIS (3/3)")
    print("=" * 5)
    section_coordinates()
    section_projections()
    section_application()
    section_tick_trajectory()
    section_quantization()
    section_balance_sector()
    section_model_scale()
    corpus = (_REPO_ROOT / "docs" / "Findings" / "Analysis_Gravity_Note.md").read_bytes()[:40000]
    print(f"\n  corpus: {len(corpus)} bytes")
    chi, fam, q6 = ledger_stream(corpus)
    data = section_event_imprint(chi, fam, q6, corpus)
    section_one_clock(data, corpus)
    section_generalization_bound()
    section_depth_law()
    section_temporal_compilation(chi, fam, corpus)
    section_uv_routing(chi, fam, q6, corpus)
    section_learned_controller(corpus)
    print("\nPART 3 DONE")
    print("=" * 5)


if __name__ == "__main__":
    main()
