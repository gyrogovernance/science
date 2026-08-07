#!/usr/bin/env python3
"""
hQVM Moments unified analysis (part 2/2): sections 9-16.

Role: parity commitments, medium policy, golden vectors, genealogy integration,
QR receipt layouts (17B / 18B), FNV-1a native profile, convergence accounting,
and daily geometry measurements. Imports shared library from part 1.

Inputs: none (deterministic kernel + fixed golden vectors).
Outputs: printed tables, counts, and PASS/FAIL checks only.

Companions:
  hqvm_moments_fiat_analysis_1.py  -- shared library and sections 1-8
  hqvm_moments_fiat_analysis_run.py  -- full study runner
"""

from __future__ import annotations

import struct
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gyroscopic.hQVM.api import (  # noqa: E402
    BYTES_BY_Q6,
    is_in_omega24,
    mask12_for_byte,
    pack_omega12,
    q_word6,
    q_word6_for_items,
    trajectory_parity_commitment,
    unpack_omega12,
)
from gyroscopic.hQVM.constants import (  # noqa: E402
    GENE_MAC_REST,
    OMEGA_SIZE,
    byte_to_intron,
    intron_family,
    intron_micro_ref,
)
from gyroscopic.hQVM.kernel import Gyroscopic  # noqa: E402

from hqvm_moments_fiat_analysis_1 import (  # noqa: E402
    DAYS_PER_YEAR,
    EPOCH_YEAR,
    FNV_OFFSET_BASIS,
    FNV_PRIME,
    GOLDEN_ALICE_ANCHOR,
    GOLDEN_BOB_ANCHOR,
    GOLDEN_MASK48,
    GOLDEN_META_ROOT,
    GOLDEN_PARITY_E,
    GOLDEN_PARITY_O,
    GOLDEN_PARITY_P,
    GOLDEN_PHI_A,
    GOLDEN_PHI_B,
    GOLDEN_SHELL_SEAL,
    HOURS_PER_DAY,
    MU_PER_HOUR,
    POPULATION,
    SECONDS_PER_HOUR,
    TIER_MULTIPLIERS,
    UHI_HOURS_PER_DAY,
    UHI_PER_DAY,
    UHI_PER_YEAR,
    Coverage,
    Grant,
    Shell,
    _pass,
    csm_total,
    extract_fiber_m12,
    extract_m12,
    fnv1a_64,
    frame_record,
    frame_sequence,
    identity_anchor,
    make_shell,
    meta_root_from_payloads,
    ntp_timestamp,
)

# ---------------------------------------------------------------------------
# QR receipt helpers
# ---------------------------------------------------------------------------

QR_BYTE_CAPACITY: dict[int, dict[str, int]] = {
    1: {"L": 17, "M": 14, "Q": 11, "H": 7},
    2: {"L": 32, "M": 26, "Q": 20, "H": 14},
}
QR_MODULES: dict[int, int] = {1: 21, 2: 25}
REGISTER_ATOM_BITS = 32


def parity24_for_payload(payload: bytes) -> int:
    o, e, _p = trajectory_parity_commitment(payload)
    return ((o & 0xFFF) << 12) | (e & 0xFFF)


def event_byte_for_payload(payload: bytes) -> int:
    q = q_word6_for_items(payload)
    return min(BYTES_BY_Q6[q])


def receipt_seal24(
    sec32: int, frac32: int, identity_state24: int, payload: bytes
) -> int:
    header = (
        struct.pack(">I", sec32)
        + struct.pack(">I", frac32)
        + identity_state24.to_bytes(3, "big")
    )
    return Gyroscopic().route_from_archetype(header + payload).state24


def encode_receipt(
    identity: bytes, payload: bytes, dt: datetime, event_byte: bool
) -> bytes:
    sec32, frac32 = ntp_timestamp(dt)
    id24 = Gyroscopic().route_from_archetype(identity).state24
    seal24 = receipt_seal24(sec32, frac32, id24, payload)
    rec = (
        struct.pack(">I", sec32)
        + struct.pack(">I", frac32)
        + id24.to_bytes(3, "big")
        + seal24.to_bytes(3, "big")
        + parity24_for_payload(payload).to_bytes(3, "big")
    )
    if event_byte:
        rec += bytes([event_byte_for_payload(payload)])
    return rec


def verify_receipt(rec: bytes, identity: bytes, payload: bytes) -> dict[str, bool]:
    sec32 = struct.unpack(">I", rec[0:4])[0]
    frac32 = struct.unpack(">I", rec[4:8])[0]
    id24 = int.from_bytes(rec[8:11], "big")
    seal24 = int.from_bytes(rec[11:14], "big")
    par24 = int.from_bytes(rec[14:17], "big")
    out = {
        "identity": id24 == Gyroscopic().route_from_archetype(identity).state24,
        "seal": seal24 == receipt_seal24(sec32, frac32, id24, payload),
        "parity": par24 == parity24_for_payload(payload),
    }
    if len(rec) == 18:
        out["event"] = rec[17] == event_byte_for_payload(payload)
    return out


def qr_fits(n_bytes: int) -> list[str]:
    fits = []
    for v, caps in QR_BYTE_CAPACITY.items():
        for ec, cap in caps.items():
            if cap >= n_bytes:
                fits.append(f"V{v}-{ec}")
    return fits


# ---------------------------------------------------------------------------
# FNV variants
# ---------------------------------------------------------------------------


def fnv1_64(data: bytes) -> int:
    h = FNV_OFFSET_BASIS
    for b in data:
        h = (h * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
        h ^= b
    return h


def fnv0_64(data: bytes) -> int:
    h = 0
    for b in data:
        h = (h * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
        h ^= b
    return h


# ---------------------------------------------------------------------------
# 9. Parity commitments
# ---------------------------------------------------------------------------


def section_parity() -> None:
    print("\n9. PARITY COMMITMENTS")
    print("=" * 5)
    payload = b"parity commitment test payload"
    c1 = trajectory_parity_commitment(payload)
    c2 = trajectory_parity_commitment(payload)
    print(f"  determinism  {_pass(c1 == c2)}")

    payload_list = list(range(40))
    orig_o, orig_e, orig_p = trajectory_parity_commitment(payload_list)
    slot_ok = True
    for i in range(len(payload_list)):
        m_old = mask12_for_byte(payload_list[i])
        b_new = None
        for c in range(256):
            if mask12_for_byte(c) != m_old:
                b_new = c
                break
        if b_new is None:
            slot_ok = False
            break
        tampered = payload_list.copy()
        tampered[i] = b_new
        new_o, new_e, new_p = trajectory_parity_commitment(tampered)
        if (i & 1) == 0:
            if new_o == orig_o:
                slot_ok = False
        else:
            if new_e == orig_e:
                slot_ok = False
        if (new_o, new_e, new_p) == (orig_o, orig_e, orig_p):
            slot_ok = False
    print(f"  mask-change slot sensitivity (n=40)  {_pass(slot_ok)}")

    o, e, p = trajectory_parity_commitment(b"structure test")
    struct_ok = (
        isinstance(o, int)
        and isinstance(e, int)
        and p in (0, 1)
        and 0 <= o < (1 << 12)
        and 0 <= e < (1 << 12)
        and p == len(b"structure test") % 2
    )
    print(f"  structure O=0x{o:03X} E=0x{e:03X} p={p}  {_pass(struct_ok)}")


# ---------------------------------------------------------------------------
# 10. Medium policy
# ---------------------------------------------------------------------------


def section_policy() -> None:
    print("\n10. MEDIUM POLICY")
    print("=" * 5)
    ident, anchor = identity_anchor("alice")
    shell_dup = Shell(
        header=b"test",
        grants=[
            Grant("alice", ident, anchor, 87_600),
            Grant("alice", ident, anchor, 87_600),
        ],
        total_capacity_mu=10**18,
    )
    shell_dup.compute_seal()
    ids = [g.identity_id for g in shell_dup.grants]
    print(f"  duplicate identity detectable  {_pass(len(ids) != len(set(ids)) and len(shell_dup.seal) == 6)}")

    shell_over = Shell(
        header=b"test",
        grants=[Grant("alice", ident, anchor, 1_000_000)],
        total_capacity_mu=500_000,
    )
    shell_over.compute_seal()
    print(
        f"  over-capacity used={shell_over.used_capacity_mu} free={shell_over.free_capacity_mu}  "
        f"{_pass(shell_over.used_capacity_mu > shell_over.total_capacity_mu and shell_over.free_capacity_mu < 0)}"
    )

    e1 = Shell(header=b"empty:2026", grants=[], total_capacity_mu=10**18)
    e1.compute_seal()
    e2 = Shell(header=b"empty:2026", grants=[], total_capacity_mu=10**18)
    e2.compute_seal()
    print(f"  empty shell seal={e1.seal}  {_pass(e1.seal == e2.seal and e1.used_capacity_mu == 0)}")


# ---------------------------------------------------------------------------
# 11. Golden vectors
# ---------------------------------------------------------------------------


def section_golden() -> None:
    print("\n11. GOLDEN VECTORS")
    print("=" * 5)
    _, alice = identity_anchor("alice")
    _, bob = identity_anchor("bob")
    print(f"  alice={alice} expect={GOLDEN_ALICE_ANCHOR}  {_pass(alice == GOLDEN_ALICE_ANCHOR)}")
    print(f"  bob={bob} expect={GOLDEN_BOB_ANCHOR}  {_pass(bob == GOLDEN_BOB_ANCHOR)}")

    shell = make_shell(b"golden:2026", [("alice", 87_600), ("bob", 175_200)])
    print(f"  shell={shell.seal} expect={GOLDEN_SHELL_SEAL}  {_pass(shell.seal == GOLDEN_SHELL_SEAL)}")

    bundles = [b"program:Alpha", b"program:Beta", b"program:Gamma"]
    root = meta_root_from_payloads(bundles)
    print(f"  meta_root={root} expect={GOLDEN_META_ROOT}  {_pass(root == GOLDEN_META_ROOT)}")

    fr = frame_record(0x00, 0x42, 0xAA, 0xFF)
    print(
        f"  frame={fr} expect=({GOLDEN_MASK48:#x},{GOLDEN_PHI_A},{GOLDEN_PHI_B})  "
        f"{_pass(fr == (GOLDEN_MASK48, GOLDEN_PHI_A, GOLDEN_PHI_B))}"
    )

    o, e, p = trajectory_parity_commitment(b"golden parity vector")
    print(
        f"  parity O=0x{o:03X} E=0x{e:03X} p={p} expect=(0x{GOLDEN_PARITY_O:03X},0x{GOLDEN_PARITY_E:03X},{GOLDEN_PARITY_P})  "
        f"{_pass((o, e, p) == (GOLDEN_PARITY_O, GOLDEN_PARITY_E, GOLDEN_PARITY_P))}"
    )


# ---------------------------------------------------------------------------
# 12. Genealogy integration
# ---------------------------------------------------------------------------


def section_genealogy_integration() -> None:
    print("\n12. GENEALOGY INTEGRATION")
    print("=" * 5)
    payload = b"genealogy replay consistency test payload!!"
    r1 = Gyroscopic()
    r1.step_bytes(payload)
    r2 = Gyroscopic()
    r2.step_bytes(payload)
    frames1 = frame_sequence(payload)
    frames2 = frame_sequence(payload)
    print(
        f"  replay state={r1.signature().state_hex} frames={len(frames1)}  "
        f"{_pass(r1.signature().state24 == r2.signature().state24 and frames1 == frames2)}"
    )

    payload2 = b"three layer certification"
    r = Gyroscopic()
    r.step_bytes(payload2)
    state = r.signature().state_hex
    frames = frame_sequence(payload2)
    parity = trajectory_parity_commitment(payload2)
    r_b = Gyroscopic()
    r_b.step_bytes(payload2)
    print(f"  layers state={state} frames={len(frames)} parity={parity}")
    print(
        f"  three-layer determinism  "
        f"{_pass(r_b.signature().state_hex == state and frame_sequence(payload2) == frames and trajectory_parity_commitment(payload2) == parity)}"
    )

    prefix = b"shared prefix data"
    suffix_a = b"branch A continuation"
    suffix_b = b"branch B continuation"
    ra = Gyroscopic()
    ra.step_bytes(prefix + suffix_a)
    rb = Gyroscopic()
    rb.step_bytes(prefix + suffix_b)
    frames_a = frame_sequence(prefix + suffix_a)
    frames_b = frame_sequence(prefix + suffix_b)
    parity_a = trajectory_parity_commitment(prefix + suffix_a)
    parity_b = trajectory_parity_commitment(prefix + suffix_b)
    prefix_n = len(prefix) // 4
    state_diff = ra.signature().state_hex != rb.signature().state_hex
    frames_diff = frames_a != frames_b
    parity_diff = parity_a != parity_b
    print(f"  fork state_diff={state_diff} frames_diff={frames_diff} parity_diff={parity_diff}")
    print(
        f"  prefix frames match then diverge  "
        f"{_pass(frames_a[:prefix_n] == frames_b[:prefix_n] and frames_diff)}"
    )

    inv_payload = b"inverse replay genealogy test"
    ri = Gyroscopic()
    ri.step_bytes(inv_payload)
    final = ri.signature().state24
    ri.step_bytes_inverse(inv_payload)
    print(
        f"  inverse genealogy  "
        f"{_pass(final != GENE_MAC_REST and ri.signature().state24 == GENE_MAC_REST)}"
    )

    part1 = bytes(range(0, 32))
    part2 = bytes(range(32, 64))
    combined = part1 + part2
    fc = frame_sequence(combined)
    f1 = frame_sequence(part1)
    f2 = frame_sequence(part2)
    print(
        f"  continuation concat frames={len(fc)}  "
        f"{_pass(len(fc) == 16 and fc[:8] == f1 and fc[8:] == f2)}"
    )


# ---------------------------------------------------------------------------
# 13. QR receipts
# ---------------------------------------------------------------------------


def section_qr_receipts() -> None:
    print("\n13. QR RECEIPTS")
    print("=" * 5)
    for v in (1, 2):
        caps = QR_BYTE_CAPACITY[v]
        m = QR_MODULES[v]
        print(
            f"  V{v}: modules={m}x{m}  data bytes L/M/Q/H="
            f"{caps['L']}/{caps['M']}/{caps['Q']}/{caps['H']}"
        )
    print(
        f"  V2-L data bytes={QR_BYTE_CAPACITY[2]['L']}  "
        f"register-atom bits={REGISTER_ATOM_BITS}"
    )

    identity = b"alice"
    payload = b"baseline_claim_user_12345"
    dt = datetime(2026, 8, 6, 0, 0, 0, tzinfo=timezone.utc)

    rec17 = encode_receipt(identity, payload, dt, event_byte=False)
    rec18 = encode_receipt(identity, payload, dt, event_byte=True)
    print(f"  L17 ({len(rec17)}B) fits: {qr_fits(len(rec17))}")
    print(f"  L18 ({len(rec18)}B) fits: {qr_fits(len(rec18))}")

    sec32 = struct.unpack(">I", rec18[0:4])[0]
    frac32 = struct.unpack(">I", rec18[4:8])[0]
    m12 = extract_m12(frac32)
    om = unpack_omega12(m12)
    print(f"  sec32=0x{sec32:08X} frac32=0x{frac32:08X}")
    print(
        f"  m12={m12} -> omega12 u6={om.u6} v6={om.v6} shell={om.shell}  "
        f"roundtrip {_pass(pack_omega12(om) == m12)}"
    )

    seal24 = int.from_bytes(rec18[11:14], "big")
    print(f"  seal24=0x{seal24:06X} on_omega {_pass(is_in_omega24(seal24))}")

    print(f"  L17 hex={rec17.hex()}")
    v17 = verify_receipt(rec17, identity, payload)
    print(f"  L17 verify {v17}  {_pass(all(v17.values()))}")

    eb = rec18[17]
    intron = byte_to_intron(eb)
    q = q_word6(eb)
    print(f"  L18 hex={rec18.hex()}")
    print(
        f"  event_byte=0x{eb:02X} intron=0x{intron:02X} "
        f"family={intron_family(intron)} micro={intron_micro_ref(intron)} q6=0x{q:02X}"
    )
    print(f"  event q6 == Q(payload)  {_pass(q == q_word6_for_items(payload))}")
    v18 = verify_receipt(rec18, identity, payload)
    print(f"  L18 verify {v18}  {_pass(all(v18.values()))}")

    tampered = bytes([payload[0] ^ 0x02]) + payload[1:]
    vt = verify_receipt(rec18, identity, tampered)
    caught = (
        vt["identity"]
        and not vt["seal"]
        and not vt["parity"]
        and not vt["event"]
    )
    print(f"  tampered payload verify {vt}  {_pass(caught)}")


# ---------------------------------------------------------------------------
# 14. FNV profile
# ---------------------------------------------------------------------------


def section_fnv_profile() -> None:
    print("\n14. FNV PROFILE")
    print("=" * 5)

    official_1a = [
        (b"", 0xCBF29CE484222325),
        (b"a", 0xAF63DC4C8601EC8C),
        (b"foobar", 0x85944171F73967E8),
        (bytes([0x48, 0x65, 0x6C, 0x6C, 0x6F, 0x21, 0x01, 0xFF, 0xED]), 0xBD51EA7094EE6FA1),
    ]
    vec_ok = all(fnv1a_64(d) == e for d, e in official_1a)
    print(f"  RFC 9923 FNV-1a vectors (n={len(official_1a)})  {_pass(vec_ok)}")
    for data, expect in official_1a:
        got = fnv1a_64(data)
        label = data.decode("latin-1", errors="replace") if data else ""
        print(f"    {label!r:12} expect=0x{expect:016X} got=0x{got:016X}  {_pass(got == expect)}")

    chongo = b"chongo <Landon Curt Noll> /\\../\\"
    ob = fnv0_64(chongo)
    print(
        f"  FNV-0(chongo)=offset_basis  "
        f"0x{ob:016X}  {_pass(ob == FNV_OFFSET_BASIS)}"
    )

    print(
        f"  prime=0x{FNV_PRIME:X} = 2^40+2^8+0xB3  "
        f"{_pass(FNV_PRIME == (2**40 + 2**8 + 0xB3))}"
    )
    pc = bin(0xB3).count("1")
    print(f"  popcount(0xB3)={pc} in {{4,5}}  {_pass(pc in (4, 5))}")
    print(f"  offset_basis=0x{FNV_OFFSET_BASIS:016X}")

    z1 = fnv0_64(b"")
    z2 = fnv0_64(b"\x00" * 8)
    print(f"  fnv0(empty)={z1}  fnv0(0x00*8)={z2}  sticky-zero {_pass(z1 == 0 and z2 == 0)}")

    h1 = fnv1_64(b"some-string-a") ^ fnv1_64(b"some-id-1231")
    h2 = fnv1_64(b"some-string-b") ^ fnv1_64(b"some-id-1232")
    print(f"  xor-correlation fnv1:  0x{h1:016X} == 0x{h2:016X}  {_pass(h1 == h2)}")
    h1a = fnv1a_64(b"some-string-a") ^ fnv1a_64(b"some-id-1231")
    h2a = fnv1a_64(b"some-string-b") ^ fnv1a_64(b"some-id-1232")
    print(f"  xor-correlation fnv1a: 0x{h1a:016X} == 0x{h2a:016X}  {_pass(h1a == h2a)}")
    d1 = fnv1_64(b"prefix-a") ^ fnv1_64(b"prefix-b")
    d1a = fnv1a_64(b"prefix-a") ^ fnv1a_64(b"prefix-b")
    print(f"  last-byte delta fnv1=0x{d1:X} (expect 0x3)  {_pass(d1 == 0x3)}")
    print(f"  last-byte delta fnv1a=0x{d1a:X} (xor-linear? {_pass(d1a == 0x3)})")

    n = 100_000
    buckets = [0] * OMEGA_SIZE
    for i in range(n):
        h = fnv1a_64(f"payload:{i}".encode())
        buckets[extract_fiber_m12(h)] += 1
    ideal = n / OMEGA_SIZE
    mean = sum(buckets) / OMEGA_SIZE
    print(
        f"  dispersion n={n}: m12 buckets min={min(buckets)} max={max(buckets)} "
        f"mean={mean:.2f} ideal={ideal:.2f} max/ideal={max(buckets) / ideal:.3f}"
    )

    sample = fnv1a_64(b"baseline_claim_user_12345")
    fm12 = extract_fiber_m12(sample)
    om = unpack_omega12(fm12)
    print(
        f"  fiber m12={fm12} -> omega12 u6={om.u6} v6={om.v6} shell={om.shell}  "
        f"roundtrip {_pass(pack_omega12(om) == fm12)}"
    )

    m = 10_000
    seals: set[int] = set()
    fnvs: set[int] = set()
    omega_ok = True
    for i in range(m):
        p = f"payload:{i}".encode()
        s = Gyroscopic().route_from_archetype(p).state24
        seals.add(s)
        omega_ok = omega_ok and is_in_omega24(s)
        fnvs.add(fnv1a_64(p))
    print(
        f"  kernel seals n={m}: distinct={len(seals)} collisions={m - len(seals)} "
        f"all_in_omega {_pass(omega_ok)}"
    )
    print(f"  fnv64        n={m}: distinct={len(fnvs)} collisions={m - len(fnvs)}")


# ---------------------------------------------------------------------------
# 15. Convergence accounting
# ---------------------------------------------------------------------------


def section_convergence() -> None:
    print("\n15. CONVERGENCE ACCOUNTING")
    print("=" * 5)

    epoch = datetime(EPOCH_YEAR, 1, 1, tzinfo=timezone.utc)
    elapsed_s = (datetime.now(timezone.utc) - epoch).total_seconds()
    moments = int(elapsed_s) * (2**32)
    print(f"  elapsed since 1900: {elapsed_s / 3.15576e7:.2f} yr")
    print(f"  tick-coordinates elapsed: {moments:.6e}")

    receipts_year = POPULATION * DAYS_PER_YEAR
    print(f"  UHI receipts/year (8.1e9 daily): {receipts_year:.6e}")

    # Ledger model: each receipt is a coordinate (anchor, depth, phase) on a
    # deterministic kernel trajectory. Storage keeps the anchor once per
    # identity and one depth delta per receipt; seal, parity, and event fields
    # are recomputed by replay.
    flat_bytes = receipts_year * 17
    coord_bytes = receipts_year * 1
    anchor_bytes = POPULATION * 3
    print(f"  flat 17B transport form: {flat_bytes:.6e} B = {flat_bytes / 1e12:.2f} TB/yr")
    print(f"  coordinate ledger (1B depth delta/receipt): {coord_bytes:.6e} B = {coord_bytes / 1e12:.2f} TB/yr")
    print(f"  identity anchors (3B, once): {anchor_bytes:.6e} B = {anchor_bytes / 1e9:.2f} GB")
    print(f"  coordinate/flat ratio: {coord_bytes / flat_bytes:.4f}")
    print(f"  ledger fits one consumer drive  {_pass((coord_bytes + anchor_bytes) < 5e12)}")

    addr_day_m12 = OMEGA_SIZE * 86_400
    addr_day_disc = addr_day_m12 * 256
    addr_year_disc = addr_day_disc * DAYS_PER_YEAR
    print(f"  grid/day m12: {addr_day_m12:.6e}  +1B discriminator: {addr_day_disc:.6e}")
    print(f"  grid/year +1B: {addr_year_disc:.6e}")
    print(f"  events per m12 bucket per day: {POPULATION / addr_day_m12:.2f}")
    print(f"  grid/receipts headroom: {addr_year_disc / receipts_year:.4f}x")
    demand = float(POPULATION) * float(UHI_PER_YEAR)
    print(f"  CSM/demand headroom: {csm_total() / demand:.6e}x")

    for n in (16, 17, 18, 20):
        print(f"    L{n}: frames={n // 4} remainder={n % 4} fits {qr_fits(n)}")
    era_years = 256 * (2**32) / 3.15576e7
    print(f"  era byte: 256 eras x {2**32 / 3.15576e7:.1f} yr = {era_years:.6e} yr")

    payload = b"baseline_claim_user_12345"
    fiber = fnv1a_64(payload)
    fstate = Gyroscopic().route_from_archetype(fiber.to_bytes(8, "big")).state24
    print(f"  fiber24=0x{fstate:06X} on_omega {_pass(is_in_omega24(fstate))}")

    m = 10_000
    pairs: set[tuple[int, int]] = set()
    for i in range(m):
        p = f"payload:{i}".encode()
        s = Gyroscopic().route_from_archetype(p).state24
        f = Gyroscopic().route_from_archetype(fnv1a_64(p).to_bytes(8, "big")).state24
        pairs.add((s, f))
    print(f"  (seal24, fiber24) pairs n={m}: distinct={len(pairs)}")


# ---------------------------------------------------------------------------
# 16. Daily geometry and tier occupation
# ---------------------------------------------------------------------------


def section_daily_geometry() -> None:
    print("\n16. DAILY GEOMETRY AND TIER OCCUPATION")
    print("=" * 5)
    min_per_day = HOURS_PER_DAY * MU_PER_HOUR
    print(f"  carrier bits=24  day hours=24  {_pass(HOURS_PER_DAY == 24)}")
    print(f"  face bits=12  half-day hours=12  {_pass(HOURS_PER_DAY // 2 == 12)}")
    print(f"  pairs per face=6  2h blocks per half-day=6  {_pass(HOURS_PER_DAY // 2 // 2 == 6)}")
    print(f"  UHI/day={UHI_PER_DAY} min  day={min_per_day} min")
    print(f"  240*6={240 * 6} == 1440  {_pass(UHI_PER_DAY * 6 == min_per_day)}")
    print(f"  4h*6={UHI_HOURS_PER_DAY * 6} == 24h  {_pass(UHI_HOURS_PER_DAY * 6 == HOURS_PER_DAY)}")
    print(f"  2 pairs*6 == 12 pairs  {_pass(2 * 6 == 12)}")

    print("  tier occupation of the daily carrier:")
    for name, mult in TIER_MULTIPLIERS.items():
        daily_mu = mult * UHI_PER_DAY
        hours = daily_mu / MU_PER_HOUR
        pairs = hours / 2
        if mult == 60:
            print(
                f"    {name}: {daily_mu:,} MU/day  extent 4h at second grain  "
                f"grain x{mult}"
            )
        else:
            print(
                f"    {name}: {daily_mu} MU/day = {hours:.0f}h = {pairs:.0f} pairs "
                f"= {pairs / 6:.3f} of one face"
            )
    print(
        f"  Tier4 mult 60 == MU_PER_HOUR  "
        f"{_pass(TIER_MULTIPLIERS['Tier 4'] == MU_PER_HOUR)}"
    )
    sec_in_4h = UHI_HOURS_PER_DAY * SECONDS_PER_HOUR
    print(
        f"  Tier4 seconds {sec_in_4h}*365={sec_in_4h * DAYS_PER_YEAR:,}  "
        f"{_pass(sec_in_4h * DAYS_PER_YEAR == TIER_MULTIPLIERS['Tier 4'] * UHI_PER_YEAR)}"
    )

    csm = csm_total()
    print("  universal-tier CSM coverage (all 8.1e9 at tier):")
    for name, mult in TIER_MULTIPLIERS.items():
        d = float(POPULATION) * float(mult * UHI_PER_YEAR)
        cov = Coverage(total_mu=csm, annual_demand_mu=d)
        print(f"    {name}: demand={d:.6e} MU/yr  coverage={cov.years:.6e} yr")


# ---------------------------------------------------------------------------
# Main (part 2)
# ---------------------------------------------------------------------------


def main() -> None:
    print("hQVM MOMENTS UNIFIED ANALYSIS (2/2)")
    print("=" * 5)
    section_parity()
    section_policy()
    section_golden()
    section_genealogy_integration()
    section_qr_receipts()
    section_fnv_profile()
    section_convergence()
    section_daily_geometry()
    print("\nPART 2 DONE")
    print("=" * 5)


if __name__ == "__main__":
    main()
