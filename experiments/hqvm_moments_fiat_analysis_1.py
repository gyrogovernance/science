#!/usr/bin/env python3
"""
hQVM Moments unified analysis (part 1/2): shared library and sections 1-8.

Role: hosts shared constants, medium-layer helpers (identity anchors, grants,
shells, archives, meta-routing), genealogy frame utilities, NTP/m12 receipt
encoding, and measurement sections 1-8 (kernel anchors through divergence
localization) over gyroscopic.hQVM.

Inputs: none (deterministic kernel + fixed golden vectors).
Outputs: printed tables, counts, and PASS/FAIL checks only.

Companions:
  hqvm_moments_fiat_analysis_2.py  -- sections 9-16
  hqvm_moments_fiat_analysis_run.py  -- full study runner
"""

from __future__ import annotations

import hashlib
import math
import struct
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gyroscopic.hQVM.api import (  # noqa: E402
    BYTES_BY_Q6,
    chirality_word6,
    depth4_mask_projection48,
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
    GENE_MIC_S,
    HORIZON_SIZE,
    OMEGA_SIZE,
    byte_to_intron,
    intron_family,
    intron_micro_ref,
    step_state_by_byte,
)
from gyroscopic.hQVM.kernel import Gyroscopic  # noqa: E402

# ---------------------------------------------------------------------------
# Physical / temporal constants
# ---------------------------------------------------------------------------

EPOCH_YEAR = 1900
F_CS = 9_192_631_770
CSM_CAPACITY = 7.944165e26  # reference; recomputed below as n_phys / |Omega|

FNV_OFFSET_BASIS = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
ANCHOR_BYTE = GENE_MIC_S  # 0xAA

EVENT_BASELINE = 0x01
EVENT_ROUTE_ACCEPT = 0x02

# ---------------------------------------------------------------------------
# Economic constants
# ---------------------------------------------------------------------------

MU_PER_MINUTE = 1
MU_PER_HOUR = 60
HOURS_PER_DAY = 24
DAYS_PER_YEAR = 365
SECONDS_PER_HOUR = 3600

UHI_HOURS_PER_DAY = 4
UHI_PER_DAY = UHI_HOURS_PER_DAY * MU_PER_HOUR
UHI_PER_YEAR = UHI_PER_DAY * DAYS_PER_YEAR

TIER_MULTIPLIERS: dict[str, int] = {
    "Tier 1": 1,
    "Tier 2": 2,
    "Tier 3": 3,
    "Tier 4": 60,
}

POPULATION = 8_100_000_000

# ---------------------------------------------------------------------------
# Golden vectors (regression pins)
# ---------------------------------------------------------------------------

GOLDEN_ALICE_ANCHOR = "aaa559"
GOLDEN_BOB_ANCHOR = "6955a9"
GOLDEN_SHELL_SEAL = "9966aa"
GOLDEN_META_ROOT = "555aa9"
GOLDEN_MASK48 = 0x333F30000CCC
GOLDEN_PHI_A = 0
GOLDEN_PHI_B = 1
GOLDEN_PARITY_O = 0xC0C
GOLDEN_PARITY_E = 0xCC0
GOLDEN_PARITY_P = 0


def _pass(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _fmt(num: float) -> str:
    if num >= 1e18:
        return f"{num / 1e18:,.2f} quintillion ({num:,.0f})"
    if num >= 1e15:
        return f"{num / 1e15:,.2f} quadrillion ({num:,.0f})"
    if num >= 1e12:
        return f"{num / 1e12:,.2f} trillion ({num:,.0f})"
    if num >= 1e9:
        return f"{num / 1e9:,.2f} billion ({num:,.0f})"
    if num >= 1e6:
        return f"{num / 1e6:,.2f} million ({num:,.0f})"
    return f"{num:,.2f}"


# ---------------------------------------------------------------------------
# Capacity helpers
# ---------------------------------------------------------------------------


def n_phys() -> float:
    return (4.0 / 3.0) * math.pi * (F_CS**3)


def csm_total() -> float:
    return n_phys() / float(OMEGA_SIZE)


@dataclass(frozen=True)
class Coverage:
    total_mu: float
    annual_demand_mu: float

    @property
    def years(self) -> float:
        return self.total_mu / self.annual_demand_mu

    @property
    def usage_pct(self) -> float:
        return 100.0 * self.annual_demand_mu / self.total_mu


# ---------------------------------------------------------------------------
# Medium layer: Identity Anchor, Grant, Shell, Archive
# ---------------------------------------------------------------------------


def identity_hash(name: str) -> bytes:
    return hashlib.sha256(name.encode("utf-8")).digest()


def identity_anchor(name: str) -> tuple[bytes, str]:
    ident = identity_hash(name)
    r = Gyroscopic()
    sig = r.route_from_archetype(ident)
    return ident, sig.state_hex


@dataclass(frozen=True)
class Grant:
    identity_label: str
    identity_id: bytes
    kernel_anchor: str
    amount_mu: int

    def canonical_receipt(self) -> bytes:
        return (
            self.identity_id
            + self.kernel_anchor.encode("ascii")
            + self.amount_mu.to_bytes(8, "big")
        )


@dataclass
class Shell:
    header: bytes
    grants: list[Grant] = field(default_factory=list)
    total_capacity_mu: int = 0
    seal: str = ""

    @property
    def used_capacity_mu(self) -> int:
        return sum(g.amount_mu for g in self.grants)

    @property
    def free_capacity_mu(self) -> int:
        return self.total_capacity_mu - self.used_capacity_mu

    def compute_seal(self) -> str:
        sorted_grants = sorted(self.grants, key=lambda g: g.identity_id)
        payload = self.header
        for g in sorted_grants:
            payload += g.canonical_receipt()
        r = Gyroscopic()
        sig = r.route_from_archetype(payload)
        self.seal = sig.state_hex
        return self.seal


def make_shell(
    header: bytes,
    grants_spec: list[tuple[str, int]],
    capacity: int = 10**18,
) -> Shell:
    grants = []
    for name, amount in grants_spec:
        ident, anchor = identity_anchor(name)
        grants.append(Grant(name, ident, anchor, amount))
    shell = Shell(header=header, grants=grants, total_capacity_mu=capacity)
    shell.compute_seal()
    return shell


@dataclass
class Archive:
    shells: list[Shell] = field(default_factory=list)

    @property
    def per_identity_totals(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for shell in self.shells:
            for g in shell.grants:
                totals[g.identity_label] = (
                    totals.get(g.identity_label, 0) + g.amount_mu
                )
        return totals

    @property
    def total_used(self) -> int:
        return sum(s.used_capacity_mu for s in self.shells)


def meta_root_from_payloads(payloads: list[bytes]) -> str:
    seals: list[bytes] = []
    for p in payloads:
        r = Gyroscopic()
        sig = r.route_from_archetype(p)
        seals.append(bytes.fromhex(sig.state_hex))
    r = Gyroscopic()
    for s in seals:
        r.step_bytes(s)
    return r.signature().state_hex


def meta_root_from_seals(seals: list[bytes]) -> str:
    r = Gyroscopic()
    for s in seals:
        r.step_bytes(s)
    return r.signature().state_hex


def seal_payload(p: bytes) -> bytes:
    r = Gyroscopic()
    sig = r.route_from_archetype(p)
    return bytes.fromhex(sig.state_hex)


# ---------------------------------------------------------------------------
# Genealogy: depth-4 frame records
# ---------------------------------------------------------------------------


def frame_record(b0: int, b1: int, b2: int, b3: int) -> tuple[int, int, int]:
    mask48 = depth4_mask_projection48(b0, b1, b2, b3)
    fams = []
    for b in (b0, b1, b2, b3):
        intron = byte_to_intron(b)
        a_bit = intron & 1
        b_bit = (intron >> 7) & 1
        fams.append((a_bit, b_bit))
    phi_a = fams[0][1] ^ fams[1][0] ^ fams[2][1] ^ fams[3][0]
    phi_b = fams[0][0] ^ fams[1][1] ^ fams[2][0] ^ fams[3][1]
    return (mask48, phi_a, phi_b)


def frame_sequence(payload: bytes) -> list[tuple[int, int, int]]:
    frames = []
    n = len(payload) - (len(payload) % 4)
    for i in range(0, n, 4):
        frames.append(
            frame_record(payload[i], payload[i + 1], payload[i + 2], payload[i + 3])
        )
    return frames


def apply_word(state: int, word: bytes) -> int:
    s = state
    for b in word:
        s = step_state_by_byte(s, b)
    return s


# ---------------------------------------------------------------------------
# NTP / FNV / 18-byte Moment Receipt
# ---------------------------------------------------------------------------


def ntp_timestamp(dt: datetime | None = None) -> tuple[int, int]:
    if dt is None:
        dt = datetime.now(timezone.utc)
    epoch = datetime(EPOCH_YEAR, 1, 1, tzinfo=timezone.utc)
    total_seconds = (dt - epoch).total_seconds()
    sec32 = int(total_seconds) & 0xFFFFFFFF
    frac_part = total_seconds - int(total_seconds)
    frac32 = int(frac_part * (2**32)) & 0xFFFFFFFF
    return sec32, frac32


def ntp_to_datetime(sec32: int, frac32: int) -> datetime:
    epoch = datetime(EPOCH_YEAR, 1, 1, tzinfo=timezone.utc)
    return epoch + timedelta(seconds=sec32 + frac32 / (2**32))


def extract_m12(frac32: int) -> int:
    return ((frac32 >> 20) & 0xFFF) % OMEGA_SIZE


def m12_time_range(m12: int) -> tuple[float, float]:
    return m12 / OMEGA_SIZE, (m12 + 1) / OMEGA_SIZE


def fnv1a_64(data_bytes: bytes) -> int:
    h = FNV_OFFSET_BASIS
    for byte in data_bytes:
        h ^= byte
        h = (h * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


def extract_fiber_m12(fiber64: int) -> int:
    lower32 = fiber64 & 0xFFFFFFFF
    return ((lower32 >> 20) & 0xFFF) % OMEGA_SIZE


def create_moment_receipt(
    event_payload: bytes | str,
    event_type: int = EVENT_BASELINE,
    dt: datetime | None = None,
) -> tuple[bytearray, dict]:
    receipt = bytearray(18)
    sec32, frac32 = ntp_timestamp(dt)
    receipt[0:4] = struct.pack(">I", sec32)
    receipt[4:8] = struct.pack(">I", frac32)
    receipt[8] = event_type & 0xFF
    if isinstance(event_payload, str):
        event_payload = event_payload.encode("utf-8")
    fiber64 = fnv1a_64(event_payload)
    receipt[9:17] = struct.pack(">Q", fiber64)
    receipt[17] = ANCHOR_BYTE
    return receipt, {
        "sec32": sec32,
        "frac32": frac32,
        "m12": extract_m12(frac32),
        "fiber64": fiber64,
        "fiber_m12": extract_fiber_m12(fiber64),
        "event_type": event_type,
        "anchor_verified": receipt[17] == ANCHOR_BYTE,
    }


def verify_moment_receipt(
    receipt_bytes: bytes | bytearray,
    original_payload: bytes | str,
) -> dict:
    if len(receipt_bytes) != 18:
        return {"valid": False, "error": "Receipt must be exactly 18 bytes"}
    result: dict = {"valid": True}
    anchor_ok = receipt_bytes[17] == ANCHOR_BYTE
    result["anchor_verified"] = anchor_ok
    if not anchor_ok:
        result["valid"] = False
        result["error"] = "Anchor byte mismatch"
    sec32 = struct.unpack(">I", receipt_bytes[0:4])[0]
    frac32 = struct.unpack(">I", receipt_bytes[4:8])[0]
    event_type = receipt_bytes[8]
    fiber64 = struct.unpack(">Q", receipt_bytes[9:17])[0]
    result["sec32"] = sec32
    result["frac32"] = frac32
    result["m12_time"] = extract_m12(frac32)
    result["event_type"] = event_type
    result["fiber64"] = fiber64
    result["m12_fiber"] = extract_fiber_m12(fiber64)
    if isinstance(original_payload, str):
        original_payload = original_payload.encode("utf-8")
    seal_ok = fiber64 == fnv1a_64(original_payload)
    result["seal_verified"] = seal_ok
    if not seal_ok:
        result["valid"] = False
        result["error"] = "FNV seal mismatch"
    return result


# ---------------------------------------------------------------------------
# 1. Kernel anchors
# ---------------------------------------------------------------------------


def section_kernel_anchors() -> None:
    print("\n1. KERNEL ANCHORS")
    print("=" * 5)
    print(f"  |Omega|={OMEGA_SIZE}  |H|={HORIZON_SIZE}  H^2==Omega {_pass(HORIZON_SIZE * HORIZON_SIZE == OMEGA_SIZE)}")
    print(f"  GENE_MIC_S=0x{GENE_MIC_S:02X}  GENE_MAC_REST=0x{GENE_MAC_REST:06X}")
    print(f"  f_Cs={F_CS:,} Hz")
    r = Gyroscopic()
    sig = r.signature()
    ok = (
        sig.state24 == GENE_MAC_REST
        and sig.state_hex == "aaa555"
        and sig.a_hex == "aaa"
        and sig.b_hex == "555"
        and sig.step == 0
    )
    print(f"  rest signature state_hex={sig.state_hex} a={sig.a_hex} b={sig.b_hex} step={sig.step}  {_pass(ok)}")


# ---------------------------------------------------------------------------
# 2. CSM / MU / UHI economy
# ---------------------------------------------------------------------------


def section_economy() -> None:
    print("\n2. ECONOMY (MU / UHI / CSM)")
    print("=" * 5)
    print(f"  MU/minute={MU_PER_MINUTE}  MU/hour={MU_PER_HOUR}")
    print(f"  UHI/day={UHI_PER_DAY:,}  UHI/year={UHI_PER_YEAR:,}")
    print(f"  UHI amounts  {_pass(UHI_PER_DAY == 240 and UHI_PER_YEAR == 87_600)}")

    expected_tiers = {
        "Tier 1": 87_600,
        "Tier 2": 175_200,
        "Tier 3": 262_800,
        "Tier 4": 5_256_000,
    }
    print("  Tier schedule (annual MU):")
    tiers_ok = True
    for name, mult in TIER_MULTIPLIERS.items():
        annual = mult * UHI_PER_YEAR
        tiers_ok = tiers_ok and annual == expected_tiers[name]
        print(f"    {name}: {mult}x = {annual:,}")
    print(f"  Tier schedule  {_pass(tiers_ok)}")

    seconds_in_4h = 4 * SECONDS_PER_HOUR
    mnemonic = seconds_in_4h * DAYS_PER_YEAR
    tier4 = TIER_MULTIPLIERS["Tier 4"] * UHI_PER_YEAR
    print(f"  Tier4 mnemonic {seconds_in_4h}*365={mnemonic:,} == tier4={tier4:,}  {_pass(mnemonic == tier4)}")

    illustrative = 4 * 4 * 52 * MU_PER_HOUR
    print(f"  work-week illustr.={illustrative:,} != UHI/year  {_pass(illustrative != UHI_PER_YEAR)}")

    csm = csm_total()
    print(f"  N_phys={n_phys():.6e}")
    print(f"  CSM={csm:.6e} MU  ({_fmt(csm)})")
    print(f"  CSM in (7e26, 8e26)  {_pass(7e26 < csm < 8e26)}")
    print(f"  CSM ref={CSM_CAPACITY:.6e}  rel_err={abs(csm - CSM_CAPACITY) / CSM_CAPACITY:.3e}")

    demand = float(POPULATION) * float(UHI_PER_YEAR)
    cov = Coverage(total_mu=csm, annual_demand_mu=demand)
    print(f"  population={POPULATION:,}")
    print(f"  annual UHI demand={_fmt(demand)} MU")
    print(f"  coverage={cov.years:.6e} years  {_pass(cov.years > 1e12)}")

    threshold = 0.01 * csm
    multiplier = threshold / demand
    print(f"  1% CSM / annual demand = {multiplier:,.0f}x  {_pass(multiplier > 1e7)}")

    distributions: dict[str, dict[str, float]] = {
        "Conservative": {"Tier 1": 95.0, "Tier 2": 4.0, "Tier 3": 0.9, "Tier 4": 0.1},
        "Plausible": {"Tier 1": 90.0, "Tier 2": 8.0, "Tier 3": 1.5, "Tier 4": 0.5},
        "Generous": {"Tier 1": 85.0, "Tier 2": 12.0, "Tier 3": 2.5, "Tier 4": 0.5},
    }
    results: dict[str, float] = {}
    print("  Tier-weighted coverage (years):")
    for name, dist in distributions.items():
        weighted_mult = sum(
            (dist[t] / 100.0) * TIER_MULTIPLIERS[t] for t in TIER_MULTIPLIERS
        )
        d = float(POPULATION) * UHI_PER_YEAR * weighted_mult
        results[name] = Coverage(total_mu=csm, annual_demand_mu=d).years
        print(f"    {name}: {results[name]:.6e}")
    ordered = results["Conservative"] > results["Plausible"] > results["Generous"]
    all_large = all(y > 1e11 for y in results.values())
    print(f"  coverage order Cons>Plaus>Gen  {_pass(ordered and all_large)}")

    reserve = float(UHI_PER_YEAR) * float(POPULATION) * 1000
    surplus = csm - reserve
    per_div = surplus / 12
    print(f"  surplus/12={per_div:.6e}  {_pass(surplus > 0 and per_div > 0)}")


# ---------------------------------------------------------------------------
# 3. NTP / m12 / receipt
# ---------------------------------------------------------------------------


def section_ntp_receipt() -> None:
    print("\n3. NTP / m12 / 18-BYTE RECEIPT")
    print("=" * 5)
    sec32, frac32 = ntp_timestamp()
    m12 = extract_m12(frac32)
    t0, t1 = m12_time_range(m12)
    print(f"  sec32=0x{sec32:08X}  frac32=0x{frac32:08X}")
    print(f"  m12={m12}  range=[{t0:.9f}, {t1:.9f}) s  bucket={1 / OMEGA_SIZE:.9f} s")

    tau_ntp = 1.0 / (2**32)
    cycles = F_CS * tau_ntp
    print(f"  cesium cycles / NTP tick={cycles:.9f}  frac={cycles - int(cycles):.9f}")

    csm = csm_total()
    mu_per_tick = csm / (2**32)
    mu_coord = int(mu_per_tick * frac32)
    print(f"  MU/NTP tick={mu_per_tick:.6e}  MU coord={mu_coord}")

    payload = b"test_event_data"
    fiber = fnv1a_64(payload)
    print(f"  FNV-1a({payload!r})=0x{fiber:016X}  fiber_m12={extract_fiber_m12(fiber)}")

    event = "baseline_claim_user_12345"
    receipt, meta = create_moment_receipt(event, EVENT_BASELINE)
    ver = verify_moment_receipt(receipt, event)
    print(f"  receipt hex={receipt.hex()}  len={len(receipt)}")
    print(
        f"  m12_time={meta['m12']}  m12_fiber={meta['fiber_m12']}  "
        f"type=0x{meta['event_type']:02X}"
    )
    print(f"  receipt verify valid={ver['valid']}  {_pass(ver['valid'])}")

    planck = datetime(1900, 12, 14, tzinfo=timezone.utc)
    p_sec, p_frac = ntp_timestamp(planck)
    print(f"  Planck 1900-12-14 NTP sec={p_sec:,}  m12={extract_m12(p_frac)}")


# ---------------------------------------------------------------------------
# 4. Kernel transport (real q6 / chirality)
# ---------------------------------------------------------------------------


def section_kernel_transport() -> None:
    print("\n4. KERNEL TRANSPORT")
    print("=" * 5)
    test_bytes = bytes([0xEE, 0x1E, 0x44, 0xC1, 0xD7, 0x28, 0xB8, 0x00])
    r = Gyroscopic()
    chi0 = chirality_word6(r.state24)
    shells = [chi0.bit_count()]
    chis = [chi0]
    for b in test_bytes:
        r.step_byte(b)
        chi = chirality_word6(r.state24)
        chis.append(chi)
        shells.append(chi.bit_count())
    print(f"  bytes={[hex(b) for b in test_bytes]}")
    print(f"  chi history={[f'0x{c:02X}' for c in chis]}")
    print(f"  shell history={shells}")
    print(f"  final state={r.signature().state_hex}  chi=0x{chis[-1]:02X} shell={shells[-1]}")

    # Chirality transport: chi' = chi xor q6(b)
    chi = chirality_word6(GENE_MAC_REST)
    q_acc = 0
    transport_ok = True
    s = GENE_MAC_REST
    for b in test_bytes:
        q = q_word6(b)
        q_acc ^= q
        s = step_state_by_byte(s, b)
        chi_next = chirality_word6(s)
        expected = (chi ^ q) & 0x3F
        if chi_next != expected:
            transport_ok = False
        chi = chi_next
    print(f"  chi == chi0 xor Q(word)  {_pass(transport_ok and chi == (chi0 ^ q_acc) & 0x3F)}")

    # Inverse stepping
    r2 = Gyroscopic()
    r2.step_bytes(test_bytes)
    fwd = r2.state24
    r2.step_bytes_inverse(test_bytes)
    print(f"  inverse restore rest  {_pass(fwd != GENE_MAC_REST and r2.state24 == GENE_MAC_REST)}")

    # Depth-4 F-word on rest via four family bytes is involution for each micro;
    # here: same byte four times is T^4 only for specific bytes — measure q closure.
    demo = 0x3C
    chi = chirality_word6(GENE_MAC_REST)
    q = q_word6(demo)
    hist = [chi]
    for _ in range(4):
        chi = (chi ^ q) & 0x3F
        hist.append(chi)
    print(f"  q6(0x{demo:02X})=0x{q:02X}  chi xor^4 hist={[f'0x{c:02X}' for c in hist]}")
    print(f"  chi xor^2 == id (order<=2)  {_pass(hist[2] == hist[0] and hist[4] == hist[0])}")


# ---------------------------------------------------------------------------
# 5. Identity / Grant / Shell / Archive
# ---------------------------------------------------------------------------


def section_medium() -> None:
    print("\n5. MEDIUM (ANCHORS / GRANTS / SHELLS / ARCHIVES)")
    print("=" * 5)

    id1, a1 = identity_anchor("alice")
    id2, a2 = identity_anchor("alice")
    _, bob = identity_anchor("bob")
    print(f"  alice anchor={a1}  bob={bob}")
    print(f"  alice determinism  {_pass(id1 == id2 and a1 == a2)}")
    print(f"  alice != bob  {_pass(a1 != bob)}")
    print(f"  anchor hex width=6  {_pass(len(a1) == 6 and int(a1, 16) >= 0)}")

    g = Grant("alice", id1, a1, UHI_PER_YEAR)
    receipt = g.canonical_receipt()
    g2 = Grant("alice", id1, a1, UHI_PER_YEAR)
    g3 = Grant("alice", id1, a1, UHI_PER_YEAR * 2)
    print(f"  grant receipt len={len(receipt)} (expect 46)  {_pass(len(receipt) == 46)}")
    print(f"  grant receipt determinism  {_pass(g2.canonical_receipt() == receipt)}")
    print(f"  amount changes receipt  {_pass(g3.canonical_receipt() != receipt)}")

    s1 = make_shell(
        b"ecology:year:2026",
        [("alice", UHI_PER_YEAR * 3), ("bob", UHI_PER_YEAR * 2)],
    )
    s2 = make_shell(
        b"ecology:year:2026",
        [("alice", UHI_PER_YEAR * 3), ("bob", UHI_PER_YEAR * 2)],
    )
    s_tamper = make_shell(
        b"ecology:year:2026",
        [("alice", UHI_PER_YEAR * 30), ("bob", UHI_PER_YEAR * 2)],
    )
    s_hdr = make_shell(b"ecology:year:2027", [("alice", UHI_PER_YEAR)])
    s_hdr0 = make_shell(b"ecology:year:2026", [("alice", UHI_PER_YEAR)])
    print(f"  shell seal={s1.seal}  used={s1.used_capacity_mu:,}")
    print(f"  shell seal determinism  {_pass(s1.seal == s2.seal)}")
    print(f"  shell tamper changes seal  {_pass(s_tamper.seal != s1.seal)}")
    print(f"  shell header sensitivity  {_pass(s_hdr.seal != s_hdr0.seal)}")

    so1 = make_shell(
        b"test",
        [("alice", UHI_PER_YEAR), ("bob", UHI_PER_YEAR * 2), ("carol", UHI_PER_YEAR * 3)],
    )
    so2 = make_shell(
        b"test",
        [("carol", UHI_PER_YEAR * 3), ("alice", UHI_PER_YEAR), ("bob", UHI_PER_YEAR * 2)],
    )
    print(f"  grant order invariance  {_pass(so1.seal == so2.seal)}")

    scap = make_shell(b"test", [("alice", 100_000), ("bob", 200_000)], capacity=1_000_000)
    print(
        f"  capacity used={scap.used_capacity_mu} free={scap.free_capacity_mu}  "
        f"{_pass(scap.used_capacity_mu == 300_000 and scap.free_capacity_mu == 700_000)}"
    )

    published = make_shell(
        b"ecology:year:2026",
        [
            ("alice", UHI_PER_YEAR),
            ("bob", UHI_PER_YEAR * 2),
            ("carol", UHI_PER_YEAR * 3),
        ],
    )
    sorted_grants = sorted(published.grants, key=lambda g: g.identity_id)
    payload = published.header
    for g in sorted_grants:
        payload += g.canonical_receipt()
    verified = Gyroscopic().route_from_archetype(payload).state_hex
    print(f"  shell replay published={published.seal} verified={verified}  {_pass(verified == published.seal)}")

    arch = Archive(
        shells=[
            make_shell(b"year:2026", [("alice", UHI_PER_YEAR * 3), ("bob", UHI_PER_YEAR * 2)]),
            make_shell(b"year:2027", [("alice", UHI_PER_YEAR * 3), ("bob", UHI_PER_YEAR * 2)]),
        ]
    )
    totals = arch.per_identity_totals
    print(f"  archive totals={totals}")
    print(
        f"  archive aggregation  "
        f"{_pass(totals['alice'] == UHI_PER_YEAR * 6 and totals['bob'] == UHI_PER_YEAR * 4)}"
    )


# ---------------------------------------------------------------------------
# 6. Meta-routing
# ---------------------------------------------------------------------------


def section_meta_routing() -> None:
    print("\n6. META-ROUTING")
    print("=" * 5)
    bundles = [b"program:A|data:abc", b"program:B|data:def", b"program:C|data:ghi"]
    root1 = meta_root_from_payloads(bundles)
    root2 = meta_root_from_payloads(bundles)
    print(f"  root={root1}  determinism  {_pass(root1 == root2)}")

    bundles_ok = [b"program:A", b"program:B", b"program:C"]
    seals = [seal_payload(b) for b in bundles_ok]
    root_ok = meta_root_from_seals(seals)
    tampered = [b"program:A", b"program:B:TAMPERED", b"program:C"]
    tampered_seals = [seal_payload(b) for b in tampered]
    root_bad = meta_root_from_seals(tampered_seals)
    diffs = [i for i, (a, b) in enumerate(zip(seals, tampered_seals)) if a != b]
    print(f"  tamper leaf index={diffs}  root_ok={root_ok} root_bad={root_bad}")
    print(f"  tamper localization  {_pass(root_bad != root_ok and diffs == [1])}")


# ---------------------------------------------------------------------------
# 7. Genealogy frames
# ---------------------------------------------------------------------------


def section_genealogy_frames() -> None:
    print("\n7. GENEALOGY (DEPTH-4 FRAMES)")
    print("=" * 5)
    rng = np.random.default_rng(2025)
    det_ok = True
    width_ok = True
    sens_ok = True
    for _ in range(1000):
        word = [int(rng.integers(0, 256)) for _ in range(4)]
        r1 = frame_record(*word)
        r2 = frame_record(*word)
        if r1 != r2:
            det_ok = False
        mask48, phi_a, phi_b = r1
        if not (0 <= mask48 < (1 << 48) and phi_a in (0, 1) and phi_b in (0, 1)):
            width_ok = False
    rng = np.random.default_rng(99)
    for _ in range(500):
        word = [int(rng.integers(0, 256)) for _ in range(4)]
        r_orig = frame_record(*word)
        for pos in range(4):
            word_alt = word.copy()
            word_alt[pos] = (word[pos] + 1) % 256
            if frame_record(*word_alt) == r_orig:
                sens_ok = False
    print(f"  frame determinism (n=1000)  {_pass(det_ok)}")
    print(f"  frame widths  {_pass(width_ok)}")
    print(f"  single-byte sensitivity (n=500)  {_pass(sens_ok)}")

    payload = b"deterministic frame sequence test!!"
    fs1 = frame_sequence(payload)
    fs2 = frame_sequence(payload)
    print(f"  sequence len={len(fs1)} (expect 8)  {_pass(fs1 == fs2 and len(fs1) == 8)}")

    frames = frame_sequence(b"abcdefghij")
    frames_ext = frame_sequence(b"abcdefghijXY")
    print(
        f"  trailing ignore / extend  "
        f"{_pass(len(frames) == 2 and len(frames_ext) == 3 and frames_ext[:2] == frames)}"
    )

    # State collision with distinct frames
    rng = np.random.default_rng(123)
    state_to_frames: dict[int, set[tuple[int, int, int]]] = {}
    for _ in range(100_000):
        word = [int(rng.integers(0, 256)) for _ in range(4)]
        final = apply_word(GENE_MAC_REST, bytes(word))
        state_to_frames.setdefault(final, set()).add(frame_record(*word))
    collisions = sum(1 for frameset in state_to_frames.values() if len(frameset) > 1)
    print(f"  states with frame-distinguishable histories={collisions}  {_pass(collisions > 0)}")

    rng = np.random.default_rng(456)
    found = False
    example = None
    for _ in range(200_000):
        w1 = [int(rng.integers(0, 256)) for _ in range(4)]
        w2 = [int(rng.integers(0, 256)) for _ in range(4)]
        if w1 == w2:
            continue
        s1 = apply_word(GENE_MAC_REST, bytes(w1))
        s2 = apply_word(GENE_MAC_REST, bytes(w2))
        if s1 == s2:
            fr1 = frame_record(*w1)
            fr2 = frame_record(*w2)
            if fr1 != fr2:
                found = True
                example = (w1, w2, s1, fr1, fr2)
                break
    if example is not None:
        w1, w2, s1, fr1, fr2 = example
        print(f"  collide words {w1} / {w2} -> state=0x{s1:06x}")
        print(f"    frames {fr1} / {fr2}")
    print(f"  same-state distinct-frame pair found  {_pass(found)}")


# ---------------------------------------------------------------------------
# 8. Divergence localization
# ---------------------------------------------------------------------------


def section_divergence() -> None:
    print("\n8. DIVERGENCE LOCALIZATION")
    print("=" * 5)
    rng = np.random.default_rng(77)
    loc_ok = True
    for _ in range(200):
        log = list(rng.integers(0, 256, size=24).astype(int))
        flip_pos = int(rng.integers(0, 24))
        log_alt = log.copy()
        log_alt[flip_pos] = (log_alt[flip_pos] + 1) % 256
        frames_orig = frame_sequence(bytes(log))
        frames_alt = frame_sequence(bytes(log_alt))
        affected = flip_pos // 4
        for i in range(affected):
            if frames_orig[i] != frames_alt[i]:
                loc_ok = False
        if frames_orig[affected] == frames_alt[affected]:
            loc_ok = False
    print(f"  localize to affected frame (n=200)  {_pass(loc_ok)}")

    rng = np.random.default_rng(88)
    localized = 0
    missed_by_state = 0
    total = 0
    for _ in range(500):
        log = list(rng.integers(0, 256, size=20).astype(int))
        flip_pos = int(rng.integers(0, 20))
        log_alt = log.copy()
        log_alt[flip_pos] = (log_alt[flip_pos] + 1) % 256
        frames_orig = frame_sequence(bytes(log))
        frames_alt = frame_sequence(bytes(log_alt))
        affected = flip_pos // 4
        if affected < len(frames_orig):
            total += 1
            if frames_orig[affected] != frames_alt[affected]:
                localized += 1
            if apply_word(GENE_MAC_REST, bytes(log)) == apply_word(
                GENE_MAC_REST, bytes(log_alt)
            ):
                missed_by_state += 1
    print(f"  divergences={total}  localized_by_frame={localized}/{total}")
    print(f"  missed_by_final_state={missed_by_state}/{total}")
    print(f"  frame catches all  {_pass(localized == total)}")


# ---------------------------------------------------------------------------
# Main (part 1)
# ---------------------------------------------------------------------------


def main() -> None:
    print("hQVM MOMENTS UNIFIED ANALYSIS (1/2)")
    print("=" * 5)
    section_kernel_anchors()
    section_economy()
    section_ntp_receipt()
    section_kernel_transport()
    section_medium()
    section_meta_routing()
    section_genealogy_frames()
    section_divergence()
    print("\nPART 1 DONE")
    print("=" * 5)


if __name__ == "__main__":
    main()
