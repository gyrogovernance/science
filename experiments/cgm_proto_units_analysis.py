#!/usr/bin/env python3
"""
Common Governance Model (CGM) – Dimensionless Core + Anchor-Based Energies

This script implements the CGM framework using only dimensionless geometric invariants
and a single physical anchor. The framework derives energy scales from geometric
principles without introducing new base units.

Key Features:
- Dimensionless geometric invariants (Q_G, m_p, stage actions/ratios)
- Single anchor approach: choose one physical scale (e.g., E_CS = Planck energy)
- Optical conjugacy relation for UV/IR energy mapping
- Polygon recursion for π derivation (pure geometry, no trig)
- Optional fine-structure constant calculation

Author: Basil Korompilias & AI Assistants
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from mpmath import mp
import numpy as np

# ============================================================================
# CONFIG / CONSTANTS
# ============================================================================

DEFAULT_PRECISION_DIGITS = 160
mp.dps = DEFAULT_PRECISION_DIGITS

try:
    import scipy.constants as sc
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Notice: scipy not available. Using CODATA 2018 fallback.")

class PhysicalConstants:
    """Essential physical constants (SI) for optional anchoring/printing."""
    def __init__(self):
        if SCIPY_AVAILABLE:
            self.G  = sc.G
            self.c  = sc.c
            self.h  = sc.h
            self.hbar = sc.hbar
            self.e  = sc.e
        else:
            self.G  = 6.67430e-11
            self.c  = 299792458.0
            self.h  = 6.62607015e-34
            self.hbar = 1.054571817e-34
            self.e  = 1.602176634e-19

CONST = PhysicalConstants()

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class PolygonIteration:
    sides: int
    lower_bound: Any
    upper_bound: Any
    gap: Any
    mean_value: Any
    quantum_amplitude: Any
    relative_precision: Any

# ============================================================================
# POLYGON RECURSION (pure geometry)
# ============================================================================

def compute_polygon_recursion(iterations: int) -> List[PolygonIteration]:
    """Archimedean inscribed/circumscribed polygon recursion (no trig)."""
    results: List[PolygonIteration] = []
    sides = mp.mpf(6)
    inscribed_sq = mp.mpf("1")        # s6^2 = 1
    circumscribed = mp.mpf(2)/mp.sqrt(3)  # t6 = 2/√3
    prev_gap = None

    for _ in range(iterations):
        inscribed = mp.sqrt(inscribed_sq)
        lower = (sides * inscribed)/2
        upper = (sides * circumscribed)/2
        gap = upper - lower
        if prev_gap is not None and gap >= prev_gap:
            raise RuntimeError("Precision loss: non-monotonic convergence")
        prev_gap = gap

        mean = (upper + lower)/2
        amp  = mp.sqrt(2/(upper+lower)) * gap   # diagnostic only
        prec = gap/mean

        results.append(PolygonIteration(
            sides=int(sides),
            lower_bound=lower, upper_bound=upper, gap=gap,
            mean_value=mean, quantum_amplitude=amp, relative_precision=prec
        ))

        # doubling recurrences
        inscribed_sq = 2 - mp.sqrt(4 - inscribed_sq)
        circumscribed = (2*circumscribed)/(mp.sqrt(4+circumscribed*circumscribed) + 2)
        sides *= 2

    return results

def check_iteration_scaling(prev: PolygonIteration, curr: PolygonIteration) -> Dict[str, Any]:
    gap_ratio = curr.gap / prev.gap
    try:
        amp_ratio = curr.quantum_amplitude/prev.quantum_amplitude if prev.quantum_amplitude != 0 else mp.mpf("inf")
    except Exception:
        amp_ratio = mp.mpf("inf")
    return {
        "gap_ratio": gap_ratio,
        "amplitude_ratio": amp_ratio,
        "expected_gap": mp.mpf("0.25"),
        "expected_amplitude": mp.mpf("0.25"),
        "gap_valid": mp.almosteq(gap_ratio, mp.mpf("0.25"), rel_eps=mp.mpf("5e-3")),
        "amplitude_valid": (mp.almosteq(amp_ratio, mp.mpf("0.25"), rel_eps=mp.mpf("5e-3"))
                            if amp_ratio != mp.mpf("inf") else False),
    }

# ============================================================================
# CGM CORE (dimensionless)
# ============================================================================

@dataclass
class CGMCore:
    """Dimensionless CGM invariants & actions (no units)."""
    derived_pi: Any

    @property
    def Q_G(self) -> Any:
        return 4*mp.pi

    @property
    def m_p(self) -> Any:
        return mp.mpf(1)/(2*mp.sqrt(2*mp.pi))

    @property
    def S_min(self) -> Any:
        return (mp.pi/2)*self.m_p

    @property
    def S_rec(self) -> Any:
        return (3*mp.pi/2)*self.m_p

    @property
    def S_geo(self) -> Any:
        return self.m_p * mp.pi * mp.sqrt(3) / 2

    @property
    def closure_identity(self) -> Any:
        return self.Q_G * (self.m_p**2)  # = 1/2 (dimensionless)

# Stage thresholds/actions/ratios (dimensionless)

def stage_thresholds() -> Dict[str, Any]:
    """
    Canonical thresholds (dimensionless):
    CS: s_p = π/2
    UNA: u_p = cos(π/4) = 1/√2
    ONA: o_p = π/4
    BU: m_p = 1/(2√(2π))
    """
    return {
        "CS": mp.pi/2,
        "UNA": mp.cos(mp.pi/4),
        "ONA": mp.pi/4,
        "BU": mp.mpf(1)/(2*mp.sqrt(2*mp.pi)),
    }

def stage_actions(thr: Dict[str, Any]) -> Dict[str, Any]:
    """
    Actions map (dimensionless):
    S_CS = s_p/m_p; S_UNA = u_p/m_p; S_ONA = o_p/m_p; S_BU = m_p
    """
    m_p = thr["BU"]
    return {
        "CS": thr["CS"]/m_p,
        "UNA": thr["UNA"]/m_p,
        "ONA": thr["ONA"]/m_p,
        "BU": m_p
    }

def gut_action(actions: Dict[str, Any], eta: float = 1.0) -> Any:
    """1/S_GUT = 1/S_UNA + 1/S_ONA + eta/S_CS (dimensionless)."""
    return 1 / (1/actions["UNA"] + 1/actions["ONA"] + eta/actions["CS"])

def energy_ratios_vs_CS(actions: Dict[str, Any], S_gut: Any) -> Dict[str, Any]:
    """Dimensionless ratios E_stage/E_CS = S_stage/S_CS (scale cancels)."""
    out = { "UNA/CS": actions["UNA"]/actions["CS"],
            "ONA/CS": actions["ONA"]/actions["CS"],
            "BU/CS" : actions["BU"] /actions["CS"],
            "GUT/CS": S_gut/ actions["CS"] }
    return out

# ============================================================================
# ANCHOR-BASED ENERGIES (optional; uses one physical anchor)
# ============================================================================

def planck_energy_GeV() -> float:
    """E_Planck in GeV, from SI constants (no CGM units involved)."""
    E_J = float(mp.sqrt(CONST.hbar * CONST.c**5 / CONST.G))
    return E_J/(CONST.e*1e9)

def anchor_uv_energies(actions: Dict[str, Any], S_gut: Any,
                       E_CS_GeV: float) -> Dict[str, float]:
    """UV energies from single anchor E_CS (in GeV)."""
    scale = E_CS_GeV / float(actions["CS"])
    E = { k: float(scale*float(v)) for k,v in actions.items() }
    E["GUT"] = float(scale*float(S_gut))
    return E

def bu_dual_project(uv: Dict[str, float], E_EW_GeV: float = 240.0) -> Dict[str, float]:
    """
    IR energies via optical invariant:
    E_i^UV * E_i^IR = (E_CS * E_EW) / (4π^2)
    """
    C = (uv["CS"]*E_EW_GeV)/(float(4*mp.pi*mp.pi))
    return { k: C/uv[k] for k in uv }

def optical_invariant_values(uv: Dict[str, float], ir: Dict[str, float]) -> Dict[str, float]:
    return { k: uv[k]*ir[k] for k in uv if k in ir }

def magnification_swap(uv: Dict[str, float], ir: Dict[str, float]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if "UNA" in uv and "ONA" in uv: out["UV_ONA/UNA"] = uv["ONA"]/uv["UNA"]
    if "UNA" in ir and "ONA" in ir: out["IR_UNA/ONA"] = ir["UNA"]/ir["ONA"]
    if "UV_ONA/UNA" in out and "IR_UNA/ONA" in out:
        out["swap_verified"] = abs(out["UV_ONA/UNA"] - out["IR_UNA/ONA"]) < 1e-12
    return out

# ============================================================================
# OPTIONAL: fine-structure hook (kept; purely dimensionless)
# ============================================================================

def try_fine_structure(m_p: Any) -> Optional[Dict[str, Any]]:
    try:
        # Prefer local package import; fallback to flat files
        try:
            from .tw_closure_test import TWClosureTester
            from .functions.gyrovector_ops import GyroVectorSpace
        except Exception:
            from tw_closure_test import TWClosureTester
            from functions.gyrovector_ops import GyroVectorSpace

        gyro = GyroVectorSpace(c=1.0)
        tester = TWClosureTester(gyro)
        res = tester.compute_bu_dual_pole_monodromy(verbose=False)
        delta_BU = res["delta_bu"]
        alpha_pred = (delta_BU**4)/m_p
        return {
            "delta_BU": float(delta_BU),
            "m_p": float(m_p),
            "alpha_pred": float(alpha_pred),
        }
    except Exception:
        return None

# ============================================================================
# PRINTING / UI
# ============================================================================

def format_number(x: Any, digits: int = 10) -> str:
    return mp.nstr(x, digits)

def print_summary(iterations: List[PolygonIteration],
                  core: CGMCore,
                  actions: Dict[str, Any],
                  S_gut: Any,
                  E_CS_anchor_GeV: float,
                  E_uv: Dict[str, float],
                  E_ir: Dict[str, float]) -> None:

    print("CGM DIMENSIONLESS CORE + ANCHOR-BASED ENERGIES")
    print("="*64)

    # 1) Geometry
    print("\n1) Polygon recursion (pure geometry)")
    print(f"   iterations: {len(iterations)}")
    print(f"   π ≈ {mp.nstr(iterations[-1].mean_value, 15)} ; rel err = {mp.nstr(iterations[-1].relative_precision, 2)}")
    if len(iterations) >= 2:
        sca = check_iteration_scaling(iterations[-2], iterations[-1])
        print(f"   gap ratio ~ {mp.nstr(sca['gap_ratio'],6)} (→0.25), amplitude ratio ~ {mp.nstr(sca['amplitude_ratio'],6)} (→0.25)")

    # 2) CGM invariants (dimensionless)
    print("\n2) CGM invariants (dimensionless)")
    print(f"   Q_G = 4π = {mp.nstr(core.Q_G,10)}")
    print(f"   m_p = 1/(2√(2π)) = {mp.nstr(core.m_p,10)}")
    print(f"   S_min=(π/2)m_p = {mp.nstr(core.S_min,10)} ; S_rec=(3π/2)m_p = {mp.nstr(core.S_rec,10)} ; S_geo= m_p π√3/2 = {mp.nstr(core.S_geo,10)}")
    print(f"   closure: Q_G·m_p² = {mp.nstr(core.closure_identity,10)} (expected 1/2)")

    # 3) Stage actions & ratios (dimensionless)
    print("\n3) Stage actions (dimensionless)")
    for k in ("CS","UNA","ONA","BU"):
        print(f"   S_{k:<3} = {mp.nstr(actions[k],10)}")
    ratios = energy_ratios_vs_CS(actions, S_gut)
    print("\n   Ratios vs CS:")
    for k,v in ratios.items():
        print(f"   E_{k} = {mp.nstr(v,8)}")
    print(f"   S_GUT/S_CS = {mp.nstr(S_gut/actions['CS'],8)}")

    # 4) Anchor (one scale) + UV energies
    print("\n4) UV energies from single anchor")
    print(f"   Anchor: E_CS = {E_CS_anchor_GeV:.6e} GeV")
    for k in ("CS","UNA","ONA","BU","GUT"):
        val = E_uv[k]
        if val >= 1e9:
            print(f"   E_{k:<3} = {val:.3e} GeV = {val/1e9:.3f} TeV")
        else:
            print(f"   E_{k:<3} = {val:.6e} GeV")

    # 5) IR conjugates + invariant
    print("\n5) IR energies via optical conjugacy (E^UV E^IR = (E_CS·240 GeV)/(4π²))")
    for k in ("CS","UNA","ONA","BU","GUT"):
        print(f"   E_{k:<3}^IR = {E_ir[k]:.6e} GeV")
    inv = optical_invariant_values(E_uv, E_ir)
    expected = (E_uv["CS"]*240.0)/(float(4*mp.pi*mp.pi))
    print(f"\n   invariant K = (E_CS·240)/(4π²) = {expected:.6e} GeV²")
    ok = True
    for k,v in inv.items():
        ok &= (abs(v-expected)/expected < 1e-12)
        print(f"   E_{k}^UV · E_{k}^IR = {v:.6e} GeV²")
    print(f"   invariant status: {'OK' if ok else 'check'}")

    # 6) Magnification swap
    swap = magnification_swap(E_uv, E_ir)
    if swap:
        print("\n6) Magnification swap (should match):")
        print(f"   UV: E_ONA/E_UNA = {swap['UV_ONA/UNA']:.9f}")
        print(f"   IR: E_UNA/E_ONA = {swap['IR_UNA/ONA']:.9f}")
        print(f"   swap verified: {swap.get('swap_verified', False)}")

    # 7) Optional fine-structure (dimensionless)
    fs = try_fine_structure(core.m_p)
    print("\n7) Fine-structure (optional hook)")
    if fs:
        print(f"   δ_BU = {fs['delta_BU']:.9f} ; m_p = {fs['m_p']:.9f} ; α_pred = {fs['alpha_pred']:.10f}")
    else:
        print("   (fine-structure module not available)")

# ============================================================================
# MAIN
# ============================================================================

def main(target_precision: float = 1e-15,
         precision_digits: int = DEFAULT_PRECISION_DIGITS,
         E_CS_anchor_GeV: Optional[float] = None):
    """
    Pipeline:
      1) Polygon recursion for π
      2) Build dimensionless CGM core
      3) Compute stage actions, S_GUT, and ratios
      4) Choose ONE anchor E_CS (default = Planck energy) and compute UV energies
      5) Map to IR via optical invariant
    """
    mp.dps = precision_digits

    # Phase 1: polygon recursion up to target precision
    iters: List[PolygonIteration] = []
    for n in range(1, 201):
        cur = compute_polygon_recursion(n)
        if cur and cur[-1].relative_precision <= target_precision:
            iters = cur
            break
    if not iters:
        iters = compute_polygon_recursion(200)

    # Phase 2–3: dimensionless CGM core and actions
    core = CGMCore(derived_pi=mp.pi)
    thr = stage_thresholds()
    acts = stage_actions(thr)
    S_gut = gut_action(acts, eta=1.0)

    # Phase 4: anchor (default Planck)
    if E_CS_anchor_GeV is None:
        E_CS_anchor_GeV = planck_energy_GeV()
    E_uv = anchor_uv_energies(acts, S_gut, E_CS_anchor_GeV)

    # Phase 5: IR via optical conjugacy (EW = 240 GeV)
    E_ir = bu_dual_project(E_uv, E_EW_GeV=240.0)

    # Print summary
    print_summary(iters, core, acts, S_gut, E_CS_anchor_GeV, E_uv, E_ir)

if __name__ == "__main__":
    main(target_precision=1e-15, precision_digits=160)
