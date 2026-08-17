#!/usr/bin/env python3
"""
Exploratory mixing-angle neighbors for CGM precession quantities.

Not part of the core holonomy identities. No representation, Hamiltonian,
or oscillation law is derived here.

Companion of cgm_precession_analysis_{1,2,run}.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

_EXP = Path(__file__).resolve().parent
_ROOT = _EXP.parent
for _p in (_EXP, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cgm_holonomy_analysis_common import CGMThresholds
from cgm_precession_analysis_1 import EdgeCache, connection_measurements, forced_basis

REF_MIXING = {
    "V_us": 0.22431,
    "th12_deg": math.degrees(math.asin(math.sqrt(0.307))),
    "th23_deg": 49.1,
    "mu_tau": 105.6583755 / 1776.86,
}


def _rel_err(a: float, b: float) -> float:
    return abs(a - b) / abs(b) if b else float("inf")


def print_mixing(state: dict[str, Any]) -> None:
    b = state["basis"]
    conn = state["conn"]
    w0, dlt, phi = b["omega0"], b["Delta"], b["phi_SU2"]
    lam = math.sin(w0 + dlt / 2.0)
    rb_pal = conn["lab"]["relative_boost_pal"]
    rows = [
        ("sin(omega0+Delta/2)", lam, REF_MIXING["V_us"], "Cabibbo |V_us|"),
        ("phi_SU2 [deg]", phi * 180.0 / math.pi, REF_MIXING["th12_deg"], "PMNS theta12"),
        ("phi_SU2+rb_pal/2 [deg]", (phi + rb_pal / 2) * 180.0 / math.pi, REF_MIXING["th23_deg"], "PMNS theta23"),
        ("2*(1-rho0)", b["two_1_rho0"], REF_MIXING["mu_tau"], "m_mu/m_tau"),
    ]
    print("MIXING PROXIES")
    print("-" * 5)
    print(f"  {'model':<26} {'value':>12}  {'ref':>12}  rel%     label")
    for name, val, ref, label in rows:
        print(f"  {name:<26} {val:12.6g}  {ref:12.6g}  {_rel_err(val, ref)*100:8.4f}  {label}")
    print()


def main() -> None:
    t = CGMThresholds.make()
    cache = EdgeCache(t)
    state = {"basis": forced_basis(t), "conn": connection_measurements(t, cache)}
    print_mixing(state)


if __name__ == "__main__":
    main()
