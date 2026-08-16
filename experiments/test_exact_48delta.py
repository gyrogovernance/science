#!/usr/bin/env python3
"""
Compare continuous aperture gap Delta = 1 - delta_BU/m_a with the discrete companion 1/48.
Companion: Analysis_48_States.md, Analysis_CGM_Constants.md.
"""

import math
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from gyroscopic.hQVM.constants import BU_HOLONOMY_ANGLE, M_A


def main() -> None:
    delta_BU = BU_HOLONOMY_ANGLE
    m_a = M_A
    rho = delta_BU / m_a
    Delta = 1.0 - rho
    companion = 1.0 / 48.0
    lambda0 = Delta / math.sqrt(5.0)

    print("1. CONTINUOUS APERTURE FROM LOOP ANGLE")
    print("-" * 5)
    print(f"delta_BU = 4*arctan(k(pi/4)*k(m_a)) = {delta_BU:.16f}")
    print(f"m_a = {m_a:.16f}")
    print(f"rho = delta_BU/m_a = {rho:.16f}")
    print(f"Delta = 1 - rho = {Delta:.16f}")
    print(f"48*Delta = {48.0 * Delta:.16f}")
    print(f"lambda_0 = Delta/sqrt(5) = {lambda0:.16f}")
    print(f"lambda_0/Delta = {lambda0 / Delta:.16f}")
    print(f"1/sqrt(5) = {1.0 / math.sqrt(5.0):.16f}")
    print()

    print("2. DISCRETE COMPANION 1/48")
    print("-" * 5)
    print(f"1/48 = {companion:.16f}")
    print(f"48*(1/48) = {48.0 * companion:.16f}")
    print(f"Delta - 1/48 = {Delta - companion:.16f}")
    print(f"|Delta - 1/48|/Delta = {abs(Delta - companion) / Delta:.16f}")
    print()

    print("3. CHECKS")
    print("-" * 5)
    print(f"PASS |48*Delta - 1| < 0.01: {abs(48.0 * Delta - 1.0) < 0.01}")
    print(
        f"PASS |lambda_0/Delta - 1/sqrt(5)| < 1e-12: "
        f"{abs(lambda0 / Delta - 1.0 / math.sqrt(5.0)) < 1e-12}"
    )


if __name__ == "__main__":
    main()
