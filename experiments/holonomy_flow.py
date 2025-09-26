#!/usr/bin/env python3
"""
Holonomy Flow Analyzer for CGM

This module tracks how χ and holonomy deficit flow as recursion depth increases,
testing for fixed points that indicate BU closure.
"""

import numpy as np
from typing import Dict, Any, List
import sys
import os

# Use absolute imports with path setup
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from functions.gyrovector_ops import GyroVectorSpace
from tw_closure_test import TWClosureTester
from helical_memory_analyzer import HelicalMemoryAnalyzer


def run_holonomy_flow(depths: List[int] = [20, 40, 80, 160]) -> List[Dict[str, Any]]:
    """
    Run holonomy flow analysis at different recursion depths.

    This tests for fixed points in the CGM toroidal closure.
    """
    gs = GyroVectorSpace(c=1.0)
    tw = TWClosureTester(gs)
    out = []

    print("HOLONOMY FLOW ANALYSIS")
    print("=" * 50)
    print("Testing for fixed points in CGM toroidal closure")
    print()

    for N in depths:
        print(f"Testing depth N = {N}")
        print("-" * 30)

        # χ should reflect sampling at this depth; use seed=N and scale sample count.
        chi = tw.compute_anatomical_tw_ratio(
            verbose=False, seed=N, n_samples=max(50, N // 2)
        )
        # Holonomy deficit on the canonical loop is depth-invariant; if you need a
        # flow-like metric, use the 8-leg path (still canonical) and track signed deviation.
        hol = tw.test_toroidal_holonomy_fullpath(verbose=False)

        # BU/ψ from helical analyzer (now SU(2)-based)
        # Pass depth parameter to make helical analysis depth-dependent
        ana = HelicalMemoryAnalyzer(gs)
        res = ana.analyze_helical_memory_structure(verbose=False, depth_param=N)
        psi = res["psi_bu_field"]

        result = {
            "depth": N,
            "chi_mean": chi["chi_mean"],
            "chi_cv": chi["coefficient_of_variation"],
            "holonomy_deficit": hol["deviation"],
            "spin_translation_coherence": psi["spin_translation_coherence"],
            "closure_residual": psi["closure_residual"],
            "helical_pitch": psi["helical_pitch"],
        }

        out.append(result)

        print(f"  χ mean: {chi['chi_mean']:.6f}")
        print(f"  χ CV: {chi['coefficient_of_variation']:.1%}")
        print(f"  Holonomy deficit: {hol['deviation']:.6f}")
        print(f"  Spin-translation coherence: {psi['spin_translation_coherence']:.6f}")
        print(f"  Closure residual: {psi['closure_residual']:.6f}")
        print(f"  Helical pitch: {psi['helical_pitch']:.6f}")
        print()

    # Analyze the flow patterns
    print("FLOW PATTERN ANALYSIS")
    print("=" * 50)

    depths_array = np.array([r["depth"] for r in out])
    chi_cv_array = np.array([r["chi_cv"] for r in out])
    holonomy_array = np.array([r["holonomy_deficit"] for r in out])

    # Check for monotone decrease (evidence of fixed point)
    chi_cv_decreasing = np.all(np.diff(chi_cv_array) <= 0)
    holonomy_decreasing = np.all(np.diff(holonomy_array) <= 0)

    print(
        f"χ coefficient of variation decreasing: {'✅ YES' if chi_cv_decreasing else '❌ NO'}"
    )
    print(
        f"Holonomy deficit decreasing: {'✅ YES' if holonomy_decreasing else '❌ NO'}"
    )

    if chi_cv_decreasing and holonomy_decreasing:
        print("\n🎯 STRONG EVIDENCE: Fixed point detected!")
        print("   Both χ stability and holonomy closure are improving with depth")
        print("   This suggests BU stage is approaching full closure")
    elif chi_cv_decreasing or holonomy_decreasing:
        print("\n✅ MODERATE EVIDENCE: Partial convergence detected")
        print("   One metric is improving, suggesting partial closure")
    else:
        print("\n⚠️  NO CONVERGENCE: System may not be approaching closure")
        print("   Consider alternative closure mechanisms")

    return out


if __name__ == "__main__":
    # Debug test first
    print("DEBUG TEST:")
    gs = GyroVectorSpace(c=1.0)
    tw = TWClosureTester(gs)

    for depth in [20, 40, 80, 160]:
        print(f"\nTesting depth {depth}:")
        result = tw.test_toroidal_holonomy(verbose=False, depth_param=depth)
        print(f"  Deviation: {result['deviation']:.6f}")

    print("\n" + "=" * 50)
    print("FULL ANALYSIS:")
    rows = run_holonomy_flow()
    print("\nSUMMARY TABLE:")
    print("Depth | χ_mean | χ_CV | Holonomy | Coherence | Residual | Pitch")
    print("-" * 70)
    for r in rows:
        print(
            f"{r['depth']:5d} | {r['chi_mean']:6.3f} | {r['chi_cv']:4.1%} | {r['holonomy_deficit']:8.3f} | {r['spin_translation_coherence']:9.3f} | {r['closure_residual']:8.3f} | {r['helical_pitch']:5.3f}"
        )
