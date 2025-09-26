"""
CGM Experiments Package

This package contains experimental modules for testing and validating
the Common Governance Model - Recursive Gyrovector Formalism.

Main Analysis Classes:
    TriadIndexAnalyzer: Analyzes scale relationships using triad source indices
    HelicalMemoryAnalyzer: Implements helical worldline evolution framework
    QuantumGravityHorizon: Quantum gravity analysis with CGM formalism
    CGMKompaneyetsAnalyzer: CMB physics analysis
    GravityCouplingAnalyzer: Gravity coupling analysis
    SingularityInfinityValidator: Singularity and infinity validation
    ElectricCalibrationValidator: Electric calibration validation

Main Functions:
    run_holonomy_flow: Holonomy flow analysis
    run_comprehensive_gravity_analysis: Complete gravity analysis
    test_tw_precession_small_angle: Thomas-Wigner precession tests
"""

# Import main analysis classes
from .triad_index_analyzer import TriadIndexAnalyzer
from .helical_memory_analyzer import HelicalMemoryAnalyzer
from .cgm_quantum_gravity_analysis import QuantumGravityHorizon, CGMConstants
from .cgm_kompaneyets_analysis import CGMKompaneyetsAnalyzer
from .cgm_gravity_analysis import (
    GravityCouplingAnalyzer,
    run_comprehensive_gravity_analysis,
)
from .singularity_infinity import SingularityInfinityValidator
from .cgm_sound_diagnostics import CGMAcousticDiagnostics
from .light_chirality_experiments import LightChiralityExperiments
from .tw_closure_test import TWClosureTester

# Import utility functions
from .holonomy_flow import run_holonomy_flow
from .tw_precession import test_tw_precession_small_angle

__all__ = [
    # Main analysis classes
    "TriadIndexAnalyzer",
    "HelicalMemoryAnalyzer",
    "QuantumGravityHorizon",
    "CGMConstants",
    "CGMKompaneyetsAnalyzer",
    "GravityCouplingAnalyzer",
    "SingularityInfinityValidator",
    "CGMAcousticDiagnostics",
    "LightChiralityExperiments",
    "TWClosureTester",
    # Utility functions
    "run_holonomy_flow",
    "run_comprehensive_gravity_analysis",
    "test_tw_precession_small_angle",
]
