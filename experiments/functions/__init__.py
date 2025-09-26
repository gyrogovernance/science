"""
Core mathematical operations for CGM (Common Governance Model)

This package provides the fundamental mathematical operations for the CGM framework,
including dimensional calibration, gyrovector operations, and recursive memory structures.

Classes:
    GyroVectorSpace: Einstein-Ungar gyrovector space implementation
    RecursivePath: Recursive path with memory accumulation
    DimensionalCalibrator: Dimensional calibration engine with {ħ, c, m⋆} basis
    DimVec: Dimensional vector for unit analysis
    GyroTriangle: Gyrotriangle defect and area calculations
    RecursiveMemory: Recursive memory structure for κ prediction
    MemoryPath: Memory path for recursive structures
    GyroTriangleDefectTheorem: Gyrotriangle defect theorem implementation
    ThomasWignerRotation: Thomas-Wigner rotation calculations
    BUAmplitudeIdentity: BU amplitude identity implementation

Functions:
    unit, P2, C4_zero_mean: Torus template functions
    tau_from_template, toroidal_opacity: Toroidal physics functions
    find_best_axis, project_P2_C4_healpix: Analysis utilities
"""

__version__ = "1.0.0"
__author__ = "CGM Framework"
__description__ = "Core mathematical operations for Common Governance Model"

# Core classes
from .gyrovector_ops import GyroVectorSpace, RecursivePath
from .dimensions import DimensionalCalibrator, DimVec, DimensionalEngineHomomorphism
from .gyrotriangle import GyroTriangle
from .recursive_memory import RecursiveMemory, MemoryPath
from .gyrogeometry import (
    GyroTriangleDefectTheorem,
    ThomasWignerRotation,
    BUAmplitudeIdentity,
)

# Torus and template functions
from .torus import (
    unit,
    P2,
    C4_zero_mean,
    torus_template,
    tau_from_template,
    toroidal_opacity,
    toroidal_y_weight,
    find_best_axis,
    project_P2_C4_healpix,
    scan_axis_angles,
)

__all__ = [
    # Core classes
    "GyroVectorSpace",
    "RecursivePath",
    "DimensionalCalibrator",
    "DimVec",
    "DimensionalEngineHomomorphism",
    "GyroTriangle",
    "RecursiveMemory",
    "MemoryPath",
    "GyroTriangleDefectTheorem",
    "ThomasWignerRotation",
    "BUAmplitudeIdentity",
    # Torus functions
    "unit",
    "P2",
    "C4_zero_mean",
    "torus_template",
    "tau_from_template",
    "toroidal_opacity",
    "toroidal_y_weight",
    "find_best_axis",
    "project_P2_C4_healpix",
    "scan_axis_angles",
]
