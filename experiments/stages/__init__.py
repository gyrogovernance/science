"""
CGM Stage implementations

This module contains the four stages of the Common Governance Model:
- CS (Common Source): Chiral seed with σ₃ rotation
- UNA (Unity Non-Absolute): SU(2) spin frame emergence
- ONA (Opposition Non-Absolute): SO(3) translation activation
- BU (Balance Universal): Closure with ψ_BU coherence field

Each stage implements:
- Stage-specific operator construction
- Memory field calculations
- Threshold validation
- Integration with the overall CGM framework
"""

from .cs_stage import CSStage
from .una_stage import UNAStage
from .ona_stage import ONAStage
from .bu_stage import BUStage

__all__ = ["CSStage", "UNAStage", "ONAStage", "BUStage"]
