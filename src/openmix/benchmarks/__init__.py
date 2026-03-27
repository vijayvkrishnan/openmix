"""
FormulaBench — benchmarks for formulation stability prediction.

Complements CheMixHub (thermophysical properties) with formulation-level
outcomes: stability, compatibility, shelf life.
"""

from openmix.benchmarks.shampoo import ShampooStability
from openmix.benchmarks.pharma_solubility import PharmaSolubility

__all__ = ["ShampooStability", "PharmaSolubility"]
