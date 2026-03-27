"""
Pluggable evaluation functions for the experiment framework.

The experiment runner calls `evaluate(formula) -> StabilityScore`.
This module provides multiple implementations:

  HeuristicScorer  — built-in rule-based scoring (default)
  ModelScorer      — trained ML model
  LabScorer        — real experimental feedback (cloud lab, manual)
  CompositeScorer  — model where confident, heuristic as fallback
"""

from openmix.scorers.base import Scorer, CompositeScorer
from openmix.scorers.model import ModelScorer
from openmix.scorers.lab import LabScorer, ManualScorer

__all__ = [
    "Scorer",
    "CompositeScorer",
    "ModelScorer",
    "LabScorer",
    "ManualScorer",
]
