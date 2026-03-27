"""
Scorer base class and composite scorer.

Any callable that takes a Formula and returns a StabilityScore works.
These classes add structure for configuration, confidence tracking,
and fallback logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

from openmix.schema import Formula
from openmix.score import score as heuristic_score, StabilityScore


class Scorer(ABC):
    """Base class for evaluation functions."""

    @abstractmethod
    def __call__(self, formula: Formula) -> StabilityScore:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    def description(self) -> str:
        return ""


class HeuristicScorer(Scorer):
    """Built-in rule-based scoring. The default."""

    def __call__(self, formula: Formula) -> StabilityScore:
        return heuristic_score(formula)

    @property
    def name(self) -> str:
        return "heuristic"

    @property
    def description(self) -> str:
        return "Rule-based heuristic (compatibility, pH, HLB, integrity, completeness)"


class CompositeScorer(Scorer):
    """
    Uses primary scorer when confident, falls back to secondary.

    Useful for: trained model on known ingredients, heuristic for unknown.
    """

    def __init__(
        self,
        primary: Scorer | Callable,
        fallback: Scorer | Callable | None = None,
        confidence_threshold: float = 0.5,
    ):
        self.primary = primary
        self.fallback = fallback or HeuristicScorer()
        self.confidence_threshold = confidence_threshold
        self._primary_calls = 0
        self._fallback_calls = 0

    def __call__(self, formula: Formula) -> StabilityScore:
        try:
            result = self.primary(formula)
            # If primary returns a meaningful score, use it
            if result.total > 0:
                self._primary_calls += 1
                return result
        except Exception:
            pass

        self._fallback_calls += 1
        return self.fallback(formula)

    @property
    def name(self) -> str:
        p = getattr(self.primary, 'name', 'primary')
        f = getattr(self.fallback, 'name', 'fallback')
        return f"composite({p}+{f})"

    @property
    def stats(self) -> dict:
        total = self._primary_calls + self._fallback_calls
        return {
            "primary_calls": self._primary_calls,
            "fallback_calls": self._fallback_calls,
            "primary_rate": self._primary_calls / total if total > 0 else 0,
        }
