"""
Knowledge base loader — reads YAML rule files into structured data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml


DATA_DIR = Path(__file__).parent / "data"


@dataclass
class InteractionRule:
    """A single ingredient interaction rule — hard or soft."""

    a: str
    b: str
    rule_type: str  # "hard" or "soft"
    mechanism: str
    confidence: float
    source: str
    category: str
    message: str

    # Hard rules have a single severity
    severity: Optional[str] = None

    # Soft rules have severity per mode
    severity_by_mode: Optional[dict[str, str]] = None

    # Conditions for soft rules (when does this interaction matter?)
    conditions: Optional[dict] = None

    # How to work around this interaction
    mitigation: Optional[str] = None

    def get_severity(self, mode: str = "safety") -> str:
        """Get the effective severity for a given validation mode."""
        if self.rule_type == "hard":
            return self.severity or "error"

        if self.severity_by_mode:
            return self.severity_by_mode.get(mode, "info")

        return self.severity or "warning"

    def should_fire(self, mode: str = "safety") -> bool:
        """Whether this rule should produce an issue in the given mode."""
        severity = self.get_severity(mode)
        return severity != "ignore"


@dataclass
class Knowledge:
    """Loaded knowledge base — all rules and data needed for validation."""

    interaction_rules: list[InteractionRule] = field(default_factory=list)
    oil_hlb: dict[str, float] = field(default_factory=dict)
    aliases: dict[str, list[str]] = field(default_factory=dict)

    @property
    def hard_rules(self) -> list[InteractionRule]:
        return [r for r in self.interaction_rules if r.rule_type == "hard"]

    @property
    def soft_rules(self) -> list[InteractionRule]:
        return [r for r in self.interaction_rules if r.rule_type == "soft"]

    def rules_for_category(self, category: str) -> list[InteractionRule]:
        return [r for r in self.interaction_rules
                if r.category in ("all", category)]

    def coverage_summary(self, category: str | None = None) -> dict:
        """Report how many rules cover a given category."""
        if category:
            applicable = self.rules_for_category(category)
            dedicated = [r for r in applicable if r.category == category]
        else:
            applicable = self.interaction_rules
            dedicated = applicable

        categories_covered = set(r.category for r in self.interaction_rules)

        return {
            "total_rules": len(self.interaction_rules),
            "hard_rules": len(self.hard_rules),
            "soft_rules": len(self.soft_rules),
            "applicable_to_category": len(applicable) if category else None,
            "dedicated_to_category": len(dedicated) if category else None,
            "category": category,
            "categories_covered": sorted(categories_covered),
        }


@lru_cache(maxsize=1)
def load_knowledge(data_dir: str | Path | None = None) -> Knowledge:
    """
    Load all knowledge files from the data directory.

    Results are cached — call with no arguments for the default bundled knowledge.
    """
    data_path = Path(data_dir) if data_dir else DATA_DIR
    kb = Knowledge()

    # Load interaction rules
    pairs_file = data_path / "incompatible_pairs.yaml"
    if pairs_file.exists():
        with open(pairs_file, "r", encoding="utf-8") as f:
            raw_pairs = yaml.safe_load(f) or []
        for p in raw_pairs:
            kb.interaction_rules.append(InteractionRule(
                a=p["a"].upper().strip(),
                b=p["b"].upper().strip(),
                rule_type=p.get("type", "soft"),
                mechanism=p.get("mechanism", "unknown"),
                confidence=float(p.get("confidence", 0.5)),
                source=p.get("source", ""),
                category=p.get("category", "all"),
                message=p.get("message", "").strip(),
                severity=p.get("severity") if isinstance(p.get("severity"), str) else None,
                severity_by_mode=p.get("severity_by_mode"),
                conditions=p.get("conditions"),
                mitigation=p.get("mitigation"),
            ))

    # Load oil HLB values
    hlb_file = data_path / "oil_hlb.yaml"
    if hlb_file.exists():
        with open(hlb_file, "r", encoding="utf-8") as f:
            kb.oil_hlb = yaml.safe_load(f) or {}

    # Load aliases
    aliases_file = data_path / "aliases.yaml"
    if aliases_file.exists():
        with open(aliases_file, "r", encoding="utf-8") as f:
            kb.aliases = yaml.safe_load(f) or {}

    return kb
