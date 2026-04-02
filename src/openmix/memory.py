"""
Experiment memory — persistence and learning across runs.

Three-layer memory system:
  Layer 1: Experiment index (~/.openmix/experiment_index.json)
           Lightweight, always loaded. One entry per experiment.
  Layer 2: Full experiment logs (~/.openmix/experiments/*.json)
           Complete trial data, fetched on demand.
  Layer 3: Discoveries (~/.openmix/discoveries.yaml)
           Cross-run findings with confidence scores.

After each run, a consolidation cycle:
  1. Extract findings from the completed experiment
  2. Merge with existing discoveries (update confidence)
  3. Prune contradicted or stale findings

Before each run, prior knowledge retrieval:
  1. Load discoveries relevant to the experiment's domain
  2. Format as a system prompt section
  3. Instruct the agent: these are hints, not ground truth
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from openmix.experiment import ExperimentLog

OPENMIX_DIR = Path.home() / ".openmix"

# Confidence scoring
INITIAL_CONFIDENCE = 0.5
CONFIRMATION_BOOST = 0.15
MAX_CONFIDENCE = 0.95
CONTRADICTION_PENALTY = 0.2


@dataclass
class IndexEntry:
    """One entry in the experiment index (Layer 1)."""
    name: str
    category: str
    date: str
    best_concerns: float
    converged: bool
    n_trials: int
    key_ingredients: list[str]
    summary: str
    log_file: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> IndexEntry:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Discovery:
    """A cross-run finding with evidence tracking (Layer 3)."""
    id: str
    finding: str
    kind: str           # "preference", "avoidance", "pattern", "concern"
    domain: str         # skincare, supplement, beverage, etc.
    ingredients: list[str]
    confidence: float
    evidence_count: int
    source_experiments: list[str]
    first_seen: str
    last_confirmed: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Discovery:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _generate_summary(log: ExperimentLog) -> str:
    """One-line summary for the experiment index."""
    parts = []

    if log.converged:
        conv_iter = next(
            (t.iteration for t in log.trials if t.concern_count == 0),
            len(log.trials),
        )
        parts.append(f"Converged at iter {conv_iter}")
    else:
        if log.best_trial:
            parts.append(
                f"Best: {log.best_concerns:.1f} concerns at iter {log.best_trial.iteration}"
            )

    if log.best_trial:
        n = len(log.best_trial.formula.ingredients)
        parts.append(f"{n} ingredients")

        key_ings = sorted(
            [i for i in log.best_trial.formula.ingredients
             if i.inci_name.upper() != "WATER"],
            key=lambda x: -x.percentage,
        )[:3]
        if key_ings:
            parts.append("key: " + ", ".join(i.inci_name for i in key_ings))

    return ". ".join(parts) if parts else "No trials completed"


def _key_ingredients(log: ExperimentLog) -> list[str]:
    """Extract top non-water ingredients from the best trial."""
    if not log.best_trial:
        return []
    return [
        i.inci_name
        for i in sorted(
            log.best_trial.formula.ingredients,
            key=lambda x: -x.percentage,
        )
        if i.inci_name.upper() != "WATER"
    ][:8]


def _extract_findings(log: ExperimentLog) -> list[Discovery]:
    """Extract findings from a single experiment run."""
    findings: list[Discovery] = []
    if not log.best_trial:
        return findings

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    category = log.config.get("constraints", {}).get("category", "general")
    exp_name = log.name

    # 1. Preferred ingredients — what was in the best trial
    for ing in log.best_trial.formula.ingredients:
        if ing.inci_name.upper() == "WATER":
            continue
        func = ing.function or "unknown"
        findings.append(Discovery(
            id=f"pref_{ing.inci_name.lower().replace(' ', '_')}_{category}",
            finding=f"{ing.inci_name} at {ing.percentage:.1f}% as {func}",
            kind="preference",
            domain=category,
            ingredients=[ing.inci_name],
            confidence=INITIAL_CONFIDENCE,
            evidence_count=1,
            source_experiments=[exp_name],
            first_seen=now,
            last_confirmed=now,
        ))

    # 2. Avoided ingredients — tried in 2+ trials but absent from best
    best_names = {i.inci_name.upper() for i in log.best_trial.formula.ingredients}
    tried_counts: dict[str, tuple[str, int]] = {}  # upper -> (original, count)

    for trial in log.trials:
        for ing in trial.formula.ingredients:
            key = ing.inci_name.upper()
            if key in tried_counts:
                tried_counts[key] = (tried_counts[key][0], tried_counts[key][1] + 1)
            else:
                tried_counts[key] = (ing.inci_name, 1)

    for key, (original_name, count) in tried_counts.items():
        if key not in best_names and key != "WATER" and count >= 2:
            findings.append(Discovery(
                id=f"avoid_{original_name.lower().replace(' ', '_')}_{category}",
                finding=f"{original_name} tried in {count} trials but dropped from best",
                kind="avoidance",
                domain=category,
                ingredients=[original_name],
                confidence=INITIAL_CONFIDENCE,
                evidence_count=1,
                source_experiments=[exp_name],
                first_seen=now,
                last_confirmed=now,
            ))

    # 3. Unresolved concerns in best trial
    if log.best_trial.observation:
        obs = log.best_trial.observation
        for concern in obs.concerns:
            subject = concern.subject
            detail = concern.detail[:120]
            # Deterministic ID from subject + category (stable across processes)
            stable_hash = hashlib.md5(
                f"{subject}:{concern.category}:{category}".encode()
            ).hexdigest()[:8]
            findings.append(Discovery(
                id=f"concern_{stable_hash}_{category}",
                finding=f"Unresolved: {detail}",
                kind="concern",
                domain=category,
                ingredients=[subject] if subject != "formula" else [],
                confidence=INITIAL_CONFIDENCE,
                evidence_count=1,
                source_experiments=[exp_name],
                first_seen=now,
                last_confirmed=now,
            ))

    # 4. Convergence pattern
    if log.converged:
        conv_iter = next(
            (t.iteration for t in log.trials if t.concern_count == 0),
            len(log.trials),
        )
        findings.append(Discovery(
            id=f"pattern_convergence_{category}",
            finding=f"Converged at iteration {conv_iter} of {len(log.trials)}",
            kind="pattern",
            domain=category,
            ingredients=[],
            confidence=INITIAL_CONFIDENCE,
            evidence_count=1,
            source_experiments=[exp_name],
            first_seen=now,
            last_confirmed=now,
        ))

    return findings


def _matches(a: Discovery, b: Discovery) -> bool:
    """Check if two discoveries are about the same thing."""
    if a.kind != b.kind or a.domain != b.domain:
        return False
    # For ingredient-level findings, match on ingredients
    if a.kind in ("preference", "avoidance", "concern"):
        return set(i.upper() for i in a.ingredients) == set(i.upper() for i in b.ingredients)
    # For patterns, match on id (stable across runs)
    return a.id == b.id


def _merge_discoveries(
    existing: list[Discovery],
    new_findings: list[Discovery],
) -> tuple[list[Discovery], list[Discovery]]:
    """
    Merge new findings into existing discoveries.

    Returns (updated_discoveries, newly_added) so callers can report
    what changed.
    """
    merged = list(existing)
    newly_added: list[Discovery] = []

    for new_d in new_findings:
        matched = False
        for ex in merged:
            if _matches(ex, new_d):
                # Confirm existing discovery
                ex.evidence_count += 1
                ex.confidence = min(MAX_CONFIDENCE, ex.confidence + CONFIRMATION_BOOST)
                ex.last_confirmed = new_d.last_confirmed
                for src in new_d.source_experiments:
                    if src not in ex.source_experiments:
                        ex.source_experiments.append(src)
                matched = True
                break

        if not matched:
            # Check for contradictions (preference vs avoidance for same ingredient)
            opposite_kind = "avoidance" if new_d.kind == "preference" else "preference"
            if new_d.kind in ("preference", "avoidance"):
                for ex in merged:
                    if (ex.kind == opposite_kind
                            and ex.domain == new_d.domain
                            and set(i.upper() for i in ex.ingredients) == set(i.upper() for i in new_d.ingredients)):
                        # Contradiction — lower confidence on existing, still add new
                        ex.confidence = max(0.1, ex.confidence - CONTRADICTION_PENALTY)
                        new_d.confidence = max(0.1, new_d.confidence - CONTRADICTION_PENALTY)
                        break

            merged.append(new_d)
            newly_added.append(new_d)

    return merged, newly_added


def format_prior_knowledge(
    discoveries: list[Discovery],
    max_entries: int = 15,
) -> str:
    """Format discoveries as a system prompt section."""
    if not discoveries:
        return ""

    lines = [
        "PRIOR KNOWLEDGE (from previous experiments):",
        "  These findings come from earlier experiments. The observation engine",
        "  is the source of truth — verify before relying on prior knowledge.",
        "",
    ]

    # Sort by confidence descending, then evidence_count
    ranked = sorted(discoveries, key=lambda d: (-d.confidence, -d.evidence_count))

    preferences = [d for d in ranked if d.kind == "preference"]
    avoidances = [d for d in ranked if d.kind == "avoidance"]
    others = [d for d in ranked if d.kind not in ("preference", "avoidance")]

    shown = 0

    if preferences:
        lines.append("  Ingredients that worked well in prior experiments:")
        for d in preferences[:max_entries]:
            conf_str = f"confidence {d.confidence:.2f}"
            ev_str = f"{d.evidence_count} experiment{'s' if d.evidence_count > 1 else ''}"
            lines.append(f"    - {d.finding} ({conf_str}, {ev_str})")
            shown += 1
        lines.append("")

    if avoidances and shown < max_entries:
        lines.append("  Ingredients to reconsider (dropped from best formulas):")
        for d in avoidances[:max_entries - shown]:
            lines.append(f"    - {d.finding} ({d.evidence_count} experiment{'s' if d.evidence_count > 1 else ''})")
            shown += 1
        lines.append("")

    if others and shown < max_entries:
        lines.append("  Patterns observed:")
        for d in others[:max_entries - shown]:
            lines.append(f"    - {d.finding}")
            shown += 1
        lines.append("")

    return "\n".join(lines)


class ExperimentMemory:
    """
    Three-layer experiment memory.

    Persists experiment results and discoveries across runs.
    Retrieves relevant prior knowledge for new experiments.

        memory = ExperimentMemory()
        prior = memory.retrieve_prior_knowledge("skincare")
        # ... run experiment ...
        new_discoveries = memory.record_experiment(log)
    """

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or OPENMIX_DIR
        self.index_path = self.base_dir / "experiment_index.json"
        self.logs_dir = self.base_dir / "experiments"
        self.discoveries_path = self.base_dir / "discoveries.yaml"

    def _ensure_dirs(self):
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Layer 1: Experiment Index
    # ------------------------------------------------------------------

    def load_index(self) -> list[IndexEntry]:
        """Load the experiment index."""
        if not self.index_path.exists():
            return []
        data = json.loads(self.index_path.read_text(encoding="utf-8"))
        return [IndexEntry.from_dict(entry) for entry in data]

    def _save_index(self, entries: list[IndexEntry]):
        self._ensure_dirs()
        data = [e.to_dict() for e in entries]
        self.index_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _add_to_index(self, log: ExperimentLog, log_file: str) -> IndexEntry:
        """Add an experiment to the index."""
        category = log.config.get("constraints", {}).get("category", "general")
        entry = IndexEntry(
            name=log.name,
            category=category,
            date=log.finished_at or datetime.now(timezone.utc).isoformat(),
            best_concerns=log.best_concerns,
            converged=log.converged,
            n_trials=len(log.trials),
            key_ingredients=_key_ingredients(log),
            summary=_generate_summary(log),
            log_file=log_file,
        )
        entries = self.load_index()
        entries.append(entry)
        self._save_index(entries)
        return entry

    # ------------------------------------------------------------------
    # Layer 2: Full Experiment Logs
    # ------------------------------------------------------------------

    def save_log(self, log: ExperimentLog) -> str:
        """Save full experiment log. Returns the filename."""
        self._ensure_dirs()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{log.name}_{timestamp}.json"
        path = self.logs_dir / filename
        log.save(path)
        return filename

    def load_log(self, filename: str) -> dict:
        """Load a full experiment log by filename."""
        path = self.logs_dir / filename
        return json.loads(path.read_text(encoding="utf-8"))

    def list_logs(self) -> list[str]:
        """List all saved experiment logs."""
        if not self.logs_dir.exists():
            return []
        return sorted(f.name for f in self.logs_dir.glob("*.json"))

    # ------------------------------------------------------------------
    # Layer 3: Discoveries
    # ------------------------------------------------------------------

    def load_discoveries(self) -> list[Discovery]:
        """Load cross-run discoveries."""
        if not self.discoveries_path.exists():
            return []
        data = yaml.safe_load(
            self.discoveries_path.read_text(encoding="utf-8")
        )
        if not data or "discoveries" not in data:
            return []
        return [Discovery.from_dict(d) for d in data["discoveries"]]

    def _save_discoveries(self, discoveries: list[Discovery]):
        self._ensure_dirs()
        data = {"discoveries": [d.to_dict() for d in discoveries]}
        self.discoveries_path.write_text(
            yaml.dump(data, default_flow_style=False, sort_keys=False,
                      allow_unicode=True),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Consolidation
    # ------------------------------------------------------------------

    def consolidate(self, log: ExperimentLog) -> list[Discovery]:
        """
        Extract findings from a completed experiment and merge with
        existing discoveries. Returns newly added discoveries.
        """
        new_findings = _extract_findings(log)
        if not new_findings:
            return []

        existing = self.load_discoveries()
        merged, newly_added = _merge_discoveries(existing, new_findings)
        self._save_discoveries(merged)
        return newly_added

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve_prior_knowledge(
        self,
        category: str,
        ingredients: list[str] | None = None,
    ) -> str:
        """
        Retrieve relevant prior knowledge formatted for the system prompt.

        Matches by domain (category). If ingredients are specified,
        prioritizes discoveries involving those ingredients.
        """
        all_discoveries = self.load_discoveries()
        if not all_discoveries:
            return ""

        # Filter: same domain, or cross-domain patterns with high confidence
        relevant: list[Discovery] = []
        for d in all_discoveries:
            if d.domain == category:
                relevant.append(d)
            elif d.confidence >= 0.8 and d.kind == "pattern":
                relevant.append(d)

        if not relevant:
            return ""

        # Boost score for discoveries that overlap with requested ingredients
        # Work on copies to avoid mutating the loaded objects
        if ingredients:
            ing_upper = {i.upper() for i in ingredients}
            boosted = []
            for d in relevant:
                if any(i.upper() in ing_upper for i in d.ingredients):
                    d_copy = copy.copy(d)
                    d_copy.confidence = min(MAX_CONFIDENCE, d_copy.confidence + 0.05)
                    boosted.append(d_copy)
                else:
                    boosted.append(d)
            relevant = boosted

        return format_prior_knowledge(relevant)

    # ------------------------------------------------------------------
    # Complete save-after-run
    # ------------------------------------------------------------------

    def record_experiment(self, log: ExperimentLog) -> list[Discovery]:
        """
        Save experiment to all three layers. Called after run().

        Returns newly discovered findings.
        """
        self._ensure_dirs()
        log_file = self.save_log(log)
        self._add_to_index(log, log_file)
        new_discoveries = self.consolidate(log)
        return new_discoveries

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Human-readable summary of memory state."""
        index = self.load_index()
        discoveries = self.load_discoveries()
        logs = self.list_logs()

        lines = [
            "",
            "=" * 60,
            "  OPENMIX EXPERIMENT MEMORY",
            "=" * 60,
            f"  Location: {self.base_dir}",
            f"  Experiments: {len(index)}",
            f"  Saved logs: {len(logs)}",
            f"  Discoveries: {len(discoveries)}",
        ]

        if index:
            lines.append("")
            lines.append("  Recent experiments:")
            for entry in index[-10:]:
                status = "OK" if entry.converged else f"{entry.best_concerns:.1f}"
                lines.append(f"    {entry.name} [{entry.category}] "
                             f"concerns={status} trials={entry.n_trials}")
                lines.append(f"      {entry.summary}")

        if discoveries:
            # Count by kind
            by_kind: dict[str, int] = {}
            for d in discoveries:
                by_kind[d.kind] = by_kind.get(d.kind, 0) + 1

            lines.append("")
            lines.append("  Discoveries by type:")
            for kind, count in sorted(by_kind.items()):
                lines.append(f"    {kind}: {count}")

            # High-confidence discoveries
            high_conf = [d for d in discoveries if d.confidence >= 0.6]
            if high_conf:
                lines.append("")
                lines.append("  High-confidence findings (>= 0.6):")
                for d in sorted(high_conf, key=lambda x: -x.confidence)[:10]:
                    lines.append(f"    [{d.confidence:.2f}] {d.finding}")
                    lines.append(f"           {d.evidence_count} experiments, "
                                 f"domain: {d.domain}")

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)
