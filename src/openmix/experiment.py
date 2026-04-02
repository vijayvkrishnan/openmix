"""
Autonomous formulation experiment runner.

    exp = Experiment.from_brief("Design a stable vitamin C serum").run()
    exp = Experiment.from_file("experiments/vitamin_c_stability.yaml").run()
"""

from __future__ import annotations

import json
import re
import time
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Callable

import yaml

from openmix.schema import Formula
from openmix.score import score as heuristic_score, StabilityScore
from openmix.observe import observe, FormulationObservation
from openmix.constraints import check_constraints
from openmix.llm import LLMProvider, AnthropicProvider, create_provider
from openmix.memory import ExperimentMemory
from openmix.discourse import evaluate_discourse


def _parse_json(text: str) -> dict | None:
    """Extract JSON from LLM response (handles markdown wrapping)."""
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        text = json_match.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start:end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _save_plan_yaml(plan: dict, brief: str, path: Path):
    """Save a generated experiment plan as YAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "name": plan.get("name", "experiment"),
        "brief": brief,
        "ingredient_pool": {
            "required": plan.get("required_ingredients", []),
            "available": plan.get("available_ingredients", []),
        },
        "constraints": {
            k: v for k, v in plan.items()
            if k in ("target_ph", "max_ingredients", "category", "product_type")
        },
        "settings": {
            "max_iterations": plan.get("max_iterations", 20),
            "target_score": plan.get("target_score", 90),
            "mode": plan.get("mode", "formulation"),
        },
    }
    output["constraints"]["total_percentage"] = 100
    path.write_text(yaml.dump(output, default_flow_style=False, sort_keys=False),
                    encoding="utf-8")


# ---------------------------------------------------------------------------
# Experiment Log — the lab notebook
# ---------------------------------------------------------------------------

@dataclass
class Trial:
    """One formulation attempt in an experiment — an artifact with lineage."""
    iteration: int
    formula: Formula
    observation: Optional[FormulationObservation] = None
    stability: Optional[StabilityScore] = None  # backward compat
    reasoning: Optional[str] = None
    parent_hash: Optional[str] = None  # what formula this was derived from
    duration_ms: int = 0
    formula_hash: str = ""

    @property
    def concern_count(self) -> float:
        if self.observation:
            return self.observation.concern_count
        if self.stability:
            return 100 - self.stability.total
        return 0

    def to_dict(self) -> dict:
        d = {
            "iteration": self.iteration,
            "concern_count": self.concern_count,
            "ingredients": [
                {"name": i.inci_name, "pct": i.percentage, "function": i.function}
                for i in self.formula.ingredients
            ],
            "reasoning": self.reasoning,
            "formula_hash": self.formula_hash,
            "parent_hash": self.parent_hash,
        }
        if self.observation:
            d["hard_violations"] = self.observation.hard_violations
            d["soft_violations"] = self.observation.soft_violations
            d["concerns"] = len(self.observation.concerns)
            d["resolution_rate"] = self.observation.resolution_rate
        if self.stability:
            d["score"] = self.stability.total
        return d


@dataclass
class ExperimentLog:
    """Complete experiment record — reproducible, shareable, citable."""
    name: str
    brief: str
    trials: list[Trial] = field(default_factory=list)
    best_trial: Optional[Trial] = None
    best_score: float = 0.0
    best_concerns: float = float("inf")
    converged: bool = False
    started_at: str = ""
    finished_at: str = ""
    total_duration_ms: int = 0
    config: dict = field(default_factory=dict)

    @property
    def scores(self) -> list[float]:
        return [t.stability.total if t.stability else (100 - t.concern_count) for t in self.trials]

    @property
    def concern_trajectory(self) -> list[float]:
        return [t.concern_count for t in self.trials]

    @property
    def best_scores_over_time(self) -> list[float]:
        """Running best concern count across iterations (lower = better)."""
        best = []
        current_best = float("inf")
        for t in self.trials:
            current_best = min(current_best, t.concern_count)
            best.append(current_best)
        return best

    @property
    def unique_formulations(self) -> int:
        return len(set(t.formula_hash for t in self.trials))

    def save(self, path: str | Path):
        """Save experiment log as JSON."""
        data = {
            "openmix_version": "0.1.0",
            "experiment": self.name,
            "brief": self.brief,
            "config": self.config,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": self.total_duration_ms / 1000,
            "converged": self.converged,
            "total_trials": len(self.trials),
            "unique_formulations": self.unique_formulations,
            "best_score": self.best_score,
            "score_trajectory": self.scores,
            "best_trajectory": self.best_scores_over_time,
            "best_formula": self.best_trial.to_dict() if self.best_trial else None,
            "trials": [t.to_dict() for t in self.trials],
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def __str__(self) -> str:
        lines = []
        lines.append("")
        lines.append("=" * 70)
        lines.append("  OPENMIX EXPERIMENT REPORT")
        lines.append("=" * 70)
        lines.append(f"  {self.name}")
        lines.append(f"  {self.brief[:100]}")
        lines.append("")
        lines.append(f"  Trials: {len(self.trials)} "
                     f"({self.unique_formulations} unique)")
        lines.append(f"  Best concerns: {self.best_concerns:.1f} (lower = better, 0 = clean)")
        lines.append(f"  Status: {'CONVERGED' if self.converged else 'MAX ITERATIONS'}")
        lines.append(f"  Duration: {self.total_duration_ms / 1000:.1f}s")

        # Concern trajectory
        if self.trials:
            lines.append("")
            lines.append("  Concern Trajectory (lower = better):")
            best_so_far = float("inf")
            for i, c in enumerate(self.concern_trajectory):
                best_so_far = min(best_so_far, c)
                bar_len = min(50, int(c))
                bar = "!" * bar_len if c > 0 else "-"
                marker = " *" if c == self.best_concerns else ""
                lines.append(f"    {i+1:>3}: {c:5.1f} (best: {best_so_far:5.1f}) |{bar}{marker}")

        # Best formula
        if self.best_trial:
            lines.append("")
            lines.append("-" * 70)
            lines.append("  BEST FORMULATION")
            lines.append("-" * 70)
            f = self.best_trial.formula
            obs = self.best_trial.observation
            lines.append(f"  Concerns: {self.best_concerns:.1f}  |  "
                        f"pH: {f.target_ph}  |  "
                        f"{len(f.ingredients)} ingredients")
            if obs:
                lines.append(f"  Violations: {obs.hard_violations} hard, "
                            f"{obs.soft_violations} soft  |  "
                            f"Resolved: {obs.resolution_rate:.0%}")
            if self.best_trial.reasoning:
                r = self.best_trial.reasoning[:200]
                lines.append(f"  Strategy: {r}")
            lines.append("")
            for ing in sorted(f.ingredients, key=lambda x: -x.percentage):
                func = ing.function or ""
                lines.append(f"    {ing.inci_name:<40} {ing.percentage:>5.1f}%  {func}")

        lines.append("")
        lines.append("=" * 70)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Experiment Definition
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Parsed experiment definition from YAML."""
    name: str
    brief: str
    required_ingredients: list[dict] = field(default_factory=list)
    available_ingredients: list[str] = field(default_factory=list)
    constraints: dict = field(default_factory=dict)
    max_iterations: int = 30
    target_score: float = 95.0
    mode: str = "formulation"
    evaluate: str = "auto"
    llm_config: dict | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> ExperimentConfig:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        pool = data.get("ingredient_pool", {})
        settings = data.get("settings", {})

        return cls(
            name=data.get("name", "unnamed"),
            brief=data.get("brief", ""),
            required_ingredients=pool.get("required", []),
            available_ingredients=pool.get("available", []),
            constraints=data.get("constraints", {}),
            max_iterations=settings.get("max_iterations", 30),
            target_score=settings.get("target_score", 95.0),
            mode=settings.get("mode", "formulation"),
            evaluate=settings.get("evaluate", "auto"),
            llm_config=data.get("llm"),
        )


# ---------------------------------------------------------------------------
# The Experiment Runner
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert formulation scientist running an autonomous experiment.
You iteratively design formulations guided by physics observations.

EXPERIMENT BRIEF:
{brief}

REQUIRED INGREDIENTS:
{required}

{pool_section}

CONSTRAINTS:
{constraints}

{prior_knowledge}
Each iteration, respond with ONLY a JSON object:
{{
  "ingredients": [
    {{"inci_name": "INCI Name", "percentage": 5.0, "phase": "A", "function": "humectant"}}
  ],
  "target_ph": 5.5,
  "protocol": {{
    "phases": [
      {{"name": "A", "label": "Water Phase", "target_temp_c": 75, "ingredients": ["Water", "Glycerin"]}},
      {{"name": "B", "label": "Oil Phase", "target_temp_c": 75, "ingredients": ["Squalane"]}},
      {{"name": "C", "label": "Cool-Down", "target_temp_c": 40, "ingredients": ["Retinol", "Phenoxyethanol"]}}
    ],
    "steps": [
      {{"action": "heat", "phase": "A", "parameters": {{"temp_c": 75, "duration_min": 10}}}},
      {{"action": "heat", "phase": "B", "parameters": {{"temp_c": 75, "duration_min": 10}}}},
      {{"action": "combine", "phase": "B", "parameters": {{"into": "A"}}, "notes": "Add oil to water slowly"}},
      {{"action": "homogenize", "phase": "all", "parameters": {{"rpm": 5000, "duration_min": 3}}}},
      {{"action": "cool", "phase": "all", "parameters": {{"target_c": 40}}}},
      {{"action": "add", "phase": "C", "parameters": {{}}}},
      {{"action": "adjust_ph", "phase": "all", "parameters": {{"target": 5.5}}}}
    ],
    "equipment": ["overhead stirrer", "homogenizer", "pH meter"]
  }},
  "reasoning": "Why I chose this formulation and what I changed from last time"
}}

Rules:
- Use proper INCI nomenclature for all ingredient names
- Percentages MUST sum to exactly 100%
- Include ALL required ingredients within their specified ranges
- Include phase (A=water, B=oil, C=cool-down) and function for each ingredient
- Include a protocol with phases, steps, and equipment
- Assign heat-sensitive ingredients (retinol, ascorbic acid, preservatives) to cool-down phase
- Include a homogenize step if oil phase > 5%
- Include adjust_ph step if target_ph is specified
- Your goal: resolve ALL violations, physics concerns, AND process concerns
- Read the assessment carefully — it includes multiple perspectives
- Hard violations MUST be resolved (dangerous combinations)
- TRUE DISAGREEMENTS are where perspectives conflict — investigate these
- Be creative — explore different ingredient combinations and strategies"""


FEEDBACK_PROMPT = """Iteration {iteration} results:

CONCERN COUNT: {concerns} (best so far: {best_concerns})

{assessment_section}

EXPERIMENT HISTORY (last 5):
{history}

{analysis}

Read the assessment carefully. Resolve violations first (hard, then soft).
Pay attention to TRUE DISAGREEMENTS — these are where perspectives conflict
and may indicate opportunities or risks worth investigating.
Address corrections, then concerns. Respond with ONLY the JSON object."""


def _hash_formula(formula: Formula) -> str:
    """Deterministic hash of a formula for dedup."""
    parts = sorted(
        f"{i.inci_name.upper().strip()}:{i.percentage:.2f}"
        for i in formula.ingredients
    )
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:12]


PLANNING_PROMPT = """You are a formulation scientist planning an experiment.
Given a research brief in natural language, generate a structured experiment plan.

Respond with ONLY a JSON object:
{{
  "name": "short-kebab-case-name",
  "category": "skincare",
  "product_type": "serum",
  "target_ph": [4.0, 6.0],
  "required_ingredients": [
    {{"name": "Ingredient", "min_pct": 5.0, "max_pct": 20.0, "function": "active"}}
  ],
  "max_ingredients": 15,
  "mode": "formulation",
  "max_iterations": 20,
  "target_score": 90
}}

Rules:
- "category" must be one of: skincare, supplement, beverage, home_care, food, pharma
- Only include "required_ingredients" that the brief specifically asks for
- Do NOT include an "available_ingredients" list — the agent has open access to all ingredients
- Set reasonable constraints based on the product type
- "mode" should be "formulation" for consumer products, "discovery" for research"""


class Experiment:
    """
    Autonomous formulation experiment.

    From natural language (recommended):
        result = Experiment.from_brief("Design a stable vitamin C serum").run()

    From YAML file (for reproducibility):
        result = Experiment.from_file("experiment.yaml").run()

    From Python (full control):
        result = Experiment(brief="...", ingredient_pool=[...]).run()
    """

    def __init__(
        self,
        name: str = "experiment",
        brief: str = "",
        ingredient_pool: list[str] | None = None,
        required_ingredients: list[dict] | None = None,
        constraints: dict | None = None,
        evaluate: Callable[[Formula], StabilityScore] | None = None,
        max_iterations: int = 30,
        target_score: float = 95.0,
        mode: str = "formulation",
        llm: LLMProvider | None = None,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        verbose: bool = True,
        use_memory: bool = True,
    ):
        self.name = name
        self.brief = brief
        self.ingredient_pool = ingredient_pool or []
        self.required_ingredients = required_ingredients or []
        self.constraints = constraints or {}
        self.evaluate = evaluate or heuristic_score
        self.max_iterations = max_iterations
        self.target_score = target_score
        self.mode = mode
        self.verbose = verbose
        self.use_memory = use_memory
        self.memory = ExperimentMemory() if use_memory else None

        # Map experiment mode to observation mode
        self._observe_mode = "discovery" if mode == "discovery" else "engineering"

        # LLM: use provided, or create default Anthropic
        if llm:
            self.llm = llm
        else:
            self.llm = AnthropicProvider(model=model, api_key=api_key)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        llm: LLMProvider | None = None,
        api_key: str | None = None,
        evaluate: Callable[[Formula], StabilityScore] | None = None,
        verbose: bool = True,
        use_memory: bool = True,
    ) -> Experiment:
        config = ExperimentConfig.from_file(path)

        # Build LLM from config if not provided
        if not llm and config.llm_config:
            llm = create_provider(config.llm_config)

        return cls(
            name=config.name,
            brief=config.brief,
            ingredient_pool=config.available_ingredients,
            required_ingredients=config.required_ingredients,
            constraints=config.constraints,
            evaluate=evaluate,
            max_iterations=config.max_iterations,
            target_score=config.target_score,
            mode=config.mode,
            llm=llm,
            api_key=api_key,
            verbose=verbose,
            use_memory=use_memory,
        )

    @classmethod
    def from_brief(
        cls,
        brief: str,
        llm: LLMProvider | None = None,
        api_key: str | None = None,
        evaluate: Callable[[Formula], StabilityScore] | None = None,
        verbose: bool = True,
        save_plan: str | Path | None = None,
        use_memory: bool = True,
    ) -> Experiment:
        """
        Create an experiment from a natural language brief.

        The LLM generates a structured experiment plan (ingredient pool,
        constraints, etc.) from plain English. The plan can optionally
        be saved as a YAML file for reproducibility.

            result = Experiment.from_brief(
                "Design a stable vitamin C serum under $30/kg"
            ).run()
        """
        provider = llm or AnthropicProvider(api_key=api_key)

        if verbose:
            print("\n  Planning experiment from brief...")
            print(f"  \"{brief[:80]}{'...' if len(brief) > 80 else ''}\"")

        response = provider.generate(PLANNING_PROMPT, [
            {"role": "user", "content": brief}
        ])

        # Parse the plan
        plan = _parse_json(response)
        if not plan:
            raise ValueError("Could not parse experiment plan from LLM response")

        # Build constraints
        constraints = {}
        if "target_ph" in plan:
            constraints["target_ph"] = plan["target_ph"]
        if "max_ingredients" in plan:
            constraints["max_ingredients"] = plan["max_ingredients"]
        constraints["total_percentage"] = 100
        if "category" in plan:
            constraints["category"] = plan["category"]
        if "product_type" in plan:
            constraints["product_type"] = plan["product_type"]

        if verbose:
            n_req = len(plan.get("required_ingredients", []))
            print(f"  Plan: {plan.get('name', 'unnamed')} | "
                  f"{n_req} required ingredients | "
                  f"open pool | mode: {plan.get('mode', 'formulation')}")
            print()

        # Optionally save the generated plan as YAML
        if save_plan:
            _save_plan_yaml(plan, brief, Path(save_plan))
            if verbose:
                print(f"  Plan saved to {save_plan}")

        return cls(
            name=plan.get("name", "experiment"),
            brief=brief,
            ingredient_pool=[],  # open pool — agent selects from full ingredient universe
            required_ingredients=plan.get("required_ingredients", []),
            constraints=constraints,
            evaluate=evaluate,
            max_iterations=plan.get("max_iterations", 20),
            target_score=plan.get("target_score", 90),
            mode=plan.get("mode", "formulation"),
            llm=provider,
            verbose=verbose,
            use_memory=use_memory,
        )

    def _log(self, msg: str, end: str = "\n"):
        if self.verbose:
            print(msg, end=end, flush=True)

    def _build_system_prompt(self) -> str:
        required_str = "\n".join(
            f"  - {r['name']}: {r.get('min_pct', 0)}-{r.get('max_pct', 100)}% ({r.get('function', '')})"
            for r in self.required_ingredients
        ) or "(none)"

        if self.ingredient_pool:
            pool_section = f"SUGGESTED INGREDIENTS (you may also use others):\n{', '.join(self.ingredient_pool)}"
        else:
            pool_section = "You have access to ALL ingredients. Use your expertise to select appropriate ones."

        constraint_str = "\n".join(
            f"  {k}: {v}" for k, v in self.constraints.items()
        ) or "(none)"

        # Retrieve prior knowledge from memory
        prior_knowledge = ""
        if self.memory:
            category = self.constraints.get("category", "general")
            ingredient_names = [r["name"] for r in self.required_ingredients]
            prior_knowledge = self.memory.retrieve_prior_knowledge(
                category=category,
                ingredients=ingredient_names or None,
            )

        return SYSTEM_PROMPT.format(
            brief=self.brief,
            required=required_str,
            pool_section=pool_section,
            constraints=constraint_str,
            prior_knowledge=prior_knowledge,
        )

    def _call_llm(self, system: str, messages: list[dict]) -> str:
        # Keep context manageable: last 6 messages (3 turns) + first message
        if len(messages) > 8:
            messages = [messages[0]] + messages[-6:]

        for attempt in range(3):
            try:
                return self.llm.generate(system, messages)
            except Exception as e:
                if "rate_limit" in str(e).lower() and attempt < 2:
                    import time as _time
                    wait = 15 * (attempt + 1)
                    self._log(f"\n    Rate limited. Waiting {wait}s...")
                    _time.sleep(wait)
                else:
                    raise

    def _parse_formula(self, text: str) -> tuple:
        from openmix.protocol import Protocol, Phase, ProcessStep

        data = _parse_json(text)
        if not data:
            return None, None, None

        reasoning = data.get("reasoning")
        category = self.constraints.get("category", "skincare")
        product_type = self.constraints.get("product_type")

        # Parse protocol if present
        protocol = None
        proto_data = data.get("protocol")
        if proto_data and isinstance(proto_data, dict):
            try:
                phases = [
                    Phase(
                        name=p.get("name", ""),
                        label=p.get("label", ""),
                        target_temp_c=p.get("target_temp_c"),
                        ingredients=p.get("ingredients", []),
                    )
                    for p in proto_data.get("phases", [])
                ]
                steps = [
                    ProcessStep(
                        action=s.get("action", ""),
                        phase=s.get("phase", "all"),
                        parameters=s.get("parameters", {}),
                        notes=s.get("notes", ""),
                    )
                    for s in proto_data.get("steps", [])
                ]
                protocol = Protocol(
                    phases=phases,
                    steps=steps,
                    equipment=proto_data.get("equipment", []),
                )
            except Exception:
                pass  # protocol parsing failed, continue without it

        try:
            formula = Formula(
                name=f"{self.name} iter",
                ingredients=data.get("ingredients", []),
                target_ph=data.get("target_ph"),
                category=category,
                product_type=product_type,
            )
            return formula, reasoning, protocol
        except Exception:
            return None, reasoning, None

    def _format_history(self, trials: list[Trial], n: int = 5) -> str:
        recent = trials[-n:]
        lines = []
        for t in recent:
            if t.observation:
                v = t.observation
                lines.append(
                    f"  Iter {t.iteration}: concerns={t.concern_count:.1f} "
                    f"({v.hard_violations}H/{v.soft_violations}S violations, "
                    f"{len(v.concerns)} physics concerns)"
                )
            elif t.stability:
                lines.append(f"  Iter {t.iteration}: score={t.stability.total:.1f}/100")
            if t.reasoning:
                lines.append(f"    Strategy: {t.reasoning[:100]}")
        return "\n".join(lines) or "(first attempt)"

    def run(self) -> ExperimentLog:
        """Run the experiment. Returns a complete log."""
        log = ExperimentLog(
            name=self.name,
            brief=self.brief,
            config={
                "max_iterations": self.max_iterations,
                "target_score": self.target_score,
                "mode": self.mode,
                "ingredient_pool": self.ingredient_pool,
                "required_ingredients": self.required_ingredients,
                "constraints": self.constraints,
            },
            started_at=datetime.now(timezone.utc).isoformat(),
        )

        start_time = time.time()
        system_prompt = self._build_system_prompt()
        messages = [{"role": "user", "content":
            "Design the first formulation. Start with a solid baseline approach."}]

        best_concerns = float("inf")
        seen_hashes = set()

        self._log("")
        self._log("=" * 70)
        self._log(f"  OPENMIX EXPERIMENT: {self.name}")
        self._log("=" * 70)
        self._log(f"  {self.brief[:100]}")
        self._log(f"  Pool: {len(self.ingredient_pool)} ingredients  |  "
                  f"Target: zero concerns  |  "
                  f"Max: {self.max_iterations} iterations")
        self._log("")

        for i in range(self.max_iterations):
            iter_num = i + 1

            # --- Propose ---
            self._log(f"  [{iter_num:>2}] ", end="")
            gen_start = time.time()
            raw = self._call_llm(system_prompt, messages)
            gen_ms = int((time.time() - gen_start) * 1000)

            formula, reasoning, protocol = self._parse_formula(raw)
            if not formula:
                self._log(f"parse error ({gen_ms/1000:.1f}s)")
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content":
                    "Invalid JSON. Respond with ONLY a JSON object."})
                continue

            # --- Check constraints ---
            constraint_result = check_constraints(
                formula, self.required_ingredients,
                self.ingredient_pool if self.ingredient_pool else None,
                self.constraints,
            )
            if not constraint_result.passed:
                self._log(f"REJECTED ({gen_ms/1000:.1f}s)")
                for v in constraint_result.violations[:2]:
                    self._log(f"    {v.constraint}: {v.message}")
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content":
                    f"Formula rejected — constraint violations:\n{constraint_result}\n"
                    f"Fix these and respond with ONLY the corrected JSON."})
                continue

            # --- Evaluate (physics + discourse if memory available) ---
            obs = observe(formula, mode=self._observe_mode)
            disc = None
            if self.memory:
                disc = evaluate_discourse(
                    formula,
                    protocol=protocol,
                    memory=self.memory,
                    observe_mode=self._observe_mode,
                )

            # Also run legacy scorer if a custom evaluate function is provided
            stability = self.evaluate(formula) if self.evaluate != heuristic_score else None

            fhash = _hash_formula(formula)
            prev_hash = log.trials[-1].formula_hash if log.trials else None
            is_dupe = fhash in seen_hashes
            seen_hashes.add(fhash)

            trial = Trial(
                iteration=iter_num,
                formula=formula,
                observation=obs,
                stability=stability,
                reasoning=reasoning,
                parent_hash=prev_hash,
                duration_ms=gen_ms,
                formula_hash=fhash,
            )
            log.trials.append(trial)

            concerns = obs.concern_count
            if concerns < best_concerns:
                best_concerns = concerns
                log.best_trial = trial
                log.best_concerns = best_concerns
                log.best_score = obs.concern_score
                marker = " *NEW BEST*"
            else:
                marker = ""

            dupe_tag = " (DUPE)" if is_dupe else ""
            v_str = f"{obs.hard_violations}H/{obs.soft_violations}S"
            self._log(
                f"concerns:{concerns:5.1f}  "
                f"violations:{v_str}  "
                f"({gen_ms/1000:.1f}s){marker}{dupe_tag}"
            )

            # --- Converged? ---
            if concerns == 0:
                self._log(f"\n  Converged at iteration {iter_num} -- zero concerns.")
                log.converged = True
                break

            # --- Prepare feedback ---
            analysis = ""
            if is_dupe:
                analysis = "WARNING: This is a duplicate formulation. Try something DIFFERENT."
            if len(log.trials) >= 3:
                recent = [t.concern_count for t in log.trials[-3:]]
                if max(recent) - min(recent) < 0.5:
                    analysis += ("\nConcerns are stagnant. Make a BIGGER change: "
                                "try different ingredients, different ratios, "
                                "or a fundamentally different approach.")

            # Build assessment: discourse if available, physics-only otherwise
            if disc and disc.topics:
                assessment_section = f"MULTI-PERSPECTIVE ASSESSMENT:\n{disc}"
            else:
                assessment_section = f"PHYSICS OBSERVATIONS:\n{obs}"

            feedback = FEEDBACK_PROMPT.format(
                iteration=iter_num,
                concerns=f"{concerns:.1f}",
                best_concerns=f"{best_concerns:.1f}",
                assessment_section=assessment_section,
                history=self._format_history(log.trials),
                analysis=analysis,
            )
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": feedback})

        log.finished_at = datetime.now(timezone.utc).isoformat()
        log.total_duration_ms = int((time.time() - start_time) * 1000)

        if self.verbose:
            print(log)

            # Post-experiment analysis
            if len(log.trials) >= 3:
                from openmix.analysis import analyze
                exp_analysis = analyze(log)
                print(exp_analysis)

        # Record to memory (consolidate discoveries)
        if self.memory and log.trials:
            try:
                new_discoveries = self.memory.record_experiment(log)
                if self.verbose and new_discoveries:
                    n_pref = sum(1 for d in new_discoveries if d.kind == "preference")
                    n_avoid = sum(1 for d in new_discoveries if d.kind == "avoidance")
                    n_other = len(new_discoveries) - n_pref - n_avoid
                    self._log("")
                    self._log("  Memory updated:")
                    self._log(f"    {n_pref} ingredient preferences, "
                              f"{n_avoid} avoidances, {n_other} patterns recorded")
                    idx = self.memory.load_index()
                    disc = self.memory.load_discoveries()
                    self._log(f"    Total: {len(idx)} experiments, "
                              f"{len(disc)} discoveries")
            except Exception as e:
                if self.verbose:
                    self._log(f"  (Memory save failed: {e})")

        return log
