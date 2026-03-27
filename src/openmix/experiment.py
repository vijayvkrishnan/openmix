"""
Autonomous formulation experiment runner.

The scientist writes a YAML experiment definition.
The agent explores the formulation space, guided by a scoring function.
Every iteration is logged. Patterns are extracted. Rules are discovered.

    exp = Experiment.from_file("experiments/vitamin_c_stability.yaml")
    results = exp.run()

This is the autoresearch pattern applied to formulation science.
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
from openmix.constraints import check_constraints
from openmix.llm import LLMProvider, AnthropicProvider, create_provider


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
    """One formulation attempt in an experiment."""
    iteration: int
    formula: Formula
    stability: StabilityScore
    reasoning: Optional[str] = None
    duration_ms: int = 0
    formula_hash: str = ""

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "score": self.stability.total,
            "subscores": {
                "compatibility": self.stability.compatibility,
                "ph": self.stability.ph_suitability,
                "hlb": self.stability.emulsion_balance,
                "integrity": self.stability.formula_integrity,
                "completeness": self.stability.system_completeness,
            },
            "ingredients": [
                {"name": i.inci_name, "pct": i.percentage, "function": i.function}
                for i in self.formula.ingredients
            ],
            "reasoning": self.reasoning,
            "formula_hash": self.formula_hash,
        }


@dataclass
class ExperimentLog:
    """Complete experiment record — reproducible, shareable, citable."""
    name: str
    brief: str
    trials: list[Trial] = field(default_factory=list)
    best_trial: Optional[Trial] = None
    best_score: float = 0.0
    converged: bool = False
    started_at: str = ""
    finished_at: str = ""
    total_duration_ms: int = 0
    config: dict = field(default_factory=dict)

    @property
    def scores(self) -> list[float]:
        return [t.stability.total for t in self.trials]

    @property
    def best_scores_over_time(self) -> list[float]:
        """Running maximum score across iterations."""
        best = []
        current_best = 0.0
        for t in self.trials:
            current_best = max(current_best, t.stability.total)
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
        lines.append(f"  Best: {self.best_score:.1f}/100")
        lines.append(f"  Status: {'CONVERGED' if self.converged else 'MAX ITERATIONS'}")
        lines.append(f"  Duration: {self.total_duration_ms / 1000:.1f}s")

        # Score trajectory
        if self.trials:
            lines.append("")
            lines.append("  Score Trajectory:")
            bests = self.best_scores_over_time
            for i, (score, best) in enumerate(zip(self.scores, bests)):
                bar = "#" * int(best / 2)
                marker = " *" if score == self.best_score else ""
                lines.append(f"    {i+1:>3}: {score:5.1f} (best: {best:5.1f}) |{bar}{marker}")

        # Best formula
        if self.best_trial:
            lines.append("")
            lines.append("-" * 70)
            lines.append("  BEST FORMULATION")
            lines.append("-" * 70)
            f = self.best_trial.formula
            lines.append(f"  Score: {self.best_score:.1f}/100  |  "
                        f"pH: {f.target_ph}  |  "
                        f"{len(f.ingredients)} ingredients")
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
You iteratively design formulations to maximize a stability score.

EXPERIMENT BRIEF:
{brief}

REQUIRED INGREDIENTS:
{required}

{pool_section}

CONSTRAINTS:
{constraints}

Each iteration, respond with ONLY a JSON object:
{{
  "ingredients": [
    {{"inci_name": "INCI Name", "percentage": 5.0, "phase": "A", "function": "humectant"}}
  ],
  "target_ph": 5.5,
  "reasoning": "Why I chose this formulation and what I changed from last time"
}}

Rules:
- Use proper INCI nomenclature for all ingredient names
- Percentages MUST sum to exactly 100%
- Include ALL required ingredients within their specified ranges
- Include phase (A=water, B=oil, C=cool-down) and function for each
- Your goal: MAXIMIZE the stability score
- Analyze sub-scores to identify which dimension to improve
- Be creative — explore different ingredient combinations and strategies"""


FEEDBACK_PROMPT = """Iteration {iteration} results:

SCORE: {score}/100 (best so far: {best_score}/100)
{score_breakdown}

EXPERIMENT HISTORY (last 5):
{history}

{analysis}

Design the next formulation. Focus on improving the weakest sub-score.
Try something DIFFERENT from previous attempts — explore the space.
Respond with ONLY the JSON object."""


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

        return SYSTEM_PROMPT.format(
            brief=self.brief,
            required=required_str,
            pool_section=pool_section,
            constraints=constraint_str,
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

    def _parse_formula(self, text: str) -> tuple[Formula | None, str | None]:
        data = _parse_json(text)
        if not data:
            return None, None

        reasoning = data.get("reasoning")
        category = self.constraints.get("category", "skincare")
        product_type = self.constraints.get("product_type")

        try:
            return Formula(
                name=f"{self.name} iter",
                ingredients=data.get("ingredients", []),
                target_ph=data.get("target_ph"),
                category=category,
                product_type=product_type,
            ), reasoning
        except Exception:
            return None, reasoning

    def _format_history(self, trials: list[Trial], n: int = 5) -> str:
        recent = trials[-n:]
        lines = []
        for t in recent:
            lines.append(
                f"  Iter {t.iteration}: {t.stability.total:.1f}/100 "
                f"(compat:{t.stability.compatibility:.0f} "
                f"pH:{t.stability.ph_suitability:.0f} "
                f"HLB:{t.stability.emulsion_balance:.0f} "
                f"integ:{t.stability.formula_integrity:.0f} "
                f"complete:{t.stability.system_completeness:.0f})"
            )
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

        best_score = 0.0
        seen_hashes = set()

        self._log("")
        self._log("=" * 70)
        self._log(f"  OPENMIX EXPERIMENT: {self.name}")
        self._log("=" * 70)
        self._log(f"  {self.brief[:100]}")
        self._log(f"  Pool: {len(self.ingredient_pool)} ingredients  |  "
                  f"Target: {self.target_score}/100  |  "
                  f"Max: {self.max_iterations} iterations")
        self._log("")

        for i in range(self.max_iterations):
            iter_num = i + 1

            # --- Propose ---
            self._log(f"  [{iter_num:>2}] ", end="")
            gen_start = time.time()
            raw = self._call_llm(system_prompt, messages)
            gen_ms = int((time.time() - gen_start) * 1000)

            formula, reasoning = self._parse_formula(raw)
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

            # --- Evaluate ---
            stability = self.evaluate(formula)
            fhash = _hash_formula(formula)
            is_dupe = fhash in seen_hashes
            seen_hashes.add(fhash)

            trial = Trial(
                iteration=iter_num,
                formula=formula,
                stability=stability,
                reasoning=reasoning,
                duration_ms=gen_ms,
                formula_hash=fhash,
            )
            log.trials.append(trial)

            if stability.total > best_score:
                best_score = stability.total
                log.best_trial = trial
                log.best_score = best_score
                marker = " *NEW BEST*"
            else:
                marker = ""

            dupe_tag = " (DUPE)" if is_dupe else ""
            self._log(
                f"{stability.total:5.1f}/100  "
                f"(c:{stability.compatibility:4.1f} "
                f"pH:{stability.ph_suitability:4.1f} "
                f"H:{stability.emulsion_balance:4.1f} "
                f"i:{stability.formula_integrity:4.1f} "
                f"s:{stability.system_completeness:4.1f}) "
                f"({gen_ms/1000:.1f}s){marker}{dupe_tag}"
            )

            # --- Converged? ---
            if stability.total >= self.target_score:
                self._log(f"\n  Converged at iteration {iter_num}.")
                log.converged = True
                break

            # --- Prepare feedback ---
            analysis = ""
            if is_dupe:
                analysis = "WARNING: This is a duplicate formulation. Try something DIFFERENT."
            if len(log.trials) >= 3:
                recent_scores = [t.stability.total for t in log.trials[-3:]]
                if max(recent_scores) - min(recent_scores) < 1.0:
                    analysis += ("\nScores are plateauing. Make a BIGGER change: "
                                "try different ingredients, different ratios, "
                                "or a fundamentally different approach.")

            feedback = FEEDBACK_PROMPT.format(
                iteration=iter_num,
                score=f"{stability.total:.1f}",
                best_score=f"{best_score:.1f}",
                score_breakdown=str(stability),
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

        return log
