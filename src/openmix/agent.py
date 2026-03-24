"""
Iterative formulation optimization agent.

Propose -> Score -> Analyze -> Improve -> Repeat.
Uses an LLM for generation and OpenMix scoring for evaluation.

Requires: pip install anthropic
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from openmix.schema import Formula, ValidationReport
from openmix.validate import validate
from openmix.score import score, StabilityScore

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


SYSTEM_PROMPT = """You are an expert formulation scientist. You design stable, effective, and safe
formulations by iteratively optimizing your designs based on quantitative feedback.

When asked to design a formulation, respond with ONLY a JSON object (no markdown, no wrapping):

{
  "name": "Product Name",
  "category": "skincare",
  "product_type": "serum",
  "target_ph": 5.5,
  "ingredients": [
    {"inci_name": "INCI Name", "percentage": 5.0, "phase": "A", "function": "humectant"}
  ],
  "reasoning": "Brief explanation of your formulation strategy and what you changed"
}

Rules:
- All percentages must sum to exactly 100%
- Use proper INCI nomenclature
- Include a complete preservative system for water-based formulas
- Specify phase (A=water, B=oil, C=cool-down) and function for each ingredient
- The "category" must be one of: skincare, supplement, beverage, home_care, food, pharma
- Use "pharma" for pharmaceutical, drug delivery, or medicinal formulations
- Include target_ph appropriate for the product type

When given scoring feedback, analyze which sub-scores are lowest and focus your
changes there. Your goal is to maximize the stability score while meeting the
original brief requirements. Think about WHY each change improves the score."""


ITERATION_PROMPT = """Here are the results of your formulation (iteration {iteration}):

STABILITY SCORE: {score}/100
{score_breakdown}

VALIDATION ISSUES:
{issues}

HISTORY (previous attempts):
{history}

Analyze what's working and what isn't. Focus on improving the lowest sub-scores.
Propose an improved formulation that addresses the specific issues while maintaining
the original requirements. Explain your reasoning in the "reasoning" field.

Respond with ONLY the JSON object."""


@dataclass
class Iteration:
    """One iteration of the optimization loop."""
    number: int
    formula: Optional[Formula] = None
    stability: Optional[StabilityScore] = None
    report: Optional[ValidationReport] = None
    reasoning: Optional[str] = None
    duration_ms: int = 0


@dataclass
class AgentResult:
    """Complete result from the formulation agent."""
    brief: str
    iterations: list[Iteration] = field(default_factory=list)
    best_formula: Optional[Formula] = None
    best_score: float = 0.0
    converged: bool = False
    total_duration_ms: int = 0

    def __str__(self) -> str:
        lines = []
        lines.append("")
        lines.append("=" * 70)
        lines.append("  OPENMIX FORMULATION AGENT")
        lines.append("=" * 70)
        lines.append(f"  Brief: {self.brief}")
        lines.append(f"  Iterations: {len(self.iterations)}")
        lines.append(f"  Best Score: {self.best_score:.1f}/100")
        lines.append(f"  Duration: {self.total_duration_ms / 1000:.1f}s")
        lines.append(f"  Status: {'CONVERGED' if self.converged else 'MAX ITERATIONS'}")
        lines.append("")

        # Score trajectory
        scores = [it.stability.total for it in self.iterations if it.stability]
        if scores:
            lines.append("  Score Trajectory:")
            for i, s in enumerate(scores):
                bar = "#" * int(s / 2)
                marker = " <-- best" if s == max(scores) else ""
                lines.append(f"    Iter {i+1}: {s:5.1f} |{bar}{marker}")
            lines.append("")

        # Show iteration details
        lines.append("-" * 70)
        for it in self.iterations:
            lines.append(f"  Iteration {it.number}")
            if it.reasoning:
                # Truncate reasoning
                r = it.reasoning[:200] + "..." if len(it.reasoning) > 200 else it.reasoning
                lines.append(f"    Strategy: {r}")
            if it.stability:
                lines.append(f"    Score: {it.stability.total:.1f}/100  "
                             f"(compat:{it.stability.compatibility:.0f} "
                             f"pH:{it.stability.ph_suitability:.0f} "
                             f"HLB:{it.stability.emulsion_balance:.0f} "
                             f"integ:{it.stability.formula_integrity:.0f} "
                             f"complete:{it.stability.system_completeness:.0f})")
            if it.report and (it.report.errors > 0 or it.report.warnings > 0):
                lines.append(f"    Issues: {it.report.errors} errors, {it.report.warnings} warnings")
                for issue in it.report.issues:
                    if issue.severity in ("error", "warning"):
                        icon = "X" if issue.severity == "error" else "!"
                        name = f"{issue.ingredient} + {issue.ingredient_b}" if issue.ingredient_b else (issue.ingredient or "")
                        msg = issue.message[:80] + "..." if len(issue.message) > 80 else issue.message
                        lines.append(f"    [{icon}] {name}: {msg}")
            lines.append(f"    ({it.duration_ms / 1000:.1f}s)")
            lines.append("")

        # Final formula
        if self.best_formula:
            lines.append("=" * 70)
            lines.append("  BEST FORMULA")
            lines.append("=" * 70)
            lines.append(f"  {self.best_formula.name or 'Untitled'}")
            lines.append(f"  Category: {self.best_formula.category}  |  "
                         f"pH: {self.best_formula.target_ph}  |  "
                         f"Score: {self.best_score:.1f}/100")
            lines.append("")
            lines.append(f"  {'INCI Name':<40} {'%':>6}  {'Phase':<6}  {'Function'}")
            lines.append(f"  {'-' * 40} {'-' * 6}  {'-' * 6}  {'-' * 20}")
            for ing in sorted(self.best_formula.ingredients, key=lambda x: -x.percentage):
                name = ing.inci_name[:40]
                phase = ing.phase or "-"
                func = ing.function or "-"
                lines.append(f"  {name:<40} {ing.percentage:>5.1f}%  {phase:<6}  {func}")
            lines.append(f"  {'':>40} {'-' * 6}")
            lines.append(f"  {'Total':>40} {self.best_formula.total_percentage:>5.1f}%")

        lines.append("")
        lines.append("=" * 70)
        return "\n".join(lines)


class FormulationAgent:
    """
    AI agent that optimizes formulations through iterative exploration.

    Unlike a simple "generate and check" loop, this agent:
    1. Proposes a candidate formulation
    2. Scores it quantitatively (stability score 0-100)
    3. Validates it qualitatively (interaction rules, safety)
    4. Analyzes what worked and what didn't
    5. Proposes an improved version
    6. Repeats until converging or hitting max iterations
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        max_iterations: int = 5,
        target_score: float = 90.0,
        verbose: bool = True,
    ):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "The anthropic package is required. Install: pip install anthropic"
            )

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_iterations = max_iterations
        self.target_score = target_score
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _call_llm(self, messages: list[dict]) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        return response.content[0].text

    def _parse_formula(self, text: str) -> tuple[Formula | None, str | None]:
        """Parse LLM response into Formula + reasoning."""
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if json_match:
            text = json_match.group(1)
        else:
            brace_start = text.find("{")
            brace_end = text.rfind("}")
            if brace_start != -1 and brace_end != -1:
                text = text[brace_start:brace_end + 1]

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None, None

        reasoning = data.get("reasoning")

        try:
            ingredients = []
            for ing in data.get("ingredients", []):
                if isinstance(ing, dict):
                    ingredients.append(ing)
                elif isinstance(ing, (list, tuple)):
                    ingredients.append({"inci_name": ing[0], "percentage": ing[1]})

            formula = Formula(
                name=data.get("name"),
                ingredients=ingredients,
                target_ph=data.get("target_ph"),
                category=data.get("category"),
                product_type=data.get("product_type"),
            )
            return formula, reasoning
        except Exception:
            return None, reasoning

    def _format_history(self, iterations: list[Iteration]) -> str:
        """Format iteration history for the LLM."""
        if not iterations:
            return "(first attempt)"

        lines = []
        for it in iterations:
            lines.append(f"Iteration {it.number}: {it.stability.total:.1f}/100")
            if it.reasoning:
                lines.append(f"  Strategy: {it.reasoning[:150]}")
            if it.stability:
                for p in it.stability.penalties[:3]:
                    lines.append(f"  Penalty: {p}")
                for b in it.stability.bonuses[:3]:
                    lines.append(f"  Bonus: {b}")
        return "\n".join(lines)

    def _format_issues(self, report: ValidationReport) -> str:
        if not report.issues:
            return "No issues found."
        lines = []
        for issue in report.issues:
            sev = issue.severity.upper()
            name = f"{issue.ingredient} + {issue.ingredient_b}" if issue.ingredient_b else (issue.ingredient or "General")
            lines.append(f"[{sev}] {name}: {issue.message[:200]}")
        return "\n".join(lines)

    def run(self, brief: str) -> AgentResult:
        """
        Run the formulation optimization loop.

        The agent iterates, scoring each attempt and using the feedback
        to improve. Converges when target_score is reached or max_iterations hit.
        """
        result = AgentResult(brief=brief)
        start_time = time.time()
        messages = [{"role": "user", "content": brief}]
        best_score = 0.0
        best_formula = None

        self._log("")
        self._log("=" * 70)
        self._log("  OPENMIX FORMULATION AGENT")
        self._log("=" * 70)
        self._log(f"  Brief: {brief}")
        self._log(f"  Target: {self.target_score:.0f}/100  |  Max iterations: {self.max_iterations}")
        self._log("")

        for i in range(self.max_iterations):
            iter_num = i + 1
            iteration = Iteration(number=iter_num)

            # --- Propose ---
            self._log(f"  [Iter {iter_num}] Proposing formulation...")
            gen_start = time.time()
            raw = self._call_llm(messages)
            iteration.duration_ms = int((time.time() - gen_start) * 1000)

            formula, reasoning = self._parse_formula(raw)
            iteration.formula = formula
            iteration.reasoning = reasoning

            if not formula:
                self._log(f"           Failed to parse. Retrying...")
                result.iterations.append(iteration)
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content":
                    "Your response was not valid JSON. Please respond with ONLY a JSON object."})
                continue

            self._log(f"           {formula.name or 'Untitled'} "
                       f"({len(formula.ingredients)} ingredients)")
            if reasoning:
                r = reasoning[:100] + "..." if len(reasoning) > 100 else reasoning
                self._log(f"           Strategy: {r}")

            # --- Score ---
            stability = score(formula)
            iteration.stability = stability

            # --- Validate ---
            report = validate(formula, mode="formulation")
            iteration.report = report

            self._log(f"           Score: {stability.total:.1f}/100  "
                       f"(compat:{stability.compatibility:.0f} "
                       f"pH:{stability.ph_suitability:.0f} "
                       f"HLB:{stability.emulsion_balance:.0f} "
                       f"integ:{stability.formula_integrity:.0f} "
                       f"complete:{stability.system_completeness:.0f})")

            if stability.penalties:
                for p in stability.penalties[:3]:
                    self._log(f"           - {p}")
            if report.errors > 0 or report.warnings > 0:
                self._log(f"           Validation: {report.errors} errors, {report.warnings} warnings")

            # Track best
            if stability.total > best_score:
                best_score = stability.total
                best_formula = formula

            result.iterations.append(iteration)
            self._log(f"           ({iteration.duration_ms / 1000:.1f}s)")
            self._log("")

            # --- Converged? ---
            if stability.total >= self.target_score and report.errors == 0:
                self._log(f"  Converged at iteration {iter_num} "
                           f"(score {stability.total:.1f} >= {self.target_score:.0f})")
                result.converged = True
                break

            # --- Prepare next iteration ---
            score_breakdown = str(stability)
            issues_text = self._format_issues(report)
            history_text = self._format_history(result.iterations)

            iteration_prompt = ITERATION_PROMPT.format(
                iteration=iter_num,
                score=f"{stability.total:.1f}",
                score_breakdown=score_breakdown,
                issues=issues_text,
                history=history_text,
            )
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": iteration_prompt})

        result.best_formula = best_formula
        result.best_score = best_score
        result.total_duration_ms = int((time.time() - start_time) * 1000)

        if self.verbose:
            print(result)

        return result
