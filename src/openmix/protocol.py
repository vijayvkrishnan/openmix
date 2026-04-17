"""
Manufacturing protocol -- the "how" that makes a formula reproducible.

A formula is a shopping list. A protocol is the instructions. The same
ingredients mixed in different orders, at different temperatures, with
different equipment produce fundamentally different products.

A machine-parseable representation of manufacturing phases (water, oil,
cool-down, active, etc.) for computational evaluation by the process perspective.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Phase:
    """A manufacturing phase -- a group of ingredients processed together."""
    name: str                           # "A", "B", "C", "D"
    label: str                          # "Water Phase", "Oil Phase", "Cool-Down"
    target_temp_c: float | None = None
    ingredients: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        temp = f" @ {self.target_temp_c}C" if self.target_temp_c else ""
        return f"Phase {self.name} ({self.label}{temp}): {', '.join(self.ingredients)}"


@dataclass
class ProcessStep:
    """One step in the manufacturing process."""
    action: str             # heat, cool, combine, add, homogenize, mix, hold, adjust_ph, measure
    phase: str              # which phase this applies to (or "all")
    parameters: dict = field(default_factory=dict)
    notes: str = ""

    def __str__(self) -> str:
        params = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
        phase_str = f" Phase {self.phase}" if self.phase != "all" else ""
        notes_str = f" -- {self.notes}" if self.notes else ""
        return f"{self.action}{phase_str}: {params}{notes_str}"


@dataclass
class Protocol:
    """
    Complete manufacturing protocol for a formulation.

    Specifies phases (ingredient groupings), process steps (what to do
    in what order), equipment requirements, and batch size.

        protocol = Protocol(
            phases=[
                Phase("A", "Water Phase", 75.0, ["Water", "Glycerin"]),
                Phase("B", "Oil Phase", 75.0, ["Squalane", "Cetyl Alcohol"]),
                Phase("C", "Cool-Down", 40.0, ["Retinol", "Phenoxyethanol"]),
            ],
            steps=[
                ProcessStep("heat", "A", {"temp_c": 75, "duration_min": 10}),
                ProcessStep("heat", "B", {"temp_c": 75, "duration_min": 10}),
                ProcessStep("combine", "B", {"into": "A", "rate": "slow"},
                            notes="Add oil phase to water phase while mixing"),
                ProcessStep("homogenize", "all", {"rpm": 5000, "duration_min": 3}),
                ProcessStep("cool", "all", {"target_c": 40, "mixing_rpm": 300}),
                ProcessStep("add", "C", {"mixing_rpm": 300}),
                ProcessStep("adjust_ph", "all", {"target": 5.5, "with": "Citric Acid 10%"}),
                ProcessStep("measure", "all", {"tests": ["pH", "viscosity"]}),
            ],
            equipment=["overhead stirrer", "homogenizer", "pH meter"],
            batch_size_g=100.0,
        )
    """
    phases: list[Phase] = field(default_factory=list)
    steps: list[ProcessStep] = field(default_factory=list)
    equipment: list[str] = field(default_factory=list)
    batch_size_g: float = 100.0

    def get_phase(self, name: str) -> Phase | None:
        """Look up a phase by name."""
        for p in self.phases:
            if p.name == name:
                return p
        return None

    def ingredients_in_phase(self, phase_name: str) -> list[str]:
        """List ingredients assigned to a specific phase."""
        phase = self.get_phase(phase_name)
        return phase.ingredients if phase else []

    def phase_for_ingredient(self, ingredient: str) -> Phase | None:
        """Find which phase an ingredient is assigned to."""
        upper = ingredient.upper()
        for phase in self.phases:
            if any(i.upper() == upper for i in phase.ingredients):
                return phase
        return None

    def __str__(self) -> str:
        lines = [f"Protocol ({self.batch_size_g}g batch):"]
        for phase in self.phases:
            lines.append(f"  {phase}")
        lines.append("")
        for i, step in enumerate(self.steps, 1):
            lines.append(f"  {i}. {step}")
        if self.equipment:
            lines.append(f"\n  Equipment: {', '.join(self.equipment)}")
        return "\n".join(lines)
