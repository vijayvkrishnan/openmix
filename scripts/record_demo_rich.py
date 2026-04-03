#!/usr/bin/env python3
"""
Rich terminal demo for GIF recording.

Single narrative: a novel drug (Memantine, NOT in the knowledge base)
is evaluated with its excipients. The discourse engine detects the
primary amine from SMILES and predicts Maillard reaction risk with
lactose -- a drug it's never seen before.

Record with:
    asciinema rec demo.cast -c "python scripts/record_demo_rich.py"
    # or screen-record with OBS/ShareX/ScreenToGif

Recommended terminal: 100+ columns, dark background.
"""

import time
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box
from rich.columns import Columns
from rich.rule import Rule

console = Console(width=90)


def pause(seconds=0.8):
    time.sleep(seconds)


def main():
    console.clear()
    pause(0.5)

    # --- Title ---
    console.print()
    title = Text("OpenMix", style="bold bright_white")
    title.append("  v0.3.1", style="dim")
    console.print(Panel(
        "[bold]RDKit is for molecules. OpenMix is for mixtures.[/bold]\n\n"
        "[dim]Multi-perspective formulation evaluation with\n"
        "evidence-based disagreement classification.[/dim]",
        title=title,
        border_style="bright_blue",
        padding=(1, 2),
    ))
    pause(1.5)

    # --- Formula ---
    console.print()
    console.print(Rule("[bold yellow]Pharma: Novel Drug Evaluation[/bold yellow]"))
    console.print()

    formula_table = Table(
        title="[bold]Memantine 10mg Tablet[/bold]",
        box=box.ROUNDED,
        border_style="cyan",
        show_header=True,
        header_style="bold",
    )
    formula_table.add_column("Ingredient", style="white", width=30)
    formula_table.add_column("%", justify="right", width=8)
    formula_table.add_column("Role", style="dim", width=20)

    ingredients = [
        ("Memantine", "5.0", "[bright_yellow]active (novel drug)[/]"),
        ("Lactose Monohydrate", "55.0", "filler"),
        ("Microcrystalline Cellulose", "28.0", "binder"),
        ("Magnesium Stearate", "1.0", "lubricant"),
        ("Povidone", "3.0", "binder"),
        ("Water", "8.0", "granulation"),
    ]
    for name, pct, role in ingredients:
        formula_table.add_row(name, pct, role)

    console.print(formula_table)
    pause(1)

    console.print()
    console.print("[dim italic]  Memantine is NOT in the 273-rule knowledge base.[/]")
    console.print("[dim italic]  Can the physics perspective predict the interaction?[/]")
    pause(1.5)

    # --- Analyzing ---
    # Pre-cache the PubChem lookup so the spinner is fast
    from openmix.resolver.resolve import _session_cache
    _session_cache.clear()
    from openmix.resolver.pubchem import lookup_pubchem
    lookup_pubchem("Memantine")  # warm cache before visual

    from openmix import Formula
    from openmix.discourse import evaluate_discourse

    tablet = Formula(
        name="Memantine 10mg Tablet",
        ingredients=[
            ("Memantine", 5.0),
            ("Lactose Monohydrate", 55.0),
            ("Microcrystalline Cellulose", 28.0),
            ("Magnesium Stearate", 1.0),
            ("Povidone", 3.0),
            ("Water", 8.0),
        ],
        category="pharma",
    )

    console.print()
    with console.status("[bold cyan]Running 4-perspective discourse...[/]", spinner="dots"):
        disc = evaluate_discourse(tablet)
        time.sleep(1.5)  # brief spinner for visual

    # --- Results ---
    console.print()
    console.print(Rule("[bold green]Discourse Results[/bold green]"))
    console.print()

    # Agreements
    for topic in disc.agreements:
        for claim in topic.claims:
            if "mechanism-based prediction" in claim.evidence.lower():
                # This is THE key finding -- mechanism-based, not from rules
                console.print(Panel(
                    f"[bold bright_yellow]MECHANISM-BASED PREDICTION[/]\n\n"
                    f"[white]{claim.detail}[/]\n\n"
                    f"[dim]Subject: {topic.subject}[/]\n"
                    f"[dim]Source: Functional group detected from SMILES + "
                    f"excipient classified as reducing sugar[/]\n"
                    f"[dim]Confidence: {claim.confidence}[/]",
                    border_style="bright_yellow",
                    title="[bold bright_yellow]Primary Amine Detected from Molecular Structure[/]",
                    padding=(1, 2),
                ))
                pause(2)
            elif claim.perspective == "chemistry" and "maillard" in claim.detail.lower():
                console.print(
                    f"  [yellow][=][/] [bold]{topic.subject}[/]\n"
                    f"      [dim]{claim.detail[:90]}[/]"
                )
                pause(0.5)
            elif claim.position == "safe":
                console.print(
                    f"  [green][=][/] [bold]{topic.subject}[/]: {claim.position}\n"
                    f"      [dim]{claim.detail[:70]}[/]"
                )
                pause(0.3)

    # True disagreements
    for topic in disc.true_disagreements:
        console.print(
            f"\n  [bright_blue][?][/] [bold]{topic.subject}[/]: "
            f"[bright_blue]true disagreement[/]\n"
            f"      [dim]{topic.summary[:80]}[/]"
        )
        pause(1)

    console.print()
    pause(1)

    # --- The punchline ---
    console.print(Panel(
        "[bold white]The physics perspective detected Memantine's primary amine\n"
        "from its SMILES string (via PubChem), matched it against Lactose\n"
        "(a reducing sugar), and predicted Maillard reaction risk.[/]\n\n"
        "[bright_yellow]No specific rule for Memantine exists in the knowledge base.\n"
        "This is a generalizable, mechanism-based prediction.[/]",
        border_style="bright_green",
        title="[bold bright_green]How it works[/]",
        padding=(1, 2),
    ))
    pause(2)

    # --- Footer ---
    console.print()

    stats = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    stats.add_column(style="bold")
    stats.add_column(style="dim")
    stats.add_row("273 rules", "6 domains")
    stats.add_row("4 perspectives", "evidence hierarchy")
    stats.add_row("95 pharma rules", "mechanism-based prediction")
    stats.add_row("pip install openmix", "github.com/vijayvkrishnan/openmix")

    console.print(Panel(stats, border_style="bright_blue", title="[bold]OpenMix[/]"))
    console.print()
    pause(3)  # hold on the final frame before terminal prompt appears


if __name__ == "__main__":
    main()
