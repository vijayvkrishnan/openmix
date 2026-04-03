#!/usr/bin/env python3
"""
Programmatic demo GIF generator.

Renders terminal-style frames with Pillow and combines into an
animated GIF. No screen recording needed.

    python scripts/generate_demo_gif.py
    # Output: assets/demo.gif
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WIDTH = 820
LINE_HEIGHT = 18
FONT_SIZE = 13
PADDING_X = 16
PADDING_Y = 12
BG_COLOR = (22, 22, 30)
BORDER_COLOR = (50, 50, 70)

# Colors (terminal palette)
C_WHITE = (220, 220, 220)
C_DIM = (120, 120, 140)
C_GREEN = (80, 220, 120)
C_YELLOW = (240, 200, 60)
C_RED = (240, 80, 80)
C_BLUE = (100, 160, 255)
C_CYAN = (80, 220, 220)
C_BRIGHT_YELLOW = (255, 220, 80)
C_BRIGHT_GREEN = (100, 255, 140)
C_PANEL_BG = (30, 30, 42)
C_PANEL_BORDER_BLUE = (80, 140, 240)
C_PANEL_BORDER_YELLOW = (220, 180, 40)
C_PANEL_BORDER_GREEN = (60, 200, 100)

FONT_PATH = "C:/Windows/Fonts/consola.ttf"

# Frame durations in ms
PAUSE_SHORT = 600
PAUSE_MED = 1200
PAUSE_LONG = 2500
PAUSE_XLONG = 3500


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def get_font():
    if os.path.exists(FONT_PATH):
        return ImageFont.truetype(FONT_PATH, FONT_SIZE)
    return ImageFont.load_default()


def render_frame(lines: list[tuple[str, tuple]], height: int | None = None) -> Image.Image:
    """Render colored text lines to an image.

    lines: list of (text, color) tuples.
    """
    if height is None:
        height = PADDING_Y * 2 + len(lines) * LINE_HEIGHT + 8
    height = max(height, 100)

    img = Image.new("RGB", (WIDTH, height), BG_COLOR)
    draw = ImageDraw.Draw(img)
    font = get_font()

    y = PADDING_Y
    for text, color in lines:
        draw.text((PADDING_X, y), text, fill=color, font=font)
        y += LINE_HEIGHT

    return img


def draw_panel(draw, font, x, y, w, lines, border_color, title=None):
    """Draw a bordered panel with text."""
    h = len(lines) * LINE_HEIGHT + 30
    # Border
    draw.rectangle([x, y, x + w, y + h], outline=border_color, width=1)
    # Fill
    draw.rectangle([x + 1, y + 1, x + w - 1, y + h - 1], fill=C_PANEL_BG)
    # Title
    if title:
        tw = font.getlength(title) if hasattr(font, 'getlength') else len(title) * 8
        tx = x + (w - tw) // 2
        draw.rectangle([tx - 4, y - 2, tx + tw + 4, y + LINE_HEIGHT - 2], fill=BG_COLOR)
        draw.text((tx, y), title, fill=border_color, font=font)
    # Content
    cy = y + 15
    for text, color in lines:
        draw.text((x + 12, cy), text, fill=color, font=font)
        cy += LINE_HEIGHT
    return h


# ---------------------------------------------------------------------------
# Frame sequences
# ---------------------------------------------------------------------------

def build_frames():
    font = get_font()
    frames = []
    durations = []

    def add(lines, duration=PAUSE_MED, height=None):
        frames.append(render_frame(lines, height))
        durations.append(duration)

    H = 520  # consistent frame height

    # Frame 1: Title
    add([
        ("", C_WHITE),
        ("  OpenMix v0.3.1", C_WHITE),
        ("", C_WHITE),
        ("  RDKit is for molecules. OpenMix is for mixtures.", C_CYAN),
        ("", C_WHITE),
        ("  Multi-perspective formulation evaluation with", C_DIM),
        ("  evidence-based disagreement classification.", C_DIM),
    ], PAUSE_LONG, H)

    # Frame 2: Formula
    add([
        ("", C_WHITE),
        ("  Pharma: Novel Drug Evaluation", C_YELLOW),
        ("  " + "-" * 50, C_DIM),
        ("", C_WHITE),
        ("  Memantine 10mg Tablet", C_WHITE),
        ("", C_WHITE),
        ("  Ingredient                      %      Role", C_DIM),
        ("  " + "-" * 50, C_DIM),
        ("  Memantine                      5.0    active (novel)", C_BRIGHT_YELLOW),
        ("  Lactose Monohydrate           55.0    filler", C_WHITE),
        ("  Microcrystalline Cellulose    28.0    binder", C_WHITE),
        ("  Magnesium Stearate             1.0    lubricant", C_WHITE),
        ("  Povidone                       3.0    binder", C_WHITE),
        ("  Water                          8.0    granulation", C_DIM),
        ("", C_WHITE),
        ("  Memantine is NOT in the 273-rule knowledge base.", C_BRIGHT_YELLOW),
    ], PAUSE_LONG, H)

    # Frame 3: Analyzing
    add([
        ("", C_WHITE),
        ("  Pharma: Novel Drug Evaluation", C_YELLOW),
        ("  " + "-" * 50, C_DIM),
        ("", C_WHITE),
        ("  Memantine 10mg Tablet", C_WHITE),
        ("", C_WHITE),
        ("  Running 4-perspective discourse...", C_CYAN),
        ("", C_WHITE),
        ("    Physics:    resolving SMILES from PubChem...", C_DIM),
        ("    Chemistry:  checking 273 interaction rules...", C_DIM),
        ("    Data:       querying experiment memory...", C_DIM),
        ("    Process:    evaluating manufacturing protocol...", C_DIM),
    ], PAUSE_MED, H)

    # Frame 4: Key finding
    add([
        ("", C_WHITE),
        ("  DISCOURSE RESULTS", C_GREEN),
        ("  " + "=" * 50, C_DIM),
        ("", C_WHITE),
        ("  [=] Lactose Monohydrate: safe", C_GREEN),
        ("      Good water solubility expected.", C_DIM),
        ("", C_WHITE),
        ("  +---------------------------------------------------+", C_PANEL_BORDER_YELLOW),
        ("  | MECHANISM-BASED PREDICTION                        |", C_BRIGHT_YELLOW),
        ("  |                                                   |", C_PANEL_BORDER_YELLOW),
        ("  | Primary amine detected from SMILES (via PubChem)  |", C_WHITE),
        ("  | + Lactose classified as reducing sugar             |", C_WHITE),
        ("  | = Maillard reaction risk                          |", C_BRIGHT_YELLOW),
        ("  |                                                   |", C_PANEL_BORDER_YELLOW),
        ("  | Subject: Memantine + LACTOSE MONOHYDRATE          |", C_DIM),
        ("  | Confidence: 0.9                                   |", C_DIM),
        ("  +---------------------------------------------------+", C_PANEL_BORDER_YELLOW),
        ("", C_WHITE),
        ("  [=] MgSt + Lactose: catalyzes Maillard reaction", C_YELLOW),
        ("      Mg2+ accelerates Amadori rearrangement.", C_DIM),
    ], PAUSE_XLONG, H)

    # Frame 5: Explanation
    add([
        ("", C_WHITE),
        ("  HOW IT WORKS", C_BRIGHT_GREEN),
        ("  " + "=" * 50, C_DIM),
        ("", C_WHITE),
        ("  1. Resolver fetched Memantine's SMILES from PubChem", C_WHITE),
        ("     CC12CC3CC(C1)(CC(C3)(C2)N)C", C_CYAN),
        ("", C_WHITE),
        ("  2. RDKit detected primary amine (-NH2) from SMARTS", C_WHITE),
        ("     pattern [NX3H2][CX4]", C_CYAN),
        ("", C_WHITE),
        ("  3. Lactose Monohydrate classified as reducing sugar", C_WHITE),
        ("     from excipient property database", C_CYAN),
        ("", C_WHITE),
        ("  4. Mechanism matched: primary_amine + reducing_sugar", C_WHITE),
        ("     = Maillard reaction (Schiff base degradation)", C_BRIGHT_YELLOW),
        ("", C_WHITE),
        ("  No specific rule for Memantine exists.", C_BRIGHT_YELLOW),
        ("  Generalizable to ANY drug with a primary amine.", C_BRIGHT_YELLOW),
    ], PAUSE_XLONG, H)

    # Frame 6: Footer
    add([
        ("", C_WHITE),
        ("  OpenMix", C_WHITE),
        ("  " + "-" * 50, C_DIM),
        ("", C_WHITE),
        ("  273 rules         6 domains", C_WHITE),
        ("  4 perspectives    evidence hierarchy", C_WHITE),
        ("  95 pharma rules   mechanism-based prediction", C_WHITE),
        ("", C_WHITE),
        ("  pip install openmix", C_CYAN),
        ("  github.com/vijayvkrishnan/openmix", C_BLUE),
    ], PAUSE_LONG, H)

    return frames, durations


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    output_path = Path(__file__).parent.parent / "assets" / "demo.gif"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Generating demo GIF frames...")
    frames, durations = build_frames()
    print(f"  {len(frames)} frames")

    # Save as animated GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
    )

    size_kb = output_path.stat().st_size / 1024
    print(f"  Saved to {output_path} ({size_kb:.0f} KB)")
    print("  Done.")


if __name__ == "__main__":
    main()
