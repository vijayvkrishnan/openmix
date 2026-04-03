#!/usr/bin/env python3
"""
OpenMix branding v2 -- cleaner, more abstract.

Instead of a crude beaker, uses overlapping translucent circles
representing multiple perspectives/ingredients mixing. The intersection
is where the interesting chemistry (and discourse) happens.
"""

from pathlib import Path
import math
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ASSETS = Path(__file__).parent.parent / "assets"

# Colors
BG = (13, 15, 21)
WHITE = (235, 237, 245)
DIM = (110, 118, 138)
SUBTLE = (35, 38, 50)

# Perspective colors (matching the discourse engine's four perspectives)
PHYSICS = (70, 130, 255)      # blue
CHEMISTRY = (60, 210, 180)    # teal
DATA = (160, 120, 255)        # purple
PROCESS = (255, 180, 60)      # amber

ACCENT = (70, 160, 255)       # primary accent

# Fonts
FONT_BOLD = "C:/Windows/Fonts/segoeuib.ttf"
FONT_LIGHT = "C:/Windows/Fonts/segoeuil.ttf"
FONT_REGULAR = "C:/Windows/Fonts/segoeui.ttf"
FONT_MONO = "C:/Windows/Fonts/consola.ttf"


def draw_glow_circle(img, cx, cy, radius, color, alpha=50):
    """Draw a soft glowing circle using a separate RGBA layer."""
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    # Multiple concentric circles with decreasing alpha for glow
    for i in range(5):
        r = radius + i * 8
        a = max(10, alpha - i * 10)
        d.ellipse(
            [cx - r, cy - r, cx + r, cy + r],
            fill=(*color, a),
        )

    # Core circle
    d.ellipse(
        [cx - radius, cy - radius, cx + radius, cy + radius],
        fill=(*color, alpha),
    )

    return Image.alpha_composite(img.convert("RGBA"), overlay)


def draw_mixing_symbol(img, cx, cy, size):
    """Draw overlapping circles representing mixing perspectives."""
    radius = size * 0.32
    offset = size * 0.22

    positions = [
        (cx - offset, cy - offset * 0.6, PHYSICS, 45),
        (cx + offset, cy - offset * 0.6, CHEMISTRY, 40),
        (cx, cy + offset * 0.7, DATA, 38),
    ]

    for x, y, color, alpha in positions:
        img = draw_glow_circle(img, x, y, radius, color, alpha)

    # Small bright dot at the intersection (where discourse happens)
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    d.ellipse([cx - 6, cy - 4, cx + 6, cy + 8], fill=(255, 255, 255, 140))
    img = Image.alpha_composite(img, overlay)

    return img


def generate_social_preview():
    """1280x640 Open Graph image."""
    img = Image.new("RGBA", (1280, 640), BG)
    draw = ImageDraw.Draw(img)

    # Subtle dot grid
    for x in range(30, 1280, 50):
        for y in range(30, 640, 50):
            draw.ellipse([x, y, x + 1, y + 1], fill=SUBTLE)

    # Mixing symbol on the left
    img = draw_mixing_symbol(img, 240, 310, 320)
    draw = ImageDraw.Draw(img)

    # Right side: text
    title_font = ImageFont.truetype(FONT_BOLD, 68)
    tagline_font = ImageFont.truetype(FONT_LIGHT, 27)
    stat_font = ImageFont.truetype(FONT_REGULAR, 21)
    mono_font = ImageFont.truetype(FONT_MONO, 19)

    # Title with accent
    draw.text((480, 145), "Open", fill=WHITE, font=title_font)
    # Measure "Open" width
    open_w = draw.textlength("Open", font=title_font)
    draw.text((480 + open_w, 145), "Mix", fill=ACCENT, font=title_font)

    # Accent line under title
    draw.line([(480, 230), (1060, 230)], fill=(*ACCENT, 120), width=1)

    # Tagline
    draw.text((480, 252), "RDKit is for molecules.", fill=(*CHEMISTRY, 220), font=tagline_font)
    draw.text((480, 288), "OpenMix is for mixtures.", fill=(*CHEMISTRY, 220), font=tagline_font)

    # Stats with colored dots
    stats = [
        (PHYSICS, "Curated interaction knowledge base, 6 domains"),
        (CHEMISTRY, "Multi-perspective discourse with evidence hierarchy"),
        (DATA, "Mechanism-based prediction from molecular structure"),
        (PROCESS, "Drug delivery, skincare, food, materials"),
    ]
    y = 360
    for color, text in stats:
        draw.ellipse([480, y + 5, 490, y + 15], fill=color)
        draw.text((502, y), text, fill=DIM, font=stat_font)
        y += 34

    # Install
    draw.text((480, 540), "pip install openmix", fill=(80, 220, 140), font=mono_font)
    url_font = ImageFont.truetype(FONT_REGULAR, 17)
    draw.text((780, 542), "github.com/vijayvkrishnan/openmix", fill=DIM, font=url_font)

    img = img.convert("RGB")
    img.save(ASSETS / "social-preview.png", quality=95)
    print(f"Social preview: {ASSETS / 'social-preview.png'}")


def generate_logo():
    """512x512 logo."""
    size = 512
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))

    # Dark circle background
    draw = ImageDraw.Draw(img)
    draw.ellipse([16, 16, size - 16, size - 16], fill=BG)

    # Mixing circles
    img = draw_mixing_symbol(img, size // 2, size // 2 - 10, size * 0.55)
    draw = ImageDraw.Draw(img)

    # "OpenMix" text at bottom
    font = ImageFont.truetype(FONT_BOLD, 42)
    text_w = draw.textlength("OpenMix", font=font)
    tx = (size - text_w) / 2
    draw.text((tx, size - 110), "Open", fill=WHITE, font=font)
    open_w = draw.textlength("Open", font=font)
    draw.text((tx + open_w, size - 110), "Mix", fill=ACCENT, font=font)

    img.save(ASSETS / "logo.png")
    print(f"Logo: {ASSETS / 'logo.png'}")

    # Favicon
    img_small = img.resize((64, 64), Image.LANCZOS)
    img_small.save(ASSETS / "favicon.ico", format="ICO")


def generate_banner():
    """1200x300 banner."""
    img = Image.new("RGBA", (1200, 300), BG)

    # Mixing symbol on left
    img = draw_mixing_symbol(img, 130, 150, 180)
    draw = ImageDraw.Draw(img)

    title_font = ImageFont.truetype(FONT_BOLD, 58)
    sub_font = ImageFont.truetype(FONT_LIGHT, 23)

    draw.text((290, 85), "Open", fill=WHITE, font=title_font)
    open_w = draw.textlength("Open", font=title_font)
    draw.text((290 + open_w, 85), "Mix", fill=ACCENT, font=title_font)

    draw.text((290, 162), "The open-source framework for computational", fill=DIM, font=sub_font)
    draw.text((290, 194), "formulation science", fill=CHEMISTRY, font=sub_font)

    img = img.convert("RGB")
    img.save(ASSETS / "banner.png", quality=95)
    print(f"Banner: {ASSETS / 'banner.png'}")


if __name__ == "__main__":
    generate_social_preview()
    generate_logo()
    generate_banner()
    print("Done.")
