"""Five colours per family, generated in OKLCH from one hue.

A family page colours five roles: attention, the MLP, the norms, the residual
stream and the family's own mark. Each role gets one hue, in two lightness
tiers: a ``fill`` that takes ink text (chips, boxes, petals, hover fills) and
a ``deep`` that sits on the paper as a line or coloured text. The hue comes
from the entry (``PALETTE["hue"]``) or from a hash of the ``model_type``, and
the other four hues follow by a fixed harmony, so the page reads the same way
on every family and related families, given neighbouring hues, read alike.

The colour maths is Ottosson's OKLab: OKLCH -> OKLab -> linear sRGB -> sRGB,
with a colour outside the sRGB gamut pulled back by reducing its chroma until
it fits. ``check`` is what the build and the tests run on the result.
"""

from __future__ import annotations

import hashlib
import math

#: Hue offsets from the base hue, in degrees, for roles 1..5: attention and the
#: MLP sit together in the diagram and take the two hues farthest apart; the
#: family's mark is the base hue itself; the norms and the stream fill the arc
#: between the MLP and attention on the far side of the mark.
HARMONY = (90, 270, 150, 210, 0)
#: Chroma and lightness of the two tiers. Muted, like a print palette: the fill
#: tier carries ink text, the deep tier is read against the paper.
CHROMA = 0.10
FILL_L = 0.74
DEEP_L = 0.50
#: What a palette must satisfy, as ``check`` measures it.
MIN_FILL_CONTRAST = 4.5     # WCAG, ink on every fill
MIN_DEEP_CONTRAST = 4.0     # WCAG, every deep on the paper
MIN_DISTANCE = 0.06         # OKLab distance between any two fills, and any two deeps
MIN_SEPARATION = 35         # degrees between any two hues

ROLES = ("attention", "mlp", "norm", "stream", "mark")

Rgb = tuple[float, float, float]


# -- colour maths -------------------------------------------------------------------

def oklab_to_linear(L: float, a: float, b: float) -> Rgb:
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
    return (
        +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
    )


def linear_to_oklab(r: float, g: float, b: float) -> tuple[float, float, float]:
    l = math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m = math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s = math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    return (
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
    )


def gamma(x: float) -> float:
    return 12.92 * x if x <= 0.0031308 else 1.055 * x ** (1 / 2.4) - 0.055


def degamma(x: float) -> float:
    return x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4


def oklch_to_linear(L: float, C: float, h: float) -> Rgb:
    rad = math.radians(h)
    return oklab_to_linear(L, C * math.cos(rad), C * math.sin(rad))


def in_gamut(rgb: Rgb) -> bool:
    return all(-1e-4 <= x <= 1 + 1e-4 for x in rgb)


def oklch_to_hex(L: float, C: float, h: float) -> str:
    """The sRGB hex of an OKLCH colour; chroma is reduced until the colour is in gamut."""
    lo, hi = 0.0, C
    rgb = oklch_to_linear(L, C, h)
    if not in_gamut(rgb):
        for _ in range(24):
            mid = (lo + hi) / 2
            if in_gamut(oklch_to_linear(L, mid, h)):
                lo = mid
            else:
                hi = mid
        rgb = oklch_to_linear(L, lo, h)
    return "#%02X%02X%02X" % tuple(round(255 * min(1, max(0, gamma(x)))) for x in rgb)


def hex_to_linear(color: str) -> Rgb:
    color = color.lstrip("#")
    return tuple(degamma(int(color[i:i + 2], 16) / 255) for i in (0, 2, 4))  # type: ignore[return-value]


def hex_to_oklch(color: str) -> tuple[float, float, float]:
    L, a, b = linear_to_oklab(*hex_to_linear(color))
    return L, math.hypot(a, b), math.degrees(math.atan2(b, a)) % 360


def luminance(color: str) -> float:
    r, g, b = hex_to_linear(color)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(one: str, two: str) -> float:
    """WCAG contrast ratio between two colours."""
    a, b = sorted((luminance(one), luminance(two)))
    return (b + 0.05) / (a + 0.05)


def distance(one: str, two: str) -> float:
    """OKLab distance (a ΔE) between two colours."""
    p, q = linear_to_oklab(*hex_to_linear(one)), linear_to_oklab(*hex_to_linear(two))
    return math.dist(p, q)


# -- the palette ----------------------------------------------------------------------

def hue_of(model_type: str) -> int:
    """The base hue a family gets when its entry does not name one."""
    return int(hashlib.sha1(model_type.encode()).hexdigest(), 16) % 360


def hues(base: float) -> list[float]:
    return [(base + offset) % 360 for offset in HARMONY]


def generate(base: float) -> dict[str, list[str]]:
    """The five fills and five deeps of a base hue."""
    return {
        "fills": [oklch_to_hex(FILL_L, CHROMA, h) for h in hues(base)],
        "deeps": [oklch_to_hex(DEEP_L, CHROMA, h) for h in hues(base)],
    }


def deepen(fills: list[str]) -> list[str]:
    """The deep tier of five given fills: the same hue and chroma at the deep lightness."""
    return [oklch_to_hex(DEEP_L, C, h) for _, C, h in map(hex_to_oklch, fills)]


#: The petals of the pages' seal (``.petals`` in encyclopedia.css), as each disc's left and top edge in
#: units of the seal's width; a disc is 0.46 of the width across.
PETALS = ((0, 0.10), (0.34, 0), (0.54, 0.30), (0.20, 0.46), (0.50, 0.56))
PETAL = 0.46


def favicon(fills: list[str]) -> str:
    """The petals as a 32x32 SVG: the five fills as overlapping discs, in the seal's order, on no background."""
    right = max(left for left, _ in PETALS) + PETAL
    bottom = max(top for _, top in PETALS) + PETAL
    scale = 32 / max(right, bottom)
    dx, dy = (32 - right * scale) / 2, (32 - bottom * scale) / 2
    r = PETAL * scale / 2
    discs = "".join(
        f"<circle cx='{dx + (left + PETAL / 2) * scale:.1f}' cy='{dy + (top + PETAL / 2) * scale:.1f}' r='{r:.1f}' fill='{fill}'/>"
        for (left, top), fill in zip(PETALS, fills)
    )
    return f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'>{discs}</svg>"


def check(fills: list[str], deeps: list[str], paper: str, ink: str) -> list[str]:
    """Every way the palette falls short, as messages; an empty list is a pass."""
    problems = []
    for role, fill in zip(ROLES, fills):
        if (c := contrast(ink, fill)) < MIN_FILL_CONTRAST:
            problems.append(f"ink on the {role} fill {fill}: contrast {c:.2f} < {MIN_FILL_CONTRAST}")
    for role, deep in zip(ROLES, deeps):
        if (c := contrast(deep, paper)) < MIN_DEEP_CONTRAST:
            problems.append(f"{role} deep {deep} on the paper: contrast {c:.2f} < {MIN_DEEP_CONTRAST}")
    for tier, colors in (("fills", fills), ("deeps", deeps)):
        for i in range(5):
            for j in range(i + 1, 5):
                if (d := distance(colors[i], colors[j])) < MIN_DISTANCE:
                    problems.append(f"{tier} {ROLES[i]} {colors[i]} and {ROLES[j]} {colors[j]}: distance {d:.3f} < {MIN_DISTANCE}")
                hi, hj = hex_to_oklch(colors[i])[2], hex_to_oklch(colors[j])[2]
                apart = abs((hi - hj + 180) % 360 - 180)
                if apart < MIN_SEPARATION:
                    problems.append(f"{tier} {ROLES[i]} and {ROLES[j]}: hues {apart:.0f}° apart < {MIN_SEPARATION}")
    return problems
