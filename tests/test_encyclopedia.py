"""Every encyclopedia entry names a shipped family and builds a page from its pinned tiny checkpoint."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "encyclopedia"))

import build  # noqa: E402
import entries  # noqa: E402
import palette  # noqa: E402
import nnterp.families  # noqa: E402


@pytest.mark.parametrize("name", entries.names())
def test_entry_builds_a_page(name):
    entry = entries.load(name)
    assert name in nnterp.families.known()
    for slug in entry.QUIRKS:
        assert slug in build.QUIRKS, slug
    page = build.build_page(entry, reference=entry.PINNED)
    assert entry.TITLE in page
    for sub in entry.BLOCK["sublayers"]:
        assert f"model.layers[i].{sub['host']}.{sub['contribution']}" in page
        for value in sub.get("interior", []):
            assert f"interior.{sub['host']}.{value}" in page
    assert '"identity"' in page and "stream.output" in page
    for k in range(1, 6):
        assert f"--c{k}: #" in page and f"--c{k}-deep: #" in page
    assert '<span class="k">with</span>' in page, "the notes' snippets are highlighted at build time"
    assert f'<span class="n role-{build.HOST_ROLES[entry.BLOCK["sublayers"][0]["host"]]}">' in page
    assert '<span class="rl">' in page, "the printout's layouts are lexed"


def test_every_quirk_has_a_label():
    for slug, (label, blurb) in build.QUIRKS.items():
        assert label and blurb, slug


def test_generated_palettes_pass_for_every_family():
    """Every hue the hash space reaches yields five readable, distinct colours in both tiers."""
    for name in nnterp.families.known():
        generated = palette.generate(palette.hue_of(name))
        assert palette.check(generated["fills"], generated["deeps"], build.PAPER, build.INK) == [], name


def test_palette_is_deterministic():
    assert palette.generate(palette.hue_of("gemma2")) == palette.generate(palette.hue_of("gemma2"))
    assert palette.hue_of("gemma2") != palette.hue_of("llama")
    assert build.palette(entries.load("gemma2")) == build.palette(entries.load("gemma2"))


def test_palette_colour_maths_round_trips():
    for color in (build.INK, build.PAPER, "#3D9F47"):
        assert palette.oklch_to_hex(*palette.hex_to_oklch(color)) == color
    assert palette.contrast("#000000", "#FFFFFF") == pytest.approx(21)
