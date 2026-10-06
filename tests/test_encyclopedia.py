"""Every encyclopedia entry names a shipped family and builds a page from its pinned tiny checkpoint."""

import importlib
import json
import os
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "encyclopedia"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "families"))

import build  # noqa: E402
import entries  # noqa: E402
import palette  # noqa: E402
import nnterp.families  # noqa: E402


def pinned(entry):
    """The checkpoint the family's own suite runs on: its ``REPO``, which is ``PINNED`` or, where the
    suite rewrites the tiny checkpoint's config, a local copy of it."""
    module = importlib.import_module(f"test_{entry.MODEL_TYPE}")
    repo = next(cls.REPO for cls in vars(module).values() if isinstance(cls, type) and "REPO" in vars(cls))
    assert repo == entry.PINNED or os.path.isdir(repo), (repo, entry.PINNED)
    return repo


def embedded(page, name):
    return json.loads(re.search(rf'id="{name}">(.*?)</script>', page, re.S)[1].replace("<\\/", "</"))


@pytest.mark.parametrize("name", entries.names())
def test_entry_builds_a_page(name):
    entry = entries.load(name)
    assert name in nnterp.families.known()
    for slug in entry.QUIRKS:
        assert slug in build.QUIRKS, slug
    page = build.build_page(entry, reference=pinned(entry))
    assert entry.TITLE in page
    schema, nodes = embedded(page, "block-schema"), embedded(page, "nodes-json")
    # The schema holds the sublayers this checkpoint's blocks draw: every one the entry lists, but
    # where a host runs a mixture on some checkpoints only (Gemma-4), the one of the pair it has.
    drawn = schema["sublayers"]
    listed = {(sub["host"], sub["kind"]) for sub in entry.BLOCK["sublayers"]}
    assert {(sub["host"], sub["kind"]) for sub in drawn} <= listed
    assert {sub["host"] for sub in drawn} == {host for host, _ in listed}
    for sub in drawn:
        key = sub.get("key", sub["host"])
        assert nodes[f"contrib.{key}"]["expr"] == f"model.layers[i].{sub['host']}.{sub['contribution']}"
        for value in sub["interior"]:
            assert f"interior.{key}.{value['name']}" in nodes
        if sub["kind"] == "moe":
            assert {f"moe.{key}.router", f"moe.{key}.experts"} <= set(nodes)
    # A family with several block shapes is checked per shape: every block has one, every
    # sublayer is drawn on some block, and each shape's identity sums the contributions it draws.
    shapes = schema.get("shapes", [{"subs": list(range(len(drawn))), "identity": schema["identity"]}])
    if "shapes" in schema:
        assert len(schema["shape_of"]) == schema["num_layers"] and len(shapes) > 1
    assert sorted({k for shape in shapes for k in shape["subs"]}) == list(range(len(drawn)))
    for shape in shapes:
        for k in shape["subs"]:
            sub = drawn[k]
            assert "identity" in entry.BLOCK or f"{sub['host']}.{sub['contribution']}" in shape["identity"]
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


def test_quirk_slugs_are_unique():
    """A dict literal with a repeated key keeps the last one silently; the source must not repeat a slug."""
    import re
    source = (Path(build.__file__)).read_text()
    body = source[source.index("QUIRKS: dict"):source.index("\n}\n", source.index("QUIRKS: dict"))]
    slugs = re.findall(r'^\s+"([a-z0-9-]+)": \(', body, re.M)
    assert len(slugs) == len(set(slugs)), [s for s in slugs if slugs.count(s) > 1]
