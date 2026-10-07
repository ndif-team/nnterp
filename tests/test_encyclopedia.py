"""Every encyclopedia entry names a shipped family and builds a page from its pinned tiny checkpoint; an entry with
vision-language wrappers builds each wrapper's tower from the wrapper's pinned tiny checkpoint."""

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


def checkpoint_data(page, checkpoint=None):
    """The page's data for one checkpoint (the default one unless named): its block schema, node texts, tower."""
    data = embedded(page, "checkpoints-json")
    return data["checkpoints"][checkpoint or data["default"]]


PANE = re.compile(r'<div class="pane" data-pane="(\w+)" data-ckpts="([^"]*)"( hidden)?>')


def panes(page, checkpoint):
    """The HTML of each per-checkpoint part the page shows for ``checkpoint``, by name: each pane runs to the next one."""
    starts = list(PANE.finditer(page))
    out = {}
    for k, m in enumerate(starts):
        if checkpoint in m[2].split():
            end = starts[k + 1].start() if k + 1 < len(starts) else len(page)
            out[m[1]] = page[m.end():end]
    return out


@pytest.mark.parametrize("name", entries.names())
def test_entry_builds_a_page(name):
    entry = entries.load(name)
    assert name in nnterp.families.known()
    for slug in entry.QUIRKS:
        assert slug in build.QUIRKS, slug
    page = build.build_page(entry, reference=pinned(entry))
    assert entry.TITLE in page
    data = checkpoint_data(page)
    schema, nodes = data["schema"], data["nodes"]
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


# -- the checkpoint selector and the vision side ------------------------------------------

def wrapped():
    """Every (entry, wrapper model_type) pair: the entries that describe vision-language wrappers."""
    return [(name, wrapper) for name in entries.names() for wrapper in getattr(entries.load(name), "WRAPPERS", {})]


@pytest.mark.parametrize("name, wrapper", wrapped())
def test_a_wrapper_builds_its_vision_encoder(name, wrapper):
    """The wrapper's pinned tiny checkpoint, selected on its family's page, brings the vision encoder: its module is
    found, its block is drawn with the encoder's names, the vision ledgers carry its values, and its config shows."""
    entry = entries.load(name)
    tiny = entry.WRAPPERS[wrapper]["pinned"]
    page = build.build_page(entry, reference=entry.PINNED, checkpoints=[entry.PINNED, tiny])
    text, vision = checkpoint_data(page), checkpoint_data(page, tiny)
    assert text["tower"] is None and vision["tower"] is not None
    tower = vision["tower"]["schema"]
    assert [sub["host"] for sub in tower["sublayers"]] == ["self_attn", "mlp"]
    assert tower["identity"] == "vision.layers[i].input + self_attn.attention_output + mlp.mlp_output == layer_output"
    nodes = vision["nodes"]
    assert nodes["v:contrib.self_attn"]["expr"] == "model.vision.layers[i].self_attn.attention_output"
    assert nodes["v:contrib.mlp"]["expr"] == "model.vision.layers[i].mlp.mlp_output"
    for sub in tower["sublayers"]:
        for value in sub["interior"]:
            assert f"v:interior.{sub['host']}.{value['name']}" in nodes
    for part, expr in [("patch_embed", "model.vision.patch_embeddings"), ("features", "model.vision.image_features"),
                       ("scatter", "model.vision.image_token_mask"), ("projector", "model.projector")]:
        assert nodes[f"v:path.{part}"]["expr"] == expr, part
    assert "layers[0].input[vision.image_token_mask] == vision.image_features" in nodes["strip.embed"]["extra"]
    shown = panes(page, tiny)
    assert 'class="tower-svg"' in shown["tower"] and 'class="tower-svg"' not in panes(page, entry.PINNED)["tower"]
    for host in ("model.<wbr>vision", "model.<wbr>vision.<wbr>layers[i].<wbr>self_attn", "model.<wbr>vision.<wbr>layers[i].<wbr>mlp"):
        assert host in shown["values"], host
    for value in ("image_token_mask", "patch_embeddings", "tower_output", "image_features"):
        assert f">{value}</span>" in shown["values"], value
    assert "Vision encoder config" in shown["config"]
    for size in build.TOWER_SIZE_NAMES:
        assert f"model.vision.{size}</span>" in shown["config"], size
    assert "The vision encoder · " in shown["tower"]
    assert vision["architecture"].endswith("ForConditionalGeneration") and text["architecture"] == "LlamaForCausalLM"
    assert f'href="https://huggingface.co/{entry.PINNED}"' in page and 'id="ckpt-hub"' in page
    assert f'href="https://huggingface.co/{tiny}"' not in page, "only the selected checkpoint is linked, by the script"
    assert "Vision-language" in shown["chips"] and "<h2>" in shown["vision_notes"]
    assert f'data-ckpt="{tiny}"' in page and 'data-vision="1"' in page


def test_an_unreadable_checkpoint_is_listed_and_skipped():
    """A checkpoint whose config cannot be read is not dropped: the selector greys it out with the reason, it has
    no data, and the page builds from the others."""
    entry = entries.load("llama")
    missing = "nnterp-encyclopedia/no-such-checkpoint"
    page = build.build_page(entry, reference=entry.PINNED, checkpoints=[entry.PINNED, missing])
    assert missing not in embedded(page, "checkpoints-json")["checkpoints"]
    option = re.search(rf'<div class="ckpt-option"[^>]*data-ckpt="{missing}"[^>]*>', page, re.S)[0]
    assert 'aria-disabled="true"' in option and 'title="not available in the encyclopedia: ' in option
    assert checkpoint_data(page)["schema"]["sublayers"]


def test_every_vision_host_in_nnterp_is_a_wrapper():
    """Every wrapper the family's tests run the vision suite on is some WRAPPERS entry's pinned checkpoint, so a new
    host in nnterp shows up here."""
    from vision_suite import VisionSuite

    for name in entries.names():
        entry = entries.load(name)
        if not getattr(entry, "WRAPPERS", None):
            continue
        module = importlib.import_module(f"test_{entry.MODEL_TYPE}")
        suites = {cls.REPO for cls in vars(module).values()
                  if isinstance(cls, type) and issubclass(cls, VisionSuite) and getattr(cls, "REPO", None)}
        pinned_wrappers = {wrapper["pinned"] for wrapper in entry.WRAPPERS.values()}
        assert suites and suites <= pinned_wrappers, (name, sorted(suites - pinned_wrappers))


def test_vision_encoder_modules_are_complete():
    """Every vision encoder module has every field, and no two cover the same vision_config.model_type."""
    import vision

    seen = {}
    for module in vision.modules():
        for field in vision.FIELDS:
            assert hasattr(module, field), (module.__name__, field)
        for slug in module.QUIRKS:
            assert slug in build.QUIRKS, (module.__name__, slug)
        for kind in module.VISION_CONFIG_TYPES:
            assert kind not in seen, (kind, seen.get(kind), module.__name__)
            seen[kind] = module.__name__
