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
    # the module's own suites: another family's suite it imports (falcon_mamba imports TestMamba) is not its pin
    repo = next(cls.REPO for cls in vars(module).values()
                if isinstance(cls, type) and cls.__module__ == module.__name__ and "REPO" in vars(cls))
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


def block_of(entry):
    """The entry's BLOCK for its pinned checkpoint: a BLOCK that varies by checkpoint is resolved on that build's config."""
    if isinstance(entry.BLOCK, dict):
        return entry.BLOCK
    info = build.read_entry(entry, reference=pinned(entry))[0][0]["info"]
    return build.resolve_block(entry.MODEL_TYPE, entry.BLOCK, info["text_config"])


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
    block = block_of(entry)
    # The schema holds the sublayers this checkpoint's blocks draw: every one the entry lists, but
    # where a host runs a mixture on some checkpoints only (Gemma-4), the one of the pair it has.
    drawn = schema["sublayers"]
    listed = {(sub["host"], sub["kind"]) for sub in block["sublayers"]}
    assert {(sub["host"], sub["kind"]) for sub in drawn} <= listed
    assert {sub["host"] for sub in drawn} == {host for host, _ in listed}
    for sub in drawn:
        key = sub.get("key", sub["host"])
        # a sublayer on a native path (OPT's fc2) names its contribution from the block: `fc2.output`
        term = sub["contribution"] if sub.get("native") else f"{sub['host']}.{sub['contribution']}"
        assert nodes[f"contrib.{key}"]["expr"] == f"model.layers[i].{term}"
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
            term = sub["contribution"] if sub.get("native") else f"{sub['host']}.{sub['contribution']}"
            assert "identity" in block or term in shape["identity"]
    assert '"identity"' in page and "stream.output" in page
    for k in range(1, 6):
        assert f"--c{k}: #" in page and f"--c{k}-deep: #" in page
    assert '<span class="k">with</span>' in page, "the notes' snippets are highlighted at build time"
    assert f'<span class="n role-{build.HOST_ROLES[block["sublayers"][0]["host"]]}">' in page
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
    # Where the suite rewrites the pinned checkpoint into a local copy (Llama 4), the page is built from the copy.
    text_pin = pinned(entry) if os.path.isdir(pinned(entry)) else entry.PINNED
    tiny = entry.WRAPPERS[wrapper]["pinned"]
    tiny = text_pin if tiny == entry.PINNED else tiny
    page = build.build_page(entry, reference=text_pin, checkpoints=[text_pin, tiny])
    text, vision = checkpoint_data(page), checkpoint_data(page, tiny)
    # A family whose every checkpoint is the wrapper (Qwen3.5) pins the same checkpoint for both: no text-only one to compare.
    text_side = tiny != text_pin
    assert (text["tower"] is None or not text_side) and vision["tower"] is not None
    nodes = vision["nodes"]
    shown = panes(page, tiny)
    for part, expr in [("patch_embed", "model.vision.patch_embeddings"), ("features", "model.vision.image_features"),
                       ("scatter", "model.vision.image_token_mask"), ("projector", "model.projector")]:
        assert nodes[f"v:path.{part}"]["expr"] == expr, part
    assert "Vision Config" in shown["config"] and '<details class="card card-tower card-fold">' in shown["config"]
    if entry.WRAPPERS[wrapper].get("tower", {}).get("BLOCK", True) is None:
        # a vision encoder with no blocks (an encoder-free embedder): the image's path and model.vision's ledger only
        assert vision["tower"]["schema"] is None and vision["tower"]["identity_html"] is None
        assert not any(key.startswith(("v:contrib.", "v:sub.", "v:interior.")) for key in nodes) and "v:path.layers" not in nodes
        assert nodes["v:path.tower_output"]["expr"] == "model.vision.tower_output"
        assert 'class="tower-svg"' not in shown["tower"] and "model.<wbr>vision.<wbr>layers" not in shown["values"]
        assert "model.<wbr>vision</span>" in shown["values"]
        return
    tower = vision["tower"]["schema"]
    assert [sub["host"] for sub in tower["sublayers"]] == ["self_attn", "mlp"]
    assert tower["identity"] == "vision.layers[i].input + self_attn.attention_output + mlp.mlp_output == layer_output"
    assert nodes["v:contrib.self_attn"]["expr"] == "model.vision.layers[i].self_attn.attention_output"
    assert nodes["v:contrib.mlp"]["expr"] == "model.vision.layers[i].mlp.mlp_output"
    for sub in tower["sublayers"]:
        for value in sub["interior"]:
            assert f"v:interior.{sub['host']}.{value['name']}" in nodes
    assert "layers[0].input[vision.image_token_mask] == vision.image_features" in nodes["strip.embed"]["extra"]
    assert 'class="tower-svg"' in shown["tower"] and ('class="tower-svg"' not in panes(page, text_pin)["tower"] or not text_side)
    for host in ("model.<wbr>vision", "model.<wbr>vision.<wbr>layers[i].<wbr>self_attn", "model.<wbr>vision.<wbr>layers[i].<wbr>mlp"):
        assert host in shown["values"], host
    for value in ("image_token_mask", "patch_embeddings", "tower_output", "image_features"):
        assert f">{value}</span>" in shown["values"], value
    assert "Vision Config" in shown["config"] and '<details class="card card-tower card-fold">' in shown["config"]
    for size in build.TOWER_SIZE_NAMES:
        assert f"model.vision.{size}</span>" in shown["config"], size
    assert '<h2 class="fold-heading">The <em>vision encoder</em></h2>' in shown["tower"]
    assert vision["architecture"].endswith("ForConditionalGeneration") and (text["architecture"].endswith("ForCausalLM") or not text_side)
    assert f'href="https://huggingface.co/{text_pin}"' in page and 'id="ckpt-hub"' in page
    assert f'href="https://huggingface.co/{tiny}"' not in page or not text_side, "only the selected checkpoint is linked, by the script"
    assert "Vision-language" in shown["chips"] and "<h2>" in shown["vision_notes"] and "<details" in shown["vision_notes"]
    assert "Vision-language" not in shown["quirk_list"] and "Vision-language" in shown["vision_notes"]
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


def test_a_greyed_checkpoint_is_listed_without_a_build(monkeypatch):
    """A checkpoint in an entry's GREYED is listed greyed out with the entry's reason, and nothing is read for it."""
    entry = entries.load("doge")
    greyed = next(iter(entry.GREYED))
    monkeypatch.setattr(build, "introspect", lambda *a, **k: pytest.fail("a GREYED checkpoint was built"))
    assert build.read_checkpoint(entry, greyed, {}) == {"id": greyed, "unavailable": entry.GREYED[greyed]}


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
        # a suite's local copy of PINNED, unless that local dir is itself a record's pinned (a wrapper the suite writes)
        suites = {entry.PINNED if os.path.isdir(repo) and repo not in pinned_wrappers else repo for repo in suites}
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


def test_every_wrapper_names_its_projector_input():
    """Every WRAPPERS record says what model.projector.input is: the image's path shows it on the projector node."""
    for name in entries.names():
        for wrapper, fields in getattr(entries.load(name), "WRAPPERS", {}).items():
            value = fields.get("projector_input")
            assert isinstance(value, str) and value.strip(), (name, wrapper)


def test_the_index_lists_every_vision_encoder_an_entry_resolves():
    """Each vision encoder some wrapper resolves to (a module, or one inline) has a filter on the index."""
    import vision
    from transformers import AutoConfig

    titles = set()
    for name in entries.names():
        entry = entries.load(name)
        for wrapper, fields in getattr(entry, "WRAPPERS", {}).items():
            repo = pinned(entry) if fields["pinned"] == entry.PINNED else fields["pinned"]  # the suite's copy, if any
            vision_type = AutoConfig.from_pretrained(repo).vision_config.model_type
            titles.add(vision.resolve(entry, wrapper, vision_type)["title"])
    cards = [{"model_type": "x", "towers": sorted(titles)}]
    shown = {tower["label"] for tower in build.index_model(cards)["towers"]}
    assert titles and shown == titles
    index = build.environment().get_template("index.html.j2").render(**build.index_model(
        [{**card, "title": "x", "subtitle": "", "palette": build.site_palette(), "checkpoints": [], "quirks": [],
          "vllm": False, "architecture": "", "family_module": "", "wrappers": [], "blocks": "1",
          "tower_slugs": [build.tower_slug(t) for t in card["towers"]]} for card in cards]))
    for title in titles:
        assert f'data-filter="{build.tower_slug(title)}">{title}</button>' in index, title


# -- what the Hub says: orgs, parameter counts, dates (hub_cache.json) ----------------------

import hub  # noqa: E402


def test_an_entrys_org_resolves_from_the_cache_offline(monkeypatch):
    """The org is the reference checkpoint's author (or the entry's ORG), named and pictured from the committed cache
    without asking the Hub; an id the cache does not hold shows as itself, with no avatar."""
    monkeypatch.setattr(hub, "REFRESH", False)
    for name, org, display in [("llama", "meta-llama", "Meta Llama"), ("gemma2", "google", "Google"), ("dbrx", "databricks", None)]:
        found = hub.org(entries.load(name))
        assert found["id"] == org, name
        assert display is None or found["name"] == display, (name, found)
        assert found["avatar"] and (Path(hub.__file__).parent / "static" / found["avatar"]).is_file(), (name, found)
    monkeypatch.setattr(hub, "_cache", {"orgs": {}, "repos": {}})
    assert hub.org(entries.load("llama")) == {"id": "meta-llama", "name": "meta-llama", "avatar": None}


def test_every_entrys_org_is_in_the_cache():
    """An offline build names every org: the cache holds each entry's, so an added entry needs one online build."""
    missing = [name for name in entries.names() if hub.org_id(entries.load(name)) not in hub.cache()["orgs"]]
    assert not missing, missing


def test_the_index_has_org_filters_and_cards_carry_their_org():
    org = {"id": "meta-llama", "name": "Meta Llama", "avatar": "orgs/meta-llama.png"}
    card = {"model_type": "llama", "title": "Llama", "subtitle": "", "org": org, "palette": build.site_palette(), "checkpoints": [],
            "quirks": [], "vllm": False, "architecture": "", "family_module": "", "towers": [], "wrappers": [], "blocks": "1",
            "tower_slugs": []}
    index = build.environment().get_template("index.html.j2").render(**build.index_model([card]))
    assert index.index(">Org</p>") < index.index(">Quirks</p>")
    assert 'data-org-filter="meta-llama"' in index and 'data-org="meta-llama"' in index
    assert index.count('src="static/orgs/meta-llama.png"') == 2 and index.count("Meta Llama</") == 2


def test_parameter_counts_and_dates_read_like_the_hub():
    assert [hub.format_params(n) for n in (8_030_261_248, 137_022_720, 405_000_000_000, 70_553_706_496, 1_040_000_000_000)] \
        == ["8.03B", "137M", "405B", "70.6B", "1.04T"]
    assert hub.format_month("2024-07-14") == "Jul 2024"


def test_params_and_date_render_when_cached_and_are_blank_when_not(monkeypatch):
    """Under the selector: the Hub's parameter count and creation month when the cache holds the repo; the meta model's
    count, said to be from the config, when the Hub has no safetensors metadata; nothing when the repo is not cached."""
    monkeypatch.setattr(hub, "REFRESH", False)
    entry = entries.load("llama")
    repo = pinned(entry)

    def hub_pane(repos):
        monkeypatch.setattr(hub, "_cache", {"orgs": {}, "repos": repos})
        return panes(build.build_page(entry, reference=repo), repo)["hub"].split("</div>")[0]  # the pane, not what follows

    shown = hub_pane({repo: {"params": 8_030_261_248, "created": "2024-07-14"}})
    assert "<b>8.03B</b> parameters" in shown and "on the Hub since <b>Jul 2024</b>" in shown and "from the config" not in shown
    info = build.read_entry(entry, reference=repo)[0][0]["info"]
    shown = hub_pane({repo: {"params": None, "created": "2024-07-14"}})
    assert f"<b>{hub.format_params(info['meta_params'])}</b> parameters" in shown and "from the config" in shown
    assert hub_pane({}).strip() == "" and hub_pane({repo: {"params": None, "created": None}}).strip() == ""
