"""Every check a family's vision side must pass, written once; and the text-stack checks on a wrapper built from a config.

`VisionSuite`: a family's test file subclasses it per multimodal wrapper,
names the pinned tiny wrapper, the native paths of the tower and projector
and the family's text-only checkpoint, and inherits end-to-end statements:
the tower names alias the native modules, the tower's blocks keep the
contribution identity, the tower's ``image_token_mask`` and ``image_features``
mean what they say (``layers[0].input[image_token_mask] == image_features``),
edits are causal, and a text-only checkpoint or load lists no ``vision`` host.

`WrapperSuite`: the text names on a multimodal wrapper of a text family, for a
family whose wrapper has no tiny checkpoint to run the whole `FamilySuite` on
(the wrapper is built from the family's tiny text config) or whose processor
refuses a text-only prompt (PaliGemma).
"""

import numpy as np
import pytest
import torch
from nnsight.intervention.envoy import Envoy
from PIL import Image
from suite import PROMPT, contributions

from nnterp import StandardizedTransformer, Unavailable
from nnterp.components import (
    ImageFeatures, ImageScatter, ImageTokenMask, Patches, Pattern, Vision, VisionAttention, VisionLayer, VisionMlp, image_token_id,
)
from nnterp.components.vision import scatter_call, scatter_host

#: The tower's own values, in forward order; its block values are those of a text block.
TOWER_VALUES = ("image_token_mask", "patch_embeddings", "tower_output", "image_features")
BLOCK_VALUES = {
    "layer_input", "layer_output", "self_attn.attention_output", "self_attn.attention_probabilities", "self_attn.attention_queries",
    "self_attn.attention_keys", "self_attn.attention_values", "self_attn.attention_scores",
    "self_attn.attention_head_outputs", "mlp.mlp_output",
}

#: A fixed random image; the processor resizes it to the tower's size.
IMAGE = Image.fromarray((np.random.RandomState(0).rand(64, 64, 3) * 255).astype("uint8"))
#: A second image of another shape: a packed tower concatenates the images it is given, a variable-resolution one sizes it
#: its own way.
IMAGE_WIDE = Image.fromarray((np.random.RandomState(1).rand(48, 96, 3) * 255).astype("uint8"))


def as_rows(native):
    """A patch embedding's native output as rows of patches, ``[images, patches, vision_hidden]``: a convolution's grid
    flattened, a packed tower's ``[patches, vision_hidden]`` with a leading 1."""
    if native.dim() == 4:
        return native.flatten(2).transpose(1, 2)
    return native.unsqueeze(0) if native.dim() == 2 else native


def image_prompt(model, text="What is in this image?", images=1):
    """A prompt with ``images`` image placeholders where the processor's chat template puts them.

    A tiny checkpoint without a usable template (VipLlava's, LLaVA-NeXT's) gets the processor's image tokens first.
    """
    messages = [{"role": "user", "content": [*[{"type": "image"}] * images, {"type": "text", "text": text}]}]
    try:
        return model.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    except (ValueError, TypeError):
        return model.processor.image_token * images + f"\n{text}"


def siglip_rows(tower="model.vision_tower"):
    """Standard path -> native path for a SigLIP tower at ``tower`` and the projector beside it."""
    return {
        "vision": tower,
        "vision.layers": f"{tower}.encoder.layers",
        "vision.patch_embed": f"{tower}.embeddings.patch_embedding",
        "vision.norm": f"{tower}.post_layernorm",
        "vision.layers.0.self_attn": f"{tower}.encoder.layers.0.self_attn",
        "vision.layers.0.mlp": f"{tower}.encoder.layers.0.mlp",
        "vision.layers.0.input_layernorm": f"{tower}.encoder.layers.0.layer_norm1",
        "vision.layers.0.post_attention_layernorm": f"{tower}.encoder.layers.0.layer_norm2",
        "projector": "model.multi_modal_projector",
    }



def pixtral_rows(tower="model.vision_tower"):
    """Standard path -> native path for a Pixtral tower at ``tower`` and the projector beside it."""
    return {
        "vision": tower,
        "vision.layers": f"{tower}.transformer.layers",
        "vision.patch_embed": f"{tower}.patch_conv",
        "vision.layers.0.self_attn": f"{tower}.transformer.layers.0.attention",
        "vision.layers.0.mlp": f"{tower}.transformer.layers.0.feed_forward",
        "vision.layers.0.input_layernorm": f"{tower}.transformer.layers.0.attention_norm",
        "vision.layers.0.post_attention_layernorm": f"{tower}.transformer.layers.0.ffn_norm",
        "projector": "model.multi_modal_projector",
    }


def clip_rows(tower="model.vision_tower"):
    """Standard path -> native path for a CLIP tower at ``tower`` (no ``vision.norm``: CLIP's ``post_layernorm`` norms
    the pooled CLS token only) and the projector beside it."""
    rows = siglip_rows(tower)
    del rows["vision.norm"]
    return rows


def align_processor(model, image_processor=None, **processor):
    """Set a tiny checkpoint's processor to its model: the given processor and image-processor attributes, and the
    config's ``image_token_id`` to the processor's image token (the hf-tiny-v2 checkpoints disagree on all three)."""
    for name, value in processor.items():
        setattr(model.processor, name, value)
    for name, value in (image_processor or {}).items():
        setattr(model.processor.image_processor, name, value)
    model.config.image_token_id = model.processor.image_token_id


class VisionSuite:
    """Subclass per multimodal wrapper: set the class attributes, add the tower's own tests."""

    #: The pinned tiny wrapper checkpoint.
    REPO: str
    #: The family module it must resolve to.
    FAMILY = None
    #: Standard path -> native path, for the tower and the projector.
    VISION_NATIVE: dict
    #: A text-only checkpoint of the same family: it must list no image values.
    TEXT_REPO: str
    #: The dtype the wrapper is loaded in: float32, unless the processor hands the tower another (Llama 4's bfloat16).
    DTYPE = torch.float32
    #: Where ``vision.patch_embeddings`` is read, from the tower: the patch embedding's output (Pixtral: ``ln_pre.input``).
    PATCHES_AT = "patch_embed.output"
    #: Tower block values unavailable on every block: value -> a substring of the reason (as `FamilySuite.EXPECTED_UNAVAILABLE`).
    EXPECTED_VISION_UNAVAILABLE: dict = {}

    def patches_of(self, model, images):
        """The length of the patches axis for ``images``: the configured grid, one per row, on a fixed-resolution tower."""
        return (model.vision.image_size // model.vision.patch_size) ** 2

    @pytest.fixture(scope="class")
    def model(self, request):
        cls = request.cls
        model = StandardizedTransformer(
            cls.REPO, task="image-text-to-text", dispatch=True, attn_implementation="eager", dtype=cls.DTYPE,
        )
        cls.fix_processor(model)
        return model

    @staticmethod
    def fix_processor(model):
        """Where a tiny checkpoint's processor disagrees with its model, set it to the model's (a subclass says how)."""

    @pytest.fixture(scope="class")
    def clean(self, model):
        """One traced run on the image: the values every test compares against."""
        with model.trace(image_prompt(model), images=[IMAGE]):
            ids = model.input_ids.save()
            mask = model.vision.image_token_mask.save()
            features = model.vision.image_features.save()
            first = model.layers[0].input.save()
            logits = model.logits.save()
        return {"ids": ids, "mask": mask, "features": features, "first": first, "logits": logits}

    # -- names ------------------------------------------------------------------------

    def test_family_resolved(self, model):
        assert model.family is self.FAMILY

    def test_the_scatter_host_is_keyed(self, model):
        """The module whose forward writes the image features in: an `ImageScatter`, or the root where the family names ``ROOT_SCATTER``."""
        name, host = scatter_host(model)
        assert isinstance(host, ImageScatter) if name else (host is model and model.family.ROOT_SCATTER)
        assert model.get(scatter_call(model)[0]) is not None  # the operation is in the host's forward

    def test_tower_names_alias_native_envoys(self, model):
        for standard, native in self.VISION_NATIVE.items():
            assert isinstance(model.get(standard), Envoy), standard
            assert model.get(standard) is model.get(native), (standard, native)

    def test_tower_envoy_classes(self, model):
        assert isinstance(model.vision, Vision)
        assert len(model.vision.layers) > 1
        for layer in model.vision.layers:
            assert isinstance(layer, VisionLayer)
            assert isinstance(layer.self_attn, VisionAttention) and isinstance(layer.mlp, VisionMlp)
            assert isinstance(layer.input_layernorm, Envoy) and isinstance(layer.post_attention_layernorm, Envoy)

    def test_no_tower_name_binds_on_the_text_model(self, model):
        """The tower's keys bind on the tower alone: the text blocks keep the family's classes and names."""
        assert all(type(layer) is self.FAMILY.Layer for layer in model.layers)
        tower = {"vision", "projector", "patch_embed"}
        assert not tower & set(model.layers[0]._aliases)
        assert not tower & set(model.layers._aliases)
        assert model.layers is not model.vision.layers

    def test_sizes_are_the_towers(self, model):
        """The tower's sizes (off its config, whatever it calls them) are the ones its modules run with."""
        vision, layer = model.vision, model.vision.layers[0]
        assert vision.num_layers == len(vision.layers)
        assert layer.input_layernorm._module.weight.shape[-1] == vision.hidden_size
        assert (layer.self_attn.num_heads, layer.self_attn.head_dim) == (vision.num_heads, vision.head_dim)
        assert layer.mlp.intermediate_size == vision.intermediate_size
        assert vision.patch_size == model.config.vision_config.patch_size
        text = model.config.get_text_config()
        assert model.num_layers == len(model.layers) == text.num_hidden_layers
        assert model.hidden_size == text.hidden_size

    # -- availability ------------------------------------------------------------------

    def test_support_lists_the_tower_values(self, model):
        vision = model.vision.support()
        assert set(vision) == {*TOWER_VALUES, *BLOCK_VALUES}
        for name, reason in vision.items():
            if name in self.EXPECTED_VISION_UNAVAILABLE:
                assert set(reason) == set(range(model.vision.num_layers)), name
                assert all(self.EXPECTED_VISION_UNAVAILABLE[name] in r for r in reason.values()), (name, reason)
            else:
                assert reason is None, (name, reason)
        assert set(model.vision.support(layer=0)) == BLOCK_VALUES
        support = model.support()
        assert {name.removeprefix("vision."): reason for name, reason in support.items() if name.startswith("vision.")} == vision
        assert not {"image_token_mask", "image_features"} & set(support)

    def test_the_image_values_are_the_towers(self, model):
        assert "image_token_mask" not in type(model).__dict__ and "image_features" not in type(model).__dict__
        assert list(Vision.values())[:2] == ["image_token_mask", "patch_embeddings"]
        assert Vision.values().keys() >= set(TOWER_VALUES)

    def test_a_text_only_checkpoint_has_no_vision_host(self):
        text = StandardizedTransformer(self.TEXT_REPO)
        assert text.family is self.FAMILY
        assert "vision" not in text._aliases and "projector" not in text._aliases
        assert not any(name.startswith("vision.") for name in text.support())

    def text_generation_load(self):
        """The wrapper loaded for text generation: no processor (dispatched, transformers builds the wrapper for a config
        it has no text-only class for)."""
        return StandardizedTransformer(self.REPO, dispatch=True, dtype=self.DTYPE)

    def test_a_text_generation_load_serves_no_tower_value(self):
        """No image reaches a load without a processor: the tower never runs, so every tower and block value is
        `Unavailable` saying how to load, inside a trace too, rather than failing out of order."""
        text = self.text_generation_load()
        if "vision" not in text._aliases:
            pytest.skip(f"text-generation builds the text-only class, {type(text._module).__name__}")
        assert "projector" in text._aliases and text.processor is None
        assert isinstance(text.vision, Vision) and text.vision.support() == {}
        assert not any(name.startswith("vision.") for name in text.support())
        reads = [(text.vision, name) for name in TOWER_VALUES]
        if text.vision.num_layers:
            layer = text.vision.layers[0]
            reads += [(layer, "layer_output"), (layer.mlp, "mlp_output")]
            reads += [(layer.self_attn, name.removeprefix("self_attn.")) for name in BLOCK_VALUES if name.startswith("self_attn.")]
        for host, name in reads:
            with pytest.raises(Unavailable, match="text-only load.*task='image-text-to-text'"):
                getattr(host, name)
        host, name = reads[-1]
        with pytest.raises(Unavailable, match="text-only load"):
            with text.trace(dict(text.tokenizer(PROMPT, return_tensors="pt"))):
                getattr(host, name)

    def test_layouts(self, model):
        assert type(model.vision).image_token_mask.layout is ImageTokenMask
        assert type(model.vision).image_features.layout is ImageFeatures
        layer = model.vision.layers[0]
        assert type(layer).layer_output.layout is Patches
        assert type(layer.self_attn).attention_output.layout is Patches
        assert type(layer.mlp).mlp_output.layout is Patches
        assert type(layer.self_attn).attention_probabilities.layout is Pattern
        assert type(model.vision).patch_embeddings.layout is Patches

    # -- where the image meets the text model ------------------------------------------

    def test_image_token_mask_is_the_image_tokens(self, model, clean):
        mask, ids = clean["mask"], clean["ids"]
        assert isinstance(mask, ImageTokenMask) and mask.dtype == torch.bool
        assert torch.equal(mask, ids == image_token_id(model.config))
        assert int(mask.sum()) == clean["features"].shape[0] > 0

    def test_image_features_are_what_enters_the_text_model(self, model, clean):
        features = clean["features"]
        assert isinstance(features, ImageFeatures) and features.shape[-1] == model.hidden_size
        assert torch.equal(clean["first"][clean["mask"]], features)

    def test_zeroing_image_features_lands_and_moves_the_logits(self, model, clean):
        with model.trace(image_prompt(model), images=[IMAGE]):
            model.vision.image_features[:] = 0
            first = model.layers[0].input.save()
            logits = model.logits.save()
        assert (first[clean["mask"]] == 0).all()
        assert torch.equal(first[~clean["mask"]], clean["first"][~clean["mask"]])
        assert not torch.allclose(logits, clean["logits"])

    def test_assigning_image_features_lands(self, model, clean):
        replacement = torch.randn_like(clean["features"])
        with model.trace(image_prompt(model), images=[IMAGE]):
            model.vision.image_features = replacement
            first = model.layers[0].input.save()
        assert torch.equal(first[clean["mask"]], replacement)

    def test_a_tower_write_moves_the_image_features(self, model, clean):
        with model.trace(image_prompt(model), images=[IMAGE]):
            model.vision.layers[0].layer_output = torch.randn_like(model.vision.layers[0].layer_output)
            features = model.vision.image_features.save()
        assert not torch.allclose(features, clean["features"])

    def test_mask_and_features_are_read_only_where_derived(self, model):
        with pytest.raises(AttributeError, match="assign input_ids"):
            with model.trace(image_prompt(model), images=[IMAGE]):
                model.vision.image_token_mask = torch.zeros(1, 1, dtype=torch.bool)

    def text_input(self, model):
        """What a text-only trace is given: the prompt, or its encoding where the processor demands an image (PaliGemma)."""
        return PROMPT

    def test_a_text_only_trace(self, model):
        with model.trace(self.text_input(model)):
            mask = model.vision.image_token_mask.save()
            out = model.layers[-1].layer_output.save()
            logits = model.logits.save()
        assert not mask.any()
        assert out.shape[-1] == model.hidden_size and logits.shape[-1] == model.vocab_size

    # -- the tower ------------------------------------------------------------------------

    def test_tower_contribution_identity(self, model):
        for layer in model.vision.layers:
            with model.trace(image_prompt(model), images=[IMAGE]):
                stream = layer.input.save()
                attn = layer.self_attn.attention_output.save()
                mlp = layer.mlp.mlp_output.save()
                out = layer.layer_output.save()
            assert isinstance(out, Patches) and out.shape[-1] == model.vision.hidden_size
            torch.testing.assert_close(stream + attn + mlp, out)

    def test_tower_pattern_sums_to_one_over_keys(self, model):
        if "self_attn.attention_probabilities" in self.EXPECTED_VISION_UNAVAILABLE:
            pytest.skip("this tower serves no pattern: test_support_lists_the_tower_values")
        for layer in model.vision.layers:
            with model.trace(image_prompt(model), images=[IMAGE]):
                pattern = layer.self_attn.attention_probabilities.save()
                out = layer.layer_output.save()
            assert pattern.shape == (out.shape[0], model.vision.num_heads, out.shape[1], out.shape[1])
            torch.testing.assert_close(pattern.sum(-1), torch.ones_like(pattern[..., 0]))

    def read_patches_at(self, model):
        """The native value ``patch_embeddings`` is read at (`PATCHES_AT`), in a trace."""
        module, attribute = self.PATCHES_AT.rsplit(".", 1)
        return getattr(model.vision.get(module), attribute)

    def test_patch_embeddings_and_tower_output(self, model, clean):
        """``patch_embeddings`` is the rows at `PATCHES_AT`; ``tower_output`` the last block's stream after the final norm, if any."""
        vision = model.vision
        with model.trace(image_prompt(model), images=[IMAGE]):
            patches = vision.patch_embeddings.save()
            native = self.read_patches_at(model).save()
            # An embedder with no blocks (Gemma 4 unified) ends no stream: its tower_output is what the projector receives.
            last = vision.layers[-1].layer_output.save() if vision.num_layers else model.projector.input.save()
            out = vision.tower_output.save()
        assert isinstance(patches, Patches) and patches.shape[1:] == (self.patches_of(model, [IMAGE]), vision.hidden_size)
        assert torch.equal(patches, as_rows(native))
        norm = getattr(vision, "norm", None)
        torch.testing.assert_close(out, norm._module(last) if norm is not None else last)

    def test_patch_embeddings_edits_land(self, model, clean):
        vision = model.vision
        with model.trace(image_prompt(model), images=[IMAGE]):
            vision.patch_embeddings[:, 0] = 0
            native = self.read_patches_at(model).save()
            features = vision.image_features.save()
        assert (as_rows(native)[:, 0] == 0).all()
        assert not torch.allclose(features, clean["features"])

    def test_two_images_in_one_invoke(self, model, clean):
        """Two images of different shapes: the mask holds both images' tokens and the scatter identity holds; a packed
        tower (1 in the images axis) holds both images' patches in one row, and its pattern, where served, is zero
        between them."""
        prompt = image_prompt(model, "Compare these images.", images=2)
        pattern_served = model.vision.num_layers and "self_attn.attention_probabilities" not in self.EXPECTED_VISION_UNAVAILABLE
        saved = {}  # made outside the block: names bound inside do not survive it
        with model.trace(prompt, images=[IMAGE, IMAGE_WIDE]):
            saved["mask"] = model.vision.image_token_mask.save()
            saved["patches"] = model.vision.patch_embeddings.save()
            if pattern_served:
                saved["pattern"] = model.vision.layers[0].self_attn.attention_probabilities.save()
            saved["features"] = model.vision.image_features.save()
            saved["first"] = model.layers[0].input.save()
        mask, patches, features = saved["mask"], saved["patches"], saved["features"]
        assert torch.equal(saved["first"][mask], features)
        assert int(mask.sum()) == features.shape[0] > int(clean["mask"].sum())
        assert patches.shape[1] == self.patches_of(model, [IMAGE, IMAGE_WIDE])
        if patches.shape[0] != 1:  # one row per image, crop or tile
            return
        counts = [self.patches_of(model, [IMAGE]), self.patches_of(model, [IMAGE_WIDE])]
        assert counts[0] != counts[1] and patches.shape[1] == sum(counts)
        if pattern_served:
            pattern, one, two = saved["pattern"], slice(0, counts[0]), slice(counts[0], sum(counts))
            assert pattern.shape == (1, model.vision.num_heads, sum(counts), sum(counts))
            assert (pattern[..., one, two] == 0).all() and (pattern[..., two, one] == 0).all()


class PixtralSuite(VisionSuite):
    """`VisionSuite` for Pixtral: a packed tower, every image's patches in one row under a block-diagonal mask."""

    VISION_NATIVE = pixtral_rows()
    PATCHES_AT = "ln_pre.input"

    def patches_of(self, model, images):
        """Every image's patches, off the processor's ``image_sizes``: ``(height // patch_size) * (width // patch_size)`` each."""
        sizes = model.processor(text=image_prompt(model, images=len(images)), images=images, return_tensors="pt")["image_sizes"]
        side = model.vision.patch_size
        return sum(int(height) // side * (int(width) // side) for height, width in sizes)

    def test_no_final_norm_and_no_image_size(self, model):
        assert "norm" not in model.vision._aliases
        with pytest.raises(Unavailable, match="any resolution"):
            model.vision.image_size


def wrapper_of(text_repo, config_class, vision_config, **load):
    """A multimodal wrapper around a text family's tiny checkpoint config, random weights, loaded as a module.

    For a family whose wrapper has no tiny checkpoint: the wrapper (the class
    ``AutoModelForImageTextToText`` maps ``config_class`` to) is built from the tiny
    text config and a small ``vision_config`` (an ``out_hidden_size`` of ``None``
    takes the text width), and handed to `StandardizedTransformer` with the text
    checkpoint's tokenizer, so the task is text generation and the tree is the
    wrapper's.
    """
    import transformers
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer

    text = AutoConfig.from_pretrained(text_repo)
    if "out_hidden_size" in vision_config and vision_config["out_hidden_size"] is None:  # a merger projecting onto the text width
        vision_config = {**vision_config, "out_hidden_size": text.hidden_size}
    config = getattr(transformers, config_class)(text_config=text.to_dict(), vision_config=vision_config)
    torch.manual_seed(0)
    module = AutoModelForImageTextToText.from_config(config, attn_implementation="eager").eval()
    return StandardizedTransformer(module, tokenizer=AutoTokenizer.from_pretrained(text_repo), **load)


class WrapperSuite:
    """The text names on a family's multimodal wrapper: they bind, `support` runs, a text-only trace keeps the identity."""

    FAMILY = None
    #: Where the wrapper keeps the text model.
    CONTAINER = "model.language_model"

    @pytest.fixture(scope="class")
    def model(self, request):
        raise NotImplementedError("a WrapperSuite subclass defines the model fixture")

    def text_input(self, model):
        """What a text-only trace is given: the prompt, or its encoding where the processor demands an image."""
        return PROMPT

    def test_family_resolved(self, model):
        assert model.family is self.FAMILY
        assert hasattr(model.config, "text_config")

    def test_text_names_alias_the_wrapper_text_stack(self, model):
        for standard in ("embed_tokens", "layers", "norm"):
            assert model.get(standard) is model.get(f"{self.CONTAINER}.{standard}"), standard
        assert model.lm_head is model.get("lm_head")
        assert all(type(layer) is self.FAMILY.Layer for layer in model.layers)
        assert all(type(layer.self_attn) is self.FAMILY.Attention for layer in model.layers)

    def test_support_runs_and_lists_vision_rows_only_where_an_image_reaches(self, model):
        """A wrapper built here has no processor, so its tower never runs; PaliGemma's, loaded with one, lists its tower."""
        support = model.support()
        assert "layer_output" in support and "self_attn.attention_output" in support
        reaches = "vision" in model._aliases and model.vision.no_images() is None
        assert any(name.startswith("vision.") for name in support) == reaches

    def test_text_only_trace_keeps_the_identity(self, model):
        layer = model.layers[0]
        hosts = contributions(layer)
        parts = []  # made outside the block: names bound inside do not survive it
        with model.trace(self.text_input(model)):
            stream = layer.input.save()
            for host, value in hosts:
                parts.append(getattr(host, value).save())
            out = layer.layer_output.save()
            logits = model.logits.save()
        torch.testing.assert_close(stream + sum(parts), out)
        assert logits.shape[-1] == model.vocab_size
