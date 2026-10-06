"""The Qwen ViT's checks, written once for the six families that host it.

`QwenVisionSuite` is `VisionSuite` on a packed tower: the tower runs on
``[patches, vision_hidden]`` over every image of the invoke, so its stream
values are served ``[1, patches, vision_hidden]``, the attention interior is
`Unavailable` with the packed reason, the sizes are the Qwen vision config's
spellings, ``image_grid_thw`` says how many patches each image has, and the
features the text model receives are the tower's ``pooler_output``.
"""

import numpy as np
import pytest
import torch
from PIL import Image
from suite import INTERIOR
from vision_suite import BLOCK_VALUES, IMAGE, TOWER_VALUES, VisionSuite, image_prompt

from nnterp import StandardizedTransformer, Unavailable
from nnterp.components import PACKED, ImageFeatures, PackedVisionAttention, Patches, QwenVision

#: The tower's attention interior, unavailable on every block of a packed tower.
PACKED_VALUES = {f"self_attn.{name}" for name in (*INTERIOR, "attention_probabilities")}
#: What a `FamilySuite` on a wrapper loaded with its processor expects of the tower's rows.
PACKED_UNAVAILABLE = {f"vision.{name}": "a packed tower" for name in PACKED_VALUES}

#: A second image, larger and not square: more patches than `IMAGE`, and several attention windows on Qwen2.5-VL.
WIDE = Image.fromarray((np.random.RandomState(1).rand(256, 320, 3) * 255).astype("uint8"))


def images_prompt(model, count):
    """A prompt with ``count`` image placeholders."""
    content = [{"type": "image"} for _ in range(count)] + [{"type": "text", "text": "What is in these images?"}]
    return model.processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False)


class QwenVisionSuite(VisionSuite):
    """Subclass per family hosting the Qwen ViT: set the class attributes, as for `VisionSuite`."""

    VISION_NATIVE = {
        "vision": "model.visual",
        "vision.layers": "model.visual.blocks",
        "vision.patch_embed": "model.visual.patch_embed",
        "vision.layers.0.self_attn": "model.visual.blocks.0.attn",
        "vision.layers.0.mlp": "model.visual.blocks.0.mlp",
        "vision.layers.0.input_layernorm": "model.visual.blocks.0.norm1",
        "vision.layers.0.post_attention_layernorm": "model.visual.blocks.0.norm2",
        "projector": "model.visual.merger",
    }
    #: A text-only load of the family, or ``None`` where every checkpoint loads as the wrapper.
    TEXT_REPO = None
    #: Whether the tower holds its patches in window order (Qwen2.5-VL), so the merger's output is not in scatter order.
    WINDOWED = False

    def patches_of(self, model, *images):
        """How many patches the processor cuts ``images`` into: ``image_grid_thw``'s products, summed."""
        grid = model.processor(text=images_prompt(model, len(images)), images=list(images), return_tensors="pt")["image_grid_thw"]
        return int(grid.prod(-1).sum())

    # -- names and sizes --------------------------------------------------------------

    def test_the_tower_is_packed(self, model):
        assert isinstance(model.vision, QwenVision)
        assert all(isinstance(layer.self_attn, PackedVisionAttention) for layer in model.vision.layers)
        assert model.projector._module is model.vision._module.merger  # the projector is inside the tower

    def test_sizes_are_the_towers(self, model):
        vision, config = model.vision, model.config.vision_config
        width = getattr(config, "embed_dim", None) or config.hidden_size
        assert vision.num_layers == len(vision.layers) == config.depth
        assert (vision.hidden_size, vision.num_heads, vision.patch_size, vision.spatial_merge_size) == (
            width, config.num_heads, config.patch_size, config.spatial_merge_size)
        assert vision.window_size == getattr(config, "window_size", None)
        with pytest.raises(Unavailable, match="any resolution"):
            vision.image_size
        assert vision.head_dim * vision.num_heads == vision.hidden_size
        attn = vision.layers[0].self_attn
        assert (attn.num_heads, attn.head_dim) == (vision.num_heads, vision.head_dim)
        assert vision.layers[0].mlp.intermediate_size == vision.intermediate_size
        assert vision.layers[0].mlp._module(torch.zeros(1, width, dtype=model.dtype, device=model.device)).shape[-1] == width
        text = model.config.get_text_config()
        assert model.num_layers == len(model.layers) == text.num_hidden_layers
        assert model.hidden_size == text.hidden_size

    # -- availability ------------------------------------------------------------------

    def test_a_text_only_checkpoint_has_no_vision_host(self):
        if self.TEXT_REPO is None:
            pytest.skip("transformers has no text-only class for this config: every checkpoint loads as the wrapper")
        super().test_a_text_only_checkpoint_has_no_vision_host()

    def test_support_lists_the_tower_values(self, model):
        vision = model.vision.support()
        assert set(vision) == {*TOWER_VALUES, *BLOCK_VALUES}
        assert all(reason is None for name, reason in vision.items() if name not in PACKED_VALUES), vision
        for name in PACKED_VALUES:
            assert set(vision[name]) == set(range(model.vision.num_layers)), name
            assert all(reason == PACKED for reason in vision[name].values()), name
        support = model.support()
        assert {name.removeprefix("vision."): reason for name, reason in support.items() if name.startswith("vision.")} == vision

    def test_the_interior_is_unavailable_with_the_packed_reason(self, model):
        attn = model.vision.layers[0].self_attn
        for name in PACKED_VALUES:
            with pytest.raises(Unavailable, match="one interface call per image"):
                getattr(attn, name.removeprefix("self_attn."))
        assert "eager" not in PACKED  # loading eager would not help: the reason does not say it would

    def test_the_packed_reason_holds_whatever_the_attention_implementation(self):
        """Under sdpa the tower's reason is still the packed one (loading eager would not help); the text side's says eager."""
        sdpa = StandardizedTransformer(self.REPO, task="image-text-to-text", dispatch=True, attn_implementation="sdpa", dtype=torch.float32)
        support = sdpa.vision.support()
        assert all(reason == PACKED for name in PACKED_VALUES for reason in support[name].values())
        assert "attn_implementation='eager'" in str(sdpa.support()["self_attn.attention_probabilities"])

    def test_tower_pattern_sums_to_one_over_keys(self, model):
        pytest.skip("a packed tower serves no pattern: test_the_interior_is_unavailable_with_the_packed_reason")

    # -- the packed layout -----------------------------------------------------------------

    def test_values_are_the_packed_stream_with_a_leading_one(self, model):
        vision, layer = model.vision, model.vision.layers[0]
        with model.trace(image_prompt(model), images=[IMAGE]):
            patches = vision.patch_embeddings.save()
            native_patches = vision.patch_embed.output.save()
            attn = layer.self_attn.attention_output.save()
            native_attn = layer.self_attn.output.save()
            mlp = layer.mlp.mlp_output.save()
            out = layer.layer_output.save()
            native_out = layer.output.save()
            tower = vision.tower_output.save()
        n = self.patches_of(model, IMAGE)
        for value in (patches, attn, mlp, out, tower):
            assert isinstance(value, Patches) and value.shape == (1, n, vision.hidden_size)
        assert native_out.shape == (n, vision.hidden_size)  # natively packed, no images axis
        assert torch.equal(patches[0], native_patches) and torch.equal(attn[0], native_attn) and torch.equal(out[0], native_out)

    def test_patch_embeddings_and_tower_output(self, model, clean):
        vision = model.vision
        with model.trace(image_prompt(model), images=[IMAGE]):
            patches = vision.patch_embeddings.save()
            last = vision.layers[-1].layer_output.save()
            out = vision.tower_output.save()
        assert patches.shape == (1, self.patches_of(model, IMAGE), vision.hidden_size)
        torch.testing.assert_close(out, last)  # no final norm over the patches: the merger norms its own input

    def test_patch_embeddings_edits_land(self, model, clean):
        vision = model.vision
        with model.trace(image_prompt(model), images=[IMAGE]):
            vision.patch_embeddings[:, 0] = 0
            native = vision.patch_embed.output.save()
            features = vision.image_features.save()
        assert (native[0] == 0).all()
        assert not torch.allclose(features, clean["features"])

    def test_writes_with_a_leading_one_land(self, model, clean):
        layer = model.vision.layers[0]
        with model.trace(image_prompt(model), images=[IMAGE]):
            replacement = torch.zeros_like(layer.layer_output)
            layer.layer_output = replacement
            entering = model.vision.layers[1].input.save()
            features = model.vision.image_features.save()
        assert entering.dim() == 2 and (entering == 0).all()
        assert not torch.allclose(features, clean["features"])

    # -- the image features ------------------------------------------------------------------

    def test_image_features_are_the_towers_pooler_output(self, model, clean):
        with model.trace(image_prompt(model), images=[IMAGE]):
            merged = model.projector.output.save()
            pooled = model.vision.output.pooler_output.save()
            features = model.vision.image_features.save()
        assert torch.equal(features, pooled) and torch.equal(features, clean["features"])
        assert merged.shape == features.shape
        n = self.patches_of(model, IMAGE)
        assert features.shape[0] * model.vision.spatial_merge_size ** 2 == n

    def test_the_merger_output_is_in_scatter_order_unless_windowed(self, model):
        """On a large image, Qwen2.5-VL's merger output is in window order and the tower restores the order after it."""
        with model.trace(image_prompt(model), images=[WIDE]):
            mask = model.vision.image_token_mask.save()
            merged = model.projector.output.save()
            features = model.vision.image_features.save()
            first = model.layers[0].input.save()
        assert torch.equal(first[mask], features)
        if self.WINDOWED:
            assert not torch.equal(merged, features)
            assert torch.equal(merged.sort(0).values, features.sort(0).values)  # the same rows, permuted
        else:
            assert torch.equal(merged, features)

    def test_two_images_in_one_invoke(self, model, clean):
        """The patches of both images in one row; the mask holds both images' tokens; the scatter still holds."""
        with model.trace(images_prompt(model, 2), images=[IMAGE, WIDE]):
            mask = model.vision.image_token_mask.save()
            patches = model.vision.patch_embeddings.save()
            out = model.vision.layers[-1].layer_output.save()
            features = model.vision.image_features.save()
            first = model.layers[0].input.save()
        n = self.patches_of(model, IMAGE, WIDE)
        assert n > self.patches_of(model, IMAGE)
        assert patches.shape == out.shape == (1, n, model.vision.hidden_size)
        assert int(mask.sum()) == features.shape[0] == n // model.vision.spatial_merge_size ** 2
        assert int(mask.sum()) > int(clean["mask"].sum())
        assert torch.equal(first[mask], features)
        image_tokens = mask[0].nonzero().flatten()
        assert (image_tokens.diff() > 1).sum() == 1  # two runs of image tokens: one per image


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class MRopeSuite:
    """Mixed into a Qwen-VL text family's `FamilySuite`: the queries and keys at the interface are M-RoPE-rotated."""

    def test_queries_and_keys_are_rotated_by_mrope(self, model):
        """``rotary_emb`` folds the three position streams into one cos/sin; the attention rotates with it before the interface."""
        attn = model.layers[0].self_attn
        rotary = model.layers.parent.rotary_emb
        with model.trace(image_prompt(model), images=[IMAGE]):
            positions = rotary.inputs[0][1].save()
            cos, sin = attn.inputs[1]["position_embeddings"]
            cos, sin = cos.save(), sin.save()
            q = (attn.q_norm if hasattr(attn, "q_norm") else attn.q_proj).output.save()
            k = (attn.k_norm if hasattr(attn, "k_norm") else attn.k_proj).output.save()
        with model.trace(image_prompt(model), images=[IMAGE]):  # the interface is reached by drilling in before the module runs
            queries = attn.attention_queries.save()
            keys = attn.attention_keys.save()
        assert positions.shape[0] in (3, 4)  # temporal, height, width (a fourth, the text positions, rides in front on Qwen3-VL)
        streams = positions[-3:]
        assert not torch.equal(streams[1], streams[2])  # the image's rows and columns differ: three streams, not one
        batch, seq = queries.shape[0], queries.shape[2]
        q = q.reshape(batch, seq, -1, attn.head_dim).transpose(1, 2)
        k = k.reshape(batch, seq, -1, attn.head_dim).transpose(1, 2)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        torch.testing.assert_close(queries, q * cos + rotate_half(q) * sin)
        torch.testing.assert_close(keys, k * cos + rotate_half(k) * sin)


class DeepstackSuite:
    """Mixed into Qwen3-VL's text families' `FamilySuite`: the deepstack features the text model adds after its first blocks."""

    def entering(self, model, k):
        """What receives block ``k``'s stream: block ``k + 1``, or the final norm after the last block."""
        return model.layers[k + 1] if k + 1 < len(model.layers) else model.norm

    def expected_values(self, model):
        return super().expected_values(model) | {"deepstack_output"}

    def test_values_match_their_annotations(self, model, monkeypatch):
        """The base traces a text-only prompt, where the text model makes no deepstack call; the deepstack value's
        layout is checked on an image (`test_deepstack_output_is_what_the_text_model_adds`)."""
        layer = type(model.layers[0])
        support = layer.support
        monkeypatch.setattr(layer, "support", lambda self: {**support(self), "deepstack_output": "needs an image"})
        super().test_values_match_their_annotations(model)

    def deepstack_count(self, model):
        return min(len(model.config.vision_config.deepstack_visual_indexes), model.num_layers)

    def test_deepstack_is_listed_and_available_on_the_first_blocks(self, model):
        support = model.support()
        count = self.deepstack_count(model)
        if count == model.num_layers:
            assert support["deepstack_output"] is None
        else:
            assert set(support["deepstack_output"]) == set(range(count, model.num_layers))
        assert type(model.layers[0]).deepstack_output.layout is ImageFeatures

    def test_deepstack_output_is_what_the_text_model_adds(self, model):
        """``layers[k+1].input[mask] == layers[k].layer_output[mask] + layers[k].deepstack_output``; the rest of the stream passes."""
        count = self.deepstack_count(model)
        outs, added, entering, merged = [], [], [], []   # made outside the block: names bound inside do not survive it
        with model.trace(image_prompt(model), images=[IMAGE]):
            mask = model.vision.image_token_mask.save()
            for k in range(count):
                merged.append(model.vision.deepstack_merger_list[k].output.save())
            for k in range(count):
                outs.append(model.layers[k].layer_output.save())
                added.append(model.layers[k].deepstack_output.save())
                entering.append(self.entering(model, k).input.save())
        for k in range(count):
            assert isinstance(added[k], ImageFeatures) and added[k].shape == (int(mask.sum()), model.hidden_size)
            assert torch.equal(added[k], merged[k])  # merger k's output, for block k
            assert torch.equal(entering[k][mask], outs[k][mask] + added[k])
            assert torch.equal(entering[k][~mask], outs[k][~mask])

    def test_reading_a_later_blocks_deepstack_alone(self, model):
        """Each block's read asks for its own occurrence of the call, not the next one the model reaches."""
        last = self.deepstack_count(model) - 1
        with model.trace(image_prompt(model), images=[IMAGE]):
            alone = model.layers[last].deepstack_output.save()
        with model.trace(image_prompt(model), images=[IMAGE]):
            merged = model.vision.deepstack_merger_list[last].output.save()
        assert torch.equal(alone, merged)

    def test_deepstack_writes_land(self, model):
        with model.trace(image_prompt(model), images=[IMAGE]):
            clean = model.logits.save()
        with model.trace(image_prompt(model), images=[IMAGE]):
            out = model.layers[0].layer_output.save()
            model.layers[0].deepstack_output[:] = 0
            entering = self.entering(model, 0).input.save()
            zeroed = model.logits.save()
        assert torch.equal(entering, out)
        assert not torch.allclose(zeroed, clean)
        with model.trace(image_prompt(model), images=[IMAGE]):
            mask = model.vision.image_token_mask.save()
            out = model.layers[0].layer_output.save()
            replacement = torch.ones_like(model.layers[0].deepstack_output)
            model.layers[0].deepstack_output = replacement
            entering = self.entering(model, 0).input.save()
        assert torch.equal(entering[mask], out[mask] + 1)

    def test_deepstack_is_unavailable_past_the_taps(self):
        """With more text blocks than tower taps, the blocks past them have no deepstack_output and the stream passes."""
        from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

        config = AutoConfig.from_pretrained(self.REPO)
        taps = len(config.vision_config.deepstack_visual_indexes)
        config.text_config.num_hidden_layers = taps + 2
        torch.manual_seed(0)
        module = AutoModelForImageTextToText.from_config(config, attn_implementation="eager", dtype=torch.float32).eval()
        model = StandardizedTransformer(module, processor=AutoProcessor.from_pretrained(self.REPO), task="image-text-to-text")
        support = model.support()["deepstack_output"]
        assert set(support) == {taps, taps + 1}
        assert all(f"blocks 0..{taps - 1} only" in reason for reason in support.values())
        with pytest.raises(Unavailable, match="deepstack features after blocks"):
            model.layers[taps].deepstack_output
        with model.trace(image_prompt(model), images=[IMAGE]):
            out = model.layers[taps].layer_output.save()
            entering = model.layers[taps + 1].input.save()
        assert torch.equal(entering, out)
