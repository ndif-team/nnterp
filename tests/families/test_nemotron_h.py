"""Nemotron-H, end to end: one mixer per block, a Mamba-2 mixer, attention or a mixture of experts."""

import glob
import importlib.util
import os
import tempfile

import pytest
import torch
from safetensors.torch import load_file, save_file
from ssd import StateSpaceChecks
from suite import LINEAR, VALUES, FamilySuite, PROMPT
from vision_suite import IMAGE, VisionSuite, image_prompt

from nnterp import StandardizedTransformer, Unavailable, route_kernels
from nnterp.components.vision import scatter_host
from nnterp.families import nemotron_h

KINDS = {"linear_attention": "linear_attn", "full_attention": "self_attn", "moe": "mlp", "mlp": "mlp"}


def _patched_checkpoint(repo="hf-tiny-v2/tiny-random-NemotronHForCausalLM"):
    """The tiny checkpoint with its embedding under the name transformers 5.17 loads.

    It stores ``backbone.embedding.weight``; the model's module is
    ``model.embeddings``, so a load would leave the embedding randomly
    initialized and two loads would differ. The weights are rewritten with
    the key renamed; everything else is symlinked. The family needs nothing.
    """
    snapshot = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{repo.replace('/', '--')}/snapshots/*"))[0]
    patched = tempfile.mkdtemp(prefix="nemotron_h-")
    for name in os.listdir(snapshot):
        if name != "model.safetensors":
            os.symlink(os.path.realpath(os.path.join(snapshot, name)), os.path.join(patched, name))
    weights = load_file(os.path.join(snapshot, "model.safetensors"))
    weights["backbone.embeddings.weight"] = weights.pop("backbone.embedding.weight")
    save_file(weights, os.path.join(patched, "model.safetensors"), metadata={"format": "pt"})
    return patched


class TestNemotronH(StateSpaceChecks, FamilySuite):
    REPO = _patched_checkpoint()
    FAMILY = nemotron_h
    ATTENTION_NORM = "norm"                # the block norm keeps its native name
    # ``layers_block_type``: Mamba-2, MoE, Mamba-2, attention, MoE
    NATIVE = {
        "embed_tokens": "model.embeddings",
        "layers": "model.layers",
        "layers.0.linear_attn": "model.layers.0.mixer",
        "layers.1.mlp": "model.layers.1.mixer",
        "layers.3.self_attn": "model.layers.3.mixer",
        "norm": "model.norm_f",
        "lm_head": "lm_head",
    }
    MLP_WIDTH_KEY = "moe_intermediate_size"
    EXPECTED_UNAVAILABLE = {
        **{name: "no self_attn module" for name in VALUES if name.startswith("self_attn.")},
        **{f"linear_attn.{name}": "no linear_attn module" for name in LINEAR if name not in ("state", "states")},
        "linear_attn.state": "",   # missing off the Mamba-2 blocks, and unavailable on them
        "linear_attn.states": "",
        "linear_attn.set_state_after": "",
        "mlp.mlp_output": "no mlp module",
    }

    def test_every_layer_is_renamed(self, model):
        """Each block's ``mixer`` answers to the standard name of what it is, and to nothing else."""
        for layer, kind in zip(model.layers, model.config.layers_block_type):
            name = KINDS[kind]
            assert getattr(layer, name) is layer.mixer, (layer.path, kind)
            assert all(getattr(layer, other, None) is None for other in set(KINDS.values()) - {name}), (layer.path, kind)
            assert layer._aliases == {name: "mixer"}

    def test_support_is_per_block(self, model):
        support = model.support()
        kinds = model.config.layers_block_type
        assert set(support["self_attn.attention_probabilities"]) == {i for i, k in enumerate(kinds) if k != "full_attention"}
        assert set(support["linear_attn.state_output"]) == {i for i, k in enumerate(kinds) if k != "linear_attention"}
        assert set(support["mlp.mlp_output"]) == {i for i, k in enumerate(kinds) if k not in ("moe", "mlp")}
        assert support["layer_output"] is None

    def test_one_contribution_per_block(self, model):
        """``input + <the block's one sublayer> == layer_output`` on every block."""
        parts = {}
        with model.trace(PROMPT):
            for i, (layer, kind) in enumerate(zip(model.layers, model.config.layers_block_type)):
                host = getattr(layer, KINDS[kind])
                value = "mlp_output" if KINDS[kind] == "mlp" else "attention_output"
                parts[i] = (layer.input.save(), getattr(host, value).save(), layer.layer_output.save())
        for i, (x, added, out) in parts.items():
            torch.testing.assert_close(x + added, out, msg=f"layer {i}")

    def test_aliases_survive_dispatch(self):
        """The block binds its aliases from the mixer's class at build and again when real weights arrive."""
        from nnterp import StandardizedTransformer

        lazy = StandardizedTransformer(self.REPO)
        assert lazy.layers[3].self_attn is lazy.layers[3].mixer
        with lazy.trace(PROMPT):
            out = lazy.layers[3].self_attn.attention_output.save()
        assert lazy.layers[3].self_attn is lazy.layers[3].mixer and out.shape[-1] == lazy.hidden_size


OMNI_REPO = "hf-tiny-v2/tiny-random-NemotronH_Omni_Reasoning_V3"


def omni_processor():
    """The tiny Omni checkpoint ships no processor files: the processor built from its tokenizer, with an image processor
    that keeps the grids small (16 to 24 patches; the tiny tower interpolates its 4x4 position table)."""
    from transformers import AutoTokenizer, ParakeetFeatureExtractor
    from transformers.models.nemotron_h_omni import (
        image_processing_nemotron_h_omni as image, processing_nemotron_h_omni as processing,
        video_processing_nemotron_h_omni as video,
    )

    tokenizer = AutoTokenizer.from_pretrained(OMNI_REPO)
    return processing.NemotronH_Omni_Reasoning_V3Processor(
        image_processor=image.NemotronH_Omni_Reasoning_V3ImageProcessor(min_num_patches=16, max_num_patches=24),
        video_processor=video.NemotronH_Omni_Reasoning_V3VideoProcessor(), feature_extractor=ParakeetFeatureExtractor(),
        tokenizer=tokenizer, chat_template=tokenizer.chat_template,
    )


@pytest.mark.skipif(
    importlib.util.find_spec("transformers.models.nemotron_h_omni") is None, reason="NemotronH Omni is in transformers 5.18 and later"
)
class TestNemotronHOmniVision(VisionSuite):
    """NemotronH Omni's RADIO tower (packed behind its CLS and register tokens, layer-scaled blocks), its projector after
    the pixel shuffle, and the root's own scatter."""

    REPO = OMNI_REPO
    FAMILY = nemotron_h
    TEXT_REPO = TestNemotronH.REPO
    VISION_NATIVE = {
        "vision": "vision_model",
        "vision.layers": "vision_model.encoder.layer",
        "vision.patch_embed": "vision_model.embeddings.patch_projection",
        "vision.layers.0.self_attn": "vision_model.encoder.layer.0.attention",
        "vision.layers.0.mlp": "vision_model.encoder.layer.0.mlp",
        "vision.layers.0.input_layernorm": "vision_model.encoder.layer.0.norm1",
        "vision.layers.0.post_attention_layernorm": "vision_model.encoder.layer.0.norm2",
        "projector": "multi_modal_projector",
    }
    EXPECTED_VISION_UNAVAILABLE = {
        f"self_attn.{name}": "one per image" for name in ("attention_scores", "attention_probabilities", "attention_head_outputs")
    }

    @pytest.fixture(scope="class", autouse=True)
    def torch_kernels(self):
        route_kernels(nemotron_h, "torch")
        yield
        route_kernels(nemotron_h, "default")

    @pytest.fixture(scope="class")
    def model(self, torch_kernels):
        model = StandardizedTransformer(
            OMNI_REPO, processor=omni_processor(), task="image-text-to-text", dispatch=True, attn_implementation="eager",
            dtype=torch.float32,
        )
        model.config.image_token_id = model._module.image_token_id = model.processor.image_token_id  # unset on the tiny config
        return model

    def patches_of(self, model, images):
        """Each image's grid of patches (``image_grid_hw``), summed; the CLS and register tokens are not patches."""
        grid = model.processor(text=image_prompt(model, images=len(images)), images=images, return_tensors="pt")["image_grid_hw"]
        return int(grid.prod(-1).sum())

    def test_the_root_is_the_scatter_host_and_the_tower_is_radio(self, model):
        assert scatter_host(model) == ("", model) and nemotron_h.ROOT_SCATTER == "inputs_embeds_masked_scatter_0"
        assert isinstance(model.vision, nemotron_h.RadioVision)
        assert all(isinstance(layer.self_attn, nemotron_h.RadioAttention) for layer in model.vision.layers)
        with pytest.raises(Unavailable, match="any resolution"):
            model.vision.image_size

    def test_contributions_are_the_layer_scales(self, model):
        layer = model.vision.layers[0]
        with model.trace(image_prompt(model), images=[IMAGE]):
            raw_attn = layer.self_attn.output[0].save()
            attn = layer.self_attn.attention_output.save()
            scaled_attn = layer.layer_scale1.output.save()
            mlp = layer.mlp.mlp_output.save()
            scaled_mlp = layer.layer_scale2.output.save()
        assert torch.equal(attn, scaled_attn) and torch.equal(mlp, scaled_mlp)
        torch.testing.assert_close(attn, raw_attn * layer.layer_scale1._module.lambda1)

    def test_the_tokens_are_the_prefix_then_the_patches(self, model, clean):
        """Each image's row holds ``num_cls_tokens + num_registers`` prefix tokens before its patches."""
        config = model.config.vision_config
        with model.trace(image_prompt(model), images=[IMAGE]):
            queries = model.vision.layers[0].self_attn.attention_queries.save()
            out = model.vision.layers[0].layer_output.save()
        prefix = config.num_cls_tokens + config.num_registers
        assert out.shape == (1, prefix + self.patches_of(model, [IMAGE]), model.vision.hidden_size)
        assert queries.shape == (1, model.vision.num_heads, out.shape[1], model.vision.head_dim)

    def test_the_text_names_bind_through_language_model(self, model):
        """The whole Nemotron-H model sits at ``language_model``: the text names and the mixer names bind through it."""
        for standard, native in (("embed_tokens", "model.embeddings"), ("layers", "model.layers"), ("norm", "model.norm_f"), ("lm_head", "lm_head")):
            assert model.get(standard)._module is model.get(f"language_model.{native}")._module, standard
        kinds = model.config.text_config.layers_block_type
        for i, kind in enumerate(kinds):
            assert model.layers[i].get(KINDS[kind])._module is model.get(f"language_model.model.layers.{i}.mixer")._module
        with model.trace(image_prompt(model), images=[IMAGE]):
            stream = model.layers[1].input.save()
            added = model.layers[1].mlp.mlp_output.save()
            out = model.layers[1].layer_output.save()
        torch.testing.assert_close(stream + added, out)
