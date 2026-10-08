"""Qwen2.5-VL's language model on vLLM, against the transformers engine: the VL wrapper run as a language model, M-RoPE positions."""

import torch
from vllm_suite import VLLMFamilySuite

from nnterp.families.vllm import qwen2_5_vl_text

PREFIX = "language_model.model"


class TestVLLMQwen2_5VLText(VLLMFamilySuite):
    REPO = "Qwen/Qwen2.5-VL-3B-Instruct"
    FAMILY = qwen2_5_vl_text
    NATIVE = {
        "embed_tokens": f"{PREFIX}.embed_tokens",
        "layers": f"{PREFIX}.layers",
        "norm": f"{PREFIX}.norm",
        "lm_head": "language_model.lm_head",
        "layers.0.self_attn": f"{PREFIX}.layers.0.self_attn",
        "layers.0.mlp": f"{PREFIX}.layers.0.mlp",
    }
    MEMORY = 0.3
    ENGINE = {"language_model_only": True}   # text-only: the multimodal mode runs on inputs_embeds, with no ids
    REFERENCE = {"task": "image-text-to-text"}   # no causal-LM class: transformers loads the wrapper

    def test_text_positions_are_one_dimensional(self, model, reference):
        """M-RoPE's three position rows are the same 1-D positions on a text-only prompt, so the rotation is plain RoPE's."""
        with self.run(model, reference):
            positions = model.layers[0].inputs[0][0].cpu().save()
        tokens = len(reference["ids"])
        assert positions.shape == (3, tokens)
        assert torch.equal(positions, torch.arange(tokens).expand(3, tokens).to(positions))
