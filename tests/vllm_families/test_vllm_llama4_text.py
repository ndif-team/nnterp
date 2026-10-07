"""Llama 4, text, on vLLM, against the transformers engine: dense and MoE blocks, RoPE and NoPE blocks, chunked attention.

The pinned checkpoint, ``yujiepan/llama-4-tiny-random``, is a
``Llama4ForConditionalGeneration`` (vLLM would run it through its multimodal
model), so the test runs a text-only copy of it: its ``text_config`` as the
config, with ``architectures`` ``Llama4ForCausalLM`` and
``attn_temperature_tuning`` as the bool transformers validates, and the
language model's tensors with the ``language_model.`` prefix dropped (the
tokenizer is symlinked). Four blocks: dense MLPs on 0 and 2, mixtures on 1
and 3, block 3 NoPE, the others chunked (``attention_chunk_size`` 128).
"""

import json
import os
import tempfile

import nnsight
import pytest
import torch
from huggingface_hub import snapshot_download
from vllm_suite import LLAMA_ROWS, VLLMFamilySuite

from nnterp import StandardizedTransformer, Unavailable
from nnterp.families.vllm import llama4_text


def _text_checkpoint(repo="yujiepan/llama-4-tiny-random"):
    """The wrapper checkpoint's language model alone, as a ``Llama4ForCausalLM`` checkpoint."""
    from safetensors.torch import load_file, save_file

    snapshot = snapshot_download(repo, local_files_only=True)
    patched = os.path.join(tempfile.gettempdir(), f"nnterp-vllm-llama4-text-{os.path.basename(snapshot)}")
    if os.path.exists(os.path.join(patched, "model.safetensors")):
        return patched
    os.makedirs(patched, exist_ok=True)
    config = json.load(open(os.path.join(snapshot, "config.json")))["text_config"]
    config.update(architectures=["Llama4ForCausalLM"], attn_temperature_tuning=bool(config["attn_temperature_tuning"]))
    json.dump(config, open(os.path.join(patched, "config.json"), "w"))
    tensors = load_file(os.path.join(snapshot, "model.safetensors"))
    prefix = "language_model."
    save_file({name[len(prefix):]: tensor for name, tensor in tensors.items() if name.startswith(prefix)}, os.path.join(patched, "model.safetensors"))
    for name in os.listdir(snapshot):
        if name.startswith(("tokenizer", "special_tokens")) or name == "generation_config.json":
            target = os.path.join(patched, name)
            if not os.path.exists(target):
                os.symlink(os.path.realpath(os.path.join(snapshot, name)), target)
    return patched


class TestVLLMLlama4Text(VLLMFamilySuite):
    REPO = _text_checkpoint()
    FAMILY = llama4_text
    NATIVE = {**LLAMA_ROWS, "layers.0.mlp": "model.layers.0.feed_forward"}
    ENGINE = {"override_generation_config": {"attn_temperature_tuning": True}}  # as the config says, which transformers follows

    def test_every_kind_of_block(self, model):
        """Dense and MoE blocks, RoPE and NoPE blocks, chunked and full attention; a mixture's shared expert serves no contribution."""
        config = model.config
        assert [type(layer.mlp._module).__name__ for layer in model.layers] == ["LlamaMLP", "Llama4MoE", "LlamaMLP", "Llama4MoE"]
        assert [type(layer.self_attn.attn._module).__name__ for layer in model.layers] == ["ChunkedLocalAttention"] * 3 + ["Attention"]
        assert config.no_rope_layers == [1, 1, 1, 0] and config.attention_chunk_size == 128
        shared = model.layers[1].mlp.shared_experts
        assert type(shared) is llama4_text.Mlp
        with pytest.raises(Unavailable, match="vLLM"):
            shared.mlp_output

    def test_pattern_is_zero_across_chunks(self, model, reference):
        """Past ``attention_chunk_size`` tokens a chunked block's recomputed pattern is transformers': nothing across the boundary."""
        ids = (reference["ids"] * 20)[:160]
        with model.trace(ids, temperature=0.0, max_tokens=1):
            got = nnsight.save({i: model.layers[i].self_attn.attention_probabilities.cpu() for i in (0, 3)})
        hf = StandardizedTransformer(self.REPO, device_map="cuda", dtype=torch.float32, attn_implementation="eager")
        with torch.no_grad(), hf.trace(torch.tensor([ids])):
            want = nnsight.save({i: hf.layers[i].self_attn.attention_probabilities.cpu() for i in (0, 3)})
        del hf
        assert got[0][..., 128:, :128].abs().max() == 0 and got[3][..., 128:, :128].abs().max() > 0
        for i in (0, 3):
            torch.testing.assert_close(got[i], want[i], atol=self.KERNEL_TOLERANCE, rtol=0)

    def test_query_channels_are_transformers(self, model, reference):
        """vLLM permutes each head's query channels at load; one channel zeroed is the same edit on both engines."""
        middle = reference["middle"]
        with self.run(model, reference):
            model.layers[middle].self_attn.attention_queries[..., 1] = 0
            got = model.logits.cpu().save()
        hf = StandardizedTransformer(self.REPO, device_map="cuda", dtype=torch.float32, attn_implementation="eager")
        with torch.no_grad(), hf.trace(torch.tensor([reference["ids"]])):
            hf.layers[middle].self_attn.attention_queries[..., 1] = 0
            want = hf.logits[:, -1:].cpu().save()
        del hf
        assert not torch.allclose(got, reference["logits"])
        torch.testing.assert_close(got, want, atol=self.TOLERANCE * want.abs().max().item(), rtol=0)
