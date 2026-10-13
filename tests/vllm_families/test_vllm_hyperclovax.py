"""HyperCLOVA X on vLLM, against the transformers engine: a sandwich block returning a pair, Granite's multipliers.

The released ``HyperCLOVAXForCausalLM`` checkpoints are 14B and up (the small
SEED ones are Llama), so the test runs hf-tiny-v2's tiny config resized for
vLLM, with every multiplier away from one and the post-norms on.
"""

import os
import shutil
import tempfile

import torch
from huggingface_hub import snapshot_download
from vllm_suite import LLAMA_ROWS, VLLMFamilySuite

from nnterp.families.vllm import hyperclovax


def _runnable_checkpoint(repo, tag, **changes):
    """The tiny checkpoint resized so vLLM runs it, with seeded weights; written once per snapshot.

    hf-tiny-v2's tiny checkpoints have 16-wide heads and ``hidden_act: gelu``;
    vLLM's kernels need heads at least 32 wide and its MLP takes SiLU only. The
    config is the tiny one with ``changes``; the weights a seeded draw with a
    larger spread than the initialisation's, so the logits have a clear
    winner; the tokenizer is the tiny one's.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    snapshot = snapshot_download(repo, local_files_only=True)
    out = os.path.join(tempfile.gettempdir(), f"nnterp-vllm-{tag}-{os.path.basename(snapshot)}")
    if os.path.exists(os.path.join(out, "model.safetensors")):
        return out
    config = AutoConfig.from_pretrained(snapshot, **changes)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.dim() == 2:
                parameter.normal_(0, 0.08)
    partial = f"{out}.{os.getpid()}"
    model.save_pretrained(partial)
    for name in os.listdir(snapshot):
        if "token" in name or name in ("tekken.json", "chat_template.jinja"):
            shutil.copy(os.path.join(snapshot, name), os.path.join(partial, name))
    os.replace(partial, out)
    return out


class TestVLLMHyperCLOVAX(VLLMFamilySuite):
    REPO = _runnable_checkpoint(
        "hf-tiny-v2/tiny-random-HyperCLOVAXForCausalLM", "hyperclovax", hidden_act="silu", hidden_size=256, num_attention_heads=8,
        num_key_value_heads=4, head_dim=32, intermediate_size=512, num_hidden_layers=4, residual_multiplier=0.5,
        embedding_multiplier=3.0, logits_scaling=0.7, attention_multiplier=0.125, use_post_norm=True,
    )
    FAMILY = hyperclovax
    NATIVE = LLAMA_ROWS

    def test_first_block_input_is_the_multiplied_embeddings(self, model, reference):
        with self.run(model, reference):
            embeddings = model.token_embeddings.cpu().save()
            stream = model.layers[0].layer_input.cpu().save()
        torch.testing.assert_close(embeddings * model.config.embedding_multiplier, stream)

    def test_contributions_are_the_scaled_post_norms(self, model, reference):
        layer = model.layers[reference["middle"]]
        with self.run(model, reference):
            post_attn = layer.post_norm1.output.clone().cpu().save()
            attn = layer.self_attn.attention_output.cpu().save()
            post_mlp = layer.post_norm2.output.clone().cpu().save()
            mlp = layer.mlp.mlp_output.cpu().save()
        multiplier = model.config.residual_multiplier
        torch.testing.assert_close(attn, post_attn.unsqueeze(0) * multiplier)
        torch.testing.assert_close(mlp, post_mlp.unsqueeze(0) * multiplier)
