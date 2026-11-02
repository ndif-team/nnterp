"""Ministral 3 (text-only) on vLLM, against the transformers engine: vLLM's Mistral classes.

No small text-only Ministral-3 checkpoint is published (the released ones are
the vision-language wrapper, 15 GB), so the test runs hf-tiny-v2's tiny config
resized for vLLM, with the released checkpoints' rope.
"""

import os
import shutil
import tempfile

import torch
from huggingface_hub import snapshot_download
from vllm_suite import LLAMA_ROWS, VLLMFamilySuite

from nnterp.families.vllm import ministral3

#: The released checkpoints' rope: YaRN whose ``mscale`` and ``mscale_all_dim`` cancel, so it scales no attention.
ROPE = {
    "rope_type": "yarn", "type": "yarn", "rope_theta": 1000000.0, "factor": 16.0, "original_max_position_embeddings": 16384,
    "beta_fast": 32.0, "beta_slow": 1.0, "mscale": 1.0, "mscale_all_dim": 1.0, "llama_4_scaling_beta": 0.1,
}


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


class TestVLLMMinistral3(VLLMFamilySuite):
    REPO = _runnable_checkpoint(
        "hf-tiny-v2/tiny-random-Ministral3ForCausalLM", "ministral3", hidden_act="silu", hidden_size=256, num_attention_heads=8,
        num_key_value_heads=4, head_dim=32, intermediate_size=512, num_hidden_layers=4, vocab_size=131072,
        max_position_embeddings=262144, rope_parameters=ROPE,
    )  # the full vocabulary: the tekken tokenizer's ids reach past the tiny one's 99
    FAMILY = ministral3
    NATIVE = LLAMA_ROWS
    # vLLM's plain YaRN ignores ``mscale`` / ``mscale_all_dim`` and multiplies cos and sin by 0.1 * ln(16) + 1 = 1.277,
    # so every query and key comes out 1.277 times transformers' (the logits differed by 62% of their scale). Mistral's
    # own params.json says ``apply_scale: false``, which vLLM's Mistral-format loader turns into this switch; an
    # HF-format checkpoint has to be given it.
    ENGINE = {"hf_overrides": {"rope_parameters": {**ROPE, "apply_yarn_scaling": False}}}
