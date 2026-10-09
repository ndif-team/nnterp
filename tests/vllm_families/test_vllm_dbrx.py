"""DBRX on vLLM, against the transformers engine: the attention inside ``norm_attn_norm``, a mixture of experts for an MLP.

The checkpoint is config-patched: it says its rotary base the way DBRX's
checkpoints do, ``attn_config.rope_theta`` (500000), which vLLM reads and
transformers 5.15's ``DbrxConfig`` ignores, falling back to 10000. The copy
adds a top-level ``rope_parameters`` with the checkpoint's own base, so both
engines rotate alike (the weights and tokenizer are symlinked).

vLLM 0.27.1 cannot load any DBRX checkpoint: its mixture keeps the experts'
weights under ``ffn.experts.routed_experts``, and ``DbrxModel.load_weights``
still looks for them under ``ffn.experts`` (``KeyError: ...w13_weight``). The
class is skipped on a vLLM whose loader does not know the new place; with the
loader's names rewritten (``ffn.experts.mlp.`` to
``ffn.experts.routed_experts.mlp.``) every test passes.
"""

import inspect
import json
import os
import tempfile

import pytest
from huggingface_hub import snapshot_download
from vllm_suite import VLLMFamilySuite

from nnterp.families.vllm import dbrx


def _rope_checkpoint(repo="yujiepan/dbrx-tiny256-random"):
    """The tiny checkpoint with its ``attn_config.rope_theta`` also given as ``rope_parameters``; everything else symlinked."""
    snapshot = snapshot_download(repo, local_files_only=True)
    patched = os.path.join(tempfile.gettempdir(), f"nnterp-vllm-dbrx-rope-{os.path.basename(snapshot)}")
    os.makedirs(patched, exist_ok=True)
    for name in os.listdir(snapshot):
        target = os.path.join(patched, name)
        if name != "config.json" and not os.path.exists(target):
            os.symlink(os.path.realpath(os.path.join(snapshot, name)), target)
    config = json.load(open(os.path.join(snapshot, "config.json")))
    config["rope_parameters"] = {"rope_type": "default", "rope_theta": float(config["attn_config"]["rope_theta"])}
    json.dump(config, open(os.path.join(patched, "config.json"), "w"))
    return patched


def _loader_finds_the_experts():
    """Whether vLLM's DBRX loader names the experts where its mixture keeps them (``routed_experts`` since the MoE runner)."""
    from vllm.model_executor.models import dbrx as vllm_dbrx

    return "routed_experts" not in inspect.getsource(vllm_dbrx) or "routed_experts" in inspect.getsource(vllm_dbrx.DbrxModel.load_weights)


@pytest.mark.skipif(not _loader_finds_the_experts(), reason="vLLM's DbrxModel.load_weights looks for the experts under ffn.experts, not ffn.experts.routed_experts")
class TestVLLMDbrx(VLLMFamilySuite):
    REPO = _rope_checkpoint()
    FAMILY = dbrx
    NATIVE = {
        "embed_tokens": "transformer.wte",
        "layers": "transformer.blocks",
        "norm": "transformer.norm_f",
        "lm_head": "lm_head",
        "layers.0.self_attn": "transformer.blocks.0.norm_attn_norm.attn",
        "layers.0.mlp": "transformer.blocks.0.ffn",
    }
