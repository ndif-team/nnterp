"""Kimi K2's language model on vLLM, inside a Kimi-K2.5 checkpoint, against the transformers engine: DeepSeek-V3's text model."""

import json
import os
import tempfile

from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from vllm_suite import VLLMFamilySuite

from nnterp.families.vllm import deepseek_v3, kimi_k2

UNAVAILABLE = frozenset({"attention_mask"} | {
    f"self_attn.{value}" for value in ("attention_queries", "attention_keys", "attention_values", "attention_scores", "attention_probabilities", "attention_head_outputs")
})
PREFIX = "language_model.model"


def _checkpoint(repo="hf-tiny-v2/tiny-random-Kimi_K25ForConditionalGeneration"):
    """The tiny Kimi-K2.5 checkpoint with the released configs' text model type and routing, as the transformers test has it.

    The rewrite sets the text model type to ``kimi_k2``, one expert group
    and two experts per token (see ``tests/families/test_kimi_k2.py``), and
    names the router K2's released configs name and transformers assumes
    (``topk_method="noaux_tc"``, ``scoring_func="sigmoid"``), which vLLM reads
    and would otherwise take as a softmax router with no score correction. The
    weights are renamed too: the tiny checkpoint names the text blocks
    ``language_model.model.blocks`` and the projector's norm
    ``model.mm_projector.pre_norm``, which transformers renames on load and
    vLLM does not, so they are saved as the released checkpoints name them,
    ``language_model.model.layers`` and ``mm_projector.pre_norm``. The rest is symlinked. Written once per snapshot.
    """
    snapshot = snapshot_download(repo, local_files_only=True)
    patched = os.path.join(tempfile.gettempdir(), f"nnterp-vllm-kimi-k2-v3-{os.path.basename(snapshot)}")
    if os.path.exists(os.path.join(patched, "config.json")):
        return patched
    os.makedirs(patched, exist_ok=True)
    for name in os.listdir(snapshot):
        if name not in ("config.json", "model.safetensors") and not os.path.exists(os.path.join(patched, name)):
            os.symlink(os.path.realpath(os.path.join(snapshot, name)), os.path.join(patched, name))
    weights = load_file(os.path.join(snapshot, "model.safetensors"))
    renamed = {"language_model.model.blocks.": "language_model.model.layers.", "model.mm_projector.": "mm_projector."}
    for old, new in renamed.items():
        weights = {new + key.removeprefix(old) if key.startswith(old) else key: value for key, value in weights.items()}
    save_file(weights,
              os.path.join(patched, "model.safetensors"), metadata={"format": "pt"})
    config = json.load(open(os.path.join(snapshot, "config.json")))
    config["text_config"].update(model_type="kimi_k2", n_group=1, num_experts_per_tok=2, topk_method="noaux_tc", scoring_func="sigmoid")
    partial = os.path.join(patched, f"config.json.{os.getpid()}")
    json.dump(config, open(partial, "w"))
    os.replace(partial, os.path.join(patched, "config.json"))
    return patched


class TestVLLMKimiK2(VLLMFamilySuite):
    """float32, on vLLM's ordinary attention layer (``VLLM_MLA_DISABLE=1``), as `deepseek_v3`'s first class."""

    REPO = _checkpoint()
    FAMILY = kimi_k2
    NATIVE = {
        "embed_tokens": f"{PREFIX}.embed_tokens",
        "layers": f"{PREFIX}.layers",
        "norm": f"{PREFIX}.norm",
        "lm_head": "language_model.lm_head",
        "layers.0.self_attn": f"{PREFIX}.layers.0.self_attn",
        "layers.0.mlp": f"{PREFIX}.layers.0.mlp",
    }
    SERVED = ()
    UNAVAILABLE = UNAVAILABLE
    ENGINE = {
        "language_model_only": True,   # text-only: the multimodal mode runs on inputs_embeds, with no ids
        "hf_overrides": {"architectures": ["KimiK25ForConditionalGeneration"]},   # vLLM's name for transformers' Kimi_K25ForConditionalGeneration
    }

    REFERENCE = {"task": "image-text-to-text"}   # transformers loads the wrapper; no causal-LM class takes its config

    @classmethod
    def setup_class(cls):
        os.environ["VLLM_MLA_DISABLE"] = "1"   # read by the engine's worker, which inherits the environment

    @classmethod
    def teardown_class(cls):
        os.environ.pop("VLLM_MLA_DISABLE", None)

    def test_the_family_is_deepseek_v3s(self, model):
        assert model.config.text_config.model_type == "kimi_k2"
        assert kimi_k2.ENVOYS is deepseek_v3.ENVOYS
        assert type(model.layers[0].mlp) is deepseek_v3.Mlp and type(model.layers[-1].mlp) is deepseek_v3.Mlp
