"""Phi-1.5 on vLLM, against the transformers engine: a parallel block, a head with a bias."""

from vllm_suite import VLLMFamilySuite

from nnterp.families.vllm import phi


class TestVLLMPhi(VLLMFamilySuite):
    REPO = "microsoft/phi-1_5"
    FAMILY = phi
    NATIVE = {
        "embed_tokens": "model.embed_tokens",
        "layers": "model.layers",
        "norm": "model.final_layernorm",
        "lm_head": "lm_head",
        "layers.0.self_attn": "model.layers.0.self_attn",
        "layers.0.mlp": "model.layers.0.mlp",
    }
