"""GPT-J on vLLM, against the transformers engine: a parallel block, a head with a bias."""

from vllm_suite import VLLMFamilySuite

from nnterp.families.vllm import gptj


class TestVLLMGPTJ(VLLMFamilySuite):
    REPO = "EleutherAI/gpt-j-6b"
    FAMILY = gptj
    NATIVE = {
        "embed_tokens": "transformer.wte",
        "layers": "transformer.h",
        "norm": "transformer.ln_f",
        "lm_head": "lm_head",
        "layers.0.self_attn": "transformer.h.0.attn",
        "layers.0.mlp": "transformer.h.0.mlp",
        "layers.0.input_layernorm": "transformer.h.0.ln_1",
    }
    MEMORY = 0.5
