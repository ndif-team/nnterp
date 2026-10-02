"""GPT-2 on vLLM, against the transformers engine: a block that is not fused."""

from vllm_suite import VLLMFamilySuite

from nnterp.families.vllm import gpt2


class TestVLLMGPT2(VLLMFamilySuite):
    REPO = "openai-community/gpt2"
    FAMILY = gpt2
    NATIVE = {
        "embed_tokens": "transformer.wte",
        "layers": "transformer.h",
        "norm": "transformer.ln_f",
        "lm_head": "lm_head",
        "layers.0.self_attn": "transformer.h.0.attn",
        "layers.0.mlp": "transformer.h.0.mlp",
        "layers.0.input_layernorm": "transformer.h.0.ln_1",
        "layers.0.post_attention_layernorm": "transformer.h.0.ln_2",
    }
