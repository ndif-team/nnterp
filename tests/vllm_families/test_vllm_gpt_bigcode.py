"""GPT-BigCode on vLLM's transformers backend, against the transformers engine: GPT-2's tree under ``model``, multi-query attention."""

from vllm_suite import VLLMFamilySuite

from nnterp.families.vllm import gpt_bigcode


class TestVLLMGPTBigCode(VLLMFamilySuite):
    REPO = "bigcode/tiny_starcoder_py"
    FAMILY = gpt_bigcode
    NATIVE = {
        "embed_tokens": "model.wte",
        "layers": "model.h",
        "norm": "model.ln_f",
        "lm_head": "lm_head",
        "layers.0.self_attn": "model.h.0.attn",
        "layers.0.mlp": "model.h.0.mlp",
        "layers.0.input_layernorm": "model.h.0.ln_1",
        "layers.0.post_attention_layernorm": "model.h.0.ln_2",
    }
