"""Falcon-7B on vLLM, against the transformers engine: a parallel block whose sublayers return ``(output, bias)``."""

from vllm_suite import VLLMFamilySuite

from nnterp.families.vllm import falcon


class TestVLLMFalcon(VLLMFamilySuite):
    REPO = "tiiuae/falcon-7b"
    FAMILY = falcon
    NATIVE = {
        "embed_tokens": "transformer.word_embeddings",
        "layers": "transformer.h",
        "norm": "transformer.ln_f",
        "lm_head": "lm_head",
        "layers.0.self_attn": "transformer.h.0.self_attention",
        "layers.0.mlp": "transformer.h.0.mlp",
    }
    MEMORY = 0.55
