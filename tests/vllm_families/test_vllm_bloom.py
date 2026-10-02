"""BLOOM on vLLM, against the transformers engine: ALiBi attention, a plain block with the positions first."""

from vllm_suite import VLLMFamilySuite

from nnterp.families.vllm import bloom


class TestVLLMBloom(VLLMFamilySuite):
    REPO = "bigscience/bloom-560m"
    FAMILY = bloom
    NATIVE = {
        "embed_tokens": "transformer.word_embeddings",
        "layers": "transformer.h",
        "norm": "transformer.ln_f",
        "lm_head": "lm_head",
        "layers.0.self_attn": "transformer.h.0.self_attention",
        "layers.0.mlp": "transformer.h.0.mlp",
    }
