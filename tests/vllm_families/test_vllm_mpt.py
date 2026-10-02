"""MPT on vLLM, against the transformers engine: ALiBi attention, a plain block with the positions first."""

from vllm_suite import VLLMFamilySuite

from nnterp.families.vllm import mpt


class TestVLLMMPT(VLLMFamilySuite):
    REPO = "anas-awadalla/mpt-7b"
    FAMILY = mpt
    NATIVE = {
        "embed_tokens": "transformer.wte",
        "layers": "transformer.blocks",
        "norm": "transformer.norm_f",
        "layers.0.self_attn": "transformer.blocks.0.attn",
        "layers.0.mlp": "transformer.blocks.0.ffn",
        "layers.0.input_layernorm": "transformer.blocks.0.norm_1",
        "layers.0.post_attention_layernorm": "transformer.blocks.0.norm_2",
    }
    MEMORY = 0.55
