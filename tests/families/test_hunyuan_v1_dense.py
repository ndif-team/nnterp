"""Hunyuan dense V1, end to end."""

from suite import FamilySuite, LLAMA_ROWS

from nnterp.families import hunyuan_v1_dense


class TestHunyuanV1Dense(FamilySuite):
    REPO = "hf-tiny-v2/tiny-random-HunYuanDenseV1ForCausalLM"
    FAMILY = hunyuan_v1_dense
    NATIVE = LLAMA_ROWS

    def rotated_as_served(self, attn, name, rotated):
        """The queries and keys are normed after the rotary, by ``query_layernorm`` / ``key_layernorm``."""
        return (attn.query_layernorm if name == "attention_queries_prerope" else attn.key_layernorm)(rotated)
