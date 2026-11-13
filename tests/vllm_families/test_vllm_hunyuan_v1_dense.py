"""Hunyuan dense V1 on vLLM, against the transformers engine: a fused block that also returns the keys and values."""

from vllm_suite import LLAMA_ROWS, VLLMFamilySuite

from nnterp.families.vllm import hunyuan_v1_dense


class TestVLLMHunYuanDenseV1(VLLMFamilySuite):
    REPO = "tencent/Hunyuan-0.5B-Instruct"
    FAMILY = hunyuan_v1_dense
    NATIVE = LLAMA_ROWS
    MEMORY = 0.3

    def rotated_as_served(self, attn, name, rotated):
        """The queries and keys are normed per head after the rotary, by ``query_layernorm`` / ``key_layernorm``, tokens first."""
        norm = attn.query_layernorm if name == "attention_queries_prerope" else attn.key_layernorm
        return norm(rotated[0].transpose(0, 1).contiguous()).transpose(0, 1).unsqueeze(0)
