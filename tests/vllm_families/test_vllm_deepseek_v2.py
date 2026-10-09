"""DeepSeek-V2 on vLLM, against the transformers engine: a fused block, latent attention, a mixture of experts.

Two engines, so two files' worth of classes would share a card; each class
here is run in its own process (``pytest -k``): float32 on vLLM's ordinary
attention layer, which is where the values can be held to transformers' at
the suite's tolerance, and the engine's default, its latent-attention kernel,
which only runs in half precision.
"""

import os

from vllm_suite import LLAMA_ROWS, VLLMFamilySuite

from nnterp.families.vllm import deepseek_v2

UNAVAILABLE = frozenset({"attention_mask"} | {
    f"self_attn.{value}" for value in ("attention_queries", "attention_keys", "attention_values", "attention_scores", "attention_probabilities", "attention_head_outputs")
})


class TestVLLMDeepseekV2(VLLMFamilySuite):
    """float32, on vLLM's ordinary attention layer (``VLLM_MLA_DISABLE=1``)."""

    REPO = "hmellor/tiny-random-DeepseekV2ForCausalLM"
    FAMILY = deepseek_v2
    NATIVE = LLAMA_ROWS
    SERVED = ()
    UNAVAILABLE = UNAVAILABLE

    @classmethod
    def setup_class(cls):
        os.environ["VLLM_MLA_DISABLE"] = "1"   # read by the engine's worker, which inherits the environment

    @classmethod
    def teardown_class(cls):
        os.environ.pop("VLLM_MLA_DISABLE", None)


class TestVLLMDeepseekV2Latent(VLLMFamilySuite):
    """The default engine: vLLM's latent-attention kernel, bfloat16, held to the float32 reference at bfloat16's resolution."""

    REPO = "hmellor/tiny-random-DeepseekV2ForCausalLM"
    FAMILY = deepseek_v2
    NATIVE = LLAMA_ROWS
    SERVED = ()
    UNAVAILABLE = UNAVAILABLE
    DTYPE = "bfloat16"
    TOLERANCE = 6e-2
