"""GLM-4-MoE-Lite on vLLM, against the transformers engine: a fused block, DeepSeek's latent attention, a mixture of experts.

Two engines, so two files' worth of classes would share a card; each class
here is run in its own process (``pytest -k``): float32 on vLLM's ordinary
attention layer, which is where the values can be held to transformers' most
closely, and the engine's default, its latent-attention kernel, which only
runs in half precision.

The two engines do not build the same latent norms: transformers'
``q_a_layernorm`` and ``kv_a_layernorm`` take their own ``eps`` of 1e-6 and
vLLM's take the config's ``rms_norm_eps`` (1e-5 on GLM-4.7-Flash, where
DeepSeek's checkpoints say 1e-6 and the two agree). On this checkpoint that
alone moves block 0's queries by 8e-4 of their scale. The reference is built
with the config's ``eps`` on those norms, so what is held here is nnterp's
mapping, not that difference.
"""

import os

from vllm_suite import LLAMA_ROWS, VLLMFamilySuite

from nnterp.families.vllm import glm4_moe_lite

UNAVAILABLE = frozenset({"attention_mask"} | {
    f"self_attn.{value}" for value in ("attention_queries", "attention_keys", "attention_values", "attention_scores", "attention_probabilities", "attention_head_outputs")
})

REPO = "ngxson/GLM-4.7-Flash-small-test"


def latent_norms_at_config_eps(on):
    """Build transformers' latent norms with the config's ``rms_norm_eps``, as vLLM does, or with their own 1e-6 again."""
    from transformers import AutoConfig  # here, not at the top: imported at collection, after vLLM, the module segfaults
    from transformers.models.glm4_moe_lite import modeling_glm4_moe_lite

    norm = modeling_glm4_moe_lite.Glm4MoeLiteRMSNorm.__init__
    norm.__defaults__ = (AutoConfig.from_pretrained(REPO).rms_norm_eps if on else 1e-6,)


class TestVLLMGlm4MoeLite(VLLMFamilySuite):
    """float32, on vLLM's ordinary attention layer (``VLLM_MLA_DISABLE=1``)."""

    REPO = REPO
    MEMORY = 0.4
    FAMILY = glm4_moe_lite
    NATIVE = LLAMA_ROWS
    SERVED = ()
    UNAVAILABLE = UNAVAILABLE
    # 47 blocks of real weights: block 0's queries, keys and values agree with transformers' to 2e-7 and vLLM's
    # float32 kernel moves its head outputs by 1.2e-3; depth compounds that, so block 23's stream entering agrees to
    # 2.2e-3, the queries, keys and values its latent norms make of it to 1.4e-2, and its attention output to 1.5e-2.
    TOLERANCE = 2e-2

    @classmethod
    def setup_class(cls):
        latent_norms_at_config_eps(True)
        os.environ["VLLM_MLA_DISABLE"] = "1"   # read by the engine's worker, which inherits the environment

    @classmethod
    def teardown_class(cls):
        os.environ.pop("VLLM_MLA_DISABLE", None)
        latent_norms_at_config_eps(False)


class TestVLLMGlm4MoeLiteLatent(VLLMFamilySuite):
    """The default engine: vLLM's latent-attention kernel, bfloat16, held to the float32 reference at bfloat16's resolution."""

    REPO = REPO
    MEMORY = 0.4
    FAMILY = glm4_moe_lite
    NATIVE = LLAMA_ROWS
    SERVED = ()
    UNAVAILABLE = UNAVAILABLE
    DTYPE = "bfloat16"
    # bfloat16 against the float32 reference, compounded over 47 blocks: block 23's attention output is 6.9e-2 off.
    TOLERANCE = 1e-1

    @classmethod
    def setup_class(cls):
        latent_norms_at_config_eps(True)

    @classmethod
    def teardown_class(cls):
        latent_norms_at_config_eps(False)
