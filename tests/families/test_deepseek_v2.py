"""DeepSeek-V2, end to end: multi-head latent attention."""

from suite import FamilySuite, LLAMA_ROWS, built, draw_routers, moe_host, patched_checkpoint

from nnterp.components import Moe
from nnterp.families import deepseek_v2

REPO = "hf-tiny-v2/tiny-random-DeepseekV2ForCausalLM"


class TestDeepseekV2(FamilySuite):
    REPO = REPO
    FAMILY = deepseek_v2
    NATIVE = LLAMA_ROWS
    KV_HEADS_EXPANDED = True  # latent attention projects keys and values for every head
    EXPECTED_UNAVAILABLE = {"self_attn.value_weight": "latent attention"}

    def test_latent_attention_widths(self, model):
        assert model.qk_head_dim == model.config.qk_nope_head_dim + model.config.qk_rope_head_dim
        assert model.head_dim == model.config.v_head_dim


class TestDeepseekV2Released(TestDeepseekV2):
    """The released checkpoints' branches, which the tiny checkpoint takes none of: DeepSeek-V2-Lite's queries
    straight from ``q_proj`` (``q_lora_rank`` null), a dense first block (``first_k_dense_replace`` 1) and YaRN with
    ``mscale`` 0.707; the 236B models' ``group_limited_greedy`` routing (the best ``topk_group`` groups' experts) and
    ``routed_scaling_factor`` 16. The tiny config otherwise, weights drawn for it, the routers drawn too."""

    REPO = built(
        patched_checkpoint, REPO, init=draw_routers, q_lora_rank=None, first_k_dense_replace=1,
        rope_parameters={"rope_type": "yarn", "factor": 40.0, "mscale": 0.707, "mscale_all_dim": 0.707, "beta_fast": 32,
                         "beta_slow": 1, "original_max_position_embeddings": 4096},
        topk_method="group_limited_greedy", n_group=4, topk_group=2, routed_scaling_factor=16.0,
    )

    def test_the_branches_are_taken(self, model):
        attn = model.layers[0].self_attn._module
        assert attn.q_proj is not None and getattr(attn, "q_a_proj", None) is None
        assert model._module.model.rotary_emb.rope_type == "yarn"
        assert type(model.layers[0].mlp) is deepseek_v2.Mlp and not isinstance(model.layers[0].mlp, Moe)
        gate = moe_host(model).router._module
        assert (gate.topk_method, gate.num_group, gate.topk_group, gate.routed_scaling_factor) == ("group_limited_greedy", 4, 2, 16.0)
