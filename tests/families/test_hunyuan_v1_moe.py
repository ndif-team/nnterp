"""Hunyuan MoE V1, end to end."""

from suite import FamilySuite, LLAMA_ROWS, PROMPT, built, patched_checkpoint

from nnterp.families import hunyuan_v1_moe


class TestHunyuanV1Moe(FamilySuite):
    REPO = "hf-tiny-v2/tiny-random-HunYuanMoEV1ForCausalLM"
    FAMILY = hunyuan_v1_moe
    NATIVE = LLAMA_ROWS

    def rotated_as_served(self, attn, name, rotated, positions):
        """The queries and keys are normed after the rotary, by ``query_layernorm`` / ``key_layernorm``."""
        return (attn.query_layernorm if name == "attention_queries_prerope" else attn.key_layernorm)(rotated)


class TestHunyuanV1MoeA13B(TestHunyuanV1Moe):
    """Hunyuan-A13B's layout: ``moe_topk`` a list (each block its own top-k: 2 and 3 here), the ``dynamic`` rotary
    with ``alpha`` 1000 (``ntk_alpha``), tied embeddings. The tiny checkpoint's weights."""

    REPO = built(
        patched_checkpoint, TestHunyuanV1Moe.REPO, moe_topk=[2, 3],
        rope_parameters={"rope_type": "dynamic", "alpha": 1000.0, "factor": 1.0}, tie_word_embeddings=True,
    )

    def test_each_block_routes_its_own_top_k(self, model):
        assert [layer.mlp.top_k for layer in model.layers] == [2, 3]
        with model.trace(PROMPT):
            first = model.layers[0].mlp.expert_indices.save()
        with model.trace(PROMPT):
            second = model.layers[1].mlp.expert_indices.save()
        assert (first.shape[-1], second.shape[-1]) == (2, 3)
