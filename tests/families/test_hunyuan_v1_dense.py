"""Hunyuan dense V1, end to end."""

from suite import FamilySuite, LLAMA_ROWS, built, patched_checkpoint

from nnterp.families import hunyuan_v1_dense


class TestHunyuanV1Dense(FamilySuite):
    REPO = "hf-tiny-v2/tiny-random-HunYuanDenseV1ForCausalLM"
    FAMILY = hunyuan_v1_dense
    NATIVE = LLAMA_ROWS

    def rotated_as_served(self, attn, name, rotated, positions):
        """The queries and keys are normed after the rotary, by ``query_layernorm`` / ``key_layernorm``."""
        return (attn.query_layernorm if name == "attention_queries_prerope" else attn.key_layernorm)(rotated)


class TestHunyuanV1DenseReleased(TestHunyuanV1Dense):
    """The released Hunyuan dense models' rotary: ``dynamic`` with ``alpha`` 1000, which transformers builds as
    ``ntk_alpha`` (a fixed larger base), and tied embeddings. The tiny checkpoint's weights."""

    REPO = built(
        patched_checkpoint, TestHunyuanV1Dense.REPO, rope_parameters={"rope_type": "dynamic", "alpha": 1000.0, "factor": 1.0},
        tie_word_embeddings=True,
    )

    def test_the_rotary_is_ntk_alpha(self, model):
        assert model._module.model.rotary_emb.rope_type == "ntk_alpha"
