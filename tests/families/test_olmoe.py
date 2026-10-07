"""OLMoE, end to end: Llama's block with query/key norms and a mixture-of-experts MLP."""

import torch
from suite import FamilySuite, LLAMA_ROWS, PROMPT

from nnterp.families import olmoe


class TestOlmoe(FamilySuite):
    REPO = "hf-internal-testing/tiny-random-OlmoeForCausalLM"
    FAMILY = olmoe
    NATIVE = LLAMA_ROWS

    def test_queries_and_keys_are_read_after_their_norms(self, model):
        """``q_norm``/``k_norm`` run on the projections before the head split and the rotary embedding."""
        attn = model.layers[0].self_attn
        attn.source  # nnsight instruments a forward on the first `.source` access; a child's output read before that in the same trace leaves the call uninstrumented
        with model.trace(PROMPT):
            q_normed = attn.q_norm.output.save()
            k_normed = attn.k_norm.output.save()
            queries = attn.attention_queries.save()
        with model.trace(PROMPT):
            keys = attn.attention_keys.save()
        head_dim = model.head_dim
        batch, seq, _ = q_normed.shape
        assert queries.shape == (batch, model.num_heads, seq, head_dim)
        assert keys.shape == (batch, model.num_kv_heads, seq, head_dim)
        # Position 0 has no rotation: the interface's queries and keys there are the normed projections.
        torch.testing.assert_close(queries[:, :, 0], q_normed[:, 0].view(batch, model.num_heads, head_dim))
        torch.testing.assert_close(keys[:, :, 0], k_normed[:, 0].view(batch, model.num_kv_heads, head_dim))

    def test_mlp_output_is_the_routed_mixture(self, model):
        with model.trace(PROMPT):
            out = model.layers[0].mlp.output.save()
            std = model.layers[0].mlp.mlp_output.save()
        assert isinstance(out, torch.Tensor) and torch.equal(out, std)
