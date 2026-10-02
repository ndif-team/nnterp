"""Cohere 2 (Command-R7B), end to end: a parallel block, sliding layers with rotary, full layers without."""

import torch
from suite import FamilySuite, PROMPT, rows

from nnter.families import cohere, cohere2


class TestCohere2(FamilySuite):
    REPO = "trl-internal-testing/tiny-Cohere2ForCausalLM"
    FAMILY = cohere2
    NATIVE = rows("model", "layers", "embed_tokens", "norm", ln2=None)
    MLP_NORM = "input_layernorm"  # parallel: one norm feeds both sublayers

    def test_both_layer_types_are_covered(self, model):
        assert set(model.config.layer_types) == {"sliding_attention", "full_attention"}

    def test_full_layers_take_the_unrotated_projections(self, model):
        """Only sliding layers apply the rotary: on a full layer the queries are ``q_proj``'s output, heads first."""
        kinds = model.config.layer_types
        full, sliding = kinds.index("full_attention"), kinds.index("sliding_attention")
        projected = {}
        for i in (full, sliding):
            with model.trace(PROMPT):
                attn = model.layers[i].self_attn
                q_proj = attn.q_proj.output.save()
                queries = attn.attention_queries.save()
            projected[i] = (q_proj.view(*q_proj.shape[:2], model.num_heads, model.head_dim).transpose(1, 2), queries)
        torch.testing.assert_close(*projected[full])
        assert not torch.allclose(*projected[sliding])

    def test_logits_scale_is_cohere_s(self, model):
        assert model.family.project_on_vocab is cohere.project_on_vocab
