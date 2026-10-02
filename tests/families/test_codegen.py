"""CodeGen, end to end: GPT-J's tuple parallel block, arithmetic in ``_attn`` with the mask added before the scaling."""

import math

import torch
from suite import FamilySuite, rows, PROMPT

from nnter.families import codegen


class TestCodeGen(FamilySuite):
    REPO = "hf-internal-testing/tiny-random-CodeGenForCausalLM"
    FAMILY = codegen
    NATIVE = rows("transformer", "h", "wte", "ln_f", attn="attn", ln1="ln_1", ln2=None)
    MLP_NORM = "input_layernorm"             # parallel: ln_1 feeds both sublayers

    def test_block_returns_a_tuple(self, model):
        with model.trace(PROMPT):
            raw = model.layers[0].output.save()
        assert isinstance(raw, tuple)

    def test_scores_are_scaled_after_the_mask(self, model):
        """The scores are ``(q @ k^T + mask) / sqrt(head_dim)`` where the mask lets them through."""
        with model.trace(PROMPT):
            q = model.layers[0].self_attn.attention_queries.save()
            k = model.layers[0].self_attn.attention_keys.save()
            scores = model.layers[0].self_attn.attention_scores.save()
        raw = q.float() @ k.float().transpose(-1, -2) / math.sqrt(model.head_dim)
        causal = torch.ones(raw.shape[-2:], dtype=torch.bool).tril()
        torch.testing.assert_close(scores[..., causal], raw[..., causal])

    def test_interior_needs_no_eager_load(self):
        """CodeGen has one attention implementation, so a default load serves the pattern."""
        from nnter import StandardizedTransformer

        model = StandardizedTransformer(self.REPO, dispatch=True)
        assert model.support()["self_attn.attention_probabilities"] is None
        with model.trace(PROMPT):
            pattern = model.layers[0].self_attn.attention_probabilities.save()
        torch.testing.assert_close(pattern.sum(-1), torch.ones_like(pattern.sum(-1)))
