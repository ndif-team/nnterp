"""GPT-NeoX-Japanese, end to end: tuple block, the last block's attention bias added in the block, arithmetic in ``_attn``."""

import pytest
import torch
from suite import FamilySuite, rows, PROMPT

from nnterp.families import gpt_neox_japanese


class TestGPTNeoXJapanese(FamilySuite):
    REPO = "hf-internal-testing/tiny-random-GPTNeoXJapaneseForCausalLM"
    FAMILY = gpt_neox_japanese
    NATIVE = {**rows("gpt_neox_japanese", "layers", "embed_in", "final_layer_norm", attn="attention"), "lm_head": "embed_out"}

    def test_block_returns_a_tuple(self, model):
        with model.trace(PROMPT):
            raw = model.layers[0].output.save()
        assert isinstance(raw, tuple)

    def test_block_is_sequential(self, model):
        """``post_attention_layernorm`` normalizes the stream after the attention is added, not the block input."""
        with model.trace(PROMPT):
            x = model.layers[0].input.save()
            attn = model.layers[0].self_attn.attention_output.save()
            mid = model.layers[0].post_attention_layernorm.input.save()
        torch.testing.assert_close(mid, x + attn)

    @pytest.fixture
    def biased(self, model):
        """The last block's ``dense_bias``, zero on the tiny checkpoint, set to noise for the test and restored after."""
        bias = model.layers[-1].self_attn._module.dense_bias
        saved = bias.detach().clone()
        with torch.no_grad():
            bias.copy_(torch.randn_like(bias))
        yield bias
        with torch.no_grad():
            bias.copy_(saved)

    def test_only_the_last_block_has_a_bias(self, model):
        assert model.layers[-1].self_attn._module.dense_bias is not None
        assert all(layer.self_attn._module.dense_bias is None for layer in model.layers[:-1])

    def test_last_block_attention_output_carries_the_bias(self, model, biased):
        """The block adds ``dense_bias`` to the attention's output; ``attention_output`` includes it and the identity holds."""
        last = model.layers[-1]
        with model.trace(PROMPT):
            x = last.input.save()
            module = last.self_attn.output[0].save()
            contribution = last.self_attn.attention_output.save()
            mlp = last.mlp.mlp_output.save()
            out = last.layer_output.save()
        torch.testing.assert_close(contribution, module + biased)
        torch.testing.assert_close(x + contribution + mlp, out)

    def test_biased_read_leaves_the_model_unchanged(self, model, biased):
        """A read of the sum hands nothing back: the logits are bit-identical to a run without the read."""
        with model.trace(PROMPT):
            clean = model.logits.save()
        with model.trace(PROMPT):
            model.layers[-1].self_attn.attention_output.save()
            logits = model.logits.save()
        assert torch.equal(logits, clean)

    def test_biased_writes_land(self, model, biased):
        """Zeroing the sum, in place or by assignment, leaves the stream after the attention equal to the block input."""
        last = model.layers[-1]
        with model.trace(PROMPT):
            x = last.input.save()
            last.self_attn.attention_output[:] = 0
            mid = last.post_attention_layernorm.input.save()
        torch.testing.assert_close(mid, x)
        with model.trace(PROMPT):
            x = last.input.save()
            last.self_attn.attention_output = torch.zeros_like(x)
            mid = last.post_attention_layernorm.input.save()
        torch.testing.assert_close(mid, x)

    def test_interior_needs_no_eager_load(self):
        """GPT-NeoX-Japanese has one attention implementation, so a default load serves the pattern."""
        from nnterp import StandardizedTransformer

        model = StandardizedTransformer(self.REPO, dispatch=True)
        assert model.support()["self_attn.attention_probabilities"] is None
        with model.trace(PROMPT):
            pattern = model.layers[0].self_attn.attention_probabilities.save()
        torch.testing.assert_close(pattern.sum(-1), torch.ones_like(pattern.sum(-1)))
