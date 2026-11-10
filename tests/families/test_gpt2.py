"""GPT-2, end to end."""

import torch
from suite import NO_ROTARY, PROMPT, FamilySuite, rows

from nnterp.families import gpt2


class TestGPT2(FamilySuite):
    REPO = "hf-internal-testing/tiny-random-gpt2"
    FAMILY = gpt2
    NATIVE = rows("transformer", "h", "wte", "ln_f", attn="attn", ln1="ln_1", ln2="ln_2")
    REFUSES_IN_PLACE_QKV = True  # q/k/v are split views of one c_attn tensor
    EXPECTED_UNAVAILABLE = NO_ROTARY


class TestGPT2Upcast(TestGPT2):
    """The ``stanford-crfm`` GPT-2 checkpoints' config: ``reorder_and_upcast_attn``, under which the eager forward calls
    GPT-2's own ``_upcast_and_reordered_attn`` in place of the interface, and ``scale_attn_by_inverse_layer_idx``.
    The whole suite runs on the interior read at that call."""

    LOAD_KWARGS = {"reorder_and_upcast_attn": True, "scale_attn_by_inverse_layer_idx": True}

    def test_the_interior_is_read_at_the_upcast_call(self, model):
        attn = model.layers[1].self_attn
        assert attn._module.reorder_and_upcast_attn and gpt2.upcast(attn)
        assert gpt2.Attention.attention_queries.path(attn) == f"source.{gpt2.UPCAST}.inputs"
        support = model.support(layer=1)
        interior = ("queries", "keys", "values", "scores", "probabilities", "head_outputs")
        assert all(support[f"self_attn.attention_{name}"] is None for name in interior) and support["self_attn.interface"] is None
        with model.trace(PROMPT):
            q, k, v = attn.attention_queries.save(), attn.attention_keys.save(), attn.attention_values.save()
            scores = attn.attention_scores.save()
            pattern = attn.attention_probabilities.save()
            heads = attn.attention_head_outputs.save()
            returned = attn.source.self__upcast_and_reordered_attn_0.output.save()
        assert torch.equal(heads, returned[0]) and torch.equal(pattern, returned[1])
        # Block 1 scales by head_dim ** -0.5 / (layer_idx + 1), in float32, before the mask.
        raw = (q.float() @ k.float().transpose(-1, -2)) * attn.head_dim ** -0.5 / 2
        causal = torch.ones(raw.shape[-2:], dtype=torch.bool).tril()
        assert scores.dtype == torch.float32
        torch.testing.assert_close(scores[..., causal], raw[..., causal])
        torch.testing.assert_close(heads, (pattern @ v).transpose(1, 2))
