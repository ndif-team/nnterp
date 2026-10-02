"""GPT-Neo, end to end: tuple block output, ``self_attn`` on the inner module, arithmetic in ``_attn``."""

import torch
from suite import FamilySuite, rows, PROMPT

from nnterp.families import gpt_neo


class TestGPTNeo(FamilySuite):
    REPO = "hf-internal-testing/tiny-random-GPTNeoForCausalLM"
    FAMILY = gpt_neo
    NATIVE = rows("transformer", "h", "wte", "ln_f", attn="attn.attention", ln1="ln_1", ln2="ln_2")

    def test_block_returns_a_tuple(self, model):
        with model.trace(PROMPT):
            raw = model.layers[0].output.save()
        assert isinstance(raw, tuple)

    def test_self_attn_is_the_module_the_wrapper_returns(self, model):
        """``attn`` returns ``attn.attention``'s output unchanged; its first element is ``attention_output``."""
        with model.trace(PROMPT):
            contribution = model.layers[0].self_attn.attention_output.save()  # the inner module returns first
            wrapper = model.layers[0].attn.output[0].save()
        torch.testing.assert_close(wrapper, contribution)

    def test_scores_are_unscaled(self, model):
        """No ``1/sqrt(head_dim)``: the scores are ``q @ k^T`` where the mask lets them through."""
        with model.trace(PROMPT):
            q = model.layers[0].self_attn.attention_queries.save()
            k = model.layers[0].self_attn.attention_keys.save()
        with model.trace(PROMPT):
            scores = model.layers[0].self_attn.attention_scores.save()
        raw = q.float() @ k.float().transpose(-1, -2)
        causal = torch.ones(raw.shape[-2:], dtype=torch.bool).tril()
        torch.testing.assert_close(scores[..., causal], raw[..., causal])

    def test_local_layers_mask_outside_the_window(self, model):
        """A ``local`` layer's pattern is zero more than ``window_size`` tokens back."""
        types = model.config.attention_layers
        window = model.config.window_size
        local = types.index("local")
        prompt = " ".join(["token"] * (window + 4))
        with model.trace(prompt):
            pattern = model.layers[local].self_attn.attention_probabilities.save()
        n = pattern.shape[-1]
        assert n > window
        outside = torch.ones(n, n, dtype=torch.bool).tril(-window)
        assert pattern[..., outside].abs().max() == 0
