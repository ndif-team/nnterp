"""JetMoE, end to end: a mixture of attention heads under ``self_attention``, a mixture-of-experts MLP."""

import torch
from suite import FamilySuite, HEAD_WEIGHTS, PROMPT, rows

from nnterp import axis_size
from nnterp.families import jetmoe


class TestJetMoe(FamilySuite):
    REPO = "hf-tiny-v2/tiny-random-JetMoeForCausalLM"
    FAMILY = jetmoe
    NATIVE = rows("model", "layers", "embed_tokens", "norm", attn="self_attention")
    MOE_UNAVAILABLE = {"expert_outputs": "sorted by expert"}
    KV_HEADS_EXPANDED = True  # keys and values are tiled to every routing slot before the interface
    EXPECTED_UNAVAILABLE = {"self_attn.attention_premix": "a mixture of attention heads", HEAD_WEIGHTS: "a mixture of attention heads"}

    def rotated_as_served(self, attn, name, rotated):
        """The keys are tiled to every routing slot after the rotary (``repeat(1, top_k, 1, 1)``)."""
        return rotated.repeat(1, attn._module.top_k, 1, 1) if name == "attention_keys_prerope" else rotated

    def model_size(self, host, axis, name=None):
        """The keys before the rotary are not tiled yet: ``num_kv_heads`` wide."""
        if name == "attention_keys_prerope":
            return axis_size(host, axis)
        return super().model_size(host, axis, name)

    def test_attention_returns_router_logits_third(self, model):
        with model.trace(PROMPT):
            raw = model.layers[0].self_attn.output.save()
            std = model.layers[0].self_attn.attention_output.save()
        assert isinstance(raw, tuple) and len(raw) == 3 and torch.equal(std, raw[0])

    def test_heads_are_routing_slots_over_the_kv_heads(self, model):
        config = model.config
        assert model.num_heads == config.num_experts_per_tok * config.num_key_value_heads
        with model.trace(PROMPT):
            keys = model.layers[0].self_attn.attention_keys.save()
        kv = config.num_key_value_heads
        assert torch.equal(keys[:, :kv], keys[:, kv : 2 * kv])  # tiled, not interleaved
