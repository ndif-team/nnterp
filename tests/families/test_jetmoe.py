"""JetMoE, end to end: a mixture of attention heads under ``self_attention``, a mixture-of-experts MLP."""

import torch
from suite import FamilySuite, PROMPT, rows, built, patched_checkpoint

from nnterp import axis_size
from nnterp.families import jetmoe


REPO = "hf-tiny-v2/tiny-random-JetMoeForCausalLM"


class TestJetMoe(FamilySuite):
    REPO = REPO
    FAMILY = jetmoe
    NATIVE = rows("model", "layers", "embed_tokens", "norm", attn="self_attention")
    MOE_UNAVAILABLE = {"expert_outputs": "sorted by expert"}
    KV_HEADS_EXPANDED = True  # keys and values are tiled to every routing slot before the interface
    EXPECTED_UNAVAILABLE = {"self_attn.attention_premix": "a mixture of attention heads", "self_attn.value_weight": "a mixture of attention heads", "self_attn.output_weight": "a mixture of attention heads"}

    def rotated_as_served(self, attn, name, rotated, positions):
        """The keys are tiled to every routing slot after the rotary (``repeat(1, top_k, 1, 1)``)."""
        return rotated.repeat(1, attn._module.top_k, 1, 1) if name == "attention_keys_prerope" else rotated

    def mixture_output(self, model, host):
        """The mixture adds its own ``bias`` after the routed sum, and ``routed_output`` is before it (no shared expert):
        what ``routed_output`` equals is the mixture's output minus that bias."""
        return super().mixture_output(model, host) - host._module.bias

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


def draw_biases(model):
    """jetmoe-8b's mixtures add a trained bias (``torch.empty`` at init, zeros on the tiny checkpoint): drawn here,
    on the MLP's mixture and on the attention's."""
    for layer in model.model.layers:
        layer.mlp.bias.normal_(0, 0.05)
        layer.self_attention.experts.bias.normal_(0, 0.05)


class TestJetMoeWithBias(TestJetMoe):
    """The tiny config with the mixtures' biases away from zero, as on every trained JetMoE; weights drawn for it."""

    REPO = built(patched_checkpoint, REPO, init=draw_biases)

    def test_mlp_output_is_the_routed_output_plus_the_bias(self, model):
        """``mlp_output - routed_output == bias`` at every position, as the family documents."""
        host = model.layers[0].mlp
        assert host._module.bias.abs().max() > 0
        with model.trace(PROMPT):
            routed = host.routed_output.save()
            out = host.mlp_output.save()
        torch.testing.assert_close(out - routed, host._module.bias.expand_as(out))
