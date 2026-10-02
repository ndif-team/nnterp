"""Doge (``DogeForCausalLM``, SmallDoge).

Llama's tree with a **gated residual**: the block scales the stream it carries,
not what it adds. Each sublayer's output is added to the residual times a learned
per-channel gate (``input_residual``, then ``post_attention_residual``, both
``[hidden_size]`` parameters of the block)::

    h = input_residual * x + self_attn(input_layernorm(x))
    out = post_attention_residual * h + mlp(post_attention_layernorm(h))

What each sublayer adds is its module's output, unscaled, so the base holds for
``attention_output`` and ``mlp_output``; the contribution identity is
``layer_output == post_attention_residual * (input_residual * input + attention_output) + mlp_output``,
which is the plain one only where both gates are one (their initial value). The
gates are ``layers[i]._module.input_residual`` / ``post_attention_residual``;
nothing is computed or divided for you. The attention runs the shared eager
forward with q/k norms and a *dynamic mask*: ``dt_proj`` of the values, through
``A``, gives each key a per-head bias, added to the causal mask (and, past
``keep_window_size`` keys, used to drop all but the top keys), so
``attention_scores`` carry it. The MLP is a gated ``DogeMLP``, or the cross-domain
mixture ``DogeCDMoE`` under ``is_moe``, which returns ``(hidden_states, router_logits)``.
"""

from transformers.models.doge.modeling_doge import DogeAttention, DogeCDMoE, DogeDecoderLayer, DogeMLP

from ..components import Attention, Layer, Mlp, Moe

MODEL_TYPES = ("doge",)

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


class Layer(Layer):
    """Doge's decoder block; returns a bare tensor, so the base holds (the gates scale the residual, not the contributions)."""


class Attention(Attention):
    """Doge's attention; the shared eager forward with a dynamic mask, and its output added unscaled, so the base holds."""


class Mlp(Mlp):
    """Doge's MLP or cross-domain mixture (``(hidden_states, router_logits)``); the output is added unscaled, so the base holds."""


#: Why none of the cross-domain mixture's values are served.
CANNOT_RUN = "transformers 5.17 cannot run DogeCDMoE: the block drops out its tuple output"


class Moe(Moe, Mlp):
    """Doge's cross-domain mixture (``is_moe``): product-key routing over a dense MLP. Every mixture value is unavailable: transformers cannot run it."""

    def no_mixture(self) -> str:
        return CANNOT_RUN


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {DogeDecoderLayer: Layer, DogeAttention: Attention, DogeMLP: Mlp, DogeCDMoE: Moe}
