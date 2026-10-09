"""DBRX (``DbrxForCausalLM``).

``transformer.{wte, blocks[i].{norm_attn_norm.{norm_1, attn, norm_2}, ffn},
norm_f}`` and ``lm_head``. The attention sits inside a ``norm_attn_norm``
module that norms before it, adds the residual after it, and norms again for
the FFN; the block adds the FFN's output to the state ``norm_attn_norm``
returns. So the attention module's own output is its contribution (the add
happens outside it, in ``norm_attn_norm``) and the base classes hold; the
aliases reach through ``norm_attn_norm`` so the block reads like any other.
The attention is on the shared interface since transformers 5.17. The FFN is
a mixture of experts (`Moe`) returning the hidden states. Its router's projection is
a child ``nn.Linear``, ``router.layer``, whose output is the logits; the FFN's own
``route_tokens_to_experts`` takes a softmax top-k and p-normalizes the weights
(``moe_normalize_expert_weights``). The experts are DBRX's own loop over experts,
taking and returning ``[batch, seq, hidden]``, so the per-slot outputs are
unavailable.
"""

from typing import TYPE_CHECKING

from transformers.models.dbrx.modeling_dbrx import DbrxAttention, DbrxBlock, DbrxFFN

from ..components import Attention, Layer, Moe, RouterLogits, TokenEProperty, unavailable

if TYPE_CHECKING:
    from ..standardized import StandardizedTransformer

RENAME = {
    "transformer.wte": "embed_tokens",
    "transformer.blocks": "layers",
    "transformer.norm_f": "norm",
    "norm_attn_norm.attn": "self_attn",
    "norm_attn_norm.norm_1": "input_layernorm",
    "norm_attn_norm.norm_2": "post_attention_layernorm",
    "ffn": "mlp",
}


class Layer(Layer):
    """DBRX's block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """DBRX's attention; the shared eager forward, and the residual added outside it in ``norm_attn_norm``."""


class Mlp(Moe):
    """DBRX's mixture of experts; the residual is added in the block, so the base holds."""

    @TokenEProperty("router.layer.output", description=Moe.router_logits.description)
    def router_logits(self, value) -> RouterLogits:
        return value

    expert_outputs = unavailable("DBRX's experts loop over the experts in their own forward; no tensor holds each slot's output")


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {DbrxBlock: Layer, DbrxAttention: Attention, DbrxFFN: Mlp}


# -- sizes: what DBRX's config calls them ------------------------------------------

def intermediate_size(model: "StandardizedTransformer") -> int:
    """The experts' width, ``ffn_config.ffn_hidden_size``; every MLP is a mixture of experts."""
    return model.config.ffn_config.ffn_hidden_size
