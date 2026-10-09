"""GPT-2 (``GPT2LMHeadModel``).

Its tree is ``transformer.{wte, wpe, drop, h[i].{ln_1, attn, ln_2, mlp}, ln_f}``
plus ``lm_head``. The container-level keys are anchored at the root
(``transformer.h``) so the aliases land on the root envoy: ``model.layers``,
not ``model.model.layers``. ``wpe`` (learned absolute positions) and ``drop``
have no standard name; they stay reachable under their own.

A checkpoint with ``reorder_and_upcast_attn`` set takes GPT-2's own
``_upcast_and_reordered_attn`` path instead of the shared eager forward, so
``attention_probabilities`` is unavailable there.
"""

from typing import TYPE_CHECKING

from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block, GPT2MLP

from ..components import Attention, Layer, Mlp

if TYPE_CHECKING:
    from ..standardized import StandardizedTransformer

RENAME = {
    "transformer.wte": "embed_tokens",
    "transformer.h": "layers",
    "transformer.ln_f": "norm",
    "ln_1": "input_layernorm",
    "attn": "self_attn",
    "ln_2": "post_attention_layernorm",
}


class Layer(Layer):
    """GPT-2's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """GPT-2's attention; the shared eager forward and the residual added in the block, so the base holds.

    A checkpoint with ``reorder_and_upcast_attn`` takes GPT-2's own upcast
    path instead, where nothing on the interface is reachable.
    """

    def off_interface(self):
        if self._module.config.reorder_and_upcast_attn:
            return "this checkpoint sets reorder_and_upcast_attn, which takes GPT-2's own upcast attention path"
        return super().off_interface()


class Mlp(Mlp):
    """GPT-2's MLP; the residual is added in the block, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {GPT2Block: Layer, GPT2Attention: Attention, GPT2MLP: Mlp}


# -- sizes: what GPT-2's config calls them ------------------------------------------

def intermediate_size(model: "StandardizedTransformer") -> int:
    """The MLP width is ``n_inner``, ``None`` meaning four times the hidden size; the config's ``intermediate_size`` is never read by the model."""
    return model.config.n_inner or 4 * model.hidden_size
