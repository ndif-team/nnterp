"""Bamba (``BambaForCausalLM``).

Llama's containers over one block class that holds either a Mamba-2 mixer or
attention, by ``config.layers_block_type`` (``attn_layer_indices`` name the
attention blocks)::

    model.layers[i]          BambaDecoderLayer
        .input_layernorm
        .mamba               BambaMixer, an SSD mixer   (or .self_attn, BambaAttention)
        .pre_ff_layernorm
        .feed_forward        BambaMLP

A pre-norm block, as Llama: the mixer's output and then the MLP's are added
to the stream as the modules return them. The names are Llama's but for three:
``mamba`` is ``linear_attn`` (`nnterp.StateSpace`), ``feed_forward`` is
``mlp`` and ``pre_ff_layernorm`` is ``post_attention_layernorm``. Each block
has ``self_attn`` or ``linear_attn``, never both. The block returns
``(hidden_states, attention_weights)``.
"""

from transformers.models.bamba.modeling_bamba import BambaAttention, BambaDecoderLayer, BambaMixer, BambaMLP

from ..components import Attention, Layer, Mlp, StateSpace

MODEL_TYPES = ("bamba",)

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.final_layernorm": "norm",
    "mamba": "linear_attn",
    "feed_forward": "mlp",
    "pre_ff_layernorm": "post_attention_layernorm",
}


class Layer(Layer):
    """Bamba's block, either kind; returns ``(hidden_states, attention_weights)``."""

    returns_tuple = True


class Attention(Attention):
    """Bamba's attention: the shared eager forward, so the base holds."""


class StateSpace(StateSpace):
    """Bamba's Mamba-2 mixer: transformers' Mamba-2 scan and the residual added in the block, so the base holds."""


class Mlp(Mlp):
    """Bamba's gated MLP; returns what the block adds, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {BambaDecoderLayer: Layer, BambaAttention: Attention, BambaMixer: StateSpace, BambaMLP: Mlp}
