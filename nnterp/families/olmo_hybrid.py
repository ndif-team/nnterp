"""OLMo-Hybrid (``OlmoHybridForCausalLM``).

Llama's containers over two block classes, chosen by ``config.layer_types``:

- ``OlmoHybridLinearAttentionDecoderLayer`` (``linear_attention``) is pre-norm, as
  Llama: ``input_layernorm -> linear_attn`` (a gated DeltaNet, see
  `nnterp.LinearAttention`), then ``post_attention_layernorm -> mlp``, each added
  to the stream as the module returns it.
- ``OlmoHybridAttentionDecoderLayer`` (``full_attention``) is OLMo-3's post-norm
  sandwich: ``self_attn -> post_attention_layernorm`` and
  ``mlp -> post_feedforward_layernorm``, the post-norms' outputs added to the
  stream; it has no ``input_layernorm``, so the block input enters the attention.

Each block has ``self_attn`` or ``linear_attn``, never both. The attention runs
the shared eager forward (query and key norms before it; no rotary embedding
when the checkpoint sets no ``rope_theta``), so its contribution is OLMo-3's
override at the post-attention norm. Both block classes hold one MLP class,
``OlmoHybridMLP``, whose contribution is its own output on a linear block and the
post-feedforward norm's output on an attention block, so ``mlp_output`` is a
`EProperty` whose key is chosen by the parent block's layer type.
The DeltaNet calls transformers' ``torch_chunk_gated_delta_rule`` /
``torch_recurrent_gated_delta_rule`` as Qwen3-Next does, so the base
`LinearAttention` holds; with ``linear_allow_neg_eigval`` the betas are doubled
before the kernel and lie in (0, 2).
"""

from transformers.models.olmo_hybrid.modeling_olmo_hybrid import (
    OlmoHybridAttention,
    OlmoHybridAttentionDecoderLayer,
    OlmoHybridGatedDeltaNet,
    OlmoHybridLinearAttentionDecoderLayer,
    OlmoHybridMLP,
)

from ..components import Attention, EProperty, Layer, LinearAttention, Mlp, Residual

MODEL_TYPES = ("olmo_hybrid",)

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


class Layer(Layer):
    """OLMo-Hybrid's block, either class; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """OLMo-Hybrid's softmax attention: the shared eager forward, but what reaches the residual stream is the post-attention norm's output."""

    @EProperty(
        "../post_attention_layernorm.output",
        description="What the attention adds to the residual stream: the post-attention norm's output",
    )
    def attention_output(self, value) -> Residual:
        return value


class LinearAttention(LinearAttention):
    """OLMo-Hybrid's gated DeltaNet mixer; transformers' pure-torch chunked rule and the residual added in the block, so the base holds."""


def by_block(envoy) -> str:
    """The key of this MLP's contribution: the post-feedforward norm on an attention block, the MLP's own output on a linear one."""
    if isinstance(envoy.parent._module, OlmoHybridAttentionDecoderLayer):
        return "../post_feedforward_layernorm.output"
    return "output"


class Mlp(Mlp):
    """OLMo-Hybrid's MLP, one class in both block types: what reaches the residual stream depends on the block holding it."""

    @EProperty(
        by_block,
        description="What the MLP adds to the residual stream: the post-feedforward norm's output on an attention block, the MLP's own on a linear block",
    )
    def mlp_output(self, value) -> Residual:
        return value


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {
    OlmoHybridAttentionDecoderLayer: Layer,
    OlmoHybridLinearAttentionDecoderLayer: Layer,
    OlmoHybridAttention: Attention,
    OlmoHybridGatedDeltaNet: LinearAttention,
    OlmoHybridMLP: Mlp,
}
