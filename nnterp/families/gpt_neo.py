"""GPT-Neo (``GPTNeoForCausalLM``).

``transformer.{wte, wpe, drop, h[i].{ln_1, attn.attention, ln_2, mlp}, ln_f}`` and
``lm_head``. A sequential block that returns ``(hidden_states, attn_weights)``.
``attn`` is a ``GPTNeoAttention`` wrapper that forwards to ``attn.attention``, the
``GPTNeoSelfAttention`` holding the projections, and returns its output as is;
``self_attn`` is bound to that inner module, so its input is ``ln_1``'s output and
its first output is what the block adds. The inner module does its own arithmetic
in ``_attn``: an fp32 ``q @ k^T`` with no ``1/sqrt(head_dim)`` scaling, a causal
mask by ``torch.where`` (a sliding window of ``window_size`` on the ``local``
layers of ``attention_layers``, which alternate with ``global`` ones), the padding
mask added, softmax, a cast to the values' dtype, ``attn_dropout``, then
``@ v``. The model knows only ``eager`` and ``flash_attention_2``; the interior is
read around the ``_attn`` call and needs ``eager``. ``wpe`` and ``drop`` have no
standard name.
"""

from typing import TYPE_CHECKING

from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoBlock, GPTNeoMLP, GPTNeoSelfAttention

from ..components import (
    Attention, EProperty, HeadOutputs, Keys, Layer, Mlp, Pattern, Queries, Values, needs_eager, seq_first,
)

if TYPE_CHECKING:
    from ..standardized import StandardizedTransformer

RENAME = {
    "transformer.wte": "embed_tokens",
    "transformer.h": "layers",
    "transformer.ln_f": "norm",
    "ln_1": "input_layernorm",
    "attn.attention": "self_attn",
    "ln_2": "post_attention_layernorm",
}


class Layer(Layer):
    """GPT-Neo's decoder block; returns ``(hidden_states, attn_weights)``, which the base unwraps."""

    returns_tuple = True


class Attention(Attention):
    """GPT-Neo's inner self-attention: the residual is added in the block; the pattern lives in its ``_attn`` method."""

    # The interior lives around the ``_attn`` method call: its arguments are the
    # queries, keys and values split into heads (no rotary embeddings, no
    # grouped-query attention, so all three are ``num_heads`` wide), its softmax
    # takes the masked, unscaled fp32 scores, and its first return is the head
    # outputs, heads first. The queries and keys are cast to fp32 inside
    # ``_attn``, after the call's inputs are read, so they are served in the
    # model's dtype.

    @EProperty("source.self__attn_0.inputs", select=0, description=Attention.attention_queries.description, unavailable=needs_eager)
    def attention_queries(self, value) -> Queries:
        return value

    @EProperty("source.self__attn_0.inputs", select=1, description=Attention.attention_keys.description, unavailable=needs_eager)
    def attention_keys(self, value) -> Keys:
        return value

    @EProperty("source.self__attn_0.inputs", select=2, description=Attention.attention_values.description, unavailable=needs_eager)
    def attention_values(self, value) -> Values:
        return value

    @EProperty(
        "source.self__attn_0.source.nn_functional_softmax_0.input",
        description="The attention scores entering the softmax, masked and unscaled, in fp32",
        unavailable=needs_eager,
    )
    def attention_scores(self, value) -> Pattern:
        return value

    @EProperty("source.self__attn_0.output", select=0, description=Attention.attention_head_outputs.description, unavailable=needs_eager)
    def attention_head_outputs(self, value) -> HeadOutputs:
        return seq_first(value)

    @attention_head_outputs.postprocess
    def attention_head_outputs(self, value):
        return seq_first(value)

    @EProperty(
        "source.self__attn_0.source.self_attn_dropout_0.output",
        description="The attention pattern the values are mixed with",
        unavailable=needs_eager,
    )
    def attention_probabilities(self, value) -> Pattern:
        return value


class Mlp(Mlp):
    """GPT-Neo's MLP; the residual is added in the block, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``. ``GPTNeoSelfAttention``
#: also covers its flash subclass, which nnsight matches through the MRO.
ENVOYS = {GPTNeoBlock: Layer, GPTNeoSelfAttention: Attention, GPTNeoMLP: Mlp}


# -- sizes: what GPT-Neo's config calls them ----------------------------------------

def intermediate_size(model: "StandardizedTransformer") -> int:
    """The MLP width is ``intermediate_size``, ``None`` meaning four times the hidden size."""
    return model.config.intermediate_size or 4 * model.hidden_size
