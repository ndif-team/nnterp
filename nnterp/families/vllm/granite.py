"""Granite on vLLM (``vllm.model_executor.models.granite``).

Llama's names and a plain block: it is called with the positions and the
residual stream and returns the stream. Granite's four scalars are where they
are on transformers, and the values carry the same meanings:

- ``residual_multiplier``: the block adds each sublayer's output times it, so
  ``attention_output`` and ``mlp_output`` are the module's output times the
  multiplier, the term that reaches the stream. That product is a computed
  copy (a `Flat` with a ``factor``): an assignment or an in-place edit is
  divided back into the module's output, and a read that edits nothing hands
  the output back untouched.
- ``embedding_multiplier``: the model multiplies the embedding module's output
  by it (in place) before the first block, so ``token_embeddings`` is the
  lookup before the multiplier, as on transformers, and ``layers[0].layer_input``
  the multiplied stream.
- ``attention_multiplier`` is the attention layer's scale, which the
  recomputed scores use.
- ``logits_scaling``: vLLM's logits processor divides by it (its ``scale``),
  so the default `project_on_vocab` through that processor already does.
"""

from nnsight.intervention.envoy import Envoy

from vllm.model_executor.models.granite import GraniteAttention, GraniteDecoderLayer, GraniteMLP

from ...components import Residual
from ...components.vllm import Attention, Flat, Layer, Mlp

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


def residual_multiplier(envoy: Envoy) -> float:
    """What the block multiplies each sublayer's output by before adding it: the parent block's own scalar."""
    return envoy.parent._module.residual_multiplier



class Layer(Layer):
    """Granite's block: called with the positions and the stream, adding each sublayer's output times ``residual_multiplier``."""

    STREAM = 1


class Attention(Attention):
    """Granite's attention: what reaches the residual stream is its output times ``residual_multiplier``."""

    @Flat("output", factor=residual_multiplier, description="What the attention adds to the residual stream: its output times residual_multiplier, [1, tokens, hidden]")
    def attention_output(self, value) -> Residual:
        return value


class Mlp(Mlp):
    """Granite's MLP: what reaches the residual stream is its output times ``residual_multiplier``."""

    @Flat("output", factor=residual_multiplier, description="What the MLP adds to the residual stream: its output times residual_multiplier, [1, tokens, hidden]")
    def mlp_output(self, value) -> Residual:
        return value


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {GraniteDecoderLayer: Layer, GraniteAttention: Attention, GraniteMLP: Mlp}
