"""HyperCLOVA X on vLLM (``vllm.model_executor.models.hyperclovax``).

Llama's names, a sandwich block and Granite's µP multipliers. The block is
not fused, whatever its signature says: it is called ``forward(positions,
hidden_states, residual)``, ignores the ``residual`` it is handed, adds each
sublayer's term itself and returns ``(hidden_states, residual)`` with the
*whole* stream first, as Exaone4's does::

    h = x + post_norm1(self_attn(input_layernorm(x))) * residual_multiplier
    out = h + post_norm2(mlp(post_attention_layernorm(h))) * residual_multiplier

The values carry transformers' meanings:

- ``attention_output`` and ``mlp_output`` are the term that reaches the
  stream: each post-norm's output times ``residual_multiplier`` (the module's
  own output times it on a checkpoint without ``use_post_norm``, where vLLM
  builds no post-norms). That product is a computed copy (a `Flat` with a
  ``factor``, as on Granite): a write is divided back into the post-norm's
  output, and a read that edits nothing hands it back untouched.
- ``embedding_multiplier``: the model multiplies the embedding module's
  output by it (in place) before the first block, so ``token_embeddings`` is
  the lookup before it, as on transformers, and ``layers[0].layer_input`` the
  multiplied stream.
- ``attention_multiplier`` is the attention layer's scale, which the
  recomputed scores use.
- ``logits_scaling``: vLLM's logits processor multiplies by it (its
  ``scale``), so the default `project_on_vocab` through that processor does
  what the model does.
"""

from nnsight.intervention.envoy import Envoy
from vllm.model_executor.models.hyperclovax import HyperCLOVAXAttention, HyperCLOVAXDecoderLayer, HyperCLOVAXMLP

from ...components import Residual
from ...components.vllm import Attention, Flat, Layer, Mlp
from .granite import residual_multiplier

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


def post_norm(norm: str):
    """The path of what a sublayer hands the multiplier: its post-norm's output, or its own where the block has no post-norms."""

    def path(envoy: Envoy) -> str:
        return f"../{norm}.output" if envoy.parent._module.use_post_norm else "output"

    path.__name__ = f"{norm}_or_output"
    return path


class Layer(Layer):
    """The decoder block: called with the positions and the stream, returning ``(stream, residual)``."""

    STREAM = 1
    returns_tuple = True


class Attention(Attention):
    """The attention: what reaches the residual stream is the post-attention norm's output times ``residual_multiplier``."""

    @Flat(
        post_norm("post_norm1"), factor=residual_multiplier,
        description="What the attention adds to the residual stream: the post-attention norm's output times residual_multiplier, [1, tokens, hidden]",
    )
    def attention_output(self, value) -> Residual:
        return value


class Mlp(Mlp):
    """The MLP: what reaches the residual stream is the post-MLP norm's output times ``residual_multiplier``."""

    @Flat(
        post_norm("post_norm2"), factor=residual_multiplier,
        description="What the MLP adds to the residual stream: the post-MLP norm's output times residual_multiplier, [1, tokens, hidden]",
    )
    def mlp_output(self, value) -> Residual:
        return value


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {HyperCLOVAXDecoderLayer: Layer, HyperCLOVAXAttention: Attention, HyperCLOVAXMLP: Mlp}
