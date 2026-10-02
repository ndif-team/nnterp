"""Granite SWA (``GraniteSWAForCausalLM``).

Granite's tree, block and multipliers (``granite.py``: scaled residual adds,
``embedding_multiplier``, ``logits_scaling``), with ``layer_types`` mixing
sliding-window and full-attention blocks, a rotary embedding per block (or none,
NoPE, where the model passes no position embeddings), and a learned **attention
sink** per head. The sink does not join the softmax as a column (as on GPT-OSS):
the eager forward takes the softmax over the real keys, mixes the values with it,
and then scales each query's head output by ``sigmoid(logsumexp(scores) - sink)``,
the share of the mass a softmax with the sink as one more logit would leave on the
real keys. So ``attention_probabilities`` is that softmax (``F_dropout_0``), whose
rows sum to one, ``attention_scores`` its input, and ``attention_head_outputs``
the interface's output, after the sink scale. The eager forward spells its
softmax and dropout ``F.softmax`` / ``F.dropout``, so both are redefined on those
operations.
"""

from transformers.models.granite_swa.modeling_granite_swa import GraniteSWAAttention, GraniteSWADecoderLayer, GraniteSWAMLP

from ..components import EProperty, INTERFACE, Layer, Pattern, interface_reason
from .granite import Attention as GraniteAttention
from .granite import Mlp as GraniteMlp
from .granite import project_on_vocab  # noqa: F401  the logit lens divides by logits_scaling, as Granite's

MODEL_TYPES = ("granite_swa",)

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


class Layer(Layer):
    """Granite SWA's decoder block; returns a bare tensor, so the base holds."""


class Attention(GraniteAttention):
    """Granite SWA's attention: Granite's scaled contribution, and a sink applied to the head outputs after the softmax.

    The pattern is the softmax over the real keys, rows summing to one; the
    sink's share scales the head outputs, not the pattern.
    """

    @EProperty(f"source.{INTERFACE}.source.F_softmax_0.input", description=GraniteAttention.attention_scores.description, unavailable=interface_reason)
    def attention_scores(self, value) -> Pattern:
        return value

    @EProperty(f"source.{INTERFACE}.source.F_dropout_0.output", description=GraniteAttention.attention_probabilities.description, unavailable=interface_reason)
    def attention_probabilities(self, value) -> Pattern:
        return value


class Mlp(GraniteMlp):
    """Granite SWA's MLP: Granite's, the block adds its output times ``residual_multiplier``."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {GraniteSWADecoderLayer: Layer, GraniteSWAAttention: Attention, GraniteSWAMLP: Mlp}
