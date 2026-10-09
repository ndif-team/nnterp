"""MiMo-V2-Flash on vLLM (``vllm.model_executor.models.mimo_v2``).

Llama's names and Llama's fused block: it takes and returns the stream as
``(hidden_states, residual)``. vLLM reads the configuration under the names
Xiaomi's released ``config.json`` gives it (``hybrid_layer_pattern``,
``moe_layer_freq``, ``swa_*``, ``layernorm_epsilon``), beside the ones
transformers reads; both describe the same model.

The attention is transformers' (see the transformers family): queries and
keys ``head_dim`` wide, values and head outputs ``v_head_dim`` wide, the
values scaled by ``attention_value_scale`` before the attention layer, and on
a sliding-window block twice the key/value heads and a learned sink logit per
head that joins the softmax as one extra key and is dropped after it. The
values and head outputs are split into heads ``v_head_dim`` wide here, and
the recomputed pattern is the softmax with the block's sink column, dropped,
so a sliding block's rows sum to less than one, as on transformers; the
scores are the masked scores before the sink joins. ``mlp`` is a dense MLP or
a mixture of experts (``moe_layer_freq``), whose output is what the block
adds; the routing is inside vLLM's fused MoE kernel and has no values here.
"""

import torch

from vllm.model_executor.models.mimo_v2 import MiMoV2Attention, MiMoV2FlashDecoderLayer, MiMoV2MLP, MiMoV2MoE

from ...components import HeadOutputs, Pattern, Values
from ...components.eproperty import DerivedEProperty
from ...components.vllm import Attention, Flat, FusedLayer, Mlp, on_decode_step
from ..mimo_v2_flash import head_dim, qk_head_dim  # noqa: F401  values v_head_dim wide, queries and keys head_dim, as on transformers

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


class ValueHeads(Flat):
    """A `Flat` over the values or the head outputs: split into heads by the attention layer's value width, ``head_size_v``."""

    def _layout(self, obj) -> dict:
        layout = super()._layout(obj)
        *walk, _ = self.path(obj).split(".")
        layout["head_dim"] = obj.get(".".join(walk))._module.head_size_v
        return layout


def probabilities(self: "Attention") -> Pattern:
    """The softmax of `attention_scores` in float32, with the block's sink logits as one more key, dropped after."""
    values = self.attention_scores
    sinks = self._module.attention_sink_bias
    if sinks is None:
        return values.float().softmax(-1).to(values.dtype)
    sink = sinks.float()[None, :, None, None].expand(*values.shape[:-1], 1)
    return torch.cat([values.float(), sink], dim=-1).softmax(-1)[..., :-1].to(values.dtype)


class Layer(FusedLayer):
    """The decoder block: ``forward(positions, hidden_states, residual) -> (hidden_states, residual)``."""


class Attention(Attention):
    """The attention: values ``v_head_dim`` wide, and on a sliding-window block a sink that takes part of each query's attention."""

    #: The pattern's rows sum to less than one on the sliding-window blocks, where the sink takes the rest.
    SINK = True

    attention_probabilities = DerivedEProperty(
        probabilities,
        description="The attention pattern, [1, heads, query, key], recomputed from the queries and keys with the sink; prefill only, read-only",
        unavailable=on_decode_step,
    )

    @ValueHeads("attn.inputs", select=2, heads="first", description="The values entering vLLM's attention layer, [1, kv_heads, tokens, v_head_dim]")
    def attention_values(self, value: torch.Tensor) -> Values:
        """This step's values as the attention layer receives them, after ``attention_value_scale``."""
        return value

    @ValueHeads("attn.output", heads="last", description="The per-head outputs before the output projection, [1, tokens, heads, v_head_dim]")
    def attention_head_outputs(self, value: torch.Tensor) -> HeadOutputs:
        """Each head's output as the attention layer returns it, before the output projection mixes them."""
        return value


class Mlp(Mlp):
    """The dense MLP or the mixture of experts; its output is what the block adds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {MiMoV2FlashDecoderLayer: Layer, MiMoV2Attention: Attention, MiMoV2MLP: Mlp, MiMoV2MoE: Mlp}
