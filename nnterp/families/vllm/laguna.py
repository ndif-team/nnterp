"""Laguna (poolside's Laguna) on vLLM (``vllm.model_executor.models.laguna``).

Llama's names and Llama's fused block: it takes and returns the stream as
``(hidden_states, residual)``. What differs is inside the sublayers, as on
transformers:

* **Per-block head counts.** ``num_attention_heads_per_layer`` gives each
  block its own number of query heads; ``layers[i].self_attn``'s interior is
  split by that block's attention layer, and the root's ``num_heads`` is the
  config's top-level count.
* **Gated attention.** Per-head query and key norms before the rotary
  embedding, then a softplus gate (``g_proj``, per head or per channel) on the
  attention layer's output before ``o_proj``: the head outputs are ungated and
  ``attention_output`` is the gated projection the block adds.
* **Mixture of experts.** ``mlp_layer_types`` names the dense blocks and the
  sparse ones; a sparse block's ``mlp`` returns the routed experts (times
  ``moe_routed_scaling_factor``) plus its shared expert, which is what the
  block adds. The routing is inside vLLM's fused MoE kernel and has no values
  here. vLLM calls the shared expert ``shared_expert``, aliased to
  ``shared_experts`` as on transformers; it is the dense MLP's class, so it is
  an `Mlp` too, and its ``mlp_output`` is unavailable.
"""

from vllm.model_executor.models.laguna import LagunaAttention, LagunaDecoderLayer, LagunaMLP, LagunaMoE

from ...components import Residual
from ...components.vllm import Attention, Flat, FusedLayer, Mlp

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    "gate": "router",
    "shared_expert": "shared_experts",
}


def _not_a_block_mlp(envoy) -> str | None:
    if envoy.path.rsplit(".", 1)[-1] == "mlp":
        return None
    return "on vLLM this is the shared expert inside the mixture of experts; what the block adds is the mixture's output, at layers[i].mlp"


class Layer(FusedLayer):
    """The decoder block: ``forward(positions, hidden_states, residual) -> (hidden_states, residual)``."""


class Attention(Attention):
    """The gated attention; its output, the gated projection, is what the block adds, so the base holds."""


class Mlp(Mlp):
    """The dense MLP or the mixture of experts; its output is what the block adds, but on the shared expert."""

    @Flat("output", description="What the MLP adds to the residual stream, [1, tokens, hidden]", unavailable=_not_a_block_mlp)
    def mlp_output(self, value) -> Residual:
        return value


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {LagunaDecoderLayer: Layer, LagunaAttention: Attention, LagunaMLP: Mlp, LagunaMoE: Mlp}
