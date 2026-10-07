"""AFMoE (Arcee's Trinity) on vLLM (``vllm.model_executor.models.afmoe``).

Llama's names and a fused block with AFMoE's sandwich norms: it takes and
returns the stream as ``(hidden_states, residual)``, and each sublayer is
normed before and after. What the block adds is each *post* norm's output,
as on transformers: ``post_attention_layernorm`` follows the attention and
``post_mlp_layernorm`` the MLP, so the contributions point at them (the
pre-MLP norm fuses the attention's add into the residual, keeps its native
name, and its output is ``mlp.input``). The attention norms its queries and
keys per head, rotates them only on its sliding-window blocks, and gates the
attention layer's output with a sigmoid before ``o_proj``, so the head
outputs are ungated, as on transformers. The first ``num_dense_layers``
blocks have a dense MLP, the others a mixture of experts whose output,
shared expert included, is what the block adds; the routing is inside vLLM's
fused MoE kernel and has no values here. The shared expert is the dense MLP's
class, so it is an `Mlp` too; its ``mlp_output`` is unavailable.
"""

from vllm.model_executor.models.afmoe import AfmoeAttention, AfmoeDecoderLayer, AfmoeMLP, AfmoeMoE

from ...components import Residual
from ...components.vllm import Attention, Flat, FusedLayer, Mlp

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


def _not_a_block_mlp(envoy) -> str | None:
    if envoy.path.rsplit(".", 1)[-1] == "mlp":
        return None
    return "on vLLM this is the shared expert inside the mixture of experts; what the block adds is the mixture's post-normed output, at layers[i].mlp"


class Layer(FusedLayer):
    """The decoder block: ``forward(positions, hidden_states, residual) -> (hidden_states, residual)``."""


class Attention(Attention):
    """The attention: what reaches the residual stream is the post-attention norm's output."""

    @Flat(
        "../post_attention_layernorm.output",
        description="What the attention adds to the residual stream: the post-attention norm's output, [1, tokens, hidden]",
    )
    def attention_output(self, value) -> Residual:
        return value


class Mlp(Mlp):
    """The dense MLP or the mixture of experts: what reaches the residual stream is the post-MLP norm's output."""

    @Flat(
        "../post_mlp_layernorm.output",
        description="What the MLP adds to the residual stream: the post-MLP norm's output, [1, tokens, hidden]",
        unavailable=_not_a_block_mlp,
    )
    def mlp_output(self, value) -> Residual:
        return value


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {AfmoeDecoderLayer: Layer, AfmoeAttention: Attention, AfmoeMLP: Mlp, AfmoeMoE: Mlp}
