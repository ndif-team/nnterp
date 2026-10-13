"""StableLM / StableLM-2 (``StableLmForCausalLM``).

Llama's containers. With ``use_parallel_residual`` the block is parallel with one
``input_layernorm`` and a ``dropout``; without it there is a
``post_attention_layernorm`` too. Under ``qk_layernorm`` the attention norms its
queries and keys per head (``q_layernorm`` / ``k_layernorm``). Of the released
checkpoints only StableLM-2-12B sets both (as does the pinned tiny); StableLM-2-1.6B
and StableLM-3B-4E1T are sequential with no q/k norms.
"""

from transformers.models.stablelm.modeling_stablelm import StableLmAttention, StableLmDecoderLayer, StableLmMLP

from ..components import Attention, EProperty, Layer, Mlp, PARALLEL, Residual

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


class Layer(Layer):
    """StableLM's block; returns a bare tensor, so the base holds.

    With ``use_parallel_residual`` there is no stream between the sublayers, and the MLP reads
    ``input_layernorm``'s output; without it the block is the pre-norm one.
    """

    layer_mid = EProperty(
        "post_attention_layernorm.input",
        description="The residual stream after the attention add",
        unavailable=lambda block: PARALLEL if block.root.config.use_parallel_residual else None,
    )

    @EProperty(
        lambda block: "input_layernorm.output" if block.root.config.use_parallel_residual else "post_attention_layernorm.output",
        description="The normed stream entering the MLP",
    )
    def mlp_norm_output(self, value) -> Residual:
        return value



class Attention(Attention):
    """StableLM's attention; the shared eager forward and the residual added in the block, so the base holds."""


class Mlp(Mlp):
    """StableLM's MLP; the residual is added in the block, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {StableLmDecoderLayer: Layer, StableLmAttention: Attention, StableLmMLP: Mlp}
