"""SmolLM3 (``SmolLM3ForCausalLM``).

Llama's tree and Llama's block: ``model.{embed_tokens, layers[i].{input_layernorm,
self_attn, post_attention_layernorm, mlp}, norm}`` and ``lm_head``, the residual
added in the block, attention through the shared eager forward. ``no_rope_layers`` leaves every fourth block
without a rotary embedding. A config can put a sliding window on those blocks (``use_sliding_window``); the
released checkpoints leave it off, so every block attends over the whole prompt.
"""

from transformers.models.smollm3.modeling_smollm3 import SmolLM3Attention, SmolLM3DecoderLayer, SmolLM3MLP

from ..components import Attention, Layer, Mlp

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


class Layer(Layer):
    """SmolLM3's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """SmolLM3's attention; the shared eager forward and the residual added in the block, so the base holds."""

    def no_rotary(self) -> str | None:
        if not self._module.use_rope:
            return "no rotary embedding on this block: SmolLM3 skips it on the blocks no_rope_layers marks"
        return None


class Mlp(Mlp):
    """SmolLM3's MLP; the residual is added in the block, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {SmolLM3DecoderLayer: Layer, SmolLM3Attention: Attention, SmolLM3MLP: Mlp}
