"""SmolLM3 on vLLM, through its transformers backend (``SmolLM3ForCausalLM`` -> ``TransformersForCausalLM``).

vLLM runs transformers' modules (see `nnterp.components.vllm.transformers_backend`),
with Llama's names. The block is Llama's, pre-norm with the residual added in
the block, so the contributions are the modules' outputs and the backend's
block and MLP envoys hold as they are. Every fourth block's attention skips
the rotary embedding (``no_rope_layers``); the queries and keys are served as
the attention layer receives them, rotated or not.
"""

from transformers.models.smollm3.modeling_smollm3 import SmolLM3Attention, SmolLM3DecoderLayer, SmolLM3MLP

from ...components.vllm.transformers_backend import Attention, Layer, Mlp

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


class Attention(Attention):
    """SmolLM3's attention: the backend's, saying which blocks apply no rotary embedding."""

    def no_rotary(self) -> str | None:
        if not self._module.use_rope:
            return "no rotary embedding on this block: SmolLM3 skips it on the blocks no_rope_layers marks"
        return super().no_rotary()


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {SmolLM3DecoderLayer: Layer, SmolLM3Attention: Attention, SmolLM3MLP: Mlp}
