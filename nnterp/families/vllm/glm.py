"""GLM-4, the first generation (``GlmForCausalLM``), on vLLM, through its transformers backend (``TransformersForCausalLM``).

vLLM runs transformers' modules (see `nnterp.components.vllm.transformers_backend`),
with Llama's names. The block is Llama's, pre-norm with the residual added in
the block, so the contributions are the modules' outputs and the backend's
envoys hold as they are. The MLP's gate and up projections are one
``gate_up_proj``.
"""

from transformers.models.glm.modeling_glm import GlmAttention, GlmDecoderLayer, GlmMLP

from ...components.vllm.transformers_backend import Attention, Layer, Mlp

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    "activation_fn": "act_fn",
}

#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {GlmDecoderLayer: Layer, GlmAttention: Attention, GlmMLP: Mlp}
