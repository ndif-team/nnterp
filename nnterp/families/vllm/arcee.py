"""Arcee (AFM) on vLLM, through its transformers backend (``ArceeForCausalLM`` -> ``TransformersForCausalLM``).

vLLM runs transformers' modules (see `nnterp.components.vllm.transformers_backend`),
with Llama's names. The block is Llama's, pre-norm with the residual added in
the block, so the contributions are the modules' outputs and the backend's
envoys hold as they are. The MLP has no gate (``down_proj(relu2(up_proj(x)))``).
"""

from transformers.models.arcee.modeling_arcee import ArceeAttention, ArceeDecoderLayer, ArceeMLP

from ...components.vllm.transformers_backend import Attention, Layer, Mlp

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}

#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {ArceeDecoderLayer: Layer, ArceeAttention: Attention, ArceeMLP: Mlp}
