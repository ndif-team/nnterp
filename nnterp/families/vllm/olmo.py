"""OLMo 1 on vLLM, through its transformers backend (``OlmoForCausalLM`` -> ``TransformersForCausalLM``).

vLLM runs transformers' modules (see `nnterp.components.vllm.transformers_backend`),
with Llama's names. The block is Llama's, pre-norm with the residual added in
the block, so the contributions are the modules' outputs and the backend's
envoys hold as they are. The norms are OLMo's non-parametric layer norms.
"""

from transformers.models.olmo.modeling_olmo import OlmoAttention, OlmoDecoderLayer, OlmoMLP

from ...components.vllm.transformers_backend import Attention, Layer, Mlp

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}

#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {OlmoDecoderLayer: Layer, OlmoAttention: Attention, OlmoMLP: Mlp}
