"""Hunyuan MoE V1 (Hunyuan-A13B) on vLLM, through its transformers backend (``HunYuanMoEV1ForCausalLM`` -> ``TransformersMoEForCausalLM``).

vLLM runs transformers' modules (see `nnterp.components.vllm.transformers_backend`),
with Llama's names. The block is Llama's, pre-norm with the residual added in
the block, so the contributions are the modules' outputs and the backend's
envoys hold as they are. The attention norms its queries and keys per head
after the rotary embedding, and the attention layer receives them after both.
Every block's ``mlp`` is the mixture of experts, whose output (the shared
expert included) is what the block adds; the routing and the experts are
inside vLLM's fused MoE kernel and have no values here.
"""

from transformers.models.hunyuan_v1_moe.modeling_hunyuan_v1_moe import (
    HunYuanMoEV1Attention,
    HunYuanMoEV1DecoderLayer,
    HunYuanMoEV1Moe,
)

from ...components.vllm.transformers_backend import Attention, Layer, Mlp

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}

#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {HunYuanMoEV1DecoderLayer: Layer, HunYuanMoEV1Attention: Attention, HunYuanMoEV1Moe: Mlp}
