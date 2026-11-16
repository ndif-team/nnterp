"""Hunyuan dense V1 on vLLM, through its transformers backend (``HunYuanDenseV1ForCausalLM`` -> ``TransformersForCausalLM``).

vLLM runs transformers' modules (see `nnterp.components.vllm.transformers_backend`),
with Llama's names. The block is Llama's, pre-norm with the residual added in
the block, so the contributions are the modules' outputs and the backend's
envoys hold as they are. The attention norms its queries and keys per head
after the rotary embedding, and the attention layer receives them after both.
"""

from transformers.models.hunyuan_v1_dense.modeling_hunyuan_v1_dense import (
    HunYuanDenseV1Attention,
    HunYuanDenseV1DecoderLayer,
    HunYuanDenseV1MLP,
)

from ...components.vllm.transformers_backend import Attention, Layer, Mlp

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}

#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {HunYuanDenseV1DecoderLayer: Layer, HunYuanDenseV1Attention: Attention, HunYuanDenseV1MLP: Mlp}
