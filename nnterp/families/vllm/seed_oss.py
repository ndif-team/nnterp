"""Seed-OSS on vLLM, through its transformers backend (``SeedOssForCausalLM`` -> ``TransformersForCausalLM``).

vLLM runs transformers' modules (see `nnterp.components.vllm.transformers_backend`),
with Llama's names. The block is Llama's, pre-norm with the residual added in
the block, so the contributions are the modules' outputs and the backend's
envoys hold as they are.
"""

from transformers.models.seed_oss.modeling_seed_oss import SeedOssAttention, SeedOssDecoderLayer, SeedOssMLP

from ...components.vllm.transformers_backend import Attention, Layer, Mlp

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}

#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {SeedOssDecoderLayer: Layer, SeedOssAttention: Attention, SeedOssMLP: Mlp}
