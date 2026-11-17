"""StarCoder2 on vLLM, through its transformers backend (``Starcoder2ForCausalLM`` -> ``TransformersForCausalLM``).

vLLM runs transformers' modules (see `nnterp.components.vllm.transformers_backend`),
with Llama's names. The block is Llama's, pre-norm with the residual added in
the block, so the contributions are the modules' outputs and the backend's
envoys hold as they are. The norms are layer norms and the MLP is
``c_fc``/``c_proj`` with no gate.
"""

from transformers.models.starcoder2.modeling_starcoder2 import (
    Starcoder2Attention,
    Starcoder2DecoderLayer,
    Starcoder2MLP,
)

from ...components.vllm.transformers_backend import Attention, Layer, Mlp

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    # The sublayers' interiors: the output projection, the activation and the down projection under their standard names.
    "act": "act_fn",
    "c_proj": "down_proj",
}

#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {Starcoder2DecoderLayer: Layer, Starcoder2Attention: Attention, Starcoder2MLP: Mlp}
