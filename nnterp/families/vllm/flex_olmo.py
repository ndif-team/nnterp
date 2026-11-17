"""FlexOlmo on vLLM, through its transformers backend (``FlexOlmoForCausalLM`` -> ``TransformersMoEForCausalLM``).

vLLM runs transformers' modules (see `nnterp.components.vllm.transformers_backend`),
with Llama's names and OLMo 2's post-norm block around a mixture of experts,
so the names and the envoys are `olmo2`'s, keyed on FlexOlmo's classes. Every
block's ``mlp`` is the mixture; the routing and the experts are inside vLLM's
fused MoE kernel and have no values here.
"""

from transformers.models.flex_olmo.modeling_flex_olmo import FlexOlmoAttention, FlexOlmoDecoderLayer, FlexOlmoSparseMoeBlock

from .olmo2 import RENAME, Attention, Layer, Mlp  # noqa: F401  OLMo 2's post-norm block, unchanged

#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {FlexOlmoDecoderLayer: Layer, FlexOlmoAttention: Attention, FlexOlmoSparseMoeBlock: Mlp}
