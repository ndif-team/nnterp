"""OLMo 3 on vLLM, through its transformers backend (``Olmo3ForCausalLM`` -> ``TransformersForCausalLM``).

vLLM runs transformers' modules (see `nnterp.components.vllm.transformers_backend`),
with Llama's names and OLMo 2's post-norm block, so the names and the envoys
are `olmo2`'s, keyed on OLMo 3's classes.
"""

from transformers.models.olmo3.modeling_olmo3 import Olmo3Attention, Olmo3DecoderLayer, Olmo3MLP

from .olmo2 import RENAME, Attention, Layer, Mlp  # noqa: F401  OLMo 2's post-norm block, unchanged

#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {Olmo3DecoderLayer: Layer, Olmo3Attention: Attention, Olmo3MLP: Mlp}
