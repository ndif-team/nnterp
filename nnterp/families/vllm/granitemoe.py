"""GraniteMoE on vLLM (``vllm.model_executor.models.granitemoe``).

Granite's block and scalars (see ``granite.py``) with a mixture of experts for
the MLP: ``block_sparse_moe``, aliased to ``mlp``. The block is plain: called
with the positions and the stream, it adds each sublayer's output times
``residual_multiplier``, so ``attention_output`` and ``mlp_output`` are those
products, computed copies divided back on a write (a `Flat` with a
``factor``), as on transformers. ``token_embeddings`` is the lookup before
``embedding_multiplier``, and vLLM's logits processor divides by
``logits_scaling``. The routing is inside vLLM's fused MoE kernel and has no
values here.
"""

from vllm.model_executor.models.granitemoe import GraniteMoeAttention, GraniteMoeDecoderLayer, GraniteMoeMoE

from .granite import Attention as GraniteAttention
from .granite import Layer as GraniteLayer
from .granite import Mlp as GraniteMlp

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    "block_sparse_moe": "mlp",
}


class Layer(GraniteLayer):
    """GraniteMoE's block: Granite's, called with the positions and the stream."""


class Attention(GraniteAttention):
    """GraniteMoE's attention: the block adds its output times ``residual_multiplier``."""


class Mlp(GraniteMlp):
    """GraniteMoE's mixture of experts: the block adds its output times ``residual_multiplier``."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {GraniteMoeDecoderLayer: Layer, GraniteMoeAttention: Attention, GraniteMoeMoE: Mlp}
