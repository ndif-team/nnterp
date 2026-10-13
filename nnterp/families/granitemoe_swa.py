"""GraniteMoE SWA (``GraniteMoeSWAForCausalLM``).

Granite SWA's attention (``granite_swa.py``: sliding-window and full blocks, a
per-head sink that scales the head outputs after the softmax) in GraniteMoE's
block (``granitemoe.py``): ``block_sparse_moe``, routed experts only, is ``mlp``,
and the block adds each sublayer's output times ``residual_multiplier``, so
``attention_output`` and ``mlp_output`` are the scaled copies. ``embedding_multiplier``
and ``logits_scaling`` are as on Granite.

Where the config sets ``shared_intermediate_size`` (1280 on
``ibm-granite/granite-swash-3b-a600m``; 0 on the pinned tiny) the block also runs a
``shared_mlp`` beside the experts and adds the sum. This family binds ``mlp_output`` to
the routed experts' output times ``residual_multiplier`` only, and
``shared_expert_output`` reports no shared expert, so on that checkpoint
``input + attention_output + mlp_output`` misses ``layer_output`` by the shared
expert's term; read ``layers[i]._module.shared_mlp`` (or its envoy) for it.
"""

from transformers.models.granitemoe_swa.modeling_granitemoe_swa import (
    GraniteMoeSWAAttention,
    GraniteMoeSWADecoderLayer,
    GraniteMoeSWAMoE,
)

from .granite import project_on_vocab  # noqa: F401  the logit lens divides by logits_scaling, as Granite's
from .granite_swa import Attention as GraniteSWAAttention
from .granitemoe import Layer as GraniteMoeLayer
from .granitemoe import Mlp as GraniteMoeMlp

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    "block_sparse_moe": "mlp",
}


class Layer(GraniteMoeLayer):
    """GraniteMoE SWA's decoder block: GraniteMoE's."""


class Attention(GraniteSWAAttention):
    """GraniteMoE SWA's attention: Granite SWA's, with its sink after the softmax and the scaled contribution."""


class Mlp(GraniteMoeMlp):
    """GraniteMoE SWA's mixture of experts: GraniteMoE's (a `Moe`), the block adds its output times ``residual_multiplier``."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {GraniteMoeSWADecoderLayer: Layer, GraniteMoeSWAAttention: Attention, GraniteMoeSWAMoE: Mlp}
