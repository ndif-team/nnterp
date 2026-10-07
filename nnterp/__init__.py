"""nnterp: one module vocabulary across transformer architectures.

`StandardizedTransformer` is an nnsight `TransformersModel` whose envoy tree
answers to the same names whatever the checkpoint's family. The vocabulary is
Llama's::

    model.embed_tokens
    model.layers[i].input_layernorm
    model.layers[i].self_attn
    model.layers[i].post_attention_layernorm
    model.layers[i].mlp
    model.norm
    model.lm_head

and the root answers for the whole model: ``logits``, ``token_embeddings``,
``next_token_probs`` and the sizes (``num_layers``, ``num_heads``, ...).

Each family under `nnterp.families` says how its own names map onto those, and
`nnterp.components` holds the envoys that give standard modules standard values
(``layer_output``, ``attention_output``, ``mlp_output``,
``attention_probabilities``), which each family subclasses.
"""

try:
    from ._version import version as __version__  # written by setuptools_scm from the git tag at install
except ImportError:  # a source tree that was never installed
    from importlib.metadata import PackageNotFoundError, version as _version

    try:
        __version__ = _version("nnterp")
    except PackageNotFoundError:
        __version__ = "0+unknown"

from .components import (
    Attention, DerivedEProperty, EProperty, Layer, LinearAttention, Mlp, Moe, RecurrentMixer, SelectiveScan, Standard,
    StateSpace, Unavailable, chunk_per_token, route_delta_rule, route_kernels, unavailable,
)
from .families import UnsupportedFamily
from .interventions import (
    TargetPrompt, TargetPromptBatch, it_repeat_prompt, logit_lens, patch_object_attn_lens, patchscope_generate,
    patchscope_lens, repeat_prompt,
)
from .standardized import StandardizedTransformer

__all__ = [
    "Attention", "DerivedEProperty", "EProperty", "Layer", "LinearAttention", "Mlp", "Moe", "RecurrentMixer", "SelectiveScan",
    "Standard", "StandardizedTransformer", "StateSpace", "TargetPrompt", "TargetPromptBatch", "Unavailable",
    "UnsupportedFamily", "chunk_per_token", "it_repeat_prompt", "logit_lens", "patch_object_attn_lens",
    "patchscope_generate", "patchscope_lens", "repeat_prompt", "route_delta_rule", "route_kernels",
    "unavailable",
]
