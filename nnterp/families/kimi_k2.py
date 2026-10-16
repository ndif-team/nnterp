"""Kimi K2 (``model_type`` ``kimi_k2``): DeepSeek-V3's architecture under Moonshot's own model type.

Kimi K2 and its successors' language models (K2.5, K2.6, K2.7-Code) are
DeepSeek-V3's ``DeepseekV3ForCausalLM`` with Moonshot's tokenizer; their configs say
``kimi_k2``. transformers builds them with its native DeepSeek-V3 classes: the
multimodal ``kimi_k25`` checkpoints through ``Kimi_K25Config``, whose ``text_config``
is a ``DeepseekV3Config`` (its ``model_type`` is the checkpoint's ``kimi_k2`` before
transformers 5.20 and ``deepseek_v3`` from it, so `nnterp.families.WRAPPERS` names this
family for the ``kimi_k25`` wrapper), and the
text-only K2 checkpoints from a ``DeepseekV3ForCausalLM`` loaded directly, since
``AutoConfig`` maps no ``kimi_k2``. The tree, the envoys and the sizes are
DeepSeek-V3's (``deepseek_v3.py``); this module names them under ``kimi_k2``,
because the family registry is the module's file name.
"""

from .deepseek_v3 import ENVOYS, RENAME, Attention, Layer, Mlp, Moe, head_dim, qk_head_dim  # noqa: F401  DeepSeek-V3's, unchanged
