"""Kimi K2.5 on vLLM (``model_type`` ``kimi_k25``): Kimi K2's family, named by the wrapper's own type.

The wrapper's ``text_config`` says ``deepseek_v3`` from transformers 5.20, so the
wrapper resolves here (`nnterp.families.resolve`), to ``kimi_k2``'s toolkit.
"""

from .kimi_k2 import *  # noqa: F401, F403
