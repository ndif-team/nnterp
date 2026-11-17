"""Kimi K2.5 (``model_type`` ``kimi_k25``): the multimodal wrapper around Kimi K2's text model.

Its ``text_config`` is a ``DeepseekV3Config`` whose ``model_type`` says ``kimi_k2``
in the checkpoint's files and ``deepseek_v3`` where transformers rewrites it at
load, so the wrapper's own type names the family (`nnterp.families.resolve`): Kimi
K2's, which is DeepSeek-V3's tree, envoys and sizes (``kimi_k2.py``).
"""

from .kimi_k2 import *  # noqa: F401, F403
