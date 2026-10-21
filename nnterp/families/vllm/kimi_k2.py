"""Kimi K2 on vLLM (``model_type`` ``kimi_k2``): the text-only checkpoints, DeepSeek-V3's family under Moonshot's model type.

vLLM builds a text-only Kimi K2 checkpoint (Kimi-K2-Instruct and the other
``DeepseekV3ForCausalLM`` ones) as its own ``DeepseekV2ForCausalLM``, with the
stack at ``model.{embed_tokens, layers, norm}``: DeepSeek-V3's tree, so its names
and envoys hold unchanged. Kimi K2.5's wrapper, which puts the language model
under ``language_model``, is ``kimi_k25``.
"""

from .deepseek_v3 import ENVOYS, LATENT, RENAME, Attention, Layer, Mlp, head_dim, qk_head_dim  # noqa: F401  DeepSeek-V3's, unchanged
