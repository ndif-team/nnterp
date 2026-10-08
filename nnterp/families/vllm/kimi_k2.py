"""Kimi K2's language model on vLLM, inside Kimi-K2.5 (``vllm.model_executor.models.kimi_k25``), text-only prompts.

Kimi K2's language model is DeepSeek-V3's, under Moonshot's ``kimi_k2``
model type (see the transformers family). vLLM runs a K2.5 checkpoint as
``KimiK25ForConditionalGeneration``, whose language model is vLLM's own
``DeepseekV2ForCausalLM`` at ``language_model``: the stack at
``language_model.model.{embed_tokens, layers, norm}``, the head and the
logits processor beside it. The block, the latent attention and the mixture
of experts are `deepseek_v3`'s classes and envoys, unchanged, under that
prefix; what `deepseek_v3` says of them (the fused block, the latent
attention's interior not mapped, ``VLLM_MLA_DISABLE=1`` for float32) holds
here.

vLLM registers the wrapper as ``KimiK25ForConditionalGeneration``; a
checkpoint saved by transformers says ``Kimi_K25ForConditionalGeneration``,
so the engine needs ``hf_overrides={"architectures":
["KimiK25ForConditionalGeneration"]}`` to load it.

Text-only. The vision tower is not mapped and images are out of scope here;
run the engine as a language model (``language_model_only=True``). In its
multimodal mode vLLM embeds every prompt outside the model's forward and
calls it with ``inputs_embeds`` and no ids, so ``input_ids`` and
``token_embeddings`` are not there to read.
"""

from .deepseek_v3 import ENVOYS, LATENT, Attention, Layer, Mlp, head_dim, qk_head_dim  # noqa: F401  DeepSeek-V3's, unchanged

RENAME = {
    "language_model.model.embed_tokens": "embed_tokens",
    "language_model.model.layers": "layers",
    "language_model.model.norm": "norm",
    "language_model.lm_head": "lm_head",
    "language_model.logits_processor": "logits_processor",
}
