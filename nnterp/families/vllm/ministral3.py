"""Ministral 3, text-only (``Ministral3ForCausalLM``), on vLLM.

vLLM runs ``Ministral3ForCausalLM`` through its ``mistral`` module
(``MistralForCausalLM``), so this is the `mistral` family: Llama's names,
the fused block and the classes it keys. The queries' position scaling
(``llama_4_scaling_beta``) is applied by vLLM's ``MistralAttention`` before
its attention layer, so ``attention_queries`` are the scaled ones, as on
transformers; vLLM reads it from a ``llama_4_scaling`` entry that only its
Mistral-format config loader writes, and the factor is exactly 1 below
``original_max_position_embeddings`` (16384 on the released checkpoints),
so the two engines agree on any prompt shorter than that.
"""

from .mistral import ENVOYS, RENAME, Attention, Layer, Mlp  # noqa: F401  vLLM's own module for this model_type
