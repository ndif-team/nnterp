"""Qwen3-VL's language model on vLLM (``vllm.model_executor.models.qwen3_vl``), text-only prompts.

vLLM runs the whole ``Qwen3VLForConditionalGeneration``, whose language model
is ``Qwen3LLMForCausalLM`` at ``language_model``: the stack at
``language_model.model.{embed_tokens, layers, norm}``, built from vLLM's
Qwen3 classes, so the fused block (``(hidden_states, residual)``) and the
attention (queries and keys normed, then rotated) are `qwen3`'s under that
prefix. The positions are interleaved M-RoPE's ``[3, tokens]``; on a
text-only prompt the three rows are the same 1-D positions, so the rotation
is plain Qwen3's.

DeepStack. The model adds a deepstack feature to ``hidden_states`` after
each of the first blocks, outside the block, only when it is called with
``inputs_embeds``, which is vLLM's multimodal mode. As a language model there
is none, and ``layers[k+1].layer_input == layers[k].layer_output`` on every
block; ``deepstack_output`` is not mapped.

Text-only. The vision tower is not mapped and images are out of scope here;
run the engine as a language model (``language_model_only=True``, or every
``limit_mm_per_prompt`` at 0). In its multimodal mode vLLM embeds every
prompt outside the model's forward and calls it with ``inputs_embeds`` and
no ids, so ``input_ids`` and ``token_embeddings`` are not there to read.
"""

from vllm.model_executor.models.qwen3 import Qwen3Attention, Qwen3DecoderLayer, Qwen3MLP

from ...components.vllm import Attention, FusedLayer, Mlp

RENAME = {
    "language_model.model.embed_tokens": "embed_tokens",
    "language_model.model.layers": "layers",
    "language_model.model.norm": "norm",
    "language_model.lm_head": "lm_head",
    "language_model.logits_processor": "logits_processor",
}


class Layer(FusedLayer):
    """Qwen3's decoder block: ``forward(positions, hidden_states, residual) -> (hidden_states, residual)``, with M-RoPE positions."""


class Attention(Attention):
    """Qwen3's attention; its output is what the block adds, so the base holds."""


class Mlp(Mlp):
    """Qwen3's MLP (vLLM's ``Qwen2MLP``); its output is what the block adds, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {Qwen3DecoderLayer: Layer, Qwen3Attention: Attention, Qwen3MLP: Mlp}
