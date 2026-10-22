"""Qwen2-VL's language model on vLLM (``vllm.model_executor.models.qwen2_vl``), text-only prompts.

vLLM runs the whole ``Qwen2VLForConditionalGeneration``, whose language model
is vLLM's own ``Qwen2ForCausalLM`` at ``language_model``: the stack at
``language_model.model.{embed_tokens, layers, norm}``, the head and the
logits processor beside it. So the classes, and the fused block (``(hidden_states,
residual)``), are `qwen2`'s; only the prefix differs. The positions are
M-RoPE's ``[3, tokens]`` (temporal, height, width); on a text-only prompt the
three rows are the same 1-D positions, so the rotation, and the queries and
keys, are plain Qwen2's, as on transformers.

Text-only. The vision tower is not mapped and images are out of scope here;
run the engine as a language model (``language_model_only=True``, or every
``limit_mm_per_prompt`` at 0). In its multimodal mode vLLM embeds every
prompt, text or not, outside the model's forward and calls it with
``inputs_embeds`` and no ids, so ``input_ids`` and ``token_embeddings`` are
not there to read.
"""

from vllm.model_executor.models.qwen2 import Qwen2Attention, Qwen2DecoderLayer, Qwen2MLP

from ...components.vllm import Attention, FusedLayer, Mlp

RENAME = {
    "language_model.model.embed_tokens": "embed_tokens",
    "language_model.model.layers": "layers",
    "language_model.model.norm": "norm",
    "language_model.lm_head": "lm_head",
    "language_model.logits_processor": "logits_processor",
}


class Layer(FusedLayer):
    """Qwen2's decoder block: ``forward(positions, hidden_states, residual) -> (hidden_states, residual)``, with M-RoPE positions."""


class Attention(Attention):
    """Qwen2's attention; its output is what the block adds, so the base holds."""


class Mlp(Mlp):
    """Qwen2's MLP; its output is what the block adds, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {Qwen2DecoderLayer: Layer, Qwen2Attention: Attention, Qwen2MLP: Mlp}
