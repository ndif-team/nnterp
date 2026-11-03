"""DeepSeek-V2 (``DeepseekV2ForCausalLM``).

Llama's tree and Llama's block: ``model.{embed_tokens, layers[i].{input_layernorm,
self_attn, post_attention_layernorm, mlp}, norm}`` and ``lm_head``, the residual
added in the block, attention through the shared eager forward. Multi-head latent attention: queries and keys are ``qk_head_dim`` wide (``qk_nope_head_dim + qk_rope_head_dim``), values ``v_head_dim``, and the interface sees ``num_heads`` key/value heads whatever ``num_key_value_heads`` says. The first ``first_k_dense_replace`` blocks have a dense MLP, the rest a mixture of experts.
"""

import sys
from typing import TYPE_CHECKING

import torch
from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2Attention, DeepseekV2DecoderLayer, DeepseekV2MLP, DeepseekV2Moe

from ..components import Attention, EProperty, Layer, Mlp, Moe, Queries, SharedRopeKeys, rotary_reason

if TYPE_CHECKING:
    from ..standardized import StandardizedTransformer

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    "gate": "router",
}


class Layer(Layer):
    """DeepSeek-V2's decoder block; returns a bare tensor, so the base holds."""


def _rope_call_inputs(attn: "Attention") -> str:
    """The rope call's arguments, whichever function the family rotates with (`Attention.rope_call`)."""
    return f"source.{attn.rope_call()}_0.inputs"


class Attention(Attention):
    """DeepSeek-V2's latent attention; the shared eager forward and the residual added in the block.

    The rotary rotates each query head's trailing ``qk_rope_head_dim``
    channels, and one rope key per position that every head shares; the rest
    of each key comes out of the latent and carries no position. So the
    queries before the rotary are whole heads (what the split into the two
    parts receives, `Queries`), and the keys are that shared rope key
    (`SharedRopeKeys`, ``[batch, 1, seq, qk_rope_head_dim]``). DeepSeek-V3
    and the families that copy its attention subclass this, naming their own
    rope call.
    """

    def rope_call(self) -> str:
        """The modeling module's function the forward rotates the rope parts with: V2's complex ``apply_rotary_emb``."""
        return "apply_rotary_emb"

    @property
    def qk_rope_head_dim(self) -> int:
        """Width of the rotated part of a query head, and of the shared rope key."""
        return self._module.qk_rope_head_dim

    @EProperty("source.torch_split_0.input", description=Attention.attention_queries_prerope.description, unavailable=rotary_reason)
    def attention_queries_prerope(self, value) -> Queries:
        return value

    @EProperty(_rope_call_inputs, select=1, description=Attention.attention_keys_prerope.description, unavailable=rotary_reason)
    def attention_keys_prerope(self, value) -> SharedRopeKeys:
        return value

    def apply_rotary(self, x, embedding):
        """The rope call on the trailing ``qk_rope_head_dim`` channels (all of a rope key's); the leading ones pass, as the forward concatenates them."""
        rope = self.qk_rope_head_dim
        apply = getattr(sys.modules[type(self._module).__module__], self.rope_call())
        args = embedding if isinstance(embedding, tuple) else (embedding,)  # (cos, sin), or V2's complex frequencies
        rot = x[..., -rope:]
        heads = rot.reshape(args[0].shape[0], -1, *rot.shape[-2:])
        return torch.cat((x[..., :-rope], apply(heads, heads, *args)[0].reshape(rot.shape)), dim=-1)


class Mlp(Mlp):
    """DeepSeek's dense MLP or mixture of experts; both return the hidden states, and the residual is added in the block."""


class Moe(Moe, Mlp):
    """DeepSeek-V2's mixture of experts: a softmax router (greedy or group-limited), routed experts and shared experts."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {DeepseekV2DecoderLayer: Layer, DeepseekV2Attention: Attention, DeepseekV2MLP: Mlp, DeepseekV2Moe: Moe}


# -- sizes: multi-head latent attention gives queries and keys their own width ------

def head_dim(model: "StandardizedTransformer") -> int:
    """Width of one head's values and outputs: ``v_head_dim`` (the config's ``head_dim`` is set to ``qk_rope_head_dim``, the rotary slice's width, which no served value has)."""
    return model.config.get_text_config().v_head_dim


def qk_head_dim(model: "StandardizedTransformer") -> int:
    """Width of one head's queries and keys: the non-rotary part plus the rotary part."""
    config = model.config.get_text_config()  # a multimodal wrapper's (Kimi K2.5) nests it
    return config.qk_nope_head_dim + config.qk_rope_head_dim
