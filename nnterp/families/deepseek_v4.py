"""DeepSeek-V4 (``DeepseekV4ForCausalLM``): a hyper-connection residual, compressed attention, all-MoE.

``model.{embed_tokens, layers[i].{attn_hc, input_layernorm, self_attn, ffn_hc,
post_attention_layernorm, mlp}, hc_head, norm}`` and ``lm_head``.

**The residual is ``hc_mult`` parallel streams**, ``[batch, seq, streams, hidden]``,
between every pair of blocks (the model copies the embedding into each stream
before block 0). Each sublayer reads one collapse of the streams and writes back
into all of them through a hyper-connection (``attn_hc`` before the attention,
``ffn_hc`` before the MLP), which returns ``(post, comb, collapsed)``:
``collapsed`` is the ``pre``-weighted sum of the streams that the sublayer's norm
reads, ``post`` ``[batch, seq, streams]`` how much of the sublayer's output each
stream receives, ``comb`` ``[batch, seq, streams, streams]`` a doubly stochastic
matrix mixing the streams, consumed transposed. The block, per token::

    h   = attention_combᵀ · x  + attention_post ⊗ self_attn(input_layernorm(collapsed_a))
    out = mlp_combᵀ       · h  + mlp_post       ⊗ mlp(post_attention_layernorm(collapsed_f))

* **``layer_output`` is the block's own stream tensor**, under the `Streams`
  layout, and so is ``layers[i].input``; writes land natively, and
  ``skip_layers`` and ``steer`` work unchanged (a ``[hidden]`` steering vector
  broadcasts over the streams).
* **The contributions are the sublayers' own outputs**, ``[batch, seq, hidden]``,
  unscaled: ``attention_output`` is ``self_attn.output[0]``, ``mlp_output`` the
  mixture's return (routed plus shared experts). The weights that put them into
  the streams are the block's ``attention_post``, ``attention_comb``,
  ``mlp_post`` and ``mlp_comb``, read off the hyper-connections' outputs. The
  identity is the block's formula above, not a sum; its stream mean is additive
  (``comb`` is doubly stochastic) up to the Sinkhorn projection's residual, about
  2e-6 relative in float32:
  ``layer_output.mean(2) == input.mean(2) + attention_post.mean(-1, keepdim=True)
  * attention_output + mlp_post.mean(-1, keepdim=True) * mlp_output``.
* **The readout** collapses the streams once more: ``norm(hc_head(streams))``,
  ``hc_head`` a learned, content-dependent weighting. The family's
  ``project_on_vocab`` does the same, so the lens on the last block is
  ``logits`` exactly.

The attention runs the shared interface, with a per-head sink (``s_aux``), so
the scores are GPT-OSS's binding. It is multi-query (one key/value head), and
the keys and values are **one tensor**: the same object is passed as both, so an
in-place edit of one edits the other. On a compressed block
(``compressed_sparse_attention``, ``heavily_compressed_attention``) the
compressor's entries are concatenated after the token keys once the prompt is
``compress_rates[type]`` tokens long, so the key axis is longer than the
sequence there. The interface's output keeps a rotation on its rotary slice
(the values are the rotated keys); the module rotates it back before the grouped
output projection, and ``attention_head_outputs`` is that de-rotated tensor, what
the projection reads. Not latent attention: the plain root sizes hold.

Every MLP is a mixture (``DeepseekV4SparseMoeBlock``, a `Moe`; its router, ``gate``,
is aliased ``router``); on the ``hash_moe`` blocks the router picks experts from the
token ids (``tid2eid[input_ids]``), so a run needs ``input_ids``. There the logits only
weight the experts the ids chose: writing ``router_logits`` changes ``expert_weights``,
not ``expert_indices``.
"""

from typing import TYPE_CHECKING

import torch
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4Attention, DeepseekV4DecoderLayer, DeepseekV4SparseMoeBlock,
)

from ..components import (
    Attention, EProperty, HeadOutputs, INTERFACE, Layer, Moe, Pattern, StreamMixing, Streams, StreamWeights,
    first_tensor, interface_reason, rewrap,
)

if TYPE_CHECKING:
    from ..standardized import StandardizedTransformer

MODEL_TYPES = ("deepseek_v4",)

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    "gate": "router",
}


class Layer(Layer):
    """DeepSeek-V4's decoder block; returns the bare stream tensor, so the base holds, under the `Streams` layout.

    Adds the hyper-connections' stream weights: ``attention_post`` /
    ``attention_comb`` (``attn_hc``) and ``mlp_post`` / ``mlp_comb``
    (``ffn_hc``), float32 whatever the model's dtype.
    """

    @EProperty(key="output", description="The residual streams leaving the block")
    def layer_output(self, value) -> Streams:
        return first_tensor(value)

    @layer_output.postprocess
    def layer_output(self, value):
        return rewrap(self, value)

    @EProperty("attn_hc.output", select=0, description="How much of the attention's output each stream receives")
    def attention_post(self, value) -> StreamWeights:
        return value

    @EProperty("attn_hc.output", select=1, description="The matrix mixing the streams around the attention, applied transposed")
    def attention_comb(self, value) -> StreamMixing:
        return value

    @EProperty("ffn_hc.output", select=0, description="How much of the MLP's output each stream receives")
    def mlp_post(self, value) -> StreamWeights:
        return value

    @EProperty("ffn_hc.output", select=1, description="The matrix mixing the streams around the MLP, applied transposed")
    def mlp_comb(self, value) -> StreamMixing:
        return value


class Attention(Attention):
    """DeepSeek-V4's attention: the shared interface with a sink column, and head outputs rotated back after it.

    The scores are the masked scores bound just before the sink column is
    concatenated (GPT-OSS's binding); the pattern is the base's. The keys and
    values are one tensor object.
    """

    #: The pattern's rows sum to less than one: the sink takes the rest.
    SINK = True

    @EProperty(f"source.{INTERFACE}.source.attn_weights_1.output", description=Attention.attention_scores.description, unavailable=interface_reason)
    def attention_scores(self, value) -> Pattern:
        return value

    @EProperty("source.attn_output_0.output", description=Attention.attention_head_outputs.description)
    def attention_head_outputs(self, value) -> HeadOutputs:
        """The interface's output with its rotary slice rotated back, ``[batch, seq, heads, head_dim]``: what the grouped output projection reads."""
        return value


class Mlp(Moe):
    """DeepSeek-V4's mixture of experts returns routed plus shared experts as a bare tensor, so the base holds."""

    @property
    def SCORING(self) -> str:  # noqa: N802  the class attribute of every other family, per block here
        """``"hash"`` on a ``hash_moe`` block (the token id picks the experts), else the config's ``scoring_func`` (``"sqrtsoftplus"``, per expert)."""
        module = self._module
        return "hash" if module.is_hash else module.experts.config.scoring_func


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {DeepseekV4DecoderLayer: Layer, DeepseekV4Attention: Attention, DeepseekV4SparseMoeBlock: Mlp}


def project_on_vocab(model: "StandardizedTransformer", hidden: torch.Tensor) -> torch.Tensor:
    """The logit lens as the model makes its logits: ``hc_head`` over the streams, the final norm, ``lm_head``.

    A tensor with the stream axis, ``[batch, seq, streams, hidden]`` (rank 4)
    or one position of it, ``[streams, hidden]`` (rank 2 with ``hc_mult``
    rows), is collapsed by the model's ``hc_head`` first; the result drops the
    stream axis, ``[..., vocab]``. Any other tensor is a plain ``[..., hidden]``
    stream (a sublayer's output, one stream ``layer_output[:, :, k]``) and goes
    through ``norm`` and ``lm_head`` alone.
    """
    streams = model.config.hc_mult
    if hidden.dim() == 4 or (hidden.dim() == 2 and hidden.shape[0] == streams):
        collapsed = model.model.hc_head(hidden.reshape(-1, 1, streams, hidden.shape[-1]))
        return model.lm_head(model.norm(collapsed)).reshape(*hidden.shape[:-2], -1)
    return model.lm_head(model.norm(hidden))
