"""OPT on vLLM (``vllm.model_executor.models.opt``).

Its tree is transformers' (``model.decoder.{embed_tokens, embed_positions,
layers[i].{self_attn_layer_norm, self_attn, final_layer_norm, fc1, fc2},
final_layer_norm}``). The block is plain: called with the residual stream
alone, it returns the stream. As on transformers there is no MLP module:
``fc1`` and ``fc2`` sit on the block, so ``mlp`` and ``mlp_output`` do not
exist here either, and what the feed-forward path adds is ``fc2``'s output
(vLLM's linear layers return ``(output, bias)``, the bias already added: read
``layers[i].fc2.output[0]``).

transformers scales the queries by ``head_dim**-0.5`` at ``q_proj``'s output;
vLLM leaves them unscaled and its attention layer applies the scale. So
``attention_queries`` is served times that scale, the pre-scaled queries
transformers serves, as a computed copy divided back on a write, and the
recomputed scores are ``queries @ keys.T`` with no further scale.

When the checkpoint ties its embeddings (OPT-125m and most of the family),
vLLM's ``lm_head`` *is* ``embed_tokens``: the same module under both names.
On a checkpoint with ``do_layer_norm_before`` false (OPT-350m) the block
norms after each add, and the stream is not ``layer_input`` plus the two
contributions. OPT-350m also has no final norm (no ``norm``) and projects the
stream with ``project_out`` before the head, as on transformers; the decoder
applies whichever of the two it has, and `project_on_vocab` does the same.
"""

import torch

from vllm.model_executor.models.opt import OPTAttention, OPTDecoderLayer

from ...components import POST_NORM_ATTENTION, POST_NORM_MLP, Pattern, Queries, Residual
from ...components.eproperty import DerivedEProperty
from ...components.vllm import Attention, Flat, Layer, Mlp, on_decode_step
from ..opt import intermediate_size  # noqa: F401  the width of fc1/fc2 is ffn_dim, as on transformers

RENAME = {
    "model.decoder.embed_tokens": "embed_tokens",
    "model.decoder.layers": "layers",
    "model.decoder.final_layer_norm": "norm",
    "self_attn_layer_norm": "input_layernorm",
    # The sublayers' interiors: the output projection, the activation and the down projection under their standard names.
    "out_proj": "o_proj",
}


def norms_before(block) -> bool:
    """The block's ``do_layer_norm_before``: the norms precede the sublayers (every OPT but the 350m, whose norms follow the adds)."""
    return block._module.do_layer_norm_before


def query_scale(envoy) -> float:
    """``head_dim**-0.5``, which transformers applies to the queries and vLLM's attention layer to the scores."""
    return envoy._module.scaling


def scores(self: "Attention") -> Pattern:
    """The causal scores, ``[1, num_heads, query, key]``: the pre-scaled queries times the keys, later keys at ``-inf``."""
    values = self.attention_queries @ self.attention_keys.transpose(-1, -2)
    position = torch.arange(values.shape[-1], device=values.device)
    return values.masked_fill(position[None, :] > position[:, None], float("-inf"))


class Layer(Layer):
    """OPT's block: ``forward(hidden_states) -> hidden_states``.

    The pre-MLP norm is the block's own ``final_layer_norm`` and the feed-forward is ``fc1``/``fc2`` on the block
    itself, so ``layer_mid`` and ``mlp_norm_output`` read the norm and there is no ``mlp`` module. vLLM's rows are
    the flattened stream transformers holds there (``FlatResidual``); they are served ``[1, tokens, hidden_size]``,
    as every value on vLLM is. With ``do_layer_norm_before`` false (OPT-350m) the norms follow the adds, as on
    transformers: the attention reads the stream itself, ``layer_mid`` is ``self_attn_layer_norm``'s output, and no
    norm precedes the feed-forward path.
    """

    @Flat("input_layernorm.output", unavailable=lambda block: None if norms_before(block) else POST_NORM_ATTENTION,
          description="The normed stream entering the attention, [1, tokens, hidden_size]")
    def attention_norm_output(self, value) -> Residual:
        return value

    @Flat(lambda block: "final_layer_norm.input" if norms_before(block) else "input_layernorm.output",
          description="The residual stream after the attention add, [1, tokens, hidden_size]", in_place=True)
    def layer_mid(self, value) -> Residual:
        return value

    @Flat("final_layer_norm.output", unavailable=lambda block: None if norms_before(block) else POST_NORM_MLP,
          description="The normed stream entering the feed-forward path, [1, tokens, hidden_size]")
    def mlp_norm_output(self, value) -> Residual:
        return value


class Attention(Attention):
    """OPT's attention: its output is what the block adds; its queries are served pre-scaled, as on transformers."""

    def no_rotary(self) -> str | None:
        return "no rotary embedding: OPT adds learned absolute position embeddings (embed_positions) to the stream"

    attention_scores = DerivedEProperty(
        scores,
        description="The scores entering the softmax, [1, num_heads, query, key], recomputed from the pre-scaled queries and the keys; prefill only, read-only",
        unavailable=on_decode_step,
    )

    @Flat(
        "attn.inputs", select=0, heads="first", factor=query_scale,
        description="The queries entering vLLM's attention layer times head_dim**-0.5, as transformers scales them, [1, num_heads, tokens, head_dim]",
    )
    def attention_queries(self, value: torch.Tensor) -> Queries:
        return value


class Mlp(Mlp):
    """OPT has no MLP module; this class exists so the family declares one, never instantiated."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``. No MLP module to key.
ENVOYS = {OPTDecoderLayer: Layer, OPTAttention: Attention}


# -- the logit lens ------------------------------------------------------------------

def project_on_vocab(model, hidden: torch.Tensor) -> torch.Tensor:
    """The decoder's final norm where it has one, ``project_out`` where it has one, then vLLM's logits processor over ``lm_head``.

    As on transformers: OPT-350m has no final norm and projects the stream from
    ``hidden_size`` to ``word_embed_proj_dim`` before the head; every other
    released OPT norms and does not project. vLLM's ``project_out`` returns
    ``(output, bias)``.
    """
    decoder = model.model.decoder
    rows = hidden.reshape(-1, hidden.shape[-1]).contiguous()
    if decoder.final_layer_norm is not None:
        rows = decoder.final_layer_norm(rows)
    if decoder.project_out is not None:
        rows = decoder.project_out(rows)[0]
    return model.logits_processor(model.lm_head, rows).reshape(*hidden.shape[:-1], -1)
