"""Jamba (``JambaForCausalLM``): Mamba-1 blocks, attention blocks and a mixture of experts.

Its tree is ``model.{embed_tokens, layers[i], final_layernorm}`` plus
``lm_head``, with two block classes (``config.layers_block_type``): most
blocks are ``JambaMambaDecoderLayer``, ``{input_layernorm, mamba,
pre_ff_layernorm, feed_forward}``, and one in ``attn_layer_period`` is
``JambaAttentionDecoderLayer``, the same with ``self_attn`` in place of
``mamba``. Both are Llama's pre-norm sequential block, so the identity is
Llama's. Each block has one mixer, never both: on a Mamba block every
``self_attn`` value is reported missing and the selective-scan values live
at ``layers[i].linear_attn`` (see `nnterp.SelectiveScan`); on an attention
block the reverse. ``feed_forward`` is a dense MLP or, one block in
``expert_layer_period``, a sparse mixture of experts; both are ``mlp``, and
``pre_ff_layernorm`` is ``post_attention_layernorm``.

The attention has no positional encoding (the Mamba blocks carry position)
and runs the shared eager interface, so the base holds. The Mamba mixer puts
RMS norms on the step size, ``B`` and ``C`` before the scan (with weights,
where Falcon-Mamba's are weightless); the values are read at the kernel
call, after them.
With ``mamba_ssm`` installed, call ``nnterp.route_kernels(model.family,
"torch")`` before the first trace.
"""

import torch
from transformers.models.jamba.modeling_jamba import (
    JambaAttention, JambaAttentionDecoderLayer, JambaMambaDecoderLayer, JambaMambaMixer, JambaMLP, JambaSparseMoeBlock,
)

from ..components import LOGITS, Attention, Layer, Mlp, Moe, RouterLogits, SelectiveScan, TokenEProperty, mixture_reason

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.final_layernorm": "norm",
    "mamba": "linear_attn",
    "feed_forward": "mlp",
    "pre_ff_layernorm": "post_attention_layernorm",
}


class Layer(Layer):
    """Jamba's two block classes; both return a bare tensor, so the base holds."""


class Attention(Attention):
    """Jamba's attention (one block in ``attn_layer_period``); the shared eager forward, so the base holds."""


class SelectiveScan(SelectiveScan):
    """Jamba's Mamba mixer; the norms on ``B``, ``C`` and the step size come before the scan, so the base holds."""


class Mlp(Mlp):
    """Jamba's dense MLP or sparse mixture of experts; both return the hidden states, added in the block."""


def router_is_linear(moe) -> bool:
    """Whether the router is a bare ``nn.Linear`` (transformers through 5.20), whose output is the logits; from 5.21 it is a
    ``JambaTopKRouter`` whose forward computes them, the base's pattern."""
    return isinstance(moe.router._module, torch.nn.Linear)


class Moe(Moe, Mlp):
    """Jamba's mixture of experts: a softmax top-k of the router's logits, the selected weights left unnormalized.

    The router is a bare ``nn.Linear`` through transformers 5.20, so its output is the
    logits; from 5.21 it is a ``JambaTopKRouter`` and the logits are the ``F.linear`` inside
    its forward, as on every other mixture.
    """

    @TokenEProperty(
        lambda moe: "router.output" if router_is_linear(moe) else f"router.source.{LOGITS}.output",
        description=Moe.router_logits.description,
        unavailable=mixture_reason,
    )
    def router_logits(self, value) -> RouterLogits:
        return value


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {
    JambaAttentionDecoderLayer: Layer, JambaMambaDecoderLayer: Layer, JambaAttention: Attention,
    JambaMambaMixer: SelectiveScan, JambaMLP: Mlp, JambaSparseMoeBlock: Moe,
}
