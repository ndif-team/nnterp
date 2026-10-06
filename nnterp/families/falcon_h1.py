"""Falcon-H1 (``FalconH1ForCausalLM``).

Llama's containers over a parallel hybrid block: every block runs a Mamba-2
mixer and attention side by side on the same normed input, then an MLP::

    model.layers[i]          FalconH1DecoderLayer
        .input_layernorm
        .mamba               FalconH1Mixer, an SSD mixer
        .self_attn           FalconH1Attention
        .pre_ff_layernorm
        .feed_forward        FalconH1MLP

    h = x + mamba(n) * ssm_out_multiplier + self_attn(n * attention_in_multiplier) * attention_out_multiplier
    out = h + feed_forward(pre_ff_layernorm(h))          with n = input_layernorm(x)

``mamba`` is ``linear_attn`` (`nnterp.StateSpace`), ``feed_forward`` is
``mlp`` and ``pre_ff_layernorm`` is ``post_attention_layernorm``. Every block
has both mixers, so the contribution identity has four terms,
``input + self_attn.attention_output + linear_attn.attention_output +
mlp_output == layer_output``.

The config's µP multipliers, and where each is handled:

- ``ssm_out_multiplier`` / ``attention_out_multiplier``: the block scales
  each mixer's output before adding it, so ``attention_output`` on each is
  that product, the block's own binding of it (``mamba_hidden_states_1``,
  ``attention_hidden_states_0``): the tensor the block adds, so reads, writes
  and in-place edits need no arithmetic.
- ``lm_head_multiplier``: the model multiplies the head's output by it, so the
  family defines ``project_on_vocab`` with that step.
- ``embedding_multiplier``: the model multiplies the embedding module's
  output by it before the first block, so ``token_embeddings`` (the module's
  output) times the multiplier is ``layers[0].input``.
- ``attention_in_multiplier``: the attention's input is the normed stream
  times it (``1.0`` on the released checkpoints), so ``self_attn.input`` is the
  norm's output only then.
- ``mlp_multipliers``, ``key_multiplier``, ``ssm_in_multiplier``,
  ``ssm_multipliers``: applied inside the MLP, the attention and the mixer; the
  values read where the forward already applied them.

With ``mamba_rms_norm`` false the decode step passes the gate into the update
kernel, so a decode step's ``attention_head_outputs`` is gated by
``silu(z)`` where a prompt's is not; with it true (the released checkpoints)
both are ungated. The block returns a one-element tuple.
"""

from typing import TYPE_CHECKING

import torch
from transformers.models.falcon_h1.modeling_falcon_h1 import FalconH1Attention, FalconH1DecoderLayer, FalconH1Mixer, FalconH1MLP

from ..components import Attention, EProperty, Layer, Mlp, Residual, StateSpace

if TYPE_CHECKING:
    from ..standardized import StandardizedTransformer

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.final_layernorm": "norm",
    "mamba": "linear_attn",
    "feed_forward": "mlp",
    "pre_ff_layernorm": "post_attention_layernorm",
}


#: The block's ``mamba_hidden_states * self.ssm_out_multiplier``: what the Mamba-2 mixer adds to the stream.
MAMBA_SCALED = "mamba_hidden_states_1"
#: The block's ``attention_hidden_states * self.attn_out_multiplier``: what the attention adds to the stream.
ATTENTION_SCALED = "attention_hidden_states_0"


class Layer(Layer):
    """Falcon-H1's parallel hybrid block; returns ``(hidden_states,)``.

    Both mixers' contributions are operations in this forward, read after
    the block has started, so the forward is instrumented at build.
    """

    returns_tuple = True
    sourced = True


class Attention(Attention):
    """Falcon-H1's attention: the shared eager forward, but the block adds its output times ``attention_out_multiplier``."""

    @EProperty(f"../source.{ATTENTION_SCALED}.output", description="What the attention adds to the residual stream: its output times attention_out_multiplier")
    def attention_output(self, value) -> Residual:
        return value


class StateSpace(StateSpace):
    """Falcon-H1's Mamba-2 mixer: transformers' Mamba-2 scan, but the block adds its output times ``ssm_out_multiplier``."""

    @EProperty(f"../source.{MAMBA_SCALED}.output", description="What the Mamba-2 mixer adds to the residual stream: its output times ssm_out_multiplier")
    def attention_output(self, value) -> Residual:
        return value


class Mlp(Mlp):
    """Falcon-H1's gated MLP; its ``mlp_multipliers`` are applied inside, so what it returns is what the block adds."""


def project_on_vocab(model: "StandardizedTransformer", hidden: torch.Tensor) -> torch.Tensor:
    """The logit lens as the model makes its logits: the final norm, ``lm_head``, then times ``lm_head_multiplier``."""
    return model.lm_head(model.norm(hidden)) * model.config.lm_head_multiplier


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {FalconH1DecoderLayer: Layer, FalconH1Attention: Attention, FalconH1Mixer: StateSpace, FalconH1MLP: Mlp}
