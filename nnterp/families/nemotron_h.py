"""Nemotron-H (``NemotronHForCausalLM``; Nemotron-H, Nemotron-Nano-2, Nemotron-3 Nano and Super).

Every block holds one norm and one sublayer, the ``mixer``, whose class the
block's entry in ``config.layers_block_type`` picks::

    model.embeddings
    model.layers[i]          NemotronHBlock
        .norm                RMSNorm
        .mixer               one of:
                               NemotronHMamba2Mixer   ("linear_attention", an SSD mixer)
                               NemotronHAttention     ("full_attention", no rotary embedding)
                               NemotronHMoE           ("moe": routed experts plus a shared expert)
                               NemotronHMLP           ("mlp")
    model.norm_f
    lm_head

Each block is ``hidden + mixer(norm(hidden))``. One native name, four
meanings, so the standard name is keyed on the mixer's class in ``RENAME``:
``linear_attn`` on a Mamba-2 block, ``self_attn`` on an attention block,
``mlp`` on an MoE or MLP block. The block's ``norm`` keeps its name (its output
is the sublayer's ``.input``). A block has exactly one of the three, so
`support` reports the other two missing on it, per block, and the
contribution identity is ``input + <the one sublayer's output> ==
layer_output``.

The attention runs the shared eager forward (no rotary embedding), so the
base `Attention` holds; the Mamba-2 mixer is transformers' Mamba-2 code, so the
base `StateSpace` holds; the MoE and the MLP return what the block adds. The
model makes its logits in float32 (``lm_head(...).float()``), so
`project_on_vocab` does too. Sizes: the attention's heads are the config's
``num_attention_heads`` / ``num_key_value_heads`` / ``head_dim``; the dense
MLP is ``intermediate_size`` wide, an MoE block's experts
``moe_intermediate_size``.
"""

import torch
from transformers.models.nemotron_h.modeling_nemotron_h import (
    NemotronHAttention,
    NemotronHBlock,
    NemotronHMamba2Mixer,
    NemotronHMLP,
    NemotronHMoE,
)

from ..components import (
    Attention, ExpertOutputs, Layer, Mlp, Moe, Residual, StateSpace, TokenEProperty, needs_grouped_experts,
)

MIXER_NAMES = {
    NemotronHMamba2Mixer: "linear_attn",
    NemotronHAttention: "self_attn",
    NemotronHMoE: "mlp",
    NemotronHMLP: "mlp",
}

RENAME = {
    "model.embeddings": "embed_tokens",
    "model.layers": "layers",
    "model.norm_f": "norm",
    # One native name, four meanings: the standard name follows the mixer's class.
    **MIXER_NAMES,
    "gate": "router",
}

#: The standard name of a block's ``mixer``, by the mixer's class.


class Layer(Layer):
    """Nemotron-H's block: ``norm`` then one ``mixer``, named by what the mixer is; returns a bare tensor."""


class Attention(Attention):
    """Nemotron-H's attention: the shared eager forward without a rotary embedding, so the base holds."""


class StateSpace(StateSpace):
    """Nemotron-H's Mamba-2 mixer: transformers' Mamba-2 scan, so the base holds."""


class Mlp(Mlp):
    """Nemotron-H's mixture of experts (routed plus shared) or dense MLP; either returns what the block adds."""


def project_on_vocab(model, hidden: torch.Tensor) -> torch.Tensor:
    """The logit lens as the model makes its logits: the final norm, ``lm_head``, then float32."""
    return model.lm_head(model.norm(hidden)).float()


def _latent(envoy) -> bool:
    return envoy._module.config.moe_latent_size is not None


def _routed(envoy) -> str:
    return "fc2_latent_proj.output" if _latent(envoy) else "experts.output"


def _no_residual_slots(envoy) -> str | None:
    if _latent(envoy):
        return "the experts run in the latent width (moe_latent_size); their per-slot outputs are not residual contributions"
    return needs_grouped_experts(envoy)


class Moe(Moe, Mlp):
    """Nemotron-H's mixture of experts: DeepSeek-V3's sigmoid router, routed experts and a shared expert.

    With ``moe_latent_size`` the experts run between two projections,
    ``fc1_latent_proj`` down and ``fc2_latent_proj`` back up: ``routed_output`` is
    the up projection's output, and the per-slot outputs, latent-width, are
    unavailable. Without it the projections are identities and the base holds.
    """

    SCORING = "sigmoid"

    @TokenEProperty(Moe.expert_outputs.key, description=Moe.expert_outputs.description, unavailable=_no_residual_slots)
    def expert_outputs(self, value) -> ExpertOutputs:
        return value

    @TokenEProperty(_routed, description=Moe.routed_output.description)
    def routed_output(self, value) -> Residual:
        return value


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {
    NemotronHBlock: Layer,
    NemotronHAttention: Attention,
    NemotronHMamba2Mixer: StateSpace,
    NemotronHMoE: Moe,
    NemotronHMLP: Mlp,
}
