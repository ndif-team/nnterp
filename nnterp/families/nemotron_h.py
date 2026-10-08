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

NemotronH Omni (``NemotronH_Omni_Reasoning_V3``, model type ``nemotron_h_omni``,
transformers 5.18 and later) holds this model whole at ``language_model``
(``language_model.model.*``, ``language_model.lm_head``) beside a RADIO tower at
``vision_model``, ``vision`` (a `RadioVision`), and ``multi_modal_projector``,
``projector`` (after the pixel shuffle). The root scatters the image features in
its own forward (``ROOT_SCATTER``), so ``vision.image_features`` is read there.
RADIO packs every image of an invoke into one row behind its CLS and register
tokens, so its blocks' values are `Patches` with 1 in the images axis and
``prefix + patches`` tokens per image; ``patch_embeddings`` is the patch
projection's output, before those tokens and the position embeddings. A
block scales what each sublayer adds (``layer_scale1``, ``layer_scale2``), so
``attention_output`` and ``mlp_output`` are the scales' outputs. The attention
makes one interface call over the row for one image and one per image for
several, so its queries, keys and values are served whole and its scores,
pattern and head outputs are `Unavailable` (`RADIO_PER_IMAGE`).
"""

import torch
from transformers.models.nemotron_h.modeling_nemotron_h import (
    NemotronHAttention,
    NemotronHBlock,
    NemotronHMamba2Mixer,
    NemotronHMLP,
    NemotronHMoE,
)

from transformers.models.radio import modeling_radio

from ..components import (
    Attention, EProperty, ExpertOutputs, Keys, Layer, Mlp, Moe, Patches, Queries, Residual, StateSpace, TokenEProperty,
    Values, Vision, VisionAttention, VisionLayer, VisionMlp, needs_grouped_experts,
)
from ..components.vision import no_tower_run, variable_resolution

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
    # NemotronH Omni: the same model whole under ``language_model``, its RADIO tower and projector. The tower's inner
    # keys are names no text block has.
    "language_model.model.embeddings": "embed_tokens",
    "language_model.model.layers": "layers",
    "language_model.model.norm_f": "norm",
    "language_model.lm_head": "lm_head",
    "vision_model": "vision",
    "multi_modal_projector": "projector",
    "embeddings.patch_projection": "patch_embed",
    "encoder.layer": "layers",
    "attention": "self_attn",
    "norm1": "input_layernorm",
    "norm2": "post_attention_layernorm",
}

#: The operation of NemotronH Omni's own forward that scatters the image features (the root is the host:
#: ``inputs_embeds.masked_scatter(image_mask, image_embeds)``, the first of its three).
ROOT_SCATTER = "inputs_embeds_masked_scatter_0"

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


#: Why RADIO's attention scores, pattern and head outputs are unavailable.
RADIO_PER_IMAGE = (
    "RADIO's attention makes one interface call over the packed row for one image (attention_interface_1) and one per "
    "image for several (attention_interface_3), so no one call holds the block's pattern for every input; read "
    "attention_queries and attention_keys, whole, and split them at the attention's cu_seqlens argument "
    "(self_attn.inputs[1]['cu_seqlens'], absent for one image)"
)


class RadioVision(Vision):
    """RADIO, NemotronH Omni's tower: packed, every image's CLS and register tokens then its patches, in one row.

    ``hidden_size``, ``num_heads`` and ``patch_size`` are the config's;
    ``intermediate_size`` is ``hidden_size * mlp_ratio``; ``image_size`` is
    `Unavailable` (the processor sizes each image's grid; ``image_grid_hw``).
    No final norm: ``tower_output`` is the last block's stream.
    """

    image_size = property(variable_resolution)

    @property
    def intermediate_size(self) -> int:
        config = self._module.config
        return int(config.hidden_size * config.mlp_ratio)


class RadioAttention(VisionAttention):
    """RADIO's attention: what it adds is ``layer_scale1``'s output; the interior is read whole around the calls (`RADIO_PER_IMAGE`)."""

    def off_interface(self) -> str | None:
        return no_tower_run(self) or RADIO_PER_IMAGE

    @EProperty("../layer_scale1.output", description=VisionAttention.attention_output.description, unavailable=no_tower_run)
    def attention_output(self, value: torch.Tensor) -> Patches:
        """What the attention adds to the tower's stream, ``[1, tokens, vision_hidden]``: its output times ``layer_scale1``."""
        return value

    @attention_output.postprocess
    def attention_output(self, value: torch.Tensor) -> torch.Tensor:
        return value

    @EProperty("source.transpose_0.output", description=Attention.attention_queries.description, unavailable=no_tower_run)
    def attention_queries(self, value: torch.Tensor) -> Queries:
        """Every image's queries, ``[1, heads, tokens, head_dim]``, before any per-image split; assign or edit in place."""
        return value

    @EProperty("source.transpose_1.output", description=Attention.attention_keys.description, unavailable=no_tower_run)
    def attention_keys(self, value: torch.Tensor) -> Keys:
        """Every image's keys, ``[1, heads, tokens, head_dim]``, before any per-image split; assign or edit in place."""
        return value

    @EProperty("source.transpose_2.output", description=Attention.attention_values.description, unavailable=no_tower_run)
    def attention_values(self, value: torch.Tensor) -> Values:
        """Every image's values, ``[1, heads, tokens, head_dim]``, before any per-image split; assign or edit in place."""
        return value


class RadioMlp(VisionMlp):
    """RADIO's MLP: what it adds is ``layer_scale2``'s output."""

    @EProperty("../layer_scale2.output", description=VisionMlp.mlp_output.description, unavailable=no_tower_run)
    def mlp_output(self, value: torch.Tensor) -> Patches:
        """What the MLP adds to the tower's stream, ``[1, tokens, vision_hidden]``: its output times ``layer_scale2``."""
        return value

    @mlp_output.postprocess
    def mlp_output(self, value: torch.Tensor) -> torch.Tensor:
        return value


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {
    NemotronHBlock: Layer,
    NemotronHAttention: Attention,
    NemotronHMamba2Mixer: StateSpace,
    NemotronHMoE: Moe,
    NemotronHMLP: Mlp,
    # NemotronH Omni's RADIO tower.
    modeling_radio.RadioModel: RadioVision, modeling_radio.RadioLayer: VisionLayer,
    modeling_radio.RadioAttention: RadioAttention, modeling_radio.RadioMLP: RadioMlp, modeling_radio.RadioSwiGLUFFN: RadioMlp,
}
