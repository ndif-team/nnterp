"""Llama 4, text (``Llama4ForCausalLM``, model_type ``llama4_text``): Scout and Maverick's language model.

``model.{embed_tokens, layers[i].{input_layernorm, self_attn,
post_attention_layernorm, feed_forward}, norm}`` and ``lm_head``: Llama's
tree with the feed-forward called ``feed_forward``, renamed to ``mlp``. A
``llama4`` checkpoint (``Llama4ForConditionalGeneration``, the published
Scout and Maverick repos) nests its config's ``text_config``, of this type;
the text-generation task builds ``Llama4ForCausalLM`` from it, and a wrapper
module passed in already loaded keeps the text model at
``language_model.{model.*, lm_head}``, so ``RENAME`` carries both spellings
and whichever the tree has binds.

The block adds both residuals itself, attention through the shared eager
forward (Llama 4's own copy, which keeps the softmax in the model dtype).
What differs from Llama, inside the attention and the feed-forward:

* **iRoPE.** ``no_rope_layers[i] == 0`` marks a NoPE block: no rotary
  embedding, no qk-norm, full causal attention, and *attention temperature
  tuning* (``attn_temperature_tuning``) scales its queries by a factor that
  grows with the position. The other blocks apply the rotary embedding, then
  an L2 qk-norm (``use_qk_norm``), and attend within chunks of
  ``attention_chunk_size`` tokens (``config.layer_types``). The queries and
  keys served are what the interface receives, after all of that; a chunked
  block's pattern is zero across chunk boundaries.
* **Interleaved mixture of experts.** Blocks in ``moe_layers`` carry a
  ``Llama4TextMoe`` (a shared expert plus routed experts) returning
  ``(out, router_logits)`` with ``out`` flattened to ``[batch * seq, hidden]``;
  the others a dense ``Llama4TextMLP``. The block views either back into the
  residual's shape before adding it, so ``mlp_output`` is read at that view in
  the block's forward: ``[batch, seq, hidden]`` on both kinds, and edits to it
  reach the feed-forward's own output tensor.
  The shared expert is a ``Llama4TextMLP`` too, so it is an `Mlp`; its
  ``mlp_output`` is unavailable, since the block adds the mixture's sum.
  The mixture is a `Moe` whose router returns dense scores ``[tokens, experts]``
  (the sigmoid of the top-k logits, zero elsewhere): the experts run on every
  token, each scaled on its *input* by its score, and the routed sum is added
  into the shared expert's output tensor in place. So ``router_logits`` (the
  router's projection) and ``expert_indices`` (its top-k) are the router's own;
  ``routed_output`` is the mixture's sum over the experts; ``shared_expert_output``
  is a copy taken as the shared expert returns, whose edits are carried back; and
  ``expert_weights`` / ``expert_outputs`` are unavailable.
* **Sizes.** ``intermediate_size`` in this config is the experts' width
  (and the shared expert's); the dense MLP's is ``intermediate_size_mlp``, which
  the root's ``intermediate_size`` reports. Scout has a mixture on every block,
  so there it names a width the model never uses, as on Qwen3-MoE; Maverick
  alternates dense and mixture blocks.

On the wrapper the vision tower ``vision_model`` is ``vision`` (a `Vision`)
and ``multi_modal_projector``, whose output is what the wrapper scatters, is
``projector``. The tower is a ViT over each image tile: the patch embedding
(``patch_embedding``, an unfold and a linear) is ``vision.patch_embed``, the
blocks (``model.layers``, pre-norm, on the shared attention interface with a
2D rotary embedding) are ``vision.layers``, and ``layernorm_post`` is
``vision.norm``. A CLS token is appended *after* the patches, so the tower's
stream is ``[tiles, patches + 1, vision_hidden]`` with the CLS last (CLIP's is
first); the tower drops it after ``layernorm_post``, then pixel-shuffles and
projects the patches in ``vision.vision_adapter``, so ``vision.tower_output``
is ``layernorm_post``'s output, CLS included, and ``projector.input`` is the
adapter's output, flattened over the tiles. Loaded with
``task="image-text-to-text"`` (the processor), the tower serves
``vision.image_token_mask`` and ``vision.image_features``. The text-generation
task builds ``Llama4ForCausalLM``, which has no tower.
"""

from typing import TYPE_CHECKING

from transformers.models.llama4.modeling_llama4 import (
    Llama4TextAttention, Llama4TextDecoderLayer, Llama4TextMLP, Llama4TextMoe, Llama4VisionAttention,
    Llama4VisionEncoderLayer, Llama4VisionMLP, Llama4VisionModel,
)

from ..components import (
    Attention, EProperty, ExpertIndices, Layer, Mlp, Moe, Patches, Residual, RouterLogits, TokenEProperty, Vision,
    VisionAttention, VisionLayer, VisionMlp, unavailable,
)

if TYPE_CHECKING:
    from nnsight.intervention.envoy import Envoy

    from ..standardized import StandardizedTransformer

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    # A Llama4ForConditionalGeneration module: the same text model under ``language_model``.
    "language_model.model.embed_tokens": "embed_tokens",
    "language_model.model.layers": "layers",
    "language_model.model.norm": "norm",
    "language_model.lm_head": "lm_head",
    "feed_forward": "mlp",
    "shared_expert": "shared_experts",
    # The wrapper's tower and projector. The tower's blocks are its ``model.layers``, which
    # the text key above binds on the tower; its other keys are names no text block has.
    "vision_model": "vision",
    "multi_modal_projector": "projector",
    "patch_embedding": "patch_embed",
    "layernorm_post": "norm",
}

#: The wrappers (config ``model_type``) whose projector's output is what they scatter into the text stream.
IMAGE_WRAPPERS = ("llama4",)

#: The block's ``hidden_states.view(residual.shape)``: the feed-forward's output in the residual's shape.
FEED_FORWARD_VIEW = "hidden_states_view_0"


def _not_a_block_feed_forward(envoy: "Envoy") -> str | None:
    if envoy.path.rsplit(".", 1)[-1] == "feed_forward":
        return None
    return "this is the shared expert inside a mixture of experts; what the block adds is the mixture's output, at layers[i].mlp"


class Layer(Layer):
    """Llama 4's decoder block; returns a bare tensor, so the base holds.

    `Mlp.mlp_output` is an operation in this forward, read after the block
    has started (its attention has returned), so the forward is instrumented
    at build.
    """

    sourced = True


class Attention(Attention):
    """Llama 4's attention; the shared eager forward and the residual added in the block, so the base holds.

    The queries and keys it serves are after the rotary embedding and the
    qk-norm on a RoPE block, and the queries after temperature tuning on a
    NoPE block.
    """


class Mlp(Mlp):
    """Llama 4's dense MLP or mixture of experts: the contribution is the block's view of its output in the residual's shape.

    The mixture returns ``(out, router_logits)`` with ``out`` flattened over
    batch and sequence; the block views it back before adding it. That view is
    the value on every block, dense or not, so the layout is the same.
    """

    @EProperty(
        f"../source.{FEED_FORWARD_VIEW}.output",
        description="What the MLP adds to the residual stream: its output viewed in the residual's shape",
        unavailable=_not_a_block_feed_forward,
    )
    def mlp_output(self, value) -> Residual:
        return value


#: Why a value of the per-slot routing is not served on Llama 4.
DENSE_SCORES = "Llama 4's router scales each expert's input by a dense score over every expert; not mapped yet"


class Moe(Moe, Mlp):
    """Llama 4's mixture of experts: dense sigmoid scores, every expert on every token, the routed sum added into the shared expert's output.

    ``routed_output`` is an operation of this forward read after the experts
    have run, so the forward is instrumented at build.
    """

    SCORING = "sigmoid"
    sourced = True

    @TokenEProperty("router.source.forward_0.output", description=Moe.router_logits.description)
    def router_logits(self, value) -> RouterLogits:
        """The router's projection, ``[batch, seq, experts]`` (the router is an ``nn.Linear`` whose forward calls its parent's)."""
        return value

    @TokenEProperty("router.source.torch_topk_0.output", select=1, description=Moe.expert_indices.description)
    def expert_indices(self, value) -> ExpertIndices:
        """The router's top-k over the logits, ``[batch, seq, top_k]``; the scores it scatters are built from them."""
        return value

    expert_weights = unavailable(DENSE_SCORES)
    expert_outputs = unavailable(DENSE_SCORES)

    @TokenEProperty("source.sum_0.output", description=Moe.routed_output.description)
    def routed_output(self, value) -> Residual:
        """The experts' outputs summed over the experts, ``[batch, seq, hidden]``: what the mixture adds into the shared expert's output."""
        return value

    @TokenEProperty("shared_experts.output", description="The shared expert's output (a copy, since the mixture adds the routed sum into the live tensor in place)")
    def shared_expert_output(self, value) -> Residual:
        """The shared expert's output as it returns, a copy: the mixture then adds the routed sum into that tensor in place."""
        return value.clone()

    @shared_expert_output.transform
    def shared_expert_output(self, value, raw):
        # Fires on the model side, after the read, with the copy the read was a view of:
        # hand back a second copy, so the mixture's in-place add leaves the user's tensor clean.
        return value.clone()


class Vision(Vision):
    """Llama 4's ViT: ``tower_output`` is the stream after ``layernorm_post``, before the CLS is dropped and the adapter runs.

    The tower's own return is the pixel-shuffle adapter's output
    (``vision.vision_adapter``), a quarter as many rows as patches and
    ``projector_output_dim`` wide; that is ``projector.input``, flattened.
    """

    @EProperty("norm.output", description="The tower's stream after its final norm, CLS last")
    def tower_output(self, value) -> Patches:
        """The last block's stream after ``layernorm_post``, ``[tiles, patches + 1, vision_hidden]``, the CLS token last.

        The tower then drops the CLS and runs ``vision_adapter`` on the
        patches; an assignment here reaches the adapter and the text model.
        """
        return value


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {
    Llama4TextDecoderLayer: Layer, Llama4TextAttention: Attention, Llama4TextMLP: Mlp, Llama4TextMoe: Moe,
    # The ViT's pre-norm blocks on the shared attention interface: the vision components hold as they are.
    Llama4VisionModel: Vision, Llama4VisionEncoderLayer: VisionLayer, Llama4VisionAttention: VisionAttention,
    Llama4VisionMLP: VisionMlp,
}


# -- sizes: what Llama 4's config calls them ------------------------------------

def intermediate_size(model: "StandardizedTransformer") -> int:
    """The dense MLP's width, ``intermediate_size_mlp``; this config's ``intermediate_size`` is the experts'."""
    return model.config.get_text_config().intermediate_size_mlp
