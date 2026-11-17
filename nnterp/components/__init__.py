"""The base envoys and the descriptor every family builds its toolkit from.

A family gives its modules standard *values* by wrapping them in `Envoy`
subclasses (nnsight's ``envoys=``): `Layer` on the decoder block, `Attention`
on the softmax-attention module, `Mlp` on the feed-forward, `Moe` on a mixture
of experts, a `RecurrentMixer` (`LinearAttention`, `SelectiveScan`,
`StateSpace`) on a recurrent mixer, and `Vision` with `VisionLayer`,
`VisionAttention` and `VisionMlp` on a multimodal wrapper's vision tower. A
family subclasses these, overriding only what its forward spells differently,
and keys them on its own module types in its ``ENVOYS``.

Every value is an `EProperty`, whose key is a path from the host envoy, or a
`DerivedEProperty`, computed from several. A value a checkpoint does not have
raises `Unavailable` with the reason, which `Standard.support` reports
without raising.
"""

from .attention import (
    INTERFACE, NOT_ON_INTERFACE, OWN_ARITHMETIC, ROTARY, SHARDED_WEIGHTS, Attention, HeadOutputs, Keys, Pattern, Queries, SharedRopeKeys, Values, attention_call,
    call_reason, holding, interface_reason, needs_eager, projection_weight, rotary_reason, seq_first,
)
from .eproperty import (
    DerivedEProperty, EProperty, Unavailable, unavailable,
)
from .layer import ONE_SUBLAYER, PARALLEL, POST_NORM_ATTENTION, POST_NORM_MLP, STATE_SPACE, FlatResidual, Layer, Residual, StreamMixing, Streams, StreamWeights
from .linear_attention import ChannelGates, Gates, LinearAttention, LinearQK, LinearV
from .mlp import Mlp, Neurons
from .moe import (
    DISPATCH, LOGITS, PER_SLOT, ExpertIndices, ExpertOutputs, ExpertWeights, Moe, RouterLogits, mixture_reason,
    needs_grouped_experts, no_shared_expert,
)
from .recurrent import (
    RecurrentMixer, State, States, needs_recurrent_routing, needs_torch_kernels, per_call, pinned,
    route_delta_rule, route_kernels,
)
from .selective_scan import (
    ScanDecays, ScanQK, ScanState, ScanStates, ScanSteps, ScanValues, SelectiveScan, needs_token_loop,
)
from .standard import (
    BATCH, CHANNELS, EXPERTS, GROUPS, HEAD_DIM, HEADS, HIDDEN, IMAGE_TOKENS, IMAGES, INTERMEDIATE, KEY, KEY_DIM, KV_HEADS, MIXER_HEADS, TOKENS,
    PATCHES, QK_HEAD_DIM, QK_ROPE_HEAD_DIM, QUERY, SEQ, STATE_DIM, STREAMS, TOP_K, VALUE_DIM, VISION_HIDDEN, VOCAB, Standard,
    axis_size, blocks_support, first_tensor, rewrap,
)
from .tokens import TokenEProperty
from .state_space import (
    SSDHeadOutputs, SSDKeys, SSDQueries, SSDValues, StateSpace, chunk_per_token, needs_per_token_chunks,
)
from .vision import (
    PER_IMAGE, ImageFeatures, ImageScatter, ImageTokenMask, Patches, PixtralVision, QwenVision, QwenVisionAttention, Vision,
    VisionAttention, VisionLayer, VisionMlp, image_token_id,
)

__all__ = [
    "BATCH", "CHANNELS", "EXPERTS", "GROUPS", "HEAD_DIM", "HEADS", "HIDDEN", "IMAGE_TOKENS", "IMAGES", "INTERMEDIATE", "KEY", "KEY_DIM", "KV_HEADS", "TOKENS",
    "MIXER_HEADS", "PATCHES", "QK_HEAD_DIM", "QK_ROPE_HEAD_DIM", "QUERY", "SEQ", "STATE_DIM", "STREAMS", "TOP_K", "VALUE_DIM", "VISION_HIDDEN", "VOCAB",
    "Attention", "ChannelGates", "DISPATCH", "DerivedEProperty", "EProperty", "ExpertIndices", "ExpertOutputs", "ExpertWeights", "Gates", "HeadOutputs", "INTERFACE", "ROTARY", "ImageFeatures", "ImageTokenMask", "Keys", "Layer", "LinearAttention", "Patches",
    "LOGITS", "LinearQK", "LinearV", "Mlp", "Moe", "PER_SLOT", "Pattern", "Queries", "RecurrentMixer", "Residual", "RouterLogits", "ScanDecays", "ScanQK", "ScanState",
    "ScanStates", "ScanSteps", "ScanValues", "SelectiveScan", "SharedRopeKeys", "State", "States", "StreamMixing", "StreamWeights", "Streams", "TokenEProperty",
    "Values", "Vision", "VisionAttention", "VisionLayer", "VisionMlp", "axis_size", "blocks_support",
    "ImageScatter", "PER_IMAGE", "PixtralVision", "QwenVision", "QwenVisionAttention",
    "NOT_ON_INTERFACE", "OWN_ARITHMETIC", "SHARDED_WEIGHTS", "SSDHeadOutputs", "SSDKeys", "SSDQueries", "SSDValues", "Standard", "StateSpace", "Unavailable",
    "attention_call", "call_reason", "chunk_per_token", "first_tensor", "holding", "image_token_id", "interface_reason", "mixture_reason", "needs_eager",
    "needs_grouped_experts", "needs_per_token_chunks", "needs_recurrent_routing", "needs_token_loop", "needs_torch_kernels", "no_shared_expert", "projection_weight", "rotary_reason",
    "per_call", "pinned", "rewrap", "route_delta_rule", "route_kernels", "seq_first", "unavailable",
    "Neurons",
    "FlatResidual",
    "PARALLEL",
    "POST_NORM_ATTENTION",
    "POST_NORM_MLP",
]
