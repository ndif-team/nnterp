"""GPT-2 (``GPT2LMHeadModel``).

Its tree is ``transformer.{wte, wpe, drop, h[i].{ln_1, attn, ln_2, mlp}, ln_f}``
plus ``lm_head``. The container-level keys are anchored at the root
(``transformer.h``) so the aliases land on the root envoy: ``model.layers``,
not ``model.model.layers``. ``wpe`` (learned absolute positions) and ``drop``
have no standard name; they stay reachable under their own.

A checkpoint with ``reorder_and_upcast_attn`` set (the ``stanford-crfm`` GPT-2
checkpoints) calls GPT-2's own ``_upcast_and_reordered_attn`` under eager in
place of the shared interface: the queries, keys, values and mask without the
module the interface takes first, the scores computed in float32, the same
return. The attention's interior is read at that call there (`Attention`).
"""

from typing import TYPE_CHECKING

from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block, GPT2MLP

from ..components import (
    INTERFACE, Attention, EProperty, HeadOutputs, Keys, Layer, Mlp, Pattern, Queries, Values, interface_reason, projection_weight,
)

if TYPE_CHECKING:
    from ..standardized import StandardizedTransformer

RENAME = {
    "transformer.wte": "embed_tokens",
    "transformer.h": "layers",
    "transformer.ln_f": "norm",
    "ln_1": "input_layernorm",
    "attn": "self_attn",
    "ln_2": "post_attention_layernorm",
    # The sublayers' interiors: the output projection, the activation and the down projection under their standard names.
    "attn.c_proj": "attn.o_proj",
    "mlp.c_proj": "mlp.down_proj",
    "act": "act_fn",
}


class Layer(Layer):
    """GPT-2's decoder block; returns a bare tensor, so the base holds."""


#: GPT-2's own attention call, under eager on a ``reorder_and_upcast_attn`` checkpoint:
#: ``self._upcast_and_reordered_attn(query, key, value, attention_mask)``.
UPCAST = "self__upcast_and_reordered_attn_0"


def upcast(attn) -> bool:
    """Whether the forward takes GPT-2's own upcast call: the module's flag under eager, the test the forward makes."""
    module = attn._module
    return bool(module.reorder_and_upcast_attn) and module.config._attn_implementation == "eager"


def at_call(rest: str):
    """A key inside the attention call the forward makes: ``source.<call>.<rest>``, the call `UPCAST` or `INTERFACE`."""

    def locate(attn) -> str:
        return f"source.{UPCAST if upcast(attn) else INTERFACE}.{rest}"

    locate.__name__ = f"call.{rest}"
    return locate


def probabilities(attn) -> str:
    """The pattern after the dropout that follows the softmax: the upcast method's own ``attn_dropout``, the interface's functional dropout."""
    if upcast(attn):
        return f"source.{UPCAST}.source.self_attn_dropout_0.output"
    return f"source.{INTERFACE}.source.nn_functional_dropout_0.output"


def argument(position: int):
    """``select`` for the call's ``position``-th tensor (queries 0, keys 1, values 2): the interface takes the module first."""

    def select(attn) -> int:
        return position if upcast(attn) else position + 1

    select.__name__ = f"argument({position})"
    return select


class Attention(Attention):
    """GPT-2's attention; the shared eager forward and the residual added in the block, so the base holds.

    A checkpoint with ``reorder_and_upcast_attn`` takes GPT-2's own
    ``_upcast_and_reordered_attn(query, key, value, attention_mask)`` under
    eager (`UPCAST`) in place of the interface call: the same queries, keys
    and values without the module argument, the scores scaled and computed in
    float32 (``baddbmm``) and then masked, the softmax, the cast back, the
    module's ``attn_dropout``, and the head outputs tokens first, as the
    interface returns them. The interior values and `interface` are read at
    whichever call the module's flag picks, so they mean the same on both; the
    scores there are float32 whatever the model's dtype.
    """

    def no_rotary(self) -> str | None:
        return "no rotary embedding: GPT-2 adds learned absolute position embeddings (wpe) to the stream"

    @EProperty(at_call("inputs"), select=argument(0), description=Attention.attention_queries.description, unavailable=interface_reason)
    def attention_queries(self, value) -> Queries:
        return value

    @EProperty(at_call("inputs"), select=argument(1), description=Attention.attention_keys.description, unavailable=interface_reason)
    def attention_keys(self, value) -> Keys:
        return value

    @EProperty(at_call("inputs"), select=argument(2), description=Attention.attention_values.description, unavailable=interface_reason)
    def attention_values(self, value) -> Values:
        return value

    @EProperty(at_call("source.nn_functional_softmax_0.input"), description=Attention.attention_scores.description, unavailable=interface_reason)
    def attention_scores(self, value) -> Pattern:
        return value

    @EProperty(probabilities, description=Attention.attention_probabilities.description, unavailable=interface_reason)
    def attention_probabilities(self, value) -> Pattern:
        return value

    @EProperty(at_call("output"), select=0, description=Attention.attention_head_outputs.description, unavailable=interface_reason)
    def attention_head_outputs(self, value) -> HeadOutputs:
        return value

    def value_weight(self, kv_head):
        """``c_attn`` is a ``Conv1D``, ``[hidden_size, 3 * hidden_size]``, whose columns are every query, key, then value."""
        return projection_weight(self, "c_attn").unflatten(1, (3, self.num_heads, self.head_dim))[:, 2, kv_head].T

    def output_weight(self, head):
        """``c_proj`` is a ``Conv1D``, ``[hidden_size, hidden_size]``, whose rows are the heads' outputs."""
        return projection_weight(self, "o_proj").unflatten(0, (self.num_heads, self.head_dim))[head].T


class Mlp(Mlp):
    """GPT-2's MLP; the residual is added in the block, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {GPT2Block: Layer, GPT2Attention: Attention, GPT2MLP: Mlp}


# -- sizes: what GPT-2's config calls them ------------------------------------------

def intermediate_size(model: "StandardizedTransformer") -> int:
    """The MLP width is ``n_inner``, ``None`` meaning four times the hidden size; the config's ``intermediate_size`` is never read by the model."""
    return model.config.n_inner or 4 * model.hidden_size
