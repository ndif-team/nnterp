"""`Attention`: a softmax-attention module, its contribution, its pattern and its interior."""

from __future__ import annotations

from typing import Any

import torch
from nnsight.intervention.envoy import Envoy

from jaxtyping import Float
from torch import Tensor

from .eproperty import EProperty
from .layer import Residual
from .standard import Standard, first_tensor, in_width, module_int, rewrap, unsized


def needs_eager(envoy: Envoy) -> str | None:
    """Why a value read inside the eager attention forward is unavailable, or ``None``."""
    implementation = envoy._module.config._attn_implementation
    if implementation != "eager":
        return f"read inside the eager attention forward, but this model runs {implementation!r}; load with attn_implementation='eager'"
    return None


#: The call every family on transformers' shared attention path makes:
#: ``attention_interface(module, query, key, value, attention_mask, ...)``.
INTERFACE = "attention_interface_1"

#: The layouts of the interior, as transformers hands them to its attention interface: heads before tokens on
#: the queries, keys and values; ``kv_heads`` wide under grouped-query attention; queries and keys ``qk_head_dim``
#: deep and values ``head_dim`` deep (the same number outside latent attention).
Queries = Float[Tensor, "batch heads seq qk_head_dim"]
Keys = Float[Tensor, "batch kv_heads seq qk_head_dim"]
Values = Float[Tensor, "batch kv_heads seq head_dim"]
#: A pattern over the tokens: the scores entering the softmax and the probabilities leaving it.
Pattern = Float[Tensor, "batch heads query key"]
#: Each head's output before they are concatenated and projected, tokens before heads.
HeadOutputs = Float[Tensor, "batch seq heads head_dim"]

#: The reason a family gives for an interface value it has not mapped onto its own arithmetic.
NOT_ON_INTERFACE = (
    "The attention does its own arithmetic rather than transformers' shared "
    "attention interface; not mapped for this family yet"
)


def seq_first(value: torch.Tensor) -> torch.Tensor:
    """``[batch, heads, seq, head_dim]`` <-> ``[batch, seq, heads, head_dim]``, as a view.

    The standard layout of ``attention_head_outputs`` is what the shared
    interface returns, sequence before heads. A family whose own arithmetic
    keeps heads first serves a transposed view on read, so in-place edits
    still land, and transposes back on write; the transpose is its own inverse.
    """
    return value.transpose(1, 2)


def interface_reason(envoy: Envoy) -> str | None:
    return envoy.off_interface()


class Attention(Standard):
    """A softmax-attention module: its contribution, its pattern, and its interior.

    Everything but ``attention_output`` is read inside transformers' shared
    ``eager_attention_forward``, reached through the module's
    ``attention_interface`` call (`INTERFACE`): the queries, keys and values
    it receives, the scores entering its softmax, the pattern leaving it, and
    the per-head outputs it returns. They are unavailable unless the model was
    loaded with ``attn_implementation="eager"``; `off_interface` is the one
    place that decides, so a family with another reason (GPT-2's
    ``reorder_and_upcast_attn``) overrides that method. A family whose
    attention does its own arithmetic redefines the pattern on its own op and
    marks the rest ``unavailable(NOT_ON_INTERFACE)``.

    Its sizes (`num_heads`, `num_kv_heads`, `head_dim`, `qk_head_dim`) are
    this block's, read off the module, so they hold on a model whose blocks
    differ (Gemma-4, MiMo-V2-Flash) and outside a trace.

    The pattern is the dropout *after* the softmax, not the softmax itself:
    that is the tensor the values are mixed with on every family, after the
    cast back to the model dtype and, on a model with an attention sink
    (GPT-OSS), after the sink column is dropped.
    """

    #: Whether the softmax has an attention sink: the pattern's rows then sum to less than one, by the
    #: mass the sink took, and the scores are read just before the sink column joins them (GPT-OSS).
    SINK = False

    def off_interface(self) -> str | None:
        """Why the shared attention interface does not run on this module, or ``None``."""
        return needs_eager(self)

    # -- sizes (off the module) ---------------------------------------------------
    # What this block runs with, read off the module's own attributes and, where
    # it keeps none, its projections' shapes. A family whose module spells a size
    # its own way overrides the property on its subclass.

    @property
    def head_dim(self) -> int:
        """Width of one head's values and outputs: the module's ``v_head_dim`` (latent attention, MiMo-V2-Flash), else its ``head_dim`` / ``head_size``."""
        size = module_int(self._module, "v_head_dim", "head_dim", "head_size")
        if size is None:
            raise unsized(self, "head_dim")
        return size

    @property
    def qk_head_dim(self) -> int:
        """Width of one head's queries and keys: the module's ``qk_head_dim`` (latent attention), else its ``head_dim`` / ``head_size``."""
        size = module_int(self._module, "qk_head_dim", "head_dim", "head_size")
        if size is None:
            raise unsized(self, "qk_head_dim")
        return size

    @property
    def num_heads(self) -> int:
        """Query heads: the module's ``num_heads`` / ``num_attention_heads`` / ``n_heads`` / ``n_head``, else the output projection's input width over `head_dim`."""
        size = module_int(self._module, "num_heads", "num_attention_heads", "n_heads", "n_head")
        if size is None:
            width = in_width(self._module, "o_proj", "out_proj", "dense", "c_proj", "wo")
            if width is None:
                raise unsized(self, "num_heads")
            size = width // self.head_dim
        return size

    @property
    def num_kv_heads(self) -> int:
        """Key/value heads as projected: the module's ``num_key_value_heads`` / ``num_kv_heads`` / ``kv_heads``, else `num_heads` over its ``num_key_value_groups``, else the key projection's width over `qk_head_dim`, else `num_heads`."""
        module = self._module
        size = module_int(module, "num_key_value_heads", "num_kv_heads", "kv_heads", "n_kv_heads")
        if size is not None:
            return size
        groups = module_int(module, "num_key_value_groups")
        if groups:
            return self.num_heads // groups
        width = getattr(getattr(module, "k_proj", None), "out_features", None)
        return width // self.qk_head_dim if width else self.num_heads

    @EProperty(f"source.{INTERFACE}.inputs", select=1, description="The queries entering attention", unavailable=interface_reason)
    def attention_queries(self, value: torch.Tensor) -> Queries:
        """The queries the attention interface receives, ``[batch, heads, seq, head_dim]``.

        After the query projection and, on a family with rotary embeddings,
        after they are applied. Assign to replace them. In-place edits reach
        the model where torch allows them: GPT-2's queries, keys and values are
        split views of one ``c_attn`` tensor and torch refuses to edit those in
        place, so assign there.
        """
        return value

    @EProperty(f"source.{INTERFACE}.inputs", select=2, description="The keys entering attention", unavailable=interface_reason)
    def attention_keys(self, value: torch.Tensor) -> Keys:
        """The keys the attention interface receives, ``[batch, kv_heads, seq, qk_head_dim]``.

        Before ``repeat_kv``, so under grouped-query attention the head axis
        is ``num_kv_heads`` wide. Assign to replace them; in-place edits reach
        the model except on GPT-2 (see `attention_queries`).
        """
        return value

    @EProperty(f"source.{INTERFACE}.inputs", select=3, description="The values entering attention", unavailable=interface_reason)
    def attention_values(self, value: torch.Tensor) -> Values:
        """The values the attention interface receives, ``[batch, kv_heads, seq, head_dim]``.

        Before ``repeat_kv``, like the keys. Assign to replace them; in-place
        edits reach the model except on GPT-2 (see `attention_queries`).
        """
        return value

    @EProperty(f"source.{INTERFACE}.source.nn_functional_softmax_0.input", description="The attention scores entering the softmax, masked", unavailable=interface_reason)
    def attention_scores(self, value: torch.Tensor) -> Pattern:
        """The scaled, masked scores entering the softmax, ``[batch, heads, query, key]``.

        ``softmax(attention_scores)`` is ``attention_probabilities`` up to the
        dtype cast. Assign to replace them; in-place edits reach the model.
        """
        return value

    @EProperty(key="output", description="What the attention adds to the residual stream")
    def attention_output(self, value: Any) -> Residual:
        """The attention sublayer's contribution to the residual stream.

        The tensor the block adds to its input, as a tensor even when the
        module returns ``(attn_output, attn_weights)``. On a family whose
        attention adds the residual inside the module, the family's subclass
        reads the pre-residual value instead, so this always means the same
        thing. In-place edits and assignment reach the model.
        """
        return first_tensor(value)

    @attention_output.postprocess
    def attention_output(self, value: torch.Tensor) -> Any:
        return rewrap(self, value)

    @EProperty(
        f"source.{INTERFACE}.source.nn_functional_dropout_0.output",
        description="The attention pattern the values are mixed with",
        unavailable=interface_reason,
    )
    def attention_probabilities(self, value: torch.Tensor) -> Pattern:
        """The attention pattern, ``[batch, heads, query, key]``.

        The post-softmax probabilities as the values are mixed with them: in
        the model's dtype, and with an attention sink's column already dropped.
        Rows sum to one (to less than one on a sink model). Read it, edit it in
        place, or assign a tensor of the same shape::

            with model.trace(prompt):
                pattern = model.layers[3].self_attn.attention_probabilities.save()
                model.layers[3].self_attn.attention_probabilities[:, 5] = 0
        """
        return value

    @EProperty(f"source.{INTERFACE}.output", select=0, description="The per-head outputs before the output projection", unavailable=interface_reason)
    def attention_head_outputs(self, value: torch.Tensor) -> HeadOutputs:
        """Each head's output before they are concatenated and projected, ``[batch, seq, heads, head_dim]``.

        What the attention interface returns; the module reshapes it to
        ``[batch, seq, hidden]`` and applies the output projection to get
        ``attention_output``. Assign to replace it, or edit it in place: the
        interface returns a contiguous tensor on most families and a plain
        transposed view on GPT-2, and torch accepts in-place edits on both.
        """
        return value
