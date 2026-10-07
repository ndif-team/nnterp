"""The base envoys of a family vLLM runs through its transformers backend: transformers' own modules, inside vLLM.

vLLM has no implementation of its own for some architectures (OLMo, OLMo 2,
SmolLM3, StarCoder2, GPT-BigCode, VaultGemma); its registry sends them to
``TransformersForCausalLM``, which builds transformers' model and runs it.
The block, the attention and the MLP are transformers' classes, called the
way transformers calls them, so the values sit where they sit on transformers.
Three things differ, and these envoys absorb them:

* **The batch axis is there.** The backend adds a batch axis of one to the
  engine's flat rows before calling the model and strips it after, so inside
  the model a request's activations are ``[1, tokens, hidden]``. The values
  are served as private copies all the same (`Flat` with ``batch=True``),
  handed back when the block moves on, so they behave as every vLLM value
  does: a saved one stays what it was read as, and an edit reaches the model.
* **The attention layer is vLLM's.** The model is built with
  ``_attn_implementation = "vllm"``: transformers' attention interface hands
  the queries, keys and values to the engine's attention layer for that
  block, which the backend keeps in a dict on the model rather than in the
  module tree. `Attention` mounts it as the module's ``attn`` child the
  first time a block in the worker asks for it (the worker's tree is built
  before any family is in play, and the client's must not hold it, since a
  request names its modules by their paths in the worker's), so it is the
  layer the vLLM `Attention` base reads: the queries, keys and values are
  its inputs and the head outputs its output, ``[tokens, heads * head_dim]``
  as on every vLLM family, and the pattern is recomputed from them, on the
  prefill only.
* **Some modules are vLLM's.** The backend fuses the query, key and value
  projections into one ``qkv_proj`` (and a gated MLP's into ``gate_up_proj``),
  rewriting the module's forward to call it, and swaps RMSNorms for vLLM's.
  The values do not read inside those forwards: nnsight's ``.source``
  instruments a module from its class's source, which still calls the
  projections the fusion removed.
"""

from __future__ import annotations

from typing import Any

import torch
from nnsight.intervention.envoy import Envoy

from .. import layer as standard
from ..layer import Residual
from .attention import Attention as VLLMAttention
from .flat import Flat
from .mlp import Mlp as VLLMMlp


def on_transformers_backend(envoy: Envoy) -> bool:
    """Whether ``envoy``'s model runs on vLLM's transformers backend, where every activation carries the batch axis."""
    from vllm.model_executor.models.transformers.base import Base

    return isinstance(envoy.root._module, Base)


def engine_attention(envoy: Envoy) -> torch.nn.Module | None:
    """The engine's attention layer for ``envoy``'s attention module, or ``None`` on a model with none.

    The backend keeps one per block in the model's ``attention_instances``,
    keyed by the block index transformers' attention carries as
    ``layer_idx``. The model is the root of the tree the module sits in: the
    request's own, or, when the request carries only part of it (a block
    bound outside the trace), the tree the worker built over the engine's model.
    """
    for tree in (envoy, envoy.interleaver.envoys.get(id(envoy._module))):
        instances = getattr(tree.root._module, "attention_instances", None) if tree is not None else None
        if instances is not None:
            return instances[envoy._module.layer_idx]
    return None


class Layer(standard.Layer):
    """A transformers block on vLLM: called with the stream ``[1, tokens, hidden]``, returning it."""

    @Flat("input", batch=True, description="The residual stream entering the block, [1, tokens, hidden]")
    def layer_input(self, value: torch.Tensor) -> Residual:
        return value

    @Flat("output", select=lambda envoy: 0 if envoy.returns_tuple else None, batch=True, description="The residual stream leaving the block, [1, tokens, hidden]")
    def layer_output(self, value: torch.Tensor) -> Residual:
        return value


class Attention(VLLMAttention):
    """A transformers attention module on vLLM, with the engine's attention layer mounted as its ``attn`` child.

    It returns ``(attn_output, attn_weights)``; the first is what the block
    adds, unless a family's block norms it first.
    """

    def __getattr__(self, name: str) -> Any:
        """``attn``: the engine's attention layer, mounted as this module's child in the worker, on first use.

        The worker builds its tree over the engine's model before any family
        is in play, so the layer is no module of that tree until a block
        mounts it, which it does inside a trace: the client's tree, which is
        what a request carries to the worker, never holds it.
        """
        if name == "attn" and self.interleaver.interleaving:
            layer = engine_attention(self)
            if layer is not None:
                return self._add_envoy("attn", layer)
        return super().__getattr__(name)

    @Flat("output", select=0, batch=True, description="What the attention adds to the residual stream, [1, tokens, hidden]")
    def attention_output(self, value: torch.Tensor) -> Residual:
        return value


class Mlp(VLLMMlp):
    """A transformers MLP on vLLM: its output is what the block adds, unless a family's block norms it first."""

    @Flat("output", batch=True, description="What the MLP adds to the residual stream, [1, tokens, hidden]")
    def mlp_output(self, value: torch.Tensor) -> Residual:
        return value
