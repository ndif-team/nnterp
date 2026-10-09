"""`StandardizedVLLM`: nnsight's ``VLLM`` engine renamed to the standard vocabulary."""

from __future__ import annotations

from types import ModuleType
from typing import Any

import torch
from nnsight.modeling.vllm import VLLM

from . import families
from .components import EProperty, Residual, unavailable
from .components.vllm import Flat, argument, on_transformers_backend, project
from .standardized import Logits, NextTokenProbs, Standardized, StandardizedCapability, Tokens


class StandardizedVLLM(Standardized, VLLM):
    """A causal language model on the vLLM engine whose modules answer to the standard names.

    `StandardizedTransformer`'s counterpart for nnsight's ``VLLM``: the same
    vocabulary (``model.layers[i].self_attn``, ``.mlp``, ``model.norm``,
    ``model.lm_head``), the same values with the same layouts, and the same
    methods, over the engine's continuous batching. The family is vLLM's own
    implementation of the checkpoint's ``model_type``, a module under
    `nnterp.families.vllm`.

    It is nnsight's ``VLLM`` otherwise, with vLLM's defaults: a trace is a
    generation request (``max_tokens=16`` and ``temperature=1.0`` unless you
    say), sampling settings go on ``trace``/``invoke``, one prompt per invoke,
    and a block runs in the engine's worker, where nnterp has to be importable.
    What that changes for the values:

    * A request is one sequence, so every value's batch axis is 1:
      ``layer_output`` is ``[1, tokens, hidden]``, the whole prompt on the
      prefill and one token on each decode step (``tracer.iter``).
    * ``logits`` is this step's, ``[1, 1, vocab]``: the engine computes the
      last position only. ``logits[:, -1]`` and ``next_token_probs`` read the
      same on both engines.
    * Values are private copies, so a saved one stays what it was read as.
    * The queries, keys, values and head outputs are what goes into and
      comes out of the engine's attention layer, for this step's tokens. The
      scores and the pattern are inside its kernel, so they are recomputed
      from the queries and keys: read-only, and on the prefill only (a decode
      step raises `Unavailable`). There is no ``attention_mask``.
    * `project_on_vocab` runs the engine's modules, so it works inside a
      trace only.

    Args:
        repo_id: A HuggingFace repo id.
        family: The family to use instead of the one looked up: a module or
            any object with ``RENAME`` and ``ENVOYS`` written against vLLM's
            module classes (and any size function or ``project_on_vocab`` it
            defines). It applies to this model only; the config is not read
            for it, and no model type is named or checked.
        rename: Extra aliases, merged over the family's; a key given here wins.
        envoys: Extra ``envoys=`` entries, merged over the family's ``ENVOYS``
            and nnsight's envoys for vLLM's parallel layers; a key given here wins.
        **kwargs: Passed through to ``VLLM`` (``dispatch``, ``mode``, ``taps``,
            and vLLM's engine arguments).

    Attributes:
        family: The vLLM family the checkpoint resolved to, or the ``family`` passed.

    Raises:
        UnsupportedFamily: when no family is passed and vLLM's implementation
            of the checkpoint's ``model_type`` has no family under
            `nnterp.families.vllm`.
    """

    def __init__(
        self,
        repo_id: str,
        *args: Any,
        family: ModuleType | None = None,
        rename: dict[str, str | list[str]] | None = None,
        envoys: dict | None = None,
        **kwargs: Any,
    ) -> None:
        from nnsight.modeling.vllm.envoys import parallel_envoys

        if family is None:
            config = self._read_config(repo_id, kwargs)
            family = families.lookup(getattr(config, "text_config", config).model_type, engine="vllm")
        self.family = family
        super().__init__(
            repo_id,
            *args,
            rename={**self.family.RENAME, **(rename or {})},
            envoys={**parallel_envoys(), **self.family.ENVOYS, **(envoys or {})},
            **kwargs,
        )

    @property
    def config(self) -> Any:
        """The checkpoint's transformers config, as vLLM's model holds it."""
        return self._module.config

    # -- whole-model values (inside a trace) ---------------------------------

    @EProperty("logits", description="This step's logits, [1, 1, vocab]: the last position, before sampling")
    def logits(self, value: torch.Tensor) -> Logits:
        """The logits of this request's current step, ``[1, 1, vocab]``.

        The engine computes logits for the last position only: the prompt's
        on the prefill, the newest token's on each decode step. Softcapping
        is applied (vLLM's logits processor does it). Assigning, or editing in
        place, changes what the sampler sees.
        """
        return value.unsqueeze(1)

    @logits.postprocess
    def logits(self, value: torch.Tensor) -> torch.Tensor:
        return value.squeeze(1)

    @EProperty("logits", description="The next-token distribution, [1, vocab]; derived, read-only")
    def next_token_probs(self, value: torch.Tensor) -> NextTokenProbs:
        """The next-token distribution of this step, ``[1, vocab]``: the softmax of `logits`. Read-only."""
        return value.softmax(-1)

    @next_token_probs.postprocess
    def next_token_probs(self, value: Any) -> Any:
        raise AttributeError(
            "next_token_probs is derived from the logits and cannot be assigned; "
            "assign model.logits instead"
        )

    @Flat("embed_tokens.output", batch=on_transformers_backend, description="The token embeddings entering the first block, [1, tokens, hidden]")
    def token_embeddings(self, value: torch.Tensor) -> Residual:
        """The embedding module's output, ``[1, tokens, hidden]``; what a family scales or adds positions to comes after."""
        return value

    # -- the input (inside a trace) ----------------------------------------------

    @EProperty("inputs", description="The token ids of this step, [1, tokens]; read-only")
    def input_ids(self, value: tuple) -> Tokens:
        """The token ids the engine runs this step, ``[1, tokens]``: the prompt on the prefill, then one token a step."""
        return argument(value, 0, "input_ids").clone().unsqueeze(0)

    @input_ids.postprocess
    def input_ids(self, value: Any) -> Any:
        raise AttributeError("input_ids is read-only on vLLM; trace the ids you want, or assign token_embeddings")

    @EProperty("inputs", description="[1, tokens] of this step; read-only")
    def input_size(self, value: tuple) -> torch.Size:
        """``[1, tokens]`` of the current step, from the ids; read-only."""
        return torch.Size((1, *argument(value, 0, "input_ids").shape))

    @input_size.postprocess
    def input_size(self, value: Any) -> Any:
        raise AttributeError("input_size is the ids' shape and cannot be assigned")

    attention_mask = unavailable("a vLLM request is one unpadded sequence; there is no mask")

    # -- the logit lens -----------------------------------------------------------

    @StandardizedCapability
    def project_on_vocab(self, hidden: torch.Tensor) -> torch.Tensor:
        """Logits for a residual-stream tensor: the final norm, ``lm_head``, and vLLM's logits processor.

        The logit lens, inside a trace: the modules are the engine's, so there
        is nothing to call outside one. The processor applies the model's
        softcapping and scaling where it has them. ``hidden`` is ``[...,
        hidden]``; the leading dimensions are kept. A family whose model
        projects another way defines ``def project_on_vocab(model, hidden)``
        in its module.
        """
        return project(self, hidden, self.lm_head)

    # -- forward order -------------------------------------------------------------

    def order(self, layer: int | None = None) -> dict[str, int]:
        """Not supported on the vllm engine: the forward order is measured on `StandardizedTransformer` only."""
        raise NotImplementedError("model.order is not supported on the vllm engine; it is measured on a StandardizedTransformer")

    def rank(self, name: str, layer: int | None = None) -> tuple[int, int]:
        """Not supported on the vllm engine: the forward order is measured on `StandardizedTransformer` only."""
        raise NotImplementedError("model.rank is not supported on the vllm engine; it is measured on a StandardizedTransformer")

    # -- remote ------------------------------------------------------------------

    def _remoteable_class(self) -> type:
        """The class in this model's remote key: ``VLLM``, what a server deploys (see `StandardizedTransformer`)."""
        return VLLM
