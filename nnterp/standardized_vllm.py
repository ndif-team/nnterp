"""`StandardizedVLLM`: nnsight's ``VLLM`` engine renamed to the standard vocabulary."""

from __future__ import annotations

import contextlib
import copy
from types import ModuleType
from typing import Any

import torch
from nnsight.modeling.vllm import VLLM

from . import families, order as forward_order
from .components import EProperty, Residual, Unavailable, unavailable
from .components.standard import values
from .components.vllm import Flat, argument, on_transformers_backend, project
from .standardized import Logits, NextTokenProbs, Standardized, StandardizedCapability, StandardizedTransformer, Tokens

#: vLLM's engine arguments that say where the checkpoint is and how to fetch it, by the name transformers gives each.
HUB_ARGUMENTS = {
    "revision": "revision",
    "code_revision": "code_revision",
    "trust_remote_code": "trust_remote_code",
    "download_dir": "cache_dir",
    "hf_token": "token",
}


def hub_kwargs(kwargs: dict) -> dict:
    """The load's arguments that say where the checkpoint is and how to fetch it, under transformers' names (`HUB_ARGUMENTS`)."""
    return {HUB_ARGUMENTS[name]: value for name, value in kwargs.items() if name in HUB_ARGUMENTS and value is not None}


def past_the_head(model: Any) -> str | None:
    """Why ``lm_head_output`` is unavailable: what vLLM's logits processor applies after the projection, or ``None``.

    The processor projects with the head's weight (adding the head's bias where
    the model passes one) and then softcaps and scales in the same call, so
    where it does either the projection alone is never a tensor.
    """
    processor = model.logits_processor._module
    after = []
    if processor.soft_cap is not None:
        after.append(f"softcaps it at {processor.soft_cap}")
    if processor.scale != 1.0:
        after.append(f"multiplies it by {processor.scale}")
    if not after:
        return None
    return (
        f"vLLM never calls lm_head: its logits processor projects with lm_head's weight and {' and '.join(after)} "
        "in the same call, so the head's own projection is never a tensor; read logits"
    )


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
      ``layer_output`` is ``[1, tokens, hidden_size]``, the whole prompt on the
      prefill and one token on each decode step (``tracer.iter``).
    * ``logits`` is this step's, ``[1, 1, vocab_size]``: the engine computes the
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
    * ``lm_head_output`` is `logits` where vLLM's logits processor applies
      nothing past the head, and unavailable where it softcaps or scales,
      naming which: vLLM never calls ``lm_head``.

    Outside a trace, `order` and `rank` give the forward order of the values
    this engine serves, measured on the transformers model of the checkpoint
    (a weightless `StandardizedTransformer`, built on first use): the values
    sit at the same module boundaries on both engines. `generation_config` is
    the checkpoint's ``transformers.GenerationConfig``.

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

        self._order_model = None  # the transformers model `order` and `rank` read: see `_transformers_model`
        self._generation_config = None
        if family is None:
            config = self._read_config(repo_id, **hub_kwargs(kwargs))
            family = families.resolve(config, engine="vllm")
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

    @EProperty("logits", description="This step's logits, [1, 1, vocab_size]: the last position, before sampling")
    def logits(self, value: torch.Tensor) -> Logits:
        """The logits of this request's current step, ``[1, 1, vocab_size]``.

        The engine computes logits for the last position only: the prompt's
        on the prefill, the newest token's on each decode step. Softcapping
        is applied (vLLM's logits processor does it). Assigning, or editing in
        place, changes what the sampler sees.
        """
        return value.unsqueeze(1)

    @logits.postprocess
    def logits(self, value: torch.Tensor) -> torch.Tensor:
        return value.squeeze(1)

    @EProperty("logits", description="The next-token distribution, [1, vocab_size]; derived, read-only")
    def next_token_probs(self, value: torch.Tensor) -> NextTokenProbs:
        """The next-token distribution of this step, ``[1, vocab_size]``: the softmax of `logits`. Read-only."""
        return value.softmax(-1)

    @next_token_probs.postprocess
    def next_token_probs(self, value: Any) -> Any:
        raise AttributeError(
            "next_token_probs is derived from the logits and cannot be assigned; "
            "assign model.logits instead"
        )

    @Flat("embed_tokens.output", batch=on_transformers_backend, description="The token embeddings entering the first block, [1, tokens, hidden_size]")
    def token_embeddings(self, value: torch.Tensor) -> Residual:
        """The embedding module's output, ``[1, tokens, hidden_size]``; what a family scales or adds positions to comes after."""
        return value

    @Flat("norm.output", batch=on_transformers_backend, description="The residual stream after the final norm, what the head reads, [1, tokens, hidden_size]")
    def final_norm_output(self, value: torch.Tensor) -> Residual:
        """The final norm's output, ``[1, tokens, hidden_size]``: every position of the step, of which the engine projects the last.

        A family whose blocks hand the stream on in two halves calls the final
        norm with both, and the norm returns ``(normed, residual)``: this is
        the normed half. Assign to replace it; in-place edits reach the model.
        """
        return value

    @EProperty(
        "logits",
        description="The head's logits of this step, [1, 1, vocab_size]: the logits, where vLLM applies no softcap or scale past the head",
        unavailable=past_the_head,
    )
    def lm_head_output(self, value: torch.Tensor) -> Logits:
        """The head's projection of this step's last position, ``[1, 1, vocab_size]``: `logits`, at the same location.

        vLLM never calls ``lm_head``: its logits processor projects with the
        head's weight (and the head's bias, on GPT-J and Phi) and applies any
        softcap and scale in the same call. Where it applies neither, the
        logits are the head's projection, as ``lm_head.output`` and ``logits``
        are one tensor on transformers there; read, edit in place and assign
        it as `logits`. Where it applies one (Gemma-2's softcap; the scale of
        Cohere, Granite and HyperCLOVA X checkpoints), the projection alone is
        never a tensor, and this is unavailable, naming what follows the head.
        """
        return value.unsqueeze(1)

    @lm_head_output.postprocess
    def lm_head_output(self, value: torch.Tensor) -> torch.Tensor:
        return value.squeeze(1)

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

    # -- forward order (outside a trace) -------------------------------------------

    def order(self, layer: int | None = None) -> dict[str, int]:
        """The values this engine serves in forward order: name -> rank; equal ranks are one location, either read first.

        `StandardizedTransformer.order` of the transformers model of the
        checkpoint (`_transformers_model`), kept to the values `support` lists
        as available here: the standard values sit at the same module
        boundaries on both engines, so the forward reaches them in the same
        order. With ``layer``, that block's values by dotted name; without,
        the root's, numbered across the whole forward. The ranks are the
        transformers model's, so a value this engine lacks leaves a gap.
        """
        table = self._transformers_model().order(layer)
        served = self._served(layer)
        return {name: rank for name, rank in table.items() if served.get(name, "absent") is None}

    def rank(self, name: str, layer: int | None = None) -> tuple[int, int]:
        """Where a value falls in the forward: ``(side, r)``, which sorts reads into forward order.

        `StandardizedTransformer.rank` of the transformers model of the
        checkpoint (`_transformers_model`), for a value this engine serves:
        ``side`` is ``-1`` or ``num_layers`` for a root value served before or
        after the blocks, else ``layer``; ``r`` is `order`'s rank.

        Raises:
            Unavailable: this engine does not serve the value on the checkpoint
                (`support` gives the reason), or the transformers model does not.
            KeyError: there is no value of that name (a block value needs ``layer``).
            NotImplementedError: no transformers family covers the checkpoint.
        """
        reference = self._transformers_model()
        if layer is not None:
            layer = range(self.num_layers)[layer]  # negative layers count from the end; IndexError past it
        served = self._served(layer)
        if name not in served:
            where = "the root" if layer is None else f"block {layer}"
            raise KeyError(f"{name!r} is not a value of {where}; a block value takes layer=, by its dotted support() name")
        if served[name]:
            raise Unavailable(f"{name} is not available{'' if layer is None else f' on block {layer}'}: {served[name]}")
        try:
            return reference.rank(name, layer)
        except Unavailable as error:
            raise Unavailable(f"{error}, on the transformers model of this checkpoint, where the forward order is measured") from None

    def _served(self, layer: int | None) -> dict[str, Any]:
        """`support`'s rows for one block, or for the root's own values (without the blocks')."""
        if layer is None:
            return {name: value.reason(self) for name, value in values(type(self)).items()}
        return self.support(layer)

    def _transformers_model(self) -> StandardizedTransformer:
        """The weightless transformers model of this checkpoint whose forward order `order` and `rank` give; built and measured on first use.

        Loaded as transformers loads the checkpoint (as its vision-language
        wrapper, ``task="image-text-to-text"``, where transformers has no
        causal-LM class for the config), with the load's hub arguments
        (`hub_kwargs`), eager attention and ``dispatch=False``: its probe
        runs on meta tensors, except where the forward cannot
        (`nnterp.order.compute`), and then it loads the weights on the CPU.

        Raises:
            NotImplementedError: no transformers family covers the checkpoint's
                ``model_type`` (nor its text config's).
        """
        if self._order_model is None:
            from transformers.models.auto.modeling_auto import (
                MODEL_FOR_CAUSAL_LM_MAPPING,
                MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING,
            )

            config = self.config
            text = getattr(config, "text_config", config)
            if not (families.covers(config.model_type) or families.covers(text.model_type)):
                raise NotImplementedError(
                    f"model.order and model.rank are measured on the transformers model of the checkpoint, and no "
                    f"transformers family covers model_type {config.model_type!r}"
                )
            wrapper = type(config) not in MODEL_FOR_CAUSAL_LM_MAPPING and type(config) in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING
            built = contextlib.nullcontext()
            if on_transformers_backend(self):
                from vllm.config import CompilationConfig, CompilationMode, set_current_vllm_config

                # vLLM's transformers backend decorates transformers' decoder class in place for its compiler, and the
                # decorated class reads vLLM's current config whenever it is built (here, and again where the probe
                # loads the weights): the backend model's own config, with nothing compiled.
                engine = copy.copy(self._module.vllm_config)
                engine.compilation_config = CompilationConfig(mode=CompilationMode.NONE)
                built = set_current_vllm_config(engine)
            with built:
                model = StandardizedTransformer(
                    self.args[0],
                    dispatch=False,
                    device_map=None,
                    attn_implementation="eager",
                    task="image-text-to-text" if wrapper else "text-generation",
                    **hub_kwargs(self.kwargs),
                )
                forward_order.cached(model)
            self._order_model = model
        return self._order_model

    # -- generation (outside a trace) ----------------------------------------------

    @property
    def generation_config(self) -> Any:
        """The checkpoint's ``transformers.GenerationConfig``, loaded once on first use.

        Read from the repo's ``generation_config.json`` with the load's hub
        arguments, or made from the model's config
        (``GenerationConfig.from_model_config``) where the repo has none, as
        transformers does when it loads a model: the checkpoint's EOS ids and
        its own sampling settings. A trace samples with vLLM's defaults unless
        it is given settings (``trace(..., temperature=0.0)``), not with these.
        """
        if self._generation_config is None:
            from transformers import GenerationConfig

            hub = {key: value for key, value in hub_kwargs(self.kwargs).items() if key in ("revision", "cache_dir", "token")}
            try:
                self._generation_config = GenerationConfig.from_pretrained(self.args[0], **hub)
            except OSError:  # the repo has no generation_config.json
                self._generation_config = GenerationConfig.from_model_config(self.config)
        return self._generation_config

    # -- remote ------------------------------------------------------------------

    def _remoteable_class(self) -> type:
        """The class in this model's remote key: ``VLLM``, what a server deploys (see `StandardizedTransformer`)."""
        return VLLM

    def __getstate__(self) -> dict:
        state = super().__getstate__()
        # A block that names the model ships it to the engine's worker: the transformers model `order` reads and the
        # generation config stay here (the first would resolve its own modules to the engine's on the far side).
        state["_order_model"] = None
        state["_generation_config"] = None
        return state
