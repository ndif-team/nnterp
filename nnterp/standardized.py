"""`StandardizedTransformer`: a `TransformersModel` renamed to the standard vocabulary."""

from __future__ import annotations

from collections import Counter
from types import ModuleType
import functools
from typing import Any, Callable, Sequence

import torch
from jaxtyping import Bool, Float, Int
from nnsight.intervention.envoy import Envoy
from nnsight.modeling.transformers import TransformersModel
from torch import Tensor

from . import families
from .components import EProperty, Layer, Residual, Standard
from .components.standard import blocks_support, standard_children, values
from .components.vision import Vision

#: The layouts of the root's values: the logits, the next-token distribution at the last position, and one
#: integer per token (``input_ids``, ``attention_mask``).
Logits = Float[Tensor, "batch seq vocab"]
NextTokenProbs = Float[Tensor, "batch vocab"]
Tokens = Int[Tensor, "batch seq"]
#: Which positions of the text batch hold an image token: ``image_token_mask``.
ImageTokenMask = Bool[Tensor, "batch seq"]
#: What the text model receives at the image tokens, flat over every image token of the batch in row-major
#: (scatter) order, the text model's ``hidden`` wide: ``image_features``.
ImageFeatures = Float[Tensor, "image_tokens hidden"]

#: The root values a checkpoint has only when images reach the model.
IMAGE_VALUES = ("image_token_mask", "image_features")


def image_token_id(config: Any) -> int | None:
    """The id of the token the processor puts where an image's features go: ``image_token_id`` (``image_token_index`` on older configs)."""
    for name in ("image_token_id", "image_token_index"):
        value = getattr(config, name, None)
        if isinstance(value, int):
            return value
    return None


def text_only(model: Any) -> str | None:
    """Why no image reaches this model, or ``None``: a text-only class, or a load without a processor.

    A value this returns a reason for is left out of `StandardizedTransformer.support` altogether.
    """
    if "projector" not in model._aliases:
        return "a text-only checkpoint or class: the tree has no image projector"
    if getattr(model, "processor", None) is None:
        return "a text-only load: no processor, so no image reaches the model; load with task='image-text-to-text'"
    return None


def no_image_tokens(model: Any) -> str | None:
    """Why ``image_token_mask`` is unavailable, or ``None``."""
    reason = text_only(model)
    if reason is None and image_token_id(model.config) is None:
        reason = "the config names no image_token_id"
    return reason


def no_image_features(model: Any) -> str | None:
    """Why ``image_features`` is unavailable, or ``None``.

    ``image_features`` is the projector's output, which is what the wrapper
    scatters into the text stream on the wrappers the family lists in
    ``IMAGE_WRAPPERS`` (each verified by the suite:
    ``layers[0].input[image_token_mask] == image_features``). A wrapper that
    rearranges the projector's output before the scatter (LLaVA-NeXT's
    unpadding and newline tokens) binds the same names but is not listed, so
    the value says so rather than serving the wrong tensor.
    """
    reason = no_image_tokens(model)
    if reason is None and model.config.model_type not in getattr(model.family, "IMAGE_WRAPPERS", ()):
        reason = (
            f"the {model.config.model_type!r} wrapper is not one whose projector output is known to be what it "
            f"scatters into the text stream (the {model.family.__name__.rsplit('.', 1)[-1]} family lists "
            f"{tuple(getattr(model.family, 'IMAGE_WRAPPERS', ()))})"
        )
    return reason


class StandardizedProperty:
    """A read-only value of the model that a family may define instead.

    Wraps the standard implementation. On read, a function of the same name
    in the model's family module wins (``def num_kv_heads(model): ...`` in
    ``falcon.py``), so a family whose config spells a size its own way keeps
    that knowledge beside its names and values, and the implementation here
    stays the plain case.
    """

    def __init__(self, fget: Callable[[Any], Any]) -> None:
        self.fget = fget
        functools.update_wrapper(self, fget)

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    def __get__(self, obj: Any, owner: type | None = None) -> Any:
        if obj is None:
            return self
        override = getattr(obj.family, self.name, None)
        return override(obj) if override is not None else self.fget(obj)

    def __set__(self, obj: Any, value: Any) -> None:
        raise AttributeError(f"{self.name} is read off the config; a family defines `def {self.name}(model)` to say it otherwise")


class StandardizedCapability(StandardizedProperty):
    """A method of the model that a family may define instead.

    `StandardizedProperty` for a method: on attribute access, a function of
    the same name in the model's family module (``def project_on_vocab(model,
    hidden): ...`` in ``cohere.py``) is bound in place of the standard
    implementation, so a family whose model does something of its own keeps
    that beside its names and values, and the implementation here stays the
    plain case.
    """

    def __get__(self, obj: Any, owner: type | None = None) -> Any:
        if obj is None:
            return self
        return functools.partial(getattr(obj.family, self.name, None) or self.fget, obj)


class StandardizedTransformer(TransformersModel):
    """A causal language model whose modules answer to one set of names.

    The checkpoint's config is read first (its ``model_type``), the matching
    family toolkit is looked up in `nnterp.families.REGISTRY`, and the family's
    ``RENAME`` is handed to nnsight's ``rename`` so every envoy in the tree also
    answers to the standard name. The original names keep working: an alias is
    an extra attribute on the same envoy, not a replacement.

    The family's ``ENVOYS`` wraps its decoder blocks in its `Layer`
    (``layer_output``), its attention modules in its `Attention`
    (``attention_output``, ``attention_probabilities``) and its feed-forwards
    in its `Mlp` (``mlp_output``). The pattern is read inside the eager
    attention forward, so it is unavailable unless the model is loaded with
    ``attn_implementation="eager"``; `support` says which values this
    checkpoint has. See `nnterp.components`.

    Args:
        repo_id: A HuggingFace repo id, or an already-loaded ``torch.nn.Module``
            (its own ``config`` is read then).
        rename: Extra aliases, merged over the family's; a key given here wins.
        envoys: Extra ``envoys=`` entries, merged over the family's ``ENVOYS``
            (and nnsight's tensor-parallel envoys when the load shards); a key
            given here wins.
        tokenizer_kwargs: Attributes to set on the loaded tokenizer, such as
            ``padding_side="left"`` or a ``pad_token``.
        **kwargs: Passed through to `TransformersModel`. ``task`` defaults to
            ``"text-generation"``; pass ``task="image-text-to-text"`` to load a
            multimodal checkpoint as its wrapper, with its processor.

    Over the values, `skip_layers`, `steer`, `project_on_vocab` and
    `get_topk_closest_tokens` do the common things (see each). Inside a
    trace `input_ids`, `input_size` and `attention_mask` are what the model
    was called with.

    The root envoy also answers for the whole model, inside a trace: ``logits``
    (the logits the model returns, after any softcap or scale past the head), ``token_embeddings``
    (the embedding's output) and ``next_token_probs``; and outside one: the
    sizes ``num_layers``, ``num_heads``, ``num_kv_heads``, ``head_dim``,
    ``qk_head_dim``, ``hidden_size``, ``intermediate_size`` and ``vocab_size``,
    read off the config, each a `StandardizedProperty` the family can define
    instead; `project_on_vocab` is a `StandardizedCapability`, a method the family
    can define the same way.

    On an image-text-to-text checkpoint loaded with ``task="image-text-to-text"``
    the text names keep their meaning (the language model's), the vision tower
    is ``vision`` (a `Vision`, with its own blocks and sizes) and the module
    whose output is scattered into the text stream is ``projector``; the root
    adds ``image_token_mask`` and ``image_features``, where
    ``layers[0].input[image_token_mask] == image_features``.

    Attributes:
        family: The toolkit module the checkpoint resolved to.
        layers: The decoder blocks, each a `Layer` (the family's subclass).
        embed_tokens, norm, lm_head: The embedding, the final norm, the unembedding.
        vision: The vision tower, a `Vision`, on a multimodal wrapper whose family names it.
        projector: The module whose output the wrapper scatters into the text stream at the image tokens.

    Raises:
        UnsupportedFamily: when no family covers the checkpoint's ``model_type``.
    """

    family: ModuleType
    layers: Sequence[Layer]
    embed_tokens: Envoy
    norm: Envoy
    lm_head: Envoy
    vision: Vision
    projector: Envoy

    def __init__(
        self,
        repo_id: Any,
        *args: Any,
        rename: dict[str, str | list[str]] | None = None,
        envoys: dict | None = None,
        tokenizer_kwargs: dict | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("task", "text-generation")
        self._add_prefix_false_tokenizer = None
        config = self._read_config(repo_id, kwargs)
        # A multimodal checkpoint's config nests the language model's; the
        # text-generation task builds that model, so its family is the one.
        self.family = families.lookup(getattr(config, "text_config", config).model_type)
        super().__init__(
            repo_id,
            *args,
            rename={**self.family.RENAME, **(rename or {})},
            envoys={
                **self._base_envoys(repo_id, kwargs),
                **self.family.ENVOYS,
                **(envoys or {}),
            },
            **kwargs,
        )
        for key, value in (tokenizer_kwargs or {}).items():
            setattr(self.tokenizer, key, value)

    @staticmethod
    def _base_envoys(repo_id: Any, kwargs: dict) -> dict:
        """What `TransformersModel` would have installed had we passed no ``envoys``.

        It only sets its tensor-parallel envoys as a default, so passing our own
        map would silently drop them on a sharded load; start from them instead.
        """
        from nnsight.modeling.tp.envoys import tp_envoys, wants_tensor_parallel

        return tp_envoys() if wants_tensor_parallel(repo_id, kwargs) else {}

    # -- whole-model values (inside a trace) ---------------------------------

    @EProperty(key="output", description="The logits the model returns, after anything it does past lm_head (a softcap, a scale)")
    def logits(self, value: Any) -> Logits:
        """The logits the model returns, ``[batch, seq, vocab]``.

        Read off the model's output, so a family that softcaps after
        ``lm_head`` (Gemma-2) is already accounted for; ``lm_head.output`` is
        the raw projection. Assigning replaces the logits in the model's output.
        """
        return value.logits

    @logits.postprocess
    def logits(self, value: torch.Tensor) -> Any:
        output = self.output
        output.logits = value
        return output

    @EProperty("embed_tokens.output", description="The embedding module's output; layers[0].input is what enters the first block")
    def token_embeddings(self, value: torch.Tensor) -> Residual:
        """The embedding module's output, ``[batch, seq, hidden]``.

        Positional embeddings and embedding norms (GPT-2's ``wpe``, BLOOM's
        ``word_embeddings_layernorm``) are applied after this by the families
        that have them. Assign to replace it.
        """
        return value

    @EProperty(key="output", description="The next-token distribution at the last position; derived, read-only")
    def next_token_probs(self, value: Any) -> NextTokenProbs:
        """The next-token distribution at the last position, ``[batch, vocab]``.

        ``logits[:, -1].softmax(-1)``, derived from the model's output; the
        last position is the last token of every row only under left padding.
        Read-only: there is no inverse, so assign ``logits`` instead.
        """
        return value.logits[:, -1].softmax(-1)

    @next_token_probs.postprocess
    def next_token_probs(self, value: Any) -> Any:
        raise AttributeError(
            "next_token_probs is derived from the logits and cannot be assigned; "
            "assign model.logits instead"
        )

    # -- methods over the values (inside a trace unless said otherwise) ----------

    def skip_layers(self, start: int, end: int, skip_with: torch.Tensor | None = None) -> None:
        """Skip blocks ``start`` through ``end`` inclusive.

        The residual stream entering block ``start`` (or ``skip_with``) is
        handed straight to block ``end + 1``; the skipped blocks do not run.
        Negative indices count from the end. Inside a trace::

            with model.trace(prompt):
                model.skip_layers(4, 7)
                logits = model.logits.save()
        """
        start, end = range(self.num_layers)[start], range(self.num_layers)[end]
        hidden = self.layers[start].input if skip_with is None else skip_with
        for i in range(start, end + 1):
            self.layers[i].skip_with(hidden)

    def steer(
        self,
        layers: int | list[int],
        vector: torch.Tensor,
        factor: float = 1.0,
        token_positions: int | list[int] | slice | None = None,
        batch_index: int | None = None,
    ) -> None:
        """Add ``factor * vector`` to the residual stream leaving the given blocks.

        ``vector`` is ``[hidden]`` (or broadcastable to the selected slice).
        ``token_positions`` restricts the positions and ``batch_index`` the
        row; both default to all. The add is in place on ``layer_output``, so
        it reaches the model. Inside a trace, with ``layers`` ascending.
        """
        rows = slice(None) if batch_index is None else batch_index
        cols = slice(None) if token_positions is None else token_positions
        for i in [layers] if isinstance(layers, int) else layers:
            out = self.layers[i].layer_output
            out[rows, cols] += factor * vector.to(out)

    @StandardizedCapability
    def project_on_vocab(self, hidden: torch.Tensor) -> torch.Tensor:
        """Logits for a residual-stream tensor: the final norm, ``lm_head``, and what the model does after the head.

        The logit lens: applied to a block's ``layer_output`` it reads that
        layer's prediction; applied to the last block's, it is `logits`.
        Works inside a trace on a live value and outside on a saved one.
        After the head the plain case is the text config's
        ``final_logit_softcapping``, when set (Gemma-2's own config; a
        multimodal checkpoint's ``text_config``). A family whose model does
        something else there (Cohere multiplies by ``logit_scale``, Granite
        divides by ``logits_scaling``) defines
        ``def project_on_vocab(model, hidden)`` in its module, which is bound
        in this one's place (`StandardizedCapability`).
        """
        logits = self.lm_head(self.norm(hidden))
        cap = getattr(self.config.get_text_config(), "final_logit_softcapping", None)
        return cap * torch.tanh(logits / cap) if cap else logits

    def probs_to_dict(self, probs: torch.Tensor, k: int = 5) -> dict[str, float]:
        """The ``k`` most likely tokens of one ``[vocab]`` distribution, as ``{token: probability}``, most likely first.

        Keyed by the decoded text. Where two or more of the ``k`` decode to
        the same text (partial UTF-8 byte tokens all decode to ``'\ufffd'``),
        those entries are keyed by the tokenizer's raw vocabulary token
        instead (``convert_ids_to_tokens``, unique per id), with the id
        appended should even that repeat, so the dict always holds ``k``
        entries, each with its own probability.
        """
        values, indices = probs.topk(k)
        ids = indices.tolist()
        keys = [self.tokenizer.decode(index) for index in ids]
        counts = Counter(keys)
        keys = [
            str(self.tokenizer.convert_ids_to_tokens(index)) if counts[key] > 1 else key
            for key, index in zip(keys, ids)
        ]
        counts = Counter(keys)
        keys = [f"{key}#{index}" if counts[key] > 1 else key for key, index in zip(keys, ids)]
        return {key: value.item() for key, value in zip(keys, values)}

    def get_topk_closest_tokens(self, hidden: torch.Tensor, k: int = 5) -> list[dict[str, float]]:
        """The ``k`` most likely next tokens for each position of ``hidden``, ``[..., hidden]``.

        `project_on_vocab` then softmax, one ``{token: probability}`` per
        position, in row-major order over the leading dimensions.
        """
        probs = self.project_on_vocab(hidden).softmax(-1).reshape(-1, self.vocab_size)
        return [self.probs_to_dict(row, k) for row in probs]

    # -- availability ----------------------------------------------------------

    def support(self, layer: int | None = None) -> dict[str, Any]:
        """Which standard values this checkpoint has, without running anything.

        With ``layer``, that block's values by dotted name (``"layer_output"``,
        ``"self_attn.attention_probabilities"``, ``"mlp.mlp_output"``): ``None``
        when available, else the reason, including ``"no <module> module"``
        when the block has no such module at all (a hybrid's blocks have either
        ``self_attn`` or ``linear_attn``). Without, the root's values
        plus every block value: ``None`` when available on every block, else
        ``{layer: reason}`` for the blocks where it is not, so a hybrid reads
        as a short dict.

        The tree decides what is listed: every child of a block that carries
        standard values (a `Standard` envoy) is walked under its standard
        name, so a value installed through ``envoys=`` or a registered family
        appears here as it does in the envoy's own `Standard.support`, and a
        module no block has (OPT's ``mlp``) has no entry. The image values
        (``image_token_mask``, ``image_features``) are listed only where images
        reach the model: a multimodal checkpoint loaded with its processor. A
        vision tower's values are ``model.vision.support()``.
        """
        if layer is not None:
            return blocks_support(self.layers, layer)
        hidden = IMAGE_VALUES if text_only(self) else ()
        support: dict[str, Any] = {
            name: value.reason(self) for name, value in values(type(self)).items() if name not in hidden
        }
        support.update(blocks_support(self.layers))
        return support

    #: The block's children that carry standard values, by standard name (see `standard_children`).
    _standard_children = staticmethod(standard_children)

    # -- the input (inside a trace) ----------------------------------------------

    @EProperty(key="inputs", description="The token ids the model was called with")
    def input_ids(self, value: Any) -> Tokens:
        """The token ids the model was called with, ``[batch, seq]``. Assign to run the model on other ids."""
        return value[1]["input_ids"]

    @input_ids.postprocess
    def input_ids(self, value: Tensor) -> Any:
        args, kwargs = self.inputs
        return args, {**kwargs, "input_ids": value}

    @EProperty(key="inputs", description="The attention mask the model was called with; zeros are padding")
    def attention_mask(self, value: Any) -> Tokens:
        """The attention mask the model was called with, ``[batch, seq]``; zeros are padding. Assignable."""
        return value[1]["attention_mask"]

    @attention_mask.postprocess
    def attention_mask(self, value: Tensor) -> Any:
        args, kwargs = self.inputs
        return args, {**kwargs, "attention_mask": value}

    @EProperty(key="inputs", description="[batch, seq] of the current call; read-only")
    def input_size(self, value: Any) -> torch.Size:
        """``[batch, seq]`` of the current call, from the ids; read-only."""
        return value[1]["input_ids"].shape

    @input_size.postprocess
    def input_size(self, value: Any) -> Any:
        raise AttributeError("input_size is the ids' shape and cannot be assigned; assign input_ids")

    # -- the image values (inside a trace, on a multimodal load) --------------------

    @EProperty(key="inputs", description="Which positions hold image tokens: input_ids == the config's image_token_id; read-only", unavailable=no_image_tokens)
    def image_token_mask(self, value: Any) -> ImageTokenMask:
        """Which positions hold an image token, ``[batch, seq]`` bool: ``input_ids == config.image_token_id``.

        Read off the call's inputs, so like `input_ids` it is read before
        anything else in the invoke. All false on a text-only trace. Read-only.
        """
        return value[1]["input_ids"] == image_token_id(self.config)

    @image_token_mask.postprocess
    def image_token_mask(self, value: Any) -> Any:
        raise AttributeError("image_token_mask is derived from the ids and cannot be assigned; assign input_ids")

    @EProperty("projector.output", description="The image features the text model receives at the image tokens, flat over them", unavailable=no_image_features)
    def image_features(self, value: torch.Tensor) -> ImageFeatures:
        """The image features the text model receives, ``[image_tokens, hidden]``, flat over every image token of the batch.

        The projector's output, which the wrapper scatters into the token
        embeddings at the image tokens in row-major order, so
        ``layers[0].input[image_token_mask] == image_features``. A view of the
        projector's output: in-place edits land, and an assigned tensor of the
        same shape replaces it. Read after `image_token_mask` and before
        ``layers[0].input``. Never reached on a text-only trace.
        """
        return value.reshape(-1, value.shape[-1])

    @image_features.postprocess
    def image_features(self, value: torch.Tensor) -> torch.Tensor:
        return value.reshape(self.projector.output.shape)

    # -- tokenizers ---------------------------------------------------------------

    @property
    def add_prefix_false_tokenizer(self) -> Any:
        """The checkpoint's tokenizer loaded with ``add_prefix_space=False``, so ``"word"`` and ``" word"`` differ.

        What `nnterp.prompt_utils.get_first_tokens` uses. Loaded once, on first use.
        """
        if self._add_prefix_false_tokenizer is None:
            from transformers import AutoTokenizer

            self._add_prefix_false_tokenizer = AutoTokenizer.from_pretrained(self.repo_id, add_prefix_space=False)
        return self._add_prefix_false_tokenizer

    # -- sizes (from the config) ----------------------------------------------
    # Each is the plain case, read off the text config (a multimodal
    # checkpoint's ``text_config``, else the config itself); a family whose
    # config says it otherwise defines a function of the same name (see
    # `StandardizedProperty`). Each is the model-wide value, equal to every
    # block's where the blocks agree; where they differ (Gemma-4, MiMo-V2-Flash)
    # it is the config's top-level value, and the block's own is on its
    # `Attention` / `Mlp`, read off the module.

    @StandardizedProperty
    def num_layers(self) -> int:
        return len(self.layers)

    @StandardizedProperty
    def hidden_size(self) -> int:
        return self.config.get_text_config().hidden_size

    @StandardizedProperty
    def vocab_size(self) -> int:
        return self.config.get_text_config().vocab_size

    @StandardizedProperty
    def num_heads(self) -> int:
        return self.config.get_text_config().num_attention_heads

    @StandardizedProperty
    def num_kv_heads(self) -> int:
        """Key/value heads: ``num_key_value_heads`` under grouped-query attention, else `num_heads`."""
        return getattr(self.config.get_text_config(), "num_key_value_heads", None) or self.num_heads

    @StandardizedProperty
    def head_dim(self) -> int:
        """Width of one attention head: the config's ``head_dim`` when it says (Qwen3, Gemma), else ``hidden_size // num_heads``."""
        return getattr(self.config.get_text_config(), "head_dim", None) or self.hidden_size // self.num_heads

    @StandardizedProperty
    def qk_head_dim(self) -> int:
        """Width of one head's queries and keys: `head_dim`, unless the family separates them (DeepSeek's latent attention)."""
        return self.head_dim

    @StandardizedProperty
    def intermediate_size(self) -> int:
        """Width of the dense MLP's hidden layer, ``config.intermediate_size``; a mixture of experts' experts are ``moe_intermediate_size`` wide."""
        return self.config.get_text_config().intermediate_size

    # -- remote ------------------------------------------------------------------

    def _remoteable_class(self) -> type:
        """The class in this model's remote key: `TransformersModel`, what a server deploys.

        A remote trace re-runs the block against the client's envoy tree, so
        the aliases and the family's envoy classes travel with the request
        (by reference: the server needs nnterp installed). The deployed model
        itself is a plain `TransformersModel`, so the key says so; a
        subclass-specific key would match nothing on the server.
        """
        return TransformersModel

    # -- loading ---------------------------------------------------------------

    @staticmethod
    def _read_config(repo_id: Any, kwargs: dict) -> Any:
        """The checkpoint's config, before any model is built.

        A ready module carries its own; a repo id is read with ``AutoConfig``,
        so a config transformers cannot parse fails here with its own error.
        The Hub caches the file, and the meta build reads it again.
        """
        if isinstance(repo_id, torch.nn.Module):
            return repo_id.config
        from transformers import AutoConfig

        return AutoConfig.from_pretrained(
            repo_id,
            revision=kwargs.get("revision"),
            trust_remote_code=bool(kwargs.get("trust_remote_code", False)),
        )
