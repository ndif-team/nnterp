"""`Standardized`, what every standardized model shares, and `StandardizedTransformer`, the `TransformersModel` one."""

from __future__ import annotations

from collections import Counter
from types import ModuleType
import functools
from typing import Any, Callable, Sequence

import torch
from jaxtyping import Float, Int
from nnsight.intervention.envoy import Envoy
from nnsight.modeling.transformers import TransformersModel
from torch import Tensor

from . import families, order as forward_order
from .families import default
from .components import EProperty, Layer, Residual, Unavailable
from .components.standard import blocks_support, standard_children, values
from .components.vision import Vision

#: The layouts of the root's values: the logits, the next-token distribution at the last position, and one
#: integer per token (``input_ids``, ``attention_mask``).
Logits = Float[Tensor, "batch seq vocab"]
NextTokenProbs = Float[Tensor, "batch vocab"]
Tokens = Int[Tensor, "batch seq"]


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


class Standardized:
    """What a standardized model is on any engine: a family, its sizes, its availability, and the methods over its values.

    Mixed in ahead of the nnsight model class that runs the checkpoint
    (`StandardizedTransformer` over `TransformersModel`,
    `nnterp.StandardizedVLLM` over nnsight's ``VLLM``). The leaf resolves
    ``family`` and hands its ``RENAME`` and ``ENVOYS`` to nnsight, and defines
    the root's values (``logits``, ``token_embeddings``, ...), which are read
    where its engine keeps them. Everything here is written against the
    standard names and values only, so it holds on both.
    """

    family: ModuleType
    layers: Sequence[Layer]
    embed_tokens: Envoy
    norm: Envoy
    lm_head: Envoy

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
        hidden = self.layers[start].layer_input if skip_with is None else skip_with
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
        module no block has (OPT's ``mlp``) has no entry. A standard child of
        the root (the vision tower, `Vision`) is listed the same way, its own
        `support` rows under its standard name (``"vision.image_features"``,
        ``"vision.self_attn.attention_probabilities"``); a tower no image
        reaches lists none.
        """
        if layer is not None:
            return blocks_support(self.layers, layer)
        support: dict[str, Any] = {name: value.reason(self) for name, value in values(type(self)).items()}
        support.update(blocks_support(self.layers))
        for host, child in standard_children(self).items():
            support.update((f"{host}.{name}", reason) for name, reason in child.support().items())
        return support

    #: The block's children that carry standard values, by standard name (see `standard_children`).
    _standard_children = staticmethod(standard_children)

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


class StandardizedTransformer(Standardized, TransformersModel):
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
        family: The family to use instead of the one looked up: a module or
            any object with ``RENAME`` and ``ENVOYS`` (and, like a shipped
            family, any size function or ``project_on_vocab`` it defines). It
            applies to this model only; the config is not read for it, and
            no model type is named or checked. ``nnterp.families.default``
            itself runs its load-time check; a copy of it (a
            ``SimpleNamespace`` spread from its ``vars``) does not, so call
            ``default.check(model)`` after the load.
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
    is ``vision`` (a `Vision`, with its own blocks, sizes and values, among them
    ``vision.image_token_mask`` and ``vision.image_features``) and the last
    module before the scatter into the text stream is ``projector``.

    Attributes:
        family: The toolkit module the checkpoint resolved to, or the ``family`` passed.
        layers: The decoder blocks, each a `Layer` (the family's subclass).
        embed_tokens, norm, lm_head: The embedding, the final norm, the unembedding.
        vision: The vision tower, a `Vision`, on a multimodal wrapper whose family names it.
        projector: The last module before the wrapper scatters the image features into the text stream.

    Raises:
        UnsupportedFamily: when no family is passed and none covers the checkpoint's ``model_type``
            and the best-effort default family cannot standardize it either.
    """

    vision: Vision
    projector: Envoy

    def __init__(
        self,
        repo_id: Any,
        *args: Any,
        family: ModuleType | None = None,
        rename: dict[str, str | list[str]] | None = None,
        envoys: dict | None = None,
        tokenizer_kwargs: dict | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("task", "text-generation")
        self._add_prefix_false_tokenizer = None
        self._order = None  # the measured ranks: see `order`
        if family is None:
            config = self._read_config(repo_id, kwargs)
            # A multimodal checkpoint's config nests the language model's; the
            # text-generation task builds that model, so its family is the one.
            family = families.lookup(getattr(config, "text_config", config).model_type)
        self.family = family
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
        # The default family is a guess, confirmed on the built tree (a copy of it calls the check itself).
        if self.family is default:
            default.check(self)
        for key, value in (tokenizer_kwargs or {}).items():
            setattr(self.tokenizer, key, value)
        self._source_root_scatter()

    def _update(self, module: Any) -> None:
        super()._update(module)  # real weights replacing meta ones reinstall the plain forward
        self._source_root_scatter()

    def _source_root_scatter(self) -> None:
        """Instrument the root's forward ahead of any run where it scatters the image features (the family's ``ROOT_SCATTER``)."""
        if getattr(self.family, "ROOT_SCATTER", None) and "projector" in self._aliases:
            self.source

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

    # -- the logit lens -----------------------------------------------------------

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
        """The attention mask the model was called with, ``[batch, seq]``; zeros are padding. Assignable.

        ``generate`` on transformers >= 5.18 calls the model with no mask when
        no prompt is padded; the all-ones mask that stands for is installed
        into the call on read (what earlier versions passed), so it is served
        and edits to it reach the model. On a decode step it covers the cache
        and the new token, ``[batch, past + seq]``, as the passed mask does.
        """
        kwargs = value[1]
        if kwargs.get("attention_mask") is None:
            tokens = kwargs.get("input_ids")
            if tokens is None:
                tokens = kwargs["inputs_embeds"]
            cache = kwargs.get("past_key_values")
            past = cache.get_seq_length() if cache is not None else 0
            kwargs["attention_mask"] = torch.ones(tokens.shape[0], past + tokens.shape[1], dtype=torch.long, device=tokens.device)
        return kwargs["attention_mask"]

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

    # -- forward order (outside a trace) -------------------------------------------

    def order(self, layer: int | None = None) -> dict[str, int]:
        """The available values in forward order: name -> rank; equal ranks are one location, either read first.

        With ``layer``, that block's values by dotted name (`support`'s), from
        0. Without, the root's, numbered across the whole forward. Measured by
        a probe run on first use (`nnterp.order.probe`) and kept on the model,
        so route kernels (`route_kernels`, `chunk_per_token`) before the first.
        """
        table = forward_order.cached(self)
        if layer is None:
            return {**table[-1], **table[self.num_layers]}
        return dict(table[range(self.num_layers)[layer]])  # negative layers count from the end; IndexError past it

    def rank(self, name: str, layer: int | None = None) -> tuple[int, int]:
        """Where a value falls in the forward: ``(side, r)``, which sorts reads into forward order.

        ``side`` is ``-1`` or ``num_layers`` for a root value served before or
        after the blocks, else ``layer``; ``r`` is `order`'s rank.

        Raises:
            Unavailable: the checkpoint has no such value (`support` gives the reason).
            KeyError: there is no value of that name (a block value needs ``layer``).
        """
        table = forward_order.cached(self)
        if layer is None:
            for side in (-1, self.num_layers):
                if name in table[side]:
                    return side, table[side][name]
            value = values(type(self)).get(name)
            reason = value and value.reason(self)
        else:
            layer = range(self.num_layers)[layer]  # negative layers count from the end; IndexError past it
            if name in table[layer]:
                return layer, table[layer][name]
            reason = self.support(layer).get(name)
        if not reason:
            where = "the root" if layer is None else f"block {layer}"
            raise KeyError(f"{name!r} is not a value of {where}; a block value takes layer=, by its dotted support() name")
        raise Unavailable(f"{name} is not available{'' if layer is None else f' on block {layer}'}: {reason}")

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
