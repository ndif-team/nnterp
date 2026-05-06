from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal
from enum import Enum
import inspect

import torch as th

from .logging import logger
from nnsight import Envoy
from .utils import (
    TraceTensor,
    is_notebook,
    display_markdown,
    try_with_scan,
    dummy_inputs,
)
from .utils import (
    OPTForCausalLM,
    BloomForCausalLM,
    GPT2LMHeadModel,
    GPTJForCausalLM,
    Qwen2MoeForCausalLM,
    DbrxForCausalLM,
    StableLmForCausalLM,
    GptOssForCausalLM,
)

IgnoreType = Literal["mlp", "attention"]


class RenamingError(Exception):
    """Exception raised when the renaming of modules is not properly done."""


class AttnProbFunction(ABC):
    @abstractmethod
    def get_attention_prob_source(
        self, attention_module, return_module_source: bool = False
    ):
        """
        Get the attention probabilities source for a given attention module. If return_module_source is True,
        return the full module source from where the attention probabilities are computed.
        """
        pass

    def __call__(self, attention_module, return_module_source: bool = False):
        return self.get_attention_prob_source(attention_module, return_module_source)


@dataclass
class RenameConfig:
    """
    Configuration for renaming transformer model modules to standardized names.

    This dataclass specifies how to map model-specific module names to standardized names
    used by nnterp. It allows customization for different transformer architectures.

    Parameters
    ----------
    attn_name : str or list of str, optional
        Name(s) of the attention module to rename to 'self_attn'.

    mlp_name : str or list of str, optional
        Name(s) of the MLP/feed-forward module to rename to 'mlp'.

    ln_final_name : str or list of str, optional
        Name(s) of the final layer normalization to rename to 'ln_final'.

    lm_head_name : str or list of str, optional
        Name(s) of the language model head to rename to 'lm_head'.

    model_name : str or list of str, optional
        Name(s) of the main model container to rename to 'model'.

    layers_name : str or list of str, optional
        Name(s) of the transformer layers container to rename to 'layers'.

    attn_prob_source : AttnProbFunction, optional
        Custom function for accessing attention probabilities.
        Should be an instance of AttnProbFunction that defines how to extract
        attention weights from the attention module.

    ignore_mlp : bool, optional
        Whether to skip MLP module processing for this architecture.
        Some models (e.g., OPT) don't have a unified MLP module.

    ignore_attn : bool, optional
        Whether to skip attention module processing for this architecture.
        Rarely used, for architectures without standard attention.

    attn_head_config_key : str, list of str, or int, optional
        Custom key name for the number of attention heads in model config,
        or the number of heads directly. Defaults to standard keys:
        ['n_heads', 'num_attention_heads', 'n_head'].

    hidden_size_config_key : str, list of str, or int, optional
        Custom key name for hidden size in model config,
        or the hidden size directly. Defaults to standard keys:
        ['hidden_size', 'd_model', 'n_embd'].

    vocab_size_config_key : str, list of str, or int, optional
        Custom key name for vocab size in model config,
        or the vocab size directly. Defaults to standard keys:
        ['vocab_size', 'n_vocab', 'text_config.vocab_size'].

    Example
    -------
    Custom configuration for a non-standard architecture::

        config = RenameConfig(
            attn_name="custom_attention",
            mlp_name=["feed_forward", "ffn"]
        )

    """

    attn_name: str | list[str] | None = None
    mlp_name: str | list[str] | None = None
    ln_final_name: str | list[str] | None = None
    lm_head_name: str | list[str] | None = None
    model_name: str | list[str] | None = None
    layers_name: str | list[str] | None = None
    attn_prob_source: AttnProbFunction | None = None
    ignore_mlp: bool | None = None
    ignore_attn: bool | None = None
    attn_head_config_key: str | list[str] | int | None = None
    hidden_size_config_key: str | list[str] | int | None = None
    vocab_size_config_key: str | list[str] | int | None = None


MODEL_NAMES = ["transformer", "gpt_neox", "decoder", "language_model"]


def expand_path_with_model(paths: list[str]) -> list[str]:
    all_paths = [
        [
            (path.replace("model.", f"model.{model_path}."))
            for path in paths
            if path.startswith("model.")
        ]
        for model_path in MODEL_NAMES
    ]
    return paths + sum(all_paths, [])


# Configuration keys for getting the number of attention heads and hidden size
def default_attn_head_config_keys():
    return ["n_heads", "num_attention_heads", "n_head", "num_heads"]


def default_hidden_size_config_keys():
    return ["hidden_size", "d_model", "n_embd"]


def default_vocab_size_config_keys():
    return ["vocab_size", "n_vocab"]


# Models with no mlp module
IGNORE_MLP_MODELS = (OPTForCausalLM,)

# Alternative names for LLM layers
ATTENTION_NAMES = ["attn", "self_attention", "attention", "norm_attn_norm", "linear_attn"]
LAYER_NAMES = expand_path_with_model(
    [
        "h",
        "blocks",
        "model.layers",
    ]
)
LN_NAMES = expand_path_with_model(
    [
        "final_layer_norm",
        "final_layernorm",
        "ln_f",
        "norm_f",
        "norm",
        "embedding_norm",
        "model.ln_final",
    ]
)
LM_HEAD_NAMES = expand_path_with_model(["embed_out", "model.lm_head"])
MLP_NAMES = ["block_sparse_moe", "feed_forward", "ffn"]
EMBED_TOKENS_NAMES = expand_path_with_model(
    [
        "wte",
        "embed_in",
        "word_embeddings",
        "model.embed_tokens",
    ]
)


def get_rename_dict(
    rename_config: RenameConfig | None = None,
) -> dict[str, str]:
    rename_dict = {}
    if rename_config is not None:

        def update_rename_dict(renaming: str, value: str | list[str] | None):
            if value is not None:
                if isinstance(value, str):
                    rename_dict[value] = renaming
                else:
                    for name in value:
                        rename_dict[name] = renaming

        update_rename_dict("model", rename_config.model_name)
        update_rename_dict("layers", rename_config.layers_name)
        update_rename_dict("self_attn", rename_config.attn_name)
        update_rename_dict("mlp", rename_config.mlp_name)
        update_rename_dict("ln_final", rename_config.ln_final_name)
        update_rename_dict("lm_head", rename_config.lm_head_name)

    rename_dict.update(
        {name: "model" for name in MODEL_NAMES}
        | {name: "layers" for name in LAYER_NAMES}
        | {name: "self_attn" for name in ATTENTION_NAMES}
        | {name: "mlp" for name in MLP_NAMES}
        | {name: "ln_final" for name in LN_NAMES}
        | {name: "lm_head" for name in LM_HEAD_NAMES}
        | {name: "embed_tokens" for name in EMBED_TOKENS_NAMES}
    )
    return rename_dict


def text_config(model):
    cfg = model.config
    if "text_config" in cfg:
        cfg = getattr(cfg, "text_config")
    return cfg


def get_num_attention_heads(
    model, raise_error: bool = True, rename_config: RenameConfig | None = None
) -> int | None:
    cfg = text_config(model)
    attn_cfg_keys = default_attn_head_config_keys()
    if rename_config is not None and rename_config.attn_head_config_key is not None:
        if isinstance(rename_config.attn_head_config_key, str):
            attn_cfg_keys.append(rename_config.attn_head_config_key)
        elif isinstance(rename_config.attn_head_config_key, list):
            attn_cfg_keys.extend(rename_config.attn_head_config_key)
        elif isinstance(rename_config.attn_head_config_key, int):
            return rename_config.attn_head_config_key
        else:
            raise ValueError(
                f"Invalid attn head config key: {rename_config.attn_head_config_key}, expected None, str, list[str] or int"
            )
    for attn_head_key in attn_cfg_keys:
        if attn_head_key in cfg:
            return getattr(cfg, attn_head_key)
    if raise_error:
        raise ValueError(f"No attn head config key found in {model}")
    return None


def get_hidden_size(
    model, raise_error: bool = True, rename_config: RenameConfig | None = None
) -> int | None:
    cfg = text_config(model)
    hidden_size_keys = default_hidden_size_config_keys()
    if rename_config is not None and rename_config.hidden_size_config_key is not None:
        if isinstance(rename_config.hidden_size_config_key, str):
            hidden_size_keys.append(rename_config.hidden_size_config_key)
        elif isinstance(rename_config.hidden_size_config_key, list):
            hidden_size_keys.extend(rename_config.hidden_size_config_key)
        elif isinstance(rename_config.hidden_size_config_key, int):
            return rename_config.hidden_size_config_key
        else:
            raise ValueError(
                f"Invalid hidden size config key: {rename_config.hidden_size_config_key}, expected None, str, list[str] or int"
            )
    for hidden_size_key in hidden_size_keys:
        if hidden_size_key in cfg:
            return getattr(cfg, hidden_size_key)
    if raise_error:
        raise ValueError(f"No hidden size config key found in {model}")
    else:
        logger.warning(
            f"Couldn't find the number of attention heads in {model.name_or_path}."
            "You should pass the number of attention heads as an integer or look at the config and pass the key in the attn_head_config_key argument of a RenameConfig."
        )
    return None


def get_vocab_size(
    model, raise_error: bool = True, rename_config: RenameConfig | None = None
) -> int | None:
    cfg = text_config(model)
    vocab_size_keys = default_vocab_size_config_keys()
    if rename_config is not None and rename_config.vocab_size_config_key is not None:
        if isinstance(rename_config.vocab_size_config_key, str):
            vocab_size_keys.append(rename_config.vocab_size_config_key)
        elif isinstance(rename_config.vocab_size_config_key, list):
            vocab_size_keys.extend(rename_config.vocab_size_config_key)
        elif isinstance(rename_config.vocab_size_config_key, int):
            return rename_config.vocab_size_config_key
    for vocab_size_key in vocab_size_keys:
        if vocab_size_key in cfg:
            return getattr(cfg, vocab_size_key)
    if raise_error:
        raise ValueError(f"No vocab size config key found in {model}")
    else:
        return None


class IOType(Enum):
    """Enum to specify input or output access"""

    INPUT = "input"
    OUTPUT = "output"


class InputLayout(Enum):
    """Layer/sublayer input convention.

    ``HIDDEN``: ``module.input`` is the hidden-state tensor (HF, vLLM GPT-2,
        vLLM Llama MLP).
    ``POSITIONS_HIDDEN``: ``module.input`` is a 1D ``positions`` int tensor;
        hidden states live at ``module.inputs[0][1]`` (vLLM Llama attention).
    ``POSITIONS_HIDDEN_RESIDUAL``: ``(positions, hidden_states, residual)``
        positional args; ``residual`` is ``None`` for layer 0
        (vLLM Llama decoder layer).
    """

    HIDDEN = "hidden"
    POSITIONS_HIDDEN = "positions_hidden"
    POSITIONS_HIDDEN_RESIDUAL = "positions_hidden_residual"


class OutputLayout(Enum):
    """Layer/sublayer output convention.

    ``SINGLE``: ``module.output`` is a single hidden-state tensor.
    ``TUPLE_FIRST``: ``module.output`` is a tuple whose first element is the
        hidden-state tensor (HF transformer convention; the rest is metadata
        like past-kv that is not a same-shape tensor).
    ``DUAL_STREAM``: ``module.output`` is ``(hidden_states, residual)`` —
        two same-shape float tensors that sum to the combined residual stream
        (vLLM Llama decoder layer).
    """

    SINGLE = "single"
    TUPLE_FIRST = "tuple_first"
    DUAL_STREAM = "dual_stream"


_POSITIONS_PARAM_NAMES = ("positions", "input_pos", "position_ids")
_RESIDUAL_PARAM_NAMES = ("residual",)
_HIDDEN_STATES_PARAM_NAME = "hidden_states"


def _infer_input_layout(module: Envoy) -> InputLayout:
    """Infer input layout from the wrapped ``nn.Module``'s forward signature.

    We rely on argument names because the input layout is a property of the
    forward contract, not the runtime values (e.g. residual is ``None`` at
    layer 0 but the layout is still positions+hidden+residual).
    """
    underlying = getattr(module, "_module", module)
    sig = inspect.signature(underlying.forward)
    params = [
        name
        for name, p in sig.parameters.items()
        if name != "self"
        and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if not params or params[0] not in _POSITIONS_PARAM_NAMES:
        return InputLayout.HIDDEN
    if len(params) >= 3 and params[2] in _RESIDUAL_PARAM_NAMES:
        return InputLayout.POSITIONS_HIDDEN_RESIDUAL
    return InputLayout.POSITIONS_HIDDEN


def _parent_calls_with_kwargs(parent_layer_module: Envoy, sub_attr_name: str) -> bool:
    """Whether the parent decoder layer calls ``self.<sub_attr_name>(...)`` with keyword args.

    vLLM's Llama decoder forward calls ``self.self_attn(positions=positions,
    hidden_states=hidden_states)`` (kwargs) but ``self.mlp(hidden_states)``
    (positional). At trace time we need to read ``hidden_states`` from the
    right place; this peeks at the parent's source to decide.

    Returns ``False`` (positional) if the source can't be read.
    """
    underlying = getattr(parent_layer_module, "_module", parent_layer_module)
    try:
        src = inspect.getsource(type(underlying).forward)
    except (TypeError, OSError):
        return False
    needle = f"self.{sub_attr_name}("
    idx = src.find(needle)
    if idx < 0:
        return False
    start = idx + len(needle)
    depth = 1
    i = start
    while depth > 0 and i < len(src):
        c = src[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        i += 1
    return "=" in src[start : i - 1]


def _infer_output_layout(target) -> OutputLayout:
    """Infer output layout from the runtime value of ``module.output``.

    Must be called inside a trace context. A 2-tuple of same-shape float
    tensors is the dual-stream signature (vLLM Llama decoder ``(hidden, residual)``);
    HF's ``(hidden, past_kv, ...)`` tuple has a non-tensor (or differently-shaped)
    second element and falls into ``TUPLE_FIRST``.

    Raises ``RenamingError`` if the output looks structurally like dual-stream
    (2-tuple of float tensors) but the shapes disagree — silently summing
    those would broadcast in surprising ways or error at use sites; better to
    fail standardization with a clear message.
    """
    if not isinstance(target, tuple):
        return OutputLayout.SINGLE
    if not (
        len(target) == 2
        and all(isinstance(t, th.Tensor) for t in target)
        and target[0].dtype.is_floating_point
        and target[1].dtype.is_floating_point
    ):
        return OutputLayout.TUPLE_FIRST
    if target[0].shape != target[1].shape:
        raise RenamingError(
            f"Output looks like dual-stream (2-tuple of float tensors) but the "
            f"streams have mismatched shapes {tuple(target[0].shape)} vs "
            f"{tuple(target[1].shape)}. Either the architecture isn't dual-stream "
            f"and this tuple shape is coincidental, or the streams are corrupted."
        )
    return OutputLayout.DUAL_STREAM


class LayerAccessor:
    """Layout-aware I/O accessor for a layer (or one of its named sub-modules).

    Handles three orthogonal axes:

    1. **Module**: the decoder layer itself (``attr_name=None``) or a named
       sub-module like ``self_attn``/``mlp`` (``attr_name="self_attn"``).
    2. **I/O direction**: ``IOType.INPUT`` or ``IOType.OUTPUT`` (or ``None``
       to return the underlying ``Envoy``).
    3. **Layout**: HF vs vLLM-single vs vLLM-dual-stream — inferred per
       accessor (see ``InputLayout``/``OutputLayout``). The input layout comes
       from the forward signature; the output layout from a runtime probe of
       layer 0 (cached after first read).
    """

    def __init__(
        self,
        model,
        attr_name: str | None,
        io_type: IOType | None,
    ):

        self.model = model
        self.attr_name = attr_name
        self.io_type = io_type
        self._input_layout: InputLayout | None = None
        # ``_kwargs_call`` only matters when ``attr_name`` is a sub-module name and
        # the layout is positions-prefixed: vLLM's LlamaDecoderLayer calls
        # ``self.self_attn(positions=..., hidden_states=...)`` (kwargs) but
        # ``self.mlp(hidden_states)`` (positional). Detected per accessor.
        self._kwargs_call: bool = False
        self._output_layout: OutputLayout | None = None

    def get_module(self, layer: int) -> Envoy:
        module = self.model.layers[layer]
        if self.attr_name is not None:
            module = getattr(module, self.attr_name)
        return module

    def _ensure_input_layout(self, module: Envoy) -> InputLayout:
        """Infer and cache the input layout (and parent calling convention) on first use."""
        if self._input_layout is None:
            self._input_layout = _infer_input_layout(module)
            if (
                self.attr_name is not None
                and self._input_layout is not InputLayout.HIDDEN
            ):
                self._kwargs_call = _parent_calls_with_kwargs(
                    self.model.layers[0], self.attr_name
                )
        return self._input_layout

    def _ensure_output_layout(self, target) -> OutputLayout:
        """Infer and cache the output layout from the runtime ``module.output``.

        Validates consistency on subsequent calls — a layer-to-layer drift in
        layout indicates a renaming bug, not normal variation.
        """
        observed = _infer_output_layout(target)
        if self._output_layout is None:
            self._output_layout = observed
        elif observed != self._output_layout:
            raise RenamingError(
                f"Inconsistent output layout: observed {observed.value} but expected "
                f"{self._output_layout.value}. attr_name={self.attr_name!r}."
            )
        return self._output_layout

    def _hs_proxy(self, module: Envoy) -> TraceTensor:
        """Return the ``hidden_states`` proxy for a positions-prefixed input.

        Routes through args or kwargs depending on ``self._kwargs_call``.
        """
        args, kwargs = module.inputs
        if self._kwargs_call:
            return kwargs[_HIDDEN_STATES_PARAM_NAME]
        return args[1]

    def _residual_proxy(self, module: Envoy) -> TraceTensor:
        """Return the ``residual`` proxy. Caller must know layer N>0 (else residual is ``None``)."""
        args, kwargs = module.inputs
        if self._kwargs_call:
            return kwargs[_RESIDUAL_PARAM_NAMES[0]]
        return args[2]

    def _read_input(self, module: Envoy, layer_idx: int) -> TraceTensor:
        """Return the hidden-state input regardless of underlying layout.

        For ``POSITIONS_HIDDEN_RESIDUAL``, layer 0 has ``residual=None`` so we
        return only ``hidden_states``; layers >0 sum ``hidden_states + residual``
        to recover the combined stream the next layer would observe at output.

        For vLLM we ``.clone()`` single-stream reads (see ``_read_output`` for
        why); the residual-sum path already produces a fresh tensor.
        """
        layout = self._ensure_input_layout(module)
        if layout is InputLayout.HIDDEN:
            out = module.input
            if getattr(self.model, "is_vllm", False):
                out = out.clone()
            return out
        hs = self._hs_proxy(module)
        if layout is InputLayout.POSITIONS_HIDDEN_RESIDUAL and layer_idx > 0:
            return hs + self._residual_proxy(module)
        if getattr(self.model, "is_vllm", False):
            hs = hs.clone()
        return hs

    def _write_input(self, module: Envoy, value: TraceTensor, layer_idx: int) -> None:
        """Write a hidden-state input regardless of underlying layout.

        For ``POSITIONS_HIDDEN_RESIDUAL`` at layer >0 we put ``value`` in
        ``hidden_states`` and zero ``residual`` so the layer sees
        ``hidden + residual == value``. At layer 0 (residual is ``None``)
        we just set ``hidden_states``.
        """
        layout = self._ensure_input_layout(module)
        if layout is InputLayout.HIDDEN:
            module.input = value
            return
        # In-place writes (per nnsight VLLM_GUIDE) avoid the whole-tuple-replacement crash.
        self._hs_proxy(module)[:] = value
        if layout is InputLayout.POSITIONS_HIDDEN_RESIDUAL and layer_idx > 0:
            self._residual_proxy(module)[:] = 0

    def _read_output(self, module: Envoy) -> TraceTensor:
        """Return the hidden-state output regardless of underlying layout.

        For vLLM, clone the resolved tensor: vLLM reuses inference-mode
        buffers across layers (e.g. layer N+1's fused RMSNorm mutates layer
        N's mlp/attn output buffer in-place), so a saved reference would
        surface the post-mutation value. ``DUAL_STREAM`` already produces
        a new tensor via the sum; ``SINGLE`` and ``TUPLE_FIRST`` need an
        explicit clone. nnsight's clone-on-save for inference-mode tensors
        (see VLLM_GUIDE) doesn't reach every path here.
        """
        target = module.output
        layout = self._ensure_output_layout(target)
        if layout is OutputLayout.DUAL_STREAM:
            return target[0] + target[1]
        out = target if layout is OutputLayout.SINGLE else target[0]
        if getattr(self.model, "is_vllm", False):
            out = out.clone()
        return out

    def _write_output(self, module: Envoy, value: TraceTensor) -> None:
        """Write a hidden-state output regardless of underlying layout.

        For ``DUAL_STREAM``, per ``nnsight VLLM_GUIDE``: whole-tuple replacement
        crashes the engine; use in-place index assignment on each stream. Zero
        ``mlp_out`` and put ``value`` in ``residual`` so they sum to ``value``.
        """
        # Reading module.output here also caches the output layout if not yet set.
        layout = self._ensure_output_layout(module.output)
        if layout is OutputLayout.SINGLE:
            module.output = value
        elif layout is OutputLayout.DUAL_STREAM:
            module.output[0][:] = 0
            module.output[1][:] = value
        else:  # TUPLE_FIRST
            module.output = (value, *module.output[1:])

    def __getitem__(self, layer: int) -> TraceTensor | Envoy:
        module = self.get_module(layer)
        if self.io_type is None:
            return module
        if self.io_type is IOType.INPUT:
            return self._read_input(module, layer)
        return self._read_output(module)

    def __setitem__(self, layer: int, value: TraceTensor):
        if self.io_type is None:
            name = self.attr_name or "layers"
            raise ValueError(
                f"Cannot set the value of a module accessor. Did you mean {name}_input/output"
            )
        module = self.get_module(layer)
        if self.io_type is IOType.INPUT:
            self._write_input(module, value, layer)
        else:
            self._write_output(module, value)

    def __call__(self, layer: int) -> TraceTensor | Envoy:
        return self[layer]

    @property
    def returns_tuple(self) -> bool | None:
        """Whether ``module.output`` is a tuple (any kind), or ``None`` if not yet probed.

        Kept for backwards compatibility with code that previously inspected
        ``_detected_is_tuple``. Use ``output_layout`` for the structured form.
        """
        if self._output_layout is None:
            return None
        return self._output_layout is not OutputLayout.SINGLE

    @property
    def output_layout(self) -> OutputLayout | None:
        """The detected ``OutputLayout`` for this accessor, or ``None`` if not yet probed."""
        return self._output_layout

    @property
    def input_layout(self) -> InputLayout | None:
        """The detected ``InputLayout`` for this accessor, or ``None`` if not yet probed."""
        return self._input_layout


def bloom_attention_prob_source(attention_module, return_module_source: bool = False):
    if return_module_source:
        return attention_module.source
    else:
        return attention_module.source.self_attention_dropout_0


def default_attention_prob_source(attention_module, return_module_source: bool = False):
    if return_module_source:
        return attention_module.source.attention_interface_0.source
    else:
        return (
            attention_module.source.attention_interface_0.source.nn_functional_dropout_0
        )


def gpt2_attention_prob_source(attention_module, return_module_source: bool = False):
    if return_module_source:
        return attention_module.source.attention_interface_0.source
    else:
        return (
            attention_module.source.attention_interface_0.source.module_attn_dropout_0
        )


def gptj_attention_prob_source(attention_module, return_module_source: bool = False):
    if return_module_source:
        return attention_module.source.self__attn_0.source
    else:
        return attention_module.source.self__attn_0.source.self_attn_dropout_0


def qwen2moe_attention_prob_source(attention_module, return_module_source: bool = False):
    if return_module_source:
        return attention_module.source
    else:
        return attention_module.source.nn_functional_dropout_0


def dbrx_attention_prob_source(attention_module, return_module_source: bool = False):
    if return_module_source:
        return attention_module.attn.source
    else:
        return attention_module.attn.source.nn_functional_dropout_0


def stablelm_attention_prob_source(attention_module, return_module_source: bool = False):
    if return_module_source:
        return attention_module.source
    else:
        return attention_module.source.self_attention_dropout_0


def gptoss_attention_prob_source(attention_module, return_module_source: bool = False):
    if return_module_source:
        return attention_module.source.attention_interface_0.source
    else:
        return (
            attention_module.source.attention_interface_0.source.nn_functional_dropout_0
        )


class AttentionProbabilitiesAccessor:
    def __init__(
        self,
        model,
        rename_config: RenameConfig | None = None,
        initialized_with_enable: bool = False,
    ):
        self.model = model
        self.initialized_with_enable = initialized_with_enable
        self.attn_probs_dont_sum_to_one = False
        if rename_config is not None and rename_config.attn_prob_source is not None:
            self.source_attr = rename_config.attn_prob_source
        elif isinstance(model._model, BloomForCausalLM):
            self.source_attr = bloom_attention_prob_source
        elif isinstance(model._model, GPT2LMHeadModel):
            self.source_attr = gpt2_attention_prob_source
        elif isinstance(model._model, GPTJForCausalLM):
            self.source_attr = gptj_attention_prob_source
        elif isinstance(model._model, Qwen2MoeForCausalLM):
            self.source_attr = qwen2moe_attention_prob_source
        elif isinstance(model._model, DbrxForCausalLM):
            self.source_attr = dbrx_attention_prob_source
        elif isinstance(model._model, StableLmForCausalLM):
            self.source_attr = stablelm_attention_prob_source
        elif isinstance(model._model, GptOssForCausalLM):
            self.source_attr = gptoss_attention_prob_source
            self.attn_probs_dont_sum_to_one = True
        else:
            self.source_attr = default_attention_prob_source
        self.enabled = True

    def disable(self):
        self.enabled = False

    def _check_enabled(self):
        if not self.enabled:
            if self.initialized_with_enable:
                raise RenamingError(
                    "Attention probabilities are disabled for this model."
                )
            else:
                raise RenamingError(
                    "Attention probabilities are disabled for this model. "
                    "Set enable_attention_probs=True when loading the model to enable them."
                )

    def __getitem__(self, layer: int) -> TraceTensor:
        self._check_enabled()
        return self.source_attr(self.model.layers[layer].self_attn).output

    def __setitem__(self, layer: int, value: TraceTensor):
        self._check_enabled()
        self.source_attr(self.model.layers[layer].self_attn).output = value

    def check_source(
        self, layer: int = 0, allow_dispatch: bool = True, use_trace: bool = True
    ):
        """
        Check that the attention probabilities source is correctly configured.

        This method validates that:
        1. The attention probabilities have the expected shape (batch_size, num_heads, seq_len, seq_len)
        2. The probabilities sum to 1 along the last dimension
        3. Modifying the probabilities affects the model's output logits

        Args:
            layer (int, optional): The layer index to check. Defaults to 0.
            allow_dispatch (bool, optional): If True, allows dispatching the model when scan fails.
            use_trace (bool, optional): If False, uses scan() to validate the attention probabilities, which means attention probabilities summing to 1 and causal effect of modifying them won't be tested. Defaults to True.

        Raises:
            RenamingError: If the attention probabilities are not properly configured or if the number of attention heads is not available.
        """
        if self.model.num_heads is None:
            raise RenamingError(
                f"Can't check the shapes of the model internals because the number of attention heads is not available in {self.model.repo_id} architecture."
                "You should pass the number of attention heads as an integer or look at the config and pass the key in the attn_head_config_key argument of a RenameConfig."
            )

        def test_prob_source():
            batch_size, seq_len = self.model.input_size
            num_heads = self.model.num_heads
            probs = self[layer]
            if probs.shape != (batch_size, num_heads, seq_len, seq_len):
                raise RenamingError(
                    f"Attention probabilities have shape {probs.shape} != {(batch_size, num_heads, seq_len, seq_len)} (batch_size, n_head, seq_len, seq_len) in {self.model.repo_id} architecture. This means it's not properly initialized."
                )
            rnd = th.randn_like(probs).abs()
            rnd = rnd / rnd.sum(dim=-1, keepdim=True)
            self[layer] = rnd
            if probs.device != th.device("meta"):
                sum_last = probs.sum(dim=-1)
                if self.attn_probs_dont_sum_to_one:
                    if not (sum_last > 0).all():
                        raise RenamingError(
                            "Attention probabilities should be > 0."
                        )
                    if not (sum_last < 1 + 1e-5).all():
                        raise RenamingError(
                            "Attention probabilities should sum to < 1 for models with sink tokens."
                        )
                else:
                    atol = 1e-2 if probs.dtype == th.bfloat16 else 1e-5
                    if not th.allclose(sum_last, th.ones_like(sum_last), atol=atol):
                        raise RenamingError("Attention probabilities do not sum to 1.")

        if use_trace:
            with self.model.trace(dummy_inputs()):
                test_prob_source()
                corr_logits = self.model.logits.save()
            with self.model.trace(dummy_inputs()):
                clean_logits = self.model.logits.save()

            if th.allclose(corr_logits, clean_logits):
                raise RenamingError(
                    "Attention probabilities are not properly initialized: changing the attention probabilities should change the logits."
                )
            return

        try_with_scan(
            self.model,
            test_prob_source,
            RenamingError(
                "Can't access attention probabilities. It is most likely not yet supported for this architecture and transformers version."
            ),
            allow_dispatch=allow_dispatch,
            errors_to_raise=(RenamingError,),
        )

    def print_source(self, layer: int = 0, allow_dispatch: bool = True):
        in_notebook = is_notebook()
        if in_notebook:
            markdown_text = "## Accessing attention probabilities from:\n"
        else:
            print("Accessing attention probabilities from:")

        def print_hook_source():
            nonlocal markdown_text
            source = self.source_attr(self.model.layers[layer].self_attn)
            if in_notebook:
                markdown_text += f"```py\n{source}\n```"
            else:
                print(source)

        used_scan = try_with_scan(
            self.model,
            print_hook_source,
            RenamingError(
                "Can't access attention probabilities. It is most likely not yet supported for this architecture and transformers version."
            ),
            allow_dispatch=allow_dispatch,
        )
        if in_notebook:
            markdown_text += "\n\n## Full module source:\n"
        else:
            print("\n\nFull module source:")

        def print_attn_source():
            nonlocal markdown_text
            source = str(
                self.source_attr(
                    self.model.layers[layer].self_attn, return_module_source=True
                )
            )
            if in_notebook:
                markdown_text += f"```py\n{source}\n```"
            else:
                print(source)

        try_with_scan(
            self.model,
            print_attn_source,
            RenamingError(
                "Can't access attention probabilities. It is most likely not yet supported for this architecture and transformers version."
            ),
            allow_dispatch=allow_dispatch,
            warn_if_scan_fails=used_scan,
        )

        if in_notebook:
            display_markdown(markdown_text)


def get_ignores(model, rename_config: RenameConfig | None = None) -> list[str]:
    ignores = []
    if isinstance(model, IGNORE_MLP_MODELS):
        message = f"{model.__class__.__name__} does not have a mlp module."
        if isinstance(model, OPTForCausalLM):
            message += " You'll have to manually use layers.fc1 and layers.fc2 instead."
        logger.warning(message)
        ignores.append("mlp")
    if rename_config is not None:
        if rename_config.ignore_mlp:
            ignores.append("mlp")
        if rename_config.ignore_attn:
            ignores.append("attention")
    return ignores


def _check_tensor(tensor, name: str, expected_shape: tuple, model_name: str):
    """Validate that a tensor has the expected type and shape."""
    if not isinstance(tensor, th.Tensor):
        raise ValueError(
            f"{name} is not a tensor in {model_name} architecture. "
            f"Found type {type(tensor)}. This means it's not properly initialized."
        )
    if tensor.shape != expected_shape:
        raise ValueError(
            f"{name} has shape {tensor.shape} != {expected_shape} in {model_name} architecture. "
            "This means it's not properly initialized."
        )


def check_io(std_model, model_name: str, ignores: list[IgnoreType]):
    """Validate that standardized accessors return tensors with consistent shapes.

    Handles both HF models (``input_size = (batch, seq)``) and vLLM models
    (``input_size = (seq,)``). Shape expectations adapt via ``(*input_size, dim)``.

    For vLLM, ``lm_head.output`` is not checked because vLLM computes logits
    in a separate phase outside the model's forward pass.
    """
    input_size = std_model.input_size
    hidden_size = std_model.hidden_size
    if hidden_size is None:
        raise RenamingError(
            f"Can't check the shapes of the model internals because the hidden size is not available in {model_name} architecture. "
            "You should pass the hidden size as an integer or look at the config and pass the key in the hidden_size_config_key argument of a RenameConfig."
        )
    expected_hidden = (*input_size, hidden_size)

    _check_tensor(std_model.token_embeddings, "token_embeddings", expected_hidden, model_name)
    # The accessor reads cache layout on first hit, so each subsequent layer-N
    # access goes through the layout-aware path with no extra introspection.
    _check_tensor(std_model.layers_input[0], "layers_input[0]", expected_hidden, model_name)

    if "attention" not in ignores:
        _check_tensor(std_model.attentions_input[0], "attentions_input[0]", expected_hidden, model_name)
        _check_tensor(std_model.attentions_output[0], "attentions_output[0]", expected_hidden, model_name)

    if "mlp" not in ignores:
        _check_tensor(std_model.mlps_input[0], "mlps_input[0]", expected_hidden, model_name)
        _check_tensor(std_model.mlps_output[0], "mlps_output[0]", expected_hidden, model_name)
    # The dual-stream-shape-mismatch invariant is enforced inside
    # ``_infer_output_layout`` (it raises ``RenamingError`` when a 2-tuple of
    # float tensors has divergent shapes), so by the time we reach here the
    # accessor's read has already validated it.
    _check_tensor(std_model.layers_output[0], "layers_output[0]", expected_hidden, model_name)
    _check_tensor(std_model.ln_final.output, "ln_final.output", expected_hidden, model_name)

    # vLLM computes logits in a separate phase (not part of the model forward pass),
    # so lm_head.output is not accessible during a vLLM trace.
    if not std_model.is_vllm:
        lm_head_out = std_model.lm_head.output
        if not isinstance(lm_head_out, th.Tensor):
            raise ValueError(
                f"lm_head.output is not a tensor in {model_name} architecture. "
                f"Found type {type(lm_head_out)}. This means it's not properly initialized."
            )
        expected_vocab = (*input_size, std_model.vocab_size)
        if std_model.vocab_size is None:
            logger.warning(
                f"Couldn't find vocab_size in {model_name} config. Couldn't properly test the shape of lm_head.output."
            )
            if lm_head_out.shape[:-1] != input_size:
                raise ValueError(
                    f"lm_head.output has shape {lm_head_out.shape}, expected prefix {input_size} in {model_name} architecture."
                )
        else:
            if lm_head_out.shape != expected_vocab:
                raise ValueError(
                    f"lm_head.output has shape {lm_head_out.shape} != {expected_vocab} in {model_name} architecture."
                )


def _check_has_module(obj, attr: str, model_name: str, rename_arg: str):
    """Raise RenamingError if ``obj`` doesn't have ``attr``."""
    if not hasattr(obj, attr):
        raise RenamingError(
            f"Could not find {attr} module in {model_name} architecture. "
            f"This means that it was not properly renamed.\n"
            f"Please pass the name of the {attr} module to the {rename_arg} argument."
        )


def _warn_heterogeneous_types(accessor, num_layers: int, kind: str, model_name: str):
    """Warn if modules accessed by ``accessor[i]`` have mixed types across layers."""
    types = {type(accessor[i]._module) for i in range(num_layers)}
    if len(types) > 1:
        type_names = ", ".join(sorted(t.__name__ for t in types))
        logger.warning(
            f"Model {model_name} has heterogeneous {kind} types across layers: {type_names}. "
            "Some nnterp operations may not work consistently across all layers."
        )


def check_model_renaming(
    std_model,
    model_name: str,
    ignores: list[IgnoreType],
    allow_dispatch: bool,
    allow_multimodal: bool = False,
):
    _check_has_module(std_model, "layers", model_name, "layers_rename")

    if not allow_multimodal:
        layer_types = {type(layer._module) for layer in std_model.layers}
        if len(layer_types) > 1:
            type_names = ", ".join(sorted(t.__name__ for t in layer_types))
            raise RenamingError(
                f"Model {model_name} has heterogeneous layer types: {type_names}.\n"
                "This likely means it is a multimodal model where some layers (e.g. cross-attention) "
                "only activate with specific inputs (like images). "
                "nnterp cannot guarantee standardized access to all layers in this case.\n"
                "If you want to use this model anyway, pass allow_multimodal=True to StandardizedTransformer."
            )

    _check_has_module(std_model, "ln_final", model_name, "ln_final_rename")
    _check_has_module(std_model, "lm_head", model_name, "lm_head_rename")

    if "attention" not in ignores:
        _check_has_module(std_model.layers[0], "self_attn", model_name, "attn_rename")
        _warn_heterogeneous_types(std_model.attentions, std_model.num_layers, "attention", model_name)

    if "mlp" not in ignores:
        _check_has_module(std_model.layers[0], "mlp", model_name, "mlp_rename")
        _warn_heterogeneous_types(std_model.mlps, std_model.num_layers, "MLP", model_name)

    try_with_scan(
        std_model,
        lambda: check_io(std_model, model_name, ignores),
        RenamingError(f"Could not check the IO of {model_name}"),
        allow_dispatch,
        errors_to_raise=(RenamingError,),
    )


HF_TO_VLLM_KWARGS_MAP = dict(
    max_new_tokens="max_tokens",
)


def hf_kwargs_to_vllm_kwargs(kwargs: dict) -> dict:
    """Translate HuggingFace keyword arguments to their vLLM equivalents.

    Raises ValueError if both the HF and vLLM names are present with different values.
    """
    for hf_name, vllm_name in HF_TO_VLLM_KWARGS_MAP.items():
        if hf_name in kwargs:
            if vllm_name in kwargs and kwargs[vllm_name] != kwargs[hf_name]:
                raise ValueError(
                    f"Conflicting values for {hf_name} and {vllm_name}: "
                    f"{kwargs[hf_name]} vs {kwargs[vllm_name]}"
                )
            kwargs[vllm_name] = kwargs.pop(hf_name)
    return kwargs
