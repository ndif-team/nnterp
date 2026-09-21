import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any, Callable, Literal
from enum import Enum

import torch as th

from .logging import logger
from nnsight.intervention.envoy import Envoy
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
    FalconForCausalLM,
    GPTJForCausalLM,
    Qwen2MoeForCausalLM,
    DbrxForCausalLM,
    GptOssForCausalLM,
    MptForCausalLM,
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
        Name(s) of the softmax-attention module to rename to 'self_attn'. Linear
        attention mixers (Gated DeltaNet in Qwen3-Next / Qwen3.5 hybrids) keep
        their ``linear_attn`` name: the attention accessors are not defined on them.

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

    attn_output_source : str, optional
        Dotted path (relative to a layer, using standardized names) to the module
        whose output is the attention's additive contribution to the residual stream,
        e.g. "self_attn.dense" for BLOOM. Needed for architectures that add the
        residual *inside* the attention module, where the module output is already
        a residual-stream state (see https://github.com/ndif-team/nnterp/issues/51).
        ``attentions_output[i]`` reads/writes this module's output. Passing the
        default "self_attn" explicitly asserts that the module output is the
        additive contribution despite a ``residual``-like argument in its forward.

    mlp_output_source : str, optional
        Same as ``attn_output_source`` for the MLP, e.g. "mlp.dense_4h_to_h" for
        BLOOM or "mlp.down_proj" for MPT.

    addresses : dict of str to Address, optional
        Rows of the address table, by accessor name: replace where an existing
        accessor reads on this architecture, or add an accessor of your own
        (it appears in ``model.internals``). The general form of the three
        fields above. See ``Address``.

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
    attn_output_source: str | None = None
    mlp_output_source: str | None = None
    addresses: "dict[str, Address] | None" = None


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

def bloom_slow_but_exact(model) -> bool:
    """BLOOM checkpoints with pretraining_tp > 1 and slow_but_exact=True (e.g.
    bigscience/bigscience-small-testing) compute the output projections with
    F.linear, bypassing the dense modules, so no pre-residual module exists."""
    return (
        isinstance(model, BloomForCausalLM)
        and model.config.pretraining_tp > 1
        and model.config.slow_but_exact
    )


# Alternative names for LLM layers
ATTENTION_NAMES = [
    "attn",
    "self_attention",
    "attention",
    "norm_attn_norm",
]
# Linear-attention mixers (Gated DeltaNet in Qwen3-Next / Qwen3.5 hybrids) are not
# renamed: a block exposes exactly one of self_attn / linear_attn, and the
# attention accessors are only defined on self_attn blocks.
LINEAR_ATTENTION_NAME = "linear_attn"
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


def get_head_dim(model) -> int:
    """The width of one head's value, and so of its share of ``attentions_premix``.
    The config's own ``head_dim`` where it states one: it is not always
    ``hidden_size // num_heads`` (Qwen3, Gemma). Under multi-head latent attention
    (DeepSeek) it is ``v_head_dim``; the query and key have ``get_qk_head_dim``."""
    cfg = text_config(model)
    if getattr(cfg, "v_head_dim", None) is not None:
        return cfg.v_head_dim
    if getattr(cfg, "head_dim", None) is not None:
        return cfg.head_dim
    return get_hidden_size(model) // get_num_attention_heads(model)


def get_qk_head_dim(model) -> int:
    """The width of one head's query and key: ``head_dim`` everywhere except under
    multi-head latent attention, where it is ``qk_nope_head_dim + qk_rope_head_dim``."""
    cfg = text_config(model)
    if getattr(cfg, "qk_nope_head_dim", None) is not None:
        return cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
    return get_head_dim(model)


def get_num_kv_heads(model) -> int:
    """Key/value heads: fewer than ``num_heads`` under grouped-query attention. A
    config that does not group does not say so."""
    cfg = text_config(model)
    # Falcon-7B: one key/value head, and a `num_kv_heads` its modules never read
    if getattr(cfg, "multi_query", False) and not getattr(cfg, "new_decoder_architecture", False):
        return 1
    for key in ("num_key_value_heads", "num_kv_heads", "n_head_kv"):
        if getattr(cfg, key, None) is not None:
            return getattr(cfg, key)
    return get_num_attention_heads(model)


def get_intermediate_size(model) -> int | None:
    """The MLP's inner width. GPT-2 calls it ``n_inner`` and leaves it None to mean
    four times hidden; that is asked first because a GPT-2 config can carry a stray
    ``intermediate_size`` its modules never read (hf-internal-testing/tiny-random-gpt2
    says 37 beside 128-wide MLPs). BLOOM states none and hard-codes four times hidden."""
    cfg = text_config(model)
    if "n_inner" in cfg:
        return cfg.n_inner or 4 * get_hidden_size(model)
    for key in ("intermediate_size", "ffn_hidden_size", "ffn_dim"):
        if getattr(cfg, key, None) is not None:
            return getattr(cfg, key)
    if getattr(cfg, "ffn_config", None) is not None:  # DBRX
        return cfg.ffn_config.ffn_hidden_size
    if getattr(cfg, "expansion_ratio", None) is not None:  # MPT
        return int(cfg.expansion_ratio * get_hidden_size(model))
    if cfg.model_type == "bloom":  # hard-coded in BloomMLP
        return 4 * get_hidden_size(model)
    logger.warning(
        f"Couldn't find the MLP's inner width in the {cfg.model_type} config; "
        "model.intermediate_size is None."
    )
    return None


class IOType(Enum):
    """Enum to specify input or output access"""

    INPUT = "input"
    OUTPUT = "output"


def get_attention_layers(layers) -> tuple[list[int], list[int]]:
    """Split the block indices of ``layers`` by attention kind, from the block
    structure: softmax-attention blocks expose ``self_attn``, linear-attention
    blocks expose ``linear_attn``. Returns ``(attention_layers, linear_attention_layers)``.
    """
    attention_layers = [
        i for i, layer in enumerate(layers) if hasattr(layer, "self_attn")
    ]
    linear_attention_layers = [
        i for i, layer in enumerate(layers) if hasattr(layer, LINEAR_ATTENTION_NAME)
    ]
    return attention_layers, linear_attention_layers


def linear_attention_error(model, layer: int) -> RenamingError:
    mixer = type(getattr(model.layers[layer], LINEAR_ATTENTION_NAME)._module).__name__
    return RenamingError(
        f"layer {layer} is a linear-attention layer ({mixer}); attentions/attention_probabilities "
        f"are only defined on softmax-attention layers (model.attention_layers = {model.attention_layers})."
    )


@dataclass(frozen=True)
class Lens:
    """Where a tensor sits inside a value no path can describe: two functions,
    because a write needs the way back as much as a read needs the way in.
    ``get(value)`` is the tensor; ``put(value, tensor)`` is ``value`` with the
    tensor replaced. The escape hatch of ``Address.select``."""

    get: Callable[[Any], Any]
    put: Callable[[Any, Any], Any]


@dataclass(frozen=True)
class Address:
    """Where one tensor of a layer is, as data.

    Every accessor nnterp offers (``layers_output``, ``attentions_output``,
    ``attention_probabilities``, ...) is one of these, and a family that differs
    differs by a row of ``FAMILY_ADDRESSES`` rather than by a branch of code. A
    model nnterp does not know is supported the same way, from outside:
    ``RenameConfig(addresses={"attentions_output": Address("self_attn.dense")})``.

    Parameters
    ----------
    module : str
        Dotted path under the layer, in nnterp's standardized names. ``""`` is the
        layer itself.
    io : IOType or None
        Which side of the module (or of the operation, when ``op`` is set) carries
        the tensor. ``None`` makes the accessor return the module envoy itself.
    op : tuple of str, or callable
        Operations of the module's ``.source`` to descend through, outermost
        first: ``("attention_interface_1", "nn_functional_dropout_0")`` is the
        dropout call inside the function the attention call dispatches to. A
        callable ``fn(module_envoy) -> operation`` is accepted for a forward no
        name can describe (``RenameConfig.attn_prob_source``).
    select : tuple, Lens or None
        Where the tensor is inside the value at that place. ``None`` takes the
        first element of a tuple and a bare value as it is, decided per access,
        since layers of one model may differ. A path ``(1,)`` / ``(0, 2)`` /
        ``("hidden_states",)`` is walked to read and rebuilt around the new tensor
        to write. A ``Lens`` is for what a path cannot say.
    order : int
        Rank of this place in the block's forward pass. nnsight cannot reach back
        to a value the model has passed, so ``Internals.read`` sorts by it.
    unavailable : str or None
        Why this family has no such tensor. Any access raises a RenamingError
        carrying the reason.
    tags : frozenset of str
        Facts about the tensor a consumer or a check needs, e.g. ``"sink"``: the
        attention probabilities of a model with attention sinks sum to less than 1.
    """

    module: str = ""
    io: IOType | None = IOType.OUTPUT
    op: tuple[str, ...] | Callable[[Envoy], Any] = ()
    select: tuple[int | str, ...] | Lens | None = None
    order: int = 0
    unavailable: str | None = None
    tags: frozenset[str] = frozenset()

    @property
    def is_attention(self) -> bool:
        return self.module.split(".")[0] == "self_attn"


def _walk(value: Any, path: tuple[int | str, ...]) -> Any:
    for step in path:
        value = value[step]
    return value


def _rebuilt(value: Any, path: tuple[int | str, ...], new: Any) -> Any:
    """``value`` with the element at ``path`` replaced by ``new``; containers are
    rebuilt rather than edited, since a module's return tuple is not ours to mutate."""
    if not path:
        return new
    step, rest = path[0], path[1:]
    inner = _rebuilt(value[step], rest, new)
    if isinstance(value, tuple):
        assert isinstance(step, int)
        items = (*value[:step], inner, *value[step + 1 :])
        return type(value)(*items) if hasattr(value, "_fields") else items
    assert isinstance(value, (list, dict)), f"cannot rebuild a {type(value).__name__}"
    copied = value.copy()
    copied[step] = inner  # type: ignore[index]
    return copied


class LayerAccessor:
    """Per-layer read/write access to the tensor an ``Address`` names.

    ``accessor[i]`` reads, ``accessor[i] = value`` writes, and the address says
    how the tensor is reached: a module's input or output, or one operation
    inside its forward, and where in that value the tensor sits.

    With no ``select`` (every accessor nnterp ships by default), tuple values are
    unwrapped per access: ``accessor[i]`` returns the first element when the value
    at layer ``i`` is a tuple and the value itself otherwise, and
    ``accessor[i] = value`` rebuilds the tuple around ``value``. Nothing is
    inferred from one layer about another, so layers whose modules return
    different structures can be accessed in any order.

    Accessors rooted at ``self_attn`` raise a RenamingError on linear-attention
    layers (see ``linear_attention_error``). An accessor that is unavailable, by
    its address or because it was disabled, raises a RenamingError with the reason.

    ``LayerAccessor(model, "self_attn", IOType.OUTPUT)`` is still accepted and
    builds the address.
    """

    def __init__(
        self,
        model,
        address: "Address | str | None" = None,
        io_type: IOType | None = None,
        disabled_reason: str | None = None,
        name: str | None = None,
    ):
        if not isinstance(address, Address):
            address = Address(address or "", io_type)
        self.model = model
        self.address = address
        self.name = name or address.module or "layers"
        self.disabled_reason = disabled_reason or address.unavailable
        self._is_tuple: dict[int, bool] = {}

    # the spelling the accessor had before it took an address
    @property
    def attr_name(self) -> str | None:
        return self.address.module or None

    @property
    def io_type(self) -> IOType | None:
        return self.address.io

    @property
    def is_attention(self) -> bool:
        return self.address.is_attention

    @property
    def enabled(self) -> bool:
        return self.disabled_reason is None

    def disable(self, reason: str | None = None):
        self.disabled_reason = reason or f"{self.name} is disabled for this model."

    def get_module(self, layer: int) -> Envoy:
        if self.disabled_reason is not None:
            raise RenamingError(self.disabled_reason)
        if self.is_attention and layer in self.model.linear_attention_layers:
            raise linear_attention_error(self.model, layer)
        module = self.model.layers[layer]
        if self.address.module:
            for attr in self.address.module.split("."):
                child = getattr(module, attr, None)
                if child is None:
                    # layers of one model may differ: DeepSeek's first blocks are
                    # dense and the rest mixtures of experts
                    raise RenamingError(
                        f"{self.name} does not exist on layer {layer}: its "
                        f"{type(module._module).__name__} has no {attr!r}."
                    )
                module = child
        return module

    def get_operation(self, layer: int, containing_source: bool = False):
        """The ``.source`` operation this address names, or (``containing_source``)
        the source it is an operation of. A name the installed forward does not
        have raises a RenamingError listing the operations it does have: a moved
        forward is an error here, never a neighbouring tensor."""
        module = self.get_module(layer)
        op = self.address.op
        if callable(op):
            return op(module, containing_source) if containing_source else op(module)
        target = module
        for depth, op_name in enumerate(op):
            source = target.source
            if containing_source and depth == len(op) - 1:
                return source
            try:
                target = getattr(source, op_name)
            except AttributeError as e:
                raise RenamingError(
                    f"{self.name}: the forward of {'.'.join(('layers', str(layer), *filter(None, [self.address.module])))}"
                    f"{''.join('.' + done for done in op[:depth])} has no operation {op_name!r} "
                    f"with this transformers version. Its operations are:\n{source}"
                ) from e
        return target

    def _place(self, layer: int):
        """The object holding the value and the attribute it is under."""
        if self.address.op:
            place = self.get_operation(layer)
            return place, "inputs" if self.io_type == IOType.INPUT else "output"
        return self.get_module(layer), self.io_type.value

    def __getitem__(self, layer: int) -> TraceTensor | Envoy:
        if self.io_type is None:
            return self.get_module(layer)
        place, attribute = self._place(layer)
        value = getattr(place, attribute)
        select = self.address.select
        if select is None:
            is_tuple = isinstance(value, tuple)
            self._is_tuple[layer] = is_tuple
            return value[0] if is_tuple else value
        return select.get(value) if isinstance(select, Lens) else _walk(value, select)

    def __setitem__(self, layer: int, new: TraceTensor):
        if self.io_type is None:
            raise ValueError(
                f"Cannot set the value of a module accessor. Did you mean {self.name}_input/output"
            )
        place, attribute = self._place(layer)
        select = self.address.select
        if select is None:
            current = getattr(place, attribute)
            is_tuple = isinstance(current, tuple)
            self._is_tuple[layer] = is_tuple
            replacement = (new, *current[1:]) if is_tuple else new
        elif isinstance(select, Lens):
            replacement = select.put(getattr(place, attribute), new)
        else:
            replacement = _rebuilt(getattr(place, attribute), select, new)
        setattr(place, attribute, replacement)

    def __call__(self, layer: int) -> TraceTensor | Envoy:
        return self[layer]

    def returns_tuple(self, layer: int) -> bool | None:
        """
        Returns whether the value at ``layer`` is a tuple, as recorded by the last
        access to that layer. Returns None if the layer has not been accessed yet.
        """
        return self._is_tuple.get(layer)

    def print_source(self, layer: int | None = None, allow_dispatch: bool = True):
        """Print the operation this accessor reads, then the forward it is part of."""
        assert self.address.op, f"{self.name} is a module boundary, not an operation of a forward"
        if layer is None:
            layer = self.model.attention_layers[0] if self.is_attention else 0
        sections = []

        def collect():
            sections.append((f"Accessing {self.name} from:", str(self.get_operation(layer))))
            sections.append(
                ("Full module source:", str(self.get_operation(layer, containing_source=True)))
            )

        try_with_scan(
            self.model,
            collect,
            RenamingError(
                f"Can't access {self.name}. It is most likely not yet supported for this architecture and transformers version."
            ),
            allow_dispatch=allow_dispatch,
        )
        if is_notebook():
            display_markdown(
                "\n\n".join(f"## {title}\n```py\n{source}\n```" for title, source in sections)
            )
        else:
            for title, source in sections:
                print(f"{title}\n{source}\n")


def check_attention_probabilities(
    model,
    layer: int | None = None,
    allow_dispatch: bool = True,
    use_trace: bool = True,
):
    """
    Check that ``model.attention_probabilities`` reads the attention pattern.

    This validates that:
    1. The attention probabilities have the expected shape (batch_size, num_heads, seq_len, seq_len)
    2. The probabilities sum to 1 along the last dimension (to at most 1 on a model with attention sinks)
    3. Modifying the probabilities affects the model's output logits. An address can
       read a perfectly good pattern and be causally inert (the weights a mixer
       *returns* are one), and only a write tells the two apart.

    Args:
        layer (int, optional): The layer index to check. Defaults to the first
            softmax-attention layer (``model.attention_layers[0]``).
        allow_dispatch (bool, optional): If True, allows dispatching the model when scan fails.
        use_trace (bool, optional): If False, uses scan() to validate the attention probabilities, which means attention probabilities summing to 1 and causal effect of modifying them won't be tested. If True, the traces run on NDIF when the model is remote and dispatch it otherwise. Defaults to True.

    Raises:
        RenamingError: If the attention probabilities are not properly configured or if the number of attention heads is not available.
    """
    accessor = model.attention_probabilities
    if model.num_heads is None:
        raise RenamingError(
            f"Can't check the shapes of the model internals because the number of attention heads is not available in {model.repo_id} architecture."
            "You should pass the number of attention heads as an integer or look at the config and pass the key in the attn_head_config_key argument of a RenameConfig."
        )
    if layer is None:
        layer = model.attention_layers[0]

    def test_prob_source():
        batch_size, seq_len = model.input_size
        num_heads = model.num_heads
        probs = accessor[layer]
        if probs.shape != (batch_size, num_heads, seq_len, seq_len):
            raise RenamingError(
                f"Attention probabilities have shape {probs.shape} != {(batch_size, num_heads, seq_len, seq_len)} (batch_size, n_head, seq_len, seq_len) in {model.repo_id} architecture. This means it's not properly initialized."
            )
        rnd = th.randn_like(probs).abs()
        rnd = rnd / rnd.sum(dim=-1, keepdim=True)
        accessor[layer] = rnd
        if probs.device != th.device("meta"):
            sum_last = probs.sum(dim=-1)
            if "sink" in accessor.address.tags:
                if not (sum_last > 0).all():
                    raise RenamingError("Attention probabilities should be > 0.")
                if not (sum_last < 1 + 1e-5).all():
                    raise RenamingError(
                        "Attention probabilities should sum to < 1 for models with sink tokens."
                    )
            else:
                atol = 1e-2 if probs.dtype == th.bfloat16 else 1e-5
                if not th.allclose(sum_last, th.ones_like(sum_last), atol=atol):
                    raise RenamingError("Attention probabilities do not sum to 1.")

    if use_trace:
        remote = model.remote
        with model.trace(dummy_inputs(), remote=remote):
            test_prob_source()
            corr_logits = model.logits.save()
        with model.trace(dummy_inputs(), remote=remote):
            clean_logits = model.logits.save()

        if th.allclose(corr_logits, clean_logits):
            raise RenamingError(
                "Attention probabilities are not properly initialized: changing the attention probabilities should change the logits."
            )
        return

    try_with_scan(
        model,
        test_prob_source,
        RenamingError(
            "Can't access attention probabilities. It is most likely not yet supported for this architecture and transformers version."
        ),
        allow_dispatch=allow_dispatch,
        errors_to_raise=(RenamingError,),
    )


# --------------------------------------------------------------------------- #
# The address table
# --------------------------------------------------------------------------- #

#: The accessors of a pre-norm block whose attention goes through transformers'
#: attention interface (Llama, Mistral, Qwen, Gemma, GPT-2, ...). ``order`` is the
#: place in the block's forward pass.
DEFAULT_ADDRESSES: dict[str, Address] = {
    "layers_input": Address("", IOType.INPUT, order=0),
    "attentions": Address("self_attn", None, order=10),
    "attentions_input": Address("self_attn", IOType.INPUT, order=10),
    # The dropout call's output rather than the softmax's: it is the pattern the
    # values are mixed with on every family (after the cast, and after an
    # attention sink has been dropped), and the identity in eval mode.
    "attention_probabilities": Address(
        "self_attn",
        op=("attention_interface_1", "nn_functional_dropout_0"),
        order=20,
    ),
    "attentions_output": Address("self_attn", order=30),
    "mlps": Address("mlp", None, order=40),
    "mlps_input": Address("mlp", IOType.INPUT, order=40),
    "mlps_output": Address("mlp", order=50),
    "layers_output": Address("", order=60),
}

#: The accessors that live on a child nnterp does not rename. Families spell
#: these children differently, and which spelling a model uses is a fact its
#: module tree states, so ``structural_addresses`` reads it there: ``{name}`` is
#: the one candidate that exists on the model. Each is defined by what it is *of*
#: the block, so that these hold on every family that has them:
#:
#:     layers_mid       == layers_input + attentions_output
#:     layers_output    == layers_mid   + mlps_output
#:     mlps_norm_output == mlps_input
#:
#: ``attentions_premix`` is every head's result side by side: ``num_heads * head_dim``
#: wide, which is not ``hidden_size`` on Qwen3 or Gemma. ``mlps_neurons`` is the
#: down projection's input: the activation itself on an ungated MLP (GPT-2), and
#: ``act(gate) * up`` on a gated one, which is why it is not ``mlps_activation``.
STRUCTURAL_ADDRESSES: dict[str, tuple[Address, tuple[str, ...]]] = {
    "attentions_norm_output": (
        Address("{name}", order=5),
        ("input_layernorm", "ln_1", "self_attn_layer_norm", "ln_attn", "norm_1"),
    ),
    "attentions_premix": (
        Address("self_attn.{name}", IOType.INPUT, order=25),
        ("o_proj", "c_proj", "dense", "out_proj"),
    ),
    "layers_mid": (
        Address("{name}", IOType.INPUT, order=33),
        ("post_attention_layernorm", "ln_2", "norm_2"),
    ),
    "mlps_norm_output": (
        Address("{name}", order=36),
        ("post_attention_layernorm", "ln_2", "norm_2"),
    ),
    "mlps_activation": (
        Address("mlp.{name}", order=44),
        ("act_fn", "act", "activation_fn", "gelu_impl"),
    ),
    "mlps_neurons": (
        Address("mlp.{name}", IOType.INPUT, order=47),
        ("down_proj", "c_proj", "dense_4h_to_h", "fc_out", "fc2"),
    ),
}

BlockStructure = Literal["pre_norm", "sandwich_norm", "post_norm", "parallel", "residual_inside"]

#: The pre-MLP norm of a sandwich-norm block. Such a block has a
#: ``post_attention_layernorm`` too, which is the *attention's* post-norm there and
#: the pre-MLP norm everywhere else: the block structure says which, not the name.
_SANDWICH_PRE_MLP_NORM = ("pre_feedforward_layernorm",)

_NO_MID_STREAM = (
    "{name} does not exist on this model: its block is {structure!r}. {why}"
)
_NO_MLP_MODULE = (
    "{name} does not exist on this model: {cls} has no MLP module, its feed-forward "
    "layers are the block's own fc1 / fc2 (layers[i].fc1, layers[i].fc2)."
)

_WHY_NOT = {
    "parallel": "Attention and the MLP both read the block input, so there is no residual stream "
    "between them and no norm of it (on GPT-NeoX the second norm exists, and normalizes the block input).",
    "post_norm": "The MLP reads the residual stream itself; the norm comes after the sublayer.",
}

_NO_CONTRIBUTION = (
    "{name} is disabled for this model: no module exposes the sublayer's "
    "additive contribution to the residual stream (see the warning logged at "
    "load and https://github.com/ndif-team/nnterp/issues/51). Use "
    "layers[i].{module}.output for the raw (residual-added) module output."
)


def _row(name: str, **differs) -> Address:
    return replace(DEFAULT_ADDRESSES[name], **differs)


#: Families that normalize a sublayer's output *before* adding it to the residual
#: stream: Gemma-2/3 (sandwich norms) and OLMo-2 (post-norm). The tensor added is
#: the post-norm's output, not the attention/MLP module's.
POST_SUBLAYER_NORM_MODEL_TYPES = ("gemma2", "gemma3", "gemma3_text", "olmo2")


def post_sublayer_norm(model) -> bool:
    return text_config(model).model_type in POST_SUBLAYER_NORM_MODEL_TYPES


#: What differs per family: ``(model class or predicate on the model, rows)``.
#: Later entries win. ``attentions_output`` / ``mlps_output`` mean the sublayer's
#: additive contribution to the residual stream, so an architecture that adds the
#: residual *inside* the sublayer module (issue #51) points them at the last
#: pre-residual projection.
FAMILY_ADDRESSES: list[tuple[type | Callable[[Any], bool], dict[str, Address]]] = [
    (
        OPTForCausalLM,
        {
            name: _row(name, unavailable=_NO_MLP_MODULE.format(name=name, cls="OPTDecoderLayer"))
            for name in ("mlps", "mlps_input", "mlps_output")
        },
    ),
    (
        # layers_input + attentions_output is the mid-stream, and the mid-stream +
        # mlps_output is layers_output, only if these are the post-norm outputs
        post_sublayer_norm,
        {
            "attentions_output": _row("attentions_output", module="post_attention_layernorm"),
            "mlps_output": _row("mlps_output", module="post_feedforward_layernorm"),
        },
    ),
    (
        BloomForCausalLM,
        {
            "attentions_output": _row("attentions_output", module="self_attn.dense"),
            "mlps_output": _row("mlps_output", module="mlp.dense_4h_to_h"),
            "attention_probabilities": _row(
                "attention_probabilities", op=("self_attention_dropout_0",)
            ),
        },
    ),
    (
        # No module carries the contribution: these checkpoints compute the
        # output projections with F.linear, bypassing the dense modules.
        bloom_slow_but_exact,
        {
            name: _row(name, unavailable=_NO_CONTRIBUTION.format(name=name, module=module))
            for name, module in (("attentions_output", "self_attn"), ("mlps_output", "mlp"))
        },
    ),
    (MptForCausalLM, {
        "mlps_output": _row("mlps_output", module="mlp.down_proj"),
        "attention_probabilities": _row("attention_probabilities", op=("nn_functional_dropout_0",)),
    }),
    (
        # DbrxNormAttentionNorm returns (resid_mid, norm_2(resid_mid), attn_weights):
        # its output[0] is a residual-stream state, the inner attn output is the contribution.
        DbrxForCausalLM,
        {
            # norm_1 is *inside* what nnterp calls self_attn here, so it comes after
            # that container's input rather than before it
            "attentions_norm_output": Address("self_attn.norm_1", order=12),
            "attentions_premix": Address("self_attn.attn.out_proj", IOType.INPUT, order=25),
            "layers_mid": Address("self_attn.norm_2", IOType.INPUT, order=33),
            "mlps_norm_output": Address("self_attn.norm_2", order=36),
            "attentions_output": _row("attentions_output", module="self_attn.attn"),
            "attention_probabilities": _row(
                "attention_probabilities", module="self_attn.attn", op=("nn_functional_dropout_0",)
            ),
        },
    ),
    (
        # FalconAttention calls its dropout only on the alibi branch; the other
        # softmaxes straight into what it returns
        FalconForCausalLM,
        {"attention_probabilities": _row("attention_probabilities", op=("F_softmax_0",))},
    ),
    (
        lambda model: isinstance(model, FalconForCausalLM) and model.config.alibi,
        {"attention_probabilities": _row("attention_probabilities", op=("self_attention_dropout_0",))},
    ),
    (
        GPTJForCausalLM,
        {"attention_probabilities": _row("attention_probabilities", op=("self__attn_0", "self_attn_dropout_0"))},
    ),
    (
        Qwen2MoeForCausalLM,
        {"attention_probabilities": _row("attention_probabilities", op=("nn_functional_dropout_0",))},
    ),
    (
        # the softmax spans the keys plus a sink, and the sink is dropped
        GptOssForCausalLM,
        {"attention_probabilities": _row("attention_probabilities", tags=frozenset({"sink"}))},
    ),
]


def get_block_structure(model) -> BlockStructure:
    """How a block combines its two sublayers with the residual stream. It decides
    which accessors exist and where ``attentions_output`` / ``mlps_output`` are."""
    cfg = text_config(model)
    if isinstance(model, (BloomForCausalLM, MptForCausalLM, DbrxForCausalLM)):
        return "residual_inside"
    if (
        cfg.model_type in ("gptj", "phi", "codegen")
        or getattr(cfg, "use_parallel_residual", False)
        or getattr(cfg, "parallel_attn", False)
        or getattr(cfg, "new_decoder_architecture", False)
    ):
        return "parallel"
    if cfg.model_type == "olmo2":
        return "post_norm"
    if post_sublayer_norm(model):
        return "sandwich_norm"
    return "pre_norm"


def structural_addresses(standardized_model, structure: BlockStructure) -> dict[str, Address]:
    """The rows of ``STRUCTURAL_ADDRESSES`` for this model: each child's name read
    off the module tree, and a reason where the block has no such place."""
    rows = {}
    for name, (address, candidates) in STRUCTURAL_ADDRESSES.items():
        if name in ("layers_mid", "mlps_norm_output") and structure in _WHY_NOT:
            if structure == "post_norm" and name == "layers_mid":
                # what the MLP reads is the residual stream itself
                rows[name] = replace(address, module="mlp")
                continue
            rows[name] = replace(
                address,
                unavailable=_NO_MID_STREAM.format(
                    name=name, structure=structure, why=_WHY_NOT[structure]
                ),
            )
            continue
        if name == "attentions_norm_output" and structure == "post_norm":
            rows[name] = replace(
                address,
                unavailable=_NO_MID_STREAM.format(
                    name=name,
                    structure=structure,
                    why="Attention reads the residual stream itself; the norm comes after the sublayer.",
                ),
            )
            continue
        if name in ("layers_mid", "mlps_norm_output") and structure == "sandwich_norm":
            candidates = _SANDWICH_PRE_MLP_NORM
        parent_path = address.module.rpartition(".")[0]
        found = set()
        for layer in standardized_model.layers:
            parent = getattr(layer, parent_path, None) if parent_path else layer
            if parent is None:  # this layer has no such sublayer (OPT's mlp, a linear-attention layer)
                continue
            found |= {c for c in candidates if getattr(parent._module, c, None) is not None}
        mlps = [getattr(layer, "mlp", None) for layer in standardized_model.layers]
        experts = parent_path == "mlp" and all(
            mlp is not None and hasattr(mlp._module, "experts") for mlp in mlps
        )
        if not found and experts:
            rows[name] = replace(
                address,
                unavailable=f"{name} does not exist on this model: every MLP is a mixture of experts, "
                "where a token goes through top-k of N experts and has no single activation. "
                "mlps_input and mlps_output are the block's boundaries and work as usual.",
            )
            continue
        if len(found) != 1:
            rows[name] = replace(
                address,
                unavailable=f"{name} is not available on this model: of the children "
                f"{list(candidates)}, its {parent_path or 'layers'} have {sorted(found)}. "
                "Name the module with RenameConfig(addresses={...}).",
            )
            continue
        rows[name] = replace(address, module=address.module.format(name=found.pop()))
    return rows


def addresses_for(model, rename_config: RenameConfig | None = None) -> dict[str, Address]:
    """The address of every accessor on ``model``: the defaults, then the rows of
    each family entry that matches, then what the user's RenameConfig says."""
    addresses = dict(DEFAULT_ADDRESSES)
    for matches, rows in FAMILY_ADDRESSES:
        if isinstance(model, matches) if isinstance(matches, type) else matches(model):
            addresses.update(rows)
    if rename_config is not None:
        if rename_config.attn_output_source is not None:
            addresses["attentions_output"] = _row(
                "attentions_output", module=rename_config.attn_output_source
            )
        if rename_config.mlp_output_source is not None:
            addresses["mlps_output"] = _row("mlps_output", module=rename_config.mlp_output_source)
        if rename_config.attn_prob_source is not None:
            addresses["attention_probabilities"] = _row(
                "attention_probabilities", op=rename_config.attn_prob_source
            )
        addresses.update(rename_config.addresses or {})
    return addresses


def get_ignores(model, rename_config: RenameConfig | None = None) -> list[str]:
    ignores = []
    if isinstance(model, IGNORE_MLP_MODELS):
        message = f"{model.__class__.__name__} does not have a mlp module."
        if isinstance(model, OPTForCausalLM):
            message += " You'll have to manually use layers.fc1 and layers.fc2 instead."
        logger.warning(message)
        ignores.append("mlp")
    if bloom_slow_but_exact(model):
        logger.warning(
            f"{model.config.name_or_path} uses pretraining_tp > 1 with slow_but_exact=True, "
            "which computes the attention/MLP output projections with F.linear instead of the "
            "dense modules. No module exposes the sublayer contributions, so attentions_output "
            "/ mlps_output are disabled and attention/MLP checks are skipped "
            "(see https://github.com/ndif-team/nnterp/issues/51)."
        )
        ignores.extend(["attention", "mlp"])
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

    Every layer output is read in forward order (which also records, for
    ``skip_layers``, whether each layer returns a tuple); the attention and MLP
    probes run on the first softmax-attention layer, since the attention
    accessors are not defined on the linear-attention layers of a hybrid.

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
    probe = std_model.attention_layers[0] if std_model.attention_layers else 0

    _check_tensor(
        std_model.token_embeddings, "token_embeddings", expected_hidden, model_name
    )
    _check_tensor(
        std_model.layers_input[0], "layers_input[0]", expected_hidden, model_name
    )

    for layer in range(std_model.num_layers):
        if layer == probe and "attention" not in ignores:
            _check_tensor(
                std_model.attentions_input[layer],
                f"attentions_input[{layer}]",
                expected_hidden,
                model_name,
            )
            _check_tensor(
                std_model.attentions_output[layer],
                f"attentions_output[{layer}]",
                expected_hidden,
                model_name,
            )
        if layer == probe and "mlp" not in ignores:
            _check_tensor(
                std_model.mlps_input[layer],
                f"mlps_input[{layer}]",
                expected_hidden,
                model_name,
            )
            mlp_out = std_model.mlps_output[layer]
            _check_tensor(mlp_out, f"mlps_output[{layer}]", expected_hidden, model_name)

        layer_out = std_model.layers_output[layer]
        _check_tensor(layer_out, f"layers_output[{layer}]", expected_hidden, model_name)
        # Value-based residual-semantics check (issue #51), only when the tensors hold
        # real values (i.e. not during a scan on fake/meta tensors).
        if (
            layer == probe
            and "mlp" not in ignores
            and layer_out.device != th.device("meta")
            and th.allclose(mlp_out, layer_out)
        ):
            raise RenamingError(
                f"mlps_output[{layer}] is identical to layers_output[{layer}] in {model_name} architecture. "
                "This means the MLP module adds the residual stream to its output inside the module, "
                "so mlps_output returns residual-stream states instead of the additive MLP "
                "contribution (see https://github.com/ndif-team/nnterp/issues/51). Pass "
                "RenameConfig(mlp_output_source='<path.to.submodule>') pointing to the submodule "
                "whose output is the additive contribution (e.g. 'mlp.dense_4h_to_h' for BLOOM)."
            )
    _check_tensor(
        std_model.ln_final.output, "ln_final.output", expected_hidden, model_name
    )

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


def _check_attention_layers(std_model, model_name: str):
    """Check that every block exposes exactly one of ``self_attn`` / ``linear_attn``,
    that at least one block is softmax attention, and that the config's
    ``layer_types`` (when present) agrees with the block structure."""
    attention_layers = std_model.attention_layers
    linear_layers = std_model.linear_attention_layers
    if not attention_layers:
        raise RenamingError(
            f"Could not find a self_attn module in any layer of {model_name} architecture. "
            "This means that it was not properly renamed.\n"
            "Please pass the name of the self_attn module to the attn_rename argument."
        )
    both = sorted(set(attention_layers) & set(linear_layers))
    neither = [
        i
        for i in range(std_model.num_layers)
        if i not in attention_layers and i not in linear_layers
    ]
    if both or neither:
        raise RenamingError(
            f"Every layer must expose exactly one of self_attn / {LINEAR_ATTENTION_NAME} in "
            f"{model_name} architecture: layers {both} expose both, layers {neither} expose "
            "neither. This means the attention modules were not properly renamed.\n"
            "Please pass the name of the self_attn module to the attn_rename argument."
        )
    cfg = text_config(std_model._module)
    if "layer_types" in cfg:
        config_linear = [
            i for i, t in enumerate(cfg.layer_types) if t == "linear_attention"
        ]
        if config_linear != linear_layers:
            raise RenamingError(
                f"config.layer_types marks layers {config_linear} as linear_attention, but the "
                f"layers with a {LINEAR_ATTENTION_NAME} module are {linear_layers} in {model_name} "
                "architecture."
            )
    if linear_layers:
        mixer = type(
            getattr(std_model.layers[linear_layers[0]], LINEAR_ATTENTION_NAME)._module
        ).__name__
        logger.info(
            f"Model {model_name} is a hybrid: layers {linear_layers} use linear attention "
            f"({mixer}) and layers {attention_layers} use softmax attention. attentions[i], "
            "attentions_input[i], attentions_output[i] and attention_probabilities[i] are only "
            "defined on the softmax-attention layers (model.attention_layers)."
        )


def _warn_heterogeneous_types(accessor, layers: list[int], kind: str, model_name: str):
    """Warn if modules accessed by ``accessor[i]`` have mixed types across ``layers``."""
    types = {type(accessor[i]._module) for i in layers}
    if len(types) > 1:
        type_names = ", ".join(sorted(t.__name__ for t in types))
        logger.warning(
            f"Model {model_name} has heterogeneous {kind} types across layers: {type_names}. "
            "Some nnterp operations may not work consistently across all layers."
        )


def _check_output_source(
    std_model,
    kind: IgnoreType,
    layer: int,
    model_name: str,
    rename_config: RenameConfig | None = None,
):
    """Check that attentions_output / mlps_output expose the additive sublayer
    contribution, not a residual-added state (issue #51), on ``layer``.

    If the sublayer module takes a residual-like argument in its forward pass, it
    most likely adds the residual to its output inside the module (BLOOM, MPT), so
    an output source pointing to the pre-residual submodule must be configured.
    An explicitly configured source (built-in or via RenameConfig) is trusted,
    including an explicit default source (user asserts the module output is the
    contribution).
    """
    if kind == "attention":
        accessor, default_source, config_field = (
            std_model.attentions_output,
            "self_attn",
            "attn_output_source",
        )
    else:
        accessor, default_source, config_field = (
            std_model.mlps_output,
            "mlp",
            "mlp_output_source",
        )
    try:
        module = accessor.get_module(layer)._module
    except AttributeError as e:
        raise RenamingError(
            f"The configured {config_field}='{accessor.attr_name}' does not resolve to a module "
            f"of layer {layer} in {model_name} architecture."
        ) from e
    explicitly_configured = (
        rename_config is not None and getattr(rename_config, config_field) is not None
    )
    if explicitly_configured or accessor.attr_name != default_source:
        return
    residual_params = [
        name
        for name in inspect.signature(module.forward).parameters
        if "residual" in name
    ]
    if residual_params:
        accessor_name = "attentions_output" if kind == "attention" else "mlps_output"
        raise RenamingError(
            f"The {kind} module ({type(module).__name__}) of {model_name} takes a "
            f"`{residual_params[0]}` argument in its forward pass. This usually means the residual "
            f"stream is added to the sublayer output inside the module, in which case "
            f"{accessor_name} would return residual-stream states instead of the additive {kind} "
            "contribution (see https://github.com/ndif-team/nnterp/issues/51).\n"
            f"If so, pass RenameConfig({config_field}='<path.to.submodule>') pointing to the "
            f"submodule whose output is the additive contribution (e.g. 'self_attn.dense' for "
            f"BLOOM). If the module does not add the residual to its output, pass "
            f"RenameConfig({config_field}='{default_source}') to keep the default behavior."
        )


def check_model_renaming(
    std_model,
    model_name: str,
    ignores: list[IgnoreType],
    allow_dispatch: bool,
    allow_multimodal: bool = False,
    rename_config: RenameConfig | None = None,
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
        _check_attention_layers(std_model, model_name)
        _warn_heterogeneous_types(
            std_model.attentions, std_model.attention_layers, "attention", model_name
        )
        _check_output_source(
            std_model,
            "attention",
            std_model.attention_layers[0],
            model_name,
            rename_config,
        )

    if "mlp" not in ignores:
        _check_has_module(std_model.layers[0], "mlp", model_name, "mlp_rename")
        _warn_heterogeneous_types(
            std_model.mlps, list(range(std_model.num_layers)), "MLP", model_name
        )
        _check_output_source(std_model, "mlp", 0, model_name, rename_config)

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
