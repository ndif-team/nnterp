import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal
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

# Architectures that add the residual stream to the sublayer output *inside* the
# attention/MLP module, so the module output is a residual-stream state instead of
# the additive contribution (https://github.com/ndif-team/nnterp/issues/51).
# Maps model class -> (attn_output_source, mlp_output_source); None keeps the default.
RESIDUAL_INSIDE_SUBLAYER_SOURCES = {
    BloomForCausalLM: ("self_attn.dense", "mlp.dense_4h_to_h"),
    MptForCausalLM: (None, "mlp.down_proj"),
    # DbrxNormAttentionNorm returns (resid_mid, norm_2(resid_mid), attn_weights):
    # its output[0] is a residual-stream state, the inner attn output is the contribution.
    DbrxForCausalLM: ("self_attn.attn", None),
}


def bloom_slow_but_exact(model) -> bool:
    """BLOOM checkpoints with pretraining_tp > 1 and slow_but_exact=True (e.g.
    bigscience/bigscience-small-testing) compute the output projections with
    F.linear, bypassing the dense modules, so no pre-residual module exists."""
    return (
        isinstance(model, BloomForCausalLM)
        and model.config.pretraining_tp > 1
        and model.config.slow_but_exact
    )


def get_output_sources(
    model, rename_config: RenameConfig | None = None
) -> tuple[str | None, str | None]:
    """Resolve the module paths targeted by attentions_output / mlps_output.

    Defaults to the attention/MLP modules themselves. For architectures that add
    the residual inside the sublayer module (BLOOM, MPT), targets the last
    pre-residual projection so the accessors expose the additive contribution.
    Returns None for a sublayer whose contribution is not exposed by any module
    (slow_but_exact BLOOM): the corresponding accessor is disabled.
    """
    attn_source, mlp_source = "self_attn", "mlp"
    if bloom_slow_but_exact(model):
        attn_source = mlp_source = None
    else:
        for model_class, (
            attn_override,
            mlp_override,
        ) in RESIDUAL_INSIDE_SUBLAYER_SOURCES.items():
            if isinstance(model, model_class):
                attn_source = attn_override or attn_source
                mlp_source = mlp_override or mlp_source
    if rename_config is not None:
        if rename_config.attn_output_source is not None:
            attn_source = rename_config.attn_output_source
        if rename_config.mlp_output_source is not None:
            mlp_source = rename_config.mlp_output_source
    return attn_source, mlp_source


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


class LayerAccessor:
    """I/O accessor that provides input/output access with setter.

    Tuple values are unwrapped per access: ``accessor[i]`` returns the first
    element when the value at layer ``i`` is a tuple and the value itself
    otherwise, and ``accessor[i] = value`` rebuilds the tuple around ``value``.
    Nothing is inferred from one layer about another, so layers whose modules
    return different structures can be accessed in any order.

    Accessors rooted at ``self_attn`` raise a RenamingError on linear-attention
    layers (see ``linear_attention_error``). If ``disabled_reason`` is set, any
    access raises a RenamingError with that message (used when no module carries
    the accessor's semantics, e.g. attentions_output on slow_but_exact BLOOM).
    """

    def __init__(
        self,
        model,
        attr_name: str | None,
        io_type: IOType | None,
        disabled_reason: str | None = None,
    ):

        self.model = model
        self.attr_name = attr_name
        self.io_type = io_type
        self.disabled_reason = disabled_reason
        self._is_tuple: dict[int, bool] = {}

    @property
    def is_attention(self) -> bool:
        return (
            self.attr_name is not None and self.attr_name.split(".")[0] == "self_attn"
        )

    def get_module(self, layer: int) -> Envoy:
        if self.disabled_reason is not None:
            raise RenamingError(self.disabled_reason)
        if self.is_attention and layer in self.model.linear_attention_layers:
            raise linear_attention_error(self.model, layer)
        module = self.model.layers[layer]
        if self.attr_name is not None:
            for attr in self.attr_name.split("."):
                module = getattr(module, attr)
        return module

    def __getitem__(self, layer: int) -> TraceTensor | Envoy:
        module = self.get_module(layer)
        if self.io_type is None:
            return module
        elif self.io_type.value == "input":
            target = module.input
        elif self.io_type.value == "output":
            target = module.output
        else:
            raise ValueError(f"Invalid io_type: {self.io_type}")

        is_tuple = isinstance(target, tuple)
        self._is_tuple[layer] = is_tuple
        return target[0] if is_tuple else target

    def __setitem__(self, layer: int, value: TraceTensor):
        if self.io_type is None:
            name = self.attr_name or "layers"
            raise ValueError(
                f"Cannot set the value of a module accessor. Did you mean {name}_input/output"
            )
        module = self.get_module(layer)
        is_input = self.io_type.value == "input"
        current = module.input if is_input else module.output
        is_tuple = isinstance(current, tuple)
        self._is_tuple[layer] = is_tuple
        replacement = (value, *current[1:]) if is_tuple else value
        if is_input:
            module.input = replacement
        else:
            module.output = replacement

    def __call__(self, layer: int) -> TraceTensor | Envoy:
        return self[layer]

    def returns_tuple(self, layer: int) -> bool | None:
        """
        Returns whether the value at ``layer`` is a tuple, as recorded by the last
        access to that layer. Returns None if the layer has not been accessed yet.
        """
        return self._is_tuple.get(layer)


def bloom_attention_prob_source(attention_module, return_module_source: bool = False):
    if return_module_source:
        return attention_module.source
    return attention_module.source.self_attention_dropout_0


def falcon_attention_prob_source(attention_module, return_module_source: bool = False):
    if return_module_source:
        return attention_module.source
    return attention_module.source.F_softmax_0


def default_attention_prob_source(attention_module, return_module_source: bool = False):
    source = attention_module.source.attention_interface_1.source
    if return_module_source:
        return source
    return source.nn_functional_dropout_0


def gptj_attention_prob_source(attention_module, return_module_source: bool = False):
    source = attention_module.source.self__attn_0.source
    if return_module_source:
        return source
    return source.self_attn_dropout_0


def qwen2moe_attention_prob_source(
    attention_module, return_module_source: bool = False
):
    if return_module_source:
        return attention_module.source
    return attention_module.source.nn_functional_dropout_0


def dbrx_attention_prob_source(attention_module, return_module_source: bool = False):
    if return_module_source:
        return attention_module.attn.source
    return attention_module.attn.source.nn_functional_dropout_0


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
        elif isinstance(model._module, BloomForCausalLM):
            self.source_attr = bloom_attention_prob_source
        elif isinstance(model._module, FalconForCausalLM):
            # FalconAttention calls its dropout only on the alibi branch; the
            # other softmaxes straight into what it returns
            self.source_attr = (
                bloom_attention_prob_source
                if model.config.alibi
                else falcon_attention_prob_source
            )
        elif isinstance(model._module, GPTJForCausalLM):
            self.source_attr = gptj_attention_prob_source
        elif isinstance(model._module, (Qwen2MoeForCausalLM, MptForCausalLM)):
            self.source_attr = qwen2moe_attention_prob_source
        elif isinstance(model._module, DbrxForCausalLM):
            self.source_attr = dbrx_attention_prob_source
        else:
            if isinstance(model._module, GptOssForCausalLM):
                # the softmax spans the keys plus a sink, and the sink is dropped
                self.attn_probs_dont_sum_to_one = True
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

    def _attention_module(self, layer: int) -> Envoy:
        if layer in self.model.linear_attention_layers:
            raise linear_attention_error(self.model, layer)
        return self.model.layers[layer].self_attn

    def __getitem__(self, layer: int) -> TraceTensor:
        self._check_enabled()
        return self.source_attr(self._attention_module(layer)).output

    def __setitem__(self, layer: int, value: TraceTensor):
        self._check_enabled()
        self.source_attr(self._attention_module(layer)).output = value

    def check_source(
        self,
        layer: int | None = None,
        allow_dispatch: bool = True,
        use_trace: bool = True,
    ):
        """
        Check that the attention probabilities source is correctly configured.

        This method validates that:
        1. The attention probabilities have the expected shape (batch_size, num_heads, seq_len, seq_len)
        2. The probabilities sum to 1 along the last dimension
        3. Modifying the probabilities affects the model's output logits

        Args:
            layer (int, optional): The layer index to check. Defaults to the first
                softmax-attention layer (``model.attention_layers[0]``).
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
        if layer is None:
            layer = self.model.attention_layers[0]

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

    def print_source(self, layer: int | None = None, allow_dispatch: bool = True):
        if layer is None:
            layer = self.model.attention_layers[0]
        in_notebook = is_notebook()
        if in_notebook:
            markdown_text = "## Accessing attention probabilities from:\n"
        else:
            print("Accessing attention probabilities from:")

        def print_hook_source():
            nonlocal markdown_text
            source = self.source_attr(self._attention_module(layer))
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
                    self._attention_module(layer), return_module_source=True
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
