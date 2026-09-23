from __future__ import annotations
import warnings

from .logging import logger
import torch as th
from torch.nn import Module
from torch import Size
from nnsight import TransformersModel
from nnsight.ndif import register as ndif_register
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from .utils import (
    TraceTensor,
    DummyCache,
    try_with_scan,
)
from .internals import Internals
from .rename_utils import (
    LayerAccessor,
    RenameConfig,
    get_rename_dict,
    get_attention_layers,
    get_ignores,
    addresses_for,
    get_block_structure,
    get_head_dim,
    get_intermediate_size,
    get_num_kv_heads,
    get_qk_head_dim,
    structural_addresses,
    check_attention_probabilities,
    check_model_renaming,
    get_num_attention_heads,
    get_hidden_size,
    RenamingError,
    get_vocab_size,
)


#: The two rows whose model attribute is a property rather than the accessor:
#: public names older than the address table that return the tensor itself.
#: ``model.logits`` reads the ``logits`` row and ``model.token_embeddings`` the
#: ``embeddings_output`` row, so there is still one answer per place.
COMPATIBILITY_PROPERTIES = ("logits", "token_embeddings")


class StandardizationMixin:
    """
    Mixin class for standardizing the architecture of a model.

    This class provides built-in accessors to extract and set intermediate activations:

    - embed_tokens: Get embedding module
    - token_embeddings: Get/set token embeddings (equivalent to embed_tokens.output)
    - layers[i]: Get layer module at layer i
    - layers_input[i]: Get/set layer input at layer i
    - layers_output[i]: Get/set layer output at layer i
    - attentions[i]: Get attention module at layer i
    - attentions_input[i] / attentions_output[i]: Get/set attention input/output at layer i
    - mlps[i]: Get MLP module at layer i
    - mlps_input[i] / mlps_output[i]: Get/set MLP input/output at layer i
    - internals: every accessor of this model by name, in forward order, with what
      it is and why a family has none (``model.internals.status()``). Each row of
      the table is an attribute of the model like the ones above.
    - the whole-model places — embeddings_input, embeddings_output, ln_final_output,
      lm_head_output — are accessors called rather than indexed:
      ``model.ln_final_output()`` reads and ``model.ln_final_output[None] = value``
      writes. ``logits`` is a row of the table too, but ``model.logits`` stays the
      tensor property it always was, as ``model.token_embeddings`` (the
      ``embeddings_output`` row) stays: those two are the compatibility
      spellings, and each reads its row.
    - attention_layers / linear_attention_layers: Indices of the softmax-attention
      blocks (``layers[i].self_attn``) and of the linear-attention blocks
      (``layers[i].linear_attn``, Gated DeltaNet in Qwen3-Next / Qwen3.5 hybrids).
      The attention accessors and attention_probabilities are only defined on
      the former and raise a RenamingError on the latter.

    attentions_output[i] / mlps_output[i] never include the residual stream: on
    architectures that add the residual inside the attention/MLP module (BLOOM,
    MPT, DBRX), they target the last pre-residual submodule instead of the module
    output (see issue #51), and on architectures that normalize the sublayer's
    output before adding it (Gemma-2/3, OLMo-2) they target that post-sublayer
    layernorm, whose output is the tensor added to the residual stream. The raw
    module outputs are ``attentions[i].output`` and ``mlps[i].output``.

    Args:
        model (str or Module): Hugging Face repository ID or path of the model to load or loaded model.
        check_renaming (bool, default True): If True, the renaming of modules is validated.
            Defaults to True.
        remote (bool, default False): If True, registers nnterp for NDIF remote execution via
            cloudpickle serialization and keeps the checkpoint off the client: allow_dispatch
            is set to False and every load-time check runs with scan() on the meta model.
        allow_dispatch (bool, default True): If True, allows using trace() to dispatch the model
            when scan() fails during renaming checks. Defaults to True. Automatically set to False
            when remote=True.
        enable_attention_probs (bool, default False): If True, enables attention probabilities
            tracing by setting attn_implementation="eager" (passing attn_implementation="eager"
            yourself is accepted, any other value raises). Defaults to False.
        check_attn_probs_with_trace (bool, default None): If True, the attention probabilities are
            validated with a trace, which tests that they sum to 1 and that editing them changes
            the logits. The trace dispatches the model, or runs on NDIF when remote=True. If False,
            they are validated with scan() (shape only). Defaults to True, and to False when
            remote=True.
        rename_config (RenameConfig, default None): A RenameConfig object to use for renaming the model. If None, a default RenameConfig will be used.
    """

    num_layers: int
    attention_layers: list[int]
    linear_attention_layers: list[int]
    num_heads: int
    hidden_size: int
    vocab_size: int
    head_dim: int
    qk_head_dim: int
    num_kv_heads: int
    intermediate_size: int
    block_structure: str
    is_vllm: bool
    remote: bool

    # One accessor per row of the address table, set in _init_standardization.
    # These say so for a reader and a type checker; the table is what creates them.
    internals: Internals
    embeddings_input: LayerAccessor
    embeddings_output: LayerAccessor
    ln_final_output: LayerAccessor
    lm_head_output: LayerAccessor
    layers_input: LayerAccessor
    layers_mid: LayerAccessor
    layers_output: LayerAccessor
    attentions: LayerAccessor
    attentions_input: LayerAccessor
    attentions_norm_output: LayerAccessor
    attentions_premix: LayerAccessor
    attentions_output: LayerAccessor
    attention_probabilities: LayerAccessor
    mlps: LayerAccessor
    mlps_input: LayerAccessor
    mlps_norm_output: LayerAccessor
    mlps_activation: LayerAccessor
    mlps_neurons: LayerAccessor
    mlps_output: LayerAccessor

    def _init_standardization(
        self,
        model: str | Module,
        check_renaming: bool = True,
        remote: bool = False,
        allow_dispatch: bool = True,
        enable_attention_probs: bool = False,
        check_attn_probs_with_trace: bool | None = None,
        allow_multimodal: bool = False,
        rename_config: RenameConfig | None = None,
    ):
        """Initialize standardization after the base model has been initialized."""
        self.remote = remote
        if remote:
            # The checkpoint lives on NDIF: validate on the meta model with scan()
            # and never dispatch it on the client.
            ndif_register("nnterp")
            allow_dispatch = False
        if check_attn_probs_with_trace is None:
            check_attn_probs_with_trace = not remote
        if isinstance(model, str):
            model_name = model
        else:
            model_name = model.__class__.__name__

        # One accessor per row of the address table: the defaults, what this
        # family does differently (rename_utils.FAMILY_ADDRESSES: e.g. attentions_output
        # / mlps_output target a submodule where the residual is added inside the
        # sublayer module, issue #51), then the user's RenameConfig.
        # The children nnterp does not rename (norms, projections, activation) are
        # named off this model's module tree; a family or the user may still say
        # otherwise, so their rows win.
        self.block_structure = get_block_structure(self._module)
        addresses = structural_addresses(self, self.block_structure) | addresses_for(
            self._module, rename_config
        )
        if self.is_vllm:
            # vLLM computes the logits outside the model's forward, so model.logits
            # is nnsight's own and not this row's object: better no row than one
            # status() calls available and a read would take from the wrong place.
            addresses.pop("logits", None)
        self.internals = Internals(self, addresses)
        # Every row is an attribute of the model: model.layers_output[i] for a
        # per-layer place, model.lm_head_output() for a whole-model one. Adding a
        # place is adding a row and nothing else, and a row that would take a name
        # the model already uses is an error rather than a silent clobber.
        for name, accessor in self.internals.items():
            if name in COMPATIBILITY_PROPERTIES:
                continue
            if hasattr(self, name):
                raise RenamingError(
                    f"The address named {name!r} cannot become model.{name}: the model "
                    f"already has one ({type(getattr(self, name)).__name__}). Name the row "
                    "something else; it is reachable as model.internals[name] either way."
                )
            setattr(self, name, accessor)

        self.num_layers = len(self.layers)
        # From the block structure: a softmax-attention block exposes self_attn, a
        # linear-attention block keeps linear_attn (check_model_renaming verifies
        # that every block exposes exactly one and cross-checks config.layer_types).
        self.attention_layers, self.linear_attention_layers = get_attention_layers(
            self.layers
        )
        self.num_heads = get_num_attention_heads(
            self._module, raise_error=False, rename_config=rename_config
        )
        self.hidden_size = get_hidden_size(
            self._module, raise_error=False, rename_config=rename_config
        )
        self.vocab_size = get_vocab_size(
            self._module, raise_error=False, rename_config=rename_config
        )
        # like num_heads and hidden_size above, None where the config does not
        # say (a RenameConfig key names it there)
        known = self.num_heads is not None and self.hidden_size is not None
        self.head_dim = get_head_dim(self._module) if known else None
        self.qk_head_dim = get_qk_head_dim(self._module) if known else None
        self.num_kv_heads = get_num_kv_heads(self._module) if known else None
        self.intermediate_size = get_intermediate_size(self._module) if known else None

        # a sublayer whose contribution the table says this model does not expose
        # (OPT has no MLP module) is one the renaming checks cannot check
        ignores = get_ignores(self, rename_config)
        if check_renaming:
            check_model_renaming(
                self,
                model_name,
                ignores,
                allow_dispatch,
                allow_multimodal,
                rename_config=rename_config,
            )
        if self.is_vllm and enable_attention_probs:
            raise NotImplementedError(
                "nnterp VLLM wrapper doesn't support attention probabilities yet, please set enable_attention_probs=False."
            )
        if check_renaming and enable_attention_probs:
            check_attention_probabilities(
                self,
                allow_dispatch=allow_dispatch,
                use_trace=check_attn_probs_with_trace,
            )
        else:
            # Disable attention probabilities as we can't check them without dispatching the model or not validating the sum to 1 and causal effect of modifying them
            self.attention_probabilities.disable(
                "Attention probabilities are disabled for this model."
                + (
                    ""
                    if enable_attention_probs
                    else " Set enable_attention_probs=True when loading the model to enable them."
                )
            )
        self._add_prefix_false_tokenizer = None

    def _get_rename(
        self,
        rename_config: RenameConfig | None = None,
        user_rename: dict[str, str] | None = None,
    ):
        rename = get_rename_dict(rename_config=rename_config)
        if user_rename is not None:
            logger.info(
                f"Updating default rename with user-provided rename: {user_rename}"
            )
            rename.update(user_rename)
        return rename

    def _prepare_init_kwargs(self, enable_attention_probs, rename_config, **kwargs):
        """Preprocess kwargs shared across StandardizedTransformer, StandardizedVLM, etc.

        Returns (attn_implementation, rename, kwargs) ready to pass to super().__init__.
        """
        kwargs.setdefault("device_map", "auto")
        attn_implementation = kwargs.pop("attn_implementation", None)
        if enable_attention_probs:
            if attn_implementation not in (None, "eager"):
                raise ValueError(
                    f"Cannot use attn_implementation='{attn_implementation}' with enable_attention_probs=True. "
                    "Either set enable_attention_probs=False or don't pass attn_implementation."
                )
            attn_implementation = "eager"
        rename = self._get_rename(
            rename_config=rename_config, user_rename=kwargs.pop("rename", None)
        )
        return attn_implementation, rename, kwargs

    def detect_layer_output_type(self):
        """Record, for every layer, whether its output is a tuple (``skip_layers``
        needs it). Already done by the renaming checks; only runs the layers that
        have not been accessed yet."""
        missing = [
            layer
            for layer in range(self.num_layers)
            if self.layers_output.returns_tuple(layer) is None
        ]
        if missing:

            def read_layer_outputs():
                for layer in missing:
                    _ = self.layers_output[layer]

            try_with_scan(
                self,
                read_layer_outputs,
                RenamingError(
                    "Unable to access layer outputs. This may indicate an unsupported model architecture."
                ),
                allow_dispatch=True,
                warn_if_scan_fails=False,
            )

    @property
    def add_prefix_false_tokenizer(self) -> PreTrainedTokenizerBase:
        """
        Returns the tokenizer with add_prefix_space=False. Which means that "word" and " word" will be tokenized as different tokens.
        """
        if self.is_vllm:
            raise ValueError(
                "nnterp VLLM wrapper doesn't support add_prefix_space=False, the normal tokenizer might already work but it might be model dependent."
            )
        if self._add_prefix_false_tokenizer is None:
            self._add_prefix_false_tokenizer = AutoTokenizer.from_pretrained(
                self.name_or_path, add_prefix_space=False
            )
        return self._add_prefix_false_tokenizer

    @property
    def attn_probs_available(self) -> bool:
        if not self.attention_layers:
            return False
        probe = self.attention_layers[0]
        return self.attention_probabilities.unavailable_on(probe) is None

    @property
    def input_ids(self) -> TraceTensor:
        """Returns the input token IDs.

        For HF models: shape ``(batch_size, sequence_length)``.
        For vLLM models: shape ``(sequence_length,)`` (no batch dimension).
        """
        return self.inputs[1]["input_ids"]

    @property
    def input_size(self) -> Size:
        """Returns the shape of the input tensor.

        For HF models: ``(batch_size, sequence_length)``.
        For vLLM models: ``(sequence_length,)`` (no batch dimension).
        """
        return self.inputs[1]["input_ids"].shape

    @property
    def attention_mask(self) -> TraceTensor:
        """Returns the attention mask tensor."""
        if self.is_vllm:
            raise NotImplementedError(
                "attention_mask is not supported yet for VLLM models as it's not in the inputs dictionary."
            )
        return self.inputs[1]["attention_mask"]

    @property
    def token_embeddings(self) -> TraceTensor:
        """Returns the token embeddings: the ``embeddings_output`` row, which is
        ``embed_tokens.output``."""
        return self.embeddings_output()

    @token_embeddings.setter
    def token_embeddings(self, value: TraceTensor):
        """Sets the token embeddings, through the same row."""
        self.embeddings_output[None] = value

    @property
    def next_token_probs(self) -> TraceTensor:
        """Returns the predicted probabilities for the next token.
        Assumes padding_side is "left"."""
        return self.logits[:, -1, :].softmax(-1)

    def skip_layer(self, layer: int, skip_with: TraceTensor | None = None):
        """
        Skip the computation of a layer.

        Args:
            layer: The layer to skip
            skip_with: The input to skip the layer with. If None, the input of the layer is used.
        """
        return self.skip_layers(layer, layer, skip_with)

    def skip_layers(
        self,
        start_layer: int,
        end_layer: int,
        skip_with: TraceTensor | None = None,
        layer_returns_tuple: bool | None = None,
    ):
        """
        Skip all layers between start_layer and end_layer (inclusive).

        Args:
            start_layer: The layer to start skipping from
            end_layer: The layer to stop skipping at (inclusive)
            skip_with: The tensor to skip the layers with, will be passed as the output of the layers. If None, the input of start_layer is used.
            layer_returns_tuple: Whether the layer outputs are tuples. Doesn't need to be provided if the model's renaming has been validated or if you ran model.detect_layer_output_type() already, in which case it is known per layer.
        """
        if skip_with is None:
            skip_with = self.layers_input[start_layer]
        for layer in range(start_layer, end_layer + 1):
            returns_tuple = layer_returns_tuple
            if returns_tuple is None:
                returns_tuple = self.layers_output.returns_tuple(layer)
            if returns_tuple is None:
                raise ValueError(
                    f"Please run `model.detect_layer_output_type()` before skipping layer {layer} or provide the layer_returns_tuple argument."
                )
            replacement = skip_with
            if returns_tuple and not isinstance(skip_with, tuple):
                replacement = (skip_with, DummyCache())
            elif not returns_tuple and isinstance(skip_with, tuple):
                raise ValueError(
                    f"Skipping layer {layer} with a tuple while its output is not a tuple. This may cause unexpected behavior."
                )
            self.layers[layer].skip(replacement)

    def steer(
        self,
        layers: int | list[int],
        steering_vector: th.Tensor,
        factor: float = 1,
        positions: int | list[int] | th.Tensor | None = None,
        token_positions: int | list[int] | th.Tensor | None = None,
        batch_index: int | list[int] | th.Tensor | None = None,
    ):
        """
        Steer the hidden states of a layer using a steering vector by doing layer_output += factor * steering_vector.

        Args:
            layers: The layer(s) to steer.
            steering_vector: The steering vector to apply.
            factor: The factor to multiply the steering vector by.
            positions: Deprecated, use token_positions instead.
            token_positions: Token positions to steer along the sequence dimension. If None, all positions are steered.
            batch_index: Batch indices to steer. If None, all batch elements are steered.
        """
        if positions is not None:
            if token_positions is not None:
                raise ValueError(
                    "Cannot specify both `positions` (deprecated) and `token_positions`."
                )
            warnings.warn(
                "`positions` is deprecated, use `token_positions` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            token_positions = positions

        if isinstance(layers, int):
            layers = [layers]
        for layer in sorted(layers):  # sort to ensure execution order
            layer_output = self.layers_output[layer]
            steering_with = factor * steering_vector.to(
                device=layer_output.device, dtype=layer_output.dtype
            )
            if self.is_vllm:
                # vLLM inference tensors don't support inplace ops
                if batch_index is None and token_positions is None:
                    self.layers_output[layer] = (
                        self.layers_output[layer] + steering_with
                    )
                elif batch_index is not None and token_positions is not None:
                    out = self.layers_output[layer].clone()
                    out[batch_index, token_positions] = (
                        out[batch_index, token_positions] + steering_with
                    )
                    self.layers_output[layer] = out
                elif token_positions is not None:
                    out = self.layers_output[layer].clone()
                    out[:, token_positions] = out[:, token_positions] + steering_with
                    self.layers_output[layer] = out
                else:
                    out = self.layers_output[layer].clone()
                    out[batch_index] = out[batch_index] + steering_with
                    self.layers_output[layer] = out
            else:
                if batch_index is None and token_positions is None:
                    self.layers_output[layer] += steering_with
                elif batch_index is not None and token_positions is not None:
                    self.layers_output[layer][
                        batch_index, token_positions
                    ] += steering_with
                elif token_positions is not None:
                    self.layers_output[layer][:, token_positions] += steering_with
                else:
                    self.layers_output[layer][batch_index] += steering_with

    def project_on_vocab(self, hidden_state: TraceTensor) -> TraceTensor:
        """Project a hidden state onto the vocabulary space.

        For vLLM models, this must be called inside a ``model.trace()`` context
        because ``ln_final``/``lm_head`` weights live in the vLLM worker subprocess
        and are on ``meta`` device in the main process.
        """
        if self.is_vllm and not self.interleaving:
            raise RuntimeError(
                "project_on_vocab cannot be called outside a trace context for vLLM models "
                "because ln_final/lm_head weights are on meta device. "
                "Call it inside model.trace() instead."
            )
        hidden_state = self.ln_final(hidden_state)
        return self.lm_head(hidden_state)

    def probs_to_dict(self, tokens: th.Tensor, probs: th.Tensor) -> dict[str, float]:
        """
        Convert a tensor of probabilities to a dictionary mapping tokens to their probabilities
        """
        return {
            token: prob.item()
            for token, prob in zip(self.tokenizer.convert_ids_to_tokens(tokens), probs)
        }

    def get_topk_closest_tokens(
        self, hidden_state: th.Tensor, k=5
    ) -> dict[str, float] | list[dict[str, float]]:
        """
        Get the top-k closest tokens to the hidden state h.

        Args:
            h: The hidden state to project on the vocabulary. Shape (batch_size, hidden_size) or (hidden_size,).
            k: The number of top tokens to return.
            returns_df: If True, returns a DataFrame instead of a dictionary. Note that you need to have pandas installed for this to work.
                Pandas is included in ``pip install nnterp[display]``.

        Returns:
            A dictionary mapping tokens to their probabilities if h is 1D, or a list of dictionaries if h is 2D.
        """
        if hidden_state.shape[-1] != self.hidden_size and self.hidden_size is not None:
            raise ValueError(
                f"Hidden state shape {hidden_state.shape} does not match model hidden size {self.hidden_size}."
            )

        logits = self.project_on_vocab(hidden_state)
        probs = logits.softmax(-1)
        topk_tokens = th.topk(probs, k=k, dim=-1)
        if hidden_state.ndim == 1:
            return self.probs_to_dict(topk_tokens.indices, topk_tokens.values)
        elif hidden_state.ndim == 2:
            return [
                self.probs_to_dict(topk_tokens.indices[i], topk_tokens.values[i])
                for i in range(hidden_state.shape[0])
            ]
        else:
            raise ValueError(
                f"Unsupported hidden state shape {hidden_state.shape}. Expected 1D or 2D tensor."
            )


class StandardizedTransformer(TransformersModel, StandardizationMixin):
    """
    Renames the TransformersModel modules to match a standardized architecture.

    The model structure is organized as follows::

        StandardizedTransformer
        ├── embed_tokens
        ├── layers
        │   ├── self_attn
        │   └── mlp
        ├── ln_final
        └── lm_head

    The following properties are also available:

    - num_layers: int
    - attention_layers: list[int] (blocks with a softmax ``self_attn``)
    - linear_attention_layers: list[int] (blocks with a ``linear_attn`` mixer, e.g. Qwen3-Next / Qwen3.5 hybrids)
    - num_heads: int
    - hidden_size: int
    - vocab_size: int

    In addition to renaming modules, this class provides built-in accessors to extract and set intermediate activations:

    - embed_tokens: Get embedding module
    - token_embeddings: Get/set token embeddings (equivalent to embed_tokens.output)
    - layers[i]: Get layer module at layer i
    - layers_input[i]: Get/set layer input at layer i
    - layers_output[i]: Get/set layer output at layer i
    - attentions[i]: Get attention module at layer i
    - attentions_input[i] / attentions_output[i]: Get/set attention input/output at layer i
    - mlps[i]: Get MLP module at layer i
    - mlps_input[i] / mlps_output[i]: Get/set MLP input/output at layer i

    On hybrid models mixing linear attention (Gated DeltaNet) and softmax attention
    blocks, the attention accessors and attention_probabilities are only defined on
    the softmax-attention layers (``attention_layers``) and raise a RenamingError
    on the others (``linear_attention_layers``), whose mixer stays at
    ``layers[i].linear_attn``.

    attentions_output[i] / mlps_output[i] never include the residual stream: on
    architectures that add the residual inside the attention/MLP module (BLOOM,
    MPT, DBRX), they target the last pre-residual submodule instead of the module
    output (see issue #51), and on architectures that normalize the sublayer's
    output before adding it (Gemma-2/3, OLMo-2) they target that post-sublayer
    layernorm, whose output is the tensor added to the residual stream. The raw
    module outputs are ``attentions[i].output`` and ``mlps[i].output``.

    Args:
        model (str or Module): Hugging Face repository ID or path of the model to load or loaded model.
        check_renaming (bool, default True): If True, the renaming of modules is validated.
            Defaults to True.
        remote (bool, default False): If True, registers nnterp for NDIF remote execution via
            cloudpickle serialization and keeps the checkpoint off the client: allow_dispatch
            is set to False and every load-time check runs with scan() on the meta model.
        allow_dispatch (bool, default True): If True, allows using trace() to dispatch the model
            when scan() fails during renaming checks. Defaults to True. Automatically set to False
            when remote=True.
        enable_attention_probs (bool, default False): If True, enables attention probabilities
            tracing by setting attn_implementation="eager" (passing attn_implementation="eager"
            yourself is accepted, any other value raises). Defaults to False.
        check_attn_probs_with_trace (bool, default None): If True, the attention probabilities are
            validated with a trace, which tests that they sum to 1 and that editing them changes
            the logits. The trace dispatches the model, or runs on NDIF when remote=True. If False,
            they are validated with scan() (shape only). Defaults to True, and to False when
            remote=True.
        rename_config (RenameConfig, default None): A RenameConfig object to use for renaming the model. If None, a default RenameConfig will be used.
        text_only (bool, default False): If True and the checkpoint registers a separate text-only
            causal LM class next to its multimodal one (e.g. Mllama, Llama-4, Qwen3.5), load only that
            text tower (no vision weights). No effect on text models or on multimodal checkpoints
            without a separate text-only class. See ``detect_automodel``.
    """

    is_vllm: bool = False

    def __init__(
        self,
        model: str | Module,
        check_renaming: bool = True,
        remote: bool = False,
        allow_dispatch: bool = True,
        enable_attention_probs: bool = False,
        check_attn_probs_with_trace: bool | None = None,
        rename_config: RenameConfig | None = None,
        automodel=None,
        text_only: bool = False,
        tokenizer_kwargs: dict | None = None,
        **kwargs,
    ):
        # Detect VLMs and warn
        if automodel is None and isinstance(model, str):
            from .utils import detect_automodel
            from transformers import AutoModelForImageTextToText

            automodel = detect_automodel(
                model,
                trust_remote_code=kwargs.get("trust_remote_code", False),
                text_only=text_only,
            )
            if automodel is AutoModelForImageTextToText:
                warnings.warn(
                    f"Model {model!r} appears to be a vision-language model. "
                    "Consider using StandardizedVLM or load_model() instead for proper image input support.",
                    UserWarning,
                    stacklevel=2,
                )

        attn_implementation, rename, kwargs = self._prepare_init_kwargs(
            enable_attention_probs, rename_config, **kwargs
        )
        super().__init__(
            model,
            task="text-generation",
            attn_implementation=attn_implementation,
            rename=rename,
            **kwargs,
        )
        for key, value in (tokenizer_kwargs or {}).items():
            setattr(self.tokenizer, key, value)
        self._init_standardization(
            model=model,
            check_renaming=check_renaming,
            remote=remote,
            allow_dispatch=allow_dispatch,
            enable_attention_probs=enable_attention_probs,
            check_attn_probs_with_trace=check_attn_probs_with_trace,
            rename_config=rename_config,
        )

    def _remoteable_class(self) -> type:
        return TransformersModel

    @property
    def logits(self) -> TraceTensor:
        """Returns the predicted logits: the ``logits`` field of the model's
        output, which is what it predicts from (capped where the head's output is
        not, on Gemma-2). The row behind it is ``model.internals["logits"]``."""
        return self.internals["logits"]()


class StandardizedVLM(TransformersModel, StandardizationMixin):
    """Standardized wrapper for vision-language models (e.g. Qwen2.5-VL, LLaVA).

    Extends nnsight's ``TransformersModel`` with the same standardized
    module access as ``StandardizedTransformer``. Supports image inputs
    via the ``images`` kwarg in ``model.trace()``.

    Args:
        model (str or Module): Hugging Face repository ID or path of the model to load.
        check_renaming (bool, default True): If True, the renaming of modules is validated.
        remote (bool, default False): If True, registers nnterp for NDIF remote execution and
            keeps the checkpoint off the client (allow_dispatch=False, checks run with scan()).
        allow_dispatch (bool, default True): If True, allows using trace() to dispatch the model
            when scan() fails during renaming checks. Set to False when remote=True.
        enable_attention_probs (bool, default False): If True, enables attention probabilities
            tracing by setting attn_implementation="eager".
        check_attn_probs_with_trace (bool, default None): If True, validates attention
            probabilities with a trace (on NDIF when remote=True), otherwise with scan().
            Defaults to True, and to False when remote=True.
        allow_multimodal (bool, default False): Whether to allow heterogeneous layer types
            (e.g. cross-attention layers in Mllama). These layers only activate with image
            inputs, so text-only tracing will fail on them.
        rename_config (RenameConfig, default None): A RenameConfig object to use for renaming.
    """

    is_vllm: bool = False

    def _remoteable_class(self) -> type:
        return TransformersModel

    def __init__(
        self,
        model: str | Module,
        check_renaming: bool = True,
        remote: bool = False,
        allow_dispatch: bool = True,
        enable_attention_probs: bool = False,
        check_attn_probs_with_trace: bool | None = None,
        allow_multimodal: bool = False,
        rename_config: RenameConfig | None = None,
        tokenizer_kwargs: dict | None = None,
        **kwargs,
    ):
        attn_implementation, rename, kwargs = self._prepare_init_kwargs(
            enable_attention_probs, rename_config, **kwargs
        )
        super().__init__(
            model,
            task="image-text-to-text",
            attn_implementation=attn_implementation,
            rename=rename,
            **kwargs,
        )
        for key, value in (tokenizer_kwargs or {}).items():
            setattr(self.tokenizer, key, value)
        self._init_standardization(
            model=model,
            check_renaming=check_renaming,
            remote=remote,
            allow_dispatch=allow_dispatch,
            enable_attention_probs=enable_attention_probs,
            check_attn_probs_with_trace=check_attn_probs_with_trace,
            allow_multimodal=allow_multimodal,
            rename_config=rename_config,
        )

    @property
    def logits(self) -> TraceTensor:
        """Returns the predicted logits: the ``logits`` field of the model's
        output, which is what it predicts from (capped where the head's output is
        not, on Gemma-2). The row behind it is ``model.internals["logits"]``."""
        return self.internals["logits"]()
