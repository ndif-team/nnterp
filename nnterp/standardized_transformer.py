from loguru import logger
import torch as th
from torch.nn import Module
from torch import Size
from nnsight import LanguageModel
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from .utils import (
    TraceTensor,
    DummyCache,
    try_with_scan,
)
from .rename_utils import (
    IOType,
    LayerAccessor,
    AttentionProbabilitiesAccessor,
    RenameConfig,
    get_rename_dict,
    get_ignores,
    check_model_renaming,
    get_num_attention_heads,
    get_hidden_size,
    RenamingError,
    get_vocab_size,
)


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

    Args:
        model (str or Module): Hugging Face repository ID or path of the model to load or loaded model.
        check_renaming (bool, default True): If True, the renaming of modules is validated.
            Defaults to True.
        allow_dispatch (bool, default True): If True, allows using trace() to dispatch the model
            when scan() fails during renaming checks. Defaults to True. You should set this to false
            if you plan to use the model remotely.
        enable_attention_probs (bool, default False): If True, enables attention probabilities
            tracing by setting attn_implementation="eager". Defaults to False.
        check_attn_probs_with_trace (bool, default True): If True, the model will be dispatched and a test will ensure that the attention probabilities returned sum to 1.
        rename_config (RenameConfig, default None): A RenameConfig object to use for renaming the model. If None, a default RenameConfig will be used.
    """

    num_layers: int
    num_heads: int
    hidden_size: int
    vocab_size: int
    is_vllm: bool

    def _init_standardization(
        self,
        model: str | Module,
        check_renaming: bool = True,
        allow_dispatch: bool = True,
        enable_attention_probs: bool = False,
        check_attn_probs_with_trace: bool = True,
        rename_config: RenameConfig | None = None,
    ):
        """Initialize standardization after the base model has been initialized."""
        if isinstance(model, str):
            model_name = model
        else:
            model_name = model.__class__.__name__

        ignores = get_ignores(self._model, rename_config)

        # Create accessor instances
        self.layers_input = LayerAccessor(self, None, IOType.INPUT)
        self.layers_output = LayerAccessor(self, None, IOType.OUTPUT)
        self.attentions = LayerAccessor(self, "self_attn", None)
        self.attentions_input = LayerAccessor(self, "self_attn", IOType.INPUT)
        self.attentions_output = LayerAccessor(self, "self_attn", IOType.OUTPUT)
        self.mlps = LayerAccessor(self, "mlp", None)
        self.mlps_input = LayerAccessor(self, "mlp", IOType.INPUT)
        self.mlps_output = LayerAccessor(self, "mlp", IOType.OUTPUT)

        self.num_layers = len(self.layers)
        self.num_heads = get_num_attention_heads(
            self._model, raise_error=False, rename_config=rename_config
        )
        self.hidden_size = get_hidden_size(
            self._model, raise_error=False, rename_config=rename_config
        )
        self.vocab_size = get_vocab_size(
            self._model, raise_error=False, rename_config=rename_config
        )

        if check_renaming:
            check_model_renaming(self, model_name, ignores, allow_dispatch)
        self.attention_probabilities = AttentionProbabilitiesAccessor(
            self,
            rename_config=rename_config,
            initialized_with_enable=enable_attention_probs,
        )
        if self.is_vllm and enable_attention_probs:
            raise NotImplementedError(
                "nnterp VLLM wrapper doesn't support attention probabilities yet, please set enable_attention_probs=False."
            )
        if check_renaming and enable_attention_probs:
            self.attention_probabilities.check_source(
                allow_dispatch=allow_dispatch,
                use_trace=check_attn_probs_with_trace,
            )
        else:
            # Disable attention probabilities as we can't check them without dispatching the model or not validating the sum to 1 and causal effect of modifying them
            self.attention_probabilities.disable()
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

    def detect_layer_output_type(self):
        if self.layers_output.returns_tuple is None:

            def test_layer_0():
                _ = self.layers_output[0]

            try_with_scan(
                self,
                test_layer_0,
                RenamingError(
                    "Unable to access layer outputs. This may indicate an unsupported model architecture."
                ),
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
        return self.attention_probabilities.enabled

    @property
    def input_ids(self) -> TraceTensor:
        if self.is_vllm:
            raise NotImplementedError(
                "input_ids is not supported for VLLM models as it is flattened and without padding."
            )
        return self.inputs[1]["input_ids"]

    @property
    def input_size(self) -> Size:
        """
        Returns the shape of the input tensor (batch_size, sequence_length)
        """
        if self.is_vllm:
            raise NotImplementedError(
                "input_size is not supported for VLLM models as it is flattened and without padding."
            )
        return self.input_ids.shape

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
        """Returns the token embeddings. Equivalent to self.embed_tokens.output"""
        return self.embed_tokens.output

    @token_embeddings.setter
    def token_embeddings(self, value: TraceTensor):
        """Sets the token embeddings. Equivalent to self.embed_tokens.output = value"""
        self.embed_tokens.output = value

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
            layer_returns_tuple: Whether the layer output is a tuple. Doesn't need to be provided if the model's renaming has been validated or if you ran model.detect_layer_output_type() already.
        """
        if layer_returns_tuple is None:
            layer_returns_tuple = self.layers_output.returns_tuple
        if layer_returns_tuple is None:
            raise ValueError(
                "Please run `model.detect_layer_output_type()` before skipping layers or provide the layer_returns_tuple argument."
            )
        if skip_with is None:
            skip_with = self.layers_input[start_layer]
        if layer_returns_tuple and not isinstance(skip_with, tuple):
            skip_with = (skip_with, DummyCache())
        elif not layer_returns_tuple and isinstance(skip_with, tuple):
            raise ValueError(
                "Skipping layer with a tuple while the layer output is not a tuple. This may cause unexpected behavior."
            )
        for layer in range(start_layer, end_layer + 1):
            self.layers[layer].skip(skip_with)

    def steer(
        self,
        layers: int | list[int],
        steering_vector: th.Tensor,
        factor: float = 1,
        positions: int | list[int] | th.Tensor | None = None,
    ):
        """
        Steer the hidden states of a layer using a steering vector by doing layer_output += factor * steering_vector.

        Args:
            layers: The layer(s) to steer
            steering_vector: The steering vector to apply
            factor: The factor to multiply the steering vector by
            positions: The position to steer. If None, all positions are steered.
        """
        if isinstance(layers, int):
            layers = [layers]
        for layer in sorted(layers):  # sort to ensure execution order
            layer_device = self.layers_output[layer].device
            steering_with = factor * steering_vector.to(layer_device)
            if positions is None:
                self.layers_output[layer] += steering_with
            else:
                self.layers_output[layer][:, positions] += steering_with

    def project_on_vocab(self, hidden_state: TraceTensor) -> TraceTensor:
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


class StandardizedTransformer(LanguageModel, StandardizationMixin):
    """
    Renames the LanguageModel modules to match a standardized architecture.

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

    Args:
        model (str or Module): Hugging Face repository ID or path of the model to load or loaded model.
        check_renaming (bool, default True): If True, the renaming of modules is validated.
            Defaults to True.
        allow_dispatch (bool, default True): If True, allows using trace() to dispatch the model
            when scan() fails during renaming checks. Defaults to True. You should set this to false
            if you plan to use the model remotely.
        enable_attention_probs (bool, default False): If True, enables attention probabilities
            tracing by setting attn_implementation="eager". Defaults to False.
        check_attn_probs_with_trace (bool, default True): If True, the model will be dispatched and a test will ensure that the attention probabilities returned sum to 1.
        rename_config (RenameConfig, default None): A RenameConfig object to use for renaming the model. If None, a default RenameConfig will be used.
    """

    is_vllm: bool = False

    def __init__(
        self,
        model: str | Module,
        check_renaming: bool = True,
        allow_dispatch: bool = True,
        enable_attention_probs: bool = False,
        check_attn_probs_with_trace: bool = True,
        rename_config: RenameConfig | None = None,
        **kwargs,
    ):
        kwargs.setdefault("device_map", "auto")
        if "attn_implementation" in kwargs and enable_attention_probs:
            if kwargs["attn_implementation"] != "eager":
                raise ValueError(
                    f"Cannot use attn_implementation='{kwargs['attn_implementation']}' with enable_attention_probs=True. "
                    "Either set enable_attention_probs=False or don't pass attn_implementation."
                )
        attn_implementation = (
            "eager"
            if enable_attention_probs
            else kwargs.pop("attn_implementation", None)
        )

        rename = self._get_rename(
            rename_config=rename_config, user_rename=kwargs.pop("rename", None)
        )
        super().__init__(
            model,
            attn_implementation=attn_implementation,
            rename=rename,
            **kwargs,
        )
        self._init_standardization(
            model=model,
            check_renaming=check_renaming,
            allow_dispatch=allow_dispatch,
            enable_attention_probs=enable_attention_probs,
            check_attn_probs_with_trace=check_attn_probs_with_trace,
            rename_config=rename_config,
        )

    @property
    def logits(self) -> TraceTensor:
        """Returns the predicted logits."""
        return self.output.logits
