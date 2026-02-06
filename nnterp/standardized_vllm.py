from loguru import logger
import torch as th
from torch.nn import Module
from nnsight.modeling.vllm import VLLM
from .rename_utils import (
    RenameConfig,
    hf_kwargs_to_vllm_kwargs,
)
from .standardized_transformer import StandardizationMixin


class StandardizedVLLM(VLLM, StandardizationMixin):
    """
    Renames the VLLM modules to match a standardized architecture.

    The model structure is organized as follows::

        StandardizedVLLM
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
        remote (bool, default False): If True, sets allow_dispatch=False and registers nnterp
            for NDIF remote execution via cloudpickle serialization.
        allow_dispatch (bool, default True): If True, allows using trace() to dispatch the model
            when scan() fails during renaming checks. Defaults to True. Automatically set to False
            when remote=True.
        enable_attention_probs (bool, default False): If True, enables attention probabilities
            tracing by setting attn_implementation="eager". Defaults to False. Note: nnterp VLLM wrapper doesn't support attention probabilities yet
        check_attn_probs_with_trace (bool, default True): If True, the model will be dispatched and a test will ensure that the attention probabilities returned sum to 1.
        rename_config (RenameConfig, default None): A RenameConfig object to use for renaming the model. If None, a default RenameConfig will be used.
        force_dangerous_prefix_caching (bool, default False): If True, allows using enable_prefix_caching=True. Only use if you know what you are doing as this could lead to some of your interventions leaking to other requests or interventions being skipped for some tokens in context.
    """

    is_vllm: bool = True

    def __init__(
        self,
        model: str | Module,
        check_renaming: bool = True,
        remote: bool = False,
        allow_dispatch: bool = True,
        enable_attention_probs: bool = False,
        check_attn_probs_with_trace: bool = True,
        rename_config: RenameConfig | None = None,
        force_dangerous_prefix_caching: bool = False,
        **vllm_kwargs,
    ):
        if th.cuda.is_available():
            vllm_kwargs.setdefault("tensor_parallel_size", th.cuda.device_count())
        else:
            logger.warning("No CUDA devices found")
        if vllm_kwargs.get("enable_prefix_caching", False):
            if not force_dangerous_prefix_caching:
                raise ValueError(
                    "enable_prefix_caching=True is dangerous, as you will reuse kv cache from previous edited requests, which could lead to some of your interventions leaking to other requests or interventions being skipped for some tokens in context. Only use if you know what you are doing. To force enable it, set force_dangerous_prefix_caching=True."
                )
            else:
                logger.warning(
                    "Using enable_prefix_caching=True is dangerous, as you will reuse kv cache from previous edited requests, which could lead to some of your interventions leaking to other requests or interventions being skipped for some tokens in context. Only use if you know what you are doing."
                )
        else:
            vllm_kwargs["enable_prefix_caching"] = False
        rename = self._get_rename(
            rename_config=rename_config, user_rename=vllm_kwargs.pop("rename", None)
        )
        super().__init__(
            model,
            rename=rename,
            **vllm_kwargs,
        )
        self._init_standardization(
            model=model,
            check_renaming=check_renaming,
            remote=remote,
            allow_dispatch=allow_dispatch,
            enable_attention_probs=enable_attention_probs,
            check_attn_probs_with_trace=check_attn_probs_with_trace,
            rename_config=rename_config,
        )

    def _prepare_input(self, *args, is_trace: bool = True, **kwargs):
        """Preprocess inputs and convert HF kwargs to VLLM kwargs."""
        if is_trace:
            kwargs.setdefault("max_tokens", 1)
        if "max_tokens" in kwargs and kwargs["max_tokens"] != 1:
            logger.warning(
                "max_tokens != 1 will result in behavior similar to LanguageModel.generate() "
                "with multiple forward pass. If you want your code to be compatible with both, "
                "use vllm_model.generate."
            )
        kwargs = hf_kwargs_to_vllm_kwargs(args, kwargs)
        return super()._prepare_input(*args, **kwargs)
