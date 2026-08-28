from __future__ import annotations

import sys
from typing import Union

from .logging import logger
import torch as th
from nnsight.intervention.tracing.globals import Object
import transformers
from transformers import AutoModelForCausalLM
import nnsight

TraceTensor = Union[th.Tensor, Object]


NNSIGHT_VERSION = nnsight.__version__
TRANSFORMERS_VERSION = transformers.__version__


# Dummy class for missing transformer architectures
class ArchitectureNotFound:
    pass


try:
    from transformers import OPTForCausalLM
except ImportError:
    OPTForCausalLM = ArchitectureNotFound
try:
    from transformers import MixtralForCausalLM
except ImportError:
    MixtralForCausalLM = ArchitectureNotFound
try:
    from transformers import BloomForCausalLM
except ImportError:
    BloomForCausalLM = ArchitectureNotFound
try:
    from transformers import GPT2LMHeadModel
except ImportError:
    GPT2LMHeadModel = ArchitectureNotFound
try:
    from transformers import Qwen2MoeForCausalLM
except ImportError:
    Qwen2MoeForCausalLM = ArchitectureNotFound
try:
    from transformers import DbrxForCausalLM
except ImportError:
    DbrxForCausalLM = ArchitectureNotFound
try:
    from transformers import GPTJForCausalLM
except ImportError:
    GPTJForCausalLM = ArchitectureNotFound
try:
    from transformers import LlamaForCausalLM
except ImportError:
    LlamaForCausalLM = ArchitectureNotFound

try:
    from transformers import StableLmForCausalLM
except ImportError:
    StableLmForCausalLM = ArchitectureNotFound

try:
    from transformers import Qwen3ForCausalLM
except ImportError:
    Qwen3ForCausalLM = ArchitectureNotFound

try:
    from transformers import Qwen2ForCausalLM
except ImportError:
    Qwen2ForCausalLM = ArchitectureNotFound

try:
    from transformers import GptOssForCausalLM
except ImportError:
    GptOssForCausalLM = ArchitectureNotFound

try:
    from transformers import MptForCausalLM
except ImportError:
    MptForCausalLM = ArchitectureNotFound


class TextOnlyAutoModelForCausalLM(AutoModelForCausalLM):
    """``AutoModelForCausalLM`` as returned by ``detect_automodel(text_only=True)``.

    Builds exactly what ``AutoModelForCausalLM`` builds (auto classes dispatch on
    the inherited ``_model_mapping``). It is a distinct class because nnsight 0.7's
    ``LanguageModel`` refuses configs registered with ``AutoModelForImageTextToText``
    unless a non-default ``automodel`` is passed. To drop with the nnsight 0.8
    migration, where ``task="text-generation"`` replaces ``automodel``.
    """


def detect_automodel(
    model: str,
    trust_remote_code: bool = False,
    text_only: bool = False,
):
    """Detect the appropriate AutoModel class for a model by inspecting its config.

    Checks which ``AutoModelFor*`` classes support the model's config via their
    internal ``_model_mapping``. Returns the first match from a priority-ordered list.

    ``AutoModelForImageTextToText`` is checked before ``AutoModelForCausalLM``
    because VLM configs (e.g. Qwen2.5-VL) are registered in both mappings, but
    ``AutoModelForCausalLM`` fails at init for VLM configs that nest
    ``vocab_size`` under ``text_config``.

    Args:
        model: HuggingFace model name or path.
        trust_remote_code: Whether to trust remote code when loading the config.
        text_only: If True and the checkpoint registers a separate text-only causal
            LM class next to its multimodal one (e.g. Mllama, Llama-4, Qwen3.5),
            return that class so only the text tower is loaded. Raises if the
            installed transformers cannot build it from the checkpoint's composite
            config. No effect when there is no separate text-only class.

    Returns:
        The appropriate AutoModel class (e.g. ``AutoModelForCausalLM``).
    """
    from transformers import (
        AutoConfig,
        AutoModelForImageTextToText,
        AutoModelForSeq2SeqLM,
    )

    config = AutoConfig.from_pretrained(model, trust_remote_code=trust_remote_code)
    config_cls = type(config)

    if (
        text_only
        and config_cls in AutoModelForImageTextToText._model_mapping
        and config_cls in AutoModelForCausalLM._model_mapping
    ):
        text_cls = AutoModelForCausalLM._model_mapping[config_cls]
        if text_cls is AutoModelForImageTextToText._model_mapping[config_cls]:
            logger.info(
                f"{model} has no separate text-only class ({text_cls.__name__} is "
                "also its multimodal class), loading the multimodal model."
            )
        else:
            # Not every architecture can build its text tower from the composite
            # config (Llama4ForCausalLM reads vocab_size at the top level in 4.56).
            with th.device("meta"):
                try:
                    AutoModelForCausalLM.from_config(
                        config, trust_remote_code=trust_remote_code
                    )
                except Exception as e:
                    raise ValueError(
                        f"text_only=True but {text_cls.__name__} cannot be built from "
                        f"the {config_cls.__name__} of {model} with transformers "
                        f"{TRANSFORMERS_VERSION}: {e}\nLoad the full multimodal model "
                        "instead (text_only=False or StandardizedVLM)."
                    ) from e
            logger.info(
                f"Loading only the text tower ({text_cls.__name__}) of {model}, "
                "the vision tower is not loaded."
            )
            return TextOnlyAutoModelForCausalLM

    candidates = [
        AutoModelForImageTextToText,
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
    ]

    for cls in candidates:
        if config_cls in cls._model_mapping:
            logger.info(
                f"Auto-detected {cls.__name__} for {model} "
                f"(config: {type(config).__name__})"
            )
            return cls

    logger.info(
        f"No specific AutoModel detected for {model} "
        f"(config: {type(config).__name__}), defaulting to AutoModelForCausalLM"
    )
    return AutoModelForCausalLM


def is_notebook():
    """Detect the current Python environment"""
    try:
        get_ipython = sys.modules["IPython"].get_ipython
        if "IPKernelApp" in get_ipython().config:
            return True
        else:
            return False
    except Exception:
        return False


def display_markdown(text: str):
    from IPython.display import display, Markdown

    display(Markdown(text))


def display_source(source: str):
    display_markdown(f"```py\n{source}\n```")


class DummyCache:
    def to_legacy_cache(self):
        return None


def dummy_inputs():
    return {
        "input_ids": th.tensor([[0, 1, 1]]),
        "attention_mask": th.tensor([[1, 1, 1]]),
    }


def try_with_scan(
    model,
    function,
    error_to_throw: Exception,
    allow_dispatch: bool,
    warn_if_scan_fails: bool = True,
    errors_to_raise: tuple[type[Exception], ...] | type[Exception] | None = None,
):
    """
    Attempt to execute a function using model.scan(), falling back to model.trace() if needed.

    This function tries to execute the given function within a model.scan() context first,
    which avoids dispatching the model. If that fails and fallback is allowed, it will
    try using model.trace() instead. If model.remote is True, the trace runs remotely
    via NDIF; otherwise, the model is dispatched locally.

    Args:
        model (StandardizationMixin): A StandardizationMixin instance (e.g. StandardizedTransformer
            or StandardizedVLLM). Must have .scan(), .trace(), and .remote attributes.
        function: A callable to execute within the model context (takes no arguments)
        error_to_throw (Exception): Exception to raise if both scan and trace fail
        allow_dispatch (bool): Whether to allow fallback to .trace() if .scan() fails
        warn_if_scan_fails (bool, optional): Whether to log warnings when scan fails.
            Defaults to True.
        errors_to_raise (tuple, optional): Tuple of exception types that should be raised
            immediately if encountered during scan, without fallback to trace.

    Returns:
        bool: True if scan succeeded, False if trace was used instead
    """

    try:
        with model.scan(dummy_inputs(), use_cache=False) as tracer:
            function()
            # tracer.stop() TODO: uncomment when nnsight fix this upstream
        return True
    except Exception as e:
        if errors_to_raise is not None and isinstance(e, errors_to_raise):
            raise e
        if not allow_dispatch and not model.dispatched:
            logger.error("Scan failed and trace() fallback is disabled")
            raise error_to_throw from e
        if warn_if_scan_fails:
            logger.warning(
                "Error when trying to scan the model - using .trace() instead (which will dispatch the model)..."
            )
        try:
            with model.trace(dummy_inputs(), remote=model.remote) as tracer:
                function()
                # tracer.stop() TODO: uncomment when nnsight fix this upstream
        except Exception as e2:
            if errors_to_raise is not None and isinstance(e2, errors_to_raise):
                raise e2
            raise error_to_throw from e2
        logger.warning(
            f"Using trace() succeed! Error when trying to scan the model:\n{e}"
        )
        return False


def unpack_tuple(tensor_or_tuple: TraceTensor) -> TraceTensor:
    if isinstance(tensor_or_tuple, tuple):
        return tensor_or_tuple[0]
    return tensor_or_tuple
