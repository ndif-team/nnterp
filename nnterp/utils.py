from __future__ import annotations

import sys
from typing import Union

from .logging import logger
import torch as th
from nnsight.intervention.tracing.globals import Object
import transformers
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
    from transformers import GptOssForCausalLM
except ImportError:
    GptOssForCausalLM = ArchitectureNotFound
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
    from transformers import Qwen3ForCausalLM
except ImportError:
    Qwen3ForCausalLM = ArchitectureNotFound

try:
    from transformers import Qwen2ForCausalLM
except ImportError:
    Qwen2ForCausalLM = ArchitectureNotFound


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
    return {"input_ids": th.tensor([[0, 1, 1]])}


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
    try using model.trace() instead, which does dispatch the model.

    Args:
        model: The model object that supports .scan() and .trace() methods
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
            tracer.stop()
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
            with model.trace(dummy_inputs()) as tracer:
                function()
                tracer.stop()
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
