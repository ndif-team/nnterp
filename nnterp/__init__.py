from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .standardized_vllm import StandardizedVLLM
from .standardized_transformer import StandardizedTransformer, StandardizedVLM
from .rename_utils import get_rename_dict
from .nnsight_utils import ModuleAccessor
from .utils import detect_automodel


__all__ = [
    "StandardizedTransformer",
    "StandardizedVLM",
    "load_model",
    "detect_automodel",
    "get_rename_dict",
    "ModuleAccessor",
]


def load_model(
    model: str,
    use_vllm: bool = False,
    **model_kwargs,
) -> Union[StandardizedTransformer, "StandardizedVLLM", StandardizedVLM]:
    """Load a model using the appropriate wrapper.

    Autodetects vision-language models and uses ``StandardizedVLM`` for them.
    Use ``use_vllm=True`` to force vLLM backend.

    Args:
        model: HuggingFace model name or path.
        use_vllm: Whether to use the vLLM wrapper.
        **model_kwargs: Keyword arguments to pass to the model wrapper.

    Returns:
        A StandardizedTransformer, StandardizedVLM, or StandardizedVLLM instance.
    """
    if use_vllm:
        from .standardized_vllm import StandardizedVLLM

        return StandardizedVLLM(model, **model_kwargs)

    from transformers import AutoModelForImageTextToText

    automodel_cls = detect_automodel(
        model,
        trust_remote_code=model_kwargs.get("trust_remote_code", False),
    )
    if automodel_cls is AutoModelForImageTextToText:
        return StandardizedVLM(model, **model_kwargs)

    return StandardizedTransformer(model, **model_kwargs)
