from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .standardized_vllm import StandardizedVLLM
from .standardized_transformer import StandardizedTransformer
from .rename_utils import get_rename_dict
from .nnsight_utils import ModuleAccessor


__all__ = [
    "StandardizedTransformer",
    "load_model",
    "get_rename_dict",
    "ModuleAccessor",
]


def load_model(
    model: str, use_vllm: bool = False, **model_kwargs
) -> Union[StandardizedTransformer, "StandardizedVLLM"]:
    """
    Load a model using the appropriate wrapper.

    Args:
        model: The model to load.
        use_vllm: Whether to use the VLLM wrapper.
        **model_kwargs: Keyword arguments to pass to the model wrapper.

    Returns:
        A StandardizedTransformer or StandardizedVLLM instance.
    """
    if use_vllm:
        from .standardized_vllm import StandardizedVLLM

        return StandardizedVLLM(model, **model_kwargs)
    else:
        return StandardizedTransformer(model, **model_kwargs)
