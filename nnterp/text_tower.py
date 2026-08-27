"""Loading the text tower of a composite (vision-)language model.

Some checkpoints ship one config that registers *both* a multimodal model and a
text-only causal LM. ``Qwen/Qwen3.6-35B-A3B`` (``model_type qwen3_5_moe``) is the
motivating case: transformers maps it to ``Qwen3_5MoeForConditionalGeneration``
under ``AutoModelForImageTextToText`` **and** to ``Qwen3_5MoeForCausalLM`` under
``AutoModelForCausalLM``.

``nnsight.LanguageModel`` refuses any config registered with
``AutoModelForImageTextToText`` (``LanguageModel._check_is_text_only``), on the
premise that loading it through ``AutoModelForCausalLM`` would fail deep inside
HuggingFace because the text fields are nested under ``config.text_config``.
That premise holds for genuine VLMs such as ``llava`` or ``qwen2_vl``, whose
``AutoModelForCausalLM`` mapping is empty — but not for a dual-registered
config, where ``AutoModelForCausalLM`` resolves to the text tower and loads it
cleanly.

This module detects that case and opts out of the guard. nnsight's guard stands
down for any non-default ``automodel``, so :class:`TextTowerAutoModelForCausalLM`
opts in explicitly while resolving through the identical ``_model_mapping``.
"""

from __future__ import annotations

from transformers import AutoModelForCausalLM

from .logging import logger


class TextTowerAutoModelForCausalLM(AutoModelForCausalLM):
    """``AutoModelForCausalLM``, tagged so nnsight's multimodal guard stands down.

    Behaviourally identical to its parent: auto classes dispatch on the
    inherited ``_model_mapping``, and nnsight only ever calls ``from_config`` and
    ``from_pretrained`` on the class it is given. The subclass exists solely so
    that ``automodel is not AutoModelForCausalLM`` — the documented escape hatch
    in ``nnsight.LanguageModel._check_is_text_only`` for callers who have chosen
    their own automodel and do not want to be second-guessed.
    """


def text_tower_is_available(config) -> bool:
    """Whether ``config`` is a composite config that also offers a text tower.

    True only when the config's ``model_type`` is registered with **both**
    ``AutoModelForImageTextToText`` and ``AutoModelForCausalLM``. A genuine
    multimodal-only config (no causal-LM mapping) returns False, and so does an
    ordinary text model (no image-text-to-text mapping) — the caller needs no
    special handling in either case.
    """
    model_type = getattr(config, "model_type", None)
    if not model_type:
        return False
    try:
        from transformers.models.auto.modeling_auto import (
            MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
            MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES,
        )
    except ImportError:  # pragma: no cover - very old/very new transformers
        return False
    return (
        model_type in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
        and model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
    )


def resolve_automodel(repo_id, trust_remote_code: bool = False):
    """The automodel to hand nnsight for ``repo_id``, or None to keep its default.

    Returns :class:`TextTowerAutoModelForCausalLM` when ``repo_id`` names a
    dual-registered composite config, so the text tower is loaded and nnsight's
    multimodal refusal does not fire. Returns None otherwise, including when the
    config cannot be read — a failure to look ahead should never turn into a
    failure to load, so the caller falls back to nnsight's own handling and its
    error message.
    """
    if not isinstance(repo_id, str):
        return None
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            repo_id, trust_remote_code=trust_remote_code
        )
    except Exception as e:  # noqa: BLE001 - diagnosis only; nnsight reports for real
        logger.debug(
            f"Could not read the config of {repo_id!r} to check for a text tower "
            f"({type(e).__name__}: {e}). Leaving nnsight's automodel untouched."
        )
        return None
    if not text_tower_is_available(config):
        return None
    logger.info(
        f"{repo_id!r} (model_type {getattr(config, 'model_type', None)!r}) is a "
        "composite multimodal config that also registers a text-only causal LM. "
        "Loading the text tower via AutoModelForCausalLM; the vision tower is "
        "not loaded. Pass text_tower=False to get nnsight's default handling."
    )
    return TextTowerAutoModelForCausalLM
