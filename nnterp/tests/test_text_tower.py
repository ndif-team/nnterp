"""Text-tower loading for dual-registered composite configs.

These tests are offline and deterministic: the predicate under test reads only
transformers' auto-mapping tables, and the model_types it is exercised on are
discovered from those tables rather than hard-coded, so the tests describe the
contract instead of pinning a transformers release.
"""

import pytest
from transformers import AutoModelForCausalLM

from nnterp.text_tower import (
    TextTowerAutoModelForCausalLM,
    resolve_automodel,
    text_tower_is_available,
)


def _mappings():
    try:
        from transformers.models.auto.modeling_auto import (
            MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
            MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES,
        )
    except ImportError:
        pytest.skip("transformers does not expose the auto-mapping name tables")
    return MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES


class _Config:
    """The only thing the predicate reads off a config."""

    def __init__(self, model_type):
        self.model_type = model_type


def _first(model_types):
    return sorted(model_types)[0] if model_types else None


def test_subclass_is_the_escape_hatch_nnsight_documents():
    """nnsight stands down for a non-default automodel via an identity check.

    ``LanguageModel._check_is_text_only`` returns early when
    ``self.automodel is not AutoModelForCausalLM``. That identity is the whole
    mechanism, so assert it directly.
    """
    assert TextTowerAutoModelForCausalLM is not AutoModelForCausalLM
    assert issubclass(TextTowerAutoModelForCausalLM, AutoModelForCausalLM)


def test_subclass_dispatches_identically_to_its_parent():
    """Being a marker must not change which class gets built."""
    assert (
        TextTowerAutoModelForCausalLM._model_mapping
        is AutoModelForCausalLM._model_mapping
    )


def test_dual_registered_config_offers_a_text_tower():
    clm, itt = _mappings()
    dual = _first(set(clm) & set(itt))
    if dual is None:
        pytest.skip("no dual-registered model_type in this transformers version")
    assert text_tower_is_available(_Config(dual))


def test_multimodal_only_config_offers_no_text_tower():
    clm, itt = _mappings()
    vlm_only = _first(set(itt) - set(clm))
    if vlm_only is None:
        pytest.skip("no multimodal-only model_type in this transformers version")
    assert not text_tower_is_available(_Config(vlm_only))


def test_plain_text_config_needs_no_special_handling():
    clm, itt = _mappings()
    text_only = _first(set(clm) - set(itt))
    assert text_only is not None, "expected at least one text-only model_type"
    assert not text_tower_is_available(_Config(text_only))


def test_config_without_model_type_is_not_a_text_tower():
    assert not text_tower_is_available(_Config(None))
    assert not text_tower_is_available(object())


def test_resolve_automodel_ignores_a_preloaded_module():
    """A module was already built by the caller; there is nothing to resolve."""
    assert resolve_automodel(object()) is None


def test_resolve_automodel_is_never_the_thing_that_fails_a_load(monkeypatch):
    """An unreadable config must fall through to nnsight, not raise here."""
    import nnterp.text_tower as tt

    def boom(*args, **kwargs):
        raise OSError("no such repo")

    monkeypatch.setattr("transformers.AutoConfig.from_pretrained", boom, raising=True)
    assert tt.resolve_automodel("definitely/not-a-real-repo") is None
