"""Tests for VLM (Vision-Language Model) support."""

import warnings

import pytest
import torch as th

from nnterp import StandardizedTransformer, StandardizedVLM, load_model
from .utils import get_all_test_models, is_vlm, is_vlm_available


VLM_TEST_MODELS = [m for m in get_all_test_models() if is_vlm(m) and is_vlm_available(m)]


@pytest.fixture(
    params=VLM_TEST_MODELS
    if VLM_TEST_MODELS
    else [pytest.param(None, marks=pytest.mark.skip(reason="No VLM test models found"))]
)
def vlm_model_name(request):
    return request.param


@pytest.fixture
def vlm(vlm_model_name):
    return load_model(vlm_model_name)


# --- load_model autodetection ---


def test_load_model_returns_vlm(vlm):
    assert isinstance(vlm, StandardizedVLM)


def test_load_model_returns_transformer_for_causal():
    model = load_model("gpt2")
    assert isinstance(model, StandardizedTransformer)
    assert not isinstance(model, StandardizedVLM)


# --- StandardizedTransformer warns on VLM ---


def test_standardized_transformer_warns_on_vlm(vlm_model_name):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model = StandardizedTransformer(vlm_model_name)
        vlm_warnings = [x for x in w if "vision-language model" in str(x.message)]
        assert len(vlm_warnings) == 1, f"Expected 1 VLM warning, got {len(vlm_warnings)}"


# --- VLM properties ---


def test_vlm_properties(vlm):
    assert vlm.num_layers is not None and vlm.num_layers > 0
    assert vlm.hidden_size is not None and vlm.hidden_size > 0
    assert vlm.vocab_size is not None and vlm.vocab_size > 0
    assert vlm.is_vllm is False


# --- VLM interventions ---


def test_vlm_layer_activation(vlm):
    with vlm.trace("Hello world"):
        out = vlm.layers_output[0].save()
    assert out.shape[-1] == vlm.hidden_size


def test_vlm_project_on_vocab(vlm):
    with vlm.trace("Hello world"):
        h = vlm.layers_output[-1].save()
    vocab_logits = vlm.project_on_vocab(h)
    assert vocab_logits.shape[-1] == vlm.vocab_size


def test_vlm_skip_layer(vlm):
    with vlm.trace("Hello world"):
        baseline = vlm.logits.save()
    with vlm.trace("Hello world"):
        vlm.skip_layer(1)
        skipped = vlm.logits.save()
    assert not th.allclose(baseline, skipped, atol=1e-4)


def test_vlm_steer(vlm):
    steering_vec = th.randn(vlm.hidden_size) * 0.1
    with vlm.trace("Hello world"):
        baseline = vlm.logits.save()
    with vlm.trace("Hello world"):
        vlm.steer(layers=0, steering_vector=steering_vec)
        steered = vlm.logits.save()
    assert not th.allclose(baseline, steered, atol=1e-4)


def test_vlm_steer_batch_index(vlm):
    prompts = ["Hello world", "Goodbye world"]
    steering_vec = th.randn(vlm.hidden_size) * 0.1
    with vlm.trace(prompts):
        baseline = vlm.logits.save()
    with vlm.trace(prompts):
        vlm.steer(layers=0, steering_vector=steering_vec, batch_index=0)
        steered = vlm.logits.save()
    assert not th.allclose(baseline[0], steered[0], atol=1e-4)
    assert th.allclose(baseline[1], steered[1], atol=1e-4)
