"""Tests for StandardizedVLLM (vLLM backend) support.

Note: The nnsight+vLLM backend is experimental. The ``check_model_renaming``
step during init uses ``try_with_scan`` which falls back to ``model.trace()``,
and this often triggers ``EngineDeadError`` in vLLM. For this reason, most
integration tests use ``check_renaming=False``. A dedicated test documents
that ``check_renaming=True`` currently fails.
"""

import warnings

import pytest
import torch as th

from nnterp import load_model

# StandardizedVLLM requires vLLM with correct version — skip module if unavailable
try:
    from nnterp.standardized_vllm import StandardizedVLLM
except ImportError as e:
    pytest.skip(f"StandardizedVLLM not available: {e}", allow_module_level=True)

requires_cuda = pytest.mark.skipif(
    not th.cuda.is_available(), reason="CUDA not available"
)


# --- Experimental flag gating (no GPU needed) ---


def test_vllm_raises_without_experimental_flag():
    """StandardizedVLLM must raise RuntimeError when allow_experimental_vllm is not set."""
    with pytest.raises(RuntimeError, match="experimental"):
        StandardizedVLLM("gpt2")


def test_vllm_raises_with_experimental_flag_false():
    """StandardizedVLLM must raise RuntimeError when allow_experimental_vllm=False."""
    with pytest.raises(RuntimeError, match="experimental"):
        StandardizedVLLM("gpt2", allow_experimental_vllm=False)


def test_load_model_raises_without_experimental_flag():
    """load_model(use_vllm=True) must raise when allow_experimental_vllm is not set."""
    with pytest.raises(RuntimeError, match="experimental"):
        load_model("gpt2", use_vllm=True)


def test_load_model_raises_with_experimental_flag_false():
    """load_model(use_vllm=True) must raise when allow_experimental_vllm=False."""
    with pytest.raises(RuntimeError, match="experimental"):
        load_model("gpt2", use_vllm=True, allow_experimental_vllm=False)


# --- Prefix caching (raises before model load, no GPU needed beyond init) ---


@requires_cuda
def test_vllm_prefix_caching_raises_without_force():
    """enable_prefix_caching=True without force must raise ValueError."""
    with pytest.raises(ValueError, match="enable_prefix_caching"):
        StandardizedVLLM(
            "gpt2",
            allow_experimental_vllm=True,
            enable_prefix_caching=True,
        )


# --- GPU tests (use check_renaming=False to avoid EngineDeadError) ---


@pytest.fixture(scope="module")
def vllm_model():
    """Load a small vLLM model for the test module.

    Uses check_renaming=False because check_model_renaming triggers
    try_with_scan → model.trace() which causes EngineDeadError in vLLM.
    """
    if not th.cuda.is_available():
        pytest.skip("CUDA not available")
    model = StandardizedVLLM("gpt2", allow_experimental_vllm=True, check_renaming=False)
    return model


@requires_cuda
def test_vllm_check_renaming_fails():
    """Document that check_renaming=True currently fails with vLLM.

    The check_model_renaming step uses try_with_scan which falls back to
    model.trace(). This triggers a vLLM EngineDeadError because the renaming
    check's dummy_inputs are incompatible with vLLM's engine lifecycle.
    This test documents the current limitation.
    """
    with pytest.raises(Exception):
        StandardizedVLLM("gpt2", allow_experimental_vllm=True, check_renaming=True)


@requires_cuda
def test_vllm_warns_on_init():
    """StandardizedVLLM must emit a UserWarning about experimental status."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        StandardizedVLLM("gpt2", allow_experimental_vllm=True, check_renaming=False)
        experimental_warnings = [x for x in w if "experimental" in str(x.message)]
        assert len(experimental_warnings) >= 1, (
            f"Expected at least 1 experimental warning, got {len(experimental_warnings)}"
        )


@requires_cuda
def test_vllm_is_vllm_flag(vllm_model):
    """StandardizedVLLM must have is_vllm=True."""
    assert vllm_model.is_vllm is True


@requires_cuda
def test_vllm_properties(vllm_model):
    """StandardizedVLLM must expose num_layers, hidden_size, vocab_size."""
    assert vllm_model.num_layers is not None and vllm_model.num_layers > 0
    assert vllm_model.hidden_size is not None and vllm_model.hidden_size > 0
    assert vllm_model.vocab_size is not None and vllm_model.vocab_size > 0


@requires_cuda
def test_vllm_layer_activation(vllm_model):
    """Accessing layers_output[0] must return a tensor with correct hidden_size."""
    with vllm_model.trace("Hello world"):
        out = vllm_model.layers_output[0].save()
    assert out.shape[-1] == vllm_model.hidden_size


@requires_cuda
def test_vllm_logits(vllm_model):
    """Accessing logits must return a tensor with correct vocab_size."""
    with vllm_model.trace("Hello world"):
        logits = vllm_model.logits.save()
    assert logits.shape[-1] == vllm_model.vocab_size


@requires_cuda
def test_vllm_skip_layer(vllm_model):
    """Skipping a layer must change the logits."""
    with vllm_model.trace("Hello world"):
        baseline = vllm_model.logits.save()
    with vllm_model.trace("Hello world"):
        vllm_model.skip_layer(1)
        skipped = vllm_model.logits.save()
    assert not th.allclose(baseline, skipped, atol=1e-4)


@requires_cuda
def test_vllm_steer(vllm_model):
    """Steering a layer must change the logits."""
    steering_vec = th.randn(vllm_model.hidden_size) * 0.1
    with vllm_model.trace("Hello world"):
        baseline = vllm_model.logits.save()
    with vllm_model.trace("Hello world"):
        vllm_model.steer(layers=0, steering_vector=steering_vec)
        steered = vllm_model.logits.save()
    assert not th.allclose(baseline, steered, atol=1e-4)


@requires_cuda
def test_vllm_project_on_vocab(vllm_model):
    """project_on_vocab must return tensor with vocab_size last dim."""
    with vllm_model.trace("Hello world"):
        h = vllm_model.layers_output[-1].save()
    vocab_logits = vllm_model.project_on_vocab(h)
    assert vocab_logits.shape[-1] == vllm_model.vocab_size


# --- Unsupported features ---


@requires_cuda
def test_vllm_input_ids_not_supported(vllm_model):
    """input_ids must raise NotImplementedError for vLLM models."""
    with vllm_model.trace("Hello world"):
        with pytest.raises(NotImplementedError):
            _ = vllm_model.input_ids


@requires_cuda
def test_vllm_input_size_not_supported(vllm_model):
    """input_size must raise NotImplementedError for vLLM models."""
    with vllm_model.trace("Hello world"):
        with pytest.raises(NotImplementedError):
            _ = vllm_model.input_size


@requires_cuda
def test_vllm_attention_mask_not_supported(vllm_model):
    """attention_mask must raise NotImplementedError for vLLM models."""
    with vllm_model.trace("Hello world"):
        with pytest.raises(NotImplementedError):
            _ = vllm_model.attention_mask


@requires_cuda
def test_vllm_attention_probs_not_supported():
    """enable_attention_probs=True must raise NotImplementedError for vLLM.

    Note: with check_renaming=True this fails at renaming before reaching the
    attention probs check, so we use check_renaming=False.
    """
    with pytest.raises(NotImplementedError, match="attention probabilities"):
        StandardizedVLLM(
            "gpt2",
            allow_experimental_vllm=True,
            enable_attention_probs=True,
            check_renaming=False,
        )


@requires_cuda
def test_vllm_add_prefix_false_tokenizer_not_supported(vllm_model):
    """add_prefix_false_tokenizer must raise ValueError for vLLM models."""
    with pytest.raises(ValueError, match="add_prefix_space"):
        _ = vllm_model.add_prefix_false_tokenizer


# --- load_model integration ---


@requires_cuda
def test_load_model_vllm_returns_standardized_vllm():
    """load_model with use_vllm=True must return StandardizedVLLM."""
    model = load_model(
        "gpt2", use_vllm=True, allow_experimental_vllm=True, check_renaming=False
    )
    assert isinstance(model, StandardizedVLLM)


@requires_cuda
def test_load_model_vllm_is_not_transformer():
    """load_model with use_vllm=True must not return StandardizedTransformer."""
    from nnterp import StandardizedTransformer

    model = load_model(
        "gpt2", use_vllm=True, allow_experimental_vllm=True, check_renaming=False
    )
    assert not isinstance(model, StandardizedTransformer)
