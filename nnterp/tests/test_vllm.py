"""Tests for StandardizedVLLM (vLLM backend) support.

Each GPU test gets its own vLLM engine via the ``vllm_model`` fixture
(function-scoped) so that a crash in one test doesn't cascade to others.
"""

import gc
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


# --- Prefix caching (raises before engine start) ---


@requires_cuda
def test_vllm_prefix_caching_raises_without_force():
    """enable_prefix_caching=True without force must raise ValueError."""
    with pytest.raises(ValueError, match="enable_prefix_caching"):
        StandardizedVLLM(
            "gpt2",
            allow_experimental_vllm=True,
            enable_prefix_caching=True,
        )


# --- GPU tests (each gets its own engine for isolation) ---


@pytest.fixture
def vllm_model():
    """Load a vLLM model, yield it, then shut down the engine.

    Each test gets its own engine so a crash doesn't cascade.
    """
    if not th.cuda.is_available():
        pytest.skip("CUDA not available")
    model = StandardizedVLLM("gpt2", allow_experimental_vllm=True)
    yield model
    # Shut down the vLLM engine subprocess and distributed state
    if model.vllm_entrypoint is not None:
        model.vllm_entrypoint.llm_engine.engine_core.shutdown()
    from nnsight.modeling.vllm.vllm import VLLM
    VLLM._cleanup_distributed()
    del model
    gc.collect()


@requires_cuda
def test_vllm_check_renaming_works(vllm_model):
    """check_renaming=True must succeed for vLLM models.

    Validates that check_io correctly handles vLLM's flat tensor shapes
    (no batch dimension) and skips lm_head.output (separate logit phase).
    """
    assert vllm_model.num_layers > 0
    assert vllm_model.hidden_size > 0


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
    """Accessing logits via model.logits.output (WrapperModule) must return correct vocab_size.

    Note: Unlike StandardizedTransformer where logits are at model.output.logits,
    in vLLM logits are accessed via model.logits.output (a WrapperModule envoy).
    """
    with vllm_model.trace("Hello world"):
        logits = vllm_model.logits.output.save()
    assert logits.shape[-1] == vllm_model.vocab_size


@requires_cuda
def test_vllm_skip_layer(vllm_model):
    """Skipping a layer must change the logits.

    Uses layers[i].skip() directly because layer_returns_tuple detection
    happens in the vLLM worker subprocess and doesn't propagate back to
    the user-process LayerAccessor.
    """
    with vllm_model.trace("Hello world"):
        baseline = vllm_model.logits.output.save()
    with vllm_model.trace("Hello world"):
        skip_with = vllm_model.layers_input[1]
        vllm_model.layers[1].skip(skip_with)
        skipped = vllm_model.logits.output.save()
    assert not th.allclose(baseline, skipped, atol=1e-4)


@requires_cuda
def test_vllm_steer(vllm_model):
    """Steering a layer must change the logits."""
    steering_vec = th.randn(vllm_model.hidden_size) * 0.1
    with vllm_model.trace("Hello world"):
        baseline = vllm_model.logits.output.save()
    with vllm_model.trace("Hello world"):
        vllm_model.steer(layers=0, steering_vector=steering_vec)
        steered = vllm_model.logits.output.save()
    assert not th.allclose(baseline, steered, atol=1e-4)


@requires_cuda
def test_vllm_generate_multi_token(vllm_model):
    """model.generate() with tracer.all() must collect tokens at each step.

    Uses tracer.invoke() + tracer.all() pattern with .save() on the list
    so values are transported back from the vLLM worker subprocess.
    Prompt goes in tracer.invoke(), not in .generate().
    """
    with vllm_model.generate(max_new_tokens=3) as tracer:
        out = list().save()
        with tracer.invoke("Hello world"):
            with tracer.all():
                out.append(vllm_model.samples.output.item().save())
    assert len(out) == 3


@requires_cuda
def test_vllm_generate_with_layer_read(vllm_model):
    """model.generate() must allow reading layer activations per step."""
    with vllm_model.generate(max_new_tokens=3) as tracer:
        layers = list().save()
        with tracer.invoke("Hello world"):
            with tracer.all():
                layers.append(vllm_model.layers_output[0])
    assert len(layers) == 3
    assert all(l.shape[-1] == vllm_model.hidden_size for l in layers)


@requires_cuda
def test_vllm_trace_defaults_single_token(vllm_model):
    """model.trace() must default to single forward pass (max_tokens=1)."""
    with vllm_model.trace() as tracer:
        out = list().save()
        with tracer.invoke("Hello world"):
            with tracer.all():
                out.append(vllm_model.samples.output.item().save())
    assert len(out) == 1


@requires_cuda
def test_vllm_project_on_vocab_outside_trace_raises(vllm_model):
    """project_on_vocab outside trace must raise RuntimeError for vLLM.

    vLLM model weights (ln_final, lm_head) are on meta device in the main
    process — only the vLLM worker subprocess has real weights.
    """
    with vllm_model.trace("Hello world"):
        h = vllm_model.layers_output[-1].save()
    with pytest.raises(RuntimeError, match="outside a trace context"):
        vllm_model.project_on_vocab(h)


# --- Unsupported features ---


@requires_cuda
def test_vllm_input_ids(vllm_model):
    """input_ids for vLLM must return (seq_len,) — 1D, no batch dimension."""
    with vllm_model.trace("Hello world"):
        ids = vllm_model.input_ids.save()
    assert ids.ndim == 1, f"Expected 1D input_ids for vLLM, got shape {ids.shape}"
    assert ids.shape[0] > 0


@requires_cuda
def test_vllm_input_size_is_1d(vllm_model):
    """input_size for vLLM must return (seq_len,) — 1D, no batch dimension."""
    with vllm_model.trace("Hello world"):
        input_size = vllm_model.input_size.save()
    assert len(input_size) == 1, f"Expected 1D input_size for vLLM, got {input_size}"


@requires_cuda
def test_vllm_attention_mask_not_supported(vllm_model):
    """attention_mask must raise NotImplementedError for vLLM models."""
    with pytest.raises(NotImplementedError):
        _ = vllm_model.attention_mask


@requires_cuda
def test_vllm_attention_probs_not_supported(vllm_model):
    """enable_attention_probs=True must raise NotImplementedError for vLLM."""
    assert not vllm_model.attn_probs_available


@requires_cuda
def test_vllm_add_prefix_false_tokenizer_not_supported(vllm_model):
    """add_prefix_false_tokenizer must raise ValueError for vLLM models."""
    with pytest.raises(ValueError, match="add_prefix_space"):
        _ = vllm_model.add_prefix_false_tokenizer


# --- load_model integration ---


@requires_cuda
def test_load_model_vllm_returns_standardized_vllm(vllm_model):
    """load_model with use_vllm=True must return StandardizedVLLM."""
    assert isinstance(vllm_model, StandardizedVLLM)


@requires_cuda
def test_load_model_vllm_is_not_transformer(vllm_model):
    """load_model with use_vllm=True must not return StandardizedTransformer."""
    from nnterp import StandardizedTransformer

    assert not isinstance(vllm_model, StandardizedTransformer)
