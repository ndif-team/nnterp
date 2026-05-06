"""End-to-end equivalence between HuggingFace and vLLM standardized accessors.

Loads the same Llama-arch model under both backends, runs the same prompt,
and asserts that each ``StandardizedTransformer`` accessor returns the same
hidden states. This is the contract that lets users swap backends without
their intervention code changing meaning.

Bootstrap, not exhaustive — covers the layer-level accessors that go through
``LayerAccessor``. Sub-module accessors (``self_attn.q_proj``, etc.) and
generation paths are intentionally out of scope here.

Note: tests are not parametrized by accessor name because the project's
conftest extracts pytest parametrize ids as model names (and tries to load
them from HF). One loop-test per pattern keeps the pytest-collection name
clean while still surfacing per-accessor failures via collected errors.
"""
import gc
import pytest
import torch as th

from nnterp import StandardizedTransformer

try:
    from nnterp.standardized_vllm import StandardizedVLLM
    from nnsight.modeling.vllm.vllm import VLLM
except ImportError as e:  # vllm not installed
    pytest.skip(f"vLLM unavailable: {e}", allow_module_level=True)


pytestmark = pytest.mark.skipif(
    not th.cuda.is_available(), reason="vLLM requires CUDA"
)

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"  # Llama-arch, ~135M params
PROMPT = "The Eiffel Tower is in"

# bfloat16 has ~8 bits mantissa. Each layer accumulates error; tolerate ~0.2
# absolute diff. vLLM's PagedAttention uses different kernels than HF's eager
# attention, so exact equivalence is not expected.
BF16_ATOL = 0.2
BF16_RTOL = 0.05

# Strict forward-pass order — nnsight requires accesses inside one trace to
# walk the graph monotonically (CLAUDE.md, VLLM_GUIDE). Reading layer 5 before
# layer 0's output, or attention output before attention input, raises
# MissedProviderError.
#
# ``ln_final.output`` is intentionally excluded: vLLM Llama's fused RMSNorm
# returns ``(normalized, residual)`` while HF returns a single tensor — the
# accessor isn't yet standardized across backends.
ACCESSORS = [
    ("token_embeddings",      lambda m: m.token_embeddings),
    ("layers_input[0]",       lambda m: m.layers_input[0]),
    ("attentions_input[0]",   lambda m: m.attentions_input[0]),
    ("attentions_output[0]",  lambda m: m.attentions_output[0]),
    ("mlps_input[0]",         lambda m: m.mlps_input[0]),
    ("mlps_output[0]",        lambda m: m.mlps_output[0]),
    ("layers_output[0]",      lambda m: m.layers_output[0]),
    ("layers_input[5]",       lambda m: m.layers_input[5]),
    ("attentions_input[5]",   lambda m: m.attentions_input[5]),
    ("layers_output[5]",      lambda m: m.layers_output[5]),
]


@pytest.fixture(scope="module")
def hf_model():
    """Load the model once under HF (bfloat16) and reuse across tests."""
    model = StandardizedTransformer(MODEL, torch_dtype=th.bfloat16, device_map="cuda")
    yield model
    del model
    gc.collect()
    th.cuda.empty_cache()


@pytest.fixture(scope="module")
def vllm_model():
    """Load the model once under vLLM and reuse across tests."""
    model = StandardizedVLLM(MODEL, allow_experimental_vllm=True, dtype="bfloat16")
    yield model
    if getattr(model, "vllm_entrypoint", None) is not None:
        model.vllm_entrypoint.llm_engine.engine_core.shutdown()
    VLLM._cleanup_distributed()
    del model
    gc.collect()
    th.cuda.empty_cache()


def _gather_hf(model):
    """Run one HF trace and return ``{name: [seq, hidden] tensor}`` for each accessor."""
    saves = {}
    with model.trace(PROMPT):
        for name, getter in ACCESSORS:
            saves[name] = getter(model).save()
    out = {}
    for name, val in saves.items():
        # HF: [1, seq, hidden] → [seq, hidden]
        assert val.dim() == 3 and val.shape[0] == 1, (
            f"HF {name}: unexpected shape {tuple(val.shape)}"
        )
        out[name] = val.squeeze(0).detach().float().cpu()
    return out


def _gather_vllm(model):
    """Run one vLLM trace per accessor and return ``{name: [seq, hidden] tensor}``.

    Per-accessor traces avoid an apparent silent-drop issue we hit batching
    many saves into a single vLLM trace (the dict was missing some keys
    afterwards). Each trace is fast (~50ms after warmup) so total cost is fine.
    """
    out = {}
    for name, getter in ACCESSORS:
        with model.trace(PROMPT):
            saved = getter(model).save()
        assert isinstance(saved, th.Tensor), (
            f"vLLM {name}: save did not produce a tensor (got {type(saved).__name__})"
        )
        assert saved.dim() == 2, f"vLLM {name}: unexpected shape {tuple(saved.shape)}"
        out[name] = saved.detach().float().cpu()
    return out


def test_accessor_equivalence(hf_model, vllm_model):
    """HF and vLLM yield numerically equivalent values for each layer accessor.

    Collects all per-accessor failures and reports them together so a single
    drift doesn't mask others.
    """
    hf_vals = _gather_hf(hf_model)
    vllm_vals = _gather_vllm(vllm_model)

    failures = []
    for name, _ in ACCESSORS:
        hf_v = hf_vals[name]
        vllm_v = vllm_vals[name]
        if hf_v.shape != vllm_v.shape:
            failures.append(f"{name}: shape mismatch hf={hf_v.shape} vllm={vllm_v.shape}")
            continue
        diff = (hf_v - vllm_v).abs()
        if not th.allclose(hf_v, vllm_v, atol=BF16_ATOL, rtol=BF16_RTOL):
            failures.append(
                f"{name}: max abs diff = {diff.max().item():.4f}, "
                f"mean abs diff = {diff.mean().item():.4f}"
            )
    assert not failures, "Equivalence failures:\n  " + "\n  ".join(failures)


def test_setter_layers_output_runs_on_hf(hf_model):
    """HF ``layers_output[i] = x`` smoke test — write zeros at layer 5, read final layer.

    Counterpart for vLLM is omitted here: the existing ``test_vllm.py::test_vllm_steer``
    exercises the same ``layers_output``-write path on vLLM via ``model.steer()``.
    A direct read+write+save in a single vLLM trace exhibits a silent-failure pattern
    (deferred exception not surfaced; saved variable unbound) shared with
    ``test_vllm_logits`` — out of scope for this refactor.
    """
    layer = 5
    with hf_model.trace(PROMPT):
        hidden = hf_model.layers_output[layer]
        hf_model.layers_output[layer] = hidden * 0
        final = hf_model.layers_output[hf_model.num_layers - 1].save()

    assert isinstance(final, th.Tensor), f"got {type(final).__name__}"
    assert final.dim() == 3 and final.shape[0] == 1, (
        f"unexpected HF shape {tuple(final.shape)}"
    )
