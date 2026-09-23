"""Hybrid models mixing linear attention (Gated DeltaNet) and softmax attention
blocks (Qwen3-Next, Qwen3.5, Qwen3.6), and the per-layer tuple detection of the
accessors they rely on."""

from types import SimpleNamespace

import pytest
import torch as th
import torch.nn as nn
from nnsight import NNsight

from nnterp import StandardizedTransformer
from nnterp.rename_utils import (
    Address,
    FirstIfTuple,
    IOType,
    LayerAccessor,
    RenamingError,
    _check_attention_layers,
)

# Layers 0-2 are Gated DeltaNet (linear_attn), layer 3 is softmax attention (self_attn).
HYBRID_MODEL = "tiny-random/qwen3.5-moe"
HYBRID_KWARGS = dict(dtype=th.float32, device_map="cpu")
LINEAR_ATTENTION_ERROR = (
    r"layer 0 is a linear-attention layer \(Qwen3_5MoeGatedDeltaNet\); "
    "attentions/attention_probabilities are only defined on softmax-attention layers"
)
PROMPT = "The quick brown fox jumps over the lazy dog"


def load_hybrid(**kwargs):
    return StandardizedTransformer(HYBRID_MODEL, **HYBRID_KWARGS, **kwargs)


def test_hybrid_layer_kinds():
    """Every block exposes exactly one of self_attn / linear_attn, and the index
    lists come from that structure."""
    model = load_hybrid(attn_implementation="eager")
    assert model.attention_layers == [3]
    assert model.linear_attention_layers == [0, 1, 2]
    for layer in model.linear_attention_layers:
        assert hasattr(model.layers[layer], "linear_attn")
        assert not hasattr(model.layers[layer], "self_attn")
        assert (
            type(model.layers[layer].linear_attn._module).__name__
            == "Qwen3_5MoeGatedDeltaNet"
        )
    assert hasattr(model.layers[3], "self_attn")
    assert not hasattr(model.layers[3], "linear_attn")


def test_hybrid_attention_accessors_raise_on_linear_layers():
    model = load_hybrid(enable_attention_probs=True)
    with pytest.raises(RenamingError, match=LINEAR_ATTENTION_ERROR):
        model.attentions[0]
    with pytest.raises(RenamingError, match=LINEAR_ATTENTION_ERROR):
        model.attentions_input[0]
    with pytest.raises(RenamingError, match=LINEAR_ATTENTION_ERROR):
        model.attentions_output[0]
    with pytest.raises(RenamingError, match=LINEAR_ATTENTION_ERROR):
        model.attention_probabilities[0]
    with pytest.raises(RenamingError, match=LINEAR_ATTENTION_ERROR):
        model.attention_probabilities[0] = th.zeros(1)
    # The mixer itself stays reachable, and the other accessors work on every layer
    assert model.attentions[3] is model.layers[3].self_attn
    assert model.mlps[0] is model.layers[0].mlp


def test_hybrid_accessors_in_any_order():
    """attentions_output[3] and layers_output[0] are readable whichever is read
    first, on a fresh model each time (per-layer tuple detection)."""
    with th.no_grad():
        model = load_hybrid(attn_implementation="eager")
        with model.trace(PROMPT):
            attn_3_first = model.attentions_output[3].save()
        with model.trace(PROMPT):
            layer_0_second = model.layers_output[0].save()

        model = load_hybrid(attn_implementation="eager")
        with model.trace(PROMPT):
            layer_0_first = model.layers_output[0].save()
            attn_3_second = model.attentions_output[3].save()

    seq_len = len(model.tokenizer.encode(PROMPT))
    assert layer_0_first.shape == attn_3_first.shape == (1, seq_len, model.hidden_size)
    assert th.allclose(attn_3_first, attn_3_second)
    assert th.allclose(layer_0_first, layer_0_second)


def test_hybrid_layer_outputs_read_3_then_0_and_0_then_3():
    """Tuple-ness is detected per access: reading layer 3 before layer 0 (across
    traces) and layer 0 before layer 3 both work, without renaming checks
    pre-recording every layer."""
    with th.no_grad():
        model = load_hybrid(check_renaming=False, attn_implementation="eager")
        assert model.layers_output.returns_tuple(3) is None
        with model.trace(PROMPT):
            out_3 = model.layers_output[3].save()
        assert model.layers_output.returns_tuple(3) is not None
        assert model.layers_output.returns_tuple(0) is None
        with model.trace(PROMPT):
            out_0 = model.layers_output[0].save()
            out_3_again = model.layers_output[3].save()
    assert model.layers_output.returns_tuple(0) is not None
    assert th.allclose(out_3, out_3_again)
    assert (
        out_0.shape
        == out_3.shape
        == (1, len(model.tokenizer.encode(PROMPT)), model.hidden_size)
    )


def test_hybrid_attention_probabilities():
    """Construction validates the attention probabilities on the first
    softmax-attention layer; on layer 3 they are (batch, heads, seq, seq),
    rows sum to 1, and editing them changes the logits."""
    with th.no_grad():
        model = load_hybrid(enable_attention_probs=True)
        assert model.attn_probs_available
        seq_len = len(model.tokenizer.encode(PROMPT))
        with model.trace(PROMPT):
            probs = model.attention_probabilities[3].save()
            logits = model.logits.save()
        assert (
            probs.shape
            == (1, model.num_heads, seq_len, seq_len)
            == (1, 8, seq_len, seq_len)
        )
        rows = probs.sum(dim=-1)
        assert th.allclose(rows, th.ones_like(rows), atol=1e-5)
        assert probs.triu(diagonal=1).abs().max() == 0

        with model.trace(PROMPT):
            mask = th.ones_like(model.attention_probabilities[3]).tril()
            model.attention_probabilities[3] = mask / mask.sum(-1, keepdim=True)
            edited_logits = model.logits.save()
        assert not th.allclose(logits, edited_logits)


def test_hybrid_skip_layers():
    """skip_layer works on a linear-attention and on a softmax-attention layer
    after the renaming checks recorded every layer's output structure."""
    with th.no_grad():
        model = load_hybrid(attn_implementation="eager")
        with model.trace(PROMPT):
            baseline = model.logits.save()
        with model.trace(PROMPT):
            model.skip_layer(0)
            skipped_linear = model.logits.save()
        with model.trace(PROMPT):
            model.skip_layer(3)
            skipped_softmax = model.logits.save()
    assert not th.allclose(baseline, skipped_linear)
    assert not th.allclose(baseline, skipped_softmax)


def test_layer_types_cross_check():
    """config.layer_types must agree with the block structure."""
    model = load_hybrid(check_renaming=False, attn_implementation="eager")
    _check_attention_layers(model, HYBRID_MODEL)
    model.config.layer_types = ["full_attention"] * model.num_layers
    with pytest.raises(RenamingError, match="layer_types"):
        _check_attention_layers(model, HYBRID_MODEL)


class _Block(nn.Module):
    def __init__(self, returns_tuple: bool):
        super().__init__()
        self.returns_tuple = returns_tuple

    def forward(self, x):
        y = x * 2
        return (y, None) if self.returns_tuple else y


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Block(True), _Block(False)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            if isinstance(x, tuple):
                x = x[0]
        return x


def test_layer_accessor_mixed_tuple_layers():
    """FirstIfTuple decides per layer, as layers_output's row does: a
    tuple-returning layer 0 and a tensor-returning layer 1 are read and written in
    either order."""
    net = NNsight(_Net())
    accessor = LayerAccessor(
        SimpleNamespace(layers=net.layers, linear_attention_layers=[]),
        Address("", IOType.OUTPUT, select=FirstIfTuple()),
    )
    x = th.ones(2, 3)

    with net.trace(x):
        out_1 = accessor[1].save()
    with net.trace(x):
        out_0 = accessor[0].save()
        out_1_again = accessor[1].save()
    assert accessor.returns_tuple(0) is True
    assert accessor.returns_tuple(1) is False
    assert th.equal(out_0, x * 2)
    assert th.equal(out_1, x * 4)
    assert th.equal(out_1, out_1_again)

    with net.trace(x):
        accessor[0] = th.zeros(2, 3)
        zeroed_1 = accessor[1].save()
        accessor[1] = th.ones(2, 3)
        result = net.output.save()
    assert th.equal(zeroed_1, th.zeros(2, 3))
    assert th.equal(result, th.ones(2, 3))
