"""Constructor behaviour of StandardizedTransformer: the ``attn_implementation``
keyword next to ``enable_attention_probs``."""

import pytest
import torch as th

from nnterp import StandardizedTransformer

MODEL = "yujiepan/llama-3.3-tiny-random"
KWARGS = dict(dtype=th.float32, device_map="cpu")
PROMPT = "The quick brown fox"


def assert_attention_probabilities(model):
    assert model._module.config._attn_implementation == "eager"
    assert model.attn_probs_available
    with model.trace(PROMPT):
        probs = model.attention_probabilities[0].save()
    seq_len = len(model.tokenizer.encode(PROMPT))
    assert probs.shape == (1, model.num_heads, seq_len, seq_len)
    row_sums = probs.sum(-1)
    assert th.allclose(row_sums, th.ones_like(row_sums), atol=1e-5)


def test_attention_probs_set_eager():
    model = StandardizedTransformer(MODEL, enable_attention_probs=True, **KWARGS)
    assert_attention_probabilities(model)


def test_attention_probs_accept_explicit_eager():
    """An explicit attn_implementation="eager" is the same request as the one
    enable_attention_probs=True makes, and reaches the model once."""
    model = StandardizedTransformer(
        MODEL, enable_attention_probs=True, attn_implementation="eager", **KWARGS
    )
    assert_attention_probabilities(model)


def test_attention_probs_refuse_non_eager():
    with pytest.raises(
        ValueError,
        match="Cannot use attn_implementation='sdpa' with enable_attention_probs=True",
    ):
        StandardizedTransformer(
            MODEL, enable_attention_probs=True, attn_implementation="sdpa", **KWARGS
        )


def test_attn_implementation_without_attention_probs():
    """Without enable_attention_probs the caller's attn_implementation is the one
    the model loads with."""
    model = StandardizedTransformer(MODEL, attn_implementation="eager", **KWARGS)
    assert model._module.config._attn_implementation == "eager"
    assert not model.attn_probs_available
    model = StandardizedTransformer(MODEL, attn_implementation="sdpa", **KWARGS)
    assert model._module.config._attn_implementation == "sdpa"
