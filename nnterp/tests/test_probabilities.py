import torch as th
from nnterp import StandardizedTransformer
import pytest


def test_print_attn_probabilities_source(model_name):
    """Test printing attention probabilities source"""
    with th.no_grad():
        model = StandardizedTransformer(
            model_name,
            enable_attention_probs=True,
            tokenizer_kwargs=dict(padding_side="right"),
        )
        model.attention_probabilities.print_source()


def test_access_attn_probabilities(model_name):
    """Test accessing attention probabilities"""
    with th.no_grad():
        model = StandardizedTransformer(
            model_name,
            enable_attention_probs=True,
            tokenizer_kwargs=dict(padding_side="right"),
        )
        if not model.attn_probs_available:
            pytest.fail("Attention probabilities are not available for this model")
        prompts = ["Hello, world!", "The quick brown fox jumps"]
        with model.trace(prompts):
            attn_probs = model.attention_probabilities[0].save()
            logits = model.logits.save()
        assert attn_probs.shape == (
            logits.shape[0],
            model.num_heads,
            logits.shape[1],
            logits.shape[1],
        )  # (batch_size, num_heads, seq_len, seq_len)
        summed_probs = attn_probs.sum(dim=-1)
        atol = 1e-2 if attn_probs.dtype == th.bfloat16 else 1e-5
        assert th.allclose(
            summed_probs,
            th.ones_like(summed_probs),
            atol=atol,
        )  # last dimension is the attention of token i to all other tokens


def test_edit_attn_probabilities(model_name):
    """Test editing attention probabilities"""
    with th.no_grad():
        model = StandardizedTransformer(
            model_name,
            enable_attention_probs=True,
            tokenizer_kwargs=dict(padding_side="right"),
        )
        if not model.attn_probs_available:
            pytest.fail("Attention probabilities are not available for this model")
        prompts = ["Hello, world!", "The quick brown fox jumps"]
        with model.trace(prompts):
            # knocking out the first token
            for layer in range(model.num_layers):
                probs = model.attention_probabilities[layer]
                probs[:, :, :, 1] += probs[:, :, :, 0]
                probs[:, :, :, 0] = 0
            corrupted_logits = model.logits.save()

        with model.trace(prompts):
            baseline_logits = model.logits.save()

        assert not th.allclose(corrupted_logits, baseline_logits)
