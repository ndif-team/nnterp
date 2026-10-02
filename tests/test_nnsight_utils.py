"""Activation collection over prompts, on GPT-2 with left padding."""

import pytest
import torch

from nnter import StandardizedTransformer
from nnter.nnsight_utils import (
    collect_last_token_activations_session, collect_token_activations_batched, compute_next_token_probs,
    get_token_activations,
)

GPT2 = "hf-internal-testing/tiny-random-gpt2"
PROMPTS = ["Hello", "The cat sat on the", "Once upon a time there"]


@pytest.fixture(scope="module")
def model():
    return StandardizedTransformer(GPT2, dispatch=True, tokenizer_kwargs={"padding_side": "left", "pad_token": "<|endoftext|>"})


def test_get_token_activations_is_the_last_token_of_layer_output(model):
    acts = get_token_activations(model, PROMPTS[0])
    assert acts.shape == (model.num_layers, 1, model.hidden_size) and acts.device.type == "cpu"
    with model.trace(PROMPTS[0]):
        last = model.layers[2].layer_output[:, -1].save()
    torch.testing.assert_close(acts[2], last.cpu())
    subset = get_token_activations(model, PROMPTS[0], layers=[0, 3])
    torch.testing.assert_close(subset[1], acts[3])


def test_inside_a_callers_tracer(model):
    with model.trace(PROMPTS[0]) as tracer:
        acts = get_token_activations(model, tracer=tracer, layers=[1]).save()
    assert acts.shape == (1, 1, model.hidden_size)


def test_index_needs_the_matching_padding_side(model):
    with pytest.raises(ValueError, match="right padding"):
        get_token_activations(model, PROMPTS[0], idx=1)


def test_batched_collection_matches_one_at_a_time(model):
    batched = collect_token_activations_batched(model, PROMPTS, batch_size=2)
    one = torch.cat([get_token_activations(model, [p]) for p in PROMPTS], dim=1)
    assert batched.shape == (model.num_layers, 3, model.hidden_size)
    torch.testing.assert_close(batched, one, rtol=1e-3, atol=1e-4)


def test_session_collection_matches(model):
    session = collect_last_token_activations_session(model, PROMPTS, batch_size=2)
    batched = collect_token_activations_batched(model, PROMPTS, batch_size=2)
    torch.testing.assert_close(session, batched)


def test_compute_next_token_probs(model):
    probs = compute_next_token_probs(model, PROMPTS)
    assert probs.shape == (3, model.vocab_size) and probs.device.type == "cpu"
    torch.testing.assert_close(probs.sum(-1), torch.ones(3), atol=1e-4, rtol=0)
