"""Prompts with targets, on GPT-2 with left padding and a pad token set through tokenizer_kwargs."""

import pytest
import torch

from nnter import StandardizedTransformer
from nnter.prompt_utils import Prompt, get_first_tokens, next_token_probs_unsqueeze, run_prompts

GPT2 = "hf-internal-testing/tiny-random-gpt2"


@pytest.fixture(scope="module")
def model():
    return StandardizedTransformer(GPT2, dispatch=True, tokenizer_kwargs={"padding_side": "left", "pad_token": "<|endoftext|>"})


def test_tokenizer_kwargs_applied(model):
    assert model.tokenizer.padding_side == "left" and model.tokenizer.pad_token == "<|endoftext|>"


def test_get_first_tokens(model):
    single = get_first_tokens("hello", model)
    several = get_first_tokens(["hello", "world"], model)
    assert isinstance(single, list) and 1 <= len(single) <= 2
    assert len(several) >= 2 and len(set(several)) == len(several)
    tok = model.add_prefix_false_tokenizer  # "word" and " word" are different tokens with it
    assert tok("hello", add_special_tokens=False).input_ids[0] != tok(" hello", add_special_tokens=False).input_ids[0]


def test_hacky_implementation_agrees_where_it_applies(model):
    plain = get_first_tokens("hello", model)
    hacky = get_first_tokens("hello", model.add_prefix_false_tokenizer, use_hacky_implementation=True)
    assert hacky and hacky[0] in plain


def test_prompt_from_strings_and_collisions(model):
    prompt = Prompt.from_strings("The quick brown fox", "jumps", model)
    assert prompt.prompt == "The quick brown fox" and set(prompt.target_tokens) == {"target"}
    prompt = Prompt.from_strings("Hello world", {"greeting": "hello", "object": "world"}, model)
    assert set(prompt.target_tokens) == {"greeting", "object"}
    assert prompt.has_no_collisions()
    prompt.target_tokens["object"] = list(prompt.target_tokens["greeting"])
    assert not prompt.has_no_collisions() and prompt.has_no_collisions(ignore_targets="object")


def test_get_target_probs_reduces_per_target(model):
    prompt = Prompt.from_strings("The quick brown fox", {"target": "jumps", "animal": "fox"}, model)
    probs = torch.rand(1, 3, model.vocab_size)
    out = prompt.get_target_probs(probs)
    assert set(out) == {"target", "animal"} and out["target"].shape == (1, 3)
    torch.testing.assert_close(out["target"][0, 1], probs[0, 1, prompt.target_tokens["target"]].sum())
    assert prompt.get_target_probs(probs, layer=2)["animal"].shape == (1,)


def test_prompt_run_and_next_token_probs(model):
    prompt = Prompt.from_strings("The quick brown fox", "jumps", model)
    probs = next_token_probs_unsqueeze(model, prompt.prompt)
    assert probs.shape == (1, 1, model.vocab_size) and torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-4)
    out = prompt.run(model, next_token_probs_unsqueeze)
    assert out["target"].shape == (1, 1) and 0 <= out["target"].item() <= 1


def test_run_prompts_batched_equals_one_by_one(model):
    prompts = [Prompt.from_strings(text, {"a": "hello", "b": "world"}, model) for text in ("Hello", "The cat sat on the", "Once upon a time there")]
    batched = run_prompts(model, prompts, batch_size=2)
    single = run_prompts(model, prompts, batch_size=1)
    assert set(batched) == {"a", "b"} and batched["a"].shape == (3, 1)
    torch.testing.assert_close(batched["a"], single["a"], rtol=1e-3, atol=1e-4)  # left padding: a padded row may shift slightly


def test_run_prompts_refuses_mismatched_targets(model):
    prompts = [Prompt.from_strings("a", {"x": "hello"}, model), Prompt.from_strings("b", {"y": "hello"}, model)]
    with pytest.raises(ValueError, match="same targets"):
        run_prompts(model, prompts)
    assert run_prompts(model, []) == {}
