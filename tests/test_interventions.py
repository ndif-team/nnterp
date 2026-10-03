"""The target prompts of the patchscopes, and the lenses' checks on their inputs. The lenses themselves run per family in `families/suite.py`."""

import pytest
import torch

from nnterp import StandardizedTransformer, TargetPrompt, TargetPromptBatch, it_repeat_prompt, logit_lens, repeat_prompt
from nnterp.interventions import patchscope_lens

LLAMA = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="module")
def model():
    return StandardizedTransformer(LLAMA, dispatch=True)


def test_repeat_prompt():
    default = repeat_prompt()
    assert default == TargetPrompt("king king\n1135 1135\nhello hello\n?", -1)
    custom = repeat_prompt(words=["cat", "dog"], rel="->", sep=" | ", placeholder="X", index_to_patch=-2)
    assert custom == TargetPrompt("cat->cat | dog->dog | X", -2)


def test_it_repeat_prompt_continues_the_assistant_turn(model):
    prompt = it_repeat_prompt(model.tokenizer)
    assert prompt.index_to_patch == -1
    assert prompt.prompt.rstrip().endswith(repeat_prompt().prompt.split("\n")[-1])
    assert "guess the next word" in prompt.prompt
    bare = it_repeat_prompt(model.tokenizer, words=["test", "word"], complete_prompt=False, add_user_instr=False, use_system_prompt=False)
    assert "test test\nword word\n?" in bare.prompt and "<<SYS>>" not in bare.prompt


def test_target_prompt_batch():
    batch = TargetPromptBatch.from_target_prompts([TargetPrompt("Hello", -1), TargetPrompt("World", 0)])
    assert len(batch) == 2 and batch.prompts == ["Hello", "World"]
    assert torch.equal(batch.index_to_patch, torch.tensor([-1, 0]))
    assert [p.prompt for p in batch] == ["Hello", "World"] and batch[1].index_to_patch == 0
    repeated = TargetPromptBatch.from_target_prompt(TargetPrompt("Test", -1), 3)
    assert repeated.prompts == ["Test"] * 3 and torch.equal(repeated.index_to_patch, torch.tensor([-1, -1, -1]))
    auto = TargetPromptBatch.auto(TargetPrompt("Test", -1), 3)
    assert auto.prompts == repeated.prompts and torch.equal(auto.index_to_patch, repeated.index_to_patch)
    assert TargetPromptBatch.auto(batch, 3) is batch
    single = TargetPromptBatch.from_prompts("Hello", -1)
    assert single.prompts == ["Hello"] and torch.equal(single.index_to_patch, torch.tensor([-1]))
    assert torch.equal(TargetPromptBatch.from_prompts(["a", "b"], [-1, 0]).index_to_patch, torch.tensor([-1, 0]))
    with pytest.raises(ValueError, match="index_to_patch"):
        TargetPromptBatch.from_prompts(["a"], (-1,))


def test_logit_lens_needs_left_padding():
    right = StandardizedTransformer(LLAMA, dispatch=True, tokenizer_kwargs={"padding_side": "right"})
    with pytest.raises(ValueError, match="left padding"):
        logit_lens(right, ["Hello", "Hello world"])


def test_patchscope_lens_takes_sources_or_latents(model):
    latents = torch.zeros(1, 1, model.hidden_size)
    with pytest.raises(ValueError, match="both"):
        patchscope_lens(model, "Hello", latents=latents, layers=[0])
    with pytest.raises(ValueError, match="Either"):
        patchscope_lens(model)
    with pytest.raises(ValueError, match="does not match"):
        patchscope_lens(model, ["a", "b"], [TargetPrompt("x", -1)] * 3)
