"""Prompts with target tokens to track: nnterp's ``prompt_utils``, on the standard values.

`get_first_tokens` turns words into the token ids a model would predict for
them, `Prompt` pairs a prompt with named sets of those, and `run_prompts` runs
many prompts and returns each target's probability mass per prompt.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable

import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .nnsight_utils import compute_next_token_probs
from .standardized import StandardizedTransformer


class TokenizationError(Exception):
    """A word could not be tokenized as a standalone first token."""


def get_first_tokens(
    words: str | list[str],
    model_or_tokenizer: StandardizedTransformer | PreTrainedTokenizerBase,
    use_hacky_implementation: bool = False,
) -> list[int]:
    """The first token of ``word`` and of ``" word"`` for each word, deduplicated.

    A model's `StandardizedTransformer.add_prefix_false_tokenizer` is used when
    a model is given, so that ``"word"`` and ``" word"`` tokenize differently;
    with a tokenizer that adds a prefix space the two collide, and the hacky
    implementation (tokenize ``"🍐word"`` and drop the pear) is used instead,
    or forced with ``use_hacky_implementation``.
    """
    words = [words] if isinstance(words, str) else words
    if isinstance(model_or_tokenizer, StandardizedTransformer):
        tokenizer = model_or_tokenizer.add_prefix_false_tokenizer
    else:
        tokenizer = model_or_tokenizer
    tokens: list[int] = []
    for word in words:
        if use_hacky_implementation:
            pear = tokenizer("🍐", add_special_tokens=False).input_ids
            both = tokenizer("🍐" + word, add_special_tokens=False).input_ids
            if both[: len(pear)] != pear:
                raise TokenizationError(f"the pear trick did not tokenize {word!r} as expected")
            if len(both) > len(pear):
                tokens.append(both[len(pear)])
            continue
        token = tokenizer(word, add_special_tokens=False).input_ids[0]
        with_space = tokenizer(" " + word, add_special_tokens=False).input_ids[0]
        if token == with_space:
            try:
                hacky = get_first_tokens(words, tokenizer, use_hacky_implementation=True)
            except TokenizationError:
                raise TokenizationError(
                    "the tokenizer adds a prefix space, so 'word' and ' word' tokenize alike; "
                    "use one initialized with add_prefix_space=False"
                ) from None
            warnings.warn("the tokenizer adds a prefix space; used the hacky implementation instead", stacklevel=2)
            return hacky
        tokens.append(token)
        space = tokenizer(" ", add_special_tokens=False).input_ids
        if with_space != (space[0] if space else None):
            tokens.append(with_space)
    return list(dict.fromkeys(tokens))


@dataclass
class Prompt:
    """A prompt with named sets of target tokens to track in the next-token distribution.

    Attributes:
        prompt: The text.
        target_tokens: Target name -> the token ids that count for it.
        target_strings: What those came from, when built with `from_strings`.
    """

    prompt: str
    target_tokens: dict[str, list[int]]
    target_strings: dict[str, str | list[str]] | None = None

    @classmethod
    def from_strings(
        cls,
        prompt: str,
        target_strings: dict[str, str | list[str]] | list[str] | str,
        model_or_tokenizer: StandardizedTransformer | PreTrainedTokenizerBase,
    ) -> "Prompt":
        """Build from words: a string or list is one target named ``"target"``, a dict names them."""
        if isinstance(target_strings, (str, list)):
            target_strings = {"target": target_strings}
        target_tokens = {name: get_first_tokens(words, model_or_tokenizer) for name, words in target_strings.items()}
        return cls(prompt=prompt, target_tokens=target_tokens, target_strings=target_strings)

    def has_no_collisions(self, ignore_targets: str | list[str] | None = None) -> bool:
        """Whether no token id belongs to two targets (``ignore_targets`` left out)."""
        ignored = {ignore_targets} if isinstance(ignore_targets, str) else set(ignore_targets or [])
        tokens = [t for name, ts in self.target_tokens.items() if name not in ignored for t in ts]
        return len(tokens) == len(set(tokens))

    def get_target_probs(self, probs: torch.Tensor, layer: int | None = None) -> dict[str, torch.Tensor]:
        """Each target's probability mass from ``probs`` of shape ``[batch, layers, vocab]``; one layer if given."""
        target_probs = {name: probs[:, :, tokens].sum(dim=2).cpu() for name, tokens in self.target_tokens.items()}
        if layer is not None:
            target_probs = {name: p[:, layer] for name, p in target_probs.items()}
        return target_probs

    @torch.no_grad()
    def run(self, model: StandardizedTransformer, get_probs: Callable) -> dict[str, torch.Tensor]:
        """``get_probs(model, prompt)`` (``[batch, layers, vocab]``) reduced to each target's mass."""
        return self.get_target_probs(get_probs(model, self.prompt))


def next_token_probs_unsqueeze(model: StandardizedTransformer, prompt: str | list[str], remote: bool = False, **_) -> torch.Tensor:
    """`compute_next_token_probs` with a layer axis of one, ``[batch, 1, vocab]``: the default ``get_probs``."""
    return compute_next_token_probs(model, prompt, remote=remote).unsqueeze(1)


@torch.no_grad()
def run_prompts(
    model: StandardizedTransformer,
    prompts: list[Prompt],
    batch_size: int = 32,
    get_probs_func: Callable | None = None,
    func_kwargs: dict | None = None,
    remote: bool = False,
    tqdm=None,
) -> dict[str, torch.Tensor]:
    """Run prompts in batches; each target's probability mass per prompt, ``[num_prompts, layers]``.

    All prompts must name the same targets. ``get_probs_func(model, batch,
    remote=..., **func_kwargs)`` returns ``[batch, layers, vocab]``; the
    default is the next-token distribution with one layer. ``tqdm`` is a
    progress-bar factory to wrap the batch loop with, or ``None``.
    """
    if not prompts:
        return {}
    targets = set(prompts[0].target_tokens)
    for prompt in prompts:
        if set(prompt.target_tokens) != targets:
            raise ValueError(f"all prompts must name the same targets; got {targets} and {set(prompt.target_tokens)}")
    get_probs_func = get_probs_func or next_token_probs_unsqueeze
    texts = [prompt.prompt for prompt in prompts]
    steps = range(0, len(texts), batch_size)
    probs = torch.cat([
        get_probs_func(model, texts[i : i + batch_size], remote=remote, **(func_kwargs or {}))
        for i in (tqdm(steps) if tqdm is not None else steps)
    ])
    return {
        name: torch.stack([probs[i, :, prompt.target_tokens[name]].sum(dim=1) for i, prompt in enumerate(prompts)]).cpu()
        for name in prompts[0].target_tokens
    }
