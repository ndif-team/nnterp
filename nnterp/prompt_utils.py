from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import torch as th
from tqdm.auto import tqdm
from .nnsight_utils import LanguageModel, compute_next_token_probs
from .standardized_transformer import StandardizedTransformer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .logging import logger


class TokenizationError(Exception):
    pass


def get_first_tokens(
    words: str | list[str],
    llm_or_tokenizer: LanguageModel | StandardizedTransformer | PreTrainedTokenizerBase,
    use_hacky_implementation=False,
) -> list[int]:
    """
    Get the all the first tokens of a "word" and " word" for all words.

    Args:
        words: A string or a list of strings to get the first token of.
        llm_or_tokenizer: The tokenizer to use. If a LanguageModel or StandardizedTransformer is provided,
            the tokenizer will be extracted from it. It is recommended to use StandardizedTransformer. If you want to use your own tokenizer,
            it's recommended to initialize it with add_prefix_space=False or to use the hacky implementation.
        use_hacky_implementation: If True, use a hacky implementation to get the first token of a word by tokenizing "🍐word" and extracting the first token of word.
            While hacky, it is still guaranteed to work correctly or raise an error.

    Returns:
        A list of tokens.
    """
    if isinstance(words, str):
        words = [words]
    if isinstance(llm_or_tokenizer, StandardizedTransformer):
        try:
            tokenizer = llm_or_tokenizer.add_prefix_false_tokenizer
        except Exception as e:
            logger.warning(
                f"Error getting model.add_prefix_false_tokenizer, using model.tokenizer instead:\n{e}"
            )
            tokenizer = llm_or_tokenizer.tokenizer
    elif isinstance(llm_or_tokenizer, LanguageModel):
        tokenizer = llm_or_tokenizer.tokenizer
    else:
        tokenizer = llm_or_tokenizer
    final_tokens = []
    for word in words:
        # If you get the value error even with add_prefix_space=False,
        # you can use the following hacky code to get the token without the prefix
        if use_hacky_implementation:
            hacky_token = tokenizer("🍐", add_special_tokens=False).input_ids
            length = len(hacky_token)
            tokens = tokenizer("🍐" + word, add_special_tokens=False).input_ids
            if tokens[:length] != hacky_token:
                raise TokenizationError(
                    "I didn't expect this to happen, please check this code"
                )
            if len(tokens) > length:
                final_tokens.append(tokens[length])
        else:
            # Assuming the tokenizer was initialized with add_prefix_space=False
            token = tokenizer(word, add_special_tokens=False).input_ids[0]
            token_with_start_of_word = tokenizer(
                " " + word, add_special_tokens=False
            ).input_ids[0]
            if token == token_with_start_of_word:
                try:
                    tokens = get_first_tokens(
                        words, tokenizer, use_hacky_implementation=True
                    )
                    logger.warning(
                        "Seems like you use a tokenizer that wasn't initialized with add_prefix_space=False."
                        "add_prefix_space=False is needed to ensure proper tokenization of words without the space."
                        "Used hacky implementation instead."
                    )
                except TokenizationError:
                    raise TokenizationError(
                        "Seems like you use a tokenizer that wasn't initialized with add_prefix_space=False."
                        "add_prefix_space=False is needed to ensure proper tokenization of words without the space."
                    )
            final_tokens.append(token)
            space_token = tokenizer(" ", add_special_tokens=False).input_ids
            if space_token:
                space_token = space_token[0]
            else:
                space_token = None
            if token_with_start_of_word != space_token:
                final_tokens.append(token_with_start_of_word)
    return list(dict.fromkeys(final_tokens))


@dataclass
class Prompt:
    """
    Generic class to represent a prompt with target tokens to track during next token prediction.

    Args:
        prompt: The prompt to use
        target_tokens: A dictionary of target tokens for each target
        target_strings: A dictionary of target strings for each target
    """

    prompt: str
    target_tokens: dict[str, list[int]]
    target_strings: dict[str, str | list[str]] | None = None

    @classmethod
    def from_strings(
        cls,
        prompt: str,
        target_strings: dict[str, str | list[str]] | list[str] | str,
        tokenizer,
    ):
        if isinstance(target_strings, str) or isinstance(target_strings, list):
            target_strings = {"target": target_strings}
        target_tokens = {
            target: get_first_tokens(words, tokenizer)
            for target, words in target_strings.items()
        }
        return cls(
            target_tokens=target_tokens,
            target_strings=target_strings,
            prompt=prompt,
        )

    def has_no_collisions(self, ignore_targets: None | str | list[str] = None):
        if isinstance(ignore_targets, str):
            ignore_targets = [ignore_targets]
        if ignore_targets is None:
            ignore_targets = []
        # Collect all tokens for non-ignored targets
        all_tokens = []
        for target, tokens in self.target_tokens.items():
            if target in ignore_targets:
                continue
            all_tokens.extend(tokens)
        return len(all_tokens) == len(set(all_tokens))

    def get_target_probs(self, probs, layer=None):
        target_probs = {
            target: probs[:, :, tokens].sum(dim=2).cpu()
            for target, tokens in self.target_tokens.items()
        }
        if layer is not None:
            target_probs = {
                target: probs_[:, layer] for target, probs_ in target_probs.items()
            }
        return target_probs

    @th.no_grad
    def run(self, nn_model, get_probs: Callable):
        """
        Run the prompt through the model and return the probabilities of the next token for both the target tokens.
        """
        probs = get_probs(nn_model, self.prompt)
        return self.get_target_probs(probs)


def next_token_probs_unsqueeze(
    nn_model: LanguageModel, prompt: str | list[str], remote=False, **_kwargs
) -> th.Tensor:
    probs = compute_next_token_probs(nn_model, prompt, remote=remote)
    return probs.unsqueeze(1)  # Add a fake layer dimension


@th.no_grad
def run_prompts(
    nn_model: LanguageModel,
    prompts: list[Prompt],
    batch_size: int = 32,
    get_probs_func: Callable | None = None,
    func_kwargs: dict | None = None,
    remote: bool = False,
    tqdm=tqdm,
) -> dict[str, th.Tensor]:
    """
    Run a list of prompts through the model and return the probabilities of the next token for the target tokens.

    Args:
        nn_model: The NNSight model
        prompts: A list of prompts. All prompts must have the same target keys
        batch_size: The batch size to use
        get_probs: The function to get the probabilities of the next token, default to next token prediction
        method_kwargs: The kwargs to pass to the get_probs function
        tqdm: The tqdm function to use, default to tqdm.auto.tqdm. Use None to disable tqdm

    Returns:
        A dictionary of target names and the probabilities of the next token for the target tokens.
    """
    if len(prompts) == 0:
        return {}
    keys = set(prompts[0].target_tokens.keys())
    for prompt in prompts:
        if set(prompt.target_tokens.keys()) != keys:
            raise ValueError(
                f"All prompts must have the same target keys. Got {keys} and {set(prompt.target_tokens.keys())}"
            )
    str_prompts = [prompt.prompt for prompt in prompts]
    probs = []
    if get_probs_func is None:
        get_probs_func = next_token_probs_unsqueeze
    if func_kwargs is None:
        func_kwargs = {}

    for i in tqdm(
        range(0, len(str_prompts), batch_size),
        desc="Running prompts",
    ):
        batch = str_prompts[i : i + batch_size]
        probs.append(get_probs_func(nn_model, batch, remote=remote, **func_kwargs))
    probs = th.cat(probs)
    target_probs = {target: [] for target in prompts[0].target_tokens.keys()}
    for i, prompt in enumerate(prompts):
        for target, tokens in prompt.target_tokens.items():
            target_probs[target].append(probs[i, :, tokens].sum(dim=1))
    target_probs = {
        target: th.stack(probs).cpu() for target, probs in target_probs.items()
    }
    return target_probs
