from __future__ import annotations

from dataclasses import dataclass
import torch as th

from .logging import logger
from .nnsight_utils import (
    get_layer_output,
    get_next_token_probs,
    get_token_activations,
    get_attention,
    get_num_layers,
    LanguageModel,
    project_on_vocab,
)

__all__ = [
    "logit_lens",
    "TargetPrompt",
    "repeat_prompt",
    "TargetPromptBatch",
    "patchscope_lens",
    "patchscope_generate",
    "patch_object_attn_lens",
]


@th.no_grad
def logit_lens(
    nn_model: LanguageModel,
    prompts: list[str] | str,
    remote=False,
    return_inv_logits=False,
) -> th.Tensor | tuple[th.Tensor, th.Tensor]:
    """
    Same as logit_lens but for Llama models directly instead of Transformer_lens models.
    Get the probabilities of the next token for the last token of each prompt at each layer using the logit lens.

    Args:
        nn_model: NNSight Language Model
        prompts: List of prompts or a single prompt
        remote: If True, the function will run on the nndif server. See `nnsight.net/status` to check which models are available.
        return_inv_logits: If True, the function will return the logits applied to the negative of the hidden states.
    Returns:
        A tensor of shape (num_prompts, num_layers, vocab_size) containing the probabilities
        of the next token for each prompt at each layer. Tensor is on the CPU.
    """
    with nn_model.trace(prompts, remote=remote) as tracer:
        hiddens_l = get_token_activations(nn_model, prompts, tracer=tracer)
        probs_l = []
        probs_inv_l = []
        for hiddens in hiddens_l:
            logits = project_on_vocab(nn_model, hiddens)
            probs = logits.softmax(-1).cpu()
            probs_l.append(probs)
            if return_inv_logits:
                inv_normalized = -nn_model.ln_final(hiddens)
                inv_logits = nn_model.lm_head(inv_normalized)
                inv_logits = inv_logits.softmax(-1).cpu()
                probs_inv_l.append(inv_logits)
        probs = th.stack(probs_l).transpose(0, 1).save()
        if return_inv_logits:
            inv_logits = th.stack(probs_inv_l).transpose(0, 1).save()
    if return_inv_logits:
        return probs, inv_logits
    else:
        return probs


@dataclass
class TargetPrompt:
    prompt: str
    index_to_patch: int


def repeat_prompt(
    words=None, rel=" ", sep="\n", placeholder="?", index_to_patch=-1
) -> TargetPrompt:
    """
    Prompt used in the patchscopes paper to predict the next token.
    https://github.com/PAIR-code/interpretability/blob/master/patchscopes/code/next_token_prediction.ipynb

    Args:
        words: The words to repeat. If None, the words will be "king", "1135", "hello".
        rel: The string between the repeated words
        sep: The separator between the words
        placeholder: The placeholder to use for the last word

    Returns:
        A TargetPrompt object containing the prompt to patch and the index of the token to patch.
    """
    if words is None:
        words = [
            "king",
            "1135",
            "hello",
        ]
    prompt = sep.join([w + rel + w for w in words]) + sep + placeholder
    return TargetPrompt(prompt, index_to_patch)


def it_repeat_prompt(
    tokenizer,
    words=None,
    rel=" ",
    sep="\n",
    placeholder="?",
    complete_prompt=True,
    add_user_instr=True,
    use_system_prompt=True,
):
    """
    Same as repeat_prompt but using the chat template of the tokenizer to generate a prompt adapted to instruction-tuned models.

    Args:
        tokenizer: The tokenizer of the model
        words: The words to repeat. If None, the words will be "king", "1135", "hello".
        rel: The string between the repeated words
        sep: The separator between the words
        placeholder: The placeholder to use for the last word
        complete_prompt: If True, the repeat_prompt will be added to the end of the prompt.
        add_user_instr: If True, the prompt will include instructions from the user to the model.

    Returns:
        A TargetPrompt object containing the prompt to patch and the index of the token to patch.
    """
    prompt = repeat_prompt(words, rel, sep, placeholder).prompt
    chat = []
    if add_user_instr:
        if use_system_prompt:
            chat.append({"role": "system", "content": "You are a helpful assistant."})
        chat.extend(
            [
                {
                    "role": "user",
                    "content": "I will provide you with a series of sentences. Your task is to guess the next word in the sentence. You must answer with the next word only.",
                },
                {"role": "assistant", "content": "Ok."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": prompt if complete_prompt else ""},
            ]
        )
    else:
        if use_system_prompt:
            chat.extend(
                [
                    {
                        "role": "system",
                        "content": "The user will provide you with a sentence. Your task is to guess the next word in the sentence. You must answer with the next word only.",
                    },
                    {"role": "user", "content": prompt},
                ]
            )
        else:
            chat.append(
                {
                    "role": "user",
                    "content": f"Guess the next word in the following sentence (answer only the next word): {prompt}",
                }
            )
        chat.append({"role": "assistant", "content": prompt if complete_prompt else ""})
    prompt = tokenizer.apply_chat_template(
        chat, tokenize=False, continue_final_message=True, add_special_tokens=False
    )
    return TargetPrompt(prompt, -1)


@dataclass
class TargetPromptBatch:
    """
    A class to handle multiple target prompts with potentially different indices to patch
    """

    prompts: list[str]
    index_to_patch: th.Tensor

    @classmethod
    def from_target_prompts(cls, prompts_: list[TargetPrompt]):
        prompts = [p.prompt for p in prompts_]
        index_to_patch = th.tensor([p.index_to_patch for p in prompts_])
        return cls(prompts, index_to_patch)

    @classmethod
    def from_target_prompt(cls, prompt: TargetPrompt, batch_size: int):
        prompts = [prompt.prompt] * batch_size
        index_to_patch = th.tensor([prompt.index_to_patch] * batch_size)
        return cls(prompts, index_to_patch)

    @classmethod
    def from_prompts(
        cls, prompts: str | list[str], index_to_patch: int | list[int] | th.Tensor
    ):
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(index_to_patch, int):
            index_to_patch = th.tensor([index_to_patch] * len(prompts))
        elif isinstance(index_to_patch, list):
            index_to_patch = th.tensor(index_to_patch)
        elif not isinstance(index_to_patch, th.Tensor):
            raise ValueError(
                f"index_to_patch must be an int, a list of ints or a tensor, got {type(index_to_patch)}"
            )
        return cls(prompts, index_to_patch)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return TargetPrompt(self.prompts[idx], self.index_to_patch[idx])

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @staticmethod
    def auto(
        target_prompt: str | TargetPrompt | list[TargetPrompt] | TargetPromptBatch,
        batch_size: int,
    ):
        if isinstance(target_prompt, TargetPrompt):
            target_prompt = TargetPromptBatch.from_target_prompt(
                target_prompt, batch_size
            )
        elif isinstance(target_prompt, list):
            target_prompt = TargetPromptBatch.from_target_prompts(target_prompt)
        elif not isinstance(target_prompt, TargetPromptBatch):
            raise ValueError(
                f"patch_prompts must be a str, a TargetPrompt, a list of TargetPrompt or a TargetPromptBatch, got {type(target_prompt)}"
            )
        return target_prompt


@th.no_grad
def patchscope_lens(
    nn_model: LanguageModel,
    source_prompts: list[str] | str | None = None,
    target_patch_prompts: (
        TargetPromptBatch | list[TargetPrompt] | TargetPrompt | None
    ) = None,
    layers: int | list[int] | None = None,
    latents: th.Tensor | None = None,
    remote=False,
):
    """
    Replace the hidden state of the patch_prompt.index_to_patch token in the patch_prompt.prompt with the hidden state of the last token of each prompt at each layer.
    Returns the probabilities of the next token in patch_prompt for each prompt for each layer intervention.
    Args:
        nn_model: The NNSight TL model
        source_prompts: List of prompts or a single prompt to get the hidden states of the last token
        target_patch_prompts: TargetPrompt(s) / TargetPromptBatch containing the prompt to patch and the index of the token to patch
        layers: Layer / list of layers to intervene on. If None, all layers are intervened on.
        latents: Tensor of shape (num_layers, num_sources, hidden_size) If None, the hidden states of the last token of each source prompt at each layer are collected. You cannot provide both source_prompts and latents.
        remote: If True, the function will run on the nndif server. See `nnsight.net/status` to check which models are available.

    Returns:
        A tensor of shape (num_prompts, num_layers, vocab_size) containing the probabilities
        of the next token for each prompt at each layer. Tensor is on the CPU.
    """
    if target_patch_prompts is None:
        target_patch_prompts = repeat_prompt()
    if latents is not None:
        if len(set([len(h) for h in latents])) > 1:
            raise ValueError("Inconsistent number of hiddens")
        num_sources = len(latents[0])
    else:
        if source_prompts is None:
            raise ValueError("Either source_prompts or hiddens must be provided")
        if isinstance(source_prompts, str):
            source_prompts = [source_prompts]
        num_sources = len(source_prompts)
    target_patch_prompts = TargetPromptBatch.auto(target_patch_prompts, num_sources)
    if layers is None:
        layers = list(range(get_num_layers(nn_model)))
    elif isinstance(layers, int):
        layers = [layers]
    if len(target_patch_prompts) != num_sources:
        raise ValueError(
            f"Number of sources ({num_sources}) does not match number of patch prompts ({len(target_patch_prompts)})"
        )
    if latents is None:
        latents = get_token_activations(
            nn_model, source_prompts, layers=layers, remote=remote
        )
    elif source_prompts is not None:
        raise ValueError("You cannot provide both source_prompts and latents")

    probs_l = []

    for idx, layer in enumerate(layers):
        with nn_model.trace(
            target_patch_prompts.prompts,
            remote=remote,
        ):
            device = get_layer_output(nn_model, layer).device
            get_layer_output(nn_model, layer)[
                th.arange(num_sources), target_patch_prompts.index_to_patch
            ] = latents[idx].to(device)
            probs_l.append(get_next_token_probs(nn_model).cpu().save())
    probs = th.cat(probs_l, dim=0)
    return probs.reshape(len(layers), num_sources, -1).transpose(0, 1)


@th.no_grad
def patchscope_generate(
    nn_model: LanguageModel,
    prompts: list[str] | str,
    target_patch_prompt: TargetPrompt,
    max_length: int = 50,
    layers=None,
    remote=False,
    max_batch_size=32,
):
    """
    Replace the hidden state of the patch_prompt.index_to_patch token in the patch_prompt.prompt with the hidden state of the last token of each prompt at each layer.
    Returns the probabilities of the next token in patch_prompt for each prompt for each layer intervention.
    Args:
        nn_model: The NNSight LanguageModel with llama architecture
        prompts: List of prompts or a single prompt to get the hidden states of the last token
        target_patch_prompt: A TargetPrompt object containing the prompt to patch and the index of the token to patch
        layers: List of layers to intervene on. If None, all layers are intervened on.
        max_length: The maximum length of the generated sequence
        remote: If True, the function will run on the nndif server. See `nnsight.net/status` to check which models are available.
        max_batch_size: The maximum number of prompts to intervene on at once.

    Returns:
        A tensor of shape (num_prompts, num_layers, vocab_size) containing the probabilities
        of the next token for each prompt at each layer. Tensor is on the CPU.
    """
    if isinstance(prompts, str):
        prompts = [prompts]
    if len(prompts) > max_batch_size:
        logger.warning(
            f"Number of prompts ({len(prompts)}) exceeds max_batch_size ({max_batch_size}). This may cause memory errors."
        )
    hiddens = get_token_activations(nn_model, prompts, remote=remote, layers=layers)
    generations = {}
    gen_kwargs = dict(remote=remote, max_new_tokens=max_length)
    if layers is None:
        layers = list(range(get_num_layers(nn_model)))
    layer_batch_size = max(max_batch_size // len(prompts), 1)
    for i in range(0, len(layers), layer_batch_size):
        layer_batch = layers[i : i + layer_batch_size]
        with nn_model.generate(**gen_kwargs) as tracer:
            for layer in layer_batch:
                with tracer.invoke(
                    [target_patch_prompt.prompt] * len(prompts),
                ):
                    get_layer_output(nn_model, layer)[
                        :, target_patch_prompt.index_to_patch
                    ] = hiddens[layer]
                    gen = nn_model.generator.output.save()
                    generations[layer] = gen
    for k, v in generations.items():
        generations[k] = v.cpu()
    return generations


def patch_object_attn_lens(
    nn_model: LanguageModel,
    source_prompts: list[str] | str,
    target_prompts: list[str] | str,
    attn_idx_patch: int,
    num_patches: int = 5,
):
    """
    A complex lens that makes the model attend to the hidden states of the last token of the source prompts instead of the attn_idx_patch token of the target prompts at last token prediction. For each layer, this intervention is performed for num_patches layers.
    Args:
        nn_model: The NNSight model
        source_prompts: The prompts to get the hidden states of the last token from
        target_prompts: The prompts to predict the next token for
        attn_idx_patch: The index of the token to patch in the target prompts
        num_patches: The number of layers to patch for each layer

    Returns:
        A tensor of shape (num_target_prompts, num_layers, vocab_size) containing the probabilities
        of the next token for each target prompt at each layer. Tensor is on the CPU.
    """
    if isinstance(source_prompts, str):
        source_prompts = [source_prompts]
    if isinstance(target_prompts, str):
        target_prompts = [target_prompts]
    num_layers = get_num_layers(nn_model)
    probs_l = []

    def get_act(model, layer):
        return get_attention(model, layer).input

    source_hiddens = get_token_activations(
        nn_model,
        source_prompts,
        get_activations=get_act,
    )
    for layer in range(num_layers):
        with nn_model.trace(target_prompts):
            for next_layer in range(layer, min(num_layers, layer + num_patches)):
                get_attention(nn_model, next_layer).input[:, attn_idx_patch] = (
                    source_hiddens[next_layer]
                )
            probs = get_next_token_probs(nn_model).cpu().save()
            probs_l.append(probs)
    return (
        th.cat(probs_l, dim=0)
        .reshape(num_layers, len(target_prompts), -1)
        .transpose(0, 1)
    )
