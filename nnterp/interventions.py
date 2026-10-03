"""Lenses over the standard values: the logit lens, patchscopes, and the object-attention lens.

Each one opens its own traces and returns tensors on the CPU. They read
``layers[i].layer_output``, ``layers[i].self_attn.input``, ``next_token_probs``
and `StandardizedTransformer.project_on_vocab`, so a family's own arithmetic
(a softcap, Granite's ``logits_scaling``, DeepSeek-V4's ``hc_head`` over the
streams) is the family's, and every lens runs on every family.

A patchscope (Ghandeharioun et al., 2024) writes a hidden state read from one
prompt into a *target prompt* at one position (`TargetPrompt`) and reads what
the model predicts there; `repeat_prompt` builds the paper's default target.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor

from .components import Unavailable
from .nnsight_utils import _check_index, get_token_activations
from .standardized import StandardizedTransformer

__all__ = [
    "logit_lens",
    "TargetPrompt",
    "repeat_prompt",
    "it_repeat_prompt",
    "TargetPromptBatch",
    "patchscope_lens",
    "patchscope_generate",
    "patch_object_attn_lens",
]

#: A next-token distribution per prompt and per block: what the lenses here return.
LayerProbs = Float[Tensor, "batch layers vocab"]


@torch.no_grad()
def logit_lens(
    nn_model: StandardizedTransformer,
    prompts: list[str] | str,
    remote: bool = False,
    return_inv_logits: bool = False,
) -> LayerProbs | tuple[LayerProbs, LayerProbs]:
    """The next-token distribution at the last token of each prompt, read off every block.

    `StandardizedTransformer.project_on_vocab` on each block's
    ``layer_output`` at the last position, then softmax. On a family whose
    residual is several streams (`Streams`, DeepSeek-V4) ``project_on_vocab``
    collapses them the way the model's own readout does, so the result has
    no stream axis there either, and the last block's row is the model's
    ``next_token_probs``. The last position is the last token of every prompt
    only under left padding, which is checked.

    Args:
        nn_model: The standardized model.
        prompts: A prompt or a list of prompts.
        remote: Run on NDIF.
        return_inv_logits: Also return the lens applied to the negated hidden
            states, ``project_on_vocab(-hidden)``.

    Returns:
        ``[num_prompts, num_layers, vocab]`` probabilities on the CPU, and the
        inverse lens of the same shape when ``return_inv_logits``.
    """
    _check_index(nn_model, -1)
    probs_l, inv_probs_l = [], []  # made outside the trace: a name bound inside does not survive it
    with nn_model.trace(prompts, remote=remote):
        for layer in nn_model.layers:
            hidden = layer.layer_output[:, -1:]  # a one-token slice keeps the seq axis, so `Streams` stays rank 4
            probs_l.append(nn_model.project_on_vocab(hidden)[:, 0].softmax(-1).cpu())
            if return_inv_logits:
                inv_probs_l.append(nn_model.project_on_vocab(-hidden)[:, 0].softmax(-1).cpu())
        probs = torch.stack(probs_l, dim=1).save()
        if return_inv_logits:
            inv_probs = torch.stack(inv_probs_l, dim=1).save()
    if return_inv_logits:
        return probs, inv_probs
    return probs


@dataclass
class TargetPrompt:
    """A target prompt and the position in it that a patchscope overwrites."""

    prompt: str
    index_to_patch: int


def repeat_prompt(
    words: list[str] | None = None,
    rel: str = " ",
    sep: str = "\n",
    placeholder: str = "?",
    index_to_patch: int = -1,
) -> TargetPrompt:
    """The patchscopes paper's target prompt for next-token prediction: few-shot repetitions, then a placeholder.

    ``sep.join(w + rel + w for w in words) + sep + placeholder``, patched at
    ``index_to_patch`` (the placeholder by default): the model is primed to
    repeat whatever the placeholder's hidden state encodes. See
    https://github.com/PAIR-code/interpretability/blob/master/patchscopes/code/next_token_prediction.ipynb

    Args:
        words: The words to repeat; default ``["king", "1135", "hello"]``.
        rel: What separates a word from its repetition.
        sep: What separates the pairs.
        placeholder: The last token, the one patched.
        index_to_patch: The token position to patch.
    """
    if words is None:
        words = ["king", "1135", "hello"]
    prompt = sep.join([w + rel + w for w in words]) + sep + placeholder
    return TargetPrompt(prompt, index_to_patch)


def it_repeat_prompt(
    tokenizer,
    words: list[str] | None = None,
    rel: str = " ",
    sep: str = "\n",
    placeholder: str = "?",
    complete_prompt: bool = True,
    add_user_instr: bool = True,
    use_system_prompt: bool = True,
) -> TargetPrompt:
    """`repeat_prompt` in the tokenizer's chat template, for instruction-tuned models; patched at the last token.

    Args:
        tokenizer: The model's tokenizer; it needs a chat template.
        words, rel, sep, placeholder: As in `repeat_prompt`.
        complete_prompt: Repeat the prompt as the start of the assistant's
            answer, so the model continues it; otherwise the answer is empty.
        add_user_instr: A user turn instructs the model to guess the next
            word and the assistant agrees, before the prompt; otherwise the
            instruction is the system prompt (or part of the user turn).
        use_system_prompt: Include a system turn; off for templates that
            refuse one.
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
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, continue_final_message=True, add_special_tokens=False)
    return TargetPrompt(prompt, -1)


@dataclass
class TargetPromptBatch:
    """Several target prompts, each with its own position to patch: one row of a patchscope's batch each."""

    prompts: list[str]
    index_to_patch: torch.Tensor

    @classmethod
    def from_target_prompts(cls, prompts_: list[TargetPrompt]) -> TargetPromptBatch:
        prompts = [p.prompt for p in prompts_]
        index_to_patch = torch.tensor([p.index_to_patch for p in prompts_])
        return cls(prompts, index_to_patch)

    @classmethod
    def from_target_prompt(cls, prompt: TargetPrompt, batch_size: int) -> TargetPromptBatch:
        """``prompt`` repeated ``batch_size`` times."""
        prompts = [prompt.prompt] * batch_size
        index_to_patch = torch.tensor([prompt.index_to_patch] * batch_size)
        return cls(prompts, index_to_patch)

    @classmethod
    def from_prompts(cls, prompts: str | list[str], index_to_patch: int | list[int] | torch.Tensor) -> TargetPromptBatch:
        """Prompts and their positions; one ``int`` position applies to every prompt."""
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(index_to_patch, int):
            index_to_patch = torch.tensor([index_to_patch] * len(prompts))
        elif isinstance(index_to_patch, list):
            index_to_patch = torch.tensor(index_to_patch)
        elif not isinstance(index_to_patch, torch.Tensor):
            raise ValueError(f"index_to_patch must be an int, a list of ints or a tensor, got {type(index_to_patch)}")
        return cls(prompts, index_to_patch)

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> TargetPrompt:
        return TargetPrompt(self.prompts[idx], self.index_to_patch[idx])

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @staticmethod
    def auto(
        target_prompt: TargetPrompt | list[TargetPrompt] | TargetPromptBatch,
        batch_size: int,
    ) -> TargetPromptBatch:
        """A `TargetPromptBatch` from any of the accepted forms; a single `TargetPrompt` is repeated ``batch_size`` times."""
        if isinstance(target_prompt, TargetPrompt):
            target_prompt = TargetPromptBatch.from_target_prompt(target_prompt, batch_size)
        elif isinstance(target_prompt, list):
            target_prompt = TargetPromptBatch.from_target_prompts(target_prompt)
        elif not isinstance(target_prompt, TargetPromptBatch):
            raise ValueError(
                f"patch_prompts must be a TargetPrompt, a list of TargetPrompt or a TargetPromptBatch, got {type(target_prompt)}"
            )
        return target_prompt


@torch.no_grad()
def patchscope_lens(
    nn_model: StandardizedTransformer,
    source_prompts: list[str] | str | None = None,
    target_patch_prompts: TargetPromptBatch | list[TargetPrompt] | TargetPrompt | None = None,
    layers: int | list[int] | None = None,
    latents: torch.Tensor | None = None,
    remote: bool = False,
) -> LayerProbs:
    """Write each source's last-token hidden state into its target prompt, block by block, and read the next token.

    For each block in ``layers``, one trace of the target prompts in which
    that block's ``layer_output`` at each target's ``index_to_patch`` is
    replaced by the source's hidden state at the same block; the result is
    the target's ``next_token_probs``. Source ``i`` goes into target ``i``.

    Args:
        nn_model: The standardized model.
        source_prompts: The prompts whose last-token hidden states are patched in.
        target_patch_prompts: The target(s); default `repeat_prompt`. A single
            `TargetPrompt` is used for every source.
        layers: The block(s) to patch; default all.
        latents: The hidden states to patch in instead of reading them off
            ``source_prompts``: ``[len(layers), num_sources, hidden]``, the
            layout `nnterp.nnsight_utils.get_token_activations` returns
            (``[..., streams, hidden]`` on a `Streams` family). Give one or
            the other.
        remote: Run on NDIF.

    Returns:
        ``[num_sources, len(layers), vocab]`` probabilities on the CPU.
    """
    if target_patch_prompts is None:
        target_patch_prompts = repeat_prompt()
    if latents is not None:
        if source_prompts is not None:
            raise ValueError("You cannot provide both source_prompts and latents")
        if len(set(len(h) for h in latents)) > 1:
            raise ValueError("Inconsistent number of hiddens")
        num_sources = len(latents[0])
    else:
        if source_prompts is None:
            raise ValueError("Either source_prompts or latents must be provided")
        if isinstance(source_prompts, str):
            source_prompts = [source_prompts]
        num_sources = len(source_prompts)
    target_patch_prompts = TargetPromptBatch.auto(target_patch_prompts, num_sources)
    if layers is None:
        layers = list(range(nn_model.num_layers))
    elif isinstance(layers, int):
        layers = [layers]
    if len(target_patch_prompts) != num_sources:
        raise ValueError(
            f"Number of sources ({num_sources}) does not match number of patch prompts ({len(target_patch_prompts)})"
        )
    if latents is None:
        latents = get_token_activations(nn_model, source_prompts, layers=layers, remote=remote)
    assert len(latents) == len(layers), (len(latents), len(layers))

    rows = torch.arange(num_sources)
    probs_l = []
    for latent, layer in zip(latents, layers):
        with nn_model.trace(target_patch_prompts.prompts, remote=remote):
            out = nn_model.layers[layer].layer_output
            out[rows, target_patch_prompts.index_to_patch] = latent.to(out)
            probs_l.append(nn_model.next_token_probs.cpu().save())
    return torch.stack(probs_l, dim=1)


@torch.no_grad()
def patchscope_generate(
    nn_model: StandardizedTransformer,
    prompts: list[str] | str,
    target_patch_prompt: TargetPrompt,
    max_length: int = 50,
    layers: list[int] | None = None,
    remote: bool = False,
    max_batch_size: int = 32,
) -> dict[int, torch.Tensor]:
    """`patchscope_lens`, generating from the patched target instead of reading one distribution.

    For each block in ``layers``, the target prompt is run once per source
    with that block's ``layer_output`` at ``index_to_patch`` replaced by the
    source's last-token hidden state, on the prompt's forward; the model
    then generates ``max_length`` tokens. Blocks are batched as invokes of
    one ``generate``, about ``max_batch_size`` rows at a time.

    Args:
        nn_model: The standardized model.
        prompts: The source prompt(s).
        target_patch_prompt: The target prompt and the position to patch.
        max_length: New tokens to generate.
        layers: The blocks to patch; default all.
        remote: Run on NDIF.
        max_batch_size: Rows per ``generate``; a single block's sources are
            never split, so more prompts than this still run together.

    Returns:
        ``{layer: ids}``, the generated sequences (target prompt included),
        ``[num_prompts, seq]`` on the CPU.
    """
    if isinstance(prompts, str):
        prompts = [prompts]
    if layers is None:
        layers = list(range(nn_model.num_layers))
    hiddens = get_token_activations(nn_model, prompts, layers=layers, remote=remote)
    generations = {}
    layer_batch_size = max(max_batch_size // len(prompts), 1)
    for i in range(0, len(layers), layer_batch_size):
        with nn_model.generate(remote=remote, max_new_tokens=max_length) as tracer:
            for j in range(i, min(i + layer_batch_size, len(layers))):
                with tracer.invoke([target_patch_prompt.prompt] * len(prompts)):
                    out = nn_model.layers[layers[j]].layer_output
                    out[:, target_patch_prompt.index_to_patch] = hiddens[j].to(out)
                    generations[layers[j]] = tracer.result.save()
    return {layer: ids.cpu() for layer, ids in generations.items()}


@torch.no_grad()
def patch_object_attn_lens(
    nn_model: StandardizedTransformer,
    source_prompts: list[str] | str,
    target_prompts: list[str] | str,
    attn_idx_patch: int,
    num_patches: int = 5,
) -> LayerProbs:
    """Make the attention see each source's last token in place of one target token, over a window of blocks.

    For each block ``l``, one trace of the target prompts in which the input
    of the attention (``self_attn.input``, the normed stream it reads) at
    position ``attn_idx_patch`` is replaced, in blocks ``l`` to
    ``l + num_patches - 1``, by the source's last-token attention input at
    the same block; the result is the target's ``next_token_probs``. The
    rest of each block (the residual, the MLP) still sees the target's own
    token.

    Args:
        nn_model: The standardized model.
        source_prompts: The prompts whose last-token attention inputs are
            patched in: one, or one per target.
        target_prompts: The prompts to predict the next token of.
        attn_idx_patch: The target token position to patch.
        num_patches: How many consecutive blocks to patch from each block on.

    Returns:
        ``[num_target_prompts, num_layers, vocab]`` probabilities on the CPU.

    Raises:
        Unavailable: when a block has no softmax attention (a hybrid's
            recurrent blocks, a state-space model).
    """
    no_attn = [i for i, layer in enumerate(nn_model.layers) if getattr(layer, "self_attn", None) is None]
    if no_attn:
        raise Unavailable(
            f"patch_object_attn_lens patches every block's self_attn.input, and blocks {no_attn} have no self_attn module"
        )
    if isinstance(source_prompts, str):
        source_prompts = [source_prompts]
    if isinstance(target_prompts, str):
        target_prompts = [target_prompts]
    num_layers = nn_model.num_layers
    source_hiddens = get_token_activations(
        nn_model, source_prompts, get_activations=lambda model, layer: model.layers[layer].self_attn.input
    )
    probs_l = []
    for layer in range(num_layers):
        with nn_model.trace(target_prompts):
            for next_layer in range(layer, min(num_layers, layer + num_patches)):
                attn = nn_model.layers[next_layer].self_attn
                attn_in = attn.input.clone()  # a parallel block feeds the same tensor to its MLP, which keeps the target's
                attn_in[:, attn_idx_patch] = source_hiddens[next_layer].to(attn_in)
                attn.input = attn_in
            probs_l.append(nn_model.next_token_probs.cpu().save())
    return torch.stack(probs_l, dim=1)
