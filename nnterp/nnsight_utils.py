from __future__ import annotations

from typing import Union, Callable
import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader

from nnsight import NNsight, Object, LanguageModel
from nnsight.intervention.envoy import Envoy
from transformers import PreTrainedModel

from .utils import TraceTensor, unpack_tuple
from .standardized_transformer import StandardizedTransformer
from .rename_utils import get_rename_dict, RenameConfig

GetModuleOutput = Callable[[LanguageModel, int], TraceTensor]


def get_embed_tokens(model: LanguageModel) -> nn.Module:
    """
    Get the token embedding layer of the model
    """
    return model.embed_tokens


def get_layers(model: LanguageModel) -> list[Envoy]:
    """
    Get the layers of the model
    """
    if isinstance(model, StandardizedTransformer):
        return model.layers
    return model.model.layers


def get_num_layers(nn_model: LanguageModel):
    """
    Get the number of layers in the model
    Args:
        nn_model: The NNSight model
    Returns:
        The number of layers in the model
    """
    return len(get_layers(nn_model))


def get_layer(nn_model: LanguageModel, layer: int) -> Envoy:
    """
    Get the layer of the model
    Args:
        nn_model: The NNSight model
        layer: The layer to get
    Returns:
        The Envoy for the layer
    """
    return get_layers(nn_model)[layer]


def get_layer_input(nn_model: LanguageModel, layer: int) -> Union[int, Object]:
    """
    Get the hidden state input of a layer
    Args:
        nn_model: The NNSight model
        layer: The layer to get the input of
    Returns:
        The Proxy for the input of the layer
    """
    return get_layer(nn_model, layer).input


def get_layer_output(nn_model: LanguageModel, layer: int) -> TraceTensor:
    """
    Get the residual stream after the layer
    Args:
        nn_model: The NNSight model
        layer: The layer to get the output of
    Returns:
        The Proxy for the output of the layer
    """
    output = get_layer(nn_model, layer).output
    return unpack_tuple(output)


def get_attention(nn_model: LanguageModel, layer: int) -> Envoy:
    """
    Get the attention module of a layer
    Args:
        nn_model: The NNSight model
        layer: The layer to get the attention module of
    Returns:
        The Envoy for the attention module of the layer
    """
    return get_layer(nn_model, layer).self_attn


def get_attention_output(nn_model: LanguageModel, layer: int) -> TraceTensor:
    """
    Get the output of the attention block of a layer
    Args:
        nn_model: The NNSight model
        layer: The layer to get the output of
    Returns:
        The Proxy for the output of the attention block of the layer
    """
    return unpack_tuple(get_attention(nn_model, layer).output)


def get_mlp(nn_model: LanguageModel, layer: int) -> Envoy:
    """
    Get the MLP module of a layer
    """
    return get_layer(nn_model, layer).mlp


def get_mlp_output(nn_model: LanguageModel, layer: int) -> TraceTensor:
    """
    Get the output of the MLP of a layer
    """
    return unpack_tuple(get_mlp(nn_model, layer).output)


def get_logits(nn_model: LanguageModel) -> TraceTensor:
    """
    Get the logits of the model
    Args:
        nn_model: The NNSight model
    Returns:
        The Proxy for the logits of the model
    """
    return nn_model.output.logits


def get_unembed_norm(nn_model: LanguageModel) -> Envoy:
    """
    Get the last layer norm of the model
    Args:
        nn_model: The NNSight model
    Returns:
        The Envoy for the last layer norm of the model
    """
    if isinstance(nn_model, StandardizedTransformer):
        return nn_model.ln_final
    return nn_model.model.norm


def get_unembed(nn_model: LanguageModel) -> Envoy:
    """
    Get the unembed module of the model
    Args:
        nn_model: The NNSight model
    Returns:
        The Envoy for the unembed module of the model
    """
    return nn_model.lm_head


def project_on_vocab(nn_model: LanguageModel, h: TraceTensor) -> TraceTensor:
    """
    Project the hidden states on the vocabulary, after applying the model's last layer norm
    Args:
        nn_model: The NNSight model
        h: The hidden states to project
    Returns:
        The Proxy for the hidden states projected on the vocabulary
    """
    ln_out = get_unembed_norm(nn_model)(h)
    return nn_model.lm_head(ln_out)


def get_next_token_probs(nn_model: LanguageModel) -> TraceTensor:
    """
    Get the probabilities of the model
    Args:
        nn_model: The NNSight model
    Returns:
        The Proxy for the probabilities of the model
    """
    return get_logits(nn_model)[:, -1, :].softmax(-1)


def set_layer_output(nn_model: LanguageModel, layer: int, tensor: TraceTensor):
    """
    Set the output of a layer to a certain tensor.
    Args:
        nn_model: The NNSight model
        layer: The layer to set the output of
        tensor: The tensor to set the output of the layer to
    """
    if isinstance(get_layer(nn_model, layer).output, tuple):
        get_layer(nn_model, layer).output = (
            tensor,
            *get_layer(nn_model, layer).output[1:],
        )
    else:
        get_layer(nn_model, layer).output = tensor


class ModuleAccessor:
    """
    Module that allows to use the NNsight and nnterp renaming utilities on huggingface models, to get the pytorch nn.Module objects with the same standardized names as the StandardizedTransformer class::

        ModuleAccessor
        ├── embed_tokens
        ├── layers
        │   ├── self_attn
        │   └── mlp
        ├── ln_final
        └── lm_head

    Args:
        model: The huggingface model to access
        rename_config: An optional nnterp RenameConfig if your model has custom module names
        rename: An optional dictionary to allow you to have your own custom renaming operations

    """

    def __init__(
        self,
        model: PreTrainedModel,
        rename_config: RenameConfig | None = None,
        rename: dict[str, str] | None = None,
    ):
        full_rename = get_rename_dict(rename_config=rename_config)
        if rename is not None:
            full_rename.update(rename)
        self.nn_model = NNsight(model, rename=full_rename)

    def __getattr__(self, name: str) -> nn.Module:
        attr = getattr(self.nn_model, name)
        if hasattr(attr, "_module"):
            return attr._module
        else:
            raise AttributeError(f"Attribute {name} is not a module")

    def get_embed_tokens(self) -> nn.Module:
        return self.nn_model.embed_tokens._module

    def get_layers(self) -> nn.ModuleList:
        return self.nn_model.layers._module

    def get_mlp(self, layer: int) -> nn.Module:
        return self.nn_model.layers[layer].mlp._module

    def get_attention(self, layer: int) -> nn.Module:
        return self.nn_model.layers[layer].self_attn._module

    def get_unembed_norm(self) -> nn.Module:
        return self.nn_model.ln_final._module

    def get_unembed(self) -> nn.Module:
        return self.nn_model.lm_head._module


@th.no_grad
def get_token_activations(
    nn_model: LanguageModel,
    prompts=None,
    layers=None,
    get_activations: GetModuleOutput | None = None,
    remote=False,
    idx: int | None = None,
    tracer=None,
):
    """
    Collect the hidden states of the last token of each prompt at each layer

    Args:
        nn_model: The NNSight model
        prompts: The prompts to collect activations for. Can be None if you call this from an existing tracer.
        layers: The layers to collect activations for, default to all layers
        get_activations: The function to get the activations, default to layer output
        remote: Whether to run the model on the remote device
        idx: The index of the token to collect activations for
        tracer: A tracer object to use to collect activations. If None, a new tracer is created.

    Returns:
        The hidden states of the last token of each prompt at each layer, moved to cpu. If open_context is False, returns a list of
        Proxies. Dimensions are (num_layers, num_prompts, hidden_size)
    """
    if tracer is None and prompts is None:
        raise ValueError("prompts must be provided if tracer is None")
    if get_activations is None:
        get_activations = get_layer_output
    if idx is None:
        idx = -1
    if idx < 0 and nn_model.tokenizer.padding_side != "left":
        raise ValueError(
            f"negative index is currently only supported with left padding, not {nn_model.tokenizer.padding_side}"
        )
    if idx > 0 and nn_model.tokenizer.padding_side != "right":
        raise ValueError(
            f"positive index is currently only supported with right padding, not {nn_model.tokenizer.padding_side}"
        )
    if layers is None:
        layers = list(range(get_num_layers(nn_model)))
    last_layer = max(layers)
    if min(layers) < 0:
        last_layer = max(last_layer, get_num_layers(nn_model) + min(layers))

    # Collect the hidden states of the last token of each prompt at each layer
    acts = []
    if tracer is None:
        with nn_model.trace(prompts, remote=remote) as tracer:
            for layer in layers:
                acts.append(get_activations(nn_model, layer)[:, idx].cpu().save())
            tracer.stop()
    else:
        device = get_layer_output(nn_model, 0).device
        for layer in layers:
            acts.append(get_activations(nn_model, layer)[:, idx].to(device))
    return th.stack(acts)


@th.no_grad
def collect_last_token_activations_session(
    nn_model,
    prompts,
    batch_size,
    layers=None,
    get_activations=None,
    remote=False,
    idx=None,
):
    """
    Collect the hidden states of the specified token of each prompt at each layer in batches using a nnsight session.

    Args:
        nn_model: The NNSight model
        prompts: The prompts to collect activations for
        batch_size: The batch size to use
        layers: The layers to collect activations for, default to all layers
        get_activations: The function to get the activations, default to layer output
        remote: Whether to run the model on the remote device
        idx: The index of the token to collect activations for. Default is -1 (last token).

    Returns:
        The hidden states of the specified token of each prompt at each layer, moved to cpu.
        Dimensions are (num_layers, num_prompts, hidden_size)
    """
    if layers is None:
        layers = list(range(get_num_layers(nn_model)))
    last_layer = max(layers)
    if min(layers) < 0:
        last_layer = max(last_layer, get_num_layers(nn_model) + min(layers))
    if get_activations is None:
        get_activations = get_layer_output
    if idx is None:
        idx = -1
    if idx < 0 and nn_model.tokenizer.padding_side != "left":
        raise ValueError("negative index is currently only supported with left padding")
    if idx > 0 and nn_model.tokenizer.padding_side != "right":
        raise ValueError(
            "positive index is currently only supported with right padding"
        )
    with nn_model.session(remote=remote) as session:
        dl = DataLoader(prompts, batch_size=batch_size)
        all_acts = []
        for batch in dl:
            with nn_model.trace(batch) as tracer:
                acts = []
                for layer in layers:
                    acts.append(
                        get_activations(nn_model, layer)[
                            :,
                            idx,
                        ]
                        .cpu()
                        .save()
                    )
                tracer.stop()
            all_acts.append(th.stack(acts).save())
        all_acts = th.cat(all_acts, dim=1).save()
    return all_acts


def collect_token_activations_batched(
    nn_model: LanguageModel,
    prompts,
    batch_size,
    layers=None,
    get_activations: GetModuleOutput | None = None,
    remote=False,
    idx=None,
    tqdm=None,
    use_session=True,
):
    """
    Collect the hidden states of the last token of each prompt at each layer in batches

    Args:
        nn_model: The NNSight model
        prompts: The prompts to collect activations for
        batch_size: The batch size to use
        layers: The layers to collect activations for, default to all layers
        get_activations: The function to get the activations, default to layer output
        remote: Whether to run the model on the remote device
        idx: The index of the token to collect activations for. Default is -1 (last token).
        tqdm: Whether to use tqdm to show progress, default to None (no progress bar)
        use_session: Whether to use a nnsight session to collect activations. Not sure why you'd want turn that off but who knows

    Returns:
        The hidden states of the specified token of each prompt at each layer, moved to cpu.
        Dimensions are (num_layers, num_prompts, hidden_size)
    """
    if use_session and remote:
        return collect_last_token_activations_session(
            nn_model,
            prompts,
            batch_size,
            layers,
            get_activations,
            remote,
            idx,
        )
    num_prompts = len(prompts)
    acts = []
    it = range(0, num_prompts, batch_size)
    if tqdm is not None:
        it = tqdm(it)
    for i in it:
        batch = prompts[i : min(i + batch_size, num_prompts)]
        acts_batch = get_token_activations(
            nn_model, batch, layers, get_activations, remote, idx
        )
        acts.append(acts_batch)
    return th.cat(acts, dim=1)


def compute_next_token_probs(
    nn_model: LanguageModel, prompt: str | list[str], remote=False
) -> th.Tensor:
    """
    Get the probabilities of the next token for the prompt
    Args:
        nn_model: The NNSight model
        prompt: The prompt to get the probabilities for
        remote: Whether to run the model on the remote device
    Returns:
        The probabilities of the next token for the prompt
    """
    with nn_model.trace(prompt, remote=remote):
        out = nn_model.output.logits
        out = out[:, -1].softmax(-1).cpu().save()
    return out
