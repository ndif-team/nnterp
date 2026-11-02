# Helpers

`nnterp.nnsight_utils` collects one token position's activation at chosen blocks over many prompts, and `nnterp.prompt_utils` measures the probability mass on named target tokens over many prompts. Both read standard values (`layer_output`, `next_token_probs`), so one call runs on every family. The signatures are in the Documentation tab: [nnsight_utils](../documentation/nnsight_utils.md), [prompt_utils](../documentation/prompt_utils.md).

```python
from nnterp import StandardizedTransformer
from nnterp.nnsight_utils import get_token_activations

model = StandardizedTransformer("openai-community/gpt2", device="cpu", tokenizer_kwargs={"padding_side": "left"})
prompts = ["The Eiffel Tower is in", "Hello"]

acts = get_token_activations(model, prompts, layers=[4, 8])
print(acts.shape, acts.device)
# torch.Size([2, 2, 768]) cpu
```

`acts[i, j]` is `model.layers[layers[i]].layer_output[j, -1]`: the stream leaving block `layers[i]` at the last token of prompt `j`. The layer axis follows `layers`, not the block numbers, and the result is on the CPU. The trace stops after the last requested block, so later blocks do not run.

## The token index and the padding side

`idx` is the token position, `-1` by default. A negative index needs left padding and a positive one right padding, so that the index names the same token in every row. `0` needs neither. The check runs against `model.tokenizer.padding_side` before anything else:

```python
try:
    get_token_activations(model, prompts, idx=2)
except ValueError as error:
    print(error)
# a positive token index needs right padding, and the tokenizer pads 'left'
```

nnsight already sets a decoder-only model's tokenizer to pad on the left, and the load above only makes that explicit with `tokenizer_kwargs`. The same rule covers every function here that batches prompts: `next_token_probs` is each row's last position, which is each prompt's own last token only under left padding.

## Another value, many prompts

`get_activations(model, layer)` picks the value read at each block; any value with the sequence axis in position 1 works, because the helper indexes `[:, idx]` on it:

```python
attn = get_token_activations(model, prompts, layers=[4], get_activations=lambda m, i: m.layers[i].self_attn.attention_output)
print(attn.shape)
# torch.Size([1, 2, 768])
```

`collect_token_activations_batched` runs `get_token_activations` over batches and concatenates along the prompt axis. `tqdm=` takes a progress-bar factory to wrap the batch loop. With `remote=True` the batches go through `collect_last_token_activations_session`, one `model.session`, so a remote run is one request ([Remote](remote.md)).

```python
from nnterp.nnsight_utils import collect_token_activations_batched

many = [f"Prompt number {i}" for i in range(10)]
acts = collect_token_activations_batched(model, many, batch_size=4, layers=[0, 11])
print(acts.shape)
# torch.Size([2, 10, 768])
```

`compute_next_token_probs` is the distribution at the end, `[num_prompts, vocab_size]` on the CPU:

```python
from nnterp.nnsight_utils import compute_next_token_probs

probs = compute_next_token_probs(model, prompts)
print(probs.shape, model.probs_to_dict(probs[0], k=3))
# torch.Size([2, 50257]) {' the': 0.2401452511548996, ' a': 0.0862361341714859, ' fact': 0.02262996695935726}
```

The `tracer=` form of `get_token_activations` reads inside a trace you have open instead of opening one; the result is then a live tensor on the model's device, and yours to `.save()`.

## Target token mass over many prompts

A `Prompt` pairs a text with named sets of target token ids, and `run_prompts` runs many of them in batches and returns each target's mass per prompt:

```python
from nnterp.prompt_utils import Prompt, run_prompts

targets = [
    Prompt.from_strings("The capital of France is", {"correct": "Paris", "wrong": ["London", "Berlin"]}, model),
    Prompt.from_strings("The capital of Germany is", {"correct": "Berlin", "wrong": ["London", "Paris"]}, model),
]
print(targets[0].target_tokens, targets[0].has_no_collisions())
# {'correct': [40313, 6342], 'wrong': [23421, 3576, 24814, 11307]} True

mass = run_prompts(model, targets, batch_size=32)
print(mass["correct"].shape, mass["correct"][:, 0].tolist())
# torch.Size([2, 1]) [0.03224833309650421, 0.05272921919822693]
```

`from_strings` turns each word into token ids with `get_first_tokens`: the first token of `"Paris"` and of `" Paris"`, tokenized with `model.add_prefix_false_tokenizer` so that the two differ. A target's mass is the sum over its ids, and `has_no_collisions` says whether two targets share one. The result is `[num_prompts, layers]`, the next-token distribution counting as one layer. Every prompt must name the same targets. When you pick target ids yourself instead, take `model.tokenizer(" Paris", add_special_tokens=False).input_ids` and check that it is one token: `tokenizer.encode(" Paris")[0]` is BOS on Llama and Gemma, and some vocabularies split the word, or make `"Paris"` and `" Paris"` the same token.

`get_probs_func` gives the layer axis a meaning of your own. The logit lens at chosen blocks, through `project_on_vocab`:

```python
import torch

def lens_probs(model, batch, remote=False, layers=None):
    with model.trace(batch, remote=remote):
        rows = [model.project_on_vocab(model.layers[i].layer_output)[:, -1].softmax(-1) for i in layers]
        out = torch.stack(rows, dim=1).cpu().save()                # [batch, len(layers), vocab_size]
    return out

mass = run_prompts(model, targets, batch_size=32, get_probs_func=lens_probs, func_kwargs={"layers": [6, 9, 11]})
print(mass["correct"].shape, [round(p, 4) for p in mass["correct"][0].tolist()])
# torch.Size([2, 3]) [0.0001, 0.1821, 0.0322]
```

The function receives `remote=` as a keyword and must pass it to `model.trace`. `func_kwargs` carries the rest.
