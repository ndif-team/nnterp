# Methods

`StandardizedTransformer`'s methods are written against the standard values, so each runs unchanged on every family. `skip_layers` and `steer` edit the residual stream inside a trace, `project_on_vocab` decodes a hidden state through the model's own head, and `get_topk_closest_tokens` and `probs_to_dict` turn the result into tokens. The signatures are in the [Documentation tab](../documentation/standardized.md#nnterp.standardized.Standardized).

```python
import torch
from nnterp import StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2", device="cpu")
prompt = "The Eiffel Tower is in the city of"

with model.trace(prompt):
    before = model.layers[3].layer_output.save()
    model.skip_layers(4, 7)                           # blocks 4 to 7 do not run
    after = model.layers[7].layer_output.save()
    logits = model.logits.save()

print(torch.equal(before, after))
# True
print(model.probs_to_dict(logits[0, -1].softmax(-1), k=3))
# {' the': 0.057030126452445984, ' London': 0.03870350494980812, ' New': 0.024094339460134506}
```

Without the skip the model puts `' Paris'` first, at 0.070.

## skip_layers

`skip_layers` hands the stream entering block `start` to block `end + 1`. Both indices are inclusive, and negative ones count from the end. `skip_with`, when given, replaces that stream with a tensor of your own, `[batch, seq, hidden_size]`. A skipped block's submodules never run, so nothing under its `self_attn` or `mlp` can be read ([nnsight: skip execution](https://nnsight.net/features/13_skip/)). On a family with tuple-returning blocks, such as GPT-J, BLOOM or Falcon, the stream is packed the way the block would have returned it.

The call reads `layers[start].input` itself, so a read of that location after it is out of order:

```python
try:
    with model.trace(prompt):
        model.skip_layers(4, 7)
        entering = model.layers[4].layer_input.save()
except Exception as error:
    print(type(error).__name__, error)
# OutOfOrderError 'model.transformer.h.4.input.i0' was requested but the model already ran past it
```

Read `layers[start - 1].layer_output` before the call instead, as the first block does.

## steer

`steer` adds `factor * vector`, with `factor` 1.0 by default, to `layer_output` of each block in `layers` (an int, or an ascending list), in place, so the add reaches the model. `vector` is `[hidden_size]` or anything broadcastable to the selected slice, and is cast to the stream's dtype and device. `token_positions` (an int, a list or a slice) and `batch_index` restrict the add; both default to everything.

```python
torch.manual_seed(0)
direction = torch.randn(model.hidden_size)

with model.trace(prompt):
    clean = model.layers[6].layer_output.save()

with model.trace(prompt):
    model.steer(6, direction, factor=3.0, token_positions=-1)
    steered = model.layers[6].layer_output.save()
    logits = model.logits.save()

print(torch.allclose(steered[:, -1] - clean[:, -1], 3 * direction, atol=1e-5), torch.equal(steered[:, :-1], clean[:, :-1]))
# True True
print(model.probs_to_dict(logits[0, -1].softmax(-1), k=3))
# {' Chicago': 0.024904731661081314, ' New': 0.020505620166659355, ' Los': 0.019833678379654884}
```

The add lands on the stream leaving a block. To steer the stream entering block `i`, steer block `i - 1`. Scale the factor against the stream's norm at that block, which grows with depth and differs between models by orders of magnitude. Here the last position's stream at block 6 has norm 92.6 and the random direction 28.5, which makes `factor=3.0` about the stream's own size. [Steering](../tutorials/steering.md) derives a direction worth adding.

## project_on_vocab

`project_on_vocab` returns logits for a residual-stream tensor: the final norm, `lm_head`, then whatever the model does to the head's output before it returns `logits`. Inside a trace it takes a live value, outside a saved one, and on the last block's `layer_output` it equals `logits` exactly.

```python
with model.trace(prompt):
    mid = model.layers[6].layer_output.save()
    last = model.layers[-1].layer_output.save()
    logits = model.logits.save()

print(torch.equal(model.project_on_vocab(last), logits))
# True
print(model.get_topk_closest_tokens(mid[0, -1], k=3))
# [{' East': 0.05666778236627579, ' England': 0.04518940672278404, ' southwest': 0.038670022040605545}]
```

The step after the head is where families differ, and it is why a hand-rolled `lm_head(norm(hidden))` is not always the model's prediction. The plain case is the softcap when the text config sets `final_logit_softcapping`: on `google/gemma-2-2b` (cap 30.0) `project_on_vocab(last)` equals `logits`, and `lm_head(norm(last))` differs from them by up to 50.75 on this prompt. A family where the model does something else after the head defines its own `project_on_vocab`: Cohere multiplies by `logit_scale`, Granite divides by `logits_scaling`, and DeepSeek-V4 first collapses its parallel streams the way the model does.

A logit lens over every block is a loop inside one trace ([Logit lens](../tutorials/logit-lens.md) is the full recipe). A list bound in the block needs `nnsight.save`:

```python
import nnsight

with model.trace(prompt):
    lens = nnsight.save([model.project_on_vocab(layer.layer_output)[0, -1] for layer in model.layers])

print([next(iter(model.probs_to_dict(row.softmax(-1), k=1))) for row in lens])
# [' the', ' the', ' the', ' the', ' the', ' the', ' East', ' Ing', ' Rome', ' London', ' Paris', ' Paris']
```

## get_topk_closest_tokens and probs_to_dict

`get_topk_closest_tokens` is `project_on_vocab`, softmax, then the `k` most likely tokens (5 by default) for each row of `hidden.reshape(-1, hidden_size)`, one `{token: probability}` dict per position. `probs_to_dict` does the last step alone, on one `[vocab_size]` distribution, with the same `k`.

```python
print(model.get_topk_closest_tokens(last[0], k=1)[:3])          # the first three positions
# [{'\n': 0.013599064201116562}, {'-': 0.04670916870236397}, {'el': 0.9958784580230713}]

with model.trace(prompt):
    probs = model.next_token_probs.save()

print(model.probs_to_dict(probs[0], k=5))
# {' Paris': 0.06996913999319077, ' London': 0.0583544597029686, ' New': 0.03101949580013752, ' Amsterdam': 0.029201989993453026, ' Berlin': 0.023510510101914406}
```

Each dict holds exactly `k` entries, most likely first. Where two of them decode to the same text (partial UTF-8 byte tokens all decode to the replacement character), those entries are keyed by the tokenizer's raw vocabulary token instead. Both methods call `.item()` and `tokenizer.decode`, so their result is plain Python. Inside a trace it has to be kept with `nnsight.save`, and calling them outside, on saved tensors, is the plain form.
