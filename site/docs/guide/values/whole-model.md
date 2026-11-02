# The whole model

Inside a trace the root envoy answers for the whole run: `logits`, `lm_head_output`, `final_norm_output`, `token_embeddings`, `next_token_probs`, and the `input_ids`, `attention_mask` and `input_size` the model was called with. Outside a trace it answers for the sizes, `num_layers`, `hidden_size`, `num_heads` and the rest, read off the config.

```python
import torch
from nnterp import StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2")

with model.trace("The Eiffel Tower is in"):
    ids = model.input_ids.save()
    mask = model.attention_mask.save()
    emb = model.token_embeddings.save()
    normed = model.final_norm_output.save()
    head = model.lm_head_output.save()
    logits = model.logits.save()
    probs = model.next_token_probs.save()

print(ids.shape, emb.shape, normed.shape, logits.shape, probs.shape)
# torch.Size([1, 7]) torch.Size([1, 7, 768]) torch.Size([1, 7, 768]) torch.Size([1, 7, 50257]) torch.Size([1, 50257])
print(torch.equal(head, logits), model.probs_to_dict(probs[0], k=3))
# True {' the': 0.2401435822248459, ' a': 0.08623422682285309, ' fact': 0.022629808634519577}
```

`probs_to_dict` gives the `k` most likely tokens of one distribution ([Methods](../methods.md)).

| value | shape | what it is | writes |
| --- | --- | --- | --- |
| `input_ids` | `[batch, seq]` | the token ids the model was called with | assign |
| `attention_mask` | `[batch, seq]` | the mask the model was called with; zeros are padding | assign |
| `input_size` | `torch.Size([batch, seq])` | the ids' shape | read-only |
| `token_embeddings` | `[batch, seq, hidden_size]` | the embedding module's output | assign |
| `final_norm_output` | `[batch, seq, hidden_size]` | the final norm's output, what the head reads | assign, in place |
| `lm_head_output` | `[batch, seq, vocab_size]` | the head's own projection | assign, in place |
| `logits` | `[batch, seq, vocab_size]` | the logits the model returns, after anything it does past the head | assign |
| `next_token_probs` | `[batch, vocab_size]` | `logits[:, -1].softmax(-1)` | read-only |

The table is in forward order: the inputs first, `token_embeddings` before any block's value, and the last three after them all.

## logits and lm_head_output

`logits` is what the model returns. On most families it is `lm_head_output`. Gemma-2 softcaps the head's output (`config.final_logit_softcapping`), Cohere multiplies it by `logit_scale` and Granite divides it by `logits_scaling`, and `logits` is the result. `model.project_on_vocab` ends with the same step, so the logit lens on the last block's `layer_output` equals `logits` on every family ([Methods](../methods.md)).

```python
gemma = StandardizedTransformer("google/gemma-2-2b")

with gemma.trace("The Eiffel Tower is in"):
    head = gemma.lm_head_output.save()
    logits = gemma.logits.save()

cap = gemma.config.final_logit_softcapping
print(cap, torch.equal(head, logits), torch.allclose(logits, cap * torch.tanh(head / cap)))
# 30.0 False True
print(head[0, -1].abs().max().item(), logits[0, -1].abs().max().item())
# 42.63672637939453 26.695619583129883
```

Assigning `logits` replaces them in the model's output, so `tracer.result.logits` is what you set.

## token_embeddings and block 0

`token_embeddings` is `embed_tokens.output`, with whatever the embedding module does itself (Gemma's `sqrt(hidden_size)` scale) and nothing the model adds after it. GPT-2 adds its positional embeddings, BLOOM norms the output and Granite multiplies it by `embedding_multiplier`, so the stream entering block 0 is `layers[0].layer_input`:

```python
with model.trace("The Eiffel Tower is in"):
    emb = model.token_embeddings.save()
    pos = model.transformer.wpe.output.save()
    first = model.layers[0].layer_input.save()

print(torch.equal(emb, first), torch.equal(emb + pos, first))
# False True
```

## next_token_probs

`next_token_probs` is the softmax at the last position, derived from the output. That position is the last token of every row under left padding, which nnsight sets on a causal model's tokenizer:

```python
with model.trace(["Hi", "The Eiffel Tower is in"]):
    ids = model.input_ids.save()
    probs = model.next_token_probs.save()

print(ids)
# tensor([[50256, 50256, 50256, 50256, 50256, 50256, 17250],
#         [  464,   412,   733,   417,  8765,   318,   287]])
print(model.probs_to_dict(probs[0], k=3))
# {'.': 0.07607239484786987, ',': 0.06419526785612106, '\n': 0.035513173788785934}
print(model.probs_to_dict(probs[1], k=3))
# {' the': 0.2401452511548996, ' a': 0.0862361341714859, ' fact': 0.02262996695935726}
```

The softmax is in the model's dtype, and in float16 thousands of entries underflow to exact zeros, so take a KL on `model.logits[:, -1].float().log_softmax(-1)` instead. Assigning `next_token_probs` raises `AttributeError: next_token_probs is derived from the logits and cannot be assigned; assign model.logits instead`.

## The input

`input_ids` and `attention_mask` are what the model was called with, and the model runs on what you assign to them. Assigning `input_ids` does not resize the mask, so assign `attention_mask` too when the length changes:

```python
with model.trace("Paris is the capital of"):
    other_ids = model.input_ids.save()
    other = model.logits.save()

with model.trace("The Eiffel Tower is in"):
    model.input_ids = other_ids
    model.attention_mask = torch.ones_like(other_ids)
    swapped = model.logits.save()

print(torch.equal(swapped, other))
# True
```

`input_size` is the ids' `torch.Size`, read-only. A `torch.Size` bound inside the trace does not survive it, so save `torch.tensor(model.input_size)` to keep it.

## Sizes

The sizes are read-only attributes of the root, readable before any trace and without `dispatch`:

```python
print(model.num_layers, model.hidden_size, model.vocab_size)
# 12 768 50257
print(model.num_heads, model.num_kv_heads, model.head_dim, model.qk_head_dim, model.intermediate_size)
# 12 12 64 64 3072
print(model.layers[5].self_attn.num_heads, model.layers[5].self_attn.head_dim, model.layers[5].mlp.intermediate_size)
# 12 64 3072
```

| size | the plain rule |
| --- | --- |
| `num_layers` | `len(model.layers)` |
| `hidden_size` | `config.hidden_size` |
| `vocab_size` | `config.vocab_size` |
| `num_heads` | `config.num_attention_heads` |
| `num_kv_heads` | `config.num_key_value_heads`, else `num_heads` |
| `head_dim` | `config.head_dim` when the config has it (Qwen3, Gemma), else `hidden_size // num_heads` |
| `qk_head_dim` | `head_dim`, unless the family separates them (DeepSeek's latent attention) |
| `intermediate_size` | `config.intermediate_size` |

The rule reads the text config (a multimodal checkpoint's `text_config`), unless the family's module defines a function of the same name, which it does where its config spells a size its own way: Falcon's `num_kv_heads` is `config.num_kv_heads` on the 40B layout and 1 under `multi_query`, GPT-2's `intermediate_size` is `config.n_inner`, four times `hidden_size` when unset. `model.hidden_size = 5` raises an `AttributeError` saying that the size is read off the config and that a family defines `def hidden_size(model)` to say it otherwise.

A root size is the config's value, which equals every block's where the blocks agree. A block's own sizes are on its attention (`num_heads`, `num_kv_heads`, `head_dim`, `qk_head_dim`) and its MLP (`intermediate_size`, one routed expert's width on a mixture of experts), read off the module, and they differ from the root's on Gemma-4 (`head_dim` on every checkpoint, `num_kv_heads` on 26B-A4B and 31B, the MLP width on E2B), MiMo-V2-Flash (`num_kv_heads`) and Laguna (`num_heads`).

The signatures are on [`StandardizedTransformer`](../../documentation/standardized.md), and the axes behind every shape on this page are on [Layouts](layouts.md).
