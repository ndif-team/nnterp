# Generation

Under `model.generate(prompt, max_new_tokens=N)` the model runs one forward call for the prompt and one per new token ([nnsight: multiple token generation](https://nnsight.net/features/4_multiple_token/)), and every standard value is served once per call: `[batch, prompt_len, ...]` on the prompt call, `[batch, 1, ...]` on a decode step.

```python
from nnterp import StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2", device="cpu", attn_implementation="eager")
prompt = "The Eiffel Tower is in the city of"

resid, probs = [], []                                  # bound before the block: a list bound inside it does not survive it
with model.generate(prompt, max_new_tokens=3, do_sample=False) as tracer:
    for step in tracer.iter[:3]:                       # steps 0, 1, 2
        resid.append(model.layers[6].layer_output.save())
        probs.append(model.next_token_probs.save())
    ids = tracer.result.save()

print([tuple(r.shape) for r in resid])
# [(1, 10, 768), (1, 1, 768), (1, 1, 768)]
print([p.argmax(-1).item() for p in probs], ids[0, -3:].tolist(), repr(model.tokenizer.decode(ids[0, -3:])))
# [6342, 11, 290] [6342, 11, 290] ' Paris, and'
```

`next_token_probs` is the distribution each call samples from, so under greedy decoding its argmax is the token the step appended. Decoding follows the checkpoint's `generation_config`; pass `do_sample=False` for a deterministic run.

## A decode step's values

| value | prompt call (step 0) | decode step |
| --- | --- | --- |
| `model.input_ids` | `[batch, prompt_len]` | `[batch, 1]`, the token the previous step produced |
| `layers[i].layer_output`, `attention_output`, `mlp_output`, `token_embeddings` | `[batch, prompt_len, hidden_size]` | `[batch, 1, hidden_size]` |
| `self_attn.attention_queries`, `attention_head_outputs` | `prompt_len` on the sequence axis | 1 on the sequence axis |
| `self_attn.attention_keys`, `attention_values` | `[batch, num_kv_heads, prompt_len, head_dim]` | `[batch, num_kv_heads, prompt_len + step, head_dim]`, the cache |
| `self_attn.attention_scores`, `attention_probabilities` | `[batch, num_heads, prompt_len, prompt_len]` | `[batch, num_heads, 1, prompt_len + step]` |
| `model.logits` | `[batch, prompt_len, vocab_size]` | `[batch, 1, vocab_size]` |
| `model.next_token_probs` | `[batch, vocab_size]` | `[batch, vocab_size]` |

```python
with model.generate(prompt, max_new_tokens=3, do_sample=False) as tracer:
    for step in tracer.iter[1]:                        # the first decode step
        ids_in = model.input_ids.save()
        keys = model.layers[6].self_attn.attention_keys.save()
        pattern = model.layers[6].self_attn.attention_probabilities.save()
        out = model.layers[6].layer_output.save()
    ids = tracer.result.save()

print(ids_in.tolist(), tuple(keys.shape), tuple(pattern.shape), tuple(out.shape))
# [[6342]] (1, 12, 11, 64) (1, 12, 1, 11) (1, 1, 768)
```

The keys and values grow with the cache and the queries do not, so `pattern[0, h, 0]` is what head `h` of the new token attends to across the whole context so far, and a head-wise edit of `attention_keys` on a decode step touches the whole context. A read outside any loop binds to step 0, the prompt call. A write at a step reaches that step and every later one:

```python
with model.generate(prompt, max_new_tokens=3, do_sample=False) as tracer:
    for step in tracer.iter[1]:
        model.layers[6].layer_output[:] = 0
    ids = tracer.result.save()

print(repr(model.tokenizer.decode(ids[0, -3:])))
# ' Paris in the'
```

Step 0's token stays `' Paris'`; the two after it change.

## A read out of order inside the loop

Within one step's body the reads follow the forward, as in a trace: `input_ids` first, a block's interior values before its `attention_output`, that before the block's `layer_output`, and `logits` and `next_token_probs` last. What differs under `generate` is how a misplaced read fails.

!!! warning "A shifted list, not an error"
    In a plain trace, reading `attention_probabilities` after the same block's `attention_output` raises `OutOfOrderError`. Inside `tracer.iter` the read binds the value's next occurrence, which is the next step's, so the list comes back shifted by one step with no error. Only the read on the last step has no next occurrence. It cuts the loop short with nnsight's `was never reached` warning, which blames the loop bound, and the statements after the loop (here `tracer.result`) never run. A saved name that is missing, or a list that looks one step off, is a read-order error.

```python
outs, patterns = [], []
with model.generate(prompt, max_new_tokens=3, do_sample=False) as tracer:
    for step in tracer.iter[:3]:
        outs.append(model.layers[6].self_attn.attention_output.save())
        patterns.append(model.layers[6].self_attn.attention_probabilities.save())   # after attention_output: out of order

print(len(outs), len(patterns), [tuple(p.shape) for p in patterns])
# 3 2 [(1, 12, 1, 11), (1, 12, 1, 12)]
```

The key axis says which steps these are: 11 and 12 keys are steps 1 and 2, not 0 and 1. Check a saved shape against its step before trusting a per-step list.

A recurrent mixer has its own order inside a decode step: on Mamba-1 and Mamba-2, `state_output` comes before `attention_head_outputs` ([Mamba](architectures/mamba.md), [Mamba-2](architectures/mamba-2.md)), and on a DeltaNet hybrid `linear_attn.state_input` is the previous step's `state_output` ([Gated DeltaNet](architectures/delta-net.md)).
