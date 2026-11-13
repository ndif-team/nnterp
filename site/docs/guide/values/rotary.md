# Queries and keys before the rotary embedding

`attention_queries_prerope` and `attention_keys_prerope` on `model.layers[i].self_attn` are the queries and keys a block's rotary embedding receives: after the projection and any query or key norm, before the rotation for their positions. `attn.rotate(x, positions)` applies that block's own rotation to any such tensor, outside a trace, so rotating the served values at the positions they ran at gives `attention_queries` and `attention_keys` bit for bit.

```python
import torch
from nnterp import StandardizedTransformer

model = StandardizedTransformer("Qwen/Qwen3-0.6B", attn_implementation="eager")
attn = model.layers[5].self_attn

with model.trace("The Eiffel Tower is in Paris"):
    q_src = attn.attention_queries_prerope.save()
    k_src = attn.attention_keys_prerope.save()
with model.trace("The Eiffel Tower is in Paris"):
    k = attn.attention_keys.save()

print(q_src.shape, k_src.shape)
# torch.Size([1, 16, 8, 128]) torch.Size([1, 8, 8, 128])
assert torch.equal(attn.rotate(k_src, range(k_src.shape[2])), k)
```

| value | shape | what it is |
| --- | --- | --- |
| `attention_queries_prerope` | `[batch, num_heads, seq, qk_head_dim]` | the queries the rotary embedding receives |
| `attention_keys_prerope` | `[batch, num_kv_heads, seq, qk_head_dim]` | the keys the rotary embedding receives, before the cache and `repeat_kv` |

Both are read on the attention module's own forward, not inside the eager attention, so they are served under any `attn_implementation`, and they come before `attention_queries` in a trace. The vLLM engine serves them too, and there `rotate` runs inside a trace ([vLLM](../../vllm/index.md#before-the-rotary-embedding)).

## Moving a query or key to another position

A rotated query carries the rotation of the position it ran at; written at another position it is attended with the wrong one. Write the value before the rotary instead: the block rotates what you wrote for the position it lands at, so the transplant is one assignment. `rotate` gives what the block will make of it:

```python
with model.trace("Rome is a city in"):
    attn.attention_queries_prerope[:, :, -1] = q_src[:, :, -1]
    moved = attn.attention_queries.save()

assert torch.equal(moved[:, :, -1:], attn.rotate(q_src[:, :, -1:], [moved.shape[2] - 1]))
```

Use these values, not `self_attn.q_proj.output`, for any edit that must come before the rotation: the projection's layout is the family's (Phi-3 and GPT-NeoX fuse the queries, keys and values into one projection), and on Qwen3, OLMo-2/3 and Gemma a norm sits between the projection and the rotary.

## Families without a rotary embedding

GPT-2, OPT, BLOOM, MPT, XGLM and the other families that position tokens another way (absolute embeddings, ALiBi) have neither value, and some families skip the rotary on some blocks (Cohere 2's and AFMoE's full-attention blocks, SmolLM3's and Llama 4's `no_rope_layers`). `model.support()` names the reason per block, and `rotate` raises `nnterp.Unavailable` with it:

```python
gpt2 = StandardizedTransformer("openai-community/gpt2")
print(gpt2.support(layer=0)["self_attn.attention_queries_prerope"])
# no rotary embedding: GPT-2 adds learned absolute position embeddings (wpe) to the stream
```

Under latent attention (DeepSeek-V2/V3, Kimi-K2, GLM-5) only part of each query head is rotated, and one rope key per position that every head shares: there `attention_keys_prerope` is that key, `[batch, 1, seq, qk_rope_head_dim]` (`SharedRopeKeys`). A family's page in the [encyclopedia](https://ndif-team.github.io/nnterp/encyclopedia/) lists where it reads the two values and what `support()` says of them.

The rotated queries and keys, and the rest of the attention's interior, are on [Attention](attention.md).
